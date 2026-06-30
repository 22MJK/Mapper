#include "taskflow/taskflow.h"

#include "mapping/cost_model.h"
#include "mapping/operator_catalog.h"
#include "taskflow/json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace taskflow {
namespace {

std::uint64_t bytes_to_uint64(double bytes) {
    if (!(bytes >= 0.0)) {
        return 0;
    }
    if (bytes >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(std::llround(bytes));
}

std::uint64_t flops_to_uint64(double flops) {
    if (!(flops >= 0.0)) {
        return 0;
    }
    if (flops >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(std::llround(flops));
}

std::int64_t checked_i64_attr(std::uint64_t value, const std::string& name) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::runtime_error("Attribute " + name + " exceeds int64 range");
    }
    return static_cast<std::int64_t>(value);
}

std::int32_t checked_i32_attr(std::uint64_t value, const std::string& name) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::runtime_error("Attribute " + name + " exceeds int32 range");
    }
    return static_cast<std::int32_t>(value);
}

bool is_cpu_device(const hardware_topology::Device* device) {
    if (device == nullptr) {
        return false;
    }
    std::string type = device->type;
    for (char& ch : type) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return type == "cpu" || type == "cpu_node";
}

struct CommHop {
    std::string src;
    std::string dst;
};

std::vector<CommHop> direct_communication_route(const hardware_topology::HardwareTopology& topology,
                                                const std::string& src_device,
                                                const std::string& dst_device) {
    if (src_device != dst_device && topology.link_id(src_device, dst_device).has_value()) {
        return {{src_device, dst_device}};
    }
    return {};
}

std::uint64_t estimate_comp_duration_micros(const mapping::Task& task,
                                            const hardware_topology::Device* device) {
    if (device == nullptr) {
        return 1;
    }
    const double duration_s = mapping::estimate_task_time_seconds(task, device);
    if (duration_s <= 0.0) {
        return 1;
    }
    const long double micros = duration_s * 1e6;
    if (micros >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    const auto out = static_cast<std::uint64_t>(std::llround(micros));
    return out == 0 ? 1 : out;
}

std::uint64_t stable_device_rank(const std::string& device_id,
                                 const std::unordered_map<std::string, std::uint64_t>& rank_map) {
    const auto it = rank_map.find(device_id);
    if (it != rank_map.end()) {
        return it->second;
    }
    return 0;
}

void sort_and_dedup(std::vector<std::uint64_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

void sort_and_dedup_rank_deps(std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>& rank_deps) {
    for (auto& entry : rank_deps) {
        sort_and_dedup(entry.second);
    }
}

std::string join_features(const std::unordered_set<std::string>& features) {
    std::vector<std::string> sorted(features.begin(), features.end());
    std::sort(sorted.begin(), sorted.end());
    std::string out;
    std::size_t reserve_size = sorted.empty() ? 0 : sorted.size() - 1;
    for (const auto& value : sorted) {
        reserve_size += value.size();
    }
    out.reserve(reserve_size);
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += sorted[i];
    }
    return out;
}

struct EtAttr {
    std::string name;
    std::optional<std::uint64_t> uint64_val;
    std::optional<std::int64_t> int64_val;
    std::optional<std::int32_t> int32_val;
    std::optional<std::string> string_val;
    std::optional<bool> bool_val;
    std::optional<std::vector<std::uint64_t>> uint64_list_val;
    std::optional<std::vector<bool>> bool_list_val;
};

struct EtNode {
    std::uint64_t id{0};
    std::string name;
    std::string type;
    std::vector<std::uint64_t> ctrl_deps;
    std::vector<std::uint64_t> data_deps;
    std::optional<std::uint64_t> duration_micros;
    std::vector<EtAttr> attrs;
};

struct EtGraph {
    std::vector<EtNode> nodes;
    std::unordered_map<std::uint64_t, std::uint64_t> node_rank;
    std::vector<bool> cpu_rank;
    std::uint64_t rank_count{0};
    std::string metadata_version{"0.0.6"};
};

void add_attr_u64(std::vector<EtAttr>& attrs, const std::string& name, std::uint64_t value) {
    EtAttr attr;
    attr.name = name;
    attr.uint64_val = value;
    attrs.push_back(std::move(attr));
}

void add_attr_i64(std::vector<EtAttr>& attrs, const std::string& name, std::int64_t value) {
    EtAttr attr;
    attr.name = name;
    attr.int64_val = value;
    attrs.push_back(std::move(attr));
}

void add_attr_i64_from_u64(std::vector<EtAttr>& attrs, const std::string& name, std::uint64_t value) {
    add_attr_i64(attrs, name, checked_i64_attr(value, name));
}

void add_attr_i32(std::vector<EtAttr>& attrs, const std::string& name, std::int32_t value) {
    EtAttr attr;
    attr.name = name;
    attr.int32_val = value;
    attrs.push_back(std::move(attr));
}

void add_attr_i32_from_u64(std::vector<EtAttr>& attrs, const std::string& name, std::uint64_t value) {
    add_attr_i32(attrs, name, checked_i32_attr(value, name));
}

void add_attr_str(std::vector<EtAttr>& attrs, const std::string& name, const std::string& value) {
    EtAttr attr;
    attr.name = name;
    attr.string_val = value;
    attrs.push_back(std::move(attr));
}

void add_attr_bool(std::vector<EtAttr>& attrs, const std::string& name, bool value) {
    EtAttr attr;
    attr.name = name;
    attr.bool_val = value;
    attrs.push_back(std::move(attr));
}

void add_attr_u64_list(std::vector<EtAttr>& attrs, const std::string& name, std::vector<std::uint64_t> values) {
    EtAttr attr;
    attr.name = name;
    attr.uint64_list_val = std::move(values);
    attrs.push_back(std::move(attr));
}

void add_attr_bool_list(std::vector<EtAttr>& attrs, const std::string& name, std::vector<bool> values) {
    EtAttr attr;
    attr.name = name;
    attr.bool_list_val = std::move(values);
    attrs.push_back(std::move(attr));
}

void write_u64_array(std::ostream& out, const std::vector<std::uint64_t>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        json::write_uint64(out, values[i]);
    }
    out << "]";
}

void write_bool_array(std::ostream& out, const std::vector<bool>& values) {
    out << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << (values[i] ? "true" : "false");
    }
    out << "]";
}

void write_et_attr(std::ostream& out, const EtAttr& attr) {
    out << "{";
    out << "\"name\": ";
    json::write_string(out, attr.name);
    if (attr.uint64_val.has_value()) {
        out << ", \"uint64_val\": ";
        json::write_uint64(out, *attr.uint64_val);
    } else if (attr.int64_val.has_value()) {
        out << ", \"int64_val\": " << *attr.int64_val;
    } else if (attr.int32_val.has_value()) {
        out << ", \"int32_val\": " << *attr.int32_val;
    } else if (attr.string_val.has_value()) {
        out << ", \"string_val\": ";
        json::write_string(out, *attr.string_val);
    } else if (attr.bool_val.has_value()) {
        out << ", \"bool_val\": " << (*attr.bool_val ? "true" : "false");
    } else if (attr.uint64_list_val.has_value()) {
        out << ", \"uint64_list_val\": ";
        write_u64_array(out, *attr.uint64_list_val);
    } else if (attr.bool_list_val.has_value()) {
        out << ", \"bool_list_val\": ";
        write_bool_array(out, *attr.bool_list_val);
    }
    out << "}";
}

struct CompInputAttrs {
    std::vector<std::string> ids;
    std::vector<std::vector<std::int64_t>> shapes;
    std::vector<std::string> storage_formats;
    std::vector<std::uint64_t> nnz;
    std::vector<std::string> dtypes;
};

std::string join_strings(const std::vector<std::string>& values) {
    std::string out;
    std::size_t reserve_size = values.empty() ? 0 : values.size() - 1;
    for (const auto& value : values) {
        reserve_size += value.size();
    }
    out.reserve(reserve_size);
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += values[i];
    }
    return out;
}

std::string canonical_token(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

std::string normalize_storage_format(const std::string& value) {
    const auto normalized = canonical_token(value.empty() ? "dense" : value);
    if (normalized == "sparse_csr") {
        return "csr";
    }
    if (normalized == "sparse_csc") {
        return "csc";
    }
    if (normalized == "sparse_coo" || normalized == "coo_import") {
        return "coo";
    }
    if (normalized == "sparse_bsr") {
        return "bsr";
    }
    if (normalized == "sparse_block") {
        return "block_sparse";
    }
    return normalized;
}

bool is_sparse_storage_format(const std::string& value) {
    const auto storage = normalize_storage_format(value);
    return storage == "csr" || storage == "csc" || storage == "coo" || storage == "bsr" ||
           storage == "block_sparse";
}

void append_json_string(std::string& out, std::string_view value) {
    static constexpr char kHex[] = "0123456789abcdef";
    out.push_back('"');
    for (const unsigned char c : value) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(kHex[c >> 4]);
                    out.push_back(kHex[c & 0x0F]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
}

void append_u64_array(std::string& out, const std::vector<std::uint64_t>& values) {
    out.push_back('[');
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out += ", ";
        }
        out += std::to_string(values[i]);
    }
    out.push_back(']');
}

std::string json_string_list(const std::vector<std::string>& values) {
    std::string out;
    std::size_t reserve_size = 2;
    if (!values.empty()) {
        reserve_size += values.size() - 1;
    }
    for (const auto& value : values) {
        reserve_size += value.size() + 2;
    }
    out.reserve(reserve_size);
    out.push_back('[');
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out.push_back(',');
        }
        append_json_string(out, values[i]);
    }
    out.push_back(']');
    return out;
}

std::string json_shape_list(const std::vector<std::vector<std::int64_t>>& shapes) {
    std::string out;
    out.reserve(2 + shapes.size() * 8);
    out.push_back('[');
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        if (i != 0) {
            out.push_back(',');
        }
        out.push_back('[');
        for (std::size_t j = 0; j < shapes[i].size(); ++j) {
            if (j != 0) {
                out.push_back(',');
            }
            out += std::to_string(shapes[i][j]);
        }
        out.push_back(']');
    }
    out.push_back(']');
    return out;
}

void append_input_attrs(CompInputAttrs& out, const mapping::TaskInput& input) {
    const auto storage = normalize_storage_format(input.storage_format);
    const bool sparse = is_sparse_storage_format(storage);
    out.ids.push_back(input.tensor_id);
    out.shapes.push_back(input.shape);
    out.storage_formats.push_back(storage);
    out.nnz.push_back(sparse && input.nonzero_elements.has_value() ? *input.nonzero_elements : 0);
    out.dtypes.push_back(input.dtype.empty() ? "fp32" : input.dtype);
}

void append_unknown_input_attrs(CompInputAttrs& out, std::string tensor_id, std::string dtype = {}) {
    mapping::TaskInput input;
    input.tensor_id = std::move(tensor_id);
    input.dtype = std::move(dtype);
    append_input_attrs(out, input);
}

CompInputAttrs comp_input_attrs(const mapping::TaskGraph& graph, const mapping::Task& task) {
    CompInputAttrs out;
    const auto& deps = graph.dependencies(task.name);
    const auto expected_inputs = task.input_data.size() + deps.size();
    out.ids.reserve(expected_inputs);
    out.shapes.reserve(expected_inputs);
    out.storage_formats.reserve(expected_inputs);
    out.nnz.reserve(expected_inputs);
    out.dtypes.reserve(expected_inputs);

    for (const auto& input : task.input_data) {
        append_input_attrs(out, input);
    }

    if (deps.empty()) {
        return out;
    }

    if (task.input_data.empty() && deps.size() == 1) {
        const auto& edge = deps.front();
        append_unknown_input_attrs(out, edge.tensor_id, edge.dtype);
        return out;
    }

    std::unordered_map<std::string, std::vector<std::size_t>> tensor_indices;
    tensor_indices.reserve(expected_inputs);
    for (std::size_t index = 0; index < task.input_data.size(); ++index) {
        const auto& input = task.input_data[index];
        if (!input.tensor_id.empty()) {
            tensor_indices[input.tensor_id].push_back(index);
        }
    }

    for (const auto& edge : deps) {
        if (!edge.tensor_id.empty()) {
            const auto it = tensor_indices.find(edge.tensor_id);
            if (it != tensor_indices.end()) {
                continue;
            }
            tensor_indices[edge.tensor_id].push_back(out.ids.size());
            append_unknown_input_attrs(out, edge.tensor_id, edge.dtype);
        } else {
            append_unknown_input_attrs(out, "", edge.dtype);
        }
    }

    return out;
}

std::string canonical_comm_kind(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    if (value == "all_reduce") {
        return "allreduce";
    }
    if (value == "all_gather") {
        return "allgather";
    }
    if (value == "reduce_scatter") {
        return "reducescatter";
    }
    if (value == "all_to_all") {
        return "alltoall";
    }
    return value;
}

bool is_collective_kind(const std::string& kind) {
    const auto k = canonical_comm_kind(kind);
    return k == "allreduce" || k == "allgather" || k == "reducescatter" || k == "broadcast" || k == "alltoall";
}

std::string collective_group_key(const mapping::TaskEdge& edge) {
    const auto kind = canonical_comm_kind(edge.comm_kind);
    if (!edge.comm_group.empty()) {
        return kind + "|group|" + edge.comm_group;
    }
    if (kind == "broadcast" || kind == "allgather") {
        return kind + "|src|" + edge.src + "|" + edge.tensor_id;
    }
    if (kind == "reducescatter" || kind == "allreduce") {
        return kind + "|dst|" + edge.dst;
    }
    if (kind == "alltoall") {
        return kind + "|tensor|" + edge.tensor_id;
    }
    return kind + "|" + edge.src + "|" + edge.dst + "|" + edge.tensor_id;
}

std::string collective_comm_type(const std::string& kind) {
    const auto k = canonical_comm_kind(kind);
    if (k == "allreduce") {
        return "ALL_REDUCE";
    }
    if (k == "allgather") {
        return "ALL_GATHER";
    }
    if (k == "reducescatter") {
        return "REDUCE_SCATTER";
    }
    if (k == "broadcast") {
        return "BROADCAST";
    }
    if (k == "alltoall") {
        return "ALL_TO_ALL";
    }
    return "ALL_TO_ALL";
}

std::int64_t collective_comm_type_code(const std::string& kind) {
    const auto k = canonical_comm_kind(kind);
    if (k == "allreduce") {
        return 0;
    }
    if (k == "allgather") {
        return 2;
    }
    if (k == "broadcast") {
        return 5;
    }
    if (k == "alltoall") {
        return 6;
    }
    if (k == "reducescatter") {
        return 7;
    }
    return 6;
}

std::string collective_name_token(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (char ch : value) {
        if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-') {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "group" : out;
}

struct CollectiveGroup {
    std::string key;
    std::string kind;
    std::vector<mapping::TaskEdge> edges;
};

std::uint64_t collective_payload_bytes(const CollectiveGroup& group) {
    std::unordered_map<std::string, std::uint64_t> bytes_by_tensor;
    std::uint64_t anonymous_total = 0;
    for (const auto& edge : group.edges) {
        const auto bytes = bytes_to_uint64(edge.tensor_bytes);
        if (edge.tensor_id.empty()) {
            anonymous_total += bytes;
            continue;
        }
        auto& current = bytes_by_tensor[edge.tensor_id];
        current = std::max(current, bytes);
    }

    std::uint64_t total = anonymous_total;
    for (const auto& entry : bytes_by_tensor) {
        total += entry.second;
    }
    return total;
}

std::string rank_data_deps_json(const std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>& rank_deps) {
    std::vector<std::uint64_t> ranks;
    ranks.reserve(rank_deps.size());
    for (const auto& entry : rank_deps) {
        if (!entry.second.empty()) {
            ranks.push_back(entry.first);
        }
    }
    sort_and_dedup(ranks);

    std::string out;
    out.reserve(2 + ranks.size() * 16);
    out.push_back('{');
    for (std::size_t i = 0; i < ranks.size(); ++i) {
        if (i != 0) {
            out.push_back(',');
        }
        append_json_string(out, std::to_string(ranks[i]));
        out.push_back(':');
        append_u64_array(out, rank_deps.at(ranks[i]));
    }
    out.push_back('}');
    return out;
}

std::string rank_group_name(const std::vector<std::uint64_t>& ranks) {
    if (ranks.empty()) {
        return "ranks_empty";
    }
    std::string out;
    out.reserve(6 + ranks.size() * 4);
    out = "ranks";
    for (const auto rank : ranks) {
        out += "_";
        out += std::to_string(rank);
    }
    return out;
}

void write_et_node(std::ostream& out, const EtNode& node) {
    out << "    {\n";
    out << "      \"id\": ";
    json::write_uint64(out, node.id);
    out << ",\n";
    out << "      \"name\": ";
    json::write_string(out, node.name);
    out << ",\n";
    out << "      \"type\": ";
    json::write_string(out, node.type);
    out << ",\n";
    out << "      \"ctrl_deps\": ";
    write_u64_array(out, node.ctrl_deps);
    out << ",\n";
    out << "      \"data_deps\": ";
    write_u64_array(out, node.data_deps);
    if (node.duration_micros.has_value()) {
        out << ",\n";
        out << "      \"duration_micros\": ";
        json::write_uint64(out, *node.duration_micros);
    }
    if (!node.attrs.empty()) {
        out << ",\n";
        out << "      \"attr\": [";
        for (std::size_t i = 0; i < node.attrs.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            write_et_attr(out, node.attrs[i]);
        }
        out << "]";
    }
    out << "\n";
    out << "    }";
}

const EtAttr* first_attr(const std::vector<EtAttr>& attrs, const std::string& name) {
    for (const auto& attr : attrs) {
        if (attr.name == name) {
            return &attr;
        }
    }
    return nullptr;
}

std::string attr_string_value(const std::vector<EtAttr>& attrs,
                              const std::string& name,
                              const std::string& default_value = "") {
    const auto* attr = first_attr(attrs, name);
    if (attr == nullptr) {
        return default_value;
    }
    if (attr->string_val.has_value()) {
        return *attr->string_val;
    }
    if (attr->uint64_val.has_value()) {
        return std::to_string(*attr->uint64_val);
    }
    if (attr->int64_val.has_value()) {
        return std::to_string(*attr->int64_val);
    }
    if (attr->int32_val.has_value()) {
        return std::to_string(*attr->int32_val);
    }
    if (attr->bool_val.has_value()) {
        return *attr->bool_val ? "true" : "false";
    }
    return default_value;
}

std::vector<std::uint64_t> attr_u64_list_value(const std::vector<EtAttr>& attrs,
                                               const std::string& name) {
    const auto* attr = first_attr(attrs, name);
    if (attr == nullptr || !attr->uint64_list_val.has_value()) {
        return {};
    }
    return *attr->uint64_list_val;
}

EtNode idle_noop_node(std::uint64_t rank, bool prefer_cpu_node) {
    EtNode node;
    node.id = 10'000'000 + rank;
    node.name = "idle_noop_rank_" + std::to_string(rank);
    node.type = "COMP_NODE";
    node.duration_micros = 1;
    add_attr_bool(node.attrs, "is_cpu_op", false);
    add_attr_str(node.attrs, "compute_target", prefer_cpu_node ? "cpu_node" : "gpu");
    add_attr_u64(node.attrs, "num_ops", 1);
    add_attr_u64(node.attrs, "tensor_size", 1);
    add_attr_str(node.attrs, "assigned_device", std::to_string(rank));
    add_attr_str(node.attrs, "subtype", "noop");
    return node;
}

void add_collective_ordering_deps(std::vector<std::vector<EtNode>>& rank_nodes,
                                  const std::unordered_map<std::uint64_t, std::uint64_t>& collective_order) {
    for (auto& nodes : rank_nodes) {
        std::vector<EtNode*> collectives;
        collectives.reserve(nodes.size());
        for (auto& node : nodes) {
            if (node.type == "COMM_COLL_NODE") {
                collectives.push_back(&node);
            }
        }
        std::sort(collectives.begin(),
                  collectives.end(),
                  [&](const EtNode* a, const EtNode* b) {
                      const auto a_order = collective_order.count(a->id) ? collective_order.at(a->id) : a->id;
                      const auto b_order = collective_order.count(b->id) ? collective_order.at(b->id) : b->id;
                      if (a_order != b_order) {
                          return a_order < b_order;
                      }
                      return a->id < b->id;
                  });
        std::unordered_map<std::string, std::uint64_t> last_by_group;
        for (auto* node : collectives) {
            const auto group = attr_string_value(node->attrs, "pg_name", "__default__");
            const auto previous = last_by_group.find(group);
            if (previous != last_by_group.end() &&
                std::find(node->ctrl_deps.begin(), node->ctrl_deps.end(), previous->second) == node->ctrl_deps.end()) {
                node->ctrl_deps.push_back(previous->second);
                sort_and_dedup(node->ctrl_deps);
            }
            last_by_group[group] = node->id;
        }
    }
}

std::vector<std::vector<EtNode>> split_et_graph_by_rank(const EtGraph& graph) {
    std::vector<std::vector<EtNode>> rank_nodes(static_cast<std::size_t>(graph.rank_count));
    std::unordered_map<std::uint64_t, std::unordered_map<std::uint64_t, std::vector<std::uint64_t>>> coll_deps_by_rank;
    std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> coll_target_ranks;
    std::unordered_map<std::uint64_t, std::uint64_t> collective_order;

    for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
        const auto& node = graph.nodes[index];
        if (node.type != "COMM_COLL_NODE") {
            continue;
        }
        collective_order.emplace(node.id, static_cast<std::uint64_t>(index));
        auto targets = attr_u64_list_value(node.attrs, "involved_ranks");
        targets.erase(std::remove_if(targets.begin(),
                                     targets.end(),
                                     [&](std::uint64_t rank) { return rank >= graph.rank_count; }),
                      targets.end());
        sort_and_dedup(targets);
        if (targets.empty()) {
            for (std::uint64_t rank = 0; rank < graph.rank_count; ++rank) {
                targets.push_back(rank);
            }
        }
        coll_target_ranks.emplace(node.id, targets);
        auto& by_rank = coll_deps_by_rank[node.id];
        for (const auto dep : node.data_deps) {
            const auto owner = graph.node_rank.find(dep);
            if (owner == graph.node_rank.end() || owner->second >= graph.rank_count) {
                continue;
            }
            by_rank[owner->second].push_back(dep);
        }
        for (auto& entry : by_rank) {
            sort_and_dedup(entry.second);
        }
    }

    for (const auto& node : graph.nodes) {
        std::uint64_t owner = 0;
        const auto rank_it = graph.node_rank.find(node.id);
        if (rank_it != graph.node_rank.end()) {
            owner = rank_it->second;
        }
        if (node.type == "COMM_COLL_NODE") {
            const auto& targets = coll_target_ranks.at(node.id);
            if (std::find(targets.begin(), targets.end(), owner) == targets.end()) {
                owner = targets.front();
            }
        }
        if (owner >= graph.rank_count) {
            throw std::runtime_error("ET node " + std::to_string(node.id) + " assigned to invalid rank " +
                                     std::to_string(owner));
        }
        rank_nodes[static_cast<std::size_t>(owner)].push_back(node);
    }

    std::vector<std::unordered_set<std::uint64_t>> local_ids(static_cast<std::size_t>(graph.rank_count));
    for (std::uint64_t rank = 0; rank < graph.rank_count; ++rank) {
        for (const auto& node : rank_nodes[static_cast<std::size_t>(rank)]) {
            local_ids[static_cast<std::size_t>(rank)].insert(node.id);
        }
    }

    std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> collective_ids_by_rank;
    std::vector<EtNode> collective_templates;
    for (std::uint64_t rank = 0; rank < graph.rank_count; ++rank) {
        auto& nodes = rank_nodes[static_cast<std::size_t>(rank)];
        const auto& ids = local_ids[static_cast<std::size_t>(rank)];
        for (auto& node : nodes) {
            std::vector<std::uint64_t> kept_data;
            if (node.type == "COMM_COLL_NODE") {
                auto coll_it = coll_deps_by_rank.find(node.id);
                if (coll_it != coll_deps_by_rank.end()) {
                    auto deps_it = coll_it->second.find(rank);
                    if (deps_it != coll_it->second.end()) {
                        kept_data = deps_it->second;
                    }
                }
                collective_ids_by_rank[rank].insert(node.id);
                collective_templates.push_back(node);
            } else {
                for (const auto dep : node.data_deps) {
                    if (ids.count(dep) != 0) {
                        kept_data.push_back(dep);
                    }
                }
            }
            node.data_deps = std::move(kept_data);

            std::vector<std::uint64_t> kept_ctrl;
            for (const auto dep : node.ctrl_deps) {
                if (ids.count(dep) != 0) {
                    kept_ctrl.push_back(dep);
                }
            }
            node.ctrl_deps = std::move(kept_ctrl);
        }
    }

    for (const auto& templ : collective_templates) {
        const auto targets_it = coll_target_ranks.find(templ.id);
        if (targets_it == coll_target_ranks.end()) {
            continue;
        }
        for (const auto rank : targets_it->second) {
            if (collective_ids_by_rank[rank].count(templ.id) != 0) {
                continue;
            }
            EtNode replica = templ;
            auto deps_it = coll_deps_by_rank[templ.id].find(rank);
            replica.data_deps = deps_it == coll_deps_by_rank[templ.id].end()
                                    ? std::vector<std::uint64_t>{}
                                    : deps_it->second;
            replica.ctrl_deps.clear();
            rank_nodes[static_cast<std::size_t>(rank)].push_back(std::move(replica));
            collective_ids_by_rank[rank].insert(templ.id);
        }
    }

    add_collective_ordering_deps(rank_nodes, collective_order);

    for (std::uint64_t rank = 0; rank < graph.rank_count; ++rank) {
        auto& nodes = rank_nodes[static_cast<std::size_t>(rank)];
        if (nodes.empty()) {
            const bool prefer_cpu = rank < graph.cpu_rank.size() && graph.cpu_rank[static_cast<std::size_t>(rank)];
            nodes.push_back(idle_noop_node(rank, prefer_cpu));
        }
        std::sort(nodes.begin(), nodes.end(), [](const EtNode& a, const EtNode& b) { return a.id < b.id; });
    }

    return rank_nodes;
}

void encode_varint(std::ostream& out, std::uint64_t value) {
    while (value > 0x7F) {
        out.put(static_cast<char>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.put(static_cast<char>(value & 0x7F));
}

void append_varint(std::string& out, std::uint64_t value) {
    while (value > 0x7F) {
        out.push_back(static_cast<char>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    out.push_back(static_cast<char>(value & 0x7F));
}

void append_key(std::string& out, std::uint32_t field_num, std::uint32_t wire_type) {
    append_varint(out, (static_cast<std::uint64_t>(field_num) << 3) | wire_type);
}

void append_length_delimited(std::string& out, std::uint32_t field_num, const std::string& payload) {
    append_key(out, field_num, 2);
    append_varint(out, payload.size());
    out.append(payload);
}

void append_u64_field(std::string& out, std::uint32_t field_num, std::uint64_t value) {
    append_key(out, field_num, 0);
    append_varint(out, value);
}

void append_i64_field(std::string& out, std::uint32_t field_num, std::int64_t value) {
    append_key(out, field_num, 0);
    append_varint(out, static_cast<std::uint64_t>(value));
}

void append_bool_field(std::string& out, std::uint32_t field_num, bool value) {
    append_u64_field(out, field_num, value ? 1 : 0);
}

std::string encode_u64_list_message(const std::vector<std::uint64_t>& values) {
    std::string packed;
    for (const auto value : values) {
        append_varint(packed, value);
    }
    std::string out;
    append_length_delimited(out, 1, packed);
    return out;
}

std::string encode_bool_list_message(const std::vector<bool>& values) {
    std::string packed;
    for (bool value : values) {
        append_varint(packed, value ? 1 : 0);
    }
    std::string out;
    append_length_delimited(out, 1, packed);
    return out;
}

std::string encode_attr_message(const EtAttr& attr, const std::string& node_type) {
    std::string out;
    append_length_delimited(out, 1, attr.name);
    const bool clamp_zero_p2p_comm_size =
        attr.name == "comm_size" && (node_type == "COMM_SEND_NODE" || node_type == "COMM_RECV_NODE");
    if (attr.uint64_list_val.has_value()) {
        append_length_delimited(out, 14, encode_u64_list_message(*attr.uint64_list_val));
    } else if (attr.bool_list_val.has_value()) {
        append_length_delimited(out, 28, encode_bool_list_message(*attr.bool_list_val));
    } else if (attr.uint64_val.has_value()) {
        append_u64_field(out, 13, clamp_zero_p2p_comm_size ? std::max<std::uint64_t>(1, *attr.uint64_val) : *attr.uint64_val);
    } else if (attr.int64_val.has_value()) {
        append_i64_field(out, 9, clamp_zero_p2p_comm_size ? std::max<std::int64_t>(1, *attr.int64_val) : *attr.int64_val);
    } else if (attr.int32_val.has_value()) {
        append_i64_field(out, 7, clamp_zero_p2p_comm_size ? std::max<std::int32_t>(1, *attr.int32_val) : *attr.int32_val);
    } else if (attr.bool_val.has_value()) {
        append_bool_field(out, 27, *attr.bool_val);
    } else if (attr.string_val.has_value()) {
        append_length_delimited(out, 29, *attr.string_val);
    }
    return out;
}

std::int32_t node_type_code(const std::string& type) {
    if (type == "METADATA_NODE") {
        return 1;
    }
    if (type == "MEM_LOAD_NODE") {
        return 2;
    }
    if (type == "MEM_STORE_NODE") {
        return 3;
    }
    if (type == "COMP_NODE") {
        return 4;
    }
    if (type == "COMM_SEND_NODE") {
        return 5;
    }
    if (type == "COMM_RECV_NODE") {
        return 6;
    }
    if (type == "COMM_COLL_NODE") {
        return 7;
    }
    throw std::runtime_error("Unsupported ET node type: " + type);
}

std::string encode_node_message(const EtNode& node) {
    std::string out;
    append_u64_field(out, 1, node.id);
    append_length_delimited(out, 2, node.name);
    append_i64_field(out, 3, node_type_code(node.type));
    for (const auto dep : node.ctrl_deps) {
        append_u64_field(out, 4, dep);
    }
    for (const auto dep : node.data_deps) {
        append_u64_field(out, 5, dep);
    }
    const auto duration = std::max<std::uint64_t>(1, node.duration_micros.value_or(1));
    append_u64_field(out, 7, duration);
    for (const auto& attr : node.attrs) {
        const auto attr_message = encode_attr_message(attr, node.type);
        if (!attr_message.empty()) {
            append_length_delimited(out, 10, attr_message);
        }
    }
    return out;
}

std::string encode_global_metadata_message(const std::string& version) {
    std::string out;
    append_length_delimited(out, 1, version);
    return out;
}

void write_framed_message(std::ostream& out, const std::string& message) {
    encode_varint(out, message.size());
    out.write(message.data(), static_cast<std::streamsize>(message.size()));
}

void ensure_parent_dir(const std::filesystem::path& path) {
    const auto parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

}  // namespace

EtGraph build_et_graph(const std::string& time_unit,
                       const mapping::TaskGraph& graph,
                       const mapping::MappingPlan& mapping_plan,
                       const hardware_topology::HardwareTopology& topology) {
    (void)time_unit;  // Keep the public API stable while omitting the non-ET metadata field.

    const auto& topo_tasks = graph.topological_order();
    std::size_t graph_edge_count = 0;
    for (const auto& task : topo_tasks) {
        graph_edge_count += graph.successors(task.name).size();
    }

    std::unordered_map<std::string, std::uint64_t> task_id;
    task_id.reserve(topo_tasks.size());
    for (std::uint64_t i = 0; i < topo_tasks.size(); ++i) {
        task_id.emplace(topo_tasks[i].name, i);
    }

    std::unordered_map<std::string, std::uint64_t> device_rank;
    const auto& devices = topology.compute_devices();
    device_rank.reserve(devices.size());
    for (std::uint64_t i = 0; i < devices.size(); ++i) {
        device_rank.emplace(devices[i]->id, i);
    }

    std::vector<EtNode> nodes;
    const auto p2p_extra_upper_bound = graph_edge_count > std::numeric_limits<std::size_t>::max() / 2
                                           ? std::numeric_limits<std::size_t>::max()
                                           : graph_edge_count * 2;
    const auto extra_node_capacity =
        std::min(p2p_extra_upper_bound, topo_tasks.size() / 4 + static_cast<std::size_t>(4096));
    nodes.reserve(topo_tasks.size() + extra_node_capacity);
    std::unordered_map<std::string, std::size_t> task_name_to_node_index;
    task_name_to_node_index.reserve(topo_tasks.size());
    std::unordered_map<std::uint64_t, std::uint64_t> node_rank;
    node_rank.reserve(topo_tasks.size() + extra_node_capacity);
    std::vector<bool> cpu_rank(devices.size(), false);
    for (std::uint64_t i = 0; i < devices.size(); ++i) {
        cpu_rank[static_cast<std::size_t>(i)] = is_cpu_device(devices[static_cast<std::size_t>(i)]);
    }

    for (std::size_t i = 0; i < topo_tasks.size(); ++i) {
        const auto& task = topo_tasks[i];
        EtNode node;
        node.id = static_cast<std::uint64_t>(i);
        node.name = task.name;
        node.type = "COMP_NODE";
        node.attrs.reserve(12);
        const auto& assigned_device = mapping_plan.node_for(task.name);
        const auto* device = topology.device(assigned_device);
        const bool on_cpu = is_cpu_device(device);
        add_attr_bool(node.attrs, "is_cpu_op", on_cpu);
        // Emit the documented ET target string for dedicated CPU-node vs GPU compute.
        add_attr_str(node.attrs, "compute_target", on_cpu ? "cpu_node" : "gpu");
        add_attr_u64(node.attrs, "num_ops", flops_to_uint64(task.compute_flops));
        add_attr_u64(node.attrs, "tensor_size", bytes_to_uint64(task.memory_bytes));
        const auto input_attrs = comp_input_attrs(graph, task);
        add_attr_str(node.attrs, "input_tensor_ids", join_strings(input_attrs.ids));
        add_attr_str(node.attrs, "input_tensor_shapes", json_shape_list(input_attrs.shapes));
        add_attr_str(node.attrs, "input_tensor_storage_formats", json_string_list(input_attrs.storage_formats));
        add_attr_u64_list(node.attrs, "input_tensor_nnz", input_attrs.nnz);
        add_attr_str(node.attrs, "input_tensor_dtypes", json_string_list(input_attrs.dtypes));
        add_attr_str(node.attrs, "assigned_device", assigned_device);
        if (!task.subtype.empty()) {
            add_attr_str(node.attrs, "subtype", mapping::canonical_operator_subtype(task.subtype));
        }
        if (!task.features.empty()) {
            add_attr_str(node.attrs, "operator_features", join_features(task.features));
        }
        node.duration_micros = estimate_comp_duration_micros(task, device);
        task_name_to_node_index.emplace(task.name, nodes.size());
        node_rank.emplace(node.id, stable_device_rank(assigned_device, device_rank));
        nodes.push_back(std::move(node));
    }

    std::unordered_map<std::string, std::vector<std::uint64_t>> compute_data_deps;
    compute_data_deps.reserve(topo_tasks.size());
    std::uint64_t next_node_id = static_cast<std::uint64_t>(nodes.size());
    std::uint64_t comm_tag = 1;
    std::unordered_map<std::string, std::size_t> collective_group_index;
    collective_group_index.reserve(std::min(graph_edge_count, topo_tasks.size()));
    std::vector<CollectiveGroup> collective_groups;
    collective_groups.reserve(std::min(graph_edge_count, topo_tasks.size()));

    auto emit_p2p_transfer = [&](const mapping::TaskEdge& edge) {
        const auto src_it = task_id.find(edge.src);
        const auto dst_it = task_id.find(edge.dst);
        if (src_it == task_id.end() || dst_it == task_id.end()) {
            throw std::runtime_error("Task ID missing while building ET nodes");
        }
        const auto& src_device = mapping_plan.node_for(edge.src);
        const auto& dst_device = mapping_plan.node_for(edge.dst);
        const double raw_bytes = edge.tensor_bytes;
        const bool needs_transfer = src_device != dst_device;
        if (!needs_transfer) {
            compute_data_deps[edge.dst].push_back(src_it->second);
            return;
        }
        const std::uint64_t comm_bytes = bytes_to_uint64(raw_bytes);
        auto route = direct_communication_route(topology, src_device, dst_device);
        if (route.empty()) {
            // Preserve the original endpoint pair; the simulator resolves any indirect route.
            route.push_back({src_device, dst_device});
        }

        std::uint64_t previous_node_id = src_it->second;
        for (std::size_t hop_index = 0; hop_index < route.size(); ++hop_index) {
            const auto& hop = route[hop_index];
            const auto hop_src_rank = stable_device_rank(hop.src, device_rank);
            const auto hop_dst_rank = stable_device_rank(hop.dst, device_rank);
            const std::string hop_suffix = route.size() == 1 ? "" : "_h" + std::to_string(hop_index);
            EtNode send;
            send.id = next_node_id++;
            send.name = "send_" + edge.src + "_to_" + edge.dst + hop_suffix;
            send.type = "COMM_SEND_NODE";
            send.data_deps.reserve(1);
            send.data_deps.push_back(previous_node_id);
            send.attrs.reserve(6);
            add_attr_str(send.attrs, "subtype", "send");
            add_attr_bool(send.attrs, "is_cpu_op", false);
            add_attr_i32_from_u64(send.attrs, "comm_src", hop_src_rank);
            add_attr_i32_from_u64(send.attrs, "comm_dst", hop_dst_rank);
            add_attr_i64_from_u64(send.attrs, "comm_size", comm_bytes);
            add_attr_i32_from_u64(send.attrs, "comm_tag", comm_tag);
            node_rank.emplace(send.id, hop_src_rank);
            nodes.push_back(std::move(send));
            const std::uint64_t send_id = next_node_id - 1;

            EtNode recv;
            recv.id = next_node_id++;
            recv.name = "recv_" + edge.src + "_to_" + edge.dst + hop_suffix;
            recv.type = "COMM_RECV_NODE";
            recv.data_deps.reserve(1);
            recv.data_deps.push_back(send_id);
            recv.attrs.reserve(6);
            add_attr_str(recv.attrs, "subtype", "recv");
            add_attr_bool(recv.attrs, "is_cpu_op", false);
            add_attr_i32_from_u64(recv.attrs, "comm_src", hop_src_rank);
            add_attr_i32_from_u64(recv.attrs, "comm_dst", hop_dst_rank);
            add_attr_i64_from_u64(recv.attrs, "comm_size", comm_bytes);
            add_attr_i32_from_u64(recv.attrs, "comm_tag", comm_tag);
            node_rank.emplace(recv.id, hop_dst_rank);
            nodes.push_back(std::move(recv));
            previous_node_id = next_node_id - 1;

            if (comm_tag < static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
                comm_tag += 1;
            } else {
                throw std::runtime_error("comm_tag exceeds int32 range");
            }
        }
        compute_data_deps[edge.dst].push_back(previous_node_id);
    };

    for (const auto& task : topo_tasks) {
        for (const auto& edge : graph.successors(task.name)) {
            if (is_collective_kind(edge.comm_kind)) {
                const auto key = collective_group_key(edge);
                const auto [it, inserted] = collective_group_index.emplace(key, collective_groups.size());
                if (inserted) {
                    CollectiveGroup group;
                    group.key = key;
                    group.kind = canonical_comm_kind(edge.comm_kind);
                    collective_groups.push_back(std::move(group));
                }
                collective_groups[it->second].edges.push_back(edge);
                continue;
            }
            emit_p2p_transfer(edge);
        }
    }

    for (const auto& group : collective_groups) {
        if (group.edges.empty()) {
            continue;
        }

        std::vector<std::uint64_t> source_ids;
        std::vector<std::pair<std::uint64_t, std::string>> dst_rank_devices;
        std::unordered_set<std::string> dst_names;
        std::unordered_set<std::string> participant_devices;
        std::unordered_map<std::uint64_t, std::vector<std::uint64_t>> rank_data_deps;
        source_ids.reserve(group.edges.size());
        dst_rank_devices.reserve(group.edges.size());
        dst_names.reserve(group.edges.size());
        participant_devices.reserve(group.edges.size() * 2);
        rank_data_deps.reserve(group.edges.size() * 2);

        for (const auto& edge : group.edges) {
            const auto src_it = task_id.find(edge.src);
            const auto dst_it = task_id.find(edge.dst);
            if (src_it == task_id.end() || dst_it == task_id.end()) {
                throw std::runtime_error("Task ID missing while building collective ET nodes");
            }
            source_ids.push_back(src_it->second);
            dst_names.insert(edge.dst);
            const auto& src_device = mapping_plan.node_for(edge.src);
            const auto& dst_device = mapping_plan.node_for(edge.dst);
            participant_devices.insert(src_device);
            participant_devices.insert(dst_device);
            const auto src_rank = stable_device_rank(src_device, device_rank);
            const auto dst_rank = stable_device_rank(dst_device, device_rank);
            dst_rank_devices.emplace_back(dst_rank, dst_device);
            rank_data_deps[src_rank].push_back(src_it->second);
        }
        std::sort(dst_rank_devices.begin(), dst_rank_devices.end());
        dst_rank_devices.erase(std::unique(dst_rank_devices.begin(), dst_rank_devices.end()), dst_rank_devices.end());

        sort_and_dedup(source_ids);
        std::vector<std::uint64_t> involved_ranks;
        involved_ranks.reserve(participant_devices.size());
        bool has_cpu_participant = false;
        for (const auto& device_id : participant_devices) {
            const auto rank = stable_device_rank(device_id, device_rank);
            involved_ranks.push_back(rank);
            if (is_cpu_device(topology.device(device_id))) {
                has_cpu_participant = true;
            }
        }
        sort_and_dedup(involved_ranks);

        if (has_cpu_participant) {
            for (const auto& edge : group.edges) {
                emit_p2p_transfer(edge);
            }
            continue;
        }

        if (involved_ranks.size() <= 1) {
            for (const auto& edge : group.edges) {
                compute_data_deps[edge.dst].push_back(task_id.at(edge.src));
            }
            continue;
        }

        for (const auto& edge : group.edges) {
            const auto dst_rank = stable_device_rank(mapping_plan.node_for(edge.dst), device_rank);
            if (!rank_data_deps[dst_rank].empty()) {
                continue;
            }
            const auto deps_it = compute_data_deps.find(edge.dst);
            if (deps_it == compute_data_deps.end()) {
                continue;
            }
            auto& local_deps = rank_data_deps[dst_rank];
            for (const auto dep_id : deps_it->second) {
                const auto rank_it = node_rank.find(dep_id);
                if (rank_it != node_rank.end() && rank_it->second == dst_rank) {
                    local_deps.push_back(dep_id);
                }
            }
        }
        sort_and_dedup_rank_deps(rank_data_deps);

        std::vector<std::uint64_t> collective_data_deps;
        collective_data_deps.reserve(source_ids.size());
        for (const auto& entry : rank_data_deps) {
            collective_data_deps.insert(collective_data_deps.end(), entry.second.begin(), entry.second.end());
        }
        if (collective_data_deps.empty()) {
            collective_data_deps = source_ids;
        }
        sort_and_dedup(collective_data_deps);

        EtNode collective;
        collective.id = next_node_id++;
        collective.name =
            "collective_" + canonical_comm_kind(group.kind) + "_" + collective_name_token(group.key) + "_" +
            std::to_string(collective.id);
        collective.type = "COMM_COLL_NODE";
        collective.data_deps = std::move(collective_data_deps);
        collective.attrs.reserve(11);
        add_attr_str(collective.attrs, "subtype", canonical_comm_kind(group.kind));
        add_attr_bool(collective.attrs, "is_cpu_op", false);
        if (!dst_rank_devices.empty()) {
            add_attr_str(collective.attrs, "assigned_device", dst_rank_devices.front().second);
        }
        add_attr_i64(collective.attrs, "comm_type", collective_comm_type_code(group.kind));
        add_attr_str(collective.attrs, "comm_type_name", collective_comm_type(group.kind));
        add_attr_i32(collective.attrs, "comm_priority", 0);
        add_attr_i64_from_u64(collective.attrs, "comm_size", collective_payload_bytes(group));
        add_attr_str(collective.attrs, "pg_name", rank_group_name(involved_ranks));
        add_attr_str(collective.attrs, "rank_data_deps", rank_data_deps_json(rank_data_deps));
        add_attr_u64_list(collective.attrs, "involved_ranks", std::move(involved_ranks));
        add_attr_bool_list(collective.attrs, "involved_dim", {true, true, true, true});

        const auto collective_id = collective.id;
        if (!dst_rank_devices.empty()) {
            node_rank.emplace(collective_id, dst_rank_devices.front().first);
        }
        nodes.push_back(std::move(collective));
        for (const auto& dst : dst_names) {
            compute_data_deps[dst].push_back(collective_id);
        }
    }

    for (const auto& task : topo_tasks) {
        const auto idx_it = task_name_to_node_index.find(task.name);
        if (idx_it == task_name_to_node_index.end()) {
            continue;
        }
        auto deps_it = compute_data_deps.find(task.name);
        if (deps_it == compute_data_deps.end()) {
            continue;
        }
        auto deps = std::move(deps_it->second);
        sort_and_dedup(deps);
        nodes[idx_it->second].data_deps = std::move(deps);
    }

    std::sort(nodes.begin(), nodes.end(), [](const EtNode& a, const EtNode& b) { return a.id < b.id; });

    EtGraph out;
    out.nodes = std::move(nodes);
    out.node_rank = std::move(node_rank);
    out.cpu_rank = std::move(cpu_rank);
    out.rank_count = static_cast<std::uint64_t>(devices.size());
    out.metadata_version = "0.0.6";
    return out;
}

void write_taskflow_json_file(const std::string& path, const EtGraph& graph) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open " + path);
    }
    out << "{\n";
    out << "  \"global_metadata\": {\n";
    out << "    \"version\": ";
    json::write_string(out, graph.metadata_version);
    out << "\n";
    out << "  },\n";
    out << "  \"nodes\": [\n";
    for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
        write_et_node(out, graph.nodes[i]);
        out << (i + 1 == graph.nodes.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void write_chakra_et_files(const std::string& et_prefix, const EtGraph& graph) {
    if (et_prefix.empty()) {
        throw std::runtime_error("Chakra ET output prefix is empty");
    }
    const auto prefix = std::filesystem::path(et_prefix);
    ensure_parent_dir(prefix);
    const auto rank_nodes = split_et_graph_by_rank(graph);
    const auto metadata = encode_global_metadata_message(graph.metadata_version);
    for (std::size_t rank = 0; rank < rank_nodes.size(); ++rank) {
        const auto path = prefix.parent_path() / (prefix.filename().string() + "." + std::to_string(rank) + ".et");
        std::ofstream out(path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to open " + path.string());
        }
        write_framed_message(out, metadata);
        for (const auto& node : rank_nodes[rank]) {
            write_framed_message(out, encode_node_message(node));
        }
    }
}

void TaskflowWriter::write(const std::string& path,
                           const std::string& time_unit,
                           const mapping::TaskGraph& graph,
                           const mapping::MappingPlan& mapping_plan,
                           const hardware_topology::HardwareTopology& topology) {
    write_outputs(path, {}, time_unit, graph, mapping_plan, topology, true, false);
}

void TaskflowWriter::write_chakra_et(const std::string& et_prefix,
                                     const std::string& time_unit,
                                     const mapping::TaskGraph& graph,
                                     const mapping::MappingPlan& mapping_plan,
                                     const hardware_topology::HardwareTopology& topology) {
    write_outputs({}, et_prefix, time_unit, graph, mapping_plan, topology, false, true);
}

void TaskflowWriter::write_outputs(const std::string& json_path,
                                   const std::string& et_prefix,
                                   const std::string& time_unit,
                                   const mapping::TaskGraph& graph,
                                   const mapping::MappingPlan& mapping_plan,
                                   const hardware_topology::HardwareTopology& topology,
                                   bool emit_json,
                                   bool emit_chakra_et) {
    if (!emit_json && !emit_chakra_et) {
        return;
    }
    const auto et_graph = build_et_graph(time_unit, graph, mapping_plan, topology);
    if (emit_json) {
        write_taskflow_json_file(json_path, et_graph);
    }
    if (emit_chakra_et) {
        write_chakra_et_files(et_prefix, et_graph);
    }
}

}  // namespace taskflow
