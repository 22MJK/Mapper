#include "hardware_topology/json_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace hardware_topology {
namespace {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
    enum class Type { Null, Bool, Number, String, Object, Array };
    Type type{Type::Null};
    std::variant<std::nullptr_t, bool, double, std::string, JsonObject, JsonArray> value{nullptr};

    const JsonObject* as_object() const {
        return std::get_if<JsonObject>(&value);
    }
    const JsonArray* as_array() const {
        return std::get_if<JsonArray>(&value);
    }
    const std::string* as_string() const {
        return std::get_if<std::string>(&value);
    }
    const double* as_number() const {
        return std::get_if<double>(&value);
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string input) : input_(std::move(input)) {}

    bool parse(JsonValue& out, std::string& error) {
        skip_ws();
        if (!parse_value(out, error)) {
            return false;
        }
        skip_ws();
        if (pos_ != input_.size()) {
            error = "Trailing data after JSON";
            return false;
        }
        return true;
    }

private:
    bool parse_value(JsonValue& out, std::string& error) {
        skip_ws();
        if (pos_ >= input_.size()) {
            error = "Unexpected end of JSON";
            return false;
        }
        const char c = input_[pos_];
        if (c == '{') {
            return parse_object(out, error);
        }
        if (c == '[') {
            return parse_array(out, error);
        }
        if (c == '"') {
            std::string s;
            if (!parse_string(s, error)) {
                return false;
            }
            out.type = JsonValue::Type::String;
            out.value = std::move(s);
            return true;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) {
            double num = 0.0;
            if (!parse_number(num, error)) {
                return false;
            }
            out.type = JsonValue::Type::Number;
            out.value = num;
            return true;
        }
        if (match_literal("true")) {
            out.type = JsonValue::Type::Bool;
            out.value = true;
            return true;
        }
        if (match_literal("false")) {
            out.type = JsonValue::Type::Bool;
            out.value = false;
            return true;
        }
        if (match_literal("null")) {
            out.type = JsonValue::Type::Null;
            out.value = nullptr;
            return true;
        }
        error = "Invalid JSON value";
        return false;
    }

    bool parse_object(JsonValue& out, std::string& error) {
        if (input_[pos_] != '{') {
            error = "Expected '{'";
            return false;
        }
        ++pos_;
        skip_ws();
        JsonObject obj;
        if (pos_ < input_.size() && input_[pos_] == '}') {
            ++pos_;
            out.type = JsonValue::Type::Object;
            out.value = std::move(obj);
            return true;
        }
        while (pos_ < input_.size()) {
            skip_ws();
            std::string key;
            if (!parse_string(key, error)) {
                return false;
            }
            skip_ws();
            if (pos_ >= input_.size() || input_[pos_] != ':') {
                error = "Expected ':' after object key";
                return false;
            }
            ++pos_;
            JsonValue value;
            if (!parse_value(value, error)) {
                return false;
            }
            obj.emplace(std::move(key), std::move(value));
            skip_ws();
            if (pos_ >= input_.size()) {
                error = "Unexpected end of object";
                return false;
            }
            if (input_[pos_] == '}') {
                ++pos_;
                out.type = JsonValue::Type::Object;
                out.value = std::move(obj);
                return true;
            }
            if (input_[pos_] != ',') {
                error = "Expected ',' between object items";
                return false;
            }
            ++pos_;
        }
        error = "Unexpected end of object";
        return false;
    }

    bool parse_array(JsonValue& out, std::string& error) {
        if (input_[pos_] != '[') {
            error = "Expected '['";
            return false;
        }
        ++pos_;
        skip_ws();
        JsonArray arr;
        if (pos_ < input_.size() && input_[pos_] == ']') {
            ++pos_;
            out.type = JsonValue::Type::Array;
            out.value = std::move(arr);
            return true;
        }
        while (pos_ < input_.size()) {
            JsonValue value;
            if (!parse_value(value, error)) {
                return false;
            }
            arr.push_back(std::move(value));
            skip_ws();
            if (pos_ >= input_.size()) {
                error = "Unexpected end of array";
                return false;
            }
            if (input_[pos_] == ']') {
                ++pos_;
                out.type = JsonValue::Type::Array;
                out.value = std::move(arr);
                return true;
            }
            if (input_[pos_] != ',') {
                error = "Expected ',' between array items";
                return false;
            }
            ++pos_;
            skip_ws();
        }
        error = "Unexpected end of array";
        return false;
    }

    bool parse_string(std::string& out, std::string& error) {
        if (input_[pos_] != '"') {
            error = "Expected string";
            return false;
        }
        ++pos_;
        std::string result;
        while (pos_ < input_.size()) {
            const char c = input_[pos_++];
            if (c == '"') {
                out = std::move(result);
                return true;
            }
            if (c == '\\') {
                if (pos_ >= input_.size()) {
                    error = "Invalid escape sequence";
                    return false;
                }
                const char esc = input_[pos_++];
                switch (esc) {
                    case '"':
                        result.push_back('"');
                        break;
                    case '\\':
                        result.push_back('\\');
                        break;
                    case '/':
                        result.push_back('/');
                        break;
                    case 'b':
                        result.push_back('\b');
                        break;
                    case 'f':
                        result.push_back('\f');
                        break;
                    case 'n':
                        result.push_back('\n');
                        break;
                    case 'r':
                        result.push_back('\r');
                        break;
                    case 't':
                        result.push_back('\t');
                        break;
                    case 'u':
                        if (pos_ + 3 >= input_.size()) {
                            error = "Invalid unicode escape";
                            return false;
                        }
                        // Skip \uXXXX (ASCII subset only).
                        pos_ += 4;
                        result.push_back('?');
                        break;
                    default:
                        error = "Invalid escape sequence";
                        return false;
                }
            } else {
                result.push_back(c);
            }
        }
        error = "Unterminated string";
        return false;
    }

    bool parse_number(double& out, std::string& error) {
        const char* start = input_.c_str() + pos_;
        char* end = nullptr;
        out = std::strtod(start, &end);
        if (end == start) {
            error = "Invalid number";
            return false;
        }
        pos_ = static_cast<size_t>(end - input_.c_str());
        return true;
    }

    bool match_literal(const char* literal) {
        const size_t len = std::strlen(literal);
        if (input_.compare(pos_, len, literal) == 0) {
            pos_ += len;
            return true;
        }
        return false;
    }

    void skip_ws() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }

    std::string input_;
    size_t pos_{0};
};

const JsonValue* get(const JsonObject& obj, const std::string& key) {
    const auto it = obj.find(key);
    if (it == obj.end()) {
        return nullptr;
    }
    return &it->second;
}

std::optional<std::string> get_string(const JsonObject& obj, const std::string& key) {
    const auto* value = get(obj, key);
    if (!value) {
        return std::nullopt;
    }
    if (const auto* s = value->as_string()) {
        return *s;
    }
    return std::nullopt;
}

std::optional<std::string> first_string(const JsonObject& obj, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (const auto value = get_string(obj, key); value.has_value()) {
            return value;
        }
    }
    return std::nullopt;
}

std::unordered_map<std::string, std::string> collect_hierarchy_parent_map(const JsonObject& root_obj) {
    std::unordered_map<std::string, std::string> parent_map;
    const auto* hierarchy_val = get(root_obj, "hierarchy");
    if (hierarchy_val == nullptr) {
        return parent_map;
    }

    const JsonArray* groups = hierarchy_val->as_array();
    if (groups == nullptr) {
        if (const auto* hierarchy_obj = hierarchy_val->as_object()) {
            const auto* groups_val = get(*hierarchy_obj, "groups");
            groups = groups_val == nullptr ? nullptr : groups_val->as_array();
        }
    }
    if (groups == nullptr) {
        return parent_map;
    }

    for (const auto& group_val : *groups) {
        const auto* group = group_val.as_object();
        if (group == nullptr) {
            continue;
        }
        const auto group_id = get_string(*group, "id");
        if (!group_id.has_value() || group_id->empty()) {
            continue;
        }
        const auto* children_val = get(*group, "children");
        const auto* children = children_val == nullptr ? nullptr : children_val->as_array();
        if (children == nullptr) {
            continue;
        }
        for (const auto& child_val : *children) {
            if (const auto* child = child_val.as_string(); child != nullptr && !child->empty()) {
                parent_map.emplace(*child, *group_id);
            }
        }
    }
    return parent_map;
}

void load_device_parent(const JsonObject& obj,
                        const std::string& id,
                        const std::unordered_map<std::string, std::string>& hierarchy_parent_map,
                        Device& dev) {
    if (const auto parent = first_string(obj, {"parent", "parent_group", "machine", "machine_id"});
        parent.has_value()) {
        dev.parent = *parent;
        return;
    }
    const auto it = hierarchy_parent_map.find(id);
    if (it != hierarchy_parent_map.end()) {
        dev.parent = it->second;
    }
}

std::optional<double> get_number(const JsonObject& obj, const std::string& key) {
    const auto* value = get(obj, key);
    if (!value) {
        return std::nullopt;
    }
    if (const auto* n = value->as_number()) {
        return *n;
    }
    return std::nullopt;
}

std::optional<bool> get_bool(const JsonObject& obj, const std::string& key) {
    const auto* value = get(obj, key);
    if (!value) {
        return std::nullopt;
    }
    if (const auto* b = std::get_if<bool>(&value->value)) {
        return *b;
    }
    return std::nullopt;
}

std::optional<int> get_int(const JsonObject& obj, const std::string& key) {
    const auto num = get_number(obj, key);
    if (!num.has_value()) {
        return std::nullopt;
    }
    if (*num < static_cast<double>(std::numeric_limits<int>::min()) ||
        *num > static_cast<double>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }
    return static_cast<int>(*num);
}

std::string uppercase_ascii(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

std::string canonical_device_type(std::string type) {
    return uppercase_ascii(std::move(type));
}

std::string canonical_feature_name(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

std::string canonical_communication_group(std::string value) {
    value = canonical_feature_name(std::move(value));
    if (value.empty() || value == "p2p" || value == "send" || value == "recv") {
        return "native";
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

std::string canonical_endpoint_pair_key(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '_' || ch == ' ') {
            ch = '-';
        }
    }
    constexpr std::string_view cpu_gpu_prefix = "cpu-gpu-";
    if (value.starts_with(cpu_gpu_prefix)) {
        return "gpu-cpu-" + value.substr(cpu_gpu_prefix.size());
    }
    return value;
}

bool append_string_set_field(const JsonObject& obj,
                             const std::string& key,
                             std::unordered_set<std::string>& out,
                             std::string* error) {
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return true;
    }
    const auto* arr = value->as_array();
    if (arr == nullptr) {
        if (error) {
            *error = "'" + key + "' must be an array of strings";
        }
        return false;
    }
    for (const auto& item : *arr) {
        const auto* str = item.as_string();
        if (str == nullptr) {
            if (error) {
                *error = "'" + key + "' must contain only strings";
            }
            return false;
        }
        out.insert(canonical_feature_name(*str));
    }
    return true;
}

bool append_string_array_field(const JsonObject& obj,
                               const std::string& key,
                               std::vector<std::string>& out,
                               std::string* error) {
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return true;
    }
    const auto* arr = value->as_array();
    if (arr == nullptr) {
        if (error) {
            *error = "'" + key + "' must be an array of strings";
        }
        return false;
    }
    for (const auto& item : *arr) {
        const auto* str = item.as_string();
        if (str == nullptr) {
            if (error) {
                *error = "'" + key + "' must contain only strings";
            }
            return false;
        }
        out.push_back(*str);
    }
    return true;
}

void add_default_features(Device& dev) {
    const auto type = canonical_device_type(dev.type);
    if (type == "CPU") {
        dev.features.insert("scalar_cores");
        dev.features.insert("simd");
        dev.features.insert("cache_hierarchy");
        dev.features.insert("low_latency_launch");
        dev.features.insert("branching");
        dev.features.insert("irregular_memory");
    } else if (type == "GPU") {
        dev.features.insert("simt");
        dev.features.insert("warp_execution");
        dev.features.insert("high_bandwidth_memory");
        dev.features.insert("coalesced_memory");
        dev.features.insert("massive_parallelism");
        dev.features.insert("device_memory");
    }
}

bool load_feature_fields(const JsonObject& obj, Device& dev, std::string* error) {
    return append_string_set_field(obj, "features", dev.features, error) &&
           append_string_set_field(obj, "supported_precisions", dev.supported_precisions, error) &&
           append_string_set_field(obj, "precisions", dev.supported_precisions, error);
}

const JsonObject* choose_capability(const JsonArray& capabilities,
                                    const std::optional<std::string>& default_id = std::nullopt) {
    const JsonObject* first = nullptr;
    const JsonObject* marked_default = nullptr;
    for (const auto& value : capabilities) {
        const auto* obj = value.as_object();
        if (obj == nullptr) {
            continue;
        }
        if (first == nullptr) {
            first = obj;
        }
        if (default_id.has_value()) {
            const auto id = get_string(*obj, "id");
            if (id.has_value() && *id == *default_id) {
                return obj;
            }
        }
        const auto is_default = get_bool(*obj, "default");
        if (is_default.value_or(false) && marked_default == nullptr) {
            marked_default = obj;
        }
    }
    return marked_default != nullptr ? marked_default : first;
}

std::optional<std::string> default_compute_capability_id(const JsonObject& node_obj) {
    const auto* defaults_val = get(node_obj, "defaults");
    const auto* defaults = defaults_val == nullptr ? nullptr : defaults_val->as_object();
    if (defaults == nullptr) {
        return std::nullopt;
    }
    const auto* compute_val = get(*defaults, "compute");
    const auto* compute = compute_val == nullptr ? nullptr : compute_val->as_object();
    if (compute == nullptr) {
        return std::nullopt;
    }
    for (const auto& item : *compute) {
        if (const auto* id = item.second.as_string()) {
            return *id;
        }
    }
    return std::nullopt;
}

std::optional<std::string> default_memory_capability_id(const JsonObject& node_obj) {
    const auto* defaults_val = get(node_obj, "defaults");
    const auto* defaults = defaults_val == nullptr ? nullptr : defaults_val->as_object();
    if (defaults == nullptr) {
        return std::nullopt;
    }
    return get_string(*defaults, "memory");
}

std::optional<double> first_number(const JsonObject& obj, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (const auto value = get_number(obj, key); value.has_value()) {
            return value;
        }
    }
    return std::nullopt;
}

std::optional<int> first_int(const JsonObject& obj, std::initializer_list<const char*> keys) {
    for (const char* key : keys) {
        if (const auto value = get_int(obj, key); value.has_value()) {
            return value;
        }
    }
    return std::nullopt;
}

double latency_ms_from_link(const JsonObject& link_obj) {
    if (const auto latency_ms = get_number(link_obj, "latency_ms"); latency_ms.has_value()) {
        return *latency_ms;
    }
    if (const auto latency_ns = get_number(link_obj, "latency_ns"); latency_ns.has_value()) {
        return *latency_ns / 1e6;
    }
    return 0.0;
}

double memory_latency_ms_from(const JsonObject& obj) {
    if (const auto latency_ms = first_number(obj, {"memory_latency_ms", "mem_latency_ms"}); latency_ms.has_value()) {
        return *latency_ms;
    }
    if (const auto latency_ns = first_number(obj, {"memory_latency_ns", "mem_latency_ns"}); latency_ns.has_value()) {
        return *latency_ns / 1e6;
    }
    return 0.0;
}

bool read_json_file(const std::filesystem::path& path, JsonValue& root, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) {
            *error = "Failed to open " + path.string();
        }
        return false;
    }
    std::ostringstream oss;
    oss << in.rdbuf();

    JsonParser parser(oss.str());
    std::string parse_error;
    if (!parser.parse(root, parse_error)) {
        if (error) {
            *error = parse_error;
        }
        return false;
    }
    return true;
}

bool validate_extra_latency_us(double value, const std::string& key, std::string* error) {
    if (!std::isfinite(value) || value < 0.0) {
        if (error) {
            *error = key + " must be a non-negative finite number of microseconds";
        }
        return false;
    }
    return true;
}

std::optional<double> optional_extra_latency_us(const JsonObject& root,
                                                const std::string& key,
                                                std::string* error) {
    const auto* raw = get(root, key);
    if (raw == nullptr) {
        return std::nullopt;
    }
    const auto* value = raw->as_number();
    if (value == nullptr) {
        if (error) {
            *error = key + " must be numeric";
        }
        return std::nullopt;
    }
    if (!validate_extra_latency_us(*value, key, error)) {
        return std::nullopt;
    }
    return *value;
}

bool load_endpoint_extra_latency_config_file(const std::filesystem::path& path,
                                             HardwareTopology::EndpointExtraLatencyConfig& config,
                                             std::string* error) {
    JsonValue root;
    if (!read_json_file(path, root, error)) {
        return false;
    }
    const auto* root_obj = root.as_object();
    if (root_obj == nullptr) {
        if (error) {
            *error = path.string() + " root must be a JSON object";
        }
        return false;
    }

    const auto gpu_gpu = optional_extra_latency_us(*root_obj, "gpu-gpu-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    const auto gpu_cpu = optional_extra_latency_us(*root_obj, "gpu-cpu-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    const auto gpu_gpu_intra = optional_extra_latency_us(*root_obj, "gpu-gpu-intra-machine-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    const auto gpu_gpu_inter = optional_extra_latency_us(*root_obj, "gpu-gpu-inter-machine-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    const auto gpu_cpu_intra = optional_extra_latency_us(*root_obj, "gpu-cpu-intra-machine-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    const auto gpu_cpu_inter = optional_extra_latency_us(*root_obj, "gpu-cpu-inter-machine-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    const auto cpu_cpu_intra = optional_extra_latency_us(*root_obj, "cpu-cpu-intra-machine-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }
    const auto cpu_cpu_inter = optional_extra_latency_us(*root_obj, "cpu-cpu-inter-machine-extra-latency-us", error);
    if (error != nullptr && !error->empty()) {
        return false;
    }

    config.gpu_gpu_intra_machine_us = gpu_gpu_intra.value_or(gpu_gpu.value_or(0.0));
    config.gpu_gpu_inter_machine_us = gpu_gpu_inter.value_or(gpu_gpu.value_or(0.0));
    config.gpu_cpu_intra_machine_us = gpu_cpu_intra.value_or(gpu_cpu.value_or(0.0));
    config.gpu_cpu_inter_machine_us = gpu_cpu_inter.value_or(gpu_cpu.value_or(0.0));
    config.cpu_cpu_intra_machine_us = cpu_cpu_intra.value_or(0.0);
    config.cpu_cpu_inter_machine_us = cpu_cpu_inter.value_or(0.0);
    return true;
}

bool validate_operator_scale(double value, const std::string& field_name, std::string* error) {
    if (!std::isfinite(value) || value <= 0.0) {
        if (error) {
            *error = field_name + " must be a positive finite number";
        }
        return false;
    }
    return true;
}

bool validate_operator_overhead_us(double value, const std::string& field_name, std::string* error) {
    if (!std::isfinite(value) || value < 0.0) {
        if (error) {
            *error = field_name + " must be a non-negative finite number of microseconds";
        }
        return false;
    }
    return true;
}

bool validate_max_num_ops(double value, const std::string& field_name, std::string* error) {
    if (!std::isfinite(value) || value <= 0.0) {
        if (error) {
            *error = field_name + " must be a positive finite number";
        }
        return false;
    }
    return true;
}

bool parse_operator_scales(const JsonObject& root,
                           OperatorCostScaleTable& scales,
                           const std::string& source_name,
                           const std::string& dashed_name,
                           const std::string& underscored_name,
                           std::string* error) {
    const auto* scale_value = get(root, dashed_name);
    if (scale_value == nullptr) {
        scale_value = get(root, underscored_name);
    }
    if (scale_value == nullptr) {
        return true;
    }
    if (scale_value->type == JsonValue::Type::Null) {
        return true;
    }
    const auto* scale_root = scale_value->as_object();
    if (scale_root == nullptr) {
        if (error) {
            *error = dashed_name + " in " + source_name + " must be an object";
        }
        return false;
    }

    for (const auto& kv : *scale_root) {
        const auto* item = kv.second.as_object();
        if (item == nullptr) {
            if (error) {
                *error = dashed_name + "." + kv.first + " in " + source_name + " must be an object";
            }
            return false;
        }
        OperatorCostScale scale;
        if (const auto* raw = get(*item, "bandwidth_scale"); raw != nullptr) {
            const auto* value = raw->as_number();
            if (value == nullptr) {
                if (error) {
                    *error = "bandwidth_scale for " + kv.first + " in " + source_name + " must be numeric";
                }
                return false;
            }
            if (!validate_operator_scale(*value, "bandwidth_scale for " + kv.first, error)) {
                return false;
            }
            scale.bandwidth_scale = *value;
        }
        if (const auto* raw = get(*item, "flops_scale"); raw != nullptr) {
            const auto* value = raw->as_number();
            if (value == nullptr) {
                if (error) {
                    *error = "flops_scale for " + kv.first + " in " + source_name + " must be numeric";
                }
                return false;
            }
            if (!validate_operator_scale(*value, "flops_scale for " + kv.first, error)) {
                return false;
            }
            scale.flops_scale = *value;
        }
        if (const auto* raw = get(*item, "launch_overhead_us"); raw != nullptr) {
            const auto* value = raw->as_number();
            if (value == nullptr) {
                if (error) {
                    *error = "launch_overhead_us for " + kv.first + " in " + source_name + " must be numeric";
                }
                return false;
            }
            if (!validate_operator_overhead_us(*value, "launch_overhead_us for " + kv.first, error)) {
                return false;
            }
            scale.launch_overhead_us = *value;
        }
        if (const auto* raw = get(*item, "segments"); raw != nullptr) {
            const auto* segments = raw->as_array();
            if (segments == nullptr) {
                if (error) {
                    *error = "segments for " + kv.first + " in " + source_name + " must be an array";
                }
                return false;
            }
            for (const auto& raw_segment : *segments) {
                const auto* segment_obj = raw_segment.as_object();
                if (segment_obj == nullptr) {
                    if (error) {
                        *error = "segments for " + kv.first + " in " + source_name + " must contain objects";
                    }
                    return false;
                }
                OperatorCostScale::Segment segment;
                segment.bandwidth_scale = scale.bandwidth_scale;
                segment.flops_scale = scale.flops_scale;
                segment.launch_overhead_us = scale.launch_overhead_us;
                if (const auto* max_raw = get(*segment_obj, "max_num_ops"); max_raw != nullptr &&
                    max_raw->type != JsonValue::Type::Null) {
                    const auto* value = max_raw->as_number();
                    if (value == nullptr) {
                        if (error) {
                            *error = "max_num_ops for " + kv.first + " in " + source_name + " must be numeric";
                        }
                        return false;
                    }
                    if (!validate_max_num_ops(*value, "max_num_ops for " + kv.first, error)) {
                        return false;
                    }
                    segment.max_num_ops = *value;
                }
                if (const auto* bw_raw = get(*segment_obj, "bandwidth_scale"); bw_raw != nullptr) {
                    const auto* value = bw_raw->as_number();
                    if (value == nullptr) {
                        if (error) {
                            *error = "bandwidth_scale for " + kv.first + " segment in " + source_name + " must be numeric";
                        }
                        return false;
                    }
                    if (!validate_operator_scale(*value, "bandwidth_scale for " + kv.first + " segment", error)) {
                        return false;
                    }
                    segment.bandwidth_scale = *value;
                }
                if (const auto* flops_raw = get(*segment_obj, "flops_scale"); flops_raw != nullptr) {
                    const auto* value = flops_raw->as_number();
                    if (value == nullptr) {
                        if (error) {
                            *error = "flops_scale for " + kv.first + " segment in " + source_name + " must be numeric";
                        }
                        return false;
                    }
                    if (!validate_operator_scale(*value, "flops_scale for " + kv.first + " segment", error)) {
                        return false;
                    }
                    segment.flops_scale = *value;
                }
                if (const auto* launch_raw = get(*segment_obj, "launch_overhead_us"); launch_raw != nullptr) {
                    const auto* value = launch_raw->as_number();
                    if (value == nullptr) {
                        if (error) {
                            *error = "launch_overhead_us for " + kv.first + " segment in " + source_name + " must be numeric";
                        }
                        return false;
                    }
                    if (!validate_operator_overhead_us(*value, "launch_overhead_us for " + kv.first + " segment", error)) {
                        return false;
                    }
                    segment.launch_overhead_us = *value;
                }
                scale.segments.push_back(segment);
            }
            std::stable_sort(
                scale.segments.begin(),
                scale.segments.end(),
                [](const auto& lhs, const auto& rhs) {
                    if (!lhs.max_num_ops.has_value()) {
                        return false;
                    }
                    if (!rhs.max_num_ops.has_value()) {
                        return true;
                    }
                    return *lhs.max_num_ops < *rhs.max_num_ops;
                });
        }
        scales[canonical_feature_name(kv.first)] = scale;
    }
    return true;
}

bool load_operator_calibration(const JsonObject& obj,
                               Device& dev,
                               const std::string& source_name,
                               std::string* error) {
    const auto* calibration_val = get(obj, "calibration");
    if (calibration_val == nullptr || calibration_val->type == JsonValue::Type::Null) {
        return true;
    }
    const auto* calibration = calibration_val->as_object();
    if (calibration == nullptr) {
        if (error) {
            *error = "calibration in " + source_name + " must be an object";
        }
        return false;
    }
    const auto before = dev.calibration_operator_cost_scales.size();
    if (!parse_operator_scales(*calibration,
                               dev.calibration_operator_cost_scales,
                               source_name + ".calibration",
                               "operator-scales",
                               "operator_scales",
                               error)) {
        return false;
    }
    if (dev.calibration_operator_cost_scales.size() != before) {
        dev.has_calibration_operator_cost_scales = true;
    }
    return true;
}

bool parse_nonnegative_size_key(const std::string& key, std::size_t& out) {
    if (key.empty()) {
        return false;
    }
    std::size_t value = 0;
    for (const char ch : key) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
        const std::size_t digit = static_cast<std::size_t>(ch - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    out = value;
    return true;
}

bool parse_communication_scale_leaf(const JsonObject& obj,
                                    CommunicationCostScale& scale,
                                    const std::string& source_name,
                                    std::string* error) {
    if (const auto* raw = get(obj, "bandwidth_scale"); raw != nullptr) {
        const auto* value = raw->as_number();
        if (value == nullptr) {
            if (error) {
                *error = "bandwidth_scale in " + source_name + " must be numeric";
            }
            return false;
        }
        if (!validate_operator_scale(*value, "bandwidth_scale in " + source_name, error)) {
            return false;
        }
        scale.bandwidth_scale = *value;
    }
    if (const auto* raw = get(obj, "extra_latency_us"); raw != nullptr) {
        const auto* value = raw->as_number();
        if (value == nullptr) {
            if (error) {
                *error = "extra_latency_us in " + source_name + " must be numeric";
            }
            return false;
        }
        if (!validate_operator_overhead_us(*value, "extra_latency_us in " + source_name, error)) {
            return false;
        }
        scale.extra_latency_us = *value;
    }
    return true;
}

bool parse_communication_group_scales(const JsonObject& root,
                                      CommunicationGroupScaleTable& out,
                                      const std::string& source_name,
                                      std::string* error) {
    for (const auto& group_kv : root) {
        const auto* group_obj = group_kv.second.as_object();
        if (group_obj == nullptr) {
            if (error) {
                *error = source_name + "." + group_kv.first + " must be an object";
            }
            return false;
        }
        auto& endpoint_pairs = out[canonical_communication_group(group_kv.first)];
        for (const auto& pair_kv : *group_obj) {
            const auto* pair_obj = pair_kv.second.as_object();
            if (pair_obj == nullptr) {
                if (error) {
                    *error = source_name + "." + group_kv.first + "." + pair_kv.first + " must be an object";
                }
                return false;
            }
            const auto* by_hops_val = get(*pair_obj, "by_route_hops");
            const auto* by_hops = by_hops_val == nullptr ? nullptr : by_hops_val->as_object();
            if (by_hops == nullptr) {
                if (error) {
                    *error = source_name + "." + group_kv.first + "." + pair_kv.first +
                             ".by_route_hops must be an object";
                }
                return false;
            }
            auto& route_hops = endpoint_pairs[canonical_endpoint_pair_key(pair_kv.first)];
            for (const auto& hops_kv : *by_hops) {
                std::size_t hops = 0;
                if (!parse_nonnegative_size_key(hops_kv.first, hops)) {
                    if (error) {
                        *error = source_name + " route hop key '" + hops_kv.first +
                                 "' must be a non-negative integer string";
                    }
                    return false;
                }
                const auto* scale_obj = hops_kv.second.as_object();
                if (scale_obj == nullptr) {
                    if (error) {
                        *error = source_name + " route hop '" + hops_kv.first + "' must be an object";
                    }
                    return false;
                }
                CommunicationCostScale scale;
                if (!parse_communication_scale_leaf(
                        *scale_obj,
                        scale,
                        source_name + "." + group_kv.first + "." + pair_kv.first + "." + hops_kv.first,
                        error)) {
                    return false;
                }
                route_hops[hops] = scale;
            }
        }
    }
    return true;
}

bool load_link_calibration(const JsonObject& obj, Link& link, const std::string& source_name, std::string* error) {
    const auto* calibration_val = get(obj, "calibration");
    if (calibration_val == nullptr || calibration_val->type == JsonValue::Type::Null) {
        return true;
    }
    const auto* calibration = calibration_val->as_object();
    if (calibration == nullptr) {
        if (error) {
            *error = "calibration in " + source_name + " must be an object";
        }
        return false;
    }

    if (const auto* raw = get(*calibration, "hardware_route_communication_scales"); raw != nullptr) {
        const auto* route_scales = raw->as_object();
        if (route_scales == nullptr) {
            if (error) {
                *error = "hardware_route_communication_scales in " + source_name + " must be an object";
            }
            return false;
        }
        for (const auto& hardware_kv : *route_scales) {
            const auto* group_obj = hardware_kv.second.as_object();
            if (group_obj == nullptr) {
                if (error) {
                    *error = source_name + ".hardware_route_communication_scales." + hardware_kv.first +
                             " must be an object";
                }
                return false;
            }
            auto& groups = link.hardware_route_communication_scales[hardware_kv.first];
            if (!parse_communication_group_scales(
                    *group_obj,
                    groups,
                    source_name + ".hardware_route_communication_scales." + hardware_kv.first,
                    error)) {
                return false;
            }
        }
    }

    if (const auto* raw = get(*calibration, "communication_scales"); raw != nullptr) {
        const auto* group_obj = raw->as_object();
        if (group_obj == nullptr) {
            if (error) {
                *error = "communication_scales in " + source_name + " must be an object";
            }
            return false;
        }
        auto& groups = link.hardware_route_communication_scales[link.id];
        if (!parse_communication_group_scales(
                *group_obj,
                groups,
                source_name + ".communication_scales",
                error)) {
            return false;
        }
    }

    return true;
}

bool load_operator_scales_config_file(const std::filesystem::path& path,
                                      OperatorCostScaleTable& scales,
                                      const std::string& dashed_name,
                                      const std::string& underscored_name,
                                      std::string* error) {
    JsonValue root;
    if (!read_json_file(path, root, error)) {
        return false;
    }
    const auto* root_obj = root.as_object();
    if (root_obj == nullptr) {
        if (error) {
            *error = path.string() + " root must be a JSON object";
        }
        return false;
    }
    return parse_operator_scales(*root_obj, scales, path.string(), dashed_name, underscored_name, error);
}

std::optional<std::filesystem::path> find_global_simulator_config_from(std::filesystem::path start) {
    if (start.empty()) {
        return std::nullopt;
    }
    std::error_code ec;
    if (!std::filesystem::is_directory(start, ec)) {
        start = start.parent_path();
    }
    if (start.empty()) {
        return std::nullopt;
    }

    auto current = std::filesystem::absolute(start, ec);
    if (ec) {
        current = start;
    }
    while (!current.empty()) {
        const auto repo_candidate = current / "simulator" / "simulator_global_config.json";
        if (std::filesystem::exists(repo_candidate, ec)) {
            return repo_candidate;
        }
        const auto simulator_candidate = current / "simulator_global_config.json";
        if (std::filesystem::exists(simulator_candidate, ec)) {
            return simulator_candidate;
        }
        const auto parent = current.parent_path();
        if (parent == current) {
            break;
        }
        current = parent;
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> find_global_simulator_config(const std::string& hardware_path) {
    if (const auto from_hardware = find_global_simulator_config_from(std::filesystem::path(hardware_path));
        from_hardware.has_value()) {
        return from_hardware;
    }
    if (const auto from_cwd = find_global_simulator_config_from(std::filesystem::current_path());
        from_cwd.has_value()) {
        return from_cwd;
    }
    return find_global_simulator_config_from(std::filesystem::path(__FILE__));
}

bool load_default_endpoint_extra_latency_config(const std::string& hardware_path,
                                                HardwareTopology& out,
                                                std::string* error) {
    const auto config_path = find_global_simulator_config(hardware_path);
    if (!config_path.has_value()) {
        return true;
    }
    HardwareTopology::EndpointExtraLatencyConfig config;
    std::string local_error;
    if (!load_endpoint_extra_latency_config_file(*config_path, config, &local_error)) {
        if (error) {
            *error = "Failed to load " + config_path->string() + ": " + local_error;
        }
        return false;
    }
    out.set_endpoint_extra_latency_config(config);
    return true;
}

bool load_default_operator_scales_config(const std::string& hardware_path,
                                         HardwareTopology& out,
                                         std::string* error) {
    const auto config_path = find_global_simulator_config(hardware_path);
    if (!config_path.has_value()) {
        return true;
    }
    OperatorCostScaleTable cpu_scales;
    OperatorCostScaleTable gpu_scales;
    std::string local_error;
    if (!load_operator_scales_config_file(
            *config_path, cpu_scales, "cpu-operator-scales", "cpu_operator_scales", &local_error)) {
        if (error) {
            *error = "Failed to load " + config_path->string() + ": " + local_error;
        }
        return false;
    }
    if (!load_operator_scales_config_file(
            *config_path, gpu_scales, "gpu-operator-scales", "gpu_operator_scales", &local_error)) {
        if (error) {
            *error = "Failed to load " + config_path->string() + ": " + local_error;
        }
        return false;
    }
    out.set_cpu_operator_scales(std::move(cpu_scales));
    out.set_gpu_operator_scales(std::move(gpu_scales));
    return true;
}

bool load_legacy_devices(const JsonArray& devices,
                         const std::unordered_map<std::string, std::string>& hierarchy_parent_map,
                         HardwareTopology& out,
                         std::string* error) {
    for (const auto& item : devices) {
        const auto* dev_obj = item.as_object();
        if (!dev_obj) {
            if (error) {
                *error = "Device entry must be an object";
            }
            return false;
        }
        const auto id = get_string(*dev_obj, "id");
        const auto name = get_string(*dev_obj, "name");
        const auto type = get_string(*dev_obj, "type");
        const auto peak = get_number(*dev_obj, "peak_gflops");
        const auto mem_bw = get_number(*dev_obj, "mem_bw_gbps");
        const auto max_concurrent = get_int(*dev_obj, "max_concurrent");
        if (!id || !name || !type || !peak || !mem_bw || !max_concurrent) {
            if (error) {
                *error = "Device entry missing required fields";
            }
            return false;
        }
        Device dev;
        dev.id = *id;
        dev.name = *name;
        dev.type = canonical_device_type(*type);
        load_device_parent(*dev_obj, dev.id, hierarchy_parent_map, dev);
        dev.peak_gflops = *peak;
        dev.mem_bw_gbps = *mem_bw;
        dev.mem_latency_ms = memory_latency_ms_from(*dev_obj);
        dev.max_concurrent = *max_concurrent;
        dev.compute_capable = true;
        add_default_features(dev);
        if (!load_feature_fields(*dev_obj, dev, error)) {
            return false;
        }
        if (!load_operator_calibration(*dev_obj, dev, "device " + dev.id, error)) {
            return false;
        }
        out.add_device(std::move(dev));
    }
    return true;
}

bool load_capability_nodes(const JsonArray& nodes,
                           const std::unordered_map<std::string, std::string>& hierarchy_parent_map,
                           HardwareTopology& out,
                           std::string* error) {
    for (const auto& item : nodes) {
        const auto* node_obj = item.as_object();
        if (!node_obj) {
            if (error) {
                *error = "Node entry must be an object";
            }
            return false;
        }
        const auto id = get_string(*node_obj, "id");
        if (!id) {
            if (error) {
                *error = "Node entry missing 'id'";
            }
            return false;
        }

        const auto* capabilities_val = get(*node_obj, "capabilities");
        const auto* capabilities = capabilities_val == nullptr ? nullptr : capabilities_val->as_object();
        if (capabilities == nullptr) {
            if (error) {
                *error = "Node entry missing 'capabilities'";
            }
            return false;
        }

        const auto* compute_val = get(*capabilities, "compute");
        const auto* compute_array = compute_val == nullptr ? nullptr : compute_val->as_array();
        const auto* memory_val = get(*capabilities, "memory");
        const auto* memory_array = memory_val == nullptr ? nullptr : memory_val->as_array();
        const auto* network_val = get(*capabilities, "network");
        const auto* network = network_val == nullptr ? nullptr : network_val->as_object();

        const JsonObject* compute = nullptr;
        if (compute_array != nullptr) {
            compute = choose_capability(*compute_array, default_compute_capability_id(*node_obj));
        }
        const JsonObject* memory = nullptr;
        if (memory_array != nullptr) {
            memory = choose_capability(*memory_array, default_memory_capability_id(*node_obj));
        }

        Device dev;
        dev.id = *id;
        dev.name = get_string(*node_obj, "name").value_or(*id);
        load_device_parent(*node_obj, dev.id, hierarchy_parent_map, dev);
        dev.max_concurrent = 1;

        if (compute != nullptr) {
            const auto kind = get_string(*compute, "kind").value_or("compute");
            const auto peak_tflops = get_number(*compute, "peak_tflops");
            const auto peak_gflops = get_number(*compute, "peak_gflops");
            dev.type = canonical_device_type(kind);
            dev.peak_gflops = peak_gflops.value_or(peak_tflops.value_or(0.0) * 1000.0);
            dev.mem_bw_gbps = first_number(*compute, {"memory_bw_gbps", "mem_bw_gbps", "bandwidth_gbps"}).value_or(0.0);
            if (dev.mem_bw_gbps <= 0.0 && memory != nullptr) {
                dev.mem_bw_gbps = get_number(*memory, "bandwidth_gbps").value_or(0.0);
            }
            dev.mem_latency_ms = memory_latency_ms_from(*compute);
            if (dev.mem_latency_ms <= 0.0 && memory != nullptr) {
                dev.mem_latency_ms = memory_latency_ms_from(*memory);
            }
            dev.max_concurrent = first_int(*compute, {"max_parallelism", "max_concurrent", "slots"}).value_or(1);
            dev.compute_capable = true;
        } else if (memory != nullptr) {
            const auto kind = get_string(*memory, "kind").value_or("memory");
            dev.type = canonical_device_type(kind);
            dev.mem_bw_gbps = get_number(*memory, "bandwidth_gbps").value_or(0.0);
            dev.mem_latency_ms = memory_latency_ms_from(*memory);
            if (network != nullptr) {
                dev.max_concurrent = get_int(*network, "comm_slots").value_or(1);
            }
            dev.compute_capable = false;
        } else if (network != nullptr) {
            const auto kind = get_string(*network, "kind").value_or("network");
            dev.type = canonical_device_type(kind);
            dev.max_concurrent = get_int(*network, "comm_slots").value_or(1);
            dev.compute_capable = false;
        } else {
            if (error) {
                *error = "Node entry has no supported compute, memory, or network capability";
            }
            return false;
        }

        add_default_features(dev);
        if (!load_feature_fields(*node_obj, dev, error)) {
            return false;
        }
        if (compute != nullptr && !load_feature_fields(*compute, dev, error)) {
            return false;
        }
        if (memory != nullptr && !load_feature_fields(*memory, dev, error)) {
            return false;
        }
        if (network != nullptr && !load_feature_fields(*network, dev, error)) {
            return false;
        }
        if (compute != nullptr && !load_operator_calibration(*compute, dev, "node " + dev.id + " compute capability", error)) {
            return false;
        }

        if (dev.max_concurrent < 1) {
            dev.max_concurrent = 1;
        }
        out.add_device(std::move(dev));
    }
    return true;
}

bool load_links(const JsonObject& root_obj, HardwareTopology& out, std::string* error) {
    const auto* links_val = get(root_obj, "links");
    if (links_val == nullptr) {
        return true;
    }
    const auto* links = links_val->as_array();
    if (links == nullptr) {
        if (error) {
            *error = "Invalid 'links' array";
        }
        return false;
    }

    for (const auto& item : *links) {
        const auto* link_obj = item.as_object();
        if (!link_obj) {
            if (error) {
                *error = "Link entry must be an object";
            }
            return false;
        }
        const auto src = get_string(*link_obj, "src");
        const auto dst = get_string(*link_obj, "dst");
        const auto bw = first_number(*link_obj, {"bw_gbps", "bandwidth_gbps"});
        if (!src || !dst || !bw) {
            if (error) {
                *error = "Link entry missing required fields";
            }
            return false;
        }

        Link link;
        link.id = get_string(*link_obj, "id").value_or("");
        if (link.id.empty()) {
            link.id = "link_" + *src + "_to_" + *dst;
        }
        link.src = *src;
        link.dst = *dst;
        link.bw_gbps = *bw;
        link.latency_ms = latency_ms_from_link(*link_obj);
        if (!append_string_array_field(*link_obj, "hardware_route", link.hardware_route, error)) {
            return false;
        }
        if (!load_link_calibration(*link_obj, link, "link " + link.id, error)) {
            return false;
        }
        out.add_link(link);

        if (get_bool(*link_obj, "bidirectional").value_or(false)) {
            Link reverse = std::move(link);
            reverse.src = *dst;
            reverse.dst = *src;
            if (!reverse.id.empty()) {
                reverse.id += "_rev";
            }
            out.add_link(std::move(reverse));
        }
    }
    return true;
}

}  // namespace

bool load_from_json(const std::string& path, HardwareTopology& out, std::string* error) {
    JsonValue root;
    if (!read_json_file(path, root, error)) {
        return false;
    }

    const auto* root_obj = root.as_object();
    if (!root_obj) {
        if (error) {
            *error = "Root must be a JSON object";
        }
        return false;
    }

    if (const auto time_unit = get_string(*root_obj, "time_unit"); time_unit.has_value()) {
        out.set_time_unit(*time_unit);
    }

    const auto hierarchy_parent_map = collect_hierarchy_parent_map(*root_obj);
    const auto* devices_val = get(*root_obj, "devices");
    const auto* nodes_val = get(*root_obj, "nodes");
    if (devices_val != nullptr) {
        const auto* devices = devices_val->as_array();
        if (devices == nullptr) {
            if (error) {
                *error = "Missing or invalid 'devices' array";
            }
            return false;
        }
        if (!load_legacy_devices(*devices, hierarchy_parent_map, out, error)) {
            return false;
        }
    } else if (nodes_val != nullptr) {
        const auto* nodes = nodes_val->as_array();
        if (nodes == nullptr) {
            if (error) {
                *error = "Missing or invalid 'nodes' array";
            }
            return false;
        }
        if (!load_capability_nodes(*nodes, hierarchy_parent_map, out, error)) {
            return false;
        }
    } else {
        if (error) {
            *error = "Missing hardware topology 'devices' or 'nodes' array";
        }
        return false;
    }

    if (!load_links(*root_obj, out, error)) {
        return false;
    }
    if (!load_default_endpoint_extra_latency_config(path, out, error)) {
        return false;
    }
    if (!load_default_operator_scales_config(path, out, error)) {
        return false;
    }

    return true;
}

}  // namespace hardware_topology
