#include "workload/json_io.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "mapping/operator_catalog.h"

namespace workload {
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

std::optional<DType> parse_dtype(const std::string& value) {
    std::string normalized = value;
    for (char& ch : normalized) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    if (normalized == "fp16" || normalized == "float16" || normalized == "half") {
        return DType::FP16;
    }
    if (normalized == "bf16" || normalized == "bfloat16") {
        return DType::BF16;
    }
    if (normalized == "fp32" || normalized == "float32" || normalized == "f32") {
        return DType::FP32;
    }
    if (normalized == "fp64" || normalized == "float64" || normalized == "f64" || normalized == "double") {
        return DType::FP64;
    }
    if (normalized == "int8") {
        return DType::INT8;
    }
    if (normalized == "uint8") {
        return DType::UINT8;
    }
    if (normalized == "int32") {
        return DType::INT32;
    }
    if (normalized == "int64") {
        return DType::INT64;
    }
    return std::nullopt;
}

std::optional<DistKind> parse_dist_kind(const std::string& value) {
    if (value == "none") {
        return DistKind::NONE;
    }
    if (value == "replicated") {
        return DistKind::REPLICATED;
    }
    if (value == "block") {
        return DistKind::BLOCK;
    }
    if (value == "cyclic") {
        return DistKind::CYCLIC;
    }
    return std::nullopt;
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

std::optional<StorageFormat> parse_storage_format(const std::string& value) {
    const auto normalized = canonical_token(value);
    if (normalized == "dense") {
        return StorageFormat::DENSE;
    }
    if (normalized == "csr" || normalized == "sparse_csr") {
        return StorageFormat::CSR;
    }
    if (normalized == "csc" || normalized == "sparse_csc") {
        return StorageFormat::CSC;
    }
    if (normalized == "coo" || normalized == "coo_import" || normalized == "sparse_coo") {
        return StorageFormat::COO_IMPORT;
    }
    if (normalized == "bsr" || normalized == "sparse_bsr") {
        return StorageFormat::BSR;
    }
    if (normalized == "block_sparse" || normalized == "sparse_block") {
        return StorageFormat::BLOCK_SPARSE;
    }
    return std::nullopt;
}

AccessKind default_access_for_storage_format(StorageFormat storage) {
    switch (storage) {
        case StorageFormat::DENSE:
            return AccessKind::DENSE;
        case StorageFormat::CSR:
            return AccessKind::SPARSE_CSR;
        case StorageFormat::CSC:
            return AccessKind::SPARSE_CSC;
        case StorageFormat::COO_IMPORT:
            return AccessKind::SPARSE_COO;
        case StorageFormat::BSR:
        case StorageFormat::BLOCK_SPARSE:
            return AccessKind::SPARSE_BLOCK;
    }
    return AccessKind::DENSE;
}

std::optional<StorageFormat> storage_format_from_access(AccessKind access) {
    switch (access) {
        case AccessKind::SPARSE_CSR:
            return StorageFormat::CSR;
        case AccessKind::SPARSE_CSC:
            return StorageFormat::CSC;
        case AccessKind::SPARSE_COO:
            return StorageFormat::COO_IMPORT;
        case AccessKind::SPARSE_BLOCK:
            return StorageFormat::BLOCK_SPARSE;
        case AccessKind::DENSE:
        case AccessKind::ROW_WISE:
        case AccessKind::COL_WISE:
            return std::nullopt;
    }
    return std::nullopt;
}

std::optional<AccessKind> parse_access_kind(const std::string& value) {
    const auto normalized = canonical_token(value);
    if (normalized == "dense" || normalized == "contiguous") {
        return AccessKind::DENSE;
    }
    if (normalized == "sparse_csr" || normalized == "csr") {
        return AccessKind::SPARSE_CSR;
    }
    if (normalized == "sparse_csc" || normalized == "csc") {
        return AccessKind::SPARSE_CSC;
    }
    if (normalized == "sparse_coo" || normalized == "coo" || normalized == "coo_import") {
        return AccessKind::SPARSE_COO;
    }
    if (normalized == "sparse_block" || normalized == "sparse_bsr" || normalized == "bsr" ||
        normalized == "block_sparse") {
        return AccessKind::SPARSE_BLOCK;
    }
    if (normalized == "row_wise") {
        return AccessKind::ROW_WISE;
    }
    if (normalized == "col_wise") {
        return AccessKind::COL_WISE;
    }
    return std::nullopt;
}

bool parse_string_set_field(const JsonObject& obj,
                            const std::string& key,
                            std::unordered_set<std::string>& out,
                            std::string& error) {
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return true;
    }
    const auto* arr = value->as_array();
    if (arr == nullptr) {
        error = "'" + key + "' must be an array of strings";
        return false;
    }
    for (const auto& item : *arr) {
        const auto* str = item.as_string();
        if (str == nullptr) {
            error = "'" + key + "' must contain only strings";
            return false;
        }
        out.insert(canonical_token(*str));
    }
    return true;
}

bool parse_uint64_field(const JsonObject& obj,
                        const std::string& key,
                        std::optional<std::uint64_t>& out,
                        std::string& error) {
    const auto value = get_number(obj, key);
    if (!value.has_value()) {
        return true;
    }
    if (*value < 0.0 || !std::isfinite(*value)) {
        error = "Tensor " + key + " must be non-negative";
        return false;
    }
    double integral = 0.0;
    if (std::modf(*value, &integral) != 0.0) {
        error = "Tensor " + key + " must be integer";
        return false;
    }
    out = static_cast<std::uint64_t>(integral);
    return true;
}

bool is_sparse_storage(StorageFormat storage) {
    return storage == StorageFormat::CSR || storage == StorageFormat::CSC ||
           storage == StorageFormat::COO_IMPORT || storage == StorageFormat::BSR ||
           storage == StorageFormat::BLOCK_SPARSE;
}

bool parse_device_groups(const JsonArray& groups_array, std::vector<DeviceGroup>& groups, std::string& error) {
    groups.clear();
    groups.reserve(groups_array.size());
    std::unordered_set<std::string> seen;
    for (const auto& item : groups_array) {
        const auto* group_obj = item.as_object();
        if (!group_obj) {
            error = "Device group entry must be an object";
            return false;
        }
        const auto id = get_string(*group_obj, "id");
        if (!id || id->empty()) {
            error = "Device group missing 'id'";
            return false;
        }
        if (!seen.insert(*id).second) {
            error = "Duplicate device group id: " + *id;
            return false;
        }
        DeviceGroup group;
        group.id = *id;
        if (const auto* members_val = get(*group_obj, "members")) {
            if (const auto* members_str = members_val->as_string()) {
                if (*members_str == "all") {
                    group.members.push_back("all");
                } else {
                    error = "Device group 'members' string must be 'all'";
                    return false;
                }
            } else if (const auto* members_arr = members_val->as_array()) {
                for (const auto& member_item : *members_arr) {
                    const auto* member_str = member_item.as_string();
                    if (!member_str) {
                        error = "Device group members must be strings";
                        return false;
                    }
                    group.members.push_back(*member_str);
                }
            } else {
                error = "Device group 'members' must be array or 'all'";
                return false;
            }
        } else {
            error = "Device group missing 'members'";
            return false;
        }
        groups.push_back(std::move(group));
    }
    return true;
}

bool parse_tensors(const JsonArray& tensors_array, std::vector<Tensor>& tensors, std::string& error) {
    tensors.clear();
    tensors.reserve(tensors_array.size());
    std::unordered_set<std::string> seen_ids;
    for (const auto& item : tensors_array) {
        const auto* tensor_obj = item.as_object();
        if (!tensor_obj) {
            error = "Tensor entry must be an object";
            return false;
        }
        const auto id = get_string(*tensor_obj, "id");
        if (!id || id->empty()) {
            error = "Tensor entry missing 'id'";
            return false;
        }
        if (!seen_ids.insert(*id).second) {
            error = "Duplicate tensor id: " + *id;
            return false;
        }

        Tensor tensor;
        tensor.id = *id;
        tensor.name = get_string(*tensor_obj, "name").value_or(*id);

        if (const auto dtype_val = get_string(*tensor_obj, "dtype")) {
            const auto dtype = parse_dtype(*dtype_val);
            if (!dtype) {
                error = "Unsupported dtype: " + *dtype_val;
                return false;
            }
            tensor.dtype = *dtype;
        }

        const bool has_storage_format = get_string(*tensor_obj, "storage_format").has_value();
        if (const auto storage_val = get_string(*tensor_obj, "storage_format")) {
            const auto storage = parse_storage_format(*storage_val);
            if (!storage) {
                error = "Unsupported storage_format: " + *storage_val;
                return false;
            }
            tensor.storage_format = *storage;
            tensor.access_pattern = default_access_for_storage_format(*storage);
        }

        if (const auto* shape_val = get(*tensor_obj, "shape"); shape_val && shape_val->as_array()) {
            for (const auto& dim_item : *shape_val->as_array()) {
                const auto* dim_num = dim_item.as_number();
                if (!dim_num || !std::isfinite(*dim_num)) {
                    error = "Tensor shape must be numeric";
                    return false;
                }
                double integral = 0.0;
                if (std::modf(*dim_num, &integral) != 0.0) {
                    error = "Tensor shape must be integer";
                    return false;
                }
                tensor.shape.push_back(static_cast<std::int64_t>(integral));
            }
        }

        if (!parse_uint64_field(*tensor_obj, "num_elements", tensor.num_elements, error)) {
            return false;
        }
        if (!parse_uint64_field(*tensor_obj, "nnz", tensor.nonzero_elements, error)) {
            return false;
        }
        if (!tensor.nonzero_elements.has_value()) {
            if (!parse_uint64_field(*tensor_obj, "nonzero_elements", tensor.nonzero_elements, error)) {
                return false;
            }
        }
        if (!tensor.nonzero_elements.has_value()) {
            if (!parse_uint64_field(*tensor_obj, "nonzeros", tensor.nonzero_elements, error)) {
                return false;
            }
        }

        std::optional<double> size_bytes_val = get_number(*tensor_obj, "size_bytes");
        if (!size_bytes_val.has_value()) {
            size_bytes_val = get_number(*tensor_obj, "bytes");  // legacy alias
        }
        if (!size_bytes_val.has_value()) {
            error = "Tensor entry missing 'size_bytes'";
            return false;
        }
        if (*size_bytes_val < 0.0 || !std::isfinite(*size_bytes_val)) {
            error = "Tensor size_bytes must be non-negative";
            return false;
        }
        double size_integral = 0.0;
        if (std::modf(*size_bytes_val, &size_integral) != 0.0) {
            error = "Tensor size_bytes must be integer";
            return false;
        }
        tensor.size_bytes = static_cast<std::uint64_t>(size_integral);

        if (const auto* dist_val = get(*tensor_obj, "distribution"); dist_val && dist_val->as_object()) {
            const auto* dist_obj = dist_val->as_object();
            if (const auto kind_val = get_string(*dist_obj, "kind")) {
                const auto kind = parse_dist_kind(*kind_val);
                if (!kind) {
                    error = "Unsupported distribution kind: " + *kind_val;
                    return false;
                }
                tensor.distribution.kind = *kind;
            }
            if (const auto axis_val = get_int(*dist_obj, "axis")) {
                tensor.distribution.axis = *axis_val;
            }
            if (const auto group_val = get_string(*dist_obj, "group")) {
                tensor.distribution.group = *group_val;
            }
        }

        if (const auto access_val = get_string(*tensor_obj, "access_pattern")) {
            const auto access = parse_access_kind(*access_val);
            if (!access) {
                error = "Unsupported access_pattern: " + *access_val;
                return false;
            }
            tensor.access_pattern = *access;
            if (!has_storage_format) {
                if (const auto inferred_storage = storage_format_from_access(*access)) {
                    tensor.storage_format = *inferred_storage;
                }
            }
        }

        if (!tensor.nonzero_elements.has_value() && is_sparse_storage(tensor.storage_format) &&
            tensor.num_elements.has_value()) {
            tensor.nonzero_elements = tensor.num_elements;
        }

        if (const auto* repl_val = get(*tensor_obj, "replication"); repl_val && repl_val->as_object()) {
            const auto* repl_obj = repl_val->as_object();
            Replication repl;
            if (const auto mode_val = get_string(*repl_obj, "mode")) {
                if (*mode_val != "broadcast" && *mode_val != "cached") {
                    error = "Unsupported replication mode: " + *mode_val;
                    return false;
                }
                repl.mode = *mode_val;
            }
            if (!repl.mode.empty()) {
                tensor.replication = repl;
            }
        } else if (tensor.distribution.kind == DistKind::REPLICATED) {
            Replication repl;
            repl.mode = "cached";
            tensor.replication = repl;
        }

        if (const auto producer_val = get_int(*tensor_obj, "producer")) {
            tensor.producer_task = *producer_val;
        }

        // Optional explicit collective: either a bare string ("allreduce") or an
        // object {"type": "allreduce", ...}. Honoured by Workload::to_task_graph
        // to emit a real COMM_COLL_NODE of that type over all participant devices.
        if (const auto hint_str = get_string(*tensor_obj, "collective_hint")) {
            tensor.collective_hint = *hint_str;
        } else if (const auto* hint_val = get(*tensor_obj, "collective_hint");
                   hint_val && hint_val->as_object()) {
            if (const auto type_val = get_string(*hint_val->as_object(), "type")) {
                tensor.collective_hint = *type_val;
            }
        }

        tensors.push_back(std::move(tensor));
    }
    return true;
}

bool parse_tasks(const JsonArray& tasks_array, std::vector<Task>& tasks, std::string& error) {
    tasks.clear();
    tasks.reserve(tasks_array.size());
    std::unordered_set<int> seen_ids;
    std::unordered_set<std::string> seen_names;
    for (const auto& item : tasks_array) {
        const auto* task_obj = item.as_object();
        if (!task_obj) {
            error = "Task entry must be an object";
            return false;
        }
        const auto id = get_int(*task_obj, "id");
        const auto name = get_string(*task_obj, "name");
        const auto op = get_string(*task_obj, "op");
        if (!id || !name || !op) {
            error = "Task entry missing required fields";
            return false;
        }
        if (!seen_ids.insert(*id).second) {
            error = "Duplicate task id: " + std::to_string(*id);
            return false;
        }
        if (!seen_names.insert(*name).second) {
            error = "Duplicate task name: " + *name;
            return false;
        }

        Task task;
        task.id = *id;
        task.name = *name;
        task.op = *op;
        if (const auto level_val = get_int(*task_obj, "level")) {
            task.level = *level_val;
        }
        if (const auto* block_val = get(*task_obj, "block"); block_val && block_val->as_array()) {
            for (const auto& coord_item : *block_val->as_array()) {
                const auto* coord_num = coord_item.as_number();
                if (!coord_num || !std::isfinite(*coord_num)) {
                    error = "Task block coordinates must be numeric";
                    return false;
                }
                double integral = 0.0;
                if (std::modf(*coord_num, &integral) != 0.0) {
                    error = "Task block coordinates must be integer";
                    return false;
                }
                task.block.push_back(static_cast<int>(integral));
            }
        }
        task.compute_flops = get_number(*task_obj, "compute_flops").value_or(0.0);
        task.memory_bytes = get_number(*task_obj, "memory_bytes").value_or(0.0);
        if (task.memory_bytes <= 0.0) {
            const auto bytes_read = get_number(*task_obj, "bytes_read").value_or(0.0);
            const auto bytes_written = get_number(*task_obj, "bytes_written").value_or(0.0);
            task.memory_bytes = bytes_read + bytes_written;
        }
        if (!parse_string_set_field(*task_obj, "features", task.features, error)) {
            return false;
        }

        if (const auto* inputs_val = get(*task_obj, "inputs"); inputs_val && inputs_val->as_array()) {
            for (const auto& input_item : *inputs_val->as_array()) {
                const auto* input_obj = input_item.as_object();
                if (!input_obj) {
                    error = "Task input entry must be an object";
                    return false;
                }
                const auto tensor_id = get_string(*input_obj, "tensor");
                if (!tensor_id) {
                    error = "Task input missing 'tensor'";
                    return false;
                }
                TensorUse use;
                use.tensor_id = *tensor_id;
                if (const auto role_val = get_string(*input_obj, "role")) {
                    use.role = *role_val;
                }
                if (const auto access_val = get_string(*input_obj, "access")) {
                    const auto access = parse_access_kind(*access_val);
                    if (!access) {
                        error = "Unsupported access: " + *access_val;
                        return false;
                    }
                    use.access = *access;
                    use.access_explicit = true;
                }
                if (const auto pattern_val = get_string(*input_obj, "access_pattern")) {
                    const auto pattern = parse_access_kind(*pattern_val);
                    if (!pattern) {
                        error = "Unsupported access_pattern: " + *pattern_val;
                        return false;
                    }
                    use.access = *pattern;
                    use.access_explicit = true;
                }
                task.inputs.push_back(std::move(use));
            }
        }

        if (const auto* outputs_val = get(*task_obj, "outputs"); outputs_val && outputs_val->as_array()) {
            for (const auto& output_item : *outputs_val->as_array()) {
                if (const auto* output_str = output_item.as_string()) {
                    task.outputs.push_back(*output_str);
                    continue;
                }
                const auto* output_obj = output_item.as_object();
                if (!output_obj) {
                    error = "Task output entry must be a string or object";
                    return false;
                }
                const auto tensor_id = get_string(*output_obj, "tensor");
                if (!tensor_id) {
                    error = "Task output missing 'tensor'";
                    return false;
                }
                task.outputs.push_back(*tensor_id);
            }
        }

        if (const auto* hint_val = get(*task_obj, "placement_hint"); hint_val && hint_val->as_object()) {
            if (const auto group_val = get_string(*hint_val->as_object(), "group")) {
                task.placement_group = *group_val;
            }
            if (const auto par_val = get_string(*hint_val->as_object(), "parallelism")) {
                task.placement_parallelism = *par_val;
            }
        }

        tasks.push_back(std::move(task));
    }
    return true;
}

bool parse_string_list(const JsonArray& arr, std::vector<std::string>& out, std::string& error) {
    out.clear();
    for (const auto& item : arr) {
        const auto* str = item.as_string();
        if (!str) {
            error = "List entry must be a string";
            return false;
        }
        out.push_back(*str);
    }
    return true;
}

bool parse_raw_string_set_field(const JsonObject& obj,
                                const std::string& key,
                                std::unordered_set<std::string>& out,
                                std::string& error) {
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return true;
    }
    const auto* arr = value->as_array();
    if (arr == nullptr) {
        error = "'" + key + "' must be an array of strings";
        return false;
    }
    for (const auto& item : *arr) {
        const auto* str = item.as_string();
        if (str == nullptr) {
            error = "'" + key + "' must contain only strings";
            return false;
        }
        out.insert(*str);
    }
    return true;
}

bool parse_nonnegative_double_field(const JsonObject& obj,
                                    const std::string& key,
                                    double& out,
                                    std::string& error,
                                    double default_value = 0.0) {
    out = default_value;
    const auto value = get_number(obj, key);
    if (!value.has_value()) {
        return true;
    }
    if (*value < 0.0 || !std::isfinite(*value)) {
        error = "'" + key + "' must be a finite non-negative number";
        return false;
    }
    out = *value;
    return true;
}

bool parse_nonnegative_size_field(const JsonObject& obj,
                                  const std::string& key,
                                  std::size_t& out,
                                  std::string& error,
                                  std::size_t default_value = 0) {
    out = default_value;
    const auto value = get_number(obj, key);
    if (!value.has_value()) {
        return true;
    }
    if (*value < 0.0 || !std::isfinite(*value)) {
        error = "'" + key + "' must be a finite non-negative integer";
        return false;
    }
    double integral = 0.0;
    if (std::modf(*value, &integral) != 0.0) {
        error = "'" + key + "' must be an integer";
        return false;
    }
    if (integral > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        error = "'" + key + "' is too large";
        return false;
    }
    out = static_cast<std::size_t>(integral);
    return true;
}

bool parse_llm_taskgraph(const JsonObject& root_obj,
                         const std::string& name,
                         Workload& out,
                         std::string& error) {
    const auto* tasks_val = get(root_obj, "tasks");
    if (tasks_val == nullptr || tasks_val->as_array() == nullptr) {
        error = "LLM TaskGraph workload missing or invalid 'tasks' array";
        return false;
    }
    const auto* edges_val = get(root_obj, "edges");
    if (edges_val == nullptr || edges_val->as_array() == nullptr) {
        error = "LLM TaskGraph workload missing or invalid 'edges' array";
        return false;
    }

    std::string default_dtype = "fp32";
    if (const auto* model_val = get(root_obj, "model"); model_val != nullptr && model_val->as_object() != nullptr) {
        default_dtype = get_string(*model_val->as_object(), "dtype").value_or(default_dtype);
    }

    mapping::TaskGraph graph;
    std::unordered_set<std::string> seen_tasks;
    for (const auto& item : *tasks_val->as_array()) {
        const auto* task_obj = item.as_object();
        if (task_obj == nullptr) {
            error = "LLM TaskGraph task entry must be an object";
            return false;
        }
        const auto task_name = get_string(*task_obj, "name");
        if (!task_name.has_value() || task_name->empty()) {
            error = "LLM TaskGraph task entry missing 'name'";
            return false;
        }
        if (!seen_tasks.insert(*task_name).second) {
            error = "Duplicate LLM TaskGraph task name: " + *task_name;
            return false;
        }
        const auto raw_subtype = get_string(*task_obj, "subtype").value_or(get_string(*task_obj, "op").value_or(""));
        if (raw_subtype.empty()) {
            error = "LLM TaskGraph task '" + *task_name + "' missing 'subtype'";
            return false;
        }

        mapping::Task task;
        task.name = *task_name;
        task.type = get_string(*task_obj, "type").value_or("compute");
        try {
            task.subtype = mapping::require_operator_profile_name(raw_subtype);
        } catch (const std::exception& ex) {
            error = ex.what();
            return false;
        }
        if (!parse_nonnegative_double_field(*task_obj, "compute_flops", task.compute_flops, error)) {
            return false;
        }
        if (!parse_nonnegative_double_field(*task_obj, "memory_bytes", task.memory_bytes, error)) {
            return false;
        }
        task.access_pattern = get_string(*task_obj, "access_pattern").value_or("dense");
        if (!parse_string_set_field(*task_obj, "features", task.features, error)) {
            return false;
        }
        if (!parse_raw_string_set_field(*task_obj, "tags", task.tags, error)) {
            return false;
        }
        if (task.features.empty()) {
            task.features.insert("dense_linear_algebra");
            task.features.insert("streaming_memory");
            task.features.insert("massive_parallelism");
        }
        graph.add_task(std::move(task));
    }

    for (const auto& item : *edges_val->as_array()) {
        const auto* edge_obj = item.as_object();
        if (edge_obj == nullptr) {
            error = "LLM TaskGraph edge entry must be an object";
            return false;
        }
        const auto src = get_string(*edge_obj, "src");
        const auto dst = get_string(*edge_obj, "dst");
        if (!src.has_value() || src->empty() || !dst.has_value() || dst->empty()) {
            error = "LLM TaskGraph edge entry missing 'src' or 'dst'";
            return false;
        }

        double tensor_bytes = 0.0;
        if (!parse_nonnegative_double_field(*edge_obj, "tensor_bytes", tensor_bytes, error)) {
            return false;
        }
        std::size_t comm_participants = 0;
        if (!parse_nonnegative_size_field(*edge_obj, "comm_participants", comm_participants, error)) {
            return false;
        }

        try {
            graph.add_edge(*src,
                           *dst,
                           tensor_bytes,
                           get_string(*edge_obj, "tensor_id").value_or(""),
                           get_string(*edge_obj, "comm_kind").value_or("p2p"),
                           get_string(*edge_obj, "access_pattern").value_or("dense"),
                           comm_participants,
                           get_string(*edge_obj, "comm_group").value_or(""),
                           get_string(*edge_obj, "dtype").value_or(default_dtype));
        } catch (const std::exception& ex) {
            error = ex.what();
            return false;
        }
    }

    try {
        (void)graph.topological_order();
    } catch (const std::exception& ex) {
        error = ex.what();
        return false;
    }

    out = Workload(name, {}, {}, {}, {}, {}, std::move(graph));
    return true;
}

}  // namespace

bool load_from_json(const std::string& path, Workload& out, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) {
            *error = "Failed to open " + path;
        }
        return false;
    }
    std::ostringstream oss;
    oss << in.rdbuf();
    std::string content = oss.str();

    JsonParser parser(std::move(content));
    JsonValue root;
    std::string parse_error;
    if (!parser.parse(root, parse_error)) {
        if (error) {
            *error = parse_error;
        }
        return false;
    }

    const auto* root_obj = root.as_object();
    if (!root_obj) {
        if (error) {
            *error = "Root must be a JSON object";
        }
        return false;
    }

    const auto name = get_string(*root_obj, "name").value_or("workload");

    const auto root_kind = get_string(*root_obj, "kind").value_or("");
    if (root_kind == "llm_taskgraph_v1" || root_kind == "taskgraph_v1") {
        if (!parse_llm_taskgraph(*root_obj, name, out, parse_error)) {
            if (error) {
                *error = parse_error;
            }
            return false;
        }
        return true;
    }

    std::vector<DeviceGroup> groups;
    if (const auto* groups_val = get(*root_obj, "device_groups"); groups_val && groups_val->as_array()) {
        if (!parse_device_groups(*groups_val->as_array(), groups, parse_error)) {
            if (error) {
                *error = parse_error;
            }
            return false;
        }
    }

    const auto* tensors_val = get(*root_obj, "tensors");
    if (!tensors_val || !tensors_val->as_array()) {
        if (error) {
            *error = "Missing or invalid 'tensors' array";
        }
        return false;
    }
    std::vector<Tensor> tensors;
    if (!parse_tensors(*tensors_val->as_array(), tensors, parse_error)) {
        if (error) {
            *error = parse_error;
        }
        return false;
    }

    const auto* tasks_val = get(*root_obj, "tasks");
    if (!tasks_val || !tasks_val->as_array()) {
        if (error) {
            *error = "Missing or invalid 'tasks' array";
        }
        return false;
    }
    std::vector<Task> tasks;
    if (!parse_tasks(*tasks_val->as_array(), tasks, parse_error)) {
        if (error) {
            *error = parse_error;
        }
        return false;
    }

    std::vector<std::string> iteration_inputs;
    if (const auto* inputs_val = get(*root_obj, "iteration_inputs"); inputs_val && inputs_val->as_array()) {
        if (!parse_string_list(*inputs_val->as_array(), iteration_inputs, parse_error)) {
            if (error) {
                *error = parse_error;
            }
            return false;
        }
    } else if (get(*root_obj, "iteration_inputs") != nullptr) {
        if (error) {
            *error = "Invalid 'iteration_inputs' array";
        }
        return false;
    }

    std::vector<std::string> iteration_outputs;
    if (const auto* outputs_val = get(*root_obj, "iteration_outputs"); outputs_val && outputs_val->as_array()) {
        if (!parse_string_list(*outputs_val->as_array(), iteration_outputs, parse_error)) {
            if (error) {
                *error = parse_error;
            }
            return false;
        }
    } else if (get(*root_obj, "iteration_outputs") != nullptr) {
        if (error) {
            *error = "Invalid 'iteration_outputs' array";
        }
        return false;
    }

    if (get(*root_obj, "edges") != nullptr) {
        if (error) {
            *error = "Edges are not supported; use tensor producers and task inputs instead";
        }
        return false;
    }

    out = Workload(name,
                   std::move(tasks),
                   std::move(tensors),
                   std::move(groups),
                   std::move(iteration_inputs),
                   std::move(iteration_outputs));
    return true;
}

}  // namespace workload
