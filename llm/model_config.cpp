#include "llm/model_config.h"

#include <cctype>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

namespace llm {
namespace {

struct JsonValue;
using JsonObject = std::map<std::string, JsonValue>;
using JsonArray = std::vector<JsonValue>;

struct JsonValue {
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

    const bool* as_bool() const {
        return std::get_if<bool>(&value);
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
    void skip_ws() {
        while (pos_ < input_.size() && std::isspace(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
    }

    bool match_literal(const char* literal) {
        const std::string_view value(literal);
        if (input_.compare(pos_, value.size(), value) != 0) {
            return false;
        }
        pos_ += value.size();
        return true;
    }

    bool parse_value(JsonValue& out, std::string& error) {
        skip_ws();
        if (pos_ >= input_.size()) {
            error = "Unexpected end of JSON";
            return false;
        }
        const char ch = input_[pos_];
        if (ch == '{') {
            return parse_object(out, error);
        }
        if (ch == '[') {
            return parse_array(out, error);
        }
        if (ch == '"') {
            std::string value;
            if (!parse_string(value, error)) {
                return false;
            }
            out.value = std::move(value);
            return true;
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            double number = 0.0;
            if (!parse_number(number, error)) {
                return false;
            }
            out.value = number;
            return true;
        }
        if (match_literal("true")) {
            out.value = true;
            return true;
        }
        if (match_literal("false")) {
            out.value = false;
            return true;
        }
        if (match_literal("null")) {
            out.value = nullptr;
            return true;
        }
        error = "Invalid JSON value";
        return false;
    }

    bool parse_object(JsonValue& out, std::string& error) {
        ++pos_;
        skip_ws();
        JsonObject object;
        if (pos_ < input_.size() && input_[pos_] == '}') {
            ++pos_;
            out.value = std::move(object);
            return true;
        }
        while (pos_ < input_.size()) {
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
            object.emplace(std::move(key), std::move(value));
            skip_ws();
            if (pos_ >= input_.size()) {
                error = "Unexpected end of object";
                return false;
            }
            if (input_[pos_] == '}') {
                ++pos_;
                out.value = std::move(object);
                return true;
            }
            if (input_[pos_] != ',') {
                error = "Expected ',' between object items";
                return false;
            }
            ++pos_;
            skip_ws();
        }
        error = "Unexpected end of object";
        return false;
    }

    bool parse_array(JsonValue& out, std::string& error) {
        ++pos_;
        skip_ws();
        JsonArray array;
        if (pos_ < input_.size() && input_[pos_] == ']') {
            ++pos_;
            out.value = std::move(array);
            return true;
        }
        while (pos_ < input_.size()) {
            JsonValue value;
            if (!parse_value(value, error)) {
                return false;
            }
            array.push_back(std::move(value));
            skip_ws();
            if (pos_ >= input_.size()) {
                error = "Unexpected end of array";
                return false;
            }
            if (input_[pos_] == ']') {
                ++pos_;
                out.value = std::move(array);
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
        if (pos_ >= input_.size() || input_[pos_] != '"') {
            error = "Expected string";
            return false;
        }
        ++pos_;
        std::string result;
        while (pos_ < input_.size()) {
            const char ch = input_[pos_++];
            if (ch == '"') {
                out = std::move(result);
                return true;
            }
            if (ch != '\\') {
                result.push_back(ch);
                continue;
            }
            if (pos_ >= input_.size()) {
                error = "Invalid escape sequence";
                return false;
            }
            const char esc = input_[pos_++];
            switch (esc) {
                case '"':
                case '\\':
                case '/':
                    result.push_back(esc);
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
        }
        error = "Unterminated string";
        return false;
    }

    bool parse_number(double& out, std::string& error) {
        const auto start = pos_;
        if (input_[pos_] == '-') {
            ++pos_;
        }
        while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
            ++pos_;
        }
        if (pos_ < input_.size() && input_[pos_] == '.') {
            ++pos_;
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        }
        if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) {
                ++pos_;
            }
            while (pos_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[pos_]))) {
                ++pos_;
            }
        }
        try {
            out = std::stod(input_.substr(start, pos_ - start));
        } catch (const std::exception&) {
            error = "Invalid number";
            return false;
        }
        return true;
    }

    std::string input_;
    std::size_t pos_{0};
};

std::string canonical(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

const JsonValue* get(const JsonObject& obj, const std::string& key) {
    const auto it = obj.find(key);
    return it == obj.end() ? nullptr : &it->second;
}

std::optional<std::string> get_string(const JsonObject& obj, const std::string& key) {
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (const auto* str = value->as_string()) {
        return *str;
    }
    return std::nullopt;
}

std::optional<int> get_int(const JsonObject& obj, const std::string& key) {
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return std::nullopt;
    }
    const auto* number = value->as_number();
    if (number == nullptr || *number < static_cast<double>(std::numeric_limits<int>::min()) ||
        *number > static_cast<double>(std::numeric_limits<int>::max())) {
        return std::nullopt;
    }
    return static_cast<int>(*number);
}

std::optional<bool> get_bool(const JsonObject& obj, const std::string& key) {
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return std::nullopt;
    }
    if (const auto* boolean = value->as_bool()) {
        return *boolean;
    }
    return std::nullopt;
}

std::vector<std::string> get_string_array(const JsonObject& obj, const std::string& key) {
    std::vector<std::string> out;
    const auto* value = get(obj, key);
    if (value == nullptr) {
        return out;
    }
    const auto* array = value->as_array();
    if (array == nullptr) {
        return out;
    }
    out.reserve(array->size());
    for (const auto& item : *array) {
        if (const auto* str = item.as_string()) {
            out.push_back(canonical(*str));
        }
    }
    return out;
}

int first_int(const JsonObject& obj, std::initializer_list<const char*> keys, int default_value = 0) {
    for (const char* key : keys) {
        if (auto value = get_int(obj, key)) {
            return *value;
        }
    }
    return default_value;
}

DataType parse_dtype_string(std::string value) {
    value = canonical(std::move(value));
    if (value == "float32" || value == "fp32" || value == "f32") {
        return DataType::FP32;
    }
    if (value == "bfloat16" || value == "bf16") {
        return DataType::BF16;
    }
    if (value == "float16" || value == "fp16" || value == "half") {
        return DataType::FP16;
    }
    if (value == "float8" || value == "fp8") {
        return DataType::FP8;
    }
    if (value == "int8") {
        return DataType::INT8;
    }
    if (value == "int4" || value == "uint4") {
        return DataType::INT4;
    }
    return DataType::FP16;
}

bool is_supported_dense_family(const std::string& model_type) {
    const auto type = canonical(model_type);
    return type.find("qwen") != std::string::npos || type.find("gemma") != std::string::npos ||
           type.find("llama") != std::string::npos || type.find("mixtral") != std::string::npos ||
           type.find("gpt3") != std::string::npos || type.find("gpt_3") != std::string::npos;
}

const char* supported_dense_family_hint() {
    return "Gemma, Qwen, Mixtral, Llama, or GPT-3";
}

bool has_moe_fields(const JsonObject& obj) {
    const auto type = canonical(get_string(obj, "model_type").value_or(""));
    if (type.find("moe") != std::string::npos || type.find("mixtral") != std::string::npos) {
        return true;
    }
    return first_int(obj,
                     {"num_experts",
                      "num_local_experts",
                      "num_routed_experts",
                      "n_routed_experts",
                      "num_experts_per_tok",
                      "moe_intermediate_size"},
                     0) > 0;
}

const JsonObject& model_object_for_config(const JsonObject& root) {
    if (const auto* text_config = get(root, "text_config")) {
        if (const auto* text_obj = text_config->as_object()) {
            if (get(*text_obj, "num_hidden_layers") != nullptr || get(*text_obj, "hidden_size") != nullptr) {
                return *text_obj;
            }
        }
    }
    return root;
}

std::string join_object_keys(const JsonObject& obj) {
    std::ostringstream out;
    bool first = true;
    for (const auto& entry : obj) {
        if (!first) {
            out << ", ";
        }
        first = false;
        out << entry.first;
    }
    return out.str();
}

int first_int_from(const JsonObject* primary,
                   const JsonObject* secondary,
                   const JsonObject* tertiary,
                   std::initializer_list<const char*> keys,
                   int default_value = 0) {
    if (primary != nullptr) {
        for (const char* key : keys) {
            if (auto value = get_int(*primary, key)) {
                return *value;
            }
        }
    }
    if (secondary != nullptr) {
        for (const char* key : keys) {
            if (auto value = get_int(*secondary, key)) {
                return *value;
            }
        }
    }
    if (tertiary != nullptr) {
        for (const char* key : keys) {
            if (auto value = get_int(*tertiary, key)) {
                return *value;
            }
        }
    }
    return default_value;
}

std::optional<bool> first_bool_from(const JsonObject* primary,
                                    const JsonObject* secondary,
                                    std::initializer_list<const char*> keys) {
    if (primary != nullptr) {
        for (const char* key : keys) {
            if (auto value = get_bool(*primary, key)) {
                return *value;
            }
        }
    }
    if (secondary != nullptr) {
        for (const char* key : keys) {
            if (auto value = get_bool(*secondary, key)) {
                return *value;
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> first_string_from(const JsonObject* primary,
                                             const JsonObject* secondary,
                                             std::initializer_list<const char*> keys) {
    if (primary != nullptr) {
        for (const char* key : keys) {
            if (auto value = get_string(*primary, key)) {
                return *value;
            }
        }
    }
    if (secondary != nullptr) {
        for (const char* key : keys) {
            if (auto value = get_string(*secondary, key)) {
                return *value;
            }
        }
    }
    return std::nullopt;
}

bool load_family_size_config(const JsonObject& root,
                             const std::string& requested_size,
                             LlmModelConfig& config,
                             std::string* error) {
    const auto family = get_string(root, "family");
    if (!family) {
        return false;
    }
    const std::string family_name = canonical(*family);
    const JsonObject* sizes = nullptr;
    if (family_name == "gpt3") {
        if (const auto* dense_sizes_value = get(root, "dense_approx_sizes")) {
            sizes = dense_sizes_value->as_object();
        }
    }
    if (sizes == nullptr) {
        if (const auto* sizes_value = get(root, "sizes")) {
            sizes = sizes_value->as_object();
        }
    }
    if (sizes == nullptr || sizes->empty()) {
        if (error != nullptr) {
            *error = "LLM family config is missing sizes";
        }
        return true;
    }

    std::string selected_size = requested_size;
    if (selected_size.empty()) {
        selected_size = get_string(root, "default_size").value_or("");
    }
    if (selected_size.empty()) {
        if (family_name == "gpt3" && sizes->count("125M") > 0) {
            selected_size = "125M";
        } else {
            selected_size = sizes->begin()->first;
        }
    }

    const auto variant_it = sizes->find(selected_size);
    if (variant_it == sizes->end()) {
        if (error != nullptr) {
            *error = "Unknown LLM size '" + selected_size + "' for family " + family_name +
                     "; available sizes: " + join_object_keys(*sizes);
        }
        return true;
    }
    const auto* variant = variant_it->second.as_object();
    if (variant == nullptr) {
        if (error != nullptr) {
            *error = "LLM family size entry must be an object";
        }
        return true;
    }

    const JsonObject* transformer = nullptr;
    if (const auto* transformer_value = get(*variant, "transformer")) {
        transformer = transformer_value->as_object();
    }
    const JsonObject* architecture = nullptr;
    if (const auto* architecture_value = get(*variant, "architecture")) {
        architecture = architecture_value->as_object();
    }
    if (transformer == nullptr || architecture == nullptr) {
        if (error != nullptr) {
            *error = "LLM family size entry must include transformer and architecture objects";
        }
        return true;
    }

    config = {};
    config.model_type = get_string(*variant, "model_family").value_or(family_name);
    config.model_name = get_string(*variant, "model_name").value_or(config.model_type + "-" + selected_size);
    config.model_size = selected_size;
    config.num_layers = first_int_from(transformer, architecture, &root, {"num_hidden_layers", "n_layer", "num_layers"});
    config.hidden_size = first_int_from(transformer, architecture, &root, {"hidden_size", "n_embd", "model_dim"});
    config.intermediate_size =
        first_int_from(transformer, architecture, &root, {"intermediate_size", "ffn_hidden_size", "mlp_hidden_size"});
    config.num_attention_heads = first_int_from(transformer, architecture, &root, {"num_attention_heads", "n_head"});
    config.num_kv_heads = first_int_from(transformer,
                                         architecture,
                                         &root,
                                         {"num_key_value_heads", "num_kv_heads", "multi_query_group_num"},
                                         config.num_attention_heads);
    config.head_dim = first_int_from(transformer, architecture, &root, {"head_dim", "attention_head_dim", "kv_head_dim"});
    if (config.head_dim <= 0 && config.num_attention_heads > 0) {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }
    config.vocab_size = first_int_from(architecture, transformer, &root, {"vocab_size", "default_vocab_size"});
    config.max_position_embeddings =
        first_int_from(architecture, transformer, &root, {"max_position_embeddings", "seq_length", "n_positions"});
    config.use_bias = first_bool_from(architecture, transformer, {"attention_bias", "use_bias"}).value_or(false);
    config.tie_word_embeddings =
        first_bool_from(architecture, transformer, {"tie_word_embeddings"}).value_or(false);
    config.use_gated_mlp = false;
    if (const auto mlp_type = first_string_from(transformer, architecture, {"mlp_type", "activation_function"})) {
        const auto mlp = canonical(*mlp_type);
        config.use_gated_mlp = mlp.find("swiglu") != std::string::npos || mlp.find("gated") != std::string::npos;
    }
    if (const auto dtype = first_string_from(variant, &root, {"torch_dtype", "dtype"})) {
        config.param_dtype = parse_dtype_string(*dtype);
    }

    if (config.num_layers <= 0 || config.hidden_size <= 0 || config.intermediate_size <= 0 ||
        config.num_attention_heads <= 0 || config.num_kv_heads <= 0 || config.head_dim <= 0 ||
        config.vocab_size <= 0) {
        if (error != nullptr) {
            *error = "LLM family size config is missing required dense decoder fields";
        }
        return true;
    }
    return true;
}

}  // namespace

std::uint64_t dtype_size_bytes(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return 4;
        case DataType::FP16:
        case DataType::BF16:
            return 2;
        case DataType::FP8:
        case DataType::INT8:
            return 1;
        case DataType::INT4:
            return 1;
    }
    return 2;
}

std::string dtype_name(DataType dtype) {
    switch (dtype) {
        case DataType::FP32:
            return "fp32";
        case DataType::FP16:
            return "fp16";
        case DataType::BF16:
            return "bf16";
        case DataType::FP8:
            return "fp8";
        case DataType::INT8:
            return "int8";
        case DataType::INT4:
            return "int4";
    }
    return "fp16";
}

bool load_model_config_from_json(const std::string& path,
                                 LlmModelConfig& config,
                                 std::string* error,
                                 const std::string& requested_size) {
    namespace fs = std::filesystem;
    fs::path config_path(path);
    if (fs::is_directory(config_path)) {
        config_path /= "config.json";
    }
    std::ifstream in(config_path);
    if (!in) {
        if (error != nullptr) {
            *error = "Failed to open LLM config: " + config_path.string();
        }
        return false;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();

    JsonValue root;
    std::string parse_error;
    JsonParser parser(buffer.str());
    if (!parser.parse(root, parse_error)) {
        if (error != nullptr) {
            *error = parse_error;
        }
        return false;
    }

    const auto* root_obj = root.as_object();
    if (root_obj == nullptr) {
        if (error != nullptr) {
            *error = "LLM config root must be an object";
        }
        return false;
    }

    if (get(*root_obj, "sizes") != nullptr && get(*root_obj, "family") != nullptr) {
        std::string family_error;
        const bool handled = load_family_size_config(*root_obj, requested_size, config, &family_error);
        if (handled) {
            if (!family_error.empty()) {
                if (error != nullptr) {
                    *error = family_error;
                }
                return false;
            }
            if (!is_supported_dense_family(config.model_type)) {
                if (error != nullptr) {
                    *error = "Unsupported LLM model family for first version: " + config.model_type +
                             " (expected " + supported_dense_family_hint() + " family config)";
                }
                return false;
            }
            return true;
        }
    }

    const auto& obj = model_object_for_config(*root_obj);
    const bool has_moe_config = has_moe_fields(obj) || has_moe_fields(*root_obj);

    config = {};
    config.model_type = get_string(obj, "model_type").value_or(get_string(*root_obj, "model_type").value_or(""));
    if (config.model_type.empty()) {
        config.model_type = "unknown";
    }
    if (!is_supported_dense_family(config.model_type)) {
        if (error != nullptr) {
            *error = "Unsupported LLM model_type for first version: " + config.model_type +
                     " (expected " + supported_dense_family_hint() + " decoder config)";
        }
        return false;
    }

    config.is_moe = has_moe_config;
    config.num_layers = first_int(obj, {"num_hidden_layers", "n_layer", "num_layers"});
    config.hidden_size = first_int(obj, {"hidden_size", "n_embd", "model_dim"});
    config.intermediate_size = first_int(obj, {"intermediate_size", "ffn_hidden_size", "mlp_hidden_size"});
    config.moe_intermediate_size = first_int(obj, {"moe_intermediate_size", "expert_intermediate_size"}, 0);
    config.num_attention_heads = first_int(obj, {"num_attention_heads", "n_head"});
    config.num_kv_heads =
        first_int(obj, {"num_key_value_heads", "num_kv_heads", "multi_query_group_num"}, config.num_attention_heads);
    config.head_dim = first_int(obj, {"head_dim", "attention_head_dim"});
    if (config.head_dim <= 0 && config.num_attention_heads > 0) {
        config.head_dim = config.hidden_size / config.num_attention_heads;
    }
    config.vocab_size = first_int(obj, {"vocab_size"});
    config.max_position_embeddings = first_int(obj, {"max_position_embeddings", "seq_length", "n_positions"});
    config.sliding_window = first_int(obj, {"sliding_window"}, 0);
    config.layer_types = get_string_array(obj, "layer_types");
    config.linear_num_key_heads = first_int(obj, {"linear_num_key_heads"}, 0);
    config.linear_num_value_heads = first_int(obj, {"linear_num_value_heads"}, 0);
    config.linear_key_head_dim = first_int(obj, {"linear_key_head_dim"}, 0);
    config.linear_value_head_dim = first_int(obj, {"linear_value_head_dim"}, 0);
    config.linear_conv_kernel_dim = first_int(obj, {"linear_conv_kernel_dim"}, 0);
    config.num_experts = first_int(obj,
                                   {"num_local_experts", "num_experts", "num_routed_experts", "n_routed_experts"},
                                   0);
    config.experts_per_token = first_int(obj, {"num_experts_per_tok", "top_k_experts", "moe_top_k"}, 0);
    if (config.is_moe && config.moe_intermediate_size <= 0) {
        config.moe_intermediate_size = config.intermediate_size;
    }
    config.attn_output_gate = get_bool(obj, "attn_output_gate").value_or(false);
    config.tie_word_embeddings = get_bool(obj, "tie_word_embeddings").value_or(false);
    config.use_bias = get_bool(obj, "attention_bias").value_or(get_bool(obj, "use_bias").value_or(false));
    config.use_gated_mlp = true;
    config.use_rms_norm = true;

    if (const auto dtype = get_string(obj, "torch_dtype")) {
        config.param_dtype = parse_dtype_string(*dtype);
    } else if (const auto dtype = get_string(obj, "dtype")) {
        config.param_dtype = parse_dtype_string(*dtype);
    } else if (const auto dtype = get_string(*root_obj, "torch_dtype")) {
        config.param_dtype = parse_dtype_string(*dtype);
    }

    const JsonValue* quant_value = get(obj, "quantization_config");
    if (quant_value == nullptr) {
        quant_value = get(*root_obj, "quantization_config");
    }
    if (quant_value != nullptr) {
        if (const auto* quant = quant_value->as_object()) {
            config.quant.type = get_string(*quant, "quant_method").value_or(get_string(*quant, "type").value_or(""));
            config.quant.bits = first_int(*quant, {"bits", "weight_bits", "w_bit"}, 0);
            config.quant.group_size = first_int(*quant, {"group_size", "q_group_size"}, 0);
            if (config.quant.bits == 8) {
                config.param_dtype = DataType::INT8;
            } else if (config.quant.bits == 4) {
                config.param_dtype = DataType::INT4;
            }
        }
    }

    if (config.num_layers <= 0 || config.hidden_size <= 0 || config.intermediate_size <= 0 ||
        config.num_attention_heads <= 0 || config.num_kv_heads <= 0 || config.head_dim <= 0 ||
        config.vocab_size <= 0) {
        if (error != nullptr) {
            *error = "LLM config is missing required dense decoder fields";
        }
        return false;
    }
    if (config.is_moe) {
        if (config.num_experts <= 0 || config.experts_per_token <= 0 ||
            config.experts_per_token > config.num_experts || config.moe_intermediate_size <= 0) {
            if (error != nullptr) {
                *error = "MoE LLM config is missing valid expert count, top-k, or expert intermediate size";
            }
            return false;
        }
    }
    return true;
}

}  // namespace llm
