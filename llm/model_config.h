#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace llm {

enum class DataType {
    FP32,
    FP16,
    BF16,
    FP8,
    INT8,
    INT4,
};

struct QuantConfig {
    std::string type;
    int bits{0};
    int group_size{0};
};

struct LlmModelConfig {
    std::string model_type;
    std::string model_name;
    std::string model_size;
    int num_layers{0};
    int hidden_size{0};
    int intermediate_size{0};
    int num_attention_heads{0};
    int num_kv_heads{0};
    int head_dim{0};
    int vocab_size{0};
    int max_position_embeddings{0};
    int sliding_window{0};
    std::vector<std::string> layer_types;
    int linear_num_key_heads{0};
    int linear_num_value_heads{0};
    int linear_key_head_dim{0};
    int linear_value_head_dim{0};
    int linear_conv_kernel_dim{0};
    bool is_moe{false};
    int num_experts{0};
    int experts_per_token{0};
    int moe_intermediate_size{0};
    bool attn_output_gate{false};
    bool tie_word_embeddings{false};
    bool use_bias{false};
    bool use_gated_mlp{true};
    bool use_rms_norm{true};
    DataType param_dtype{DataType::FP16};
    QuantConfig quant;
};

std::uint64_t dtype_size_bytes(DataType dtype);
std::string dtype_name(DataType dtype);

bool load_model_config_from_json(const std::string& path,
                                 LlmModelConfig& config,
                                 std::string* error = nullptr,
                                 const std::string& requested_size = {});

}  // namespace llm
