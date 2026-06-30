#pragma once

#include <string>
#include <vector>

#include "hardware_topology/topology.h"
#include "llm/model_config.h"
#include "mapping/graph.h"

namespace llm {

struct LlmRequestConfig {
    int prefill_batch_size{1};
    int prompt_len{2048};
    int decode_batch_size{1};
    int decode_steps{0};
    int avg_context_len{2048};
};

struct LlmParallelConfig {
    int tp{1};
    int pp{1};
    int cp{1};
    int dp{1};
    bool pin_shards_to_devices{true};
    std::vector<int> stage_for_layer;
};

struct LlmTaskGraphBuildResult {
    mapping::TaskGraph graph;
    LlmModelConfig model;
    LlmRequestConfig request;
    LlmParallelConfig parallel;
    std::vector<std::string> diagnostics;
};

LlmTaskGraphBuildResult build_task_graph(const LlmModelConfig& model,
                                         const LlmRequestConfig& request,
                                         const LlmParallelConfig& parallel,
                                         const hardware_topology::HardwareTopology& topology);

LlmTaskGraphBuildResult build_task_graph_from_config(const std::string& config_path,
                                                     const LlmRequestConfig& request,
                                                     const LlmParallelConfig& parallel,
                                                     const hardware_topology::HardwareTopology& topology,
                                                     const std::string& model_size = {});

void write_task_graph_json(const std::string& path,
                           const mapping::TaskGraph& graph,
                           const LlmModelConfig& model,
                           const LlmRequestConfig& request,
                           const LlmParallelConfig& parallel,
                           bool omit_device_tags = false);

}  // namespace llm
