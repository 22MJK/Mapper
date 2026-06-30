#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

#include "hardware_topology/topology.h"
#include "workload/workload.h"

namespace mapper {

struct NamedCount {
    std::string name;
    std::size_t count{0};
};

struct NamedBytes {
    std::string name;
    std::uint64_t bytes{0};
};

struct Options {
    int parts{0};
    std::string time_unit{"s"};
    std::string mapper{"aeft"};
    std::string parallel{"none"};
    bool force_exhaustive{false};
    std::string llm_config_path;
    std::string llm_size;
    int llm_prefill_batch_size{1};
    int llm_prompt_len{2048};
    int llm_decode_batch_size{1};
    int llm_decode_steps{0};
    int llm_avg_context_len{2048};
    int llm_tp{1};
    int llm_pp{1};
    int llm_cp{1};
    int llm_dp{1};
    bool llm_auto_parallel{false};
    bool llm_rank_parallel{false};
    std::string llm_dump_taskgraph_path;
    bool workload_rank_parallel{false};
    std::string workload_dump_taskgraph_path;
    std::string output_format{"json"};
    std::string et_prefix;
};

struct RunResult {
    double estimated_makespan_s{0.0};
    std::string selected_parallel{"none"};
    std::size_t task_count{0};
    std::size_t edge_count{0};
    std::size_t source_count{0};
    std::size_t sink_count{0};
    std::size_t dag_depth{0};
    std::uint64_t total_edge_bytes{0};
    std::size_t cross_device_edge_count{0};
    std::uint64_t cross_device_edge_bytes{0};
    std::vector<NamedCount> task_subtype_counts;
    std::vector<NamedCount> device_task_counts;
    std::vector<NamedBytes> comm_kind_bytes;
};

// Mapper entrypoint: binds tasks to devices and emits taskflow JSON for simulators.
RunResult write_taskflow(const hardware_topology::HardwareTopology& topology,
                         const workload::Workload& workload,
                         const std::string& taskflow_path,
                         const Options& options = {});

}  // namespace mapper
