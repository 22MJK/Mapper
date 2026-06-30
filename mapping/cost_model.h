#pragma once

#include "hardware_topology/topology.h"
#include "mapping/graph.h"

namespace mapping {

double estimate_task_time_seconds(const Task& task, const hardware_topology::Device* device);
double estimate_cpu_task_time_seconds(const Task& task, const hardware_topology::Device* device);
double estimate_gpu_task_time_seconds(const Task& task, const hardware_topology::Device* device);

double estimate_cpu_vector_efficiency(const Task& task);
double estimate_cpu_thread_efficiency(const Task& task, const hardware_topology::Device* device);
double estimate_cpu_cache_efficiency(const Task& task);
double estimate_gpu_occupancy(const Task& task);
double estimate_gpu_simt_efficiency(const Task& task);
double estimate_gpu_coalescing_efficiency(const Task& task);
double estimate_cpu_mismatch_penalty_seconds(const Task& task);
double estimate_gpu_mismatch_penalty_seconds(const Task& task);

}  // namespace mapping

