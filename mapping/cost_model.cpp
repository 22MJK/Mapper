#include "mapping/cost_model.h"

#include "mapping/operator_catalog.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_set>

namespace mapping {
namespace {

constexpr double kEpsilon = 1e-12;

std::string canonical(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

bool is_cpu(const hardware_topology::Device* device) {
    return device != nullptr && canonical(device->type) == "cpu";
}

bool is_gpu(const hardware_topology::Device* device) {
    return device != nullptr && canonical(device->type) == "gpu";
}

bool contains_any(const std::string& value, std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (value.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string op_category(const Task& task) {
    const std::string subtype = canonical_operator_subtype(task.subtype.empty() ? task.type : task.subtype);
    if (contains_any(subtype, {"gemm", "spgemm", "trsm", "trsv", "sptrsv", "gessm", "ssssm", "tstrf", "getrf", "geqrf", "potrf",
                               "spmv", "mv"})) {
        return "matrix";
    }
    if (contains_any(subtype, {"dot", "nrm2", "allreduce"})) {
        return "reduction";
    }
    if (contains_any(subtype, {"transpose"})) {
        return "matrix_transform";
    }
    if (contains_any(subtype, {"axpy", "copy", "scal", "assemble"})) {
        return "vector";
    }
    if (contains_any(subtype, {"symbolic", "order", "etree", "colcount", "postorder", "supernode_partition"})) {
        return "symbolic";
    }
    return "unknown";
}

std::string access_pattern(const Task& task) {
    const std::string subtype = canonical_operator_subtype(task.subtype.empty() ? task.type : task.subtype);
    const std::string access = canonical(task.access_pattern);
    if (subtype == "transpose" && (access.empty() || access == "dense" || access == "contiguous")) {
        return "strided";
    }
    if (access.find("sparse") != std::string::npos || access.find("csr") != std::string::npos) {
        return "sparse";
    }
    if (access.find("random") != std::string::npos || access.find("indirect") != std::string::npos) {
        return "random";
    }
    if (access.find("row") != std::string::npos || access.find("col") != std::string::npos ||
        access.find("stride") != std::string::npos) {
        return "strided";
    }
    if (access.find("dense") != std::string::npos || access.find("contiguous") != std::string::npos ||
        access.empty()) {
        return "contiguous";
    }
    return "unknown";
}

std::unordered_set<std::string> operator_features(const Task& task) {
    std::unordered_set<std::string> features = task.features;
    const std::string subtype = canonical_operator_subtype(task.subtype.empty() ? task.type : task.subtype);
    const std::string category = op_category(task);
    const std::string access = access_pattern(task);

    if (category == "matrix") {
        if (access == "sparse" || access == "random" || subtype == "ssssm" || subtype == "spmv" ||
            subtype == "sptrsv" || subtype == "spgemm") {
            features.insert("sparse_linear_algebra");
            features.insert("irregular_access");
        } else {
            features.insert("dense_linear_algebra");
        }
        if (subtype == "trsv" || subtype == "sptrsv" || subtype == "trsm" || subtype == "ssssm") {
            features.insert("triangular_dependency");
            features.insert("data_dependency");
        }
        if (subtype == "potrf" || subtype == "getrf" || subtype == "geqrf" || subtype == "tstrf" ||
            subtype == "gessm") {
            features.insert("factorization");
        }
    } else if (category == "reduction") {
        features.insert("reduction");
    } else if (category == "matrix_transform") {
        features.insert("matrix_transform");
    } else if (category == "vector") {
        features.insert("elementwise");
    } else if (category == "symbolic") {
        features.insert("symbolic");
        features.insert("control_flow");
        features.insert("data_dependency");
        features.insert("latency_sensitive");
        features.insert("gpu_unsupported");
    }

    if (access == "contiguous") {
        features.insert("streaming_memory");
        features.insert("coalesced_access");
    } else if (access == "strided") {
        features.insert("strided_access");
    } else if (access == "random" || access == "sparse") {
        features.insert("irregular_access");
    }

    if (task.compute_flops >= 1e9) {
        features.insert("massive_parallelism");
        features.insert("compute_intensive");
    } else if (task.compute_flops >= 1e6 || task.memory_bytes >= 1024.0 * 1024.0) {
        features.insert("high_parallelism");
    } else {
        features.insert("small_working_set");
    }

    if (task.memory_bytes > 0.0 && task.compute_flops / task.memory_bytes < 1.0) {
        features.insert("memory_bound");
    }
    return features;
}

std::unordered_set<std::string> device_features(const hardware_topology::Device* device) {
    std::unordered_set<std::string> features;
    if (device == nullptr) {
        return features;
    }
    features = device->features;
    if (is_cpu(device)) {
        features.insert("scalar_cores");
        features.insert("simd");
        features.insert("cache_hierarchy");
        features.insert("low_latency_launch");
        features.insert("branching");
        features.insert("irregular_memory");
    } else if (is_gpu(device)) {
        features.insert("simt");
        features.insert("warp_execution");
        features.insert("high_bandwidth_memory");
        features.insert("coalesced_memory");
        features.insert("massive_parallelism");
        features.insert("device_memory");
    }
    return features;
}

bool has_feature(const std::unordered_set<std::string>& features, const std::string& feature) {
    return features.find(feature) != features.end();
}

bool task_has_tag_or_feature(const Task& task, const std::string& value) {
    return task.tags.find(value) != task.tags.end() || task.features.find(value) != task.features.end();
}

double feature_affinity_multiplier(const Task& task, const hardware_topology::Device* device) {
    const auto op_features = operator_features(task);
    const auto dev_features = device_features(device);
    double score = 0.0;

    if (is_cpu(device)) {
        if (has_feature(op_features, "control_flow") || has_feature(op_features, "data_dependency")) {
            score += has_feature(dev_features, "branching") ? 2.0 : -2.0;
        }
        if (has_feature(op_features, "irregular_access")) {
            score += has_feature(dev_features, "irregular_memory") ? 1.5 : -1.5;
        }
        if (has_feature(op_features, "small_working_set") || has_feature(op_features, "latency_sensitive")) {
            score += has_feature(dev_features, "low_latency_launch") ? 1.5 : -1.0;
        }
        if (has_feature(op_features, "massive_parallelism") || has_feature(op_features, "dense_linear_algebra")) {
            score -= 1.5;
        }
    } else if (is_gpu(device)) {
        if (has_feature(op_features, "massive_parallelism") || has_feature(op_features, "high_parallelism")) {
            score += has_feature(dev_features, "massive_parallelism") ? 2.0 : -2.0;
        }
        if (has_feature(op_features, "dense_linear_algebra") || has_feature(op_features, "compute_intensive")) {
            score += has_feature(dev_features, "simt") ? 1.5 : -1.5;
        }
        if (has_feature(op_features, "coalesced_access") || has_feature(op_features, "streaming_memory")) {
            score += has_feature(dev_features, "coalesced_memory") ? 1.0 : -1.0;
        }
        if (has_feature(op_features, "control_flow") || has_feature(op_features, "data_dependency")) {
            score -= 2.0;
        }
        if (has_feature(op_features, "irregular_access")) {
            score -= 1.5;
        }
        if (has_feature(op_features, "small_working_set") || has_feature(op_features, "latency_sensitive")) {
            score -= 1.0;
        }
    }

    return std::clamp(1.0 - 0.04 * score, 0.65, 1.45);
}

bool has_control_flow(const Task& task) {
    const auto features = operator_features(task);
    return op_category(task) == "symbolic" || has_feature(features, "control_flow");
}

bool has_data_dependency(const Task& task) {
    const auto features = operator_features(task);
    return has_control_flow(task) || has_feature(features, "data_dependency");
}

bool has_dynamic_shape(const Task& task) {
    const auto features = operator_features(task);
    if (has_feature(features, "dynamic_shape")) {
        return true;
    }
    const std::string subtype = canonical(task.subtype);
    return contains_any(subtype, {"dynamic", "shape"});
}

double work_items(const Task& task) {
    if (task.memory_bytes > 0.0) {
        return std::max(1.0, task.memory_bytes / 4.0);
    }
    if (task.compute_flops > 0.0) {
        return std::max(1.0, std::sqrt(task.compute_flops));
    }
    return 1.0;
}

double peak_flops_per_s(const hardware_topology::Device* device) {
    if (device == nullptr || device->peak_gflops <= 0.0) {
        return 0.0;
    }
    return device->peak_gflops * 1e9;
}

double bandwidth_bytes_per_s(const hardware_topology::Device* device) {
    if (device == nullptr || device->mem_bw_gbps <= 0.0) {
        return 0.0;
    }
    return device->mem_bw_gbps * 1e9;
}

hardware_topology::OperatorCostScale device_operator_scale(const Task& task,
                                                           const hardware_topology::Device* device) {
    if (device == nullptr) {
        return {};
    }
    const std::string subtype = canonical_operator_subtype(task.subtype.empty() ? task.type : task.subtype);
    const auto it = device->operator_cost_scales.find(subtype);
    if (it != device->operator_cost_scales.end()) {
        const auto& scale = it->second;
        for (const auto& segment : scale.segments) {
            if (!segment.max_num_ops.has_value() ||
                task.compute_flops <= *segment.max_num_ops) {
                hardware_topology::OperatorCostScale selected;
                selected.bandwidth_scale = segment.bandwidth_scale;
                selected.flops_scale = segment.flops_scale;
                selected.launch_overhead_us = segment.launch_overhead_us;
                return selected;
            }
        }
        return scale;
    }
    return {};
}

double launch_overhead_seconds(const hardware_topology::OperatorCostScale& scale) {
    return scale.launch_overhead_us / 1e6;
}

}  // namespace

double estimate_task_time_seconds(const Task& task, const hardware_topology::Device* device) {
    if (device == nullptr || !device->compute_capable) {
        return std::numeric_limits<double>::infinity();
    }
    if (is_cpu(device) && task_has_tag_or_feature(task, "cpu_unsupported")) {
        return std::numeric_limits<double>::infinity();
    }
    if (is_gpu(device) && task_has_tag_or_feature(task, "gpu_unsupported")) {
        return std::numeric_limits<double>::infinity();
    }
    if (is_cpu(device)) {
        return estimate_cpu_task_time_seconds(task, device);
    }
    if (is_gpu(device)) {
        return estimate_gpu_task_time_seconds(task, device);
    }
    const double compute_s = (task.compute_flops > 0.0 && peak_flops_per_s(device) <= 0.0)
                                 ? std::numeric_limits<double>::infinity()
                                 : task.compute_flops / std::max(peak_flops_per_s(device), kEpsilon);
    const double memory_s = (task.memory_bytes > 0.0 && bandwidth_bytes_per_s(device) <= 0.0)
                                ? std::numeric_limits<double>::infinity()
                                : task.memory_bytes / std::max(bandwidth_bytes_per_s(device), kEpsilon);
    return std::max(compute_s, memory_s);
}

double estimate_cpu_task_time_seconds(const Task& task, const hardware_topology::Device* device) {
    if (task_has_tag_or_feature(task, "cpu_unsupported")) {
        return std::numeric_limits<double>::infinity();
    }
    if (device == nullptr || peak_flops_per_s(device) <= 0.0 || bandwidth_bytes_per_s(device) <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double vector_eff = estimate_cpu_vector_efficiency(task);
    const double thread_eff = estimate_cpu_thread_efficiency(task, device);
    const double cache_eff = estimate_cpu_cache_efficiency(task);
    const auto scale = device_operator_scale(task, device);
    const double compute_s =
        task.compute_flops /
        std::max(peak_flops_per_s(device) * vector_eff * thread_eff * scale.flops_scale, kEpsilon);
    const double memory_s =
        task.memory_bytes /
        std::max(bandwidth_bytes_per_s(device) * cache_eff * scale.bandwidth_scale, kEpsilon);
    return (std::max(compute_s, memory_s) + estimate_cpu_mismatch_penalty_seconds(task)) *
               feature_affinity_multiplier(task, device) +
           launch_overhead_seconds(scale);
}

double estimate_gpu_task_time_seconds(const Task& task, const hardware_topology::Device* device) {
    if (task_has_tag_or_feature(task, "gpu_unsupported")) {
        return std::numeric_limits<double>::infinity();
    }
    if (device == nullptr || peak_flops_per_s(device) <= 0.0 || bandwidth_bytes_per_s(device) <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const auto scale = device_operator_scale(task, device);
    const double compute_s = task.compute_flops / std::max(peak_flops_per_s(device) * scale.flops_scale, kEpsilon);
    const double memory_s =
        task.memory_bytes / std::max(bandwidth_bytes_per_s(device) * scale.bandwidth_scale, kEpsilon);
    return std::max(compute_s, memory_s) + estimate_gpu_mismatch_penalty_seconds(task) +
           launch_overhead_seconds(scale);
}

double estimate_cpu_vector_efficiency(const Task& task) {
    (void)task;
    return 1.0;
}

double estimate_cpu_thread_efficiency(const Task& task, const hardware_topology::Device* device) {
    (void)task;
    (void)device;
    return 1.0;
}

double estimate_cpu_cache_efficiency(const Task& task) {
    (void)task;
    return 1.0;
}

double estimate_gpu_occupancy(const Task& task) {
    const double items = work_items(task);
    constexpr double kFullWaveItems = 120.0 * 2048.0;
    double occupancy = std::min(1.0, items / kFullWaveItems);
    if (items >= 1024.0 * 1024.0) {
        occupancy = std::max(occupancy, 0.85);
    } else if (items >= 65536.0) {
        occupancy = std::max(occupancy, 0.55);
    } else if (items >= 8192.0) {
        occupancy = std::max(occupancy, 0.25);
    }
    return std::max(0.05, occupancy);
}

double estimate_gpu_simt_efficiency(const Task& task) {
    double efficiency = 0.95;
    if (has_control_flow(task)) {
        efficiency *= 0.20;
    }
    if (has_data_dependency(task)) {
        efficiency *= 0.65;
    }
    if (has_dynamic_shape(task)) {
        efficiency *= 0.75;
    }
    if (work_items(task) < 32768.0) {
        efficiency *= 0.55;
    }
    return std::max(0.05, efficiency);
}

double estimate_gpu_coalescing_efficiency(const Task& task) {
    const std::string access = access_pattern(task);
    if (access == "contiguous") {
        return 0.95;
    }
    if (access == "strided") {
        return 0.65;
    }
    if (access == "random") {
        return 0.25;
    }
    if (access == "sparse") {
        return 0.18;
    }
    return 0.50;
}

double estimate_cpu_mismatch_penalty_seconds(const Task& task) {
    (void)task;
    return 0.0;
}

double estimate_gpu_mismatch_penalty_seconds(const Task& task) {
    (void)task;
    return 0.0;
}

}  // namespace mapping
