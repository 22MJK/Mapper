#include "mapper/mapper.h"

#include "mapping/operator_catalog.h"

#include "llm/taskgraph_builder.h"
#include "mapping/cost_model.h"
#include "mapping/mapper.h"
#include "mapping/schedule_model.h"
#include "mapping/strategies.h"
#include "taskflow/json.h"
#include "taskflow/taskflow.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mapper {
namespace {

constexpr std::uint64_t kAutoMatrixMinVectorBytes = 4 * 1024;
constexpr std::uint64_t kAutoMatrixMinMatrixBytes = 256 * 1024;
constexpr double kAutoMatrixMinTaskFlops = 1e5;
constexpr double kAutoMatrixMinTaskBytes = 64 * 1024;
constexpr std::size_t kAutoMatrixMaxShards = 4;
constexpr double kMatrixParallelMaxRelativeTime = 0.98;

using mapping::canonical_comm_kind;
using mapping::estimate_collective_time_seconds;
using mapping::estimate_makespan_seconds;
using mapping::is_collective_kind;

std::size_t dtype_size(workload::DType dtype) {
    switch (dtype) {
        case workload::DType::FP16:
        case workload::DType::BF16:
            return 2;
        case workload::DType::FP32:
        case workload::DType::INT32:
            return 4;
        case workload::DType::FP64:
        case workload::DType::INT64:
            return 8;
        case workload::DType::INT8:
        case workload::DType::UINT8:
            return 1;
    }
    return 4;
}

std::uint64_t tensor_bytes(const workload::Tensor& tensor) {
    if (tensor.size_bytes > 0) {
        return tensor.size_bytes;
    }
    if (tensor.num_elements.has_value()) {
        return static_cast<std::uint64_t>(*tensor.num_elements * dtype_size(tensor.dtype));
    }
    if (tensor.shape.empty()) {
        return static_cast<std::uint64_t>(dtype_size(tensor.dtype));
    }
    long double total = 1.0L;
    for (auto dim : tensor.shape) {
        if (dim <= 0) {
            return 0;
        }
        total *= static_cast<long double>(dim);
        if (total > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
            return std::numeric_limits<std::uint64_t>::max();
        }
    }
    total *= static_cast<long double>(dtype_size(tensor.dtype));
    if (total > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(total);
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

std::optional<std::string> pinned_device_tag(const mapping::Task& task) {
    static const std::string prefix = "device:";
    for (const auto* bag : {&task.tags, &task.features}) {
        for (const auto& entry : *bag) {
            if (entry.rfind(prefix, 0) == 0 && entry.size() > prefix.size()) {
                return entry.substr(prefix.size());
            }
        }
    }
    return std::nullopt;
}

bool has_task_tag_or_feature(const mapping::Task& task, const std::string& value) {
    return task.tags.find(value) != task.tags.end() || task.features.find(value) != task.features.end();
}

bool is_device_type(const hardware_topology::Device* device, const std::string& expected) {
    if (device == nullptr) {
        return false;
    }
    std::string type = device->type;
    for (char& ch : type) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return type == expected;
}

bool pinned_device_supported_by_task(const mapping::Task& task,
                                     const hardware_topology::Device* device) {
    if (device == nullptr || !device->compute_capable) {
        return false;
    }
    if (is_device_type(device, "cpu") && has_task_tag_or_feature(task, "cpu_unsupported")) {
        return false;
    }
    if (is_device_type(device, "gpu") && has_task_tag_or_feature(task, "gpu_unsupported")) {
        return false;
    }
    return true;
}

std::optional<mapping::MappingPlan> build_fully_pinned_mapping_plan(
    const mapping::TaskGraph& graph,
    const hardware_topology::HardwareTopology& topology) {
    mapping::MappingPlan plan;
    const auto& tasks = graph.topological_order();
    plan.assignments.reserve(tasks.size());
    for (const auto& task : tasks) {
        const auto pin = pinned_device_tag(task);
        if (!pin.has_value()) {
            return std::nullopt;
        }
        const auto* device = topology.device(*pin);
        if (device == nullptr || !device->compute_capable) {
            throw std::runtime_error("Pinned compute device not found: " + *pin);
        }
        if (!pinned_device_supported_by_task(task, device)) {
            throw std::runtime_error("Pinned device is not supported by task profile: " + task.name);
        }
        plan.assignments.emplace(task.name, *pin);
    }
    return plan;
}

std::optional<workload::AccessKind> parse_access_kind(const std::string& value) {
    const auto normalized = canonical_token(value);
    if (normalized == "dense" || normalized == "contiguous") {
        return workload::AccessKind::DENSE;
    }
    if (normalized == "sparse_csr" || normalized == "csr") {
        return workload::AccessKind::SPARSE_CSR;
    }
    if (normalized == "sparse_csc" || normalized == "csc") {
        return workload::AccessKind::SPARSE_CSC;
    }
    if (normalized == "sparse_coo" || normalized == "coo" || normalized == "coo_import") {
        return workload::AccessKind::SPARSE_COO;
    }
    if (normalized == "sparse_block" || normalized == "sparse_bsr" || normalized == "bsr" ||
        normalized == "block_sparse") {
        return workload::AccessKind::SPARSE_BLOCK;
    }
    if (normalized == "row_wise") {
        return workload::AccessKind::ROW_WISE;
    }
    if (normalized == "col_wise") {
        return workload::AccessKind::COL_WISE;
    }
    return std::nullopt;
}

int access_kind_rank(workload::AccessKind access) {
    switch (access) {
        case workload::AccessKind::SPARSE_CSR:
        case workload::AccessKind::SPARSE_CSC:
        case workload::AccessKind::SPARSE_COO:
        case workload::AccessKind::SPARSE_BLOCK:
            return 3;
        case workload::AccessKind::ROW_WISE:
        case workload::AccessKind::COL_WISE:
            return 2;
        case workload::AccessKind::DENSE:
            return 1;
    }
    return 0;
}

const char* access_kind_name(workload::AccessKind access) {
    switch (access) {
        case workload::AccessKind::DENSE:
            return "dense";
        case workload::AccessKind::SPARSE_CSR:
            return "sparse_csr";
        case workload::AccessKind::SPARSE_CSC:
            return "sparse_csc";
        case workload::AccessKind::SPARSE_COO:
            return "sparse_coo";
        case workload::AccessKind::SPARSE_BLOCK:
            return "sparse_block";
        case workload::AccessKind::ROW_WISE:
            return "row-wise";
        case workload::AccessKind::COL_WISE:
            return "col-wise";
    }
    return "dense";
}

workload::AccessKind effective_input_access(const workload::TensorUse& input, const workload::Tensor& tensor) {
    return input.access_explicit ? input.access : tensor.access_pattern;
}

bool is_row_local_access(workload::AccessKind access) {
    return access == workload::AccessKind::DENSE || access == workload::AccessKind::ROW_WISE ||
           access == workload::AccessKind::SPARSE_CSR || access == workload::AccessKind::SPARSE_BLOCK;
}

bool is_col_local_access(workload::AccessKind access) {
    return access == workload::AccessKind::DENSE || access == workload::AccessKind::COL_WISE ||
           access == workload::AccessKind::SPARSE_CSC;
}

std::string task_access_pattern(const workload::Task& task,
                                const std::unordered_map<std::string, workload::Tensor>& tensor_map) {
    workload::AccessKind access = workload::AccessKind::DENSE;
    for (const auto& input : task.inputs) {
        const auto tensor_it = tensor_map.find(input.tensor_id);
        const auto input_access =
            tensor_it == tensor_map.end() ? input.access : effective_input_access(input, tensor_it->second);
        if (access_kind_rank(input_access) > access_kind_rank(access)) {
            access = input_access;
        }
    }
    return access_kind_name(access);
}

std::string canonical_task_subtype(std::string value) {
    return mapping::canonical_operator_subtype(std::move(value));
}

std::unordered_map<std::string, std::vector<std::string>> build_group_members(
    const workload::Workload& workload,
    const hardware_topology::HardwareTopology& topology) {
    std::unordered_map<std::string, std::vector<std::string>> group_members;
    group_members.reserve(workload.device_groups().size());
    for (const auto& group : workload.device_groups()) {
        if (group.members.size() == 1 && group.members[0] == "all") {
            std::vector<std::string> members;
            for (const auto* device : topology.compute_devices()) {
                members.push_back(device->id);
            }
            group_members[group.id] = std::move(members);
        } else {
            group_members[group.id] = group.members;
        }
    }
    return group_members;
}

std::vector<std::string> gpu_compute_members(const hardware_topology::HardwareTopology& topology) {
    std::vector<std::string> members;
    for (const auto* device : topology.compute_devices()) {
        if (device != nullptr && canonical_token(device->type) == "gpu") {
            members.push_back(device->id);
        }
    }
    std::sort(members.begin(), members.end());
    return members;
}

std::size_t group_size(const workload::Distribution& dist,
                       const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    if (dist.group.empty()) {
        return 1;
    }
    const auto it = groups.find(dist.group);
    if (it == groups.end() || it->second.empty()) {
        return 1;
    }
    return it->second.size();
}

std::size_t distribution_parts(const workload::Tensor& tensor,
                               const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    return group_size(tensor.distribution, groups);
}

double estimate_transfer_bytes(const workload::Tensor& tensor,
                               workload::AccessKind access,
                               const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    const auto total_bytes = static_cast<double>(tensor_bytes(tensor));
    if (total_bytes <= 0.0) {
        return 0.0;
    }

    const std::size_t parts = distribution_parts(tensor, groups);
    switch (tensor.distribution.kind) {
        case workload::DistKind::REPLICATED:
            if (tensor.replication.has_value() && tensor.replication->mode == "cached") {
                return 0.0;
            }
            return total_bytes;
        case workload::DistKind::NONE:
            return total_bytes;
        case workload::DistKind::BLOCK:
        case workload::DistKind::CYCLIC:
            if (parts == 0) {
                return total_bytes;
            }
            if ((tensor.distribution.axis == 0 &&
                 (access == workload::AccessKind::ROW_WISE || access == workload::AccessKind::SPARSE_CSR ||
                  access == workload::AccessKind::SPARSE_BLOCK)) ||
                (tensor.distribution.axis == 1 &&
                 (access == workload::AccessKind::COL_WISE || access == workload::AccessKind::SPARSE_CSC))) {
                return total_bytes / static_cast<double>(parts);
            }
            return total_bytes;
    }
    return total_bytes;
}

double estimate_collective_bytes(const workload::Tensor& tensor,
                                 const std::string& comm_kind,
                                 const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    const auto total_bytes = static_cast<double>(tensor_bytes(tensor));
    if (total_bytes <= 0.0) {
        return 0.0;
    }
    const std::size_t parts = distribution_parts(tensor, groups);
    if (parts <= 1) {
        return total_bytes;
    }
    const std::string kind = canonical_comm_kind(comm_kind);
    if (kind == "allgather" || kind == "reducescatter" || kind == "alltoall") {
        return total_bytes / static_cast<double>(parts);
    }
    return total_bytes;
}

mapping::TaskGraph annotate_comm_bytes(const mapping::TaskGraph& graph,
                                       const workload::Workload& workload,
                                       const hardware_topology::HardwareTopology& topology) {
    mapping::TaskGraph annotated;
    const auto& ordered = graph.topological_order();
    for (const auto& task : ordered) {
        annotated.add_task(task);
    }

    std::unordered_map<std::string, workload::Tensor> tensor_map;
    tensor_map.reserve(workload.tensors().size());
    for (const auto& tensor : workload.tensors()) {
        tensor_map.emplace(tensor.id, tensor);
    }

    const auto group_members = build_group_members(workload, topology);

    for (const auto& task : ordered) {
        for (const auto& edge : graph.successors(task.name)) {
            double bytes = edge.tensor_bytes;
            if (bytes <= 0.0) {
                auto it = tensor_map.find(edge.tensor_id);
                if (it != tensor_map.end()) {
                    const auto& tensor = it->second;
                    const auto access = parse_access_kind(edge.access_pattern).value_or(tensor.access_pattern);
                    if (is_collective_kind(edge.comm_kind)) {
                        bytes = estimate_collective_bytes(tensor, edge.comm_kind, group_members);
                    } else {
                        bytes = estimate_transfer_bytes(tensor, access, group_members);
                    }
                }
            }
            annotated.add_edge(edge.src,
                               edge.dst,
                               bytes,
                               edge.tensor_id,
                               canonical_comm_kind(edge.comm_kind),
                               edge.access_pattern,
                               edge.comm_participants,
                               edge.comm_group,
                               edge.dtype);
        }
    }
    return annotated;
}

struct PlacementInfo {
    std::string group;
    std::string parallelism;
};

enum class MatrixOpKind {
    NONE,
    POINTWISE,
    REDUCTION,
    MATRIX,
};

struct PartitionCandidate {
    std::string tensor_id;
    int axis{-1};
    std::size_t parts{1};
};

std::optional<int> infer_auto_partition_axis(const workload::Tensor& tensor,
                                             workload::AccessKind access,
                                             MatrixOpKind op_kind);

std::vector<PartitionCandidate> prioritize_partition_candidates(const std::vector<PartitionCandidate>& candidates,
                                                                const std::unordered_map<std::string, workload::Tensor>& tensor_map);

double estimate_matrix_split_communication_seconds(const workload::Task& task,
                                                   const PartitionCandidate& anchor,
                                                   MatrixOpKind op_kind,
                                                   const std::unordered_map<std::string, workload::Tensor>& tensor_map,
                                                   const std::vector<double>& ratios,
                                                   std::size_t shard_count,
                                                   const hardware_topology::HardwareTopology& topology);

double task_exec_time_seconds(const mapping::Task& task, const hardware_topology::Device* device);

mapping::Task make_timing_task(double compute_flops,
                               double memory_bytes,
                               std::string subtype = {},
                               std::string access_pattern = {});

struct MatrixSplitDecision {
    bool enabled{false};
    MatrixOpKind op_kind{MatrixOpKind::NONE};
    PartitionCandidate anchor;
    std::vector<std::string> devices;
    std::vector<double> ratios;
};

enum class ParallelMode {
    NONE,
    MATRIX,
    AUTO,
};

std::optional<ParallelMode> parse_parallel_mode(const std::string& value) {
    if (value.empty() || value == "none") {
        return ParallelMode::NONE;
    }
    if (value == "matrix" || value == "matrix_parallel") {
        return ParallelMode::MATRIX;
    }
    if (value == "auto") {
        return ParallelMode::AUTO;
    }
    if (value == "llm") {
        return ParallelMode::NONE;
    }
    return std::nullopt;
}

std::unordered_map<std::string, PlacementInfo> build_placement(const workload::Workload& workload) {
    std::unordered_map<std::string, PlacementInfo> placement;
    placement.reserve(workload.tasks().size());
    for (const auto& task : workload.tasks()) {
        PlacementInfo info;
        info.group = task.placement_group;
        info.parallelism = task.placement_parallelism;
        placement.emplace(task.name, std::move(info));
    }
    return placement;
}

std::string canonical_op_name(std::string value) {
    return mapping::canonical_operator_subtype(std::move(value));
}

bool tensor_has_splittable_shape(const workload::Tensor& tensor) {
    if (tensor.shape.empty()) {
        return false;
    }
    if (tensor.shape.size() == 1) {
        return tensor.shape[0] > 1;
    }
    return tensor.shape[0] > 1 || tensor.shape[1] > 1;
}

bool tensor_large_enough_for_auto_split(const workload::Tensor& tensor) {
    if (!tensor_has_splittable_shape(tensor)) {
        return false;
    }
    const auto bytes = tensor_bytes(tensor);
    if (tensor.shape.size() >= 2) {
        return bytes >= kAutoMatrixMinMatrixBytes;
    }
    return bytes >= kAutoMatrixMinVectorBytes;
}

MatrixOpKind classify_matrix_op(const workload::Task& task) {
    const std::string op = canonical_op_name(task.op);
    if (op.empty() || op == "scal") {
        return MatrixOpKind::NONE;
    }
    if (op == "dot" || op == "nrm2") {
        return MatrixOpKind::REDUCTION;
    }
    if (op == "axpy" || op == "copy") {
        return MatrixOpKind::POINTWISE;
    }
    if (op == "spmv" || op == "sptrsv" || op == "mv" || op == "gemm" || op == "spgemm" || op == "trsm" || op == "trsv" ||
        op == "potrf") {
        return MatrixOpKind::MATRIX;
    }
    if (task.compute_flops > 0.0 && (task.inputs.size() + task.outputs.size()) > 0) {
        return MatrixOpKind::POINTWISE;
    }
    return MatrixOpKind::NONE;
}

bool op_supports_auto_matrix_parallel(const workload::Task& task) {
    const std::string op = canonical_op_name(task.op);
    return op == "spmv" || op == "mv" || op == "gemm" || op == "spgemm";
}

bool task_looks_auto_matrix_parallel_candidate(const workload::Task& task,
                                               const std::unordered_map<std::string, workload::Tensor>& tensor_map) {
    if (!op_supports_auto_matrix_parallel(task)) {
        return false;
    }
    if (task.compute_flops < kAutoMatrixMinTaskFlops &&
        task.memory_bytes < kAutoMatrixMinTaskBytes) {
        return false;
    }

    auto touches_large_tensor = [&](const std::string& tensor_id) {
        const auto it = tensor_map.find(tensor_id);
        return it != tensor_map.end() && tensor_large_enough_for_auto_split(it->second);
    };

    for (const auto& output : task.outputs) {
        if (touches_large_tensor(output)) {
            return true;
        }
    }
    for (const auto& input : task.inputs) {
        if (touches_large_tensor(input.tensor_id)) {
            return true;
        }
    }
    return false;
}

std::vector<std::string> resolve_group_members(
    const std::unordered_map<std::string, std::vector<std::string>>& group_members,
    const std::string& group,
    const hardware_topology::HardwareTopology& topology) {
    if (!group.empty()) {
        const auto it = group_members.find(group);
        if (it != group_members.end() && !it->second.empty()) {
            return it->second;
        }
    }
    std::vector<std::string> members;
    for (const auto* device : topology.compute_devices()) {
        members.push_back(device->id);
    }
    return members;
}

std::vector<std::string> filter_compute_members(const std::vector<std::string>& members,
                                                const hardware_topology::HardwareTopology& topology) {
    std::vector<std::string> filtered;
    filtered.reserve(members.size());
    for (const auto& member : members) {
        const auto* device = topology.device(member);
        if (device != nullptr && device->compute_capable) {
            filtered.push_back(member);
        }
    }
    return filtered;
}

struct WorkloadParallelShape {
    bool has_matrix_candidates{false};
    std::size_t tasks_with_matrix_hints{0};
    std::size_t tasks_with_auto_matrix_candidates{0};
    std::size_t tensors_with_distributed_layout{0};
    std::size_t tensors_with_splittable_shape{0};
};

WorkloadParallelShape analyze_workload_parallel_shape(const workload::Workload& workload) {
    WorkloadParallelShape shape;
    std::unordered_map<std::string, workload::Tensor> tensor_map;
    tensor_map.reserve(workload.tensors().size());
    for (const auto& tensor : workload.tensors()) {
        tensor_map.emplace(tensor.id, tensor);
        if (tensor_has_splittable_shape(tensor)) {
            shape.tensors_with_splittable_shape += 1;
        }
    }
    for (const auto& task : workload.tasks()) {
        if (task.placement_parallelism == "matrix_parallel") {
            shape.tasks_with_matrix_hints += 1;
        }
        if (task_looks_auto_matrix_parallel_candidate(task, tensor_map)) {
            shape.tasks_with_auto_matrix_candidates += 1;
        }
    }
    for (const auto& tensor : workload.tensors()) {
        if (tensor.distribution.kind == workload::DistKind::BLOCK ||
            tensor.distribution.kind == workload::DistKind::CYCLIC) {
            shape.tensors_with_distributed_layout += 1;
        }
    }

    const bool explicit_matrix_signal = shape.tasks_with_matrix_hints > 0 &&
                                        (shape.tensors_with_distributed_layout > 0 ||
                                         shape.tensors_with_splittable_shape > 0);
    const bool auto_matrix_signal =
        shape.tasks_with_auto_matrix_candidates > 0 && shape.tensors_with_splittable_shape > 0;
    shape.has_matrix_candidates = explicit_matrix_signal || auto_matrix_signal;
    return shape;
}

bool is_partitioned_tensor(const workload::Tensor& tensor,
                           const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    if (tensor.distribution.kind != workload::DistKind::BLOCK &&
        tensor.distribution.kind != workload::DistKind::CYCLIC) {
        return false;
    }
    return distribution_parts(tensor, groups) > 1;
}

bool input_is_locally_compatible(const PartitionCandidate& anchor,
                                 const workload::Tensor& tensor,
                                 const workload::TensorUse& use,
                                 const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    if (!is_partitioned_tensor(tensor, groups)) {
        return true;
    }
    if (tensor.shape.size() <= 1) {
        return tensor.distribution.axis == anchor.axis;
    }
    if (tensor.distribution.axis != anchor.axis) {
        return false;
    }
    const auto access = effective_input_access(use, tensor);
    if (anchor.axis == 0) {
        return is_row_local_access(access);
    }
    if (anchor.axis == 1) {
        return is_col_local_access(access);
    }
    return access == workload::AccessKind::DENSE;
}

bool output_is_locally_compatible(const PartitionCandidate& anchor,
                                  const workload::Tensor& tensor,
                                  const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    if (!is_partitioned_tensor(tensor, groups)) {
        return true;
    }
    if (tensor.shape.size() <= 1) {
        return tensor.distribution.axis == anchor.axis;
    }
    return tensor.distribution.axis == anchor.axis && distribution_parts(tensor, groups) == anchor.parts;
}

bool inferred_tensor_matches_anchor(const PartitionCandidate& anchor,
                                    const workload::Tensor& tensor,
                                    workload::AccessKind access,
                                    MatrixOpKind op_kind) {
    const auto axis = infer_auto_partition_axis(tensor, access, op_kind);
    if (!axis.has_value()) {
        return false;
    }
    if (*axis != anchor.axis) {
        return false;
    }
    if (tensor.shape.empty()) {
        return false;
    }
    const auto extent = tensor.shape.size() == 1 ? tensor.shape[0] : tensor.shape[*axis];
    return extent > 1;
}

double device_split_score(const workload::Task& task,
                          MatrixOpKind op_kind,
                          const hardware_topology::Device* device) {
    if (device == nullptr) {
        return 0.0;
    }
    const double compute_cap = std::max(0.0, device->peak_gflops);
    const double memory_cap = std::max(0.0, device->mem_bw_gbps);
    const double concurrency = std::max(1, device->max_concurrent);

    if (compute_cap <= 0.0 && memory_cap <= 0.0) {
        return 0.0;
    }

    const double compute_weight =
        (op_kind == MatrixOpKind::MATRIX) ? 0.75 : ((op_kind == MatrixOpKind::REDUCTION) ? 0.20 : 0.35);
    const double memory_weight = 1.0 - compute_weight;

    const double task_compute = std::max(0.0, task.compute_flops);
    const double task_memory = std::max(0.0, task.memory_bytes);
    const bool has_compute = task_compute > 0.0;
    const bool has_memory = task_memory > 0.0;

    double score = 0.0;
    if (has_compute && compute_cap > 0.0) {
        score += compute_weight * compute_cap;
    }
    if (has_memory && memory_cap > 0.0) {
        score += memory_weight * memory_cap;
    }
    if (!has_compute && !has_memory) {
        score = compute_cap + memory_cap;
    } else if (!has_compute) {
        score = memory_cap;
    } else if (!has_memory) {
        score = compute_cap;
    }
    return score * static_cast<double>(concurrency);
}

std::vector<PartitionCandidate> collect_partition_candidates(
    const workload::Task& task,
    const std::unordered_map<std::string, workload::Tensor>& tensor_map,
    const std::unordered_map<std::string, std::vector<std::string>>& group_members) {
    std::vector<PartitionCandidate> out;
    out.reserve(task.outputs.size() + task.inputs.size());
    auto add_candidate = [&](const std::string& tensor_id) {
        const auto it = tensor_map.find(tensor_id);
        if (it == tensor_map.end() || !is_partitioned_tensor(it->second, group_members)) {
            return;
        }
        PartitionCandidate candidate;
        candidate.tensor_id = tensor_id;
        candidate.axis = it->second.distribution.axis;
        candidate.parts = distribution_parts(it->second, group_members);
        out.push_back(std::move(candidate));
    };
    for (const auto& output : task.outputs) {
        add_candidate(output);
    }
    for (const auto& input : task.inputs) {
        add_candidate(input.tensor_id);
    }
    return out;
}

std::optional<int> infer_auto_partition_axis(const workload::Tensor& tensor,
                                             workload::AccessKind access,
                                             MatrixOpKind op_kind) {
    if (!tensor_has_splittable_shape(tensor)) {
        return std::nullopt;
    }
    if (tensor.shape.size() == 1) {
        return 0;
    }
    if (access == workload::AccessKind::SPARSE_COO) {
        return std::nullopt;
    }
    if (access == workload::AccessKind::ROW_WISE || access == workload::AccessKind::SPARSE_CSR ||
        access == workload::AccessKind::SPARSE_BLOCK) {
        return 0;
    }
    if (access == workload::AccessKind::COL_WISE || access == workload::AccessKind::SPARSE_CSC) {
        return 1;
    }
    if (op_kind == MatrixOpKind::REDUCTION) {
        return 0;
    }
    const auto rows = tensor.shape.size() > 0 ? tensor.shape[0] : 1;
    const auto cols = tensor.shape.size() > 1 ? tensor.shape[1] : 1;
    return rows >= cols ? 0 : 1;
}

std::vector<PartitionCandidate> collect_auto_partition_candidates(
    const workload::Task& task,
    MatrixOpKind op_kind,
    const std::unordered_map<std::string, workload::Tensor>& tensor_map,
    std::size_t max_parts) {
    std::vector<PartitionCandidate> out;
    out.reserve(task.outputs.size() + task.inputs.size());

    auto add_candidate = [&](const std::string& tensor_id, workload::AccessKind access) {
        const auto it = tensor_map.find(tensor_id);
        if (it == tensor_map.end()) {
            return;
        }
        const auto& tensor = it->second;
        if (!tensor_large_enough_for_auto_split(tensor)) {
            return;
        }
        const auto axis = infer_auto_partition_axis(tensor, access, op_kind);
        if (!axis.has_value()) {
            return;
        }
        const auto tensor_extent =
            tensor.shape.size() == 1 ? static_cast<std::size_t>(std::max<std::int64_t>(1, tensor.shape[0]))
                                     : static_cast<std::size_t>(std::max<std::int64_t>(1, tensor.shape[*axis]));
        const auto parts = std::min<std::size_t>(std::min<std::size_t>(max_parts, kAutoMatrixMaxShards), tensor_extent);
        if (parts <= 1) {
            return;
        }
        out.push_back(PartitionCandidate{tensor_id, *axis, parts});
    };

    for (const auto& output : task.outputs) {
        const auto tensor_it = tensor_map.find(output);
        const auto access = tensor_it == tensor_map.end() ? workload::AccessKind::DENSE : tensor_it->second.access_pattern;
        add_candidate(output, access);
    }
    for (const auto& input : task.inputs) {
        const auto tensor_it = tensor_map.find(input.tensor_id);
        const auto access =
            tensor_it == tensor_map.end() ? input.access : effective_input_access(input, tensor_it->second);
        add_candidate(input.tensor_id, access);
    }

    return prioritize_partition_candidates(out, tensor_map);
}

std::vector<PartitionCandidate> prioritize_partition_candidates(const std::vector<PartitionCandidate>& candidates,
                                                                const std::unordered_map<std::string, workload::Tensor>& tensor_map) {
    auto ranked = candidates;
    std::sort(ranked.begin(),
              ranked.end(),
              [&](const PartitionCandidate& lhs, const PartitionCandidate& rhs) {
                  const auto lhs_it = tensor_map.find(lhs.tensor_id);
                  const auto rhs_it = tensor_map.find(rhs.tensor_id);
                  const auto lhs_bytes = lhs_it == tensor_map.end() ? 0 : tensor_bytes(lhs_it->second);
                  const auto rhs_bytes = rhs_it == tensor_map.end() ? 0 : tensor_bytes(rhs_it->second);
                  if (lhs_bytes != rhs_bytes) {
                      return lhs_bytes > rhs_bytes;
                  }
                  if (lhs.parts != rhs.parts) {
                      return lhs.parts > rhs.parts;
                  }
                  return lhs.tensor_id < rhs.tensor_id;
              });
    ranked.erase(std::unique(ranked.begin(),
                             ranked.end(),
                             [](const PartitionCandidate& lhs, const PartitionCandidate& rhs) {
                                 return lhs.tensor_id == rhs.tensor_id && lhs.axis == rhs.axis && lhs.parts == rhs.parts;
                             }),
                 ranked.end());
    return ranked;
}

bool same_partition_ratios(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        if (std::abs(lhs[i] - rhs[i]) > 1e-6) {
            return false;
        }
    }
    return true;
}

MatrixSplitDecision analyze_matrix_parallel_task(
    const workload::Task& task,
    const PlacementInfo& placement,
    const std::unordered_map<std::string, workload::Tensor>& tensor_map,
    const std::unordered_map<std::string, std::vector<std::string>>& group_members,
    const hardware_topology::HardwareTopology& topology) {
    MatrixSplitDecision decision;
    decision.op_kind = classify_matrix_op(task);
    if (decision.op_kind == MatrixOpKind::NONE) {
        return decision;
    }
    const bool explicit_matrix = placement.parallelism == "matrix_parallel";

    const auto members =
        filter_compute_members(resolve_group_members(group_members, placement.group, topology), topology);
    if (members.size() <= 1) {
        return decision;
    }

    auto candidates = collect_partition_candidates(task, tensor_map, group_members);
    if (candidates.empty()) {
        const auto auto_candidates = collect_auto_partition_candidates(task, decision.op_kind, tensor_map, members.size());
        candidates.insert(candidates.end(), auto_candidates.begin(), auto_candidates.end());
    }
    candidates = prioritize_partition_candidates(candidates, tensor_map);
    if (candidates.empty()) {
        return decision;
    }

    for (const auto& candidate : candidates) {
        if (candidate.parts <= 1) {
            continue;
        }
        if (candidate.parts > members.size()) {
            continue;
        }
        bool compatible = true;
        for (const auto& input : task.inputs) {
            const auto tensor_it = tensor_map.find(input.tensor_id);
            if (tensor_it == tensor_map.end()) {
                compatible = false;
                break;
            }
            const auto input_auto = !is_partitioned_tensor(tensor_it->second, group_members);
            const auto input_access = effective_input_access(input, tensor_it->second);
            const bool input_ok = input_auto
                                      ? inferred_tensor_matches_anchor(candidate, tensor_it->second, input_access, decision.op_kind)
                                      : input_is_locally_compatible(candidate, tensor_it->second, input, group_members);
            if (!input_ok) {
                compatible = false;
                break;
            }
        }
        if (!compatible) {
            continue;
        }
        for (const auto& output_id : task.outputs) {
            const auto tensor_it = tensor_map.find(output_id);
            if (tensor_it == tensor_map.end()) {
                continue;
            }
            const auto& tensor = tensor_it->second;
            const bool scalar_output = tensor.shape.empty() || (tensor.num_elements.has_value() && *tensor.num_elements <= 1);
            if (scalar_output && decision.op_kind == MatrixOpKind::REDUCTION) {
                continue;
            }
            const auto output_auto = !is_partitioned_tensor(tensor, group_members);
            const bool output_ok = output_auto
                                       ? inferred_tensor_matches_anchor(candidate, tensor, tensor.access_pattern, decision.op_kind)
                                       : output_is_locally_compatible(candidate, tensor, group_members);
            if (!output_ok) {
                compatible = false;
                break;
            }
        }
        if (!compatible) {
            continue;
        }

        if (!explicit_matrix) {
            const auto anchor_it = tensor_map.find(candidate.tensor_id);
            if (anchor_it == tensor_map.end() || !tensor_large_enough_for_auto_split(anchor_it->second)) {
                continue;
            }
        }

        const std::size_t shard_count = std::min(candidate.parts, members.size());
        if (shard_count <= 1) {
            continue;
        }
        const double flops_per_shard = task.compute_flops / static_cast<double>(shard_count);
        const double bytes_per_shard = task.memory_bytes / static_cast<double>(shard_count);
        if (!explicit_matrix &&
            flops_per_shard < kAutoMatrixMinTaskFlops &&
            bytes_per_shard < kAutoMatrixMinTaskBytes) {
            continue;
        }

        struct ScoredDevice {
            std::string id;
            double score{0.0};
        };
        std::vector<ScoredDevice> scored;
        scored.reserve(members.size());
        for (const auto& member : members) {
            scored.push_back({member, device_split_score(task, decision.op_kind, topology.device(member))});
        }
        std::sort(scored.begin(),
                  scored.end(),
                  [](const ScoredDevice& a, const ScoredDevice& b) {
                      if (a.score != b.score) {
                          return a.score > b.score;
                      }
                      return a.id < b.id;
                  });

        decision.enabled = true;
        decision.anchor = candidate;
        decision.devices.clear();
        decision.ratios.clear();
        decision.devices.reserve(shard_count);
        decision.ratios.reserve(shard_count);

        double total_score = 0.0;
        for (std::size_t i = 0; i < shard_count; ++i) {
            decision.devices.push_back(scored[i].id);
            total_score += std::max(0.0, scored[i].score);
            decision.ratios.push_back(std::max(0.0, scored[i].score));
        }
        if (!(total_score > 0.0)) {
            decision.ratios.assign(shard_count, 1.0 / static_cast<double>(shard_count));
        } else {
            for (double& ratio : decision.ratios) {
                ratio /= total_score;
            }
        }

        double baseline_time = std::numeric_limits<double>::infinity();
        if (const auto* best_device = topology.device(decision.devices.front()); best_device != nullptr) {
            const auto timing_task =
                make_timing_task(task.compute_flops, task.memory_bytes, task.op, task_access_pattern(task, tensor_map));
            baseline_time = task_exec_time_seconds(timing_task, best_device);
        }

        double split_exec_time = 0.0;
        for (std::size_t i = 0; i < shard_count; ++i) {
            const auto* split_device = topology.device(decision.devices[i]);
            if (split_device == nullptr) {
                split_exec_time = std::numeric_limits<double>::infinity();
                break;
            }
            const auto shard_task =
                make_timing_task(task.compute_flops * decision.ratios[i],
                                 task.memory_bytes * decision.ratios[i],
                                 task.op,
                                 task_access_pattern(task, tensor_map));
            split_exec_time = std::max(split_exec_time, task_exec_time_seconds(shard_task, split_device));
        }
        const double split_comm_time =
            estimate_matrix_split_communication_seconds(task,
                                                        candidate,
                                                        decision.op_kind,
                                                        tensor_map,
                                                        decision.ratios,
                                                        shard_count,
                                                        topology);
        const double split_total_time = split_exec_time + split_comm_time;
        const double max_relative_time = explicit_matrix ? 1.05 : kMatrixParallelMaxRelativeTime;
        if (std::isfinite(baseline_time) && std::isfinite(split_total_time) &&
            split_total_time > baseline_time * max_relative_time) {
            decision.enabled = false;
            decision.devices.clear();
            decision.ratios.clear();
            continue;
        }

        return decision;
    }

    return decision;
}

bool uses_local_partition(const workload::Tensor& tensor,
                          const mapping::TaskEdge& edge,
                          const std::unordered_map<std::string, std::vector<std::string>>& groups) {
    if (!is_partitioned_tensor(tensor, groups)) {
        return false;
    }
    if (tensor.shape.size() <= 1) {
        return true;
    }

    const auto access = parse_access_kind(edge.access_pattern).value_or(tensor.access_pattern);
    if (tensor.distribution.axis == 0) {
        return is_row_local_access(access);
    }
    if (tensor.distribution.axis == 1) {
        return is_col_local_access(access);
    }
    return access == workload::AccessKind::DENSE;
}

double communication_bytes_for_edge(const workload::Tensor& tensor,
                                    const mapping::TaskEdge& edge,
                                    double src_ratio,
                                    double dst_ratio,
                                    std::size_t src_shards,
                                    std::size_t dst_shards,
                                    bool pairwise_partition) {
    if (is_collective_kind(edge.comm_kind)) {
        return static_cast<double>(tensor_bytes(tensor));
    }
    if (pairwise_partition) {
        const double total = static_cast<double>(tensor_bytes(tensor));
        return total * std::max(src_ratio, dst_ratio);
    }
    if (src_shards > 1 && dst_shards > 1) {
        const double total = static_cast<double>(tensor_bytes(tensor));
        return total * src_ratio * dst_ratio;
    }
    if (src_shards > 1) {
        return static_cast<double>(tensor_bytes(tensor)) * src_ratio;
    }
    if (dst_shards > 1) {
        if (pairwise_partition) {
            return static_cast<double>(tensor_bytes(tensor)) * dst_ratio;
        }
        return static_cast<double>(tensor_bytes(tensor));
    }
    return static_cast<double>(tensor_bytes(tensor));
}

mapping::Task make_timing_task(double compute_flops,
                               double memory_bytes,
                               std::string subtype,
                               std::string access_pattern) {
    mapping::Task task;
    task.type = "compute";
    task.subtype = mapping::canonical_operator_subtype(std::move(subtype));
    task.compute_flops = compute_flops;
    task.memory_bytes = memory_bytes;
    task.access_pattern = std::move(access_pattern);
    return task;
}

double estimate_matrix_split_communication_seconds(const workload::Task& task,
                                                   const PartitionCandidate& anchor,
                                                   MatrixOpKind op_kind,
                                                   const std::unordered_map<std::string, workload::Tensor>& tensor_map,
                                                   const std::vector<double>& ratios,
                                                   std::size_t shard_count,
                                                   const hardware_topology::HardwareTopology& topology) {
    if (shard_count <= 1) {
        return 0.0;
    }

    double communication_bytes = 0.0;
    for (const auto& input : task.inputs) {
        const auto tensor_it = tensor_map.find(input.tensor_id);
        if (tensor_it == tensor_map.end()) {
            continue;
        }
        const auto& tensor = tensor_it->second;
        const auto axis = infer_auto_partition_axis(tensor, effective_input_access(input, tensor), op_kind);
        const bool locally_partitioned =
            axis.has_value() && *axis == anchor.axis &&
            ((tensor.shape.size() == 1 && anchor.axis == 0) ||
             (tensor.shape.size() > static_cast<std::size_t>(anchor.axis)));
        if (!locally_partitioned) {
            communication_bytes += static_cast<double>(tensor_bytes(tensor));
        }
    }

    for (const auto& output_id : task.outputs) {
        const auto tensor_it = tensor_map.find(output_id);
        if (tensor_it == tensor_map.end()) {
            continue;
        }
        const auto& tensor = tensor_it->second;
        const auto axis = infer_auto_partition_axis(tensor, tensor.access_pattern, op_kind);
        const bool locally_partitioned =
            axis.has_value() && *axis == anchor.axis &&
            ((tensor.shape.size() == 1 && anchor.axis == 0) ||
             (tensor.shape.size() > static_cast<std::size_t>(anchor.axis)));
        if (!locally_partitioned) {
            communication_bytes += static_cast<double>(tensor_bytes(tensor));
        }
    }

    if (!(communication_bytes > 0.0)) {
        return 0.0;
    }

    double max_ratio = 0.0;
    for (double ratio : ratios) {
        max_ratio = std::max(max_ratio, ratio);
    }
    if (!(max_ratio > 0.0)) {
        max_ratio = 1.0 / static_cast<double>(shard_count);
    }

    const double effective_bytes = communication_bytes * max_ratio;
    return estimate_collective_time_seconds("allgather", effective_bytes, shard_count, topology);
}

std::int64_t scale_dimension(std::int64_t dim, double ratio) {
    if (dim <= 0 || !(ratio > 0.0)) {
        return dim;
    }
    const long double scaled = static_cast<long double>(dim) * static_cast<long double>(ratio);
    const auto rounded = static_cast<std::int64_t>(std::llround(scaled));
    return std::max<std::int64_t>(1, rounded);
}

std::uint64_t scale_count(std::uint64_t count, double ratio) {
    if (count == 0 || !(ratio > 0.0)) {
        return count;
    }
    const long double scaled = static_cast<long double>(count) * static_cast<long double>(ratio);
    if (scaled >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    const auto rounded = static_cast<std::uint64_t>(std::llround(scaled));
    return rounded == 0 ? 1 : rounded;
}

void scale_split_input_metadata(mapping::TaskInput& input,
                                const workload::Task& workload_task,
                                const std::unordered_map<std::string, workload::Tensor>& tensor_map,
                                const PartitionCandidate& anchor,
                                MatrixOpKind op_kind,
                                double ratio) {
    if (input.nonzero_elements.has_value()) {
        input.nonzero_elements = scale_count(*input.nonzero_elements, ratio);
    }

    const auto tensor_it = tensor_map.find(input.tensor_id);
    if (tensor_it == tensor_map.end()) {
        return;
    }

    const auto use_it = std::find_if(workload_task.inputs.begin(),
                                     workload_task.inputs.end(),
                                     [&](const workload::TensorUse& use) {
                                         return use.tensor_id == input.tensor_id;
                                     });
    const auto access = use_it == workload_task.inputs.end() ? tensor_it->second.access_pattern
                                                             : effective_input_access(*use_it, tensor_it->second);
    const auto axis = infer_auto_partition_axis(tensor_it->second, access, op_kind);
    if (!axis.has_value() || *axis != anchor.axis || input.shape.empty()) {
        return;
    }

    const std::size_t shape_axis = input.shape.size() == 1 ? 0 : static_cast<std::size_t>(*axis);
    if (shape_axis >= input.shape.size()) {
        return;
    }
    input.shape[shape_axis] = scale_dimension(input.shape[shape_axis], ratio);
}

void erase_device_tags(std::unordered_set<std::string>& tags) {
    for (auto it = tags.begin(); it != tags.end();) {
        if (it->rfind("device:", 0) == 0) {
            it = tags.erase(it);
        } else {
            ++it;
        }
    }
}

void scale_rank_input_metadata(mapping::TaskInput& input, double ratio) {
    input.tensor_bytes *= ratio;
    if (input.nonzero_elements.has_value()) {
        input.nonzero_elements = scale_count(*input.nonzero_elements, ratio);
    }
    if (input.shape.empty()) {
        return;
    }
    auto axis = std::distance(input.shape.begin(), std::max_element(input.shape.begin(), input.shape.end()));
    if (axis >= 0 && static_cast<std::size_t>(axis) < input.shape.size()) {
        input.shape[static_cast<std::size_t>(axis)] =
            scale_dimension(input.shape[static_cast<std::size_t>(axis)], ratio);
    }
}

bool task_disallows_gpu(const mapping::Task& task) {
    return task.tags.find("gpu_unsupported") != task.tags.end() ||
           task.features.find("gpu_unsupported") != task.features.end();
}

struct LlmTaskNameParts {
    bool valid{false};
    std::string phase;
    int layer{-1};
    std::string op;
};

struct InferredRankCollective {
    std::string kind;
    std::string group;
    std::size_t participants{0};
    double bytes{0.0};
};

bool all_digits(const std::string& value, std::size_t begin) {
    if (begin >= value.size()) {
        return false;
    }
    for (std::size_t i = begin; i < value.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(value[i]))) {
            return false;
        }
    }
    return true;
}

bool phase_tensor(const mapping::TaskEdge& edge, const std::string& phase, const std::string& name) {
    return !phase.empty() && edge.tensor_id == phase + "." + name;
}

LlmTaskNameParts parse_llm_task_name(std::string name) {
    const auto tp_pos = name.rfind(".tp");
    if (tp_pos != std::string::npos && all_digits(name, tp_pos + 3)) {
        name.resize(tp_pos);
    }

    LlmTaskNameParts parts;
    const auto first_dot = name.find('.');
    if (first_dot == std::string::npos || first_dot == 0 || first_dot + 1 >= name.size()) {
        return parts;
    }
    parts.phase = name.substr(0, first_dot);
    if (parts.phase != "prefill" && parts.phase != "decode") {
        parts.phase.clear();
        return parts;
    }

    const std::string rest = name.substr(first_dot + 1);
    if (rest.rfind("layer_", 0) == 0) {
        const auto op_dot = rest.find('.');
        if (op_dot == std::string::npos || op_dot + 1 >= rest.size()) {
            return parts;
        }
        const auto layer_begin = std::string("layer_").size();
        if (op_dot <= layer_begin) {
            return parts;
        }
        for (std::size_t i = layer_begin; i < op_dot; ++i) {
            if (!std::isdigit(static_cast<unsigned char>(rest[i]))) {
                return parts;
            }
        }
        parts.layer = std::stoi(rest.substr(layer_begin, op_dot - layer_begin));
        parts.op = rest.substr(op_dot + 1);
    } else {
        parts.op = rest;
    }

    parts.valid = !parts.phase.empty() && !parts.op.empty();
    return parts;
}

std::optional<InferredRankCollective> infer_llm_rank_collective(
    const mapping::TaskEdge& edge,
    const std::vector<std::string>& devices) {
    if (devices.size() <= 1) {
        return std::nullopt;
    }

    const std::string explicit_kind = canonical_comm_kind(edge.comm_kind);
    if (!explicit_kind.empty() && explicit_kind != "p2p" && is_collective_kind(explicit_kind)) {
        InferredRankCollective collective;
        collective.kind = explicit_kind;
        collective.group = edge.comm_group.empty() ? edge.tensor_id + "." + explicit_kind : edge.comm_group;
        collective.participants = edge.comm_participants == 0 ? devices.size() : edge.comm_participants;
        collective.bytes = edge.tensor_bytes;
        return collective;
    }
    if (!explicit_kind.empty() && explicit_kind != "p2p") {
        return std::nullopt;
    }

    const auto src = parse_llm_task_name(edge.src);
    const auto dst = parse_llm_task_name(edge.dst);
    if (!src.valid || !dst.valid || src.phase != dst.phase) {
        return std::nullopt;
    }

    InferredRankCollective collective;
    collective.participants = devices.size();
    collective.bytes = edge.tensor_bytes;

    auto set_layer_collective = [&](std::string kind, const std::string& suffix, int layer) {
        collective.kind = std::move(kind);
        collective.group = src.phase + ".layer_" + std::to_string(layer) + "." + suffix;
        return collective;
    };

    if (phase_tensor(edge, src.phase, "hidden") && dst.op == "qkv_proj" && dst.layer > 0) {
        return set_layer_collective("all_reduce", "hidden_all_reduce", dst.layer);
    }
    if (phase_tensor(edge, src.phase, "post_attn") && src.op == "o_proj" &&
        src.layer >= 0 && src.layer == dst.layer &&
        (dst.op == "mlp_gate_up" || dst.op == "moe_router")) {
        return set_layer_collective("all_reduce", "post_attn_all_reduce", src.layer);
    }
    if (phase_tensor(edge, src.phase, "moe_dispatch") &&
        src.op == "moe_dispatch" && dst.op == "moe_expert_ffn" &&
        src.layer >= 0 && src.layer == dst.layer) {
        return set_layer_collective("all_to_all", "moe_dispatch_all_to_all", src.layer);
    }
    if (phase_tensor(edge, src.phase, "moe_combine") &&
        src.op == "moe_expert_ffn" && dst.op == "moe_combine" &&
        src.layer >= 0 && src.layer == dst.layer) {
        return set_layer_collective("all_to_all", "moe_combine_all_to_all", src.layer);
    }
    if (phase_tensor(edge, src.phase, "final_hidden") && dst.op == "logits") {
        collective.kind = "all_reduce";
        collective.group = src.phase + ".final_hidden_all_reduce";
        return collective;
    }
    if (phase_tensor(edge, src.phase, "logits") && src.op == "logits" && dst.op == "sampling") {
        collective.kind = "all_gather";
        collective.group = src.phase + ".logits_all_gather";
        return collective;
    }

    return std::nullopt;
}

void add_rank_collective_edges(mapping::TaskGraph& expanded,
                               const std::vector<std::string>& srcs,
                               const std::vector<std::string>& dsts,
                               const mapping::TaskEdge& edge,
                               const InferredRankCollective& collective) {
    if (srcs.size() != dsts.size()) {
        throw std::runtime_error("Inferred rank collective '" + collective.group +
                                 "' has mismatched source and destination shard counts");
    }
    for (std::size_t rank = 0; rank < srcs.size(); ++rank) {
        expanded.add_edge(srcs[rank],
                          dsts[rank],
                          collective.bytes,
                          edge.tensor_id,
                          collective.kind,
                          edge.access_pattern,
                          collective.participants,
                          collective.group,
                          edge.dtype);
    }
}

mapping::TaskGraph expand_workload_rank_parallel(const mapping::TaskGraph& graph,
                                                  const hardware_topology::HardwareTopology& topology,
                                                  bool* expanded_any = nullptr) {
    const auto devices = gpu_compute_members(topology);
    if (devices.size() <= 1) {
        throw std::runtime_error("--workload-rank-parallel requires at least two GPU compute devices");
    }

    const double ratio = 1.0 / static_cast<double>(devices.size());
    std::unordered_map<std::string, std::vector<std::string>> shards_by_task;
    mapping::TaskGraph expanded;
    bool any_expanded = false;

    const auto& ordered = graph.topological_order();
    for (const auto& task : ordered) {
        if (task_disallows_gpu(task)) {
            expanded.add_task(task);
            continue;
        }

        auto& shards = shards_by_task[task.name];
        shards.reserve(devices.size());
        any_expanded = true;
        for (std::size_t rank = 0; rank < devices.size(); ++rank) {
            mapping::Task split = task;
            split.name = task.name + "@r" + std::to_string(rank);
            split.compute_flops *= ratio;
            split.memory_bytes *= ratio;
            split.comm_bytes *= ratio;
            for (auto& input : split.input_data) {
                scale_rank_input_metadata(input, ratio);
            }
            erase_device_tags(split.tags);
            split.tags.insert("device:" + devices[rank]);
            expanded.add_task(split);
            shards.push_back(split.name);
        }
    }

    for (const auto& task : ordered) {
        for (const auto& edge : graph.successors(task.name)) {
            const auto src_it = shards_by_task.find(edge.src);
            const auto dst_it = shards_by_task.find(edge.dst);
            const bool src_expanded = src_it != shards_by_task.end();
            const bool dst_expanded = dst_it != shards_by_task.end();
            if (!src_expanded && !dst_expanded) {
                expanded.add_edge(edge.src,
                                  edge.dst,
                                  edge.tensor_bytes,
                                  edge.tensor_id,
                                  edge.comm_kind,
                                  edge.access_pattern,
                                  edge.comm_participants,
                                  edge.comm_group,
                                  edge.dtype);
                continue;
            }

            if (src_expanded && dst_expanded) {
                const auto collective = infer_llm_rank_collective(edge, devices);
                if (collective.has_value()) {
                    add_rank_collective_edges(expanded, src_it->second, dst_it->second, edge, *collective);
                    continue;
                }
                const auto paired = std::min(src_it->second.size(), dst_it->second.size());
                for (std::size_t rank = 0; rank < paired; ++rank) {
                    expanded.add_edge(src_it->second[rank],
                                      dst_it->second[rank],
                                      edge.tensor_bytes * ratio,
                                      edge.tensor_id,
                                      edge.comm_kind,
                                      edge.access_pattern,
                                      edge.comm_participants,
                                      edge.comm_group,
                                      edge.dtype);
                }
                continue;
            }

            if (src_expanded) {
                const auto kind = canonical_comm_kind(edge.comm_kind);
                for (const auto& src : src_it->second) {
                    expanded.add_edge(src,
                                      edge.dst,
                                      edge.tensor_bytes * ratio,
                                      edge.tensor_id,
                                      (kind.empty() || kind == "p2p") ? "allgather" : edge.comm_kind,
                                      edge.access_pattern,
                                      edge.comm_participants,
                                      edge.comm_group,
                                      edge.dtype);
                }
                continue;
            }

            const auto kind = canonical_comm_kind(edge.comm_kind);
            for (const auto& dst : dst_it->second) {
                expanded.add_edge(edge.src,
                                  dst,
                                  edge.tensor_bytes * ratio,
                                  edge.tensor_id,
                                  (kind.empty() || kind == "p2p") ? "broadcast" : edge.comm_kind,
                                  edge.access_pattern,
                                  edge.comm_participants,
                                  edge.comm_group,
                                  edge.dtype);
            }
        }
    }

    if (expanded_any != nullptr) {
        *expanded_any = any_expanded;
    }
    return expanded;
}

std::string matrix_comm_kind(const mapping::TaskEdge& edge,
                             bool src_expanded,
                             bool dst_expanded,
                             bool pairwise_partition) {
    const std::string kind = canonical_comm_kind(edge.comm_kind);
    if (!kind.empty() && kind != "p2p") {
        return kind;
    }
    if (src_expanded && dst_expanded && !pairwise_partition) {
        return "alltoall";
    }
    if (src_expanded && !dst_expanded) {
        return "allgather";
    }
    if (!src_expanded && dst_expanded && !pairwise_partition) {
        return "broadcast";
    }
    return "p2p";
}

mapping::TaskGraph expand_matrix_parallel(const mapping::TaskGraph& graph,
                                          const workload::Workload& workload,
                                          const hardware_topology::HardwareTopology& topology,
                                          ParallelMode mode,
                                          bool* expanded_any = nullptr) {
    if (mode == ParallelMode::NONE) {
        if (expanded_any != nullptr) {
            *expanded_any = false;
        }
        return graph;
    }

    const auto placement = build_placement(workload);
    const auto group_members = build_group_members(workload, topology);
    std::unordered_map<std::string, workload::Tensor> tensor_map;
    tensor_map.reserve(workload.tensors().size());
    for (const auto& tensor : workload.tensors()) {
        tensor_map.emplace(tensor.id, tensor);
    }
    std::unordered_map<std::string, const workload::Task*> workload_tasks;
    workload_tasks.reserve(workload.tasks().size());
    for (const auto& task : workload.tasks()) {
        workload_tasks.emplace(task.name, &task);
    }

    struct ExpandedTask {
        std::vector<std::string> devices;
        std::vector<std::string> shards;
        std::vector<double> ratios;
        std::size_t shard_count{1};
    };

    std::unordered_map<std::string, ExpandedTask> expanded;
    mapping::TaskGraph expanded_graph;
    bool any_expanded = false;

    const auto& ordered = graph.topological_order();
    for (const auto& task : ordered) {
        const auto placement_it = placement.find(task.name);
        const PlacementInfo default_placement{"", ""};
        const auto& placement_info = placement_it == placement.end() ? default_placement : placement_it->second;

        const auto workload_it = workload_tasks.find(task.name);
        if (workload_it == workload_tasks.end()) {
            expanded_graph.add_task(task);
            continue;
        }

        const bool explicit_matrix = placement_info.parallelism == "matrix_parallel";
        const bool auto_matrix = task_looks_auto_matrix_parallel_candidate(*workload_it->second, tensor_map);
        if (!explicit_matrix && !auto_matrix) {
            expanded_graph.add_task(task);
            continue;
        }

        const auto decision =
            analyze_matrix_parallel_task(*workload_it->second, placement_info, tensor_map, group_members, topology);
        if (!decision.enabled || decision.devices.empty()) {
            expanded_graph.add_task(task);
            continue;
        }

        ExpandedTask info;
        info.devices = decision.devices;
        info.ratios = decision.ratios;
        info.shard_count = decision.devices.size();
        any_expanded = true;
        for (std::size_t i = 0; i < info.shard_count; ++i) {
            mapping::Task split = task;
            split.name = task.name + "@p" + std::to_string(i);
            const double ratio = (i < info.ratios.size()) ? info.ratios[i] : (1.0 / static_cast<double>(info.shard_count));
            split.compute_flops = task.compute_flops * ratio;
            split.memory_bytes = task.memory_bytes * ratio;
            for (auto& input : split.input_data) {
                input.tensor_bytes *= ratio;
                scale_split_input_metadata(input,
                                           *workload_it->second,
                                           tensor_map,
                                           decision.anchor,
                                           decision.op_kind,
                                           ratio);
            }
            split.tags.insert("device:" + info.devices[i]);
            expanded_graph.add_task(split);
            info.shards.push_back(split.name);
        }
        expanded.emplace(task.name, std::move(info));
    }

    for (const auto& task : ordered) {
        for (const auto& edge : graph.successors(task.name)) {
            const auto src_it = expanded.find(edge.src);
            const auto dst_it = expanded.find(edge.dst);
            const bool src_expanded = src_it != expanded.end();
            const bool dst_expanded = dst_it != expanded.end();

            auto tensor_it = tensor_map.find(edge.tensor_id);
            if (tensor_it == tensor_map.end()) {
                if (!src_expanded && !dst_expanded) {
                    expanded_graph.add_edge(edge.src, edge.dst, edge.tensor_bytes, edge.tensor_id, edge.comm_kind,
                                            edge.access_pattern, edge.comm_participants, edge.comm_group);
                } else if (src_expanded && dst_expanded) {
                    const auto paired = std::min(src_it->second.shards.size(), dst_it->second.shards.size());
                    for (std::size_t i = 0; i < paired; ++i) {
                        expanded_graph.add_edge(src_it->second.shards[i], dst_it->second.shards[i], edge.tensor_bytes,
                                                edge.tensor_id, edge.comm_kind, edge.access_pattern, edge.comm_participants,
                                                edge.comm_group);
                    }
                } else if (src_expanded) {
                    for (const auto& src_name : src_it->second.shards) {
                        expanded_graph.add_edge(src_name, edge.dst, edge.tensor_bytes, edge.tensor_id, edge.comm_kind,
                                                edge.access_pattern, edge.comm_participants, edge.comm_group);
                    }
                } else {
                    for (const auto& dst_name : dst_it->second.shards) {
                        expanded_graph.add_edge(edge.src, dst_name, edge.tensor_bytes, edge.tensor_id, edge.comm_kind,
                                                edge.access_pattern, edge.comm_participants, edge.comm_group);
                    }
                }
                continue;
            }

            const auto& tensor = tensor_it->second;
            const bool pairwise_partition =
                uses_local_partition(tensor, edge, group_members) &&
                ((src_expanded && dst_expanded && src_it->second.shard_count == dst_it->second.shard_count) ||
                 (src_expanded != dst_expanded));
            const bool same_layout =
                !src_expanded || !dst_expanded || same_partition_ratios(src_it->second.ratios, dst_it->second.ratios);
            const bool use_pairwise = pairwise_partition && same_layout;
            const std::string comm_kind = matrix_comm_kind(edge, src_expanded, dst_expanded, use_pairwise);
            if (!src_expanded && !dst_expanded) {
                expanded_graph.add_edge(edge.src,
                                        edge.dst,
                                        edge.tensor_bytes,
                                        edge.tensor_id,
                                        comm_kind,
                                        edge.access_pattern,
                                        edge.comm_participants,
                                        edge.comm_group);
                continue;
            }
            if (src_expanded && dst_expanded) {
                const auto& src_shards = src_it->second.shards;
                const auto& dst_shards = dst_it->second.shards;
                if (use_pairwise && src_shards.size() == dst_shards.size()) {
                    for (std::size_t i = 0; i < src_shards.size(); ++i) {
                        const double bytes =
                            communication_bytes_for_edge(tensor,
                                                         edge,
                                                         src_it->second.ratios[i],
                                                         dst_it->second.ratios[i],
                                                         src_it->second.shard_count,
                                                         dst_it->second.shard_count,
                                                         true);
                        expanded_graph.add_edge(src_shards[i], dst_shards[i], bytes, edge.tensor_id, comm_kind,
                                                edge.access_pattern, edge.comm_participants, edge.comm_group);
                    }
                } else {
                    for (std::size_t i = 0; i < src_shards.size(); ++i) {
                        for (std::size_t j = 0; j < dst_shards.size(); ++j) {
                            const double bytes =
                                communication_bytes_for_edge(tensor,
                                                             edge,
                                                             src_it->second.ratios[i],
                                                             dst_it->second.ratios[j],
                                                             src_it->second.shard_count,
                                                             dst_it->second.shard_count,
                                                             false);
                            expanded_graph.add_edge(src_shards[i], dst_shards[j], bytes, edge.tensor_id, comm_kind,
                                                    edge.access_pattern, edge.comm_participants, edge.comm_group);
                        }
                    }
                }
                continue;
            }
            if (src_expanded) {
                for (std::size_t i = 0; i < src_it->second.shards.size(); ++i) {
                    const double bytes =
                        communication_bytes_for_edge(tensor,
                                                     edge,
                                                     src_it->second.ratios[i],
                                                     1.0,
                                                     src_it->second.shard_count,
                                                     1,
                                                     pairwise_partition);
                    expanded_graph.add_edge(src_it->second.shards[i], edge.dst, bytes, edge.tensor_id, comm_kind,
                                            edge.access_pattern, edge.comm_participants, edge.comm_group);
                }
                continue;
            }
            for (std::size_t i = 0; i < dst_it->second.shards.size(); ++i) {
                const double bytes =
                    communication_bytes_for_edge(tensor,
                                                 edge,
                                                 1.0,
                                                 dst_it->second.ratios[i],
                                                 1,
                                                 dst_it->second.shard_count,
                                                 pairwise_partition);
                expanded_graph.add_edge(edge.src, dst_it->second.shards[i], bytes, edge.tensor_id, comm_kind,
                                        edge.access_pattern, edge.comm_participants, edge.comm_group);
            }
        }
    }

    if (expanded_any != nullptr) {
        *expanded_any = any_expanded;
    }
    return expanded_graph;
}

double task_exec_time_seconds(const mapping::Task& task, const hardware_topology::Device* device) {
    return mapping::estimate_task_time_seconds(task, device);
}

std::unique_ptr<mapping::Mapper> build_mapper_for_graph(const mapping::TaskGraph& graph,
                                                        const mapper::Options& options) {
    std::unique_ptr<mapping::Mapper> base_mapper;
    if (options.mapper == "heft") {
        base_mapper = std::make_unique<mapping::HeftMapper>();
    } else if (options.mapper == "aeft") {
        base_mapper = std::make_unique<mapping::AeftMapper>();
    } else if (options.mapper == "peft") {
        base_mapper = std::make_unique<mapping::PeftMapper>();
    } else if (options.mapper == "peft_lc" || options.mapper == "peft-lc") {
        base_mapper = std::make_unique<mapping::PeftLcMapper>();
    } else if (options.mapper == "hoft") {
        base_mapper = std::make_unique<mapping::HoftMapper>();
    } else if (options.mapper == "greedy") {
        base_mapper = std::make_unique<mapping::GreedyMapper>();
    } else if (options.mapper == "exhaustive" || options.mapper == "bruteforce" ||
               options.mapper == "brute_force" || options.mapper == "enumerate") {
        base_mapper = std::make_unique<mapping::ExhaustiveMapper>(options.force_exhaustive);
    } else if (options.mapper == "exhaustive_bb" || options.mapper == "exhaustive_pruned" ||
               options.mapper == "branch_and_bound" || options.mapper == "bb_exhaustive") {
        base_mapper = std::make_unique<mapping::ExhaustiveMapper>(
            options.force_exhaustive,
            true,
            true);
    } else {
        throw std::runtime_error("Unknown mapper: " + options.mapper);
    }

    if (options.parts <= 0) {
        return base_mapper;
    }

    mapping::LayerPartition partition;
    const auto task_partitions = partition.partition(graph, options.parts);
    std::vector<std::vector<std::string>> partitions;
    partitions.reserve(task_partitions.size());
    for (const auto& block : task_partitions) {
        std::vector<std::string> names;
        names.reserve(block.size());
        for (const auto& task : block) {
            names.push_back(task.name);
        }
        partitions.push_back(std::move(names));
    }
    return std::make_unique<mapping::PartitionerMapper>(std::move(base_mapper), std::move(partitions));
}

std::vector<const hardware_topology::Device*> select_llm_candidate_devices(
    const hardware_topology::HardwareTopology& topology) {
    std::vector<const hardware_topology::Device*> devices;
    for (const auto* device : topology.compute_devices()) {
        if (canonical_token(device->type) == "gpu") {
            devices.push_back(device);
        }
    }
    if (devices.empty()) {
        devices = topology.compute_devices();
    }
    std::sort(devices.begin(),
              devices.end(),
              [](const auto* lhs, const auto* rhs) {
                  return lhs->id < rhs->id;
              });
    return devices;
}

std::string llm_stage_partition_label(const std::vector<int>& stage_for_layer, int pp) {
    if (stage_for_layer.empty()) {
        return "";
    }
    std::ostringstream out;
    out << "[";
    for (int stage = 0; stage < pp; ++stage) {
        if (stage != 0) {
            out << "|";
        }
        int first = -1;
        int last = -1;
        for (std::size_t layer = 0; layer < stage_for_layer.size(); ++layer) {
            if (stage_for_layer[layer] == stage) {
                if (first < 0) {
                    first = static_cast<int>(layer);
                }
                last = static_cast<int>(layer);
            }
        }
        if (first < 0) {
            out << "-";
        } else if (first == last) {
            out << first;
        } else {
            out << first << "-" << last;
        }
    }
    out << "]";
    return out.str();
}

std::string llm_parallel_plan_label(const llm::LlmParallelConfig& parallel, const std::string& mode) {
    std::ostringstream out;
    out << "llm";
    if (!mode.empty()) {
        out << ":" << mode;
    }
    out << "(tp=" << parallel.tp
        << ",pp=" << parallel.pp
        << ",dp=" << parallel.dp;
    const auto stages = llm_stage_partition_label(parallel.stage_for_layer, parallel.pp);
    if (!stages.empty()) {
        out << ",stages=" << stages;
    }
    out << ")";
    return out.str();
}

std::string stage_map_key(const std::vector<int>& stage_for_layer) {
    std::ostringstream out;
    for (const int stage : stage_for_layer) {
        out << stage << ",";
    }
    return out.str();
}

std::vector<int> uniform_stage_map(int layers, int pp) {
    std::vector<int> out(static_cast<std::size_t>(layers), 0);
    for (int layer = 0; layer < layers; ++layer) {
        out[static_cast<std::size_t>(layer)] = std::min(pp - 1, (layer * pp) / std::max(1, layers));
    }
    return out;
}

bool is_mapper_linear_attention_type(const std::string& type) {
    return type == "linear_attention" || type == "linear";
}

bool is_mapper_sliding_attention_type(const std::string& type) {
    return type == "sliding_attention" || type == "sliding_window_attention" || type == "local_attention";
}

long double mapper_shard_heads(int heads, int tp) {
    if (heads <= 0) {
        return 0.0L;
    }
    if (tp <= 1 || heads % tp == 0) {
        return static_cast<long double>(heads / std::max(1, tp));
    }
    return heads < tp ? static_cast<long double>(heads) : static_cast<long double>(heads) / static_cast<long double>(tp);
}

long double llm_layer_phase_proxy_weight(const llm::LlmModelConfig& model,
                                         const llm::LlmRequestConfig& request,
                                         const llm::LlmParallelConfig& parallel,
                                         int layer,
                                         bool decode) {
    const long double batch = static_cast<long double>(decode ? request.decode_batch_size
                                                              : request.prefill_batch_size);
    const long double seq = static_cast<long double>(decode ? 1 : std::max(0, request.prompt_len));
    const long double steps = static_cast<long double>(decode ? std::max(1, request.decode_steps) : 1);
    const long double tokens = batch * seq * steps;
    if (!(tokens > 0.0L)) {
        return 0.0L;
    }

    const long double hidden = static_cast<long double>(model.hidden_size);
    const long double inter = static_cast<long double>(model.intermediate_size);
    const long double heads = static_cast<long double>(model.num_attention_heads);
    const long double kv_heads = static_cast<long double>(model.num_kv_heads);
    const long double head_dim = static_cast<long double>(model.head_dim);
    const long double tp = static_cast<long double>(std::max(1, parallel.tp));
    const std::string attention_type =
        (!model.layer_types.empty() && static_cast<std::size_t>(layer) < model.layer_types.size())
            ? model.layer_types[static_cast<std::size_t>(layer)]
            : "full_attention";
    const bool linear_attention = is_mapper_linear_attention_type(attention_type);
    const bool sliding_attention = is_mapper_sliding_attention_type(attention_type);

    const long double q_heads_per_shard = heads / tp;
    const long double kv_heads_per_shard =
        linear_attention
            ? mapper_shard_heads(model.linear_num_key_heads, parallel.tp)
            : ((model.num_kv_heads % parallel.tp == 0)
                   ? kv_heads / tp
                   : (model.num_kv_heads < parallel.tp ? kv_heads : kv_heads / tp));
    const long double q_dim_per_shard = q_heads_per_shard * head_dim;
    const long double key_dim_per_shard =
        linear_attention
            ? mapper_shard_heads(model.linear_num_key_heads, parallel.tp) *
                  static_cast<long double>(model.linear_key_head_dim)
            : kv_heads_per_shard * head_dim;
    const long double value_dim_per_shard =
        linear_attention
            ? mapper_shard_heads(model.linear_num_value_heads, parallel.tp) *
                  static_cast<long double>(model.linear_value_head_dim)
            : kv_heads_per_shard * head_dim;
    const long double qkv_out_per_shard = q_dim_per_shard + key_dim_per_shard + value_dim_per_shard;
    const long double qkv_proj = 2.0L * tokens * hidden * qkv_out_per_shard;
    const long double o_proj =
        2.0L * tokens *
        (linear_attention ? static_cast<long double>(model.linear_num_value_heads) *
                                static_cast<long double>(model.linear_value_head_dim)
                          : heads * head_dim) *
        hidden / tp;

    long double attention = 0.0L;
    if (linear_attention) {
        const long double kernel = static_cast<long double>(std::max(1, model.linear_conv_kernel_dim));
        attention = 2.0L * tokens * (q_dim_per_shard + key_dim_per_shard + value_dim_per_shard) +
                    2.0L * tokens * key_dim_per_shard * value_dim_per_shard /
                        std::max(1.0L, q_heads_per_shard) +
                    tokens * (key_dim_per_shard + value_dim_per_shard) * kernel;
    } else if (decode) {
        const long double context = static_cast<long double>(
            std::max(1, request.avg_context_len > 0 ? request.avg_context_len : request.prompt_len));
        attention = 4.0L * batch * steps * q_heads_per_shard * context * head_dim;
    } else {
        const long double effective_context =
            sliding_attention && model.sliding_window > 0
                ? std::min(static_cast<long double>(std::max(1, request.prompt_len)),
                           static_cast<long double>(model.sliding_window))
                : static_cast<long double>(std::max(1, request.prompt_len));
        attention = 4.0L * batch * q_heads_per_shard *
                    static_cast<long double>(std::max(1, request.prompt_len)) *
                    effective_context * head_dim;
    }

    long double ffn = 0.0L;
    if (model.is_moe) {
        const long double moe_inter = static_cast<long double>(
            model.moe_intermediate_size > 0 ? model.moe_intermediate_size : model.intermediate_size);
        const long double experts_per_token = static_cast<long double>(std::max(1, model.experts_per_token));
        ffn = 2.0L * tokens * hidden * static_cast<long double>(std::max(1, model.num_experts)) +
              4.0L * tokens * hidden * moe_inter * experts_per_token / tp;
    } else {
        const long double mlp_up_outputs = model.use_gated_mlp ? 2.0L * inter : inter;
        ffn = 2.0L * tokens * hidden * mlp_up_outputs / tp +
              2.0L * tokens * inter * hidden / tp;
    }

    const long double dtype_bytes = static_cast<long double>(llm::dtype_size_bytes(model.param_dtype));
    const long double activation_proxy = tokens * hidden * dtype_bytes;
    return qkv_proj + attention + o_proj + ffn + activation_proxy;
}

std::vector<long double> llm_layer_proxy_weights(const llm::LlmModelConfig& model,
                                                 const llm::LlmRequestConfig& request,
                                                 const llm::LlmParallelConfig& parallel) {
    std::vector<long double> weights(static_cast<std::size_t>(model.num_layers), 0.0L);
    for (int layer = 0; layer < model.num_layers; ++layer) {
        long double weight = 0.0L;
        if (request.prompt_len > 0) {
            weight += llm_layer_phase_proxy_weight(model, request, parallel, layer, false);
        }
        if (request.decode_steps > 0) {
            weight += llm_layer_phase_proxy_weight(model, request, parallel, layer, true);
        }
        weights[static_cast<std::size_t>(layer)] = std::max(1.0L, weight);
    }
    return weights;
}

std::vector<int> balanced_stage_map(const std::vector<long double>& weights, int pp) {
    const int layers = static_cast<int>(weights.size());
    if (pp <= 1) {
        return std::vector<int>(weights.size(), 0);
    }
    const long double inf = std::numeric_limits<long double>::infinity();
    std::vector<long double> prefix(static_cast<std::size_t>(layers + 1), 0.0L);
    for (int i = 0; i < layers; ++i) {
        prefix[static_cast<std::size_t>(i + 1)] = prefix[static_cast<std::size_t>(i)] + weights[static_cast<std::size_t>(i)];
    }
    std::vector<long double> dp(static_cast<std::size_t>((pp + 1) * (layers + 1)), inf);
    std::vector<int> parent(static_cast<std::size_t>((pp + 1) * (layers + 1)), -1);
    auto idx = [layers](int stages, int used_layers) {
        return static_cast<std::size_t>(stages * (layers + 1) + used_layers);
    };
    dp[idx(0, 0)] = 0.0L;
    for (int stages = 1; stages <= pp; ++stages) {
        for (int used = stages; used <= layers; ++used) {
            for (int cut = stages - 1; cut < used; ++cut) {
                const long double stage_weight =
                    prefix[static_cast<std::size_t>(used)] - prefix[static_cast<std::size_t>(cut)];
                const long double cost = std::max(dp[idx(stages - 1, cut)], stage_weight);
                if (cost < dp[idx(stages, used)]) {
                    dp[idx(stages, used)] = cost;
                    parent[idx(stages, used)] = cut;
                }
            }
        }
    }

    std::vector<int> out(weights.size(), 0);
    int used = layers;
    for (int stage = pp; stage >= 1; --stage) {
        const int cut = parent[idx(stage, used)];
        if (cut < 0) {
            return uniform_stage_map(layers, pp);
        }
        for (int layer = cut; layer < used; ++layer) {
            out[static_cast<std::size_t>(layer)] = stage - 1;
        }
        used = cut;
    }
    return out;
}

std::vector<std::vector<int>> llm_candidate_stage_maps(const llm::LlmModelConfig& model,
                                                       const llm::LlmRequestConfig& request,
                                                       const llm::LlmParallelConfig& parallel) {
    if (parallel.pp <= 1) {
        return {{}};
    }
    std::vector<std::vector<int>> maps;
    std::unordered_set<std::string> seen;
    auto add_map = [&](std::vector<int> map) {
        const auto key = stage_map_key(map);
        if (seen.insert(key).second) {
            maps.push_back(std::move(map));
        }
    };
    add_map(uniform_stage_map(model.num_layers, parallel.pp));
    add_map(balanced_stage_map(llm_layer_proxy_weights(model, request, parallel), parallel.pp));
    return maps;
}

struct LlmAutoParallelChoice {
    bool valid{false};
    llm::LlmParallelConfig parallel;
    llm::LlmTaskGraphBuildResult build_result;
    double makespan{std::numeric_limits<double>::infinity()};
    std::size_t devices_used{0};
    std::size_t task_count{0};
};

struct LlmParallelSearchPolicy {
    int exact_devices{0};
    bool model_parallel_only{false};
    std::string failure_label;
    std::string diagnostic_label;
};

bool llm_auto_choice_better(const LlmAutoParallelChoice& candidate,
                            const LlmAutoParallelChoice& incumbent) {
    if (!candidate.valid || !std::isfinite(candidate.makespan)) {
        return false;
    }
    if (!incumbent.valid || !std::isfinite(incumbent.makespan)) {
        return true;
    }
    const double epsilon = std::max(1e-18, std::abs(incumbent.makespan) * 1e-12);
    if (candidate.makespan + epsilon < incumbent.makespan) {
        return true;
    }
    if (incumbent.makespan + epsilon < candidate.makespan) {
        return false;
    }
    if (candidate.devices_used != incumbent.devices_used) {
        return candidate.devices_used < incumbent.devices_used;
    }
    if (candidate.task_count != incumbent.task_count) {
        return candidate.task_count < incumbent.task_count;
    }
    if (candidate.parallel.dp != incumbent.parallel.dp) {
        return candidate.parallel.dp < incumbent.parallel.dp;
    }
    if (candidate.parallel.pp != incumbent.parallel.pp) {
        return candidate.parallel.pp < incumbent.parallel.pp;
    }
    return candidate.parallel.tp < incumbent.parallel.tp;
}

LlmAutoParallelChoice choose_llm_parallel_by_policy(
    const hardware_topology::HardwareTopology& topology,
    const llm::LlmModelConfig& model,
    const llm::LlmRequestConfig& request,
    const Options& options,
    const LlmParallelSearchPolicy& policy) {
    if (options.llm_cp != 1) {
        throw std::runtime_error("LLM " + policy.failure_label + " currently requires --llm-cp 1");
    }

    const auto devices = select_llm_candidate_devices(topology);
    if (devices.empty()) {
        throw std::runtime_error("Topology has no compute devices for LLM " + policy.failure_label);
    }
    const int max_devices = static_cast<int>(devices.size());
    int exact_devices = policy.exact_devices;
    if (exact_devices < 0) {
        exact_devices = max_devices;
    }
    if (exact_devices > max_devices) {
        throw std::runtime_error("LLM " + policy.failure_label + " requires " +
                                 std::to_string(exact_devices) + " devices, but topology has only " +
                                 std::to_string(max_devices));
    }

    LlmAutoParallelChoice best;
    std::vector<std::string> rejected;

    for (int dp = 1; dp <= max_devices; ++dp) {
        if (policy.model_parallel_only && dp != 1) {
            continue;
        }
        const int max_pp = std::min(max_devices / dp, std::max(1, model.num_layers));
        for (int pp = 1; pp <= max_pp; ++pp) {
            for (int tp = 1; tp <= max_devices / (dp * pp); ++tp) {
                const int devices_used = tp * pp * dp;
                if (exact_devices > 0 && devices_used != exact_devices) {
                    continue;
                }

                llm::LlmParallelConfig parallel;
                parallel.tp = tp;
                parallel.pp = pp;
                parallel.cp = 1;
                parallel.dp = dp;

                for (const auto& stage_map : llm_candidate_stage_maps(model, request, parallel)) {
                    parallel.stage_for_layer = stage_map;
                    LlmAutoParallelChoice candidate;
                    candidate.parallel = parallel;
                    candidate.devices_used = static_cast<std::size_t>(devices_used);
                    try {
                        candidate.build_result = llm::build_task_graph(model, request, parallel, topology);
                        const auto& graph = candidate.build_result.graph;
                        auto pinned_mapping_plan = build_fully_pinned_mapping_plan(graph, topology);
                        mapping::MappingPlan mapping_plan;
                        if (pinned_mapping_plan.has_value()) {
                            mapping_plan = std::move(*pinned_mapping_plan);
                        } else {
                            auto mapper = build_mapper_for_graph(graph, options);
                            mapping_plan = mapper->map(graph, topology);
                        }
                        candidate.makespan = estimate_makespan_seconds(graph, mapping_plan, topology);
                        candidate.task_count = graph.topological_order().size();
                        candidate.valid = std::isfinite(candidate.makespan);
                    } catch (const std::exception& ex) {
                        if (rejected.size() < 8) {
                            rejected.push_back(llm_parallel_plan_label(parallel, "") + ": " + ex.what());
                        }
                    }
                    if (llm_auto_choice_better(candidate, best)) {
                        best = std::move(candidate);
                    }
                }
            }
        }
    }

    if (!best.valid) {
        std::string message = "LLM " + policy.failure_label +
                              " found no feasible TP/PP/DP candidate";
        if (!rejected.empty()) {
            message += "; examples: ";
            for (std::size_t i = 0; i < rejected.size(); ++i) {
                if (i != 0) {
                    message += " | ";
                }
                message += rejected[i];
            }
        }
        throw std::runtime_error(message);
    }
    best.build_result.diagnostics.push_back(
        policy.diagnostic_label + " selected " + llm_parallel_plan_label(best.parallel, "") +
        " with estimated makespan " + std::to_string(best.makespan) + " s.");
    return best;
}

LlmAutoParallelChoice choose_llm_auto_parallel(
    const hardware_topology::HardwareTopology& topology,
    const llm::LlmModelConfig& model,
    const llm::LlmRequestConfig& request,
    const Options& options) {
    return choose_llm_parallel_by_policy(
        topology,
        model,
        request,
        options,
        LlmParallelSearchPolicy{0, false, "auto parallel search", "Alpa-lite auto parallel"});
}

LlmAutoParallelChoice choose_llm_rank_parallel(
    const hardware_topology::HardwareTopology& topology,
    const llm::LlmModelConfig& model,
    const llm::LlmRequestConfig& request,
    const Options& options) {
    if (options.llm_cp != 1) {
        throw std::runtime_error("LLM rank-parallel planning currently requires --llm-cp 1");
    }

    const auto devices = select_llm_candidate_devices(topology);
    if (devices.empty()) {
        throw std::runtime_error("Topology has no compute devices for LLM rank-parallel planning");
    }
    const int device_count = static_cast<int>(devices.size());
    std::vector<std::string> rejected;

    auto tp_shape_feasible = [&](int tp) {
        if (tp <= 0) {
            return false;
        }
        if (model.hidden_size % tp != 0 || model.num_attention_heads % tp != 0 ||
            model.intermediate_size % tp != 0) {
            return false;
        }
        if (model.num_kv_heads % tp != 0 && model.num_kv_heads >= tp) {
            return false;
        }
        return true;
    };

    auto try_rank_plan = [&](int tp, int pp, int dp, const std::string& note)
        -> std::optional<LlmAutoParallelChoice> {
        llm::LlmParallelConfig parallel;
        parallel.tp = tp;
        parallel.pp = pp;
        parallel.cp = 1;
        parallel.dp = dp;

        auto stage_maps = llm_candidate_stage_maps(model, request, parallel);
        std::reverse(stage_maps.begin(), stage_maps.end());
        for (const auto& stage_map : stage_maps) {
            parallel.stage_for_layer = stage_map;
            try {
                LlmAutoParallelChoice choice;
                choice.parallel = parallel;
                choice.devices_used = static_cast<std::size_t>(device_count);
                choice.build_result = llm::build_task_graph(model, request, parallel, topology);
                choice.task_count = choice.build_result.graph.topological_order().size();
                choice.valid = true;
                choice.build_result.diagnostics.push_back(
                    "Topology-rank LLM parallel selected " + llm_parallel_plan_label(choice.parallel, "") +
                    " using all " + std::to_string(device_count) +
                    " GPU devices without per-candidate makespan search" + note + ".");
                return choice;
            } catch (const std::exception& ex) {
                if (rejected.size() < 8) {
                    rejected.push_back(llm_parallel_plan_label(parallel, "") + ": " + ex.what());
                }
            }
        }
        return std::nullopt;
    };

    for (int tp = device_count; tp >= 1; --tp) {
        if (device_count % tp != 0 || !tp_shape_feasible(tp)) {
            continue;
        }
        const int pp = device_count / tp;
        if (pp > std::max(1, model.num_layers)) {
            continue;
        }
        if (auto choice = try_rank_plan(tp, pp, 1, "")) {
            return *choice;
        }
    }

    for (int tp = device_count; tp >= 1; --tp) {
        if (device_count % tp != 0 || !tp_shape_feasible(tp)) {
            continue;
        }
        const int remaining = device_count / tp;
        const int max_pp = std::min(remaining, std::max(1, model.num_layers));
        for (int pp = max_pp; pp >= 1; --pp) {
            if (remaining % pp != 0) {
                continue;
            }
            const int dp = remaining / pp;
            if (dp <= 1) {
                continue;
            }
            if (auto choice = try_rank_plan(tp, pp, dp, " with DP fallback")) {
                return *choice;
            }
        }
    }

    std::string message = "LLM rank-parallel planning found no feasible TP/PP/DP candidate using " +
                          std::to_string(device_count) + " GPU devices";
    if (!rejected.empty()) {
        message += "; examples: ";
        for (std::size_t i = 0; i < rejected.size(); ++i) {
            if (i != 0) {
                message += " | ";
            }
            message += rejected[i];
        }
    }
    throw std::runtime_error(message);
}

llm::LlmRequestConfig make_llm_request_config(const Options& options) {
    llm::LlmRequestConfig request;
    request.prefill_batch_size = options.llm_prefill_batch_size;
    request.prompt_len = options.llm_prompt_len;
    request.decode_batch_size = options.llm_decode_batch_size;
    request.decode_steps = options.llm_decode_steps;
    request.avg_context_len = options.llm_avg_context_len;
    return request;
}

llm::LlmParallelConfig make_llm_parallel_config(const Options& options) {
    llm::LlmParallelConfig parallel;
    parallel.tp = options.llm_tp;
    parallel.pp = options.llm_pp;
    parallel.cp = options.llm_cp;
    parallel.dp = options.llm_dp;
    return parallel;
}

struct LlmPlanningResult {
    llm::LlmTaskGraphBuildResult build_result;
    std::string selected_parallel_label;
};

LlmPlanningResult plan_llm_graph_for_scheduling(
    const hardware_topology::HardwareTopology& topology,
    const Options& options) {
    if (options.llm_config_path.empty()) {
        throw std::runtime_error("--parallel=llm requires --llm-config PATH");
    }

    // LLM parallelism changes the task DAG itself: TP/PP/DP alter tasks, edges,
    // collective groups, and hard device pins. Materialize each candidate inside
    // the scheduling search instead of treating it as a fixed external workload.
    const auto request = make_llm_request_config(options);
    auto parallel = make_llm_parallel_config(options);
    llm::LlmModelConfig model;
    std::string model_error;
    if (!llm::load_model_config_from_json(options.llm_config_path, model, &model_error, options.llm_size)) {
        throw std::runtime_error(model_error);
    }

    if (options.llm_auto_parallel) {
        auto choice = choose_llm_auto_parallel(topology, model, request, options);
        parallel = choice.parallel;
        return {std::move(choice.build_result), llm_parallel_plan_label(parallel, "auto")};
    }

    if (options.llm_rank_parallel) {
        auto choice = choose_llm_rank_parallel(topology, model, request, options);
        parallel = choice.parallel;
        return {std::move(choice.build_result), llm_parallel_plan_label(parallel, "ranks")};
    }

    auto build_result = llm::build_task_graph(model, request, parallel, topology);
    const auto selected_parallel_label = llm_parallel_plan_label(build_result.parallel, "");
    return {std::move(build_result), selected_parallel_label};
}

void write_workload_task_graph_json(const std::string& path,
                                    const mapping::TaskGraph& graph,
                                    const std::string& name) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open workload taskgraph dump: " + path);
    }

    auto write_string_set_without_device_tags = [&](const auto& values) {
        std::vector<std::string> sorted;
        sorted.reserve(values.size());
        for (const auto& value : values) {
            if (value.rfind("device:", 0) == 0) {
                continue;
            }
            sorted.push_back(value);
        }
        std::sort(sorted.begin(), sorted.end());
        out << "[";
        for (std::size_t i = 0; i < sorted.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            taskflow::json::write_string(out, sorted[i]);
        }
        out << "]";
    };

    out << "{\n";
    out << "  \"kind\": \"taskgraph_v1\",\n";
    out << "  \"name\": ";
    taskflow::json::write_string(out, name.empty() ? "workload" : name);
    out << ",\n";
    out << "  \"tasks\": [\n";
    const auto& tasks = graph.topological_order();
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        out << "    {\"name\": ";
        taskflow::json::write_string(out, task.name);
        out << ", \"subtype\": ";
        taskflow::json::write_string(out, task.subtype.empty() ? task.type : task.subtype);
        out << ", \"compute_flops\": ";
        taskflow::json::write_double(out, task.compute_flops);
        out << ", \"memory_bytes\": ";
        taskflow::json::write_double(out, task.memory_bytes);
        if (!task.features.empty()) {
            out << ", \"features\": ";
            write_string_set_without_device_tags(task.features);
        }
        bool has_exported_tags = false;
        for (const auto& tag : task.tags) {
            if (tag.rfind("device:", 0) != 0) {
                has_exported_tags = true;
                break;
            }
        }
        if (has_exported_tags) {
            out << ", \"tags\": ";
            write_string_set_without_device_tags(task.tags);
        }
        out << "}" << (i + 1 == tasks.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"edges\": [\n";
    bool first = true;
    for (const auto& task : tasks) {
        for (const auto& edge : graph.successors(task.name)) {
            if (!first) {
                out << ",\n";
            }
            first = false;
            out << "    {\"src\": ";
            taskflow::json::write_string(out, edge.src);
            out << ", \"dst\": ";
            taskflow::json::write_string(out, edge.dst);
            out << ", \"tensor_bytes\": ";
            taskflow::json::write_double(out, edge.tensor_bytes);
            if (!edge.tensor_id.empty()) {
                out << ", \"tensor_id\": ";
                taskflow::json::write_string(out, edge.tensor_id);
            }
            if (!edge.comm_kind.empty()) {
                out << ", \"comm_kind\": ";
                taskflow::json::write_string(out, edge.comm_kind);
            }
            if (!edge.access_pattern.empty()) {
                out << ", \"access_pattern\": ";
                taskflow::json::write_string(out, edge.access_pattern);
            }
            if (edge.comm_participants > 0) {
                out << ", \"comm_participants\": " << edge.comm_participants;
            }
            if (!edge.comm_group.empty()) {
                out << ", \"comm_group\": ";
                taskflow::json::write_string(out, edge.comm_group);
            }
            if (!edge.dtype.empty()) {
                out << ", \"dtype\": ";
                taskflow::json::write_string(out, edge.dtype);
            }
            out << "}";
        }
    }
    out << "\n  ]\n";
    out << "}\n";
}

std::uint64_t safe_edge_bytes(double bytes) {
    if (bytes <= 0.0) {
        return 0;
    }
    const long double rounded = std::llround(bytes);
    if (rounded <= 0.0L) {
        return 0;
    }
    const auto max_u64 = static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
    if (rounded >= max_u64) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(rounded);
}

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) {
    if (std::numeric_limits<std::uint64_t>::max() - a < b) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return a + b;
}

struct CollectiveByteSummary {
    std::string kind;
    std::unordered_map<std::string, std::uint64_t> bytes_by_tensor;
    std::uint64_t anonymous_total{0};
    std::unordered_set<std::string> participant_devices;
};

std::string collective_summary_key(const mapping::TaskEdge& edge, const std::string& kind) {
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

void add_collective_summary_edge(std::unordered_map<std::string, CollectiveByteSummary>& summaries,
                                 const mapping::TaskEdge& edge,
                                 const std::string& kind,
                                 const std::string& src_device,
                                 const std::string& dst_device) {
    auto& summary = summaries[collective_summary_key(edge, kind)];
    summary.kind = kind;
    const auto bytes = safe_edge_bytes(edge.tensor_bytes);
    if (edge.tensor_id.empty()) {
        summary.anonymous_total = saturating_add(summary.anonymous_total, bytes);
    } else {
        auto& current = summary.bytes_by_tensor[edge.tensor_id];
        current = std::max(current, bytes);
    }
    if (!src_device.empty()) {
        summary.participant_devices.insert(src_device);
    }
    if (!dst_device.empty()) {
        summary.participant_devices.insert(dst_device);
    }
}

std::uint64_t collective_summary_payload_bytes(const CollectiveByteSummary& summary) {
    std::uint64_t total = summary.anonymous_total;
    for (const auto& entry : summary.bytes_by_tensor) {
        total = saturating_add(total, entry.second);
    }
    return total;
}

std::vector<NamedCount> sort_named_count(const std::unordered_map<std::string, std::size_t>& counts) {
    std::vector<NamedCount> out;
    out.reserve(counts.size());
    for (const auto& entry : counts) {
        out.push_back({entry.first, entry.second});
    }
    std::sort(out.begin(),
              out.end(),
              [](const NamedCount& a, const NamedCount& b) {
                  if (a.count != b.count) {
                      return a.count > b.count;
                  }
                  return a.name < b.name;
              });
    return out;
}

std::vector<NamedBytes> sort_named_bytes(const std::unordered_map<std::string, std::uint64_t>& bytes) {
    std::vector<NamedBytes> out;
    out.reserve(bytes.size());
    for (const auto& entry : bytes) {
        out.push_back({entry.first, entry.second});
    }
    std::sort(out.begin(),
              out.end(),
              [](const NamedBytes& a, const NamedBytes& b) {
                  if (a.bytes != b.bytes) {
                      return a.bytes > b.bytes;
                  }
                  return a.name < b.name;
              });
    return out;
}

}  // namespace

RunResult write_taskflow(const hardware_topology::HardwareTopology& topology,
                         const workload::Workload& workload,
                         const std::string& taskflow_path,
                         const Options& options) {
    const bool llm_mode = !options.llm_config_path.empty();
    mapping::TaskGraph working_graph;
    bool matrix_parallel_applied = false;
    std::string selected_parallel_label = "none";

    if (llm_mode) {
        auto llm_plan = plan_llm_graph_for_scheduling(topology, options);
        if (!options.llm_dump_taskgraph_path.empty()) {
            llm::write_task_graph_json(options.llm_dump_taskgraph_path,
                                       llm_plan.build_result.graph,
                                       llm_plan.build_result.model,
                                       llm_plan.build_result.request,
                                       llm_plan.build_result.parallel,
                                       false);
        }
        selected_parallel_label = std::move(llm_plan.selected_parallel_label);
        working_graph = std::move(llm_plan.build_result.graph);
    } else {
        working_graph = workload.to_task_graph(topology);
        const auto mode = parse_parallel_mode(options.parallel);
        if (!mode.has_value()) {
            throw std::runtime_error("Unknown parallel mode: " + options.parallel +
                                     " (expected: none|matrix|matrix_parallel|auto|llm)");
        }

        bool expanded_any = false;
        if (options.workload_rank_parallel) {
            working_graph = expand_workload_rank_parallel(working_graph, topology, &expanded_any);
            if (!expanded_any) {
                throw std::runtime_error("--workload-rank-parallel found no GPU-compatible tasks to split");
            }
            selected_parallel_label = "workload_rank_parallel";
        } else if (*mode == ParallelMode::MATRIX || *mode == ParallelMode::AUTO) {
            const bool should_expand_matrix =
                *mode == ParallelMode::MATRIX || analyze_workload_parallel_shape(workload).has_matrix_candidates;
            if (should_expand_matrix) {
                working_graph = expand_matrix_parallel(working_graph, workload, topology, ParallelMode::MATRIX, &expanded_any);
                matrix_parallel_applied = expanded_any;
                if (matrix_parallel_applied) {
                    selected_parallel_label = "matrix_parallel";
                }
            }
        }
    }

    mapping::TaskGraph annotated_storage;
    const mapping::TaskGraph* annotated_graph = &working_graph;
    if (!llm_mode) {
        annotated_storage = annotate_comm_bytes(working_graph, workload, topology);
        annotated_graph = &annotated_storage;
    }
    if (!llm_mode && !options.workload_dump_taskgraph_path.empty()) {
        write_workload_task_graph_json(options.workload_dump_taskgraph_path,
                                       *annotated_graph,
                                       workload.name());
    }
    auto pinned_mapping_plan = build_fully_pinned_mapping_plan(*annotated_graph, topology);
    mapping::MappingPlan mapping_plan;
    if (pinned_mapping_plan.has_value()) {
        mapping_plan = std::move(*pinned_mapping_plan);
    } else {
        auto mapper = build_mapper_for_graph(*annotated_graph, options);
        mapping_plan = mapper->map(*annotated_graph, topology);
    }
    const double makespan = estimate_makespan_seconds(*annotated_graph, mapping_plan, topology);

    std::unordered_map<std::string, std::size_t> subtype_counts;
    std::unordered_map<std::string, std::size_t> device_counts;
    std::unordered_map<std::string, std::uint64_t> comm_kind_bytes;
    std::unordered_map<std::string, std::size_t> longest_depth_to_task;
    std::size_t edge_count = 0;
    std::size_t source_count = 0;
    std::size_t sink_count = 0;
    std::size_t dag_depth = 0;
    std::size_t cross_device_edge_count = 0;
    std::uint64_t total_edge_bytes = 0;
    std::uint64_t cross_device_edge_bytes = 0;
    std::unordered_map<std::string, CollectiveByteSummary> collective_summaries;

    const auto& topo = annotated_graph->topological_order();

    for (const auto& task : topo) {
        const std::string raw_subtype = task.subtype.empty() ? (task.type.empty() ? "unknown" : task.type) : task.subtype;
        const std::string subtype = canonical_task_subtype(raw_subtype);
        subtype_counts[subtype] += 1;

        const auto assigned = mapping_plan.assignments.find(task.name);
        const std::string device = assigned == mapping_plan.assignments.end() ? "?" : assigned->second;
        device_counts[device] += 1;

        std::size_t max_pred_depth = 0;
        const auto& deps = annotated_graph->dependencies(task.name);
        if (deps.empty()) {
            source_count += 1;
        }
        for (const auto& dep : deps) {
            const auto it = longest_depth_to_task.find(dep.src);
            if (it != longest_depth_to_task.end()) {
                max_pred_depth = std::max(max_pred_depth, it->second);
            }
        }
        const std::size_t my_depth = max_pred_depth + 1;
        longest_depth_to_task[task.name] = my_depth;
        dag_depth = std::max(dag_depth, my_depth);

        const auto& succs = annotated_graph->successors(task.name);
        if (succs.empty()) {
            sink_count += 1;
        }
        for (const auto& edge : succs) {
            edge_count += 1;
            const std::string kind = edge.comm_kind.empty() ? "p2p" : canonical_comm_kind(edge.comm_kind);

            const auto src_it = mapping_plan.assignments.find(edge.src);
            const auto dst_it = mapping_plan.assignments.find(edge.dst);
            const bool cross_device =
                (src_it != mapping_plan.assignments.end() && dst_it != mapping_plan.assignments.end() &&
                 src_it->second != dst_it->second);
            const bool collective = is_collective_kind(kind);
            if (collective) {
                add_collective_summary_edge(collective_summaries,
                                            edge,
                                            kind,
                                            src_it == mapping_plan.assignments.end() ? "" : src_it->second,
                                            dst_it == mapping_plan.assignments.end() ? "" : dst_it->second);
                continue;
            }

            const std::uint64_t bytes = cross_device ? safe_edge_bytes(edge.tensor_bytes) : 0;
            total_edge_bytes = saturating_add(total_edge_bytes, bytes);

            if (bytes > 0) {
                auto& kind_total = comm_kind_bytes["p2p"];
                kind_total = saturating_add(kind_total, bytes);
            }

            if (cross_device) {
                cross_device_edge_count += 1;
                cross_device_edge_bytes = saturating_add(cross_device_edge_bytes, bytes);
            }
        }
    }

    for (const auto& entry : collective_summaries) {
        const auto& summary = entry.second;
        const auto bytes = collective_summary_payload_bytes(summary);
        total_edge_bytes = saturating_add(total_edge_bytes, bytes);
        auto& kind_total = comm_kind_bytes[summary.kind.empty() ? "collective" : summary.kind];
        kind_total = saturating_add(kind_total, bytes);
        if (summary.participant_devices.size() > 1) {
            cross_device_edge_count += 1;
            cross_device_edge_bytes = saturating_add(cross_device_edge_bytes, bytes);
        }
    }

    const std::string output_format = options.output_format.empty() ? "json" : options.output_format;
    const bool emit_json = output_format == "json" || output_format == "both";
    const bool emit_chakra_et = output_format == "chakra-et" || output_format == "both";
    if (!emit_json && !emit_chakra_et) {
        throw std::runtime_error("Unknown output format: " + output_format + " (expected json|chakra-et|both)");
    }
    const std::string et_prefix = emit_chakra_et
                                      ? (options.et_prefix.empty() ? taskflow_path : options.et_prefix)
                                      : std::string{};
    taskflow::TaskflowWriter::write_outputs(taskflow_path,
                                            et_prefix,
                                            options.time_unit,
                                            *annotated_graph,
                                            mapping_plan,
                                            topology,
                                            emit_json,
                                            emit_chakra_et);
    return RunResult{
        makespan,
        selected_parallel_label,
        topo.size(),
        edge_count,
        source_count,
        sink_count,
        dag_depth,
        total_edge_bytes,
        cross_device_edge_count,
        cross_device_edge_bytes,
        sort_named_count(subtype_counts),
        sort_named_count(device_counts),
        sort_named_bytes(comm_kind_bytes),
    };
}

}  // namespace mapper
