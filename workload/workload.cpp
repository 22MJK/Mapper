#include "workload/workload.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mapping/operator_catalog.h"

namespace workload {
namespace {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::FP16:
        case DType::BF16:
            return 2;
        case DType::FP32:
        case DType::INT32:
            return 4;
        case DType::FP64:
        case DType::INT64:
            return 8;
        case DType::INT8:
        case DType::UINT8:
            return 1;
    }
    return 4;
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::FP16:
            return "fp16";
        case DType::BF16:
            return "bf16";
        case DType::FP32:
            return "fp32";
        case DType::FP64:
            return "fp64";
        case DType::INT8:
            return "int8";
        case DType::UINT8:
            return "uint8";
        case DType::INT32:
            return "int32";
        case DType::INT64:
            return "int64";
    }
    return "fp32";
}

std::uint64_t tensor_bytes(const Tensor& tensor) {
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
    for (const auto dim : tensor.shape) {
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

const char* storage_format_name(StorageFormat storage) {
    switch (storage) {
        case StorageFormat::DENSE:
            return "dense";
        case StorageFormat::CSR:
            return "csr";
        case StorageFormat::CSC:
            return "csc";
        case StorageFormat::COO_IMPORT:
            return "coo";
        case StorageFormat::BSR:
            return "bsr";
        case StorageFormat::BLOCK_SPARSE:
            return "block_sparse";
    }
    return "dense";
}

bool is_sparse_storage(StorageFormat storage) {
    return storage == StorageFormat::CSR || storage == StorageFormat::CSC ||
           storage == StorageFormat::COO_IMPORT || storage == StorageFormat::BSR ||
           storage == StorageFormat::BLOCK_SPARSE;
}

std::optional<std::uint64_t> infer_sparse_nnz_from_size(const Tensor& tensor) {
    if (!is_sparse_storage(tensor.storage_format) || tensor.size_bytes == 0) {
        return std::nullopt;
    }

    const auto dtype_bytes = static_cast<std::uint64_t>(dtype_size(tensor.dtype));
    constexpr std::uint64_t kIndexBytes = 4;
    std::uint64_t overhead = 0;
    std::uint64_t bytes_per_nnz = dtype_bytes;

    switch (tensor.storage_format) {
        case StorageFormat::CSR:
            if (tensor.shape.empty() || tensor.shape[0] < 0) {
                return std::nullopt;
            }
            overhead = (static_cast<std::uint64_t>(tensor.shape[0]) + 1) * kIndexBytes;
            bytes_per_nnz += kIndexBytes;
            break;
        case StorageFormat::CSC:
            if (tensor.shape.size() < 2 || tensor.shape[1] < 0) {
                return std::nullopt;
            }
            overhead = (static_cast<std::uint64_t>(tensor.shape[1]) + 1) * kIndexBytes;
            bytes_per_nnz += kIndexBytes;
            break;
        case StorageFormat::COO_IMPORT:
            bytes_per_nnz += 2 * kIndexBytes;
            break;
        case StorageFormat::BSR:
        case StorageFormat::BLOCK_SPARSE:
            bytes_per_nnz += kIndexBytes;
            break;
        case StorageFormat::DENSE:
            return std::nullopt;
    }

    if (tensor.size_bytes >= overhead && bytes_per_nnz > 0) {
        const auto payload = tensor.size_bytes - overhead;
        if (payload > 0 && payload % bytes_per_nnz == 0) {
            return payload / bytes_per_nnz;
        }
    }
    if (dtype_bytes > 0 && tensor.size_bytes % dtype_bytes == 0) {
        return tensor.size_bytes / dtype_bytes;
    }
    return std::nullopt;
}

std::optional<std::uint64_t> tensor_nonzero_elements(const Tensor& tensor) {
    if (!is_sparse_storage(tensor.storage_format)) {
        return std::nullopt;
    }
    if (tensor.nonzero_elements.has_value()) {
        return tensor.nonzero_elements;
    }
    if (tensor.num_elements.has_value()) {
        return tensor.num_elements;
    }
    return infer_sparse_nnz_from_size(tensor);
}

const char* access_kind_name(AccessKind access) {
    switch (access) {
        case AccessKind::DENSE:
            return "dense";
        case AccessKind::SPARSE_CSR:
            return "sparse_csr";
        case AccessKind::SPARSE_CSC:
            return "sparse_csc";
        case AccessKind::SPARSE_COO:
            return "sparse_coo";
        case AccessKind::SPARSE_BLOCK:
            return "sparse_block";
        case AccessKind::ROW_WISE:
            return "row-wise";
        case AccessKind::COL_WISE:
            return "col-wise";
    }
    return "dense";
}

int access_kind_rank(AccessKind access) {
    switch (access) {
        case AccessKind::SPARSE_CSR:
        case AccessKind::SPARSE_CSC:
        case AccessKind::SPARSE_COO:
        case AccessKind::SPARSE_BLOCK:
            return 3;
        case AccessKind::ROW_WISE:
        case AccessKind::COL_WISE:
            return 2;
        case AccessKind::DENSE:
            return 1;
    }
    return 0;
}

AccessKind effective_input_access(const TensorUse& input, const Tensor& tensor) {
    return input.access_explicit ? input.access : tensor.access_pattern;
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

std::size_t collective_participants(const std::unordered_set<int>& predecessors,
                                    const std::unordered_set<int>& successors,
                                    const std::string& comm_kind) {
    const std::string kind = canonical_comm_kind(comm_kind);
    if (kind == "broadcast" || kind == "allgather") {
        return successors.size() + 1;
    }
    if (kind == "reducescatter" || kind == "allreduce") {
        return predecessors.size() + 1;
    }
    if (kind == "alltoall") {
        std::unordered_set<int> participants = predecessors;
        participants.insert(successors.begin(), successors.end());
        return participants.size();
    }
    return 0;
}

bool is_supported_collective_hint(const std::string& comm_kind) {
    const std::string kind = canonical_comm_kind(comm_kind);
    return kind == "allreduce" || kind == "allgather" || kind == "reducescatter" || kind == "broadcast" ||
           kind == "alltoall";
}

const std::unordered_set<int>& empty_int_set() {
    static const std::unordered_set<int> kEmpty;
    return kEmpty;
}

std::vector<int> sorted_ids(const std::unordered_set<int>& ids) {
    std::vector<int> out(ids.begin(), ids.end());
    std::sort(out.begin(), out.end());
    return out;
}

std::string alltoall_group_key(const std::unordered_set<int>& sources,
                               const std::unordered_set<int>& destinations) {
    const auto sorted_sources = sorted_ids(sources);
    const auto sorted_destinations = sorted_ids(destinations);
    std::string out = "src:";
    for (std::size_t i = 0; i < sorted_sources.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(sorted_sources[i]);
    }
    out += "|dst:";
    for (std::size_t i = 0; i < sorted_destinations.size(); ++i) {
        if (i != 0) {
            out += ",";
        }
        out += std::to_string(sorted_destinations[i]);
    }
    return out;
}

bool is_complete_alltoall_relation(
    const std::unordered_set<int>& sources,
    const std::unordered_set<int>& destinations,
    const std::unordered_map<int, std::unordered_set<int>>& successors) {
    if (sources.size() < 2 || destinations.size() < 2) {
        return false;
    }
    for (const int source : sources) {
        const auto succ_it = successors.find(source);
        if (succ_it == successors.end()) {
            return false;
        }
        for (const int destination : destinations) {
            if (source == destination) {
                continue;
            }
            if (succ_it->second.find(destination) == succ_it->second.end()) {
                return false;
            }
        }
    }
    return true;
}

void add_operator_features(mapping::Task& task) {
    const std::string op = mapping::canonical_operator_profile_name(task.subtype.empty() ? task.type : task.subtype);
    const std::string access = task.access_pattern;
    if (op == "gemm" || op == "mv" || op == "trsm" || op == "trsv" || op == "potrf" || op == "getrf" ||
        op == "geqrf" || op == "tstrf" || op == "gessm") {
        task.features.insert("dense_linear_algebra");
    }
    if (op == "spmv" || op == "sptrsv" || op == "spgemm" || op == "ssssm") {
        task.features.insert("sparse_linear_algebra");
        task.features.insert("irregular_access");
    }
    if (op == "trsv" || op == "sptrsv" || op == "trsm" || op == "ssssm") {
        task.features.insert("triangular_dependency");
        task.features.insert("data_dependency");
    }
    if (op == "potrf" || op == "getrf" || op == "geqrf" || op == "tstrf" || op == "gessm") {
        task.features.insert("factorization");
    }
    if (op == "dot" || op == "nrm2" || op == "allreduce") {
        task.features.insert("reduction");
    }
    if (op == "allreduce") {
        task.features.insert("communication");
    }
    if (op == "axpy" || op == "copy" || op == "scal") {
        task.features.insert("elementwise");
    }
    if (op == "transpose") {
        task.features.insert("matrix_transform");
        task.features.insert("strided_access");
    }
    if (op == "symbolic" || op == "order" || op == "etree" || op == "colcount" || op == "postorder" ||
        op == "supernode_partition") {
        task.features.insert("symbolic");
        task.features.insert("control_flow");
        task.features.insert("data_dependency");
        task.features.insert("latency_sensitive");
        task.features.insert("gpu_unsupported");
        task.tags.insert("gpu_unsupported");
    }
    if (access == "dense" || access == "contiguous") {
        task.features.insert("streaming_memory");
        task.features.insert("coalesced_access");
    } else if (access.find("sparse") != std::string::npos) {
        task.features.insert("irregular_access");
    } else if (access.find("row") != std::string::npos || access.find("col") != std::string::npos ||
               access.find("stride") != std::string::npos) {
        task.features.insert("strided_access");
    }
    if (task.compute_flops >= 1e9) {
        task.features.insert("massive_parallelism");
        task.features.insert("compute_intensive");
    } else if (task.compute_flops >= 1e6 || task.memory_bytes >= 1024.0 * 1024.0) {
        task.features.insert("high_parallelism");
    } else {
        task.features.insert("small_working_set");
    }
    if (task.memory_bytes > 0.0 && task.compute_flops / task.memory_bytes < 1.0) {
        task.features.insert("memory_bound");
    }
}

}  // namespace

Workload::Workload(std::string name,
                   std::vector<Task> tasks,
                   std::vector<Tensor> tensors,
                   std::vector<DeviceGroup> device_groups,
                   std::vector<std::string> iteration_inputs,
                   std::vector<std::string> iteration_outputs,
                   std::optional<mapping::TaskGraph> explicit_task_graph)
    : name_(std::move(name)),
      tasks_(std::move(tasks)),
      tensors_(std::move(tensors)),
      device_groups_(std::move(device_groups)),
      iteration_inputs_(std::move(iteration_inputs)),
      iteration_outputs_(std::move(iteration_outputs)),
      explicit_task_graph_(std::move(explicit_task_graph)) {}

mapping::TaskGraph Workload::to_task_graph(const hardware_topology::HardwareTopology& topology) const {
    (void)topology;
    if (explicit_task_graph_.has_value()) {
        return *explicit_task_graph_;
    }

    mapping::TaskGraph graph;
    std::unordered_map<int, std::string> id_to_name;
    id_to_name.reserve(tasks_.size());

    std::unordered_map<std::string, Tensor> tensor_map;
    tensor_map.reserve(tensors_.size());
    for (const auto& tensor : tensors_) {
        tensor_map.emplace(tensor.id, tensor);
    }

    for (const auto& task : tasks_) {
        mapping::Task mapped;
        mapped.name = task.name;
        mapped.type = "compute";
        mapped.subtype = mapping::require_operator_profile_name(task.op);
        mapped.compute_flops = task.compute_flops;
        mapped.comm_bytes = 0.0;
        mapped.features = task.features;
        AccessKind task_access = AccessKind::DENSE;
        for (const auto& input : task.inputs) {
            const auto tensor_it = tensor_map.find(input.tensor_id);
            if (tensor_it == tensor_map.end()) {
                throw std::runtime_error("Input tensor not found: " + input.tensor_id);
            }
            const auto& tensor = tensor_it->second;
            mapped.input_data.push_back(mapping::TaskInput{input.tensor_id,
                                                           static_cast<double>(tensor_bytes(tensor)),
                                                           tensor.shape,
                                                           storage_format_name(tensor.storage_format),
                                                           tensor_nonzero_elements(tensor),
                                                           dtype_name(tensor.dtype)});
            const auto input_access = effective_input_access(input, tensor_it->second);
            if (access_kind_rank(input_access) > access_kind_rank(task_access)) {
                task_access = input_access;
            }
        }
        mapped.access_pattern = access_kind_name(task_access);

        double estimated_memory_bytes = task.memory_bytes;
        if (estimated_memory_bytes <= 0.0) {
            for (const auto& input : task.inputs) {
                const auto tensor_it = tensor_map.find(input.tensor_id);
                if (tensor_it == tensor_map.end()) {
                    throw std::runtime_error("Input tensor not found: " + input.tensor_id);
                }
                estimated_memory_bytes += static_cast<double>(tensor_bytes(tensor_it->second));
            }
            for (const auto& output_id : task.outputs) {
                const auto tensor_it = tensor_map.find(output_id);
                if (tensor_it == tensor_map.end()) {
                    continue;
                }
                estimated_memory_bytes += static_cast<double>(tensor_bytes(tensor_it->second));
            }
        }
        mapped.memory_bytes = estimated_memory_bytes;
        add_operator_features(mapped);

        graph.add_task(std::move(mapped));
        id_to_name.emplace(task.id, task.name);
    }

    std::unordered_map<int, std::unordered_set<int>> task_successors;
    std::unordered_map<int, std::unordered_set<int>> task_predecessors;
    for (const auto& task : tasks_) {
        for (const auto& input : task.inputs) {
            const auto tensor_it = tensor_map.find(input.tensor_id);
            if (tensor_it == tensor_map.end() || !tensor_it->second.producer_task.has_value()) {
                continue;
            }
            const int producer = *tensor_it->second.producer_task;
            if (producer == task.id) {
                continue;
            }
            task_successors[producer].insert(task.id);
            task_predecessors[task.id].insert(producer);
        }
    }

    for (const auto& task : tasks_) {
        const auto dst_it = id_to_name.find(task.id);
        if (dst_it == id_to_name.end()) {
            throw std::runtime_error("Task id not found while building graph");
        }
        for (const auto& input : task.inputs) {
            const auto tensor_it = tensor_map.find(input.tensor_id);
            if (tensor_it == tensor_map.end()) {
                throw std::runtime_error("Input tensor not found: " + input.tensor_id);
            }
            const auto& tensor = tensor_it->second;
            if (!tensor.producer_task.has_value()) {
                continue;
            }
            const auto src_it = id_to_name.find(*tensor.producer_task);
            if (src_it == id_to_name.end()) {
                throw std::runtime_error("Tensor producer task not found");
            }
            std::string comm_kind = "p2p";
            std::string comm_group;
            std::size_t comm_participants = 0;
            const auto successor_it = task_successors.find(*tensor.producer_task);
            const auto predecessor_it = task_predecessors.find(task.id);
            const auto& successors = successor_it == task_successors.end() ? empty_int_set() : successor_it->second;
            const auto& predecessors = predecessor_it == task_predecessors.end() ? empty_int_set() : predecessor_it->second;

            if (!tensor.collective_hint.empty()) {
                // Explicit collective: compile this tensor's producer->consumer
                // transfers into ONE collective of the hinted type. Forcing a
                // per-tensor comm_group makes taskflow group every such edge into a
                // single COMM_COLL_NODE spanning all participant (pinned) devices,
                // regardless of the dependency shape.
                comm_kind = canonical_comm_kind(tensor.collective_hint);
                if (!is_supported_collective_hint(comm_kind)) {
                    throw std::runtime_error("Unsupported collective_hint '" + tensor.collective_hint +
                                             "' for tensor: " + tensor.id);
                }
                comm_group = "hint|" + tensor.id;
                comm_participants = std::max<std::size_t>(2, successors.size() + 1);
            } else if (is_complete_alltoall_relation(predecessors, successors, task_successors)) {
                comm_kind = "alltoall";
                comm_group = alltoall_group_key(predecessors, successors);
                comm_participants = collective_participants(predecessors, successors, comm_kind);
            } else {
                comm_kind = "p2p";
                comm_participants = collective_participants(predecessors, successors, comm_kind);
            }
            const std::string access_pattern = access_kind_name(effective_input_access(input, tensor));
            graph.add_edge(src_it->second,
                           dst_it->second,
                           0.0,
                           tensor.id,
                           comm_kind,
                           access_pattern,
                           comm_participants,
                           comm_group,
                           dtype_name(tensor.dtype));
        }
    }

    return graph;
}

const std::string& Workload::name() const {
    return name_;
}

const std::vector<Task>& Workload::tasks() const {
    return tasks_;
}

const std::vector<Tensor>& Workload::tensors() const {
    return tensors_;
}

const std::vector<DeviceGroup>& Workload::device_groups() const {
    return device_groups_;
}

const std::vector<std::string>& Workload::iteration_inputs() const {
    return iteration_inputs_;
}

const std::vector<std::string>& Workload::iteration_outputs() const {
    return iteration_outputs_;
}

}  // namespace workload
