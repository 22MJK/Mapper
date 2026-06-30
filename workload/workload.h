#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "hardware_topology/topology.h"
#include "mapping/graph.h"

namespace workload {

enum class DType { FP16, BF16, FP32, FP64, INT8, UINT8, INT32, INT64 };
enum class DistKind { NONE, REPLICATED, BLOCK, CYCLIC };
enum class StorageFormat { DENSE, CSR, CSC, COO_IMPORT, BSR, BLOCK_SPARSE };
enum class AccessKind { DENSE, SPARSE_CSR, SPARSE_CSC, SPARSE_COO, SPARSE_BLOCK, ROW_WISE, COL_WISE };

struct Distribution {
    DistKind kind{DistKind::NONE};
    int axis{-1};
    std::string group;
};

struct Replication {
    std::string mode;
};

struct Tensor {
    std::string id;
    std::string name;
    DType dtype{DType::FP32};
    StorageFormat storage_format{StorageFormat::DENSE};
    std::vector<std::int64_t> shape;
    std::uint64_t size_bytes{0};
    std::optional<std::uint64_t> num_elements;
    std::optional<std::uint64_t> nonzero_elements;
    Distribution distribution;
    std::optional<Replication> replication;
    AccessKind access_pattern{AccessKind::DENSE};
    std::optional<int> producer_task;
    // When set (e.g. "allreduce", "allgather", "reducescatter", "alltoall",
    // "broadcast"), the producer->consumer transfers of this tensor are
    // compiled into a single collective of that type over every participant
    // device, instead of being inferred from the dependency topology.
    std::string collective_hint;
};

struct TensorUse {
    std::string tensor_id;
    std::string role;
    AccessKind access{AccessKind::DENSE};
    bool access_explicit{false};
};

struct Task {
    int id{0};
    std::string name;
    std::string op;
    std::optional<int> level;
    std::vector<int> block;
    double compute_flops{0.0};
    double memory_bytes{0.0};
    std::vector<TensorUse> inputs;
    std::vector<std::string> outputs;
    std::string placement_group;
    std::string placement_parallelism;
    std::unordered_set<std::string> features;
};

struct DeviceGroup {
    std::string id;
    std::vector<std::string> members;
};

class Workload {
public:
    Workload(std::string name,
             std::vector<Task> tasks,
             std::vector<Tensor> tensors,
             std::vector<DeviceGroup> device_groups,
             std::vector<std::string> iteration_inputs = {},
             std::vector<std::string> iteration_outputs = {},
             std::optional<mapping::TaskGraph> explicit_task_graph = std::nullopt);
    mapping::TaskGraph to_task_graph(const hardware_topology::HardwareTopology& topology) const;
    const std::string& name() const;
    const std::vector<Task>& tasks() const;
    const std::vector<Tensor>& tensors() const;
    const std::vector<DeviceGroup>& device_groups() const;
    const std::vector<std::string>& iteration_inputs() const;
    const std::vector<std::string>& iteration_outputs() const;

private:
    std::string name_;
    std::vector<Task> tasks_;
    std::vector<Tensor> tensors_;
    std::vector<DeviceGroup> device_groups_;
    std::vector<std::string> iteration_inputs_;
    std::vector<std::string> iteration_outputs_;
    std::optional<mapping::TaskGraph> explicit_task_graph_;
};

}  // namespace workload
