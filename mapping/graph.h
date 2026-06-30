#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mapping {

struct TaskInput {
    std::string tensor_id;
    double tensor_bytes{0.0};
    std::vector<std::int64_t> shape;
    std::string storage_format{"dense"};
    std::optional<std::uint64_t> nonzero_elements;
    std::string dtype{"fp32"};
};

struct Task {
    std::string name;
    std::string type;
    std::string subtype;
    double compute_flops{0.0};
    double memory_bytes{0.0};
    double comm_bytes{0.0};
    std::string access_pattern;
    std::unordered_set<std::string> features;
    std::unordered_set<std::string> tags;
    std::vector<TaskInput> input_data;
};

struct TaskEdge {
    std::string src;
    std::string dst;
    double tensor_bytes{0.0};
    std::string tensor_id;
    std::string comm_kind;
    std::string access_pattern;
    std::size_t comm_participants{0};
    std::string comm_group;
    std::string dtype;
};

class TaskGraph {
public:
    void add_task(Task task);
    void add_edge(const std::string& src,
                  const std::string& dst,
                  double tensor_bytes = 0.0,
                  std::string tensor_id = {},
                  std::string comm_kind = {},
                  std::string access_pattern = {},
                  std::size_t comm_participants = 0,
                  std::string comm_group = {},
                  std::string dtype = {});

    const std::vector<TaskEdge>& dependencies(const std::string& name) const;
    const std::vector<TaskEdge>& successors(const std::string& name) const;
    const std::vector<Task>& topological_order() const;
    const std::vector<Task>& source_tasks() const;
    const std::vector<Task>& sink_tasks() const;

    bool has_task(const std::string& name) const;
    const Task& task(const std::string& name) const;

private:
    void invalidate_caches();
    void rebuild_topological_order() const;
    void rebuild_source_tasks() const;
    void rebuild_sink_tasks() const;

    std::unordered_map<std::string, Task> tasks_;
    std::unordered_map<std::string, std::vector<TaskEdge>> edges_;
    std::unordered_map<std::string, std::vector<TaskEdge>> reverse_edges_;
    std::vector<std::string> insertion_order_;
    mutable bool topo_valid_{false};
    mutable bool source_valid_{false};
    mutable bool sink_valid_{false};
    mutable std::vector<Task> topo_cache_;
    mutable std::vector<Task> source_cache_;
    mutable std::vector<Task> sink_cache_;
};

}  // namespace mapping
