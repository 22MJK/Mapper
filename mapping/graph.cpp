#include "mapping/graph.h"

namespace mapping {

namespace {

const std::vector<TaskEdge>& empty_edges() {
    static const std::vector<TaskEdge> kEmpty;
    return kEmpty;
}

}  // namespace

void TaskGraph::add_task(Task task) {
    std::string name = task.name;
    const auto [it, inserted] = tasks_.try_emplace(name, std::move(task));
    if (inserted) {
        insertion_order_.push_back(std::move(name));
    } else {
        it->second = std::move(task);
    }
    invalidate_caches();
}

void TaskGraph::add_edge(const std::string& src,
                         const std::string& dst,
                         double tensor_bytes,
                         std::string tensor_id,
                         std::string comm_kind,
                         std::string access_pattern,
                         std::size_t comm_participants,
                         std::string comm_group,
                         std::string dtype) {
    if (!has_task(src)) {
        throw std::runtime_error("Source task " + src + " not known");
    }
    if (!has_task(dst)) {
        throw std::runtime_error("Destination task " + dst + " not known");
    }
    TaskEdge edge{src,
                  dst,
                  tensor_bytes,
                  std::move(tensor_id),
                  std::move(comm_kind),
                  std::move(access_pattern),
                  comm_participants,
                  std::move(comm_group),
                  std::move(dtype)};
    edges_[src].push_back(edge);
    reverse_edges_[dst].push_back(edge);
    invalidate_caches();
}

const std::vector<TaskEdge>& TaskGraph::dependencies(const std::string& name) const {
    const auto it = reverse_edges_.find(name);
    if (it == reverse_edges_.end()) {
        return empty_edges();
    }
    return it->second;
}

const std::vector<TaskEdge>& TaskGraph::successors(const std::string& name) const {
    const auto it = edges_.find(name);
    if (it == edges_.end()) {
        return empty_edges();
    }
    return it->second;
}

const std::vector<Task>& TaskGraph::topological_order() const {
    if (!topo_valid_) {
        rebuild_topological_order();
    }
    return topo_cache_;
}

const std::vector<Task>& TaskGraph::source_tasks() const {
    if (!source_valid_) {
        rebuild_source_tasks();
    }
    return source_cache_;
}

const std::vector<Task>& TaskGraph::sink_tasks() const {
    if (!sink_valid_) {
        rebuild_sink_tasks();
    }
    return sink_cache_;
}

bool TaskGraph::has_task(const std::string& name) const {
    return tasks_.find(name) != tasks_.end();
}

const Task& TaskGraph::task(const std::string& name) const {
    return tasks_.at(name);
}

void TaskGraph::invalidate_caches() {
    topo_valid_ = false;
    source_valid_ = false;
    sink_valid_ = false;
}

void TaskGraph::rebuild_topological_order() const {
    std::unordered_map<std::string, int> indegree;
    indegree.reserve(insertion_order_.size());
    for (const auto& name : insertion_order_) {
        indegree[name] = 0;
    }
    for (const auto& entry : edges_) {
        for (const auto& edge : entry.second) {
            indegree[edge.dst] += 1;
        }
    }

    std::deque<std::string> queue;
    for (const auto& name : insertion_order_) {
        if (indegree[name] == 0) {
            queue.push_back(name);
        }
    }

    std::vector<Task> order;
    order.reserve(tasks_.size());
    while (!queue.empty()) {
        const auto current = queue.front();
        queue.pop_front();
        order.push_back(tasks_.at(current));
        const auto succ_it = edges_.find(current);
        if (succ_it == edges_.end()) {
            continue;
        }
        for (const auto& edge : succ_it->second) {
            indegree[edge.dst] -= 1;
            if (indegree[edge.dst] == 0) {
                queue.push_back(edge.dst);
            }
        }
    }

    if (order.size() != tasks_.size()) {
        throw std::runtime_error("TaskGraph contains a cycle");
    }
    topo_cache_ = std::move(order);
    topo_valid_ = true;
}

void TaskGraph::rebuild_source_tasks() const {
    std::vector<Task> result;
    result.reserve(insertion_order_.size());
    for (const auto& name : insertion_order_) {
        if (reverse_edges_.find(name) == reverse_edges_.end()) {
            result.push_back(tasks_.at(name));
        }
    }
    source_cache_ = std::move(result);
    source_valid_ = true;
}

void TaskGraph::rebuild_sink_tasks() const {
    std::vector<Task> result;
    result.reserve(insertion_order_.size());
    for (const auto& name : insertion_order_) {
        if (edges_.find(name) == edges_.end()) {
            result.push_back(tasks_.at(name));
        }
    }
    sink_cache_ = std::move(result);
    sink_valid_ = true;
}

}  // namespace mapping
