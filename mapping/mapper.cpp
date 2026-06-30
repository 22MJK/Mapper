#include "mapping/mapper.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "mapping/cost_model.h"
#include "mapping/schedule_model.h"

namespace mapping {

const std::string& MappingPlan::node_for(const std::string& task_name) const {
    return assignments.at(task_name);
}

namespace {

double compute_time_seconds(const Task& task, const hardware_topology::Device* device);

constexpr std::size_t kDependencyLocalityMinDeviceCount = 12;
constexpr std::size_t kDependencyLocalityMinNearbyDevices = 4;
constexpr std::size_t kDependencyLocalityMaxNearbyDevices = 32;
constexpr std::size_t kDependencyLocalitySpilloverDevices = 4;
constexpr std::size_t kPeftLcPreferredDevices = 8;
constexpr std::size_t kPeftLcLocalityMinNearbyDevices = 8;
constexpr std::size_t kPeftLcLocalityMaxNearbyDevices = 24;
constexpr std::size_t kPeftLcFullCommCacheMaxShapes = 1024;
constexpr std::size_t kPeftLcFullCommCacheMaxEntries = 16 * 1024 * 1024;
constexpr double kPeftLcMinAverageEligibleDevices = 24.0;

std::optional<std::string> pinned_device_tag(const Task& task) {
    static const std::string prefix = "device:";
    // Hard device pins arrive either as tags (LLM taskgraph builder) or as
    // features (mapper.workload.v2 JSON copies a task's "features" array onto the
    // mapping task but never its tags). Scan both so a synthetic workload can pin
    // a producer/consumer to a specific compute device id, e.g. "device:gpu5".
    for (const auto* bag : {&task.tags, &task.features}) {
        for (const auto& entry : *bag) {
            if (entry.rfind(prefix, 0) == 0 && entry.size() > prefix.size()) {
                return entry.substr(prefix.size());
            }
        }
    }
    return std::nullopt;
}

bool has_tag_or_feature(const Task& task, const std::string& value) {
    return task.tags.find(value) != task.tags.end() || task.features.find(value) != task.features.end();
}

bool is_cpu_device(const hardware_topology::Device* device) {
    if (device == nullptr) {
        return false;
    }
    std::string type = device->type;
    for (char& ch : type) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return type == "cpu";
}

bool is_gpu_device(const hardware_topology::Device* device) {
    if (device == nullptr) {
        return false;
    }
    std::string type = device->type;
    for (char& ch : type) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return type == "gpu";
}

std::string normalized_device_type(const hardware_topology::Device* device) {
    if (device == nullptr) {
        return "unknown";
    }
    std::string type = device->type.empty() ? device->id : device->type;
    for (char& ch : type) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return type.empty() ? "unknown" : type;
}

bool is_device_allowed_for_task(const Task& task, const hardware_topology::Device* device) {
    if (device == nullptr || !device->compute_capable) {
        return false;
    }
    if (is_cpu_device(device) && has_tag_or_feature(task, "cpu_unsupported")) {
        return false;
    }
    if (is_gpu_device(device) && has_tag_or_feature(task, "gpu_unsupported")) {
        return false;
    }
    return true;
}

std::size_t bytes_to_size_t(double bytes) {
    if (!(bytes > 0.0)) {
        return 0;
    }
    const auto limit = static_cast<long double>(std::numeric_limits<std::size_t>::max());
    if (static_cast<long double>(bytes) >= limit) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(bytes);
}

std::size_t dependency_locality_neighbor_limit(std::size_t device_count) {
    if (device_count <= kDependencyLocalityMinDeviceCount) {
        return device_count;
    }
    const std::size_t scaled_limit = std::max<std::size_t>(kDependencyLocalityMinNearbyDevices, device_count / 4);
    return std::min(device_count, std::min(kDependencyLocalityMaxNearbyDevices, scaled_limit));
}

std::size_t peft_lc_locality_neighbor_limit(std::size_t device_count) {
    if (device_count <= kDependencyLocalityMinDeviceCount) {
        return device_count;
    }
    const std::size_t scaled_limit = std::max<std::size_t>(kPeftLcLocalityMinNearbyDevices, device_count / 6);
    return std::min(device_count, std::min(kPeftLcLocalityMaxNearbyDevices, scaled_limit));
}

struct IndexedEdge {
    const TaskEdge* edge{nullptr};
    std::size_t src_task{0};
    std::size_t dst_task{0};
    bool collective{false};
    std::size_t communication_shape{std::numeric_limits<std::size_t>::max()};
    double average_comm_time{0.0};
};

struct CommunicationShapeKey {
    double tensor_bytes{0.0};
    std::string comm_kind;
    std::string access_pattern;
    std::size_t comm_participants{0};
    std::string comm_group;
    std::string dtype;

    bool operator==(const CommunicationShapeKey& other) const {
        return tensor_bytes == other.tensor_bytes &&
               comm_kind == other.comm_kind &&
               access_pattern == other.access_pattern &&
               comm_participants == other.comm_participants &&
               comm_group == other.comm_group &&
               dtype == other.dtype;
    }
};

struct CommunicationShapeKeyHash {
    std::size_t operator()(const CommunicationShapeKey& key) const {
        std::size_t seed = 0;
        auto combine = [&](std::size_t value) {
            seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        };
        combine(std::hash<double>{}(key.tensor_bytes));
        combine(std::hash<std::string>{}(key.comm_kind));
        combine(std::hash<std::string>{}(key.access_pattern));
        combine(std::hash<std::size_t>{}(key.comm_participants));
        combine(std::hash<std::string>{}(key.comm_group));
        combine(std::hash<std::string>{}(key.dtype));
        return seed;
    }
};

struct CommunicationShapeCache {
    bool collective{false};
    bool sparse_p2p{false};
    const TaskEdge* representative_edge{nullptr};
    std::vector<double> comm_times;
    std::unordered_map<std::uint64_t, double> sparse_comm_times;
    double collective_time{std::numeric_limits<double>::infinity()};
    double average_comm_time{0.0};
};

enum class CommunicationCacheMode {
    Full,
    Lazy,
};

struct ScheduleContext {
    const std::vector<Task>& tasks;
    const std::vector<const hardware_topology::Device*>& devices;
    const hardware_topology::HardwareTopology* topology{nullptr};
    std::vector<std::vector<std::size_t>> successors;
    std::vector<std::vector<std::size_t>> dependencies;
    std::vector<std::vector<std::size_t>> allowed_devices;
    std::vector<std::vector<std::size_t>> eligible_devices;
    std::vector<std::vector<std::size_t>> local_destinations_by_source;
    std::vector<std::size_t> all_devices;
    mutable std::vector<CommunicationShapeCache> communication_shapes;
    std::unordered_map<CommunicationShapeKey, std::size_t, CommunicationShapeKeyHash> communication_shape_index;
    std::vector<double> compute_times;
    std::vector<unsigned char> eligible_device_mask;
    std::vector<unsigned char> local_device_pairs;
    std::vector<IndexedEdge> edges;
    CommunicationCacheMode communication_cache_mode{CommunicationCacheMode::Full};
    bool locality_pruning{false};

    ScheduleContext(const TaskGraph& graph,
                    const hardware_topology::HardwareTopology& topology,
                    bool build_locality_index = true,
                    CommunicationCacheMode cache_mode = CommunicationCacheMode::Full)
        : tasks(graph.topological_order()),
          devices(topology.compute_devices()),
          topology(&topology),
          communication_cache_mode(cache_mode) {
        const std::size_t task_count = tasks.size();
        const std::size_t device_count = devices.size();
        if (build_locality_index) {
            build_device_locality(topology);
        }

        std::unordered_map<std::string, std::size_t> task_index;
        task_index.reserve(task_count);
        for (std::size_t i = 0; i < task_count; ++i) {
            task_index.emplace(tasks[i].name, i);
        }

        successors.resize(task_count);
        dependencies.resize(task_count);
        allowed_devices.resize(task_count);
        eligible_devices.resize(task_count);
        compute_times.assign(task_count * device_count, std::numeric_limits<double>::infinity());
        eligible_device_mask.assign(task_count * device_count, 0);

        for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
            const auto& task = tasks[task_idx];
            const auto pin = pinned_device_tag(task);
            auto& allowed = allowed_devices[task_idx];
            auto& eligible = eligible_devices[task_idx];
            allowed.reserve(device_count);
            eligible.reserve(device_count);

            for (std::size_t device_idx = 0; device_idx < device_count; ++device_idx) {
                const auto* device = devices[device_idx];
                if (!is_device_allowed_for_task(task, device)) {
                    continue;
                }
                allowed.push_back(device_idx);
                if (!pin.has_value() || device->id == *pin) {
                    eligible.push_back(device_idx);
                    eligible_device_mask[task_idx * device_count + device_idx] = 1;
                }
                compute_times[task_idx * device_count + device_idx] = compute_time_seconds(task, device);
            }

            if (pin.has_value() && eligible.empty()) {
                const auto* target = topology.device(*pin);
                if (target == nullptr || !target->compute_capable) {
                    throw std::runtime_error("Pinned compute device not found: " + *pin);
                }
                throw std::runtime_error("Pinned device is not supported by task profile: " + task.name);
            }
            if (eligible.empty()) {
                throw std::runtime_error("No eligible device for task: " + task.name);
            }
        }

        for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
            for (const auto& edge : graph.successors(tasks[task_idx].name)) {
                const auto dst_it = task_index.find(edge.dst);
                if (dst_it == task_index.end()) {
                    throw std::runtime_error("Destination task " + edge.dst + " not known");
                }

                IndexedEdge indexed;
                indexed.edge = &edge;
                indexed.src_task = task_idx;
                indexed.dst_task = dst_it->second;
                indexed.collective = is_collective_kind(edge.comm_kind);
                indexed.communication_shape = communication_shape_for_edge(edge, topology);
                indexed.average_comm_time = communication_shapes[indexed.communication_shape].average_comm_time;

                const std::size_t edge_idx = edges.size();
                successors[task_idx].push_back(edge_idx);
                dependencies[dst_it->second].push_back(edge_idx);
                edges.push_back(std::move(indexed));
            }
        }
    }

    double compute_time(std::size_t task_idx, std::size_t device_idx) const {
        return compute_times[task_idx * devices.size() + device_idx];
    }

    double communication_time(std::size_t edge_idx, std::size_t src_device, std::size_t dst_device) const {
        const auto& edge = edges[edge_idx];
        auto& shape = communication_shapes[edge.communication_shape];
        if (shape.collective) {
            return shape.collective_time;
        }
        const auto device_count = devices.size();
        if (src_device >= device_count || dst_device >= device_count) {
            return std::numeric_limits<double>::infinity();
        }
        if (src_device == dst_device) {
            return 0.0;
        }
        if (shape.sparse_p2p) {
            const auto key = static_cast<std::uint64_t>(src_device) * static_cast<std::uint64_t>(device_count) +
                             static_cast<std::uint64_t>(dst_device);
            if (const auto it = shape.sparse_comm_times.find(key); it != shape.sparse_comm_times.end()) {
                return it->second;
            }
            if (shape.representative_edge == nullptr || topology == nullptr) {
                return std::numeric_limits<double>::infinity();
            }
            const double time = estimate_communication_time_seconds(*shape.representative_edge,
                                                                    *topology,
                                                                    devices[src_device]->id,
                                                                    devices[dst_device]->id);
            shape.sparse_comm_times.emplace(key, time);
            return time;
        }
        return shape.comm_times[src_device * device_count + dst_device];
    }

    double exact_communication_time(std::size_t edge_idx,
                                    const hardware_topology::HardwareTopology&,
                                    std::size_t src_device,
                                    std::size_t dst_device) const {
        return communication_time(edge_idx, src_device, dst_device);
    }

    bool is_eligible(std::size_t task_idx, std::size_t device_idx) const {
        return eligible_device_mask[task_idx * devices.size() + device_idx] != 0;
    }

    bool local_device_pair(std::size_t src_device, std::size_t dst_device) const {
        return local_device_pairs[src_device * devices.size() + dst_device] != 0;
    }

    const std::vector<std::size_t>& cached_destinations_for_source(std::size_t src_device) const {
        return locality_pruning ? local_destinations_by_source[src_device] : all_devices;
    }

    void add_candidate(std::size_t device_idx,
                       std::vector<std::size_t>& candidates,
                       std::vector<unsigned char>& candidate_mask) const {
        if (candidate_mask[device_idx] != 0) {
            return;
        }
        candidate_mask[device_idx] = 1;
        candidates.push_back(device_idx);
    }

    template <typename ScoreFn>
    void add_best_scored_candidates(std::size_t task_idx,
                                    std::size_t limit,
                                    std::vector<std::size_t>& candidates,
                                    std::vector<unsigned char>& candidate_mask,
                                    ScoreFn score_fn) const {
        if (limit == 0) {
            return;
        }
        std::vector<std::pair<double, std::size_t>> scored;
        scored.reserve(eligible_devices[task_idx].size());
        for (const auto device_idx : eligible_devices[task_idx]) {
            if (candidate_mask[device_idx] != 0) {
                continue;
            }
            const double score = score_fn(device_idx);
            if (std::isfinite(score)) {
                scored.push_back({score, device_idx});
            }
        }
        if (scored.empty()) {
            return;
        }
        std::sort(scored.begin(),
                  scored.end(),
                  [&](const auto& lhs, const auto& rhs) {
                      if (lhs.first == rhs.first) {
                          return devices[lhs.second]->id < devices[rhs.second]->id;
                      }
                      return lhs.first < rhs.first;
                  });
        const std::size_t count = std::min(limit, scored.size());
        for (std::size_t i = 0; i < count; ++i) {
            add_candidate(scored[i].second, candidates, candidate_mask);
        }
    }

    bool collect_nearby_dependency_candidates(std::size_t task_idx,
                                              const std::vector<std::size_t>& assignment,
                                              std::vector<std::size_t>& candidates,
                                              std::vector<unsigned char>& nearby_mask) const {
        candidates.clear();
        if (!locality_pruning || dependencies[task_idx].empty()) {
            return false;
        }

        std::fill(nearby_mask.begin(), nearby_mask.end(), 0);
        bool initialized = false;
        for (const auto edge_idx : dependencies[task_idx]) {
            const auto& edge = edges[edge_idx];
            if (edge.collective) {
                continue;
            }
            const auto pred_device = assignment[edge.src_task];
            if (pred_device == std::numeric_limits<std::size_t>::max()) {
                throw std::runtime_error("Task dependencies must be assigned before successors");
            }
            if (!initialized) {
                for (const auto device_idx : local_destinations_by_source[pred_device]) {
                    nearby_mask[device_idx] = 1;
                }
                initialized = true;
                continue;
            }
            for (std::size_t device_idx = 0; device_idx < devices.size(); ++device_idx) {
                if (nearby_mask[device_idx] != 0 && !local_device_pair(pred_device, device_idx)) {
                    nearby_mask[device_idx] = 0;
                }
            }
        }
        if (!initialized) {
            return false;
        }

        for (const auto device_idx : eligible_devices[task_idx]) {
            if (nearby_mask[device_idx] != 0) {
                candidates.push_back(device_idx);
            }
        }
        return true;
    }

private:
    static CommunicationShapeKey communication_shape_key(const TaskEdge& edge) {
        return CommunicationShapeKey{
            edge.tensor_bytes,
            canonical_comm_kind(edge.comm_kind),
            edge.access_pattern,
            edge.comm_participants,
            edge.comm_group,
            edge.dtype,
        };
    }

    std::size_t communication_shape_for_edge(const TaskEdge& edge,
                                             const hardware_topology::HardwareTopology& topology) {
        const auto key = communication_shape_key(edge);
        if (const auto it = communication_shape_index.find(key); it != communication_shape_index.end()) {
            return it->second;
        }

        CommunicationShapeCache shape;
        shape.collective = is_collective_kind(edge.comm_kind);
        shape.representative_edge = &edge;

        const auto device_count = devices.size();
        if (shape.collective) {
            shape.collective_time =
                estimate_communication_time_seconds(edge, topology, devices.front()->id, devices.front()->id);
            shape.average_comm_time = shape.collective_time;
        } else if (communication_cache_mode == CommunicationCacheMode::Lazy) {
            shape.sparse_p2p = true;
        } else {
            shape.comm_times.assign(device_count * device_count, std::numeric_limits<double>::infinity());

            double total = 0.0;
            std::size_t count = 0;
            for (std::size_t src_device = 0; src_device < device_count; ++src_device) {
                for (std::size_t dst_device = 0; dst_device < device_count; ++dst_device) {
                    double time = 0.0;
                    if (src_device != dst_device) {
                        time = estimate_communication_time_seconds(edge,
                                                                   topology,
                                                                   devices[src_device]->id,
                                                                   devices[dst_device]->id);
                    }
                    shape.comm_times[src_device * device_count + dst_device] = time;

                    if (src_device == dst_device) {
                        continue;
                    }
                    if (std::isfinite(time)) {
                        total += time;
                        count += 1;
                    }
                }
            }
            shape.average_comm_time =
                device_count < 2 ? 0.0
                                 : (count == 0 ? std::numeric_limits<double>::infinity()
                                               : total / static_cast<double>(count));
        }

        const auto shape_idx = communication_shapes.size();
        communication_shapes.push_back(std::move(shape));
        communication_shape_index.emplace(std::move(key), shape_idx);
        return shape_idx;
    }

    void build_device_locality(const hardware_topology::HardwareTopology& topology) {
        const std::size_t device_count = devices.size();
        const std::size_t nearby_limit = dependency_locality_neighbor_limit(device_count);
        locality_pruning = nearby_limit < device_count;

        all_devices.clear();
        all_devices.reserve(device_count);
        for (std::size_t device_idx = 0; device_idx < device_count; ++device_idx) {
            all_devices.push_back(device_idx);
        }

        if (!locality_pruning) {
            return;
        }

        local_destinations_by_source.assign(device_count, {});
        local_device_pairs.assign(device_count * device_count, 0);
        for (std::size_t src_device = 0; src_device < device_count; ++src_device) {
            std::vector<std::pair<double, std::size_t>> distances;
            distances.reserve(device_count);
            for (std::size_t dst_device = 0; dst_device < device_count; ++dst_device) {
                double distance = 0.0;
                if (src_device != dst_device) {
                    distance = topology.get_transfer_time(devices[src_device]->id, devices[dst_device]->id, 0);
                }
                if (std::isfinite(distance)) {
                    distances.push_back({distance, dst_device});
                }
            }
            if (distances.empty()) {
                local_device_pairs[src_device * device_count + src_device] = 1;
                local_destinations_by_source[src_device].push_back(src_device);
                continue;
            }

            std::sort(distances.begin(),
                      distances.end(),
                      [&](const auto& lhs, const auto& rhs) {
                          if (lhs.first == rhs.first) {
                              return devices[lhs.second]->id < devices[rhs.second]->id;
                          }
                          return lhs.first < rhs.first;
                      });

            const std::size_t cutoff_pos = std::min(nearby_limit, distances.size()) - 1;
            const double cutoff = distances[cutoff_pos].first;
            for (const auto& entry : distances) {
                if (entry.first > cutoff) {
                    break;
                }
                local_device_pairs[src_device * device_count + entry.second] = 1;
            }
            local_device_pairs[src_device * device_count + src_device] = 1;

            auto& destinations = local_destinations_by_source[src_device];
            destinations.reserve(std::min(device_count, nearby_limit));
            for (std::size_t dst_device = 0; dst_device < device_count; ++dst_device) {
                if (local_device_pair(src_device, dst_device)) {
                    destinations.push_back(dst_device);
                }
            }
        }
    }
};

}  // namespace

MappingPlan GreedyMapper::map(const TaskGraph& graph, const hardware_topology::HardwareTopology& topology) const {
    const auto& devices = topology.compute_devices();
    if (devices.empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }

    std::unordered_map<std::string, double> available;
    for (const auto* device : devices) {
        available[device->id] = 0.0;
    }

    MappingPlan plan;
    for (const auto& task : graph.topological_order()) {
        const auto pin = pinned_device_tag(task);
        const hardware_topology::Device* target = nullptr;
        double best_finish = std::numeric_limits<double>::infinity();
        if (pin.has_value()) {
            target = topology.device(*pin);
            if (target == nullptr || !target->compute_capable) {
                throw std::runtime_error("Pinned compute device not found: " + *pin);
            }
            if (!is_device_allowed_for_task(task, target)) {
                throw std::runtime_error("Pinned device is not supported by task profile: " + task.name);
            }
            best_finish = available[target->id] + compute_time_seconds(task, target);
        } else {
            for (const auto* device : devices) {
                if (!is_device_allowed_for_task(task, device)) {
                    continue;
                }
                const double finish = available[device->id] + compute_time_seconds(task, device);
                if (finish < best_finish || (finish == best_finish && (target == nullptr || device->id < target->id))) {
                    best_finish = finish;
                    target = device;
                }
            }
            if (target == nullptr) {
                throw std::runtime_error("No eligible device for task: " + task.name);
            }
        }
        plan.assignments[task.name] = target->id;
        available[target->id] = best_finish;
    }
    return plan;
}

namespace {

double compute_time_seconds(const Task& task, const hardware_topology::Device* device) {
    return estimate_task_time_seconds(task, device);
}

bool should_use_peft_lc_pruning(const TaskGraph& graph,
                                const hardware_topology::HardwareTopology& topology) {
    const auto& devices = topology.compute_devices();
    if (devices.size() <= kDependencyLocalityMinDeviceCount) {
        return false;
    }

    const auto& tasks = graph.topological_order();
    std::uint64_t eligible_total = 0;
    for (const auto& task : tasks) {
        const auto pin = pinned_device_tag(task);
        std::size_t eligible = 0;
        for (const auto* device : devices) {
            if (!is_device_allowed_for_task(task, device)) {
                continue;
            }
            if (!pin.has_value() || device->id == *pin) {
                ++eligible;
            }
        }
        if (eligible == 0) {
            return false;
        }
        eligible_total += eligible;
    }
    const double average_eligible =
        static_cast<double>(eligible_total) / static_cast<double>(std::max<std::size_t>(1, tasks.size()));
    if (average_eligible < kPeftLcMinAverageEligibleDevices) {
        return false;
    }

    std::size_t positive_p2p_edges = 0;
    for (const auto& task : tasks) {
        for (const auto& edge : graph.successors(task.name)) {
            if (!is_collective_kind(edge.comm_kind) && edge.tensor_bytes > 0.0) {
                ++positive_p2p_edges;
            }
        }
    }
    return positive_p2p_edges > 0;
}

std::size_t peft_lc_communication_shape_count_limited(const TaskGraph& graph, std::size_t limit) {
    std::unordered_set<CommunicationShapeKey, CommunicationShapeKeyHash> shapes;
    shapes.reserve(std::min<std::size_t>(limit + 1, 4096));
    const auto& tasks = graph.topological_order();
    for (const auto& task : tasks) {
        for (const auto& edge : graph.successors(task.name)) {
            shapes.insert(CommunicationShapeKey{
                edge.tensor_bytes,
                canonical_comm_kind(edge.comm_kind),
                edge.access_pattern,
                edge.comm_participants,
                edge.comm_group,
                edge.dtype,
            });
            if (shapes.size() > limit) {
                return shapes.size();
            }
        }
    }
    return shapes.size();
}

bool should_use_peft_lc_full_comm_cache(const TaskGraph& graph,
                                        const hardware_topology::HardwareTopology& topology) {
    const std::size_t device_count = topology.compute_devices().size();
    if (device_count == 0) {
        return false;
    }

    const std::size_t entries_per_shape = device_count * device_count;
    if (entries_per_shape == 0 || entries_per_shape > kPeftLcFullCommCacheMaxEntries) {
        return false;
    }

    const std::size_t max_shapes_by_entries = kPeftLcFullCommCacheMaxEntries / entries_per_shape;
    const std::size_t shape_limit = std::min(kPeftLcFullCommCacheMaxShapes, max_shapes_by_entries);
    if (shape_limit == 0) {
        return false;
    }
    return peft_lc_communication_shape_count_limited(graph, shape_limit) <= shape_limit;
}

std::vector<std::vector<std::size_t>> build_peft_lc_local_destinations(const ScheduleContext& ctx) {
    const std::size_t device_count = ctx.devices.size();
    const std::size_t nearby_limit = peft_lc_locality_neighbor_limit(device_count);
    std::vector<std::vector<std::size_t>> destinations(device_count);

    if (nearby_limit >= device_count) {
        std::vector<std::size_t> all_devices;
        all_devices.reserve(device_count);
        for (std::size_t device_idx = 0; device_idx < device_count; ++device_idx) {
            all_devices.push_back(device_idx);
        }
        for (auto& per_source : destinations) {
            per_source = all_devices;
        }
        return destinations;
    }

    std::size_t representative_edge_idx = std::numeric_limits<std::size_t>::max();
    double representative_bytes = -1.0;
    for (std::size_t edge_idx = 0; edge_idx < ctx.edges.size(); ++edge_idx) {
        const auto& edge = ctx.edges[edge_idx];
        if (edge.collective || edge.edge == nullptr || edge.edge->tensor_bytes <= representative_bytes) {
            continue;
        }
        representative_edge_idx = edge_idx;
        representative_bytes = edge.edge->tensor_bytes;
    }
    if (representative_edge_idx == std::numeric_limits<std::size_t>::max()) {
        return destinations;
    }

    for (std::size_t src_device = 0; src_device < device_count; ++src_device) {
        std::vector<std::pair<double, std::size_t>> distances;
        distances.reserve(device_count);
        for (std::size_t dst_device = 0; dst_device < device_count; ++dst_device) {
            double distance = 0.0;
            if (src_device != dst_device) {
                distance = ctx.communication_time(representative_edge_idx, src_device, dst_device);
            }
            if (std::isfinite(distance)) {
                distances.push_back({distance, dst_device});
            }
        }
        if (distances.empty()) {
            destinations[src_device].push_back(src_device);
            continue;
        }

        std::sort(distances.begin(),
                  distances.end(),
                  [&](const auto& lhs, const auto& rhs) {
                      if (lhs.first == rhs.first) {
                          return ctx.devices[lhs.second]->id < ctx.devices[rhs.second]->id;
                      }
                      return lhs.first < rhs.first;
                  });

        auto& per_source = destinations[src_device];
        const std::size_t count = std::min(nearby_limit, distances.size());
        per_source.reserve(count + 1);
        bool has_source = false;
        for (std::size_t i = 0; i < count; ++i) {
            const auto device_idx = distances[i].second;
            has_source = has_source || device_idx == src_device;
            per_source.push_back(device_idx);
        }
        if (!has_source) {
            per_source.push_back(src_device);
        }
    }

    return destinations;
}

MappingPlan map_peft_exact(const TaskGraph& graph,
                           const hardware_topology::HardwareTopology& topology,
                           bool build_locality_index) {
    if (topology.compute_devices().empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }
    ScheduleContext ctx(graph, topology, build_locality_index);

    const std::size_t task_count = ctx.tasks.size();
    const std::size_t device_count = ctx.devices.size();
    const auto kInf = std::numeric_limits<double>::infinity();

    std::vector<double> oct(task_count * device_count, kInf);
    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        for (const auto device_idx : ctx.eligible_devices[pos]) {
            double value = 0.0;
            for (const auto edge_idx : ctx.successors[pos]) {
                const auto& edge = ctx.edges[edge_idx];
                double best_successor = kInf;

                for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
                    const double comm = ctx.communication_time(edge_idx, device_idx, succ_device_idx);
                    const double succ_cost = ctx.compute_time(edge.dst_task, succ_device_idx);
                    const double succ_oct = oct[edge.dst_task * device_count + succ_device_idx];
                    const double candidate = comm + succ_cost + succ_oct;
                    if (candidate < best_successor) {
                        best_successor = candidate;
                    }
                }
                value = std::max(value, best_successor);
            }
            oct[pos * device_count + device_idx] = value;
        }
    }

    constexpr std::size_t kUnassigned = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> assignment(task_count, kUnassigned);
    std::vector<double> finish_time(task_count, 0.0);
    std::vector<double> available(device_count, 0.0);

    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        double best_metric = kInf;
        double best_finish = kInf;
        std::size_t best_device = ctx.eligible_devices[task_idx].front();

        auto candidate_ready = [&](std::size_t device_idx) {
            double ready = available[device_idx];

            for (const auto edge_idx : ctx.dependencies[task_idx]) {
                const auto& edge = ctx.edges[edge_idx];
                const auto pred_device = assignment[edge.src_task];
                if (pred_device == kUnassigned) {
                    throw std::runtime_error("Task dependencies must be assigned before successors");
                }

                const double comm = ctx.communication_time(edge_idx, pred_device, device_idx);
                if (!std::isfinite(comm)) {
                    return kInf;
                }
                ready = std::max(ready, finish_time[edge.src_task] + comm);
            }
            return ready;
        };

        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const auto* device = ctx.devices[device_idx];
            const double ready = candidate_ready(device_idx);
            if (!std::isfinite(ready)) {
                continue;
            }

            const double exec = ctx.compute_time(task_idx, device_idx);
            const double eft = ready + exec;
            const double metric = eft + oct[task_idx * device_count + device_idx];
            if (metric < best_metric ||
                (metric == best_metric &&
                 (eft < best_finish || (eft == best_finish && device->id < ctx.devices[best_device]->id)))) {
                best_metric = metric;
                best_finish = eft;
                best_device = device_idx;
            }
        }

        if (!std::isfinite(best_finish)) {
            throw std::runtime_error("No eligible device for task: " + ctx.tasks[task_idx].name);
        }

        assignment[task_idx] = best_device;
        finish_time[task_idx] = best_finish;
        available[best_device] = best_finish;
    }

    MappingPlan plan;
    plan.assignments.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        plan.assignments.emplace(ctx.tasks[task_idx].name, ctx.devices[assignment[task_idx]]->id);
    }

    return plan;
}

}  // namespace

MappingPlan HeftMapper::map(const TaskGraph& graph, const hardware_topology::HardwareTopology& topology) const {
    if (topology.compute_devices().empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }
    ScheduleContext ctx(graph, topology);

    const std::size_t task_count = ctx.tasks.size();
    const double kInf = std::numeric_limits<double>::infinity();

    std::vector<double> avg_comp(task_count, kInf);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        double total = 0.0;
        std::size_t count = 0;
        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const double time = ctx.compute_time(task_idx, device_idx);
            if (!std::isfinite(time)) {
                continue;
            }
            total += time;
            count += 1;
        }
        if (count > 0) {
            avg_comp[task_idx] = total / static_cast<double>(count);
        }
    }

    std::vector<double> rank(task_count, 0.0);
    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        double max_succ = 0.0;
        for (const auto edge_idx : ctx.successors[pos]) {
            const auto& edge = ctx.edges[edge_idx];
            max_succ = std::max(max_succ, edge.average_comm_time + rank[edge.dst_task]);
        }
        rank[pos] = avg_comp[pos] + max_succ;
    }

    std::vector<std::size_t> order;
    order.reserve(task_count);
    for (std::size_t i = 0; i < task_count; ++i) {
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const double ra = rank[a];
        const double rb = rank[b];
        if (ra == rb) {
            return a < b;
        }
        return ra > rb;
    });

    struct ScheduledSlot {
        double start{0.0};
        double finish{0.0};
        std::size_t task_idx{0};
    };

    constexpr std::size_t kUnassigned = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> assignment(task_count, kUnassigned);
    std::vector<double> finish_time(task_count, 0.0);
    std::vector<std::vector<ScheduledSlot>> device_schedules(ctx.devices.size());

    auto earliest_inserted_start = [](const std::vector<ScheduledSlot>& schedule,
                                      double ready,
                                      double duration) {
        if (!std::isfinite(ready) || !std::isfinite(duration)) {
            return std::numeric_limits<double>::infinity();
        }
        double start = ready;
        const double epsilon = 1e-18;
        for (const auto& slot : schedule) {
            if (start + duration <= slot.start + epsilon) {
                return start;
            }
            start = std::max(start, slot.finish);
        }
        return start;
    };

    for (const auto task_idx : order) {
        double best_start = kInf;
        double best_finish = kInf;
        std::size_t best_device = ctx.eligible_devices[task_idx].front();

        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const double duration = ctx.compute_time(task_idx, device_idx);
            if (!std::isfinite(duration)) {
                continue;
            }

            double ready = 0.0;
            for (const auto edge_idx : ctx.dependencies[task_idx]) {
                const auto& edge = ctx.edges[edge_idx];
                const auto pred_device = assignment[edge.src_task];
                if (pred_device == kUnassigned) {
                    throw std::runtime_error("Task dependencies must be assigned before successors");
                }
                const double comm = ctx.communication_time(edge_idx, pred_device, device_idx);
                if (!std::isfinite(comm)) {
                    ready = kInf;
                    break;
                }
                ready = std::max(ready, finish_time[edge.src_task] + comm);
            }
            const double start = earliest_inserted_start(device_schedules[device_idx], ready, duration);
            const double finish = start + duration;
            const auto* device = ctx.devices[device_idx];
            if (finish < best_finish ||
                (finish == best_finish &&
                 (start < best_start ||
                  (start == best_start && device->id < ctx.devices[best_device]->id)))) {
                best_start = start;
                best_finish = finish;
                best_device = device_idx;
            }
        }

        if (!std::isfinite(best_finish)) {
            throw std::runtime_error("No eligible device for task: " + ctx.tasks[task_idx].name);
        }

        assignment[task_idx] = best_device;
        finish_time[task_idx] = best_finish;
        auto& schedule = device_schedules[best_device];
        const auto insert_at = std::lower_bound(schedule.begin(),
                                                schedule.end(),
                                                best_start,
                                                [](const ScheduledSlot& slot, double start) {
                                                    return slot.start < start;
                                                });
        schedule.insert(insert_at, ScheduledSlot{best_start, best_finish, task_idx});
    }

    MappingPlan plan;
    plan.assignments.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        plan.assignments.emplace(ctx.tasks[task_idx].name, ctx.devices[assignment[task_idx]]->id);
    }

    return plan;
}

MappingPlan AeftMapper::map(const TaskGraph& graph, const hardware_topology::HardwareTopology& topology) const {
    if (topology.compute_devices().empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }
    ScheduleContext ctx(graph, topology);

    const std::size_t task_count = ctx.tasks.size();
    const std::size_t device_count = ctx.devices.size();
    const double kInf = std::numeric_limits<double>::infinity();

    std::vector<double> ioct(task_count * device_count, kInf);
    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        for (const auto device_idx : ctx.eligible_devices[pos]) {
            const double compute = ctx.compute_time(pos, device_idx);
            if (!std::isfinite(compute)) {
                continue;
            }

            double longest_shortest_successor = 0.0;
            for (const auto edge_idx : ctx.successors[pos]) {
                const auto& edge = ctx.edges[edge_idx];
                double shortest_successor_path = kInf;
                for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
                    const double comm = ctx.communication_time(edge_idx, device_idx, succ_device_idx);
                    const double succ_tail = ioct[edge.dst_task * device_count + succ_device_idx];
                    if (!std::isfinite(comm) || !std::isfinite(succ_tail)) {
                        continue;
                    }
                    shortest_successor_path = std::min(shortest_successor_path, comm + succ_tail);
                }
                longest_shortest_successor = std::max(longest_shortest_successor, shortest_successor_path);
            }

            if (std::isfinite(longest_shortest_successor)) {
                ioct[pos * device_count + device_idx] = compute + longest_shortest_successor;
            }
        }
    }

    std::vector<double> priority(task_count, 0.0);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        double total = 0.0;
        std::size_t count = 0;
        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const double value = ioct[task_idx * device_count + device_idx];
            if (std::isfinite(value)) {
                total += value;
                ++count;
            }
        }
        priority[task_idx] = count == 0 ? 0.0 : total / static_cast<double>(count);
    }

    constexpr std::size_t kUnassigned = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> assignment(task_count, kUnassigned);
    std::vector<double> finish_time(task_count, 0.0);
    std::vector<double> available(device_count, 0.0);
    std::vector<std::size_t> remaining_dependencies(task_count, 0);
    std::vector<std::size_t> ready;
    ready.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        remaining_dependencies[task_idx] = ctx.dependencies[task_idx].size();
        if (remaining_dependencies[task_idx] == 0) {
            ready.push_back(task_idx);
        }
    }

    auto ready_before = [&](std::size_t lhs, std::size_t rhs) {
        const double lhs_priority = priority[lhs];
        const double rhs_priority = priority[rhs];
        if (lhs_priority == rhs_priority) {
            const auto lhs_out = ctx.successors[lhs].size();
            const auto rhs_out = ctx.successors[rhs].size();
            if (lhs_out == rhs_out) {
                return ctx.tasks[lhs].name < ctx.tasks[rhs].name;
            }
            return lhs_out > rhs_out;
        }
        return lhs_priority > rhs_priority;
    };

    auto candidate_ready = [&](std::size_t task_idx, std::size_t device_idx) {
        double ready_time = available[device_idx];
        for (const auto edge_idx : ctx.dependencies[task_idx]) {
            const auto& edge = ctx.edges[edge_idx];
            const auto pred_device = assignment[edge.src_task];
            if (pred_device == kUnassigned) {
                return kInf;
            }
            const double comm = ctx.communication_time(edge_idx, pred_device, device_idx);
            if (!std::isfinite(comm)) {
                return kInf;
            }
            ready_time = std::max(ready_time, finish_time[edge.src_task] + comm);
        }
        return ready_time;
    };

    auto successor_projection = [&](std::size_t edge_idx, std::size_t current_device, double current_finish) {
        const auto& edge = ctx.edges[edge_idx];
        double best = kInf;
        for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
            const double comm = ctx.communication_time(edge_idx, current_device, succ_device_idx);
            const double succ_tail = ioct[edge.dst_task * device_count + succ_device_idx];
            if (!std::isfinite(comm) || !std::isfinite(succ_tail)) {
                continue;
            }
            const double projected_start = std::max(available[succ_device_idx], current_finish + comm);
            best = std::min(best, projected_start + succ_tail);
        }
        return best;
    };

    std::size_t scheduled_count = 0;
    while (!ready.empty()) {
        const auto ready_it = std::min_element(ready.begin(), ready.end(), [&](std::size_t lhs, std::size_t rhs) {
            return ready_before(lhs, rhs);
        });
        const auto task_idx = *ready_it;
        ready.erase(ready_it);

        double best_metric = kInf;
        double best_finish = kInf;
        std::size_t best_device = ctx.eligible_devices[task_idx].front();

        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const double ready_time = candidate_ready(task_idx, device_idx);
            if (!std::isfinite(ready_time)) {
                continue;
            }
            const double compute = ctx.compute_time(task_idx, device_idx);
            if (!std::isfinite(compute)) {
                continue;
            }
            const double finish = ready_time + compute;

            double metric = finish;
            if (!ctx.successors[task_idx].empty()) {
                double projected_total = finish;
                std::size_t projected_count = 1;
                bool feasible = true;
                for (const auto edge_idx : ctx.successors[task_idx]) {
                    const double projected = successor_projection(edge_idx, device_idx, finish);
                    if (!std::isfinite(projected)) {
                        feasible = false;
                        break;
                    }
                    projected_total += projected;
                    ++projected_count;
                }
                if (feasible) {
                    metric = projected_total / static_cast<double>(projected_count);
                }
            }

            const auto* device = ctx.devices[device_idx];
            if (metric < best_metric ||
                (metric == best_metric &&
                 (finish < best_finish ||
                  (finish == best_finish && device->id < ctx.devices[best_device]->id)))) {
                best_metric = metric;
                best_finish = finish;
                best_device = device_idx;
            }
        }

        if (!std::isfinite(best_finish)) {
            throw std::runtime_error("No eligible device for task: " + ctx.tasks[task_idx].name);
        }

        assignment[task_idx] = best_device;
        finish_time[task_idx] = best_finish;
        available[best_device] = best_finish;
        ++scheduled_count;

        for (const auto edge_idx : ctx.successors[task_idx]) {
            const auto successor = ctx.edges[edge_idx].dst_task;
            if (remaining_dependencies[successor] == 0) {
                throw std::runtime_error("Task dependency bookkeeping underflow");
            }
            --remaining_dependencies[successor];
            if (remaining_dependencies[successor] == 0) {
                ready.push_back(successor);
            }
        }
    }

    if (scheduled_count != task_count) {
        throw std::runtime_error("Task graph contains unscheduled tasks");
    }

    MappingPlan plan;
    plan.assignments.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        plan.assignments.emplace(ctx.tasks[task_idx].name, ctx.devices[assignment[task_idx]]->id);
    }

    return plan;
}

MappingPlan PeftMapper::map(const TaskGraph& graph, const hardware_topology::HardwareTopology& topology) const {
    if (topology.compute_devices().empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }
    ScheduleContext ctx(graph, topology);

    const std::size_t task_count = ctx.tasks.size();
    const std::size_t device_count = ctx.devices.size();
    const auto kInf = std::numeric_limits<double>::infinity();

    std::vector<double> oct(task_count * device_count, kInf);
    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        for (const auto device_idx : ctx.eligible_devices[pos]) {
            double value = 0.0;
            for (const auto edge_idx : ctx.successors[pos]) {
                const auto& edge = ctx.edges[edge_idx];
                double best_successor = kInf;

                for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
                    const double comm = ctx.communication_time(edge_idx, device_idx, succ_device_idx);
                    const double succ_cost = ctx.compute_time(edge.dst_task, succ_device_idx);
                    const double succ_oct = oct[edge.dst_task * device_count + succ_device_idx];
                    const double candidate = comm + succ_cost + succ_oct;
                    if (candidate < best_successor) {
                        best_successor = candidate;
                    }
                }
                value = std::max(value, best_successor);
            }
            oct[pos * device_count + device_idx] = value;
        }
    }

    constexpr std::size_t kUnassigned = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> assignment(task_count, kUnassigned);
    std::vector<double> finish_time(task_count, 0.0);
    std::vector<double> available(device_count, 0.0);

    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        double best_metric = kInf;
        double best_finish = kInf;
        std::size_t best_device = ctx.eligible_devices[task_idx].front();

        auto candidate_ready = [&](std::size_t device_idx) {
            double ready = available[device_idx];

            for (const auto edge_idx : ctx.dependencies[task_idx]) {
                const auto& edge = ctx.edges[edge_idx];
                const auto pred_device = assignment[edge.src_task];
                if (pred_device == kUnassigned) {
                    throw std::runtime_error("Task dependencies must be assigned before successors");
                }

                const double comm = ctx.communication_time(edge_idx, pred_device, device_idx);
                if (!std::isfinite(comm)) {
                    return kInf;
                }
                ready = std::max(ready, finish_time[edge.src_task] + comm);
            }
            return ready;
        };

        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const auto* device = ctx.devices[device_idx];
            const double ready = candidate_ready(device_idx);
            if (!std::isfinite(ready)) {
                continue;
            }

            const double exec = ctx.compute_time(task_idx, device_idx);
            const double eft = ready + exec;
            const double metric = eft + oct[task_idx * device_count + device_idx];
            if (metric < best_metric ||
                (metric == best_metric &&
                 (eft < best_finish || (eft == best_finish && device->id < ctx.devices[best_device]->id)))) {
                best_metric = metric;
                best_finish = eft;
                best_device = device_idx;
            }
        }

        if (!std::isfinite(best_finish)) {
            throw std::runtime_error("No eligible device for task: " + ctx.tasks[task_idx].name);
        }

        assignment[task_idx] = best_device;
        finish_time[task_idx] = best_finish;
        available[best_device] = best_finish;
    }

    MappingPlan plan;
    plan.assignments.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        plan.assignments.emplace(ctx.tasks[task_idx].name, ctx.devices[assignment[task_idx]]->id);
    }

    return plan;
}

MappingPlan PeftLcMapper::map(const TaskGraph& graph, const hardware_topology::HardwareTopology& topology) const {
    if (topology.compute_devices().empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }
    if (!should_use_peft_lc_pruning(graph, topology)) {
        return map_peft_exact(graph, topology, false);
    }
    const auto communication_cache_mode = should_use_peft_lc_full_comm_cache(graph, topology)
                                              ? CommunicationCacheMode::Full
                                              : CommunicationCacheMode::Lazy;
    ScheduleContext ctx(graph, topology, false, communication_cache_mode);

    const std::size_t task_count = ctx.tasks.size();
    const std::size_t device_count = ctx.devices.size();
    const auto kInf = std::numeric_limits<double>::infinity();
    const bool peft_lc_locality_pruning = peft_lc_locality_neighbor_limit(device_count) < device_count;
    const auto peft_lc_local_destinations = build_peft_lc_local_destinations(ctx);
    std::vector<unsigned char> task_fully_eligible(task_count, 0);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        task_fully_eligible[task_idx] = ctx.eligible_devices[task_idx].size() == device_count ? 1 : 0;
    }

    constexpr std::size_t kNoDevice = std::numeric_limits<std::size_t>::max();
    std::vector<double> oct(task_count * device_count, kInf);
    std::vector<std::size_t> preferred_devices(task_count * kPeftLcPreferredDevices, kNoDevice);
    std::vector<unsigned char> preferred_device_count(task_count, 0);
    std::vector<double> best_continuation(task_count, kInf);

    auto refresh_preferred_devices = [&](std::size_t task_idx) {
        best_continuation[task_idx] = kInf;

        std::array<std::pair<double, std::size_t>, kPeftLcPreferredDevices> best{};
        for (auto& entry : best) {
            entry = {kInf, kNoDevice};
        }
        std::size_t best_count = 0;
        auto better = [&](const auto& lhs, const auto& rhs) {
            if (lhs.first == rhs.first) {
                return ctx.devices[lhs.second]->id < ctx.devices[rhs.second]->id;
            }
            return lhs.first < rhs.first;
        };

        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const double compute = ctx.compute_time(task_idx, device_idx);
            const double tail = oct[task_idx * device_count + device_idx];
            if (!std::isfinite(compute) || !std::isfinite(tail)) {
                continue;
            }
            const double score = compute + tail;
            best_continuation[task_idx] = std::min(best_continuation[task_idx], score);
            const auto entry = std::make_pair(score, device_idx);

            std::size_t insert_at = 0;
            while (insert_at < best_count && !better(entry, best[insert_at])) {
                ++insert_at;
            }
            if (insert_at >= kPeftLcPreferredDevices) {
                continue;
            }

            const std::size_t new_count = std::min(best_count + 1, kPeftLcPreferredDevices);
            for (std::size_t i = new_count - 1; i > insert_at; --i) {
                best[i] = best[i - 1];
            }
            best[insert_at] = entry;
            best_count = new_count;
        }

        const auto base = task_idx * kPeftLcPreferredDevices;
        preferred_device_count[task_idx] = static_cast<unsigned char>(best_count);
        for (std::size_t i = 0; i < kPeftLcPreferredDevices; ++i) {
            preferred_devices[base + i] = i < best_count ? best[i].second : kNoDevice;
        }
    };

    auto for_each_preferred_device = [&](std::size_t task_idx, const auto& visitor) {
        const auto base = task_idx * kPeftLcPreferredDevices;
        const auto count = static_cast<std::size_t>(preferred_device_count[task_idx]);
        for (std::size_t i = 0; i < count; ++i) {
            const auto device_idx = preferred_devices[base + i];
            if (device_idx != kNoDevice) {
                visitor(device_idx);
            }
        }
    };

    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        for (const auto device_idx : ctx.eligible_devices[pos]) {
            double value = 0.0;
            for (const auto edge_idx : ctx.successors[pos]) {
                const auto& edge = ctx.edges[edge_idx];
                double best_successor = kInf;

                auto consider_successor_device = [&](std::size_t succ_device_idx, bool check_eligible) {
                    if (check_eligible && !ctx.is_eligible(edge.dst_task, succ_device_idx)) {
                        return;
                    }
                    const double comm = ctx.communication_time(edge_idx, device_idx, succ_device_idx);
                    const double succ_cost = ctx.compute_time(edge.dst_task, succ_device_idx);
                    const double succ_oct = oct[edge.dst_task * device_count + succ_device_idx];
                    if (!std::isfinite(comm) || !std::isfinite(succ_cost) || !std::isfinite(succ_oct)) {
                        return;
                    }
                    best_successor = std::min(best_successor, comm + succ_cost + succ_oct);
                };

                if (edge.collective) {
                    const double comm = ctx.communication_time(edge_idx, device_idx, device_idx);
                    const double continuation = best_continuation[edge.dst_task];
                    if (std::isfinite(comm) && std::isfinite(continuation)) {
                        best_successor = comm + continuation;
                    }
                } else if (peft_lc_locality_pruning) {
                    const bool fully_eligible_successor = task_fully_eligible[edge.dst_task] != 0;
                    for (const auto succ_device_idx : peft_lc_local_destinations[device_idx]) {
                        consider_successor_device(succ_device_idx, !fully_eligible_successor);
                    }
                    for_each_preferred_device(edge.dst_task, [&](std::size_t succ_device_idx) {
                        consider_successor_device(succ_device_idx, false);
                    });
                    if (!std::isfinite(best_successor)) {
                        for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
                            consider_successor_device(succ_device_idx, false);
                        }
                    }
                } else {
                    for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
                        consider_successor_device(succ_device_idx, false);
                    }
                }

                value = std::max(value, best_successor);
            }
            oct[pos * device_count + device_idx] = value;
        }
        refresh_preferred_devices(pos);
    }

    constexpr std::size_t kUnassigned = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> assignment(task_count, kUnassigned);
    std::vector<double> finish_time(task_count, 0.0);
    std::vector<double> available(device_count, 0.0);
    std::vector<std::size_t> candidate_devices;
    std::vector<unsigned char> candidate_mask(device_count, 0);
    std::vector<std::size_t> local_hit_count(device_count, 0);
    std::vector<std::size_t> local_touched;
    candidate_devices.reserve(device_count);
    local_touched.reserve(device_count);

    auto clear_candidate_devices = [&]() {
        for (const auto device_idx : candidate_devices) {
            candidate_mask[device_idx] = 0;
        }
        candidate_devices.clear();
    };

    auto add_candidate_device = [&](std::size_t task_idx, std::size_t device_idx) {
        if (device_idx >= device_count || candidate_mask[device_idx] != 0) {
            return;
        }
        if (!ctx.is_eligible(task_idx, device_idx)) {
            return;
        }
        candidate_mask[device_idx] = 1;
        candidate_devices.push_back(device_idx);
    };

    auto add_low_load_candidates = [&](std::size_t task_idx) {
        std::array<std::pair<double, std::size_t>, kDependencyLocalitySpilloverDevices> best{};
        for (auto& entry : best) {
            entry = {kInf, kNoDevice};
        }
        std::size_t best_count = 0;

        auto better = [&](const auto& lhs, const auto& rhs) {
            if (lhs.first == rhs.first) {
                return ctx.devices[lhs.second]->id < ctx.devices[rhs.second]->id;
            }
            return lhs.first < rhs.first;
        };

        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            if (candidate_mask[device_idx] != 0) {
                continue;
            }
            const double compute = ctx.compute_time(task_idx, device_idx);
            const double tail = oct[task_idx * device_count + device_idx];
            if (!std::isfinite(compute) || !std::isfinite(tail)) {
                continue;
            }
            const auto entry = std::make_pair(available[device_idx] + compute + tail, device_idx);

            std::size_t insert_at = 0;
            while (insert_at < best_count && !better(entry, best[insert_at])) {
                ++insert_at;
            }
            if (insert_at >= best.size()) {
                continue;
            }

            const std::size_t new_count = std::min(best_count + 1, best.size());
            for (std::size_t i = new_count - 1; i > insert_at; --i) {
                best[i] = best[i - 1];
            }
            best[insert_at] = entry;
            best_count = new_count;
        }

        for (std::size_t i = 0; i < best_count; ++i) {
            add_candidate_device(task_idx, best[i].second);
        }
    };

    auto build_candidate_devices = [&](std::size_t task_idx) {
        clear_candidate_devices();

        if (peft_lc_locality_pruning) {
            local_touched.clear();
            std::size_t dependency_count = 0;
            for (const auto edge_idx : ctx.dependencies[task_idx]) {
                const auto& edge = ctx.edges[edge_idx];
                if (edge.collective) {
                    continue;
                }
                const auto pred_device = assignment[edge.src_task];
                if (pred_device == kUnassigned) {
                    throw std::runtime_error("Task dependencies must be assigned before successors");
                }
                ++dependency_count;
                for (const auto device_idx : peft_lc_local_destinations[pred_device]) {
                    if (!ctx.is_eligible(task_idx, device_idx)) {
                        continue;
                    }
                    if (local_hit_count[device_idx] == 0) {
                        local_touched.push_back(device_idx);
                    }
                    ++local_hit_count[device_idx];
                }
            }

            if (dependency_count > 0) {
                for (const auto device_idx : local_touched) {
                    if (local_hit_count[device_idx] == dependency_count) {
                        add_candidate_device(task_idx, device_idx);
                    }
                }
                for (const auto device_idx : local_touched) {
                    local_hit_count[device_idx] = 0;
                }
            }
        }

        for_each_preferred_device(task_idx, [&](std::size_t device_idx) {
            add_candidate_device(task_idx, device_idx);
        });
        add_low_load_candidates(task_idx);
    };

    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        double best_metric = kInf;
        double best_finish = kInf;
        std::size_t best_device = ctx.eligible_devices[task_idx].front();

        auto candidate_ready = [&](std::size_t device_idx) {
            double ready = available[device_idx];

            for (const auto edge_idx : ctx.dependencies[task_idx]) {
                const auto& edge = ctx.edges[edge_idx];
                const auto pred_device = assignment[edge.src_task];
                if (pred_device == kUnassigned) {
                    throw std::runtime_error("Task dependencies must be assigned before successors");
                }

                const double comm = ctx.communication_time(edge_idx, pred_device, device_idx);
                if (!std::isfinite(comm)) {
                    return kInf;
                }
                ready = std::max(ready, finish_time[edge.src_task] + comm);
            }
            return ready;
        };

        auto consider_device = [&](std::size_t device_idx) {
            const auto* device = ctx.devices[device_idx];
            const double ready = candidate_ready(device_idx);
            if (!std::isfinite(ready)) {
                return;
            }

            const double exec = ctx.compute_time(task_idx, device_idx);
            const double eft = ready + exec;
            const double metric = eft + oct[task_idx * device_count + device_idx];
            if (metric < best_metric ||
                (metric == best_metric &&
                 (eft < best_finish || (eft == best_finish && device->id < ctx.devices[best_device]->id)))) {
                best_metric = metric;
                best_finish = eft;
                best_device = device_idx;
            }
        };

        build_candidate_devices(task_idx);
        if (candidate_devices.empty()) {
            for (const auto device_idx : ctx.eligible_devices[task_idx]) {
                consider_device(device_idx);
            }
        } else {
            for (const auto device_idx : candidate_devices) {
                consider_device(device_idx);
            }
            if (!std::isfinite(best_finish)) {
                for (const auto device_idx : ctx.eligible_devices[task_idx]) {
                    consider_device(device_idx);
                }
            }
        }
        clear_candidate_devices();

        if (!std::isfinite(best_finish)) {
            throw std::runtime_error("No eligible device for task: " + ctx.tasks[task_idx].name);
        }

        assignment[task_idx] = best_device;
        finish_time[task_idx] = best_finish;
        available[best_device] = best_finish;
    }

    MappingPlan plan;
    plan.assignments.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        plan.assignments.emplace(ctx.tasks[task_idx].name, ctx.devices[assignment[task_idx]]->id);
    }

    return plan;
}

MappingPlan HoftMapper::map(const TaskGraph& graph, const hardware_topology::HardwareTopology& topology) const {
    if (topology.compute_devices().empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }
    ScheduleContext ctx(graph, topology);

    const std::size_t task_count = ctx.tasks.size();
    const std::size_t device_count = ctx.devices.size();
    const double kInf = std::numeric_limits<double>::infinity();

    std::unordered_map<std::string, std::size_t> type_index;
    std::vector<std::string> type_names;
    std::vector<std::size_t> device_type(device_count, 0);
    std::vector<std::vector<std::size_t>> devices_by_type;
    for (std::size_t device_idx = 0; device_idx < device_count; ++device_idx) {
        const auto type_name = normalized_device_type(ctx.devices[device_idx]);
        auto it = type_index.find(type_name);
        if (it == type_index.end()) {
            const std::size_t type_idx = type_names.size();
            it = type_index.emplace(type_name, type_idx).first;
            type_names.push_back(type_name);
            devices_by_type.emplace_back();
        }
        device_type[device_idx] = it->second;
        devices_by_type[it->second].push_back(device_idx);
    }
    const std::size_t type_count = type_names.size();

    std::vector<double> type_compute(task_count * type_count, kInf);
    std::vector<unsigned char> task_type_eligible(task_count * type_count, 0);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const auto type_idx = device_type[device_idx];
            const double time = ctx.compute_time(task_idx, device_idx);
            if (!std::isfinite(time)) {
                continue;
            }
            auto& current = type_compute[task_idx * type_count + type_idx];
            current = std::min(current, time);
            task_type_eligible[task_idx * type_count + type_idx] = 1;
        }
    }

    auto type_is_eligible = [&](std::size_t task_idx, std::size_t type_idx) {
        return task_type_eligible[task_idx * type_count + type_idx] != 0;
    };

    std::vector<double> type_comm(ctx.edges.size() * type_count * type_count, kInf);
    auto type_comm_slot = [&](std::size_t edge_idx, std::size_t src_type, std::size_t dst_type) {
        return edge_idx * type_count * type_count + src_type * type_count + dst_type;
    };
    for (std::size_t edge_idx = 0; edge_idx < ctx.edges.size(); ++edge_idx) {
        const auto& edge = ctx.edges[edge_idx];
        for (std::size_t src_type = 0; src_type < type_count; ++src_type) {
            if (!type_is_eligible(edge.src_task, src_type)) {
                continue;
            }
            for (std::size_t dst_type = 0; dst_type < type_count; ++dst_type) {
                if (!type_is_eligible(edge.dst_task, dst_type)) {
                    continue;
                }
                double best = kInf;
                if (edge.collective) {
                    best = ctx.communication_time(edge_idx, devices_by_type[src_type].front(),
                                                  devices_by_type[dst_type].front());
                } else {
                    for (const auto src_device : devices_by_type[src_type]) {
                        if (!ctx.is_eligible(edge.src_task, src_device)) {
                            continue;
                        }
                        for (const auto dst_device : devices_by_type[dst_type]) {
                            if (!ctx.is_eligible(edge.dst_task, dst_device)) {
                                continue;
                            }
                            const double comm = ctx.communication_time(edge_idx, src_device, dst_device);
                            if (std::isfinite(comm)) {
                                best = std::min(best, comm);
                            }
                        }
                    }
                }
                if (!std::isfinite(best)) {
                    for (const auto src_device : devices_by_type[src_type]) {
                        if (!ctx.is_eligible(edge.src_task, src_device)) {
                            continue;
                        }
                        for (const auto dst_device : devices_by_type[dst_type]) {
                            if (!ctx.is_eligible(edge.dst_task, dst_device)) {
                                continue;
                            }
                            const double comm = ctx.exact_communication_time(edge_idx, topology, src_device, dst_device);
                            if (std::isfinite(comm)) {
                                best = std::min(best, comm);
                            }
                        }
                    }
                }
                type_comm[type_comm_slot(edge_idx, src_type, dst_type)] = best;
            }
        }
    }

    std::vector<double> oft(task_count * type_count, kInf);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        for (std::size_t type_idx = 0; type_idx < type_count; ++type_idx) {
            const double compute = type_compute[task_idx * type_count + type_idx];
            if (!std::isfinite(compute)) {
                continue;
            }

            double head = 0.0;
            bool feasible = true;
            for (const auto edge_idx : ctx.dependencies[task_idx]) {
                const auto& edge = ctx.edges[edge_idx];
                double best_pred = kInf;
                for (std::size_t pred_type = 0; pred_type < type_count; ++pred_type) {
                    const double pred_oft = oft[edge.src_task * type_count + pred_type];
                    if (!std::isfinite(pred_oft)) {
                        continue;
                    }
                    const double comm = type_comm[type_comm_slot(edge_idx, pred_type, type_idx)];
                    if (std::isfinite(comm)) {
                        best_pred = std::min(best_pred, pred_oft + comm);
                    }
                }
                if (!std::isfinite(best_pred)) {
                    feasible = false;
                    break;
                }
                head = std::max(head, best_pred);
            }
            if (feasible) {
                oft[task_idx * type_count + type_idx] = head + compute;
            }
        }
    }

    std::vector<double> weight(task_count, 1.0);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        double min_oft = kInf;
        double max_oft = -kInf;
        double min_compute = kInf;
        for (std::size_t type_idx = 0; type_idx < type_count; ++type_idx) {
            const double value = oft[task_idx * type_count + type_idx];
            if (std::isfinite(value)) {
                if (value < min_oft) {
                    min_oft = value;
                }
                max_oft = std::max(max_oft, value);
            }
            const double compute = type_compute[task_idx * type_count + type_idx];
            if (!std::isfinite(min_oft) && std::isfinite(compute) && compute < min_compute) {
                min_compute = compute;
            }
        }
        if (std::isfinite(min_oft) && std::isfinite(max_oft) && min_oft > 1e-18) {
            weight[task_idx] = std::max(1.0, max_oft / min_oft);
        }
    }

    std::vector<double> device_tail(task_count * device_count, kInf);
    std::vector<std::size_t> successor_candidates;
    std::vector<unsigned char> successor_mask(device_count, 0);
    std::vector<double> collective_tail_successor_best(ctx.edges.size(), kInf);
    std::vector<unsigned char> collective_tail_successor_best_ready(ctx.edges.size(), 0);
    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        for (const auto device_idx : ctx.eligible_devices[pos]) {
            double tail = 0.0;
            bool feasible = true;
            for (const auto edge_idx : ctx.successors[pos]) {
                const auto& edge = ctx.edges[edge_idx];
                double best_successor = kInf;

                auto consider_successor_device = [&](std::size_t succ_device_idx, bool exact_comm) {
                    if (!ctx.is_eligible(edge.dst_task, succ_device_idx)) {
                        return;
                    }
                    const double succ_compute = ctx.compute_time(edge.dst_task, succ_device_idx);
                    const double succ_tail = device_tail[edge.dst_task * device_count + succ_device_idx];
                    if (!std::isfinite(succ_compute) || !std::isfinite(succ_tail)) {
                        return;
                    }
                    const double comm = exact_comm
                                            ? ctx.exact_communication_time(edge_idx, topology, device_idx, succ_device_idx)
                                            : ctx.communication_time(edge_idx, device_idx, succ_device_idx);
                    if (!std::isfinite(comm)) {
                        return;
                    }
                    best_successor = std::min(best_successor, comm + succ_compute + succ_tail);
                };

                if (edge.collective) {
                    if (collective_tail_successor_best_ready[edge_idx] == 0) {
                        const double comm = ctx.communication_time(edge_idx, device_idx, device_idx);
                        double best = kInf;
                        for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
                            const double succ_compute = ctx.compute_time(edge.dst_task, succ_device_idx);
                            const double succ_tail = device_tail[edge.dst_task * device_count + succ_device_idx];
                            if (std::isfinite(succ_compute) && std::isfinite(succ_tail)) {
                                best = std::min(best, succ_compute + succ_tail);
                            }
                        }
                        collective_tail_successor_best[edge_idx] =
                            std::isfinite(comm) && std::isfinite(best) ? comm + best : kInf;
                        collective_tail_successor_best_ready[edge_idx] = 1;
                    }
                    best_successor = collective_tail_successor_best[edge_idx];
                } else if (ctx.locality_pruning) {
                    std::fill(successor_mask.begin(), successor_mask.end(), 0);
                    successor_candidates.clear();
                    for (const auto succ_device_idx : ctx.local_destinations_by_source[device_idx]) {
                        if (ctx.is_eligible(edge.dst_task, succ_device_idx)) {
                            ctx.add_candidate(succ_device_idx, successor_candidates, successor_mask);
                        }
                    }
                    const std::size_t first_spillover_candidate = successor_candidates.size();
                    ctx.add_best_scored_candidates(edge.dst_task,
                                                   kDependencyLocalitySpilloverDevices,
                                                   successor_candidates,
                                                   successor_mask,
                                                   [&](std::size_t succ_device_idx) {
                                                       return ctx.compute_time(edge.dst_task, succ_device_idx) +
                                                              device_tail[edge.dst_task * device_count + succ_device_idx];
                                                   });
                    for (std::size_t i = 0; i < successor_candidates.size(); ++i) {
                        consider_successor_device(successor_candidates[i], i >= first_spillover_candidate);
                    }
                } else {
                    for (const auto succ_device_idx : ctx.eligible_devices[edge.dst_task]) {
                        consider_successor_device(succ_device_idx, false);
                    }
                }

                if (!std::isfinite(best_successor)) {
                    feasible = false;
                    break;
                }
                tail = std::max(tail, best_successor);
            }
            if (feasible) {
                device_tail[pos * device_count + device_idx] = tail;
            }
        }
    }

    std::vector<double> rank(task_count, 0.0);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        double best_path = kInf;
        for (const auto device_idx : ctx.eligible_devices[task_idx]) {
            const double compute = ctx.compute_time(task_idx, device_idx);
            const double tail = device_tail[task_idx * device_count + device_idx];
            if (std::isfinite(compute) && std::isfinite(tail)) {
                best_path = std::min(best_path, compute + tail);
            }
        }
        rank[task_idx] = std::isfinite(best_path) ? best_path : weight[task_idx];
    }

    std::vector<std::size_t> order;
    order.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        order.push_back(task_idx);
    }
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        if (rank[lhs] == rank[rhs]) {
            return lhs < rhs;
        }
        return rank[lhs] > rank[rhs];
    });

    constexpr std::size_t kUnassigned = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> assignment(task_count, kUnassigned);
    std::vector<double> finish_time(task_count, 0.0);
    std::vector<double> available(device_count, 0.0);
    std::vector<std::size_t> nearby_candidates;
    std::vector<unsigned char> candidate_mask(device_count, 0);

    struct DeviceEval {
        bool valid{false};
        double ready{0.0};
        double finish{std::numeric_limits<double>::infinity()};
        double metric{std::numeric_limits<double>::infinity()};
    };

    for (const auto task_idx : order) {
        std::vector<DeviceEval> evaluated(device_count);
        double best_metric = kInf;
        double best_finish = kInf;
        std::size_t best_device = ctx.eligible_devices[task_idx].front();
        double fallback_finish = kInf;
        std::size_t fallback_device = best_device;

        auto candidate_ready = [&](std::size_t device_idx, bool exact_comm) {
            double ready = available[device_idx];
            for (const auto edge_idx : ctx.dependencies[task_idx]) {
                const auto& edge = ctx.edges[edge_idx];
                const auto pred_device = assignment[edge.src_task];
                if (pred_device == kUnassigned) {
                    throw std::runtime_error("Task dependencies must be assigned before successors");
                }
                const double comm = exact_comm ? ctx.exact_communication_time(edge_idx, topology, pred_device, device_idx)
                                               : ctx.communication_time(edge_idx, pred_device, device_idx);
                if (!std::isfinite(comm)) {
                    return kInf;
                }
                ready = std::max(ready, finish_time[edge.src_task] + comm);
            }
            return ready;
        };

        auto consider_device = [&](std::size_t device_idx, bool exact_comm) {
            if (!ctx.is_eligible(task_idx, device_idx)) {
                return;
            }
            const double ready = candidate_ready(device_idx, exact_comm);
            const double exec = ctx.compute_time(task_idx, device_idx);
            if (!std::isfinite(ready) || !std::isfinite(exec)) {
                return;
            }
            const double finish = ready + exec;
            const double tail = device_tail[task_idx * device_count + device_idx];
            const double metric = std::isfinite(tail) ? finish + tail : kInf;
            evaluated[device_idx] = DeviceEval{true, ready, finish, metric};

            const auto* device = ctx.devices[device_idx];
            if (finish < fallback_finish ||
                (finish == fallback_finish && device->id < ctx.devices[fallback_device]->id)) {
                fallback_finish = finish;
                fallback_device = device_idx;
            }
            if (metric < best_metric ||
                (metric == best_metric &&
                 (finish < best_finish || (finish == best_finish && device->id < ctx.devices[best_device]->id)))) {
                best_metric = metric;
                best_finish = finish;
                best_device = device_idx;
            }
        };

        auto consider_candidates = [&](const std::vector<std::size_t>& candidates, bool exact_comm) {
            for (const auto device_idx : candidates) {
                consider_device(device_idx, exact_comm);
            }
        };

        if (ctx.locality_pruning) {
            std::fill(candidate_mask.begin(), candidate_mask.end(), 0);
            nearby_candidates.clear();
            ctx.collect_nearby_dependency_candidates(task_idx, assignment, nearby_candidates, candidate_mask);
            ctx.add_best_scored_candidates(task_idx,
                                           kDependencyLocalitySpilloverDevices,
                                           nearby_candidates,
                                           candidate_mask,
                                           [&](std::size_t device_idx) {
                                               return available[device_idx];
                                           });
            ctx.add_best_scored_candidates(task_idx,
                                           kDependencyLocalitySpilloverDevices,
                                           nearby_candidates,
                                           candidate_mask,
                                           [&](std::size_t device_idx) {
                                               return ctx.compute_time(task_idx, device_idx);
                                           });
            ctx.add_best_scored_candidates(task_idx,
                                           kDependencyLocalitySpilloverDevices,
                                           nearby_candidates,
                                           candidate_mask,
                                           [&](std::size_t device_idx) {
                                               return ctx.compute_time(task_idx, device_idx) +
                                                      device_tail[task_idx * device_count + device_idx];
                                           });
            consider_candidates(nearby_candidates, false);
        } else {
            consider_candidates(ctx.eligible_devices[task_idx], false);
        }
        if (!std::isfinite(fallback_finish) || !std::isfinite(best_metric)) {
            consider_candidates(ctx.eligible_devices[task_idx], true);
        }
        if (!std::isfinite(fallback_finish)) {
            throw std::runtime_error("No eligible device for task: " + ctx.tasks[task_idx].name);
        }

        std::size_t selected_device = std::isfinite(best_metric) ? best_device : fallback_device;

        if (!evaluated[selected_device].valid) {
            consider_device(selected_device, true);
        }
        if (!evaluated[selected_device].valid) {
            throw std::runtime_error("No feasible HOFT device for task: " + ctx.tasks[task_idx].name);
        }
        assignment[task_idx] = selected_device;
        finish_time[task_idx] = evaluated[selected_device].finish;
        available[selected_device] = evaluated[selected_device].finish;
    }

    MappingPlan plan;
    plan.assignments.reserve(task_count);
    for (std::size_t task_idx = 0; task_idx < task_count; ++task_idx) {
        plan.assignments.emplace(ctx.tasks[task_idx].name, ctx.devices[assignment[task_idx]]->id);
    }
    try {
        const double hoft_makespan = estimate_makespan_seconds(graph, plan, topology);
        AeftMapper aeft_fallback;
        auto fallback_plan = aeft_fallback.map(graph, topology);
        const double fallback_makespan = estimate_makespan_seconds(graph, fallback_plan, topology);
        const double epsilon = std::max(1e-18, std::abs(fallback_makespan) * 1e-12);
        if (std::isfinite(fallback_makespan) &&
            (!std::isfinite(hoft_makespan) || fallback_makespan + epsilon < hoft_makespan)) {
            return fallback_plan;
        }
    } catch (const std::exception&) {
        // HOFT remains a heuristic mapper; keep its own plan if the guard cannot
        // evaluate or produce an AEFT fallback.
    }
    return plan;
}

ExhaustiveMapper::ExhaustiveMapper(bool force_large_search,
                                   bool branch_and_bound,
                                   bool seed_upper_bound)
    : force_large_search_(force_large_search),
      branch_and_bound_(branch_and_bound),
      seed_upper_bound_(seed_upper_bound) {}

MappingPlan ExhaustiveMapper::map(const TaskGraph& graph,
                                  const hardware_topology::HardwareTopology& topology) const {
    constexpr std::uint64_t kMaxSearchCombinations = 1000000;
    const auto& devices = topology.compute_devices();
    if (devices.empty()) {
        throw std::runtime_error("Topology has no compute devices");
    }

    std::vector<const hardware_topology::Device*> ordered_devices = devices;
    std::sort(ordered_devices.begin(),
              ordered_devices.end(),
              [](const auto* lhs, const auto* rhs) {
                  return lhs->id < rhs->id;
              });

    const auto& topo = graph.topological_order();
    std::vector<std::vector<std::size_t>> eligible_by_task;
    eligible_by_task.reserve(topo.size());
    std::uint64_t combinations = 1;
    bool combinations_over_limit = false;

    for (const auto& task : topo) {
        std::vector<std::size_t> eligible;
        const auto pin = pinned_device_tag(task);
        for (std::size_t i = 0; i < ordered_devices.size(); ++i) {
            const auto* device = ordered_devices[i];
            if (pin.has_value() && device->id != *pin) {
                continue;
            }
            if (!is_device_allowed_for_task(task, device)) {
                continue;
            }
            eligible.push_back(i);
        }
        if (pin.has_value() && eligible.empty()) {
            const auto* target = topology.device(*pin);
            if (target == nullptr || !target->compute_capable) {
                throw std::runtime_error("Pinned compute device not found: " + *pin);
            }
            throw std::runtime_error("Pinned device is not supported by task profile: " + task.name);
        }
        if (eligible.empty()) {
            throw std::runtime_error("No eligible device for task: " + task.name);
        }
        if (!combinations_over_limit) {
            const auto eligible_count = static_cast<std::uint64_t>(eligible.size());
            if (combinations > kMaxSearchCombinations / eligible_count) {
                if (!force_large_search_) {
                    throw std::runtime_error("Exhaustive mapper search space too large: more than " +
                                             std::to_string(kMaxSearchCombinations) +
                                             " combinations; use --force-exhaustive to run anyway");
                }
                combinations_over_limit = true;
            } else {
                combinations *= eligible_count;
            }
        }
        eligible_by_task.push_back(std::move(eligible));
    }

    std::unordered_map<std::string, std::size_t> task_index;
    task_index.reserve(topo.size());
    for (std::size_t i = 0; i < topo.size(); ++i) {
        task_index.emplace(topo[i].name, i);
    }

    const double kInf = std::numeric_limits<double>::infinity();
    double best_makespan = kInf;
    std::vector<std::size_t> best_assignment(topo.size(), 0);
    std::vector<std::size_t> current_assignment(topo.size(), 0);
    std::vector<double> available(ordered_devices.size(), 0.0);
    std::vector<double> finish_time(topo.size(), 0.0);
    bool have_best_assignment = false;

    struct ExhaustiveSearchStats {
        std::uint64_t assignment_evaluations{0};
        std::uint64_t best_updates{0};
        std::uint64_t initial_task_device_choices{0};
        std::uint64_t prepruned_task_device_choices{0};
        std::uint64_t remaining_task_device_choices{0};
        std::uint64_t nodes_visited{0};
        std::uint64_t leaves_reached{0};
        std::uint64_t branch_attempts{0};
        std::uint64_t branch_accepted{0};
        std::uint64_t pruned_by_current_makespan{0};
        std::uint64_t pruned_by_optimistic_bound{0};
        std::uint64_t pruned_by_capacity_bound{0};
        std::uint64_t pruned_by_candidate_makespan{0};
        std::uint64_t pruned_by_candidate_bound{0};
        std::uint64_t infeasible_candidates{0};
        std::uint64_t dynamic_sort_nodes{0};
        std::uint64_t prefixes_created{0};
        std::uint64_t worker_threads{0};
        std::uint64_t exact_assignment_cache_hits{0};
        std::uint64_t exact_assignment_cache_records{0};
        std::uint64_t exact_state_cache_queries{0};
        std::uint64_t exact_state_cache_hits{0};
        std::uint64_t exact_state_cache_records{0};
        std::uint64_t exact_state_cache_saturated{0};
        std::size_t max_depth_reached{0};
    };
    ExhaustiveSearchStats stats;
    const bool collect_exhaustive_stats = std::getenv("TERRAPOD_EXHAUSTIVE_STATS") != nullptr;

    auto add_stats = [](ExhaustiveSearchStats& dst, const ExhaustiveSearchStats& src) {
        dst.assignment_evaluations += src.assignment_evaluations;
        dst.best_updates += src.best_updates;
        dst.initial_task_device_choices += src.initial_task_device_choices;
        dst.prepruned_task_device_choices += src.prepruned_task_device_choices;
        dst.remaining_task_device_choices += src.remaining_task_device_choices;
        dst.nodes_visited += src.nodes_visited;
        dst.leaves_reached += src.leaves_reached;
        dst.branch_attempts += src.branch_attempts;
        dst.branch_accepted += src.branch_accepted;
        dst.pruned_by_current_makespan += src.pruned_by_current_makespan;
        dst.pruned_by_optimistic_bound += src.pruned_by_optimistic_bound;
        dst.pruned_by_capacity_bound += src.pruned_by_capacity_bound;
        dst.pruned_by_candidate_makespan += src.pruned_by_candidate_makespan;
        dst.pruned_by_candidate_bound += src.pruned_by_candidate_bound;
        dst.infeasible_candidates += src.infeasible_candidates;
        dst.dynamic_sort_nodes += src.dynamic_sort_nodes;
        dst.prefixes_created += src.prefixes_created;
        dst.worker_threads += src.worker_threads;
        dst.exact_assignment_cache_hits += src.exact_assignment_cache_hits;
        dst.exact_assignment_cache_records += src.exact_assignment_cache_records;
        dst.exact_state_cache_queries += src.exact_state_cache_queries;
        dst.exact_state_cache_hits += src.exact_state_cache_hits;
        dst.exact_state_cache_records += src.exact_state_cache_records;
        dst.exact_state_cache_saturated += src.exact_state_cache_saturated;
        dst.max_depth_reached = std::max(dst.max_depth_reached, src.max_depth_reached);
    };

    auto parse_size_env = [](const char* name, std::size_t fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') {
            return fallback;
        }
        char* end = nullptr;
        const auto parsed = std::strtoull(value, &end, 10);
        if (end == value || parsed == 0) {
            return fallback;
        }
        return static_cast<std::size_t>(parsed);
    };

    auto parse_bool_env = [](const char* name, bool fallback) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') {
            return fallback;
        }
        std::string normalized(value);
        for (char& ch : normalized) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        return normalized != "0" && normalized != "false" && normalized != "off" &&
               normalized != "no";
    };

    const std::size_t exhaustive_progress_interval_s =
        parse_size_env("TERRAPOD_EXHAUSTIVE_PROGRESS_INTERVAL_SEC",
                       collect_exhaustive_stats ? 30 : 0);
    const bool report_exhaustive_progress = exhaustive_progress_interval_s > 0;

    struct ExhaustiveProgressCounters {
        std::atomic<std::uint64_t> nodes_visited{0};
        std::atomic<std::uint64_t> leaves_reached{0};
        std::atomic<std::uint64_t> branch_attempts{0};
        std::atomic<std::uint64_t> branch_accepted{0};
        std::atomic<std::uint64_t> pruned_by_current_makespan{0};
        std::atomic<std::uint64_t> pruned_by_optimistic_bound{0};
        std::atomic<std::uint64_t> pruned_by_capacity_bound{0};
        std::atomic<std::uint64_t> pruned_by_candidate_makespan{0};
        std::atomic<std::uint64_t> pruned_by_candidate_bound{0};
        std::atomic<std::uint64_t> infeasible_candidates{0};
        std::atomic<std::uint64_t> exact_state_cache_queries{0};
        std::atomic<std::uint64_t> exact_state_cache_hits{0};
        std::atomic<std::uint64_t> exact_state_cache_records{0};
        std::atomic<std::uint64_t> exact_state_cache_saturated{0};
        std::atomic<std::uint64_t> best_updates{0};
        std::atomic<std::uint64_t> prefixes_total{0};
        std::atomic<std::uint64_t> prefixes_done{0};
        std::atomic<std::size_t> max_depth_reached{0};
        std::atomic<double> best_makespan{std::numeric_limits<double>::infinity()};
    };

    ExhaustiveProgressCounters progress;
    std::atomic<bool> stop_progress_reporter{false};
    std::thread progress_reporter;

    struct ProgressReporterGuard {
        std::atomic<bool>& stop;
        std::thread& reporter;

        ~ProgressReporterGuard() {
            stop.store(true, std::memory_order_relaxed);
            if (reporter.joinable()) {
                reporter.join();
            }
        }
    };

    ProgressReporterGuard progress_guard{stop_progress_reporter, progress_reporter};

    auto progress_add = [&](std::atomic<std::uint64_t>& counter, std::uint64_t value = 1) {
        if (report_exhaustive_progress) {
            counter.fetch_add(value, std::memory_order_relaxed);
        }
    };

    auto progress_update_depth = [&](std::size_t depth) {
        if (!report_exhaustive_progress) {
            return;
        }
        auto observed = progress.max_depth_reached.load(std::memory_order_relaxed);
        while (depth > observed &&
               !progress.max_depth_reached.compare_exchange_weak(observed,
                                                                  depth,
                                                                  std::memory_order_relaxed,
                                                                  std::memory_order_relaxed)) {}
    };

    auto progress_update_best = [&](double makespan) {
        if (report_exhaustive_progress && std::isfinite(makespan)) {
            progress.best_makespan.store(makespan, std::memory_order_relaxed);
        }
    };

    struct SizeVectorHash {
        std::size_t operator()(const std::vector<std::size_t>& values) const {
            std::size_t seed = values.size();
            for (const auto value : values) {
                seed ^= std::hash<std::size_t>{}(value) + 0x9e3779b97f4a7c15ULL +
                        (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    struct ExactStateKey {
        std::size_t task_pos{0};
        std::vector<std::uint64_t> words;

        bool operator==(const ExactStateKey& other) const {
            return task_pos == other.task_pos && words == other.words;
        }
    };

    struct ExactStateKeyHash {
        std::size_t operator()(const ExactStateKey& key) const {
            std::size_t seed = key.task_pos;
            for (const auto word : key.words) {
                const auto mixed = static_cast<std::size_t>(
                    word ^ (word >> 32));
                seed ^= mixed + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };

    struct ExactStateDominance {
        double current_makespan{std::numeric_limits<double>::infinity()};
        double optimistic_bound{std::numeric_limits<double>::infinity()};
    };

    auto double_bits = [](double value) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        return bits;
    };

    auto assignment_lexicographically_smaller = [&](const std::vector<std::size_t>& lhs,
                                                    const std::vector<std::size_t>& rhs) {
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            const auto& left_id = ordered_devices[lhs[i]]->id;
            const auto& right_id = ordered_devices[rhs[i]]->id;
            if (left_id != right_id) {
                return left_id < right_id;
            }
        }
        return false;
    };

    auto maybe_update_best = [&](double makespan, const std::vector<std::size_t>& assignment) {
        if (!std::isfinite(makespan)) {
            return;
        }
        if (!have_best_assignment ||
            makespan < best_makespan ||
            (makespan == best_makespan &&
             assignment_lexicographically_smaller(assignment, best_assignment))) {
            best_makespan = makespan;
            best_assignment = assignment;
            have_best_assignment = true;
            if (collect_exhaustive_stats) {
                ++stats.best_updates;
            }
            progress_add(progress.best_updates);
            progress_update_best(best_makespan);
        }
    };

    const std::size_t task_count = topo.size();
    const std::size_t device_count = ordered_devices.size();
    const bool exact_cache_enabled =
        branch_and_bound_ && parse_bool_env("TERRAPOD_EXHAUSTIVE_EXACT_CACHE", true);
    const bool exact_assignment_cache_enabled =
        parse_bool_env("TERRAPOD_EXHAUSTIVE_ASSIGNMENT_CACHE", true);
    const bool shared_exact_cache_enabled =
        branch_and_bound_ && parse_bool_env("TERRAPOD_EXHAUSTIVE_SHARED_EXACT_CACHE", true);
    const std::size_t exact_state_cache_max_entries =
        parse_size_env("TERRAPOD_EXHAUSTIVE_EXACT_CACHE_MAX_ENTRIES", 262144);
    const std::size_t exact_assignment_cache_max_entries =
        parse_size_env("TERRAPOD_EXHAUSTIVE_ASSIGNMENT_CACHE_MAX_ENTRIES", 262144);
    const std::size_t requested_exact_cache_shards =
        parse_size_env("TERRAPOD_EXHAUSTIVE_EXACT_CACHE_SHARDS", 64);
    if (report_exhaustive_progress) {
        progress_reporter = std::thread([&]() {
            const auto started_at = std::chrono::steady_clock::now();
            auto next_report_at =
                started_at + std::chrono::seconds(exhaustive_progress_interval_s);
            while (!stop_progress_reporter.load(std::memory_order_relaxed)) {
                while (!stop_progress_reporter.load(std::memory_order_relaxed) &&
                       std::chrono::steady_clock::now() < next_report_at) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(250));
                }
                if (stop_progress_reporter.load(std::memory_order_relaxed)) {
                    break;
                }

                const auto now = std::chrono::steady_clock::now();
                next_report_at = now + std::chrono::seconds(exhaustive_progress_interval_s);
                const auto elapsed_s =
                    std::chrono::duration_cast<std::chrono::seconds>(now - started_at).count();
                const auto nodes = progress.nodes_visited.load(std::memory_order_relaxed);
                const auto leaves = progress.leaves_reached.load(std::memory_order_relaxed);
                const auto branch_attempts =
                    progress.branch_attempts.load(std::memory_order_relaxed);
                const auto branch_accepted =
                    progress.branch_accepted.load(std::memory_order_relaxed);
                const auto prunes =
                    progress.pruned_by_current_makespan.load(std::memory_order_relaxed) +
                    progress.pruned_by_optimistic_bound.load(std::memory_order_relaxed) +
                    progress.pruned_by_capacity_bound.load(std::memory_order_relaxed) +
                    progress.pruned_by_candidate_makespan.load(std::memory_order_relaxed) +
                    progress.pruned_by_candidate_bound.load(std::memory_order_relaxed);
                const auto exact_queries =
                    progress.exact_state_cache_queries.load(std::memory_order_relaxed);
                const auto exact_hits =
                    progress.exact_state_cache_hits.load(std::memory_order_relaxed);
                const auto exact_records =
                    progress.exact_state_cache_records.load(std::memory_order_relaxed);
                const auto exact_saturated =
                    progress.exact_state_cache_saturated.load(std::memory_order_relaxed);
                const auto best = progress.best_makespan.load(std::memory_order_relaxed);

                std::cerr << "[exhaustive progress] elapsed=" << elapsed_s
                          << "s tasks=" << task_count
                          << " devices=" << device_count
                          << " best=";
                if (std::isfinite(best)) {
                    std::cerr << best;
                } else {
                    std::cerr << "inf";
                }
                std::cerr << " best_updates="
                          << progress.best_updates.load(std::memory_order_relaxed)
                          << " nodes=" << nodes
                          << " leaves=" << leaves
                          << " max_depth="
                          << progress.max_depth_reached.load(std::memory_order_relaxed)
                          << " branches=" << branch_accepted << "/" << branch_attempts
                          << " prunes=" << prunes
                          << " exact_hits=" << exact_hits << "/" << exact_queries
                          << " exact_records=" << exact_records
                          << " exact_saturated=" << exact_saturated
                          << " prefixes="
                          << progress.prefixes_done.load(std::memory_order_relaxed)
                          << "/"
                          << progress.prefixes_total.load(std::memory_order_relaxed)
                          << "\n";
            }
        });
    }
    const auto& topology_devices = topology.devices();
    const auto& topology_links = topology.links();

    std::unordered_map<std::string, std::size_t> topology_device_index_by_id;
    topology_device_index_by_id.reserve(topology_devices.size());
    for (std::size_t i = 0; i < topology_devices.size(); ++i) {
        topology_device_index_by_id.emplace(topology_devices[i]->id, i);
    }
    std::unordered_map<std::string, std::size_t> link_index_by_id;
    link_index_by_id.reserve(topology_links.size());
    for (std::size_t i = 0; i < topology_links.size(); ++i) {
        link_index_by_id.emplace(topology_links[i].id, i);
    }

    struct RouteHop {
        std::size_t link_idx{0};
        double elapsed_s{0.0};
        double link_busy_s{0.0};
    };

    struct PrecomputedComm {
        bool same_device_noop{false};
        bool collective{false};
        bool route_valid{false};
        std::size_t source_topology_device_idx{0};
        double endpoint_extra_latency_s{0.0};
        double optimistic_time_s{std::numeric_limits<double>::infinity()};
        std::vector<RouteHop> route;
    };

    struct ExhaustiveEdge {
        std::size_t src_task{0};
        std::size_t dst_task{0};
        double min_comm_time_s{std::numeric_limits<double>::infinity()};
        std::vector<PrecomputedComm> comm_by_pair;
    };

    std::vector<double> compute_times(task_count * device_count, kInf);
    std::vector<double> min_compute_time(task_count, kInf);
    std::vector<unsigned char> eligible_device_mask(task_count * device_count, 0);
    for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
        for (const auto device_idx : eligible_by_task[task_pos]) {
            eligible_device_mask[task_pos * device_count + device_idx] = 1;
            const double compute = compute_time_seconds(topo[task_pos], ordered_devices[device_idx]);
            compute_times[task_pos * device_count + device_idx] = compute;
            if (std::isfinite(compute)) {
                min_compute_time[task_pos] = std::min(min_compute_time[task_pos], compute);
            }
        }
        if (collect_exhaustive_stats) {
            stats.initial_task_device_choices += eligible_by_task[task_pos].size();
        }
    }

    auto precompute_comm = [&](const TaskEdge& edge,
                               std::size_t src_task,
                               std::size_t dst_task) {
        ExhaustiveEdge indexed;
        indexed.src_task = src_task;
        indexed.dst_task = dst_task;
        indexed.comm_by_pair.resize(device_count * device_count);

        for (std::size_t src_device = 0; src_device < device_count; ++src_device) {
            for (std::size_t dst_device = 0; dst_device < device_count; ++dst_device) {
                auto& comm = indexed.comm_by_pair[src_device * device_count + dst_device];
                comm.collective = is_collective_kind(edge.comm_kind);
                const auto& src_id = ordered_devices[src_device]->id;
                const auto& dst_id = ordered_devices[dst_device]->id;

                if (src_device == dst_device && !comm.collective) {
                    comm.same_device_noop = true;
                    comm.optimistic_time_s = 0.0;
                } else if (comm.collective) {
                    comm.optimistic_time_s =
                        estimate_communication_time_seconds(edge, topology, src_id, dst_id);
                } else {
                    const auto src_idx_it = topology_device_index_by_id.find(src_id);
                    if (src_idx_it != topology_device_index_by_id.end()) {
                        comm.source_topology_device_idx = src_idx_it->second;
                        comm.endpoint_extra_latency_s = topology.endpoint_extra_latency_seconds(src_id, dst_id);
                        const auto route = topology.shortest_route_link_ids(
                            src_id,
                            dst_id,
                            static_cast<std::size_t>(std::max(0.0, edge.tensor_bytes)));
                        double no_contention_time = comm.endpoint_extra_latency_s;
                        bool route_valid = !route.empty();
                        for (const auto& link_id : route) {
                            const auto link_idx_it = link_index_by_id.find(link_id);
                            if (link_idx_it == link_index_by_id.end()) {
                                route_valid = false;
                                break;
                            }
                            const auto& link = topology_links[link_idx_it->second];
                            if (link.bw_gbps <= 0.0) {
                                route_valid = false;
                                break;
                            }
                            const auto timing =
                                topology.link_transfer_timing_seconds(
                                    link,
                                    src_id,
                                    dst_id,
                                    bytes_to_size_t(edge.tensor_bytes),
                                    "native",
                                    route.size());
                            if (!std::isfinite(timing.elapsed_s) || !std::isfinite(timing.link_busy_s)) {
                                route_valid = false;
                                break;
                            }
                            no_contention_time += timing.elapsed_s;
                            comm.route.push_back(RouteHop{link_idx_it->second,
                                                          timing.elapsed_s,
                                                          timing.link_busy_s});
                        }
                        if (route_valid) {
                            comm.route_valid = true;
                            comm.optimistic_time_s = no_contention_time;
                        } else {
                            comm.route.clear();
                        }
                    }
                    if (!comm.route_valid) {
                        comm.optimistic_time_s =
                            estimate_communication_time_seconds(edge, topology, src_id, dst_id);
                    }
                }

                if (eligible_device_mask[src_task * device_count + src_device] != 0 &&
                    eligible_device_mask[dst_task * device_count + dst_device] != 0 &&
                    std::isfinite(comm.optimistic_time_s)) {
                    indexed.min_comm_time_s = std::min(indexed.min_comm_time_s, comm.optimistic_time_s);
                }
            }
        }
        return indexed;
    };

    std::vector<ExhaustiveEdge> exhaustive_edges;
    std::vector<std::vector<std::size_t>> dependency_edges_by_task(task_count);
    std::vector<std::vector<std::size_t>> successor_edges_by_task(task_count);
    for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
        for (const auto& edge : graph.dependencies(topo[task_pos].name)) {
            const auto pred_it = task_index.find(edge.src);
            if (pred_it == task_index.end() || pred_it->second >= task_pos) {
                throw std::runtime_error("Task dependencies must appear before successors");
            }
            const auto edge_idx = exhaustive_edges.size();
            exhaustive_edges.push_back(precompute_comm(edge, pred_it->second, task_pos));
            dependency_edges_by_task[task_pos].push_back(edge_idx);
            successor_edges_by_task[pred_it->second].push_back(edge_idx);
        }
    }

    std::vector<double> optimistic_tail_from_task(task_count, 0.0);
    std::vector<double> optimistic_tail_after_task(task_count, 0.0);
    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        double tail_after = 0.0;
        for (const auto edge_idx : successor_edges_by_task[pos]) {
            const auto& edge = exhaustive_edges[edge_idx];
            if (!std::isfinite(edge.min_comm_time_s) ||
                !std::isfinite(optimistic_tail_from_task[edge.dst_task])) {
                tail_after = kInf;
                break;
            }
            tail_after = std::max(tail_after,
                                  edge.min_comm_time_s +
                                      optimistic_tail_from_task[edge.dst_task]);
        }
        optimistic_tail_after_task[pos] = tail_after;
        optimistic_tail_from_task[pos] =
            std::isfinite(min_compute_time[pos]) && std::isfinite(tail_after)
                ? min_compute_time[pos] + tail_after
                : kInf;
    }

    std::vector<double> remaining_min_compute_suffix(task_count + 1, 0.0);
    for (std::size_t pos = task_count; pos > 0;) {
        --pos;
        if (!std::isfinite(min_compute_time[pos]) ||
            !std::isfinite(remaining_min_compute_suffix[pos + 1])) {
            remaining_min_compute_suffix[pos] = kInf;
        } else {
            remaining_min_compute_suffix[pos] =
                min_compute_time[pos] + remaining_min_compute_suffix[pos + 1];
        }
    }

    std::vector<std::size_t> last_successor_task(task_count, 0);
    for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
        last_successor_task[task_pos] = task_pos;
        for (const auto edge_idx : successor_edges_by_task[task_pos]) {
            last_successor_task[task_pos] =
                std::max(last_successor_task[task_pos], exhaustive_edges[edge_idx].dst_task);
        }
    }
    std::vector<std::vector<std::size_t>> live_predecessors_by_depth(task_count + 1);
    for (std::size_t depth = 0; depth <= task_count; ++depth) {
        auto& live = live_predecessors_by_depth[depth];
        for (std::size_t task_pos = 0; task_pos < depth; ++task_pos) {
            if (last_successor_task[task_pos] >= depth) {
                live.push_back(task_pos);
            }
        }
    }

    std::vector<std::size_t> all_compute_device_indices;
    all_compute_device_indices.reserve(device_count);
    for (std::size_t device_idx = 0; device_idx < device_count; ++device_idx) {
        all_compute_device_indices.push_back(device_idx);
    }
    std::vector<std::size_t> all_topology_device_indices;
    all_topology_device_indices.reserve(topology_devices.size());
    for (std::size_t device_idx = 0; device_idx < topology_devices.size(); ++device_idx) {
        all_topology_device_indices.push_back(device_idx);
    }
    std::vector<std::size_t> all_link_indices;
    all_link_indices.reserve(topology_links.size());
    for (std::size_t link_idx = 0; link_idx < topology_links.size(); ++link_idx) {
        all_link_indices.push_back(link_idx);
    }

    std::vector<std::vector<std::size_t>> exact_cache_compute_devices_by_depth;
    std::vector<std::vector<std::size_t>> exact_cache_comm_sources_by_depth;
    std::vector<std::vector<std::size_t>> exact_cache_links_by_depth;

    auto make_exact_state_key = [&](std::size_t task_pos,
                                    const std::vector<std::size_t>& assignment,
                                    const std::vector<double>& task_finish_time,
                                    const std::vector<double>& device_available,
                                    const std::vector<double>& comm_device_available,
                                    const std::vector<double>& route_link_available) {
        ExactStateKey key;
        key.task_pos = task_pos;
        const auto& live = live_predecessors_by_depth[task_pos];
        const auto& live_compute_devices =
            exact_cache_compute_devices_by_depth.empty()
                ? all_compute_device_indices
                : exact_cache_compute_devices_by_depth[task_pos];
        const auto& live_comm_sources =
            exact_cache_comm_sources_by_depth.empty()
                ? all_topology_device_indices
                : exact_cache_comm_sources_by_depth[task_pos];
        const auto& live_links =
            exact_cache_links_by_depth.empty()
                ? all_link_indices
                : exact_cache_links_by_depth[task_pos];

        key.words.reserve(live_compute_devices.size() + live_comm_sources.size() +
                          live_links.size() + live.size() * 2);
        for (const auto device_idx : live_compute_devices) {
            key.words.push_back(double_bits(device_available[device_idx]));
        }
        for (const auto device_idx : live_comm_sources) {
            key.words.push_back(double_bits(comm_device_available[device_idx]));
        }
        for (const auto link_idx : live_links) {
            key.words.push_back(double_bits(route_link_available[link_idx]));
        }
        for (const auto pred_task : live) {
            key.words.push_back(static_cast<std::uint64_t>(assignment[pred_task]));
            key.words.push_back(double_bits(task_finish_time[pred_task]));
        }
        return key;
    };

    auto dominated_or_record_exact_state =
        [&](std::unordered_map<ExactStateKey, ExactStateDominance, ExactStateKeyHash>& cache,
            ExactStateKey key,
            double current_makespan,
            double optimistic_bound,
            ExhaustiveSearchStats& target_stats,
            std::size_t max_entries) {
        if (!exact_cache_enabled || max_entries == 0 ||
            !std::isfinite(current_makespan) || !std::isfinite(optimistic_bound)) {
            return false;
        }
        if (collect_exhaustive_stats) {
            ++target_stats.exact_state_cache_queries;
        }
        progress_add(progress.exact_state_cache_queries);
        auto it = cache.find(key);
        if (it != cache.end()) {
            const auto& prior = it->second;
            if (prior.current_makespan <= current_makespan &&
                prior.optimistic_bound <= optimistic_bound) {
                if (collect_exhaustive_stats) {
                    ++target_stats.exact_state_cache_hits;
                }
                progress_add(progress.exact_state_cache_hits);
                return true;
            }
            if ((current_makespan < prior.current_makespan &&
                 optimistic_bound <= prior.optimistic_bound) ||
                (current_makespan <= prior.current_makespan &&
                 optimistic_bound < prior.optimistic_bound)) {
                it->second = ExactStateDominance{current_makespan, optimistic_bound};
            }
            return false;
        }
        if (cache.size() >= max_entries) {
            if (collect_exhaustive_stats) {
                ++target_stats.exact_state_cache_saturated;
            }
            progress_add(progress.exact_state_cache_saturated);
            return false;
        }
        cache.emplace(std::move(key), ExactStateDominance{current_makespan, optimistic_bound});
        if (collect_exhaustive_stats) {
            ++target_stats.exact_state_cache_records;
        }
        progress_add(progress.exact_state_cache_records);
        return false;
    };

    std::unordered_map<std::vector<std::size_t>, double, SizeVectorHash>
        exact_assignment_makespan_cache;
    if (exact_assignment_cache_enabled && exact_assignment_cache_max_entries > 0) {
        exact_assignment_makespan_cache.reserve(
            std::min<std::size_t>(exact_assignment_cache_max_entries, 65536));
    }

    auto assignment_makespan = [&](const std::vector<std::size_t>& assignment) {
        if (collect_exhaustive_stats) {
            ++stats.assignment_evaluations;
        }
        if (assignment.size() != task_count) {
            return kInf;
        }
        if (exact_assignment_cache_enabled && exact_assignment_cache_max_entries > 0) {
            const auto cached = exact_assignment_makespan_cache.find(assignment);
            if (cached != exact_assignment_makespan_cache.end()) {
                if (collect_exhaustive_stats) {
                    ++stats.exact_assignment_cache_hits;
                }
                return cached->second;
            }
        }

        std::vector<double> local_available(device_count, 0.0);
        std::vector<double> local_finish_time(task_count, 0.0);
        std::vector<double> local_comm_available(topology_devices.size(), 0.0);
        std::vector<double> local_link_available(topology_links.size(), 0.0);
        double makespan = 0.0;

        for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
            const auto device_idx = assignment[task_pos];
            if (device_idx >= device_count ||
                eligible_device_mask[task_pos * device_count + device_idx] == 0) {
                return kInf;
            }

            double ready = local_available[device_idx];
            for (const auto edge_idx : dependency_edges_by_task[task_pos]) {
                const auto& edge = exhaustive_edges[edge_idx];
                const auto pred_idx = edge.src_task;
                const auto pred_device_idx = assignment[pred_idx];
                if (pred_device_idx >= device_count) {
                    return kInf;
                }
                const auto& comm = edge.comm_by_pair[pred_device_idx * device_count + device_idx];
                if (comm.same_device_noop) {
                    ready = std::max(ready, local_finish_time[pred_idx]);
                    continue;
                }
                if (comm.collective) {
                    if (!std::isfinite(comm.optimistic_time_s)) {
                        return kInf;
                    }
                    ready = std::max(ready, local_finish_time[pred_idx] + comm.optimistic_time_s);
                    continue;
                }
                if (comm.route_valid) {
                    double now =
                        std::max(local_finish_time[pred_idx],
                                 local_comm_available[comm.source_topology_device_idx]) +
                        comm.endpoint_extra_latency_s;
                    for (const auto& hop : comm.route) {
                        const double start = std::max(now, local_link_available[hop.link_idx]);
                        const double finish = start + hop.elapsed_s;
                        local_link_available[hop.link_idx] = start + hop.link_busy_s;
                        now = finish;
                    }
                    local_comm_available[comm.source_topology_device_idx] = now;
                    ready = std::max(ready, now);
                    continue;
                }
                if (!std::isfinite(comm.optimistic_time_s)) {
                    return kInf;
                }
                ready = std::max(ready, local_finish_time[pred_idx] + comm.optimistic_time_s);
            }

            const double exec = compute_times[task_pos * device_count + device_idx];
            if (!std::isfinite(exec)) {
                return kInf;
            }
            const double finish = ready + exec;
            local_available[device_idx] = finish;
            local_finish_time[task_pos] = finish;
            makespan = std::max(makespan, finish);
        }
        if (exact_assignment_cache_enabled &&
            exact_assignment_makespan_cache.size() < exact_assignment_cache_max_entries) {
            exact_assignment_makespan_cache.emplace(assignment, makespan);
            if (collect_exhaustive_stats) {
                ++stats.exact_assignment_cache_records;
            }
        }
        return makespan;
    };

    if (seed_upper_bound_) {
        std::unordered_map<std::string, std::size_t> device_index_by_id;
        device_index_by_id.reserve(ordered_devices.size());
        for (std::size_t i = 0; i < ordered_devices.size(); ++i) {
            device_index_by_id.emplace(ordered_devices[i]->id, i);
        }

        const bool debug_exhaustive_seeds = std::getenv("TERRAPOD_DEBUG_EXHAUSTIVE_SEEDS") != nullptr;

        auto candidate_assignment_better = [&](double candidate_makespan,
                                               const std::vector<std::size_t>& candidate_assignment,
                                               double incumbent_makespan,
                                               const std::vector<std::size_t>& incumbent_assignment) {
            if (!std::isfinite(candidate_makespan)) {
                return false;
            }
            if (!std::isfinite(incumbent_makespan) ||
                candidate_makespan < incumbent_makespan) {
                return true;
            }
            return candidate_makespan == incumbent_makespan &&
                   assignment_lexicographically_smaller(candidate_assignment, incumbent_assignment);
        };

        auto improve_seed_assignment = [&](const char* label, std::vector<std::size_t>& seed_assignment) {
            constexpr std::size_t kLocalImproveMaxTasks = 2048;
            constexpr std::size_t kLocalImprovePasses = 2;
            constexpr std::size_t kPairSwapCandidateTasks = 64;
            constexpr std::size_t kPairSwapMaxUpdates = 16;

            double current_makespan = assignment_makespan(seed_assignment);
            if (!std::isfinite(current_makespan)) {
                return current_makespan;
            }
            maybe_update_best(current_makespan, seed_assignment);

            if (task_count < 64 || task_count > kLocalImproveMaxTasks) {
                return current_makespan;
            }

            const double initial_makespan = current_makespan;
            std::vector<std::size_t> trial = seed_assignment;
            for (std::size_t pass = 0; pass < kLocalImprovePasses; ++pass) {
                bool changed = false;
                for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
                    double best_local_makespan = current_makespan;
                    std::vector<std::size_t> best_local_assignment = seed_assignment;
                    const auto original_device = seed_assignment[task_pos];

                    trial = seed_assignment;
                    for (const auto device_idx : eligible_by_task[task_pos]) {
                        if (device_idx == original_device) {
                            continue;
                        }
                        trial[task_pos] = device_idx;
                        const double candidate_makespan = assignment_makespan(trial);
                        if (candidate_assignment_better(candidate_makespan,
                                                        trial,
                                                        best_local_makespan,
                                                        best_local_assignment)) {
                            best_local_makespan = candidate_makespan;
                            best_local_assignment = trial;
                        }
                    }

                    if (candidate_assignment_better(best_local_makespan,
                                                    best_local_assignment,
                                                    current_makespan,
                                                    seed_assignment)) {
                        seed_assignment = std::move(best_local_assignment);
                        current_makespan = best_local_makespan;
                        maybe_update_best(current_makespan, seed_assignment);
                        changed = true;
                    }
                }
                if (!changed) {
                    break;
                }
            }

            std::vector<std::size_t> swap_tasks;
            swap_tasks.reserve(task_count);
            for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
                swap_tasks.push_back(task_pos);
            }
            std::sort(swap_tasks.begin(),
                      swap_tasks.end(),
                      [&](std::size_t lhs, std::size_t rhs) {
                          const double left_tail = optimistic_tail_from_task[lhs];
                          const double right_tail = optimistic_tail_from_task[rhs];
                          if (left_tail == right_tail) {
                              return topo[lhs].name < topo[rhs].name;
                          }
                          return left_tail > right_tail;
                      });
            if (swap_tasks.size() > kPairSwapCandidateTasks) {
                swap_tasks.resize(kPairSwapCandidateTasks);
            }

            std::size_t pair_updates = 0;
            for (std::size_t i = 0; i < swap_tasks.size(); ++i) {
                for (std::size_t j = i + 1; j < swap_tasks.size(); ++j) {
                    const auto lhs_task = swap_tasks[i];
                    const auto rhs_task = swap_tasks[j];
                    const auto lhs_device = seed_assignment[lhs_task];
                    const auto rhs_device = seed_assignment[rhs_task];
                    if (lhs_device == rhs_device ||
                        eligible_device_mask[lhs_task * device_count + rhs_device] == 0 ||
                        eligible_device_mask[rhs_task * device_count + lhs_device] == 0) {
                        continue;
                    }

                    trial = seed_assignment;
                    trial[lhs_task] = rhs_device;
                    trial[rhs_task] = lhs_device;
                    const double candidate_makespan = assignment_makespan(trial);
                    if (!candidate_assignment_better(candidate_makespan,
                                                     trial,
                                                     current_makespan,
                                                     seed_assignment)) {
                        continue;
                    }

                    seed_assignment = trial;
                    current_makespan = candidate_makespan;
                    maybe_update_best(current_makespan, seed_assignment);
                    ++pair_updates;
                    if (pair_updates >= kPairSwapMaxUpdates) {
                        break;
                    }
                }
                if (pair_updates >= kPairSwapMaxUpdates) {
                    break;
                }
            }

            if (debug_exhaustive_seeds && current_makespan < initial_makespan) {
                std::cerr << "[exhaustive seed] " << label
                          << " local_improved " << initial_makespan
                          << " -> " << current_makespan << "\n";
            }
            return current_makespan;
        };

        auto consider_seed_plan = [&](const char* label, const MappingPlan& seed_plan) {
            std::vector<std::size_t> seed_assignment(topo.size(), 0);
            for (std::size_t task_pos = 0; task_pos < topo.size(); ++task_pos) {
                const auto plan_it = seed_plan.assignments.find(topo[task_pos].name);
                if (plan_it == seed_plan.assignments.end()) {
                    if (debug_exhaustive_seeds) {
                        std::cerr << "[exhaustive seed] " << label
                                  << " missing assignment for " << topo[task_pos].name << "\n";
                    }
                    return;
                }
                const auto device_it = device_index_by_id.find(plan_it->second);
                if (device_it == device_index_by_id.end()) {
                    if (debug_exhaustive_seeds) {
                        std::cerr << "[exhaustive seed] " << label
                                  << " unknown device " << plan_it->second << "\n";
                    }
                    return;
                }
                const auto device_idx = device_it->second;
                const auto& eligible = eligible_by_task[task_pos];
                if (std::find(eligible.begin(), eligible.end(), device_idx) == eligible.end()) {
                    if (debug_exhaustive_seeds) {
                        std::cerr << "[exhaustive seed] " << label
                                  << " ineligible device " << plan_it->second
                                  << " for " << topo[task_pos].name << "\n";
                    }
                    return;
                }
                seed_assignment[task_pos] = device_idx;
            }
            const double seed_makespan = improve_seed_assignment(label, seed_assignment);
            if (debug_exhaustive_seeds) {
                std::unordered_map<std::string, std::size_t> seed_counts;
                for (const auto idx : seed_assignment) {
                    seed_counts[ordered_devices[idx]->id] += 1;
                }
                std::cerr << "[exhaustive seed] " << label
                          << " internal_makespan=" << seed_makespan
                          << " devices=";
                bool first = true;
                for (const auto& entry : seed_counts) {
                    if (!first) {
                        std::cerr << ",";
                    }
                    first = false;
                    std::cerr << entry.first << ":" << entry.second;
                }
                std::cerr << "\n";
            }
        };

        AeftMapper aeft_seed;
        PeftMapper peft_seed;
        HoftMapper hoft_seed;
        GreedyMapper greedy_seed;
        for (const auto* seed_mapper : std::vector<const Mapper*>{
                 &aeft_seed,
                 &peft_seed,
                 &hoft_seed,
                 &greedy_seed,
             }) {
            try {
                const char* label = seed_mapper == &aeft_seed ? "aeft"
                                   : seed_mapper == &peft_seed ? "peft"
                                   : seed_mapper == &hoft_seed ? "hoft"
                                   : "greedy";
                consider_seed_plan(label, seed_mapper->map(graph, topology));
            } catch (const std::exception&) {
                // A seed heuristic is only an upper-bound source; exhaustive search
                // below still proves optimality if a heuristic cannot produce a plan.
            }
        }
    }

    if (branch_and_bound_ && have_best_assignment) {
        std::vector<double> optimistic_head_before_task(task_count, 0.0);
        std::vector<double> optimistic_head_finish_task(task_count, 0.0);
        for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
            double head_before = 0.0;
            for (const auto edge_idx : dependency_edges_by_task[task_pos]) {
                const auto& edge = exhaustive_edges[edge_idx];
                if (!std::isfinite(optimistic_head_finish_task[edge.src_task]) ||
                    !std::isfinite(edge.min_comm_time_s)) {
                    head_before = kInf;
                    break;
                }
                head_before = std::max(head_before,
                                       optimistic_head_finish_task[edge.src_task] +
                                           edge.min_comm_time_s);
            }
            optimistic_head_before_task[task_pos] = head_before;
            optimistic_head_finish_task[task_pos] =
                std::isfinite(head_before) && std::isfinite(min_compute_time[task_pos])
                    ? head_before + min_compute_time[task_pos]
                    : kInf;
        }

        const double prune_epsilon =
            std::max(1e-18, std::abs(best_makespan) * 1e-12);
        std::size_t pruned_task_devices = 0;
        for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
            auto& eligible = eligible_by_task[task_pos];
            std::vector<std::size_t> filtered;
            filtered.reserve(eligible.size());
            for (const auto device_idx : eligible) {
                const double compute = compute_times[task_pos * device_count + device_idx];
                const double bound =
                    std::isfinite(optimistic_head_before_task[task_pos]) &&
                            std::isfinite(compute) &&
                            std::isfinite(optimistic_tail_after_task[task_pos])
                        ? optimistic_head_before_task[task_pos] +
                              compute +
                              optimistic_tail_after_task[task_pos]
                        : kInf;
                if (std::isfinite(bound) &&
                    bound > best_makespan + prune_epsilon) {
                    ++pruned_task_devices;
                    continue;
                }
                filtered.push_back(device_idx);
            }
            eligible = std::move(filtered);
        }

        if (pruned_task_devices > 0) {
            std::fill(eligible_device_mask.begin(), eligible_device_mask.end(), 0);
            for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
                for (const auto device_idx : eligible_by_task[task_pos]) {
                    eligible_device_mask[task_pos * device_count + device_idx] = 1;
                }
            }
        }
        if (collect_exhaustive_stats) {
            stats.prepruned_task_device_choices += pruned_task_devices;
        }
    }
    if (collect_exhaustive_stats) {
        for (std::size_t task_pos = 0; task_pos < task_count; ++task_pos) {
            stats.remaining_task_device_choices += eligible_by_task[task_pos].size();
        }
    }

    auto collect_live_indices = [](const std::vector<unsigned char>& live) {
        std::vector<std::size_t> indices;
        indices.reserve(live.size());
        for (std::size_t idx = 0; idx < live.size(); ++idx) {
            if (live[idx] != 0) {
                indices.push_back(idx);
            }
        }
        return indices;
    };

    auto rebuild_exact_cache_resource_indices = [&]() {
        exact_cache_compute_devices_by_depth.assign(task_count + 1, {});
        exact_cache_comm_sources_by_depth.assign(task_count + 1, {});
        exact_cache_links_by_depth.assign(task_count + 1, {});

        std::vector<unsigned char> live_compute_devices(device_count, 0);
        std::vector<unsigned char> live_comm_sources(topology_devices.size(), 0);
        std::vector<unsigned char> live_links(topology_links.size(), 0);

        for (std::size_t depth = task_count; depth > 0;) {
            --depth;
            for (const auto device_idx : eligible_by_task[depth]) {
                live_compute_devices[device_idx] = 1;
            }
            for (const auto edge_idx : dependency_edges_by_task[depth]) {
                const auto& edge = exhaustive_edges[edge_idx];
                for (const auto src_device_idx : eligible_by_task[edge.src_task]) {
                    for (const auto dst_device_idx : eligible_by_task[edge.dst_task]) {
                        const auto& comm =
                            edge.comm_by_pair[src_device_idx * device_count + dst_device_idx];
                        if (!comm.route_valid) {
                            continue;
                        }
                        live_comm_sources[comm.source_topology_device_idx] = 1;
                        for (const auto& hop : comm.route) {
                            live_links[hop.link_idx] = 1;
                        }
                    }
                }
            }
            exact_cache_compute_devices_by_depth[depth] =
                collect_live_indices(live_compute_devices);
            exact_cache_comm_sources_by_depth[depth] =
                collect_live_indices(live_comm_sources);
            exact_cache_links_by_depth[depth] = collect_live_indices(live_links);
        }
    };

    rebuild_exact_cache_resource_indices();

    std::vector<double> comm_available(topology_devices.size(), 0.0);
    std::vector<double> link_available(topology_links.size(), 0.0);

    struct StateChange {
        bool link_state{false};
        std::size_t index{0};
        double old_value{0.0};
    };

    std::vector<StateChange> state_changes;
    state_changes.reserve(std::max<std::size_t>(64, exhaustive_edges.size() * 4));

    auto update_comm_state = [&](std::size_t index, double value) {
        if (comm_available[index] == value) {
            return;
        }
        state_changes.push_back(StateChange{false, index, comm_available[index]});
        comm_available[index] = value;
    };

    auto update_link_state = [&](std::size_t index, double value) {
        if (link_available[index] == value) {
            return;
        }
        state_changes.push_back(StateChange{true, index, link_available[index]});
        link_available[index] = value;
    };

    auto rollback_state = [&](std::size_t checkpoint) {
        while (state_changes.size() > checkpoint) {
            const auto change = state_changes.back();
            state_changes.pop_back();
            if (change.link_state) {
                link_available[change.index] = change.old_value;
            } else {
                comm_available[change.index] = change.old_value;
            }
        }
    };

    auto schedule_precomputed_p2p_transfer = [&](const PrecomputedComm& comm,
                                                 double earliest_start) {
        if (!comm.route_valid) {
            return kInf;
        }
        double now = std::max(earliest_start, comm_available[comm.source_topology_device_idx]) +
                     comm.endpoint_extra_latency_s;
        for (const auto& hop : comm.route) {
            const double start = std::max(now, link_available[hop.link_idx]);
            const double finish = start + hop.elapsed_s;
            update_link_state(hop.link_idx, start + hop.link_busy_s);
            now = finish;
        }
        update_comm_state(comm.source_topology_device_idx, now);
        return now;
    };

    std::vector<double> capacity_scratch;
    capacity_scratch.reserve(device_count);
    auto remaining_capacity_lower_bound = [&](double remaining_min_work) {
        if (!std::isfinite(remaining_min_work)) {
            return kInf;
        }
        if (remaining_min_work <= 0.0) {
            return 0.0;
        }

        capacity_scratch.clear();
        for (const auto value : available) {
            if (!std::isfinite(value)) {
                return kInf;
            }
            capacity_scratch.push_back(value);
        }
        if (capacity_scratch.empty()) {
            return kInf;
        }
        std::sort(capacity_scratch.begin(), capacity_scratch.end());

        double prefix = 0.0;
        for (std::size_t i = 0; i < capacity_scratch.size(); ++i) {
            prefix += capacity_scratch[i];
            const double active = static_cast<double>(i + 1);
            double candidate = (remaining_min_work + prefix) / active;
            candidate = std::max(candidate, capacity_scratch[i]);
            if (i + 1 == capacity_scratch.size() ||
                candidate <= capacity_scratch[i + 1]) {
                return candidate;
            }
        }
        return capacity_scratch.back() + remaining_min_work /
                                             static_cast<double>(capacity_scratch.size());
    };

    struct BranchCandidate {
        std::size_t device_idx{0};
        double ready{0.0};
        double finish{0.0};
        double next_makespan{0.0};
        double next_optimistic_bound{0.0};
    };

    auto evaluate_branch_candidate = [&](std::size_t task_pos,
                                         std::size_t device_idx,
                                         double current_makespan,
                                         double optimistic_bound,
                                         BranchCandidate& candidate) {
        double ready = available[device_idx];
        bool feasible = true;

        for (const auto edge_idx : dependency_edges_by_task[task_pos]) {
            const auto& edge = exhaustive_edges[edge_idx];
            const auto pred_idx = edge.src_task;
            const auto pred_device_idx = current_assignment[pred_idx];
            const auto& comm = edge.comm_by_pair[pred_device_idx * device_count + device_idx];
            if (comm.same_device_noop) {
                ready = std::max(ready, finish_time[pred_idx]);
                continue;
            }
            if (comm.collective) {
                if (!std::isfinite(comm.optimistic_time_s)) {
                    feasible = false;
                    break;
                }
                ready = std::max(ready, finish_time[pred_idx] + comm.optimistic_time_s);
                continue;
            }
            if (comm.route_valid) {
                const double comm_finish = schedule_precomputed_p2p_transfer(comm, finish_time[pred_idx]);
                if (std::isfinite(comm_finish)) {
                    ready = std::max(ready, comm_finish);
                    continue;
                }
            }
            if (!std::isfinite(comm.optimistic_time_s)) {
                feasible = false;
                break;
            }
            ready = std::max(ready, finish_time[pred_idx] + comm.optimistic_time_s);
        }
        if (!feasible) {
            if (collect_exhaustive_stats) {
                ++stats.infeasible_candidates;
            }
            progress_add(progress.infeasible_candidates);
            return false;
        }

        const double exec = compute_times[task_pos * device_count + device_idx];
        if (!std::isfinite(exec)) {
            if (collect_exhaustive_stats) {
                ++stats.infeasible_candidates;
            }
            progress_add(progress.infeasible_candidates);
            return false;
        }

        const double finish = ready + exec;
        const double next_makespan = std::max(current_makespan, finish);
        if (branch_and_bound_ && next_makespan > best_makespan) {
            if (collect_exhaustive_stats) {
                ++stats.pruned_by_candidate_makespan;
            }
            progress_add(progress.pruned_by_candidate_makespan);
            return false;
        }

        const double tail_bound =
            std::isfinite(optimistic_tail_after_task[task_pos])
                ? finish + optimistic_tail_after_task[task_pos]
                : kInf;
        const double next_optimistic_bound =
            std::max({optimistic_bound, next_makespan, tail_bound});
        if (branch_and_bound_ && next_optimistic_bound > best_makespan) {
            if (collect_exhaustive_stats) {
                ++stats.pruned_by_candidate_bound;
            }
            progress_add(progress.pruned_by_candidate_bound);
            return false;
        }

        candidate.device_idx = device_idx;
        candidate.ready = ready;
        candidate.finish = finish;
        candidate.next_makespan = next_makespan;
        candidate.next_optimistic_bound = next_optimistic_bound;
        return true;
    };

    struct DeviceOrderEntry {
        std::size_t device_idx{0};
        double score{std::numeric_limits<double>::infinity()};
        double next_makespan{std::numeric_limits<double>::infinity()};
        double finish{std::numeric_limits<double>::infinity()};
        double ready{std::numeric_limits<double>::infinity()};
    };

    auto cheap_device_order_entry = [&](std::size_t task_pos,
                                        std::size_t device_idx,
                                        double current_makespan,
                                        double optimistic_bound) {
        DeviceOrderEntry entry;
        entry.device_idx = device_idx;

        double ready = available[device_idx];
        for (const auto edge_idx : dependency_edges_by_task[task_pos]) {
            const auto& edge = exhaustive_edges[edge_idx];
            const auto pred_idx = edge.src_task;
            const auto pred_device_idx = current_assignment[pred_idx];
            const auto& comm = edge.comm_by_pair[pred_device_idx * device_count + device_idx];
            if (!std::isfinite(comm.optimistic_time_s)) {
                return entry;
            }
            ready = std::max(ready, finish_time[pred_idx] + comm.optimistic_time_s);
        }

        const double exec = compute_times[task_pos * device_count + device_idx];
        if (!std::isfinite(exec)) {
            return entry;
        }
        const double finish = ready + exec;
        const double next_makespan = std::max(current_makespan, finish);
        const double tail_bound =
            std::isfinite(optimistic_tail_after_task[task_pos])
                ? finish + optimistic_tail_after_task[task_pos]
                : kInf;

        entry.ready = ready;
        entry.finish = finish;
        entry.next_makespan = next_makespan;
        entry.score = std::max({optimistic_bound, next_makespan, tail_bound});
        return entry;
    };

    const bool use_capacity_lower_bound = branch_and_bound_ && task_count >= 64;
    const bool use_dynamic_device_sort = branch_and_bound_ && task_count >= 64;
    const std::size_t requested_exhaustive_threads =
        parse_size_env("TERRAPOD_EXHAUSTIVE_THREADS", 1);
    const std::size_t exhaustive_thread_count =
        branch_and_bound_ && have_best_assignment && task_count > 1
            ? std::min<std::size_t>(std::max<std::size_t>(1, requested_exhaustive_threads), 64)
            : 1;
    const std::size_t requested_prefix_depth =
        parse_size_env("TERRAPOD_EXHAUSTIVE_PREFIX_DEPTH", 0);
    const std::size_t parallel_prefix_depth =
        std::min(task_count,
                 requested_prefix_depth == 0
                     ? std::min<std::size_t>(2, task_count)
                     : requested_prefix_depth);

    std::unordered_map<ExactStateKey, ExactStateDominance, ExactStateKeyHash>
        exact_state_cache;
    if (exact_cache_enabled && exact_state_cache_max_entries > 0) {
        exact_state_cache.reserve(std::min<std::size_t>(exact_state_cache_max_entries, 65536));
    }

    std::function<void(std::size_t, double, double)> search =
        [&](std::size_t task_pos, double current_makespan, double optimistic_bound) {
        if (collect_exhaustive_stats) {
            ++stats.nodes_visited;
            stats.max_depth_reached = std::max(stats.max_depth_reached, task_pos);
        }
        progress_add(progress.nodes_visited);
        progress_update_depth(task_pos);
        if (task_pos == topo.size()) {
            if (collect_exhaustive_stats) {
                ++stats.leaves_reached;
            }
            progress_add(progress.leaves_reached);
            maybe_update_best(current_makespan, current_assignment);
            return;
        }
        if (branch_and_bound_ && current_makespan > best_makespan) {
            if (collect_exhaustive_stats) {
                ++stats.pruned_by_current_makespan;
            }
            progress_add(progress.pruned_by_current_makespan);
            return;
        }
        if (branch_and_bound_ && optimistic_bound > best_makespan) {
            if (collect_exhaustive_stats) {
                ++stats.pruned_by_optimistic_bound;
            }
            progress_add(progress.pruned_by_optimistic_bound);
            return;
        }

        if (use_capacity_lower_bound) {
            const double capacity_bound =
                remaining_capacity_lower_bound(remaining_min_compute_suffix[task_pos]);
            if (std::max({current_makespan, optimistic_bound, capacity_bound}) > best_makespan) {
                if (collect_exhaustive_stats) {
                    ++stats.pruned_by_capacity_bound;
                }
                progress_add(progress.pruned_by_capacity_bound);
                return;
            }
        }

        if (exact_cache_enabled && task_pos > 0 && task_pos < task_count) {
            auto key = make_exact_state_key(task_pos,
                                            current_assignment,
                                            finish_time,
                                            available,
                                            comm_available,
                                            link_available);
            if (dominated_or_record_exact_state(exact_state_cache,
                                                std::move(key),
                                                current_makespan,
                                                optimistic_bound,
                                                stats,
                                                exact_state_cache_max_entries)) {
                return;
            }
        }

        auto visit_device = [&](std::size_t device_idx) {
            const auto state_checkpoint = state_changes.size();
            BranchCandidate candidate;
            if (collect_exhaustive_stats) {
                ++stats.branch_attempts;
            }
            progress_add(progress.branch_attempts);
            if (!evaluate_branch_candidate(task_pos,
                                           device_idx,
                                           current_makespan,
                                           optimistic_bound,
                                           candidate)) {
                rollback_state(state_checkpoint);
                return;
            }
            if (collect_exhaustive_stats) {
                ++stats.branch_accepted;
            }
            progress_add(progress.branch_accepted);

            const double saved_available = available[device_idx];
            const double saved_finish = finish_time[task_pos];
            const auto saved_assignment = current_assignment[task_pos];
            available[device_idx] = candidate.finish;
            finish_time[task_pos] = candidate.finish;
            current_assignment[task_pos] = device_idx;

            search(task_pos + 1, candidate.next_makespan, candidate.next_optimistic_bound);

            current_assignment[task_pos] = saved_assignment;
            finish_time[task_pos] = saved_finish;
            available[device_idx] = saved_available;
            rollback_state(state_checkpoint);
        };

        constexpr std::size_t kDynamicDeviceSortMaxDepth = 8;
        if (use_dynamic_device_sort && task_pos < kDynamicDeviceSortMaxDepth) {
            if (collect_exhaustive_stats) {
                ++stats.dynamic_sort_nodes;
            }
            std::vector<DeviceOrderEntry> candidates;
            candidates.reserve(eligible_by_task[task_pos].size());
            for (const auto device_idx : eligible_by_task[task_pos]) {
                const auto candidate = cheap_device_order_entry(task_pos,
                                                                device_idx,
                                                                current_makespan,
                                                                optimistic_bound);
                if (std::isfinite(candidate.score)) {
                    candidates.push_back(candidate);
                }
            }
            std::sort(candidates.begin(),
                      candidates.end(),
                      [&](const auto& lhs, const auto& rhs) {
                          if (lhs.score != rhs.score) {
                              return lhs.score < rhs.score;
                          }
                          if (lhs.next_makespan != rhs.next_makespan) {
                              return lhs.next_makespan < rhs.next_makespan;
                          }
                          if (lhs.finish != rhs.finish) {
                              return lhs.finish < rhs.finish;
                          }
                          if (lhs.ready != rhs.ready) {
                              return lhs.ready < rhs.ready;
                          }
                          return ordered_devices[lhs.device_idx]->id <
                                 ordered_devices[rhs.device_idx]->id;
                      });
            for (const auto& candidate : candidates) {
                visit_device(candidate.device_idx);
            }
        } else {
            for (const auto device_idx : eligible_by_task[task_pos]) {
                visit_device(device_idx);
            }
        }
    };

    auto run_parallel_search = [&]() {
        struct SearchPrefix {
            std::size_t task_pos{0};
            double current_makespan{0.0};
            double optimistic_bound{0.0};
            std::vector<std::size_t> assignment;
            std::vector<double> available;
            std::vector<double> finish_time;
            std::vector<double> comm_available;
            std::vector<double> link_available;
        };

        std::vector<SearchPrefix> prefixes;
        prefixes.reserve(exhaustive_thread_count * 8);

        auto save_prefix = [&](std::size_t task_pos,
                               double current_makespan,
                               double optimistic_bound) {
            SearchPrefix prefix;
            prefix.task_pos = task_pos;
            prefix.current_makespan = current_makespan;
            prefix.optimistic_bound = optimistic_bound;
            prefix.assignment = current_assignment;
            prefix.available = available;
            prefix.finish_time = finish_time;
            prefix.comm_available = comm_available;
            prefix.link_available = link_available;
            prefixes.push_back(std::move(prefix));
        };

        std::function<void(std::size_t, double, double)> collect_prefixes =
            [&](std::size_t task_pos, double current_makespan, double optimistic_bound) {
            if (collect_exhaustive_stats) {
                ++stats.nodes_visited;
                stats.max_depth_reached = std::max(stats.max_depth_reached, task_pos);
            }
            progress_add(progress.nodes_visited);
            progress_update_depth(task_pos);
            if (task_pos >= parallel_prefix_depth || task_pos == task_count) {
                save_prefix(task_pos, current_makespan, optimistic_bound);
                return;
            }
            if (branch_and_bound_ && current_makespan > best_makespan) {
                if (collect_exhaustive_stats) {
                    ++stats.pruned_by_current_makespan;
                }
                progress_add(progress.pruned_by_current_makespan);
                return;
            }
            if (branch_and_bound_ && optimistic_bound > best_makespan) {
                if (collect_exhaustive_stats) {
                    ++stats.pruned_by_optimistic_bound;
                }
                progress_add(progress.pruned_by_optimistic_bound);
                return;
            }
            if (use_capacity_lower_bound) {
                const double capacity_bound =
                    remaining_capacity_lower_bound(remaining_min_compute_suffix[task_pos]);
                if (std::max({current_makespan, optimistic_bound, capacity_bound}) > best_makespan) {
                    if (collect_exhaustive_stats) {
                        ++stats.pruned_by_capacity_bound;
                    }
                    progress_add(progress.pruned_by_capacity_bound);
                    return;
                }
            }

            auto visit_device = [&](std::size_t device_idx) {
                const auto state_checkpoint = state_changes.size();
                BranchCandidate candidate;
                if (collect_exhaustive_stats) {
                    ++stats.branch_attempts;
                }
                progress_add(progress.branch_attempts);
                if (!evaluate_branch_candidate(task_pos,
                                               device_idx,
                                               current_makespan,
                                               optimistic_bound,
                                               candidate)) {
                    rollback_state(state_checkpoint);
                    return;
                }
                if (collect_exhaustive_stats) {
                    ++stats.branch_accepted;
                }
                progress_add(progress.branch_accepted);

                const double saved_available = available[device_idx];
                const double saved_finish = finish_time[task_pos];
                const auto saved_assignment = current_assignment[task_pos];
                available[device_idx] = candidate.finish;
                finish_time[task_pos] = candidate.finish;
                current_assignment[task_pos] = device_idx;

                collect_prefixes(task_pos + 1,
                                 candidate.next_makespan,
                                 candidate.next_optimistic_bound);

                current_assignment[task_pos] = saved_assignment;
                finish_time[task_pos] = saved_finish;
                available[device_idx] = saved_available;
                rollback_state(state_checkpoint);
            };

            constexpr std::size_t kDynamicDeviceSortMaxDepth = 8;
            if (use_dynamic_device_sort && task_pos < kDynamicDeviceSortMaxDepth) {
                if (collect_exhaustive_stats) {
                    ++stats.dynamic_sort_nodes;
                }
                std::vector<DeviceOrderEntry> candidates;
                candidates.reserve(eligible_by_task[task_pos].size());
                for (const auto device_idx : eligible_by_task[task_pos]) {
                    const auto candidate = cheap_device_order_entry(task_pos,
                                                                    device_idx,
                                                                    current_makespan,
                                                                    optimistic_bound);
                    if (std::isfinite(candidate.score)) {
                        candidates.push_back(candidate);
                    }
                }
                std::sort(candidates.begin(),
                          candidates.end(),
                          [&](const auto& lhs, const auto& rhs) {
                              if (lhs.score != rhs.score) {
                                  return lhs.score < rhs.score;
                              }
                              if (lhs.next_makespan != rhs.next_makespan) {
                                  return lhs.next_makespan < rhs.next_makespan;
                              }
                              if (lhs.finish != rhs.finish) {
                                  return lhs.finish < rhs.finish;
                              }
                              if (lhs.ready != rhs.ready) {
                                  return lhs.ready < rhs.ready;
                              }
                              return ordered_devices[lhs.device_idx]->id <
                                     ordered_devices[rhs.device_idx]->id;
                          });
                for (const auto& candidate : candidates) {
                    visit_device(candidate.device_idx);
                }
            } else {
                for (const auto device_idx : eligible_by_task[task_pos]) {
                    visit_device(device_idx);
                }
            }
        };

        collect_prefixes(0, 0.0, 0.0);
        if (prefixes.empty()) {
            return;
        }
        if (collect_exhaustive_stats) {
            stats.prefixes_created += prefixes.size();
        }
        progress.prefixes_total.fetch_add(prefixes.size(), std::memory_order_relaxed);

        struct DfsState {
            std::vector<std::size_t> current_assignment;
            std::vector<double> available;
            std::vector<double> finish_time;
            std::vector<double> comm_available;
            std::vector<double> link_available;
            std::vector<StateChange> state_changes;
            std::vector<double> capacity_scratch;
        };

        std::atomic<double> shared_best(best_makespan);
        std::mutex best_mutex;
        auto try_update_shared_best = [&](double makespan,
                                          const std::vector<std::size_t>& assignment,
                                          ExhaustiveSearchStats& local_stats) {
            if (!std::isfinite(makespan)) {
                return;
            }
            const double snapshot = shared_best.load(std::memory_order_relaxed);
            if (makespan > snapshot) {
                return;
            }
            std::lock_guard<std::mutex> lock(best_mutex);
            if (!have_best_assignment ||
                makespan < best_makespan ||
                (makespan == best_makespan &&
                 assignment_lexicographically_smaller(assignment, best_assignment))) {
                best_makespan = makespan;
                best_assignment = assignment;
                have_best_assignment = true;
                shared_best.store(best_makespan, std::memory_order_relaxed);
                if (collect_exhaustive_stats) {
                    ++local_stats.best_updates;
                }
                progress_add(progress.best_updates);
                progress_update_best(best_makespan);
            }
        };

        auto update_comm_state_for = [&](DfsState& state, std::size_t index, double value) {
            if (state.comm_available[index] == value) {
                return;
            }
            state.state_changes.push_back(StateChange{false, index, state.comm_available[index]});
            state.comm_available[index] = value;
        };

        auto update_link_state_for = [&](DfsState& state, std::size_t index, double value) {
            if (state.link_available[index] == value) {
                return;
            }
            state.state_changes.push_back(StateChange{true, index, state.link_available[index]});
            state.link_available[index] = value;
        };

        auto rollback_state_for = [&](DfsState& state, std::size_t checkpoint) {
            while (state.state_changes.size() > checkpoint) {
                const auto change = state.state_changes.back();
                state.state_changes.pop_back();
                if (change.link_state) {
                    state.link_available[change.index] = change.old_value;
                } else {
                    state.comm_available[change.index] = change.old_value;
                }
            }
        };

        auto schedule_precomputed_p2p_transfer_for = [&](DfsState& state,
                                                         const PrecomputedComm& comm,
                                                         double earliest_start) {
            if (!comm.route_valid) {
                return kInf;
            }
            double now =
                std::max(earliest_start, state.comm_available[comm.source_topology_device_idx]) +
                comm.endpoint_extra_latency_s;
            for (const auto& hop : comm.route) {
                const double start = std::max(now, state.link_available[hop.link_idx]);
                const double finish = start + hop.elapsed_s;
                update_link_state_for(state, hop.link_idx, start + hop.link_busy_s);
                now = finish;
            }
            update_comm_state_for(state, comm.source_topology_device_idx, now);
            return now;
        };

        auto remaining_capacity_lower_bound_for = [&](DfsState& state, double remaining_min_work) {
            if (!std::isfinite(remaining_min_work)) {
                return kInf;
            }
            if (remaining_min_work <= 0.0) {
                return 0.0;
            }
            state.capacity_scratch.clear();
            for (const auto value : state.available) {
                if (!std::isfinite(value)) {
                    return kInf;
                }
                state.capacity_scratch.push_back(value);
            }
            if (state.capacity_scratch.empty()) {
                return kInf;
            }
            std::sort(state.capacity_scratch.begin(), state.capacity_scratch.end());

            double prefix = 0.0;
            for (std::size_t i = 0; i < state.capacity_scratch.size(); ++i) {
                prefix += state.capacity_scratch[i];
                const double active = static_cast<double>(i + 1);
                double candidate = (remaining_min_work + prefix) / active;
                candidate = std::max(candidate, state.capacity_scratch[i]);
                if (i + 1 == state.capacity_scratch.size() ||
                    candidate <= state.capacity_scratch[i + 1]) {
                    return candidate;
                }
            }
            return state.capacity_scratch.back() +
                   remaining_min_work / static_cast<double>(state.capacity_scratch.size());
        };

        auto evaluate_branch_candidate_for = [&](DfsState& state,
                                                 std::size_t task_pos,
                                                 std::size_t device_idx,
                                                 double current_makespan,
                                                 double optimistic_bound,
                                                 ExhaustiveSearchStats& local_stats,
                                                 BranchCandidate& candidate) {
            double ready = state.available[device_idx];
            bool feasible = true;

            for (const auto edge_idx : dependency_edges_by_task[task_pos]) {
                const auto& edge = exhaustive_edges[edge_idx];
                const auto pred_idx = edge.src_task;
                const auto pred_device_idx = state.current_assignment[pred_idx];
                const auto& comm = edge.comm_by_pair[pred_device_idx * device_count + device_idx];
                if (comm.same_device_noop) {
                    ready = std::max(ready, state.finish_time[pred_idx]);
                    continue;
                }
                if (comm.collective) {
                    if (!std::isfinite(comm.optimistic_time_s)) {
                        feasible = false;
                        break;
                    }
                    ready = std::max(ready, state.finish_time[pred_idx] + comm.optimistic_time_s);
                    continue;
                }
                if (comm.route_valid) {
                    const double comm_finish =
                        schedule_precomputed_p2p_transfer_for(state, comm, state.finish_time[pred_idx]);
                    if (std::isfinite(comm_finish)) {
                        ready = std::max(ready, comm_finish);
                        continue;
                    }
                }
                if (!std::isfinite(comm.optimistic_time_s)) {
                    feasible = false;
                    break;
                }
                ready = std::max(ready, state.finish_time[pred_idx] + comm.optimistic_time_s);
            }
            if (!feasible) {
                if (collect_exhaustive_stats) {
                    ++local_stats.infeasible_candidates;
                }
                progress_add(progress.infeasible_candidates);
                return false;
            }

            const double exec = compute_times[task_pos * device_count + device_idx];
            if (!std::isfinite(exec)) {
                if (collect_exhaustive_stats) {
                    ++local_stats.infeasible_candidates;
                }
                progress_add(progress.infeasible_candidates);
                return false;
            }

            const double finish = ready + exec;
            const double next_makespan = std::max(current_makespan, finish);
            const double incumbent = shared_best.load(std::memory_order_relaxed);
            if (branch_and_bound_ && next_makespan > incumbent) {
                if (collect_exhaustive_stats) {
                    ++local_stats.pruned_by_candidate_makespan;
                }
                progress_add(progress.pruned_by_candidate_makespan);
                return false;
            }

            const double tail_bound =
                std::isfinite(optimistic_tail_after_task[task_pos])
                    ? finish + optimistic_tail_after_task[task_pos]
                    : kInf;
            const double next_optimistic_bound =
                std::max({optimistic_bound, next_makespan, tail_bound});
            if (branch_and_bound_ && next_optimistic_bound > incumbent) {
                if (collect_exhaustive_stats) {
                    ++local_stats.pruned_by_candidate_bound;
                }
                progress_add(progress.pruned_by_candidate_bound);
                return false;
            }

            candidate.device_idx = device_idx;
            candidate.ready = ready;
            candidate.finish = finish;
            candidate.next_makespan = next_makespan;
            candidate.next_optimistic_bound = next_optimistic_bound;
            return true;
        };

        auto cheap_device_order_entry_for = [&](DfsState& state,
                                                std::size_t task_pos,
                                                std::size_t device_idx,
                                                double current_makespan,
                                                double optimistic_bound) {
            DeviceOrderEntry entry;
            entry.device_idx = device_idx;

            double ready = state.available[device_idx];
            for (const auto edge_idx : dependency_edges_by_task[task_pos]) {
                const auto& edge = exhaustive_edges[edge_idx];
                const auto pred_idx = edge.src_task;
                const auto pred_device_idx = state.current_assignment[pred_idx];
                const auto& comm = edge.comm_by_pair[pred_device_idx * device_count + device_idx];
                if (!std::isfinite(comm.optimistic_time_s)) {
                    return entry;
                }
                ready = std::max(ready, state.finish_time[pred_idx] + comm.optimistic_time_s);
            }

            const double exec = compute_times[task_pos * device_count + device_idx];
            if (!std::isfinite(exec)) {
                return entry;
            }
            const double finish = ready + exec;
            const double next_makespan = std::max(current_makespan, finish);
            const double tail_bound =
                std::isfinite(optimistic_tail_after_task[task_pos])
                    ? finish + optimistic_tail_after_task[task_pos]
                    : kInf;

            entry.ready = ready;
            entry.finish = finish;
            entry.next_makespan = next_makespan;
            entry.score = std::max({optimistic_bound, next_makespan, tail_bound});
            return entry;
        };

        const std::size_t actual_thread_count =
            std::min<std::size_t>(exhaustive_thread_count, prefixes.size());
        if (collect_exhaustive_stats) {
            stats.worker_threads += actual_thread_count;
        }

        struct SharedExactStateCacheShard {
            std::mutex mutex;
            std::unordered_map<ExactStateKey, ExactStateDominance, ExactStateKeyHash> cache;
            std::size_t max_entries{0};
        };

        const bool use_shared_exact_state_cache =
            exact_cache_enabled && shared_exact_cache_enabled &&
            actual_thread_count > 1 && exact_state_cache_max_entries > 0;
        std::vector<std::unique_ptr<SharedExactStateCacheShard>> shared_exact_state_cache;
        if (use_shared_exact_state_cache) {
            const std::size_t shard_count =
                std::min<std::size_t>(
                    std::max<std::size_t>(1, requested_exact_cache_shards),
                    exact_state_cache_max_entries);
            shared_exact_state_cache.reserve(shard_count);
            const std::size_t base_entries = exact_state_cache_max_entries / shard_count;
            const std::size_t extra_entries = exact_state_cache_max_entries % shard_count;
            for (std::size_t shard_idx = 0; shard_idx < shard_count; ++shard_idx) {
                auto shard = std::make_unique<SharedExactStateCacheShard>();
                shard->max_entries = base_entries + (shard_idx < extra_entries ? 1 : 0);
                shard->cache.reserve(std::min<std::size_t>(shard->max_entries, 65536));
                shared_exact_state_cache.push_back(std::move(shard));
            }
        }

        std::atomic<std::size_t> next_prefix{0};
        std::vector<ExhaustiveSearchStats> local_stats(actual_thread_count);
        std::exception_ptr worker_exception;
        std::mutex worker_exception_mutex;

        auto worker_body = [&](std::size_t thread_idx) {
                auto& thread_stats = local_stats[thread_idx];
                std::unordered_map<ExactStateKey, ExactStateDominance, ExactStateKeyHash>
                    thread_exact_state_cache;
                if (!use_shared_exact_state_cache &&
                    exact_cache_enabled && exact_state_cache_max_entries > 0) {
                    thread_exact_state_cache.reserve(
                        std::min<std::size_t>(exact_state_cache_max_entries, 65536));
                }

                auto dominated_or_record_parallel_exact_state =
                    [&](ExactStateKey key,
                        double current_makespan,
                        double optimistic_bound) {
                    if (use_shared_exact_state_cache) {
                        const auto hash = ExactStateKeyHash{}(key);
                        auto& shard =
                            *shared_exact_state_cache[hash % shared_exact_state_cache.size()];
                        std::lock_guard<std::mutex> lock(shard.mutex);
                        return dominated_or_record_exact_state(shard.cache,
                                                               std::move(key),
                                                               current_makespan,
                                                               optimistic_bound,
                                                               thread_stats,
                                                               shard.max_entries);
                    }
                    return dominated_or_record_exact_state(thread_exact_state_cache,
                                                           std::move(key),
                                                           current_makespan,
                                                           optimistic_bound,
                                                           thread_stats,
                                                           exact_state_cache_max_entries);
                };

                std::function<void(DfsState&, std::size_t, double, double)> worker_search;
                worker_search = [&](DfsState& state,
                                    std::size_t task_pos,
                                    double current_makespan,
                                    double optimistic_bound) {
                    if (collect_exhaustive_stats) {
                        ++thread_stats.nodes_visited;
                        thread_stats.max_depth_reached =
                            std::max(thread_stats.max_depth_reached, task_pos);
                    }
                    progress_add(progress.nodes_visited);
                    progress_update_depth(task_pos);
                    if (task_pos == task_count) {
                        if (collect_exhaustive_stats) {
                            ++thread_stats.leaves_reached;
                        }
                        progress_add(progress.leaves_reached);
                        try_update_shared_best(current_makespan,
                                               state.current_assignment,
                                               thread_stats);
                        return;
                    }

                    const double incumbent = shared_best.load(std::memory_order_relaxed);
                    if (branch_and_bound_ && current_makespan > incumbent) {
                        if (collect_exhaustive_stats) {
                            ++thread_stats.pruned_by_current_makespan;
                        }
                        progress_add(progress.pruned_by_current_makespan);
                        return;
                    }
                    if (branch_and_bound_ && optimistic_bound > incumbent) {
                        if (collect_exhaustive_stats) {
                            ++thread_stats.pruned_by_optimistic_bound;
                        }
                        progress_add(progress.pruned_by_optimistic_bound);
                        return;
                    }
                    if (use_capacity_lower_bound) {
                        const double capacity_bound =
                            remaining_capacity_lower_bound_for(
                                state,
                                remaining_min_compute_suffix[task_pos]);
                        if (std::max({current_makespan, optimistic_bound, capacity_bound}) >
                            incumbent) {
                            if (collect_exhaustive_stats) {
                                ++thread_stats.pruned_by_capacity_bound;
                            }
                            progress_add(progress.pruned_by_capacity_bound);
                            return;
                        }
                    }

                    if (exact_cache_enabled && task_pos > 0 && task_pos < task_count) {
                        auto key = make_exact_state_key(task_pos,
                                                        state.current_assignment,
                                                        state.finish_time,
                                                        state.available,
                                                        state.comm_available,
                                                        state.link_available);
                        if (dominated_or_record_parallel_exact_state(std::move(key),
                                                                     current_makespan,
                                                                     optimistic_bound)) {
                            return;
                        }
                    }

                    auto visit_device = [&](std::size_t device_idx) {
                        const auto state_checkpoint = state.state_changes.size();
                        BranchCandidate candidate;
                        if (collect_exhaustive_stats) {
                            ++thread_stats.branch_attempts;
                        }
                        progress_add(progress.branch_attempts);
                        if (!evaluate_branch_candidate_for(state,
                                                           task_pos,
                                                           device_idx,
                                                           current_makespan,
                                                           optimistic_bound,
                                                           thread_stats,
                                                           candidate)) {
                            rollback_state_for(state, state_checkpoint);
                            return;
                        }
                        if (collect_exhaustive_stats) {
                            ++thread_stats.branch_accepted;
                        }
                        progress_add(progress.branch_accepted);

                        const double saved_available = state.available[device_idx];
                        const double saved_finish = state.finish_time[task_pos];
                        const auto saved_assignment = state.current_assignment[task_pos];
                        state.available[device_idx] = candidate.finish;
                        state.finish_time[task_pos] = candidate.finish;
                        state.current_assignment[task_pos] = device_idx;

                        worker_search(state,
                                      task_pos + 1,
                                      candidate.next_makespan,
                                      candidate.next_optimistic_bound);

                        state.current_assignment[task_pos] = saved_assignment;
                        state.finish_time[task_pos] = saved_finish;
                        state.available[device_idx] = saved_available;
                        rollback_state_for(state, state_checkpoint);
                    };

                    constexpr std::size_t kDynamicDeviceSortMaxDepth = 8;
                    if (use_dynamic_device_sort && task_pos < kDynamicDeviceSortMaxDepth) {
                        if (collect_exhaustive_stats) {
                            ++thread_stats.dynamic_sort_nodes;
                        }
                        std::vector<DeviceOrderEntry> candidates;
                        candidates.reserve(eligible_by_task[task_pos].size());
                        for (const auto device_idx : eligible_by_task[task_pos]) {
                            const auto candidate =
                                cheap_device_order_entry_for(state,
                                                             task_pos,
                                                             device_idx,
                                                             current_makespan,
                                                             optimistic_bound);
                            if (std::isfinite(candidate.score)) {
                                candidates.push_back(candidate);
                            }
                        }
                        std::sort(candidates.begin(),
                                  candidates.end(),
                                  [&](const auto& lhs, const auto& rhs) {
                                      if (lhs.score != rhs.score) {
                                          return lhs.score < rhs.score;
                                      }
                                      if (lhs.next_makespan != rhs.next_makespan) {
                                          return lhs.next_makespan < rhs.next_makespan;
                                      }
                                      if (lhs.finish != rhs.finish) {
                                          return lhs.finish < rhs.finish;
                                      }
                                      if (lhs.ready != rhs.ready) {
                                          return lhs.ready < rhs.ready;
                                      }
                                      return ordered_devices[lhs.device_idx]->id <
                                             ordered_devices[rhs.device_idx]->id;
                                  });
                        for (const auto& candidate : candidates) {
                            visit_device(candidate.device_idx);
                        }
                    } else {
                        for (const auto device_idx : eligible_by_task[task_pos]) {
                            visit_device(device_idx);
                        }
                    }
                };

                for (;;) {
                    const auto prefix_idx = next_prefix.fetch_add(1, std::memory_order_relaxed);
                    if (prefix_idx >= prefixes.size()) {
                        break;
                    }
                    const auto& prefix = prefixes[prefix_idx];
                    DfsState state;
                    state.current_assignment = prefix.assignment;
                    state.available = prefix.available;
                    state.finish_time = prefix.finish_time;
                    state.comm_available = prefix.comm_available;
                    state.link_available = prefix.link_available;
                    state.state_changes.reserve(std::max<std::size_t>(64, exhaustive_edges.size() * 4));
                    state.capacity_scratch.reserve(device_count);
                    worker_search(state,
                                  prefix.task_pos,
                                  prefix.current_makespan,
                                  prefix.optimistic_bound);
                    progress_add(progress.prefixes_done);
                }
        };

        auto pthread_trampoline = [](void* arg) -> void* {
            (*static_cast<std::function<void()>*>(arg))();
            return nullptr;
        };

        const std::size_t stack_mb =
            parse_size_env("TERRAPOD_EXHAUSTIVE_THREAD_STACK_MB", 64);
        const std::size_t requested_stack_size = stack_mb * 1024 * 1024;
        const std::size_t stack_size =
            std::max<std::size_t>(PTHREAD_STACK_MIN, requested_stack_size);

        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0) {
            throw std::runtime_error("Failed to initialize exhaustive worker thread attributes");
        }
        if (pthread_attr_setstacksize(&attr, stack_size) != 0) {
            pthread_attr_destroy(&attr);
            throw std::runtime_error("Failed to set exhaustive worker thread stack size");
        }

        std::vector<pthread_t> workers(actual_thread_count);
        std::vector<std::unique_ptr<std::function<void()>>> worker_functions;
        worker_functions.reserve(actual_thread_count);
        try {
            for (std::size_t thread_idx = 0; thread_idx < actual_thread_count; ++thread_idx) {
                auto fn = std::make_unique<std::function<void()>>([&, thread_idx]() {
                    try {
                        worker_body(thread_idx);
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(worker_exception_mutex);
                        if (!worker_exception) {
                            worker_exception = std::current_exception();
                        }
                    }
                });
                const int rc = pthread_create(&workers[thread_idx],
                                              &attr,
                                              pthread_trampoline,
                                              fn.get());
                if (rc != 0) {
                    throw std::runtime_error("Failed to create exhaustive worker thread: " +
                                             std::to_string(rc));
                }
                worker_functions.push_back(std::move(fn));
            }
        } catch (...) {
            pthread_attr_destroy(&attr);
            throw;
        }
        pthread_attr_destroy(&attr);

        for (auto& worker : workers) {
            pthread_join(worker, nullptr);
        }
        if (worker_exception) {
            std::rethrow_exception(worker_exception);
        }
        if (collect_exhaustive_stats) {
            for (const auto& local : local_stats) {
                add_stats(stats, local);
            }
        }
    };

    if (exhaustive_thread_count > 1) {
        run_parallel_search();
    } else {
        search(0, 0.0, 0.0);
    }
    stop_progress_reporter.store(true, std::memory_order_relaxed);
    if (progress_reporter.joinable()) {
        progress_reporter.join();
    }
    if (collect_exhaustive_stats) {
        std::cerr << "[exhaustive stats] tasks=" << task_count
                  << " devices=" << device_count
                  << " worker_threads=" << stats.worker_threads
                  << " prefixes=" << stats.prefixes_created
                  << " best_makespan=" << best_makespan
                  << " best_updates=" << stats.best_updates
                  << " assignment_evaluations=" << stats.assignment_evaluations
                  << "\n";
        std::cerr << "[exhaustive stats] task_device_choices initial="
                  << stats.initial_task_device_choices
                  << " prepruned=" << stats.prepruned_task_device_choices
                  << " remaining=" << stats.remaining_task_device_choices
                  << "\n";
        std::cerr << "[exhaustive stats] search nodes=" << stats.nodes_visited
                  << " leaves=" << stats.leaves_reached
                  << " max_depth=" << stats.max_depth_reached
                  << " branch_attempts=" << stats.branch_attempts
                  << " branch_accepted=" << stats.branch_accepted
                  << " dynamic_sort_nodes=" << stats.dynamic_sort_nodes
                  << "\n";
        std::cerr << "[exhaustive stats] prunes current_makespan="
                  << stats.pruned_by_current_makespan
                  << " optimistic_bound=" << stats.pruned_by_optimistic_bound
                  << " capacity_bound=" << stats.pruned_by_capacity_bound
                  << " candidate_makespan=" << stats.pruned_by_candidate_makespan
                  << " candidate_bound=" << stats.pruned_by_candidate_bound
                  << " infeasible_candidates=" << stats.infeasible_candidates
                  << "\n";
        std::cerr << "[exhaustive stats] exact_cache assignment_hits="
                  << stats.exact_assignment_cache_hits
                  << " assignment_records=" << stats.exact_assignment_cache_records
                  << " state_queries=" << stats.exact_state_cache_queries
                  << " state_hits=" << stats.exact_state_cache_hits
                  << " state_records=" << stats.exact_state_cache_records
                  << " state_saturated=" << stats.exact_state_cache_saturated
                  << "\n";
    }
    if (!have_best_assignment || !std::isfinite(best_makespan)) {
        throw std::runtime_error("No feasible exhaustive mapping found");
    }

    MappingPlan plan;
    plan.assignments.reserve(topo.size());
    for (std::size_t i = 0; i < topo.size(); ++i) {
        plan.assignments[topo[i].name] = ordered_devices[best_assignment[i]]->id;
    }
    return plan;
}

PartitionerMapper::PartitionerMapper(std::unique_ptr<Mapper> inner, std::vector<std::vector<std::string>> partitions)
    : inner_(std::move(inner)), partitions_(std::move(partitions)) {}

MappingPlan PartitionerMapper::map(const TaskGraph& graph, const hardware_topology::HardwareTopology& topology) const {
    TaskGraph reordered;
    const auto reordered_tasks = order(graph);
    for (const auto& task : reordered_tasks) {
        reordered.add_task(task);
    }
    for (const auto& task : reordered_tasks) {
        for (const auto& edge : graph.successors(task.name)) {
            reordered.add_edge(edge.src,
                               edge.dst,
                               edge.tensor_bytes,
                               edge.tensor_id,
                               edge.comm_kind,
                               edge.access_pattern,
                               edge.comm_participants,
                               edge.comm_group,
                               edge.dtype);
        }
    }
    return inner_->map(reordered, topology);
}

std::vector<Task> PartitionerMapper::order(const TaskGraph& graph) const {
    std::vector<Task> ordered;
    std::unordered_set<std::string> seen;
    for (const auto& block : partitions_) {
        for (const auto& name : block) {
            if (!graph.has_task(name)) {
                continue;
            }
            if (seen.insert(name).second) {
                ordered.push_back(graph.task(name));
            }
        }
    }
    for (const auto& task : graph.topological_order()) {
        if (seen.insert(task.name).second) {
            ordered.push_back(task);
        }
    }
    return ordered;
}

}  // namespace mapping
