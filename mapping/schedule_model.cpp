#include "mapping/schedule_model.h"

#include "mapping/cost_model.h"
#include "mapping/mapper.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mapping {
namespace {

struct LinkStats {
    double avg_bw_gbps{0.0};
    double avg_latency_s{0.0};
    double avg_extra_latency_s{0.0};
    std::size_t link_count{0};
};

struct CollectiveScheduleGroup {
    std::string key;
    TaskEdge representative;
    std::vector<std::string> sources;
    std::vector<std::string> destinations;
};

struct LinkStatsCache {
    std::unordered_map<std::string, LinkStats> by_communication_group;
};

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

void add_unique_string(std::vector<std::string>& values, const std::string& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

std::string collective_schedule_key(const TaskEdge& edge) {
    const auto kind = canonical_comm_kind(edge.comm_kind);
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

std::unordered_map<std::string, CollectiveScheduleGroup> build_collective_schedule_groups(
    const TaskGraph& graph) {
    std::unordered_map<std::string, CollectiveScheduleGroup> groups;
    for (const auto& task : graph.topological_order()) {
        for (const auto& edge : graph.successors(task.name)) {
            if (!is_collective_kind(edge.comm_kind)) {
                continue;
            }
            const auto key = collective_schedule_key(edge);
            auto& group = groups[key];
            if (group.key.empty()) {
                group.key = key;
                group.representative = edge;
            }
            add_unique_string(group.sources, edge.src);
            add_unique_string(group.destinations, edge.dst);
        }
    }
    return groups;
}

LinkStats average_link_stats(const hardware_topology::HardwareTopology& topology,
                             const std::string& communication_group = "native") {
    const auto& links = topology.links();
    if (links.empty()) {
        return {};
    }
    constexpr std::size_t kProbeBytes = 1'000'000'000;
    double bw_sum = 0.0;
    double lat_s_sum = 0.0;
    std::size_t count = 0;
    for (const auto& link : links) {
        if (link.bw_gbps <= 0.0) {
            continue;
        }
        const double zero_s =
            topology.link_transfer_time_seconds(link, link.src, link.dst, 0, communication_group);
        const double probe_s =
            topology.link_transfer_time_seconds(link, link.src, link.dst, kProbeBytes, communication_group);
        if (std::isfinite(zero_s) && std::isfinite(probe_s) && probe_s > zero_s) {
            bw_sum += static_cast<double>(kProbeBytes) / ((probe_s - zero_s) * 1e9);
            lat_s_sum += zero_s;
        } else {
            bw_sum += link.bw_gbps;
            lat_s_sum += link.latency_ms / 1000.0;
        }
        count += 1;
    }
    if (count == 0) {
        return {};
    }
    return LinkStats{
        bw_sum / static_cast<double>(count),
        lat_s_sum / static_cast<double>(count),
        topology.average_endpoint_extra_latency_seconds(),
        count,
    };
}

const LinkStats& cached_average_link_stats(LinkStatsCache& cache,
                                           const hardware_topology::HardwareTopology& topology,
                                           const std::string& communication_group) {
    auto [it, inserted] = cache.by_communication_group.try_emplace(communication_group);
    if (inserted) {
        it->second = average_link_stats(topology, communication_group);
    }
    return it->second;
}

double allreduce_time_seconds(double bytes,
                              std::size_t participants,
                              const LinkStats& stats) {
    if (participants <= 1) {
        return 0.0;
    }
    const double payload = std::max(0.0, bytes) / 1e9;
    if (stats.link_count == 0 || (payload > 0.0 && stats.avg_bw_gbps <= 0.0)) {
        return std::numeric_limits<double>::infinity();
    }
    const double alpha = stats.avg_latency_s + stats.avg_extra_latency_s;
    const double beta = payload > 0.0 ? payload / stats.avg_bw_gbps : 0.0;
    const double p = static_cast<double>(participants);
    return 2.0 * (p - 1.0) * alpha + 2.0 * ((p - 1.0) / p) * beta;
}

double estimate_collective_time_seconds_from_stats(const std::string& kind,
                                                   double bytes,
                                                   std::size_t participants,
                                                   const LinkStats& stats) {
    const double payload = std::max(0.0, bytes) / 1e9;
    if (stats.link_count == 0 || (payload > 0.0 && stats.avg_bw_gbps <= 0.0)) {
        return std::numeric_limits<double>::infinity();
    }

    const double alpha = stats.avg_latency_s + stats.avg_extra_latency_s;
    const double beta = payload > 0.0 ? payload / stats.avg_bw_gbps : 0.0;
    const double p = static_cast<double>(participants);

    if (kind == "allreduce") {
        return allreduce_time_seconds(bytes, participants, stats);
    }
    if (kind == "allgather" || kind == "reducescatter") {
        return (p - 1.0) * alpha + ((p - 1.0) / p) * beta;
    }
    if (kind == "broadcast") {
        const double stages = std::ceil(std::log2(p));
        return stages * alpha + beta;
    }
    if (kind == "alltoall") {
        return (p - 1.0) * alpha + ((p - 1.0) / p) * beta;
    }
    return std::numeric_limits<double>::infinity();
}

double estimate_collective_time_seconds_cached(const std::string& comm_kind,
                                               double bytes,
                                               std::size_t participants,
                                               const hardware_topology::HardwareTopology& topology,
                                               LinkStatsCache& cache) {
    if (participants <= 1) {
        return 0.0;
    }
    const std::string kind = canonical_comm_kind(comm_kind);
    const std::string scale_group =
        (kind == "allreduce" || kind == "allgather" || kind == "reducescatter" || kind == "alltoall")
            ? kind
            : "generic";
    const auto& stats = cached_average_link_stats(cache, topology, scale_group);
    return estimate_collective_time_seconds_from_stats(kind, bytes, participants, stats);
}

std::string abstract_collective_kind(const std::string& comm_kind) {
    const auto kind = canonical_comm_kind(comm_kind);
    return is_collective_kind(kind) ? kind : "alltoall";
}

double direct_transfer_time_seconds(const hardware_topology::HardwareTopology& topology,
                                    const std::string& src,
                                    const std::string& dst,
                                    double bytes) {
    if (src == dst) {
        return 0.0;
    }
    const auto id = topology.link_id(src, dst);
    if (!id.has_value()) {
        return std::numeric_limits<double>::infinity();
    }
    const auto* link = topology.link_by_id(*id);
    if (link == nullptr) {
        return std::numeric_limits<double>::infinity();
    }
    return topology.link_transfer_time_seconds(*link, src, dst, bytes_to_size_t(bytes), "native", 1) +
           topology.endpoint_extra_latency_seconds(src, dst);
}

const hardware_topology::Link* find_link_by_id(const hardware_topology::HardwareTopology& topology,
                                               const std::string& id) {
    return topology.link_by_id(id);
}

}  // namespace

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

bool is_collective_kind(const std::string& kind) {
    const auto k = canonical_comm_kind(kind);
    return k == "allreduce" || k == "allgather" || k == "reducescatter" || k == "broadcast" || k == "alltoall";
}

double estimate_collective_time_seconds(const std::string& comm_kind,
                                        double bytes,
                                        std::size_t participants,
                                        const hardware_topology::HardwareTopology& topology) {
    if (participants <= 1) {
        return 0.0;
    }
    const std::string kind = canonical_comm_kind(comm_kind);
    const std::string scale_group =
        (kind == "allreduce" || kind == "allgather" || kind == "reducescatter" || kind == "alltoall")
            ? kind
            : "generic";
    const auto stats = average_link_stats(topology, scale_group);
    return estimate_collective_time_seconds_from_stats(kind, bytes, participants, stats);
}

double estimate_communication_time_seconds(const TaskEdge& edge,
                                           const hardware_topology::HardwareTopology& topology,
                                           const std::string& src,
                                           const std::string& dst) {
    if (is_collective_kind(edge.comm_kind)) {
        return estimate_collective_time_seconds(edge.comm_kind,
                                                edge.tensor_bytes,
                                                std::max<std::size_t>(2, edge.comm_participants),
                                                topology);
    }
    if (src == dst) {
        return 0.0;
    }
    if (topology.link_id(src, dst).has_value()) {
        return direct_transfer_time_seconds(topology, src, dst, edge.tensor_bytes);
    }
    const double routed = topology.get_transfer_time(src, dst, static_cast<size_t>(std::max(0.0, edge.tensor_bytes)));
    if (std::isfinite(routed)) {
        return routed;
    }
    return estimate_collective_time_seconds(abstract_collective_kind(edge.comm_kind), edge.tensor_bytes, 2, topology);
}

double estimate_average_communication_time_seconds(
    const TaskEdge& edge,
    const std::vector<const hardware_topology::Device*>& devices,
    const hardware_topology::HardwareTopology& topology) {
    if (devices.size() < 2) {
        return 0.0;
    }
    double total = 0.0;
    std::size_t count = 0;
    for (const auto* src : devices) {
        for (const auto* dst : devices) {
            if (src->id == dst->id) {
                continue;
            }
            const double time = estimate_communication_time_seconds(edge, topology, src->id, dst->id);
            if (!std::isfinite(time)) {
                continue;
            }
            total += time;
            count += 1;
        }
    }
    if (count == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return total / static_cast<double>(count);
}

double schedule_p2p_transfer_seconds(const hardware_topology::HardwareTopology& topology,
                                     const std::string& src,
                                     const std::string& dst,
                                     double bytes,
                                     double earliest_start,
                                     std::unordered_map<std::string, double>& comm_available,
                                     std::unordered_map<std::string, double>& link_available) {
    if (src == dst) {
        return earliest_start;
    }

    const auto route = topology.shortest_route_link_ids(src, dst, static_cast<std::size_t>(std::max(0.0, bytes)));
    if (route.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    double now = std::max(earliest_start, comm_available[src]) + topology.endpoint_extra_latency_seconds(src, dst);
    for (const auto& link_id : route) {
        const auto* link = find_link_by_id(topology, link_id);
        if (link == nullptr) {
            return std::numeric_limits<double>::infinity();
        }
        const auto timing = topology.link_transfer_timing_seconds(*link,
                                                                  src,
                                                                  dst,
                                                                  bytes_to_size_t(bytes),
                                                                  "native",
                                                                  route.size());
        if (!std::isfinite(timing.elapsed_s) || !std::isfinite(timing.link_busy_s)) {
            return std::numeric_limits<double>::infinity();
        }
        const double start = std::max(now, link_available[link_id]);
        const double finish = start + timing.elapsed_s;
        link_available[link_id] = start + timing.link_busy_s;
        now = finish;
    }

    // Match terrapod-sim's single send slot: a source cannot issue another p2p send until this one finishes.
    comm_available[src] = now;
    return now;
}

double estimate_makespan_seconds(const TaskGraph& graph,
                                 const MappingPlan& plan,
                                 const hardware_topology::HardwareTopology& topology) {
    const auto& devices = topology.compute_devices();
    if (devices.empty()) {
        return std::numeric_limits<double>::infinity();
    }

    std::unordered_map<std::string, double> available;
    for (const auto* device : devices) {
        available[device->id] = 0.0;
    }
    std::unordered_map<std::string, double> comm_available;
    for (const auto* device : topology.devices()) {
        comm_available[device->id] = 0.0;
    }
    std::unordered_map<std::string, double> link_available;
    for (const auto& link : topology.links()) {
        link_available[link.id] = 0.0;
    }
    std::unordered_map<std::string, double> finish_time;
    const auto collective_groups = build_collective_schedule_groups(graph);
    LinkStatsCache collective_cost_cache;

    double makespan = 0.0;
    for (const auto& task : graph.topological_order()) {
        const auto map_it = plan.assignments.find(task.name);
        if (map_it == plan.assignments.end()) {
            return std::numeric_limits<double>::infinity();
        }
        const auto& device_id = map_it->second;
        const auto* device = topology.device(device_id);
        if (device == nullptr) {
            return std::numeric_limits<double>::infinity();
        }

        double ready = available[device_id];
        std::unordered_set<std::string> handled_collective_groups;
        for (const auto& edge : graph.dependencies(task.name)) {
            const auto pred_assign = plan.assignments.find(edge.src);
            const auto pred_finish = finish_time.find(edge.src);
            if (pred_assign == plan.assignments.end() || pred_finish == finish_time.end()) {
                return std::numeric_limits<double>::infinity();
            }

            double comm = 0.0;
            if (pred_assign->second != device_id || is_collective_kind(edge.comm_kind)) {
                if (is_collective_kind(edge.comm_kind)) {
                    const auto group_key = collective_schedule_key(edge);
                    if (!handled_collective_groups.insert(group_key).second) {
                        continue;
                    }
                    const auto group_it = collective_groups.find(group_key);
                    if (group_it == collective_groups.end()) {
                        comm = estimate_communication_time_seconds(edge, topology, pred_assign->second, device_id);
                        ready = std::max(ready, pred_finish->second + comm);
                        continue;
                    }
                    double group_ready = 0.0;
                    for (const auto& src : group_it->second.sources) {
                        const auto src_finish = finish_time.find(src);
                        if (src_finish == finish_time.end()) {
                            return std::numeric_limits<double>::infinity();
                        }
                        group_ready = std::max(group_ready, src_finish->second);
                    }
                    comm = estimate_collective_time_seconds_cached(group_it->second.representative.comm_kind,
                                                                   group_it->second.representative.tensor_bytes,
                                                                   std::max<std::size_t>(
                                                                       2,
                                                                       group_it->second.representative.comm_participants),
                                                                   topology,
                                                                   collective_cost_cache);
                    ready = std::max(ready, group_ready + comm);
                    continue;
                }
                const double comm_finish = schedule_p2p_transfer_seconds(topology,
                                                                         pred_assign->second,
                                                                         device_id,
                                                                         edge.tensor_bytes,
                                                                         pred_finish->second,
                                                                         comm_available,
                                                                         link_available);
                if (!std::isfinite(comm_finish)) {
                    comm = estimate_communication_time_seconds(edge, topology, pred_assign->second, device_id);
                    ready = std::max(ready, pred_finish->second + comm);
                    continue;
                }
                ready = std::max(ready, comm_finish);
                continue;
            }
            ready = std::max(ready, pred_finish->second + comm);
        }

        const double finish = ready + estimate_task_time_seconds(task, device);
        available[device_id] = finish;
        finish_time[task.name] = finish;
        makespan = std::max(makespan, finish);
    }

    return makespan;
}

}  // namespace mapping
