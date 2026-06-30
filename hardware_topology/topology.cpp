#include "hardware_topology/topology.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <queue>
#include <limits>
#include <string>
#include <unordered_map>

namespace hardware_topology {
namespace {

std::string canonical_device_type(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    return value;
}

bool is_gpu_endpoint(const Device* device) {
    return device != nullptr && canonical_device_type(device->type) == "gpu";
}

bool is_cpu_endpoint(const Device* device) {
    return device != nullptr && canonical_device_type(device->type) == "cpu";
}

bool is_intra_machine_pair(const Device* src, const Device* dst) {
    return src != nullptr && dst != nullptr && !src->parent.empty() && src->parent == dst->parent;
}

std::string canonical_communication_group(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (ch == '-' || ch == ' ') {
            ch = '_';
        }
    }
    if (value.empty() || value == "p2p" || value == "send" || value == "recv") {
        return "native";
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

struct EndpointPairLookup {
    std::string exact_key;
    std::string type_prefix;
    bool has_machine_scope{false};
};

EndpointPairLookup endpoint_pair_lookup(const Device* src, const Device* dst) {
    EndpointPairLookup lookup;
    const bool src_gpu = is_gpu_endpoint(src);
    const bool dst_gpu = is_gpu_endpoint(dst);
    const bool src_cpu = is_cpu_endpoint(src);
    const bool dst_cpu = is_cpu_endpoint(dst);
    if (src_gpu && dst_gpu) {
        lookup.type_prefix = "gpu-gpu-";
    } else if ((src_gpu && dst_cpu) || (src_cpu && dst_gpu)) {
        lookup.type_prefix = "gpu-cpu-";
    } else if (src_cpu && dst_cpu) {
        lookup.type_prefix = "cpu-cpu-";
    } else {
        return lookup;
    }

    lookup.has_machine_scope =
        src != nullptr && dst != nullptr && !src->parent.empty() && !dst->parent.empty();
    if (lookup.has_machine_scope) {
        lookup.exact_key = lookup.type_prefix +
                           (src->parent == dst->parent ? "intra-machine" : "inter-machine");
    }
    return lookup;
}

const CommunicationCostScale* scale_for_route_hops(const CommunicationRouteHopScaleTable& table,
                                                   std::size_t route_hops) {
    const auto it = table.find(route_hops);
    if (it == table.end()) {
        return nullptr;
    }
    return &it->second;
}

const CommunicationCostScale* scale_for_endpoint_pair(const CommunicationEndpointPairScaleTable& table,
                                                      const EndpointPairLookup& lookup,
                                                      std::size_t route_hops) {
    if (lookup.has_machine_scope && !lookup.exact_key.empty()) {
        const auto exact_it = table.find(lookup.exact_key);
        if (exact_it != table.end()) {
            if (const auto* scale = scale_for_route_hops(exact_it->second, route_hops); scale != nullptr) {
                return scale;
            }
        }
    }

    if (lookup.type_prefix.empty()) {
        return nullptr;
    }
    const CommunicationCostScale* unique = nullptr;
    std::size_t matches = 0;
    for (const auto& kv : table) {
        if (!kv.first.starts_with(lookup.type_prefix)) {
            continue;
        }
        const auto* scale = scale_for_route_hops(kv.second, route_hops);
        if (scale == nullptr) {
            continue;
        }
        unique = scale;
        ++matches;
        if (matches > 1) {
            return nullptr;
        }
    }
    return unique;
}

const CommunicationCostScale* communication_scale_for_hardware_link(
    const CommunicationHardwareLinkScaleTable& table,
    const std::string& hardware_link_id,
    const std::string& communication_group,
    const EndpointPairLookup& endpoint_pair,
    std::size_t route_hops) {
    const auto hardware_it = table.find(hardware_link_id);
    if (hardware_it == table.end()) {
        return nullptr;
    }
    const auto& groups = hardware_it->second;
    const auto group_it = groups.find(communication_group);
    if (group_it != groups.end()) {
        if (const auto* scale = scale_for_endpoint_pair(group_it->second, endpoint_pair, route_hops);
            scale != nullptr) {
            return scale;
        }
    }
    if (communication_group != "native") {
        const auto generic_it = groups.find("generic");
        if (generic_it != groups.end()) {
            return scale_for_endpoint_pair(generic_it->second, endpoint_pair, route_hops);
        }
    }
    return nullptr;
}

struct CommunicationAdjustment {
    double bandwidth_scale{1.0};
    double extra_latency_s{0.0};
    bool has_bandwidth_scale{false};
    bool matched_any_scale{false};

    void apply(const CommunicationCostScale& scale) {
        matched_any_scale = true;
        if (scale.bandwidth_scale.has_value()) {
            bandwidth_scale = has_bandwidth_scale ? std::min(bandwidth_scale, *scale.bandwidth_scale)
                                                  : *scale.bandwidth_scale;
            has_bandwidth_scale = true;
        }
        extra_latency_s += scale.extra_latency_us / 1e6;
    }
};

std::string direct_link_key(std::string_view src, std::string_view dst) {
    std::string key;
    key.reserve(src.size() + 1 + dst.size());
    key.append(src);
    key.push_back('\0');
    key.append(dst);
    return key;
}

}  // namespace

std::size_t HardwareTopology::TransferCacheKeyHash::operator()(const TransferCacheKey& key) const {
    std::size_t h = std::hash<std::string>{}(key.src);
    h ^= std::hash<std::string>{}(key.dst) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::uint64_t>{}(key.bytes) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<std::string>{}(key.communication_group) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

void HardwareTopology::set_time_unit(std::string time_unit) {
    time_unit_ = std::move(time_unit);
}

const std::string& HardwareTopology::time_unit() const {
    return time_unit_;
}

void HardwareTopology::add_device(Device device) {
    apply_operator_cost_scales(device);
    devices_.insert_or_assign(device.id, std::move(device));
    invalidate_caches();
}

void HardwareTopology::add_link(Link link) {
    if (link.id.empty()) {
        link.id = "link_" + link.src + "_to_" + link.dst;
    }
    links_.push_back(std::move(link));
    invalidate_caches();
}

const Device* HardwareTopology::device(std::string_view id) const {
    const auto it = devices_.find(std::string(id));
    if (it == devices_.end()) {
        return nullptr;
    }
    return &it->second;
}

const std::vector<const Device*>& HardwareTopology::devices() const {
    if (!devices_cache_valid_) {
        rebuild_device_cache();
    }
    return devices_cache_;
}

const std::vector<const Device*>& HardwareTopology::compute_devices() const {
    if (!devices_cache_valid_) {
        rebuild_device_cache();
    }
    return compute_devices_cache_;
}

const std::vector<Link>& HardwareTopology::links() const {
    if (!links_cache_valid_) {
        rebuild_link_caches();
    }
    return links_cache_;
}

std::optional<double> HardwareTopology::bw_gbps(std::string_view src, std::string_view dst) const {
    if (!links_cache_valid_) {
        rebuild_link_caches();
    }
    const auto it = direct_link_cache_.find(direct_link_key(src, dst));
    if (it == direct_link_cache_.end()) {
        return std::nullopt;
    }
    return it->second->bw_gbps;
}

std::optional<double> HardwareTopology::latency_ms(std::string_view src, std::string_view dst) const {
    if (!links_cache_valid_) {
        rebuild_link_caches();
    }
    const auto it = direct_link_cache_.find(direct_link_key(src, dst));
    if (it == direct_link_cache_.end()) {
        return std::nullopt;
    }
    return it->second->latency_ms;
}

std::optional<std::string> HardwareTopology::link_id(std::string_view src, std::string_view dst) const {
    if (!links_cache_valid_) {
        rebuild_link_caches();
    }
    const auto it = direct_link_cache_.find(direct_link_key(src, dst));
    if (it == direct_link_cache_.end()) {
        return std::nullopt;
    }
    return it->second->id;
}

const Link* HardwareTopology::link_by_id(std::string_view id) const {
    if (!links_cache_valid_) {
        rebuild_link_caches();
    }
    const auto it = link_by_id_cache_.find(std::string(id));
    if (it == link_by_id_cache_.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<std::string> HardwareTopology::shortest_route_link_ids(std::string_view src,
                                                                    std::string_view dst,
                                                                    size_t bytes) const {
    if (src == dst) {
        return {};
    }
    const auto* src_dev = device(src);
    const auto* dst_dev = device(dst);
    if (src_dev == nullptr || dst_dev == nullptr) {
        return {};
    }

    if (!links_cache_valid_) {
        rebuild_link_caches();
    }

    const auto link_cost_seconds = [&](const Link& link) {
        const auto route_hops = link.hardware_route.empty() ? std::size_t{1} : link.hardware_route.size();
        return link_transfer_time_seconds(link, link.src, link.dst, bytes, "native", route_hops);
    };

    const double kInf = std::numeric_limits<double>::infinity();
    std::unordered_map<std::string, double> dist;
    std::unordered_map<std::string, std::string> parent_node;
    std::unordered_map<std::string, std::string> parent_link;
    dist.reserve(devices_.size());
    parent_node.reserve(devices_.size());
    parent_link.reserve(devices_.size());
    for (const auto& device_ptr : devices()) {
        dist.emplace(device_ptr->id, kInf);
    }
    dist[src_dev->id] = 0.0;

    using QueueItem = std::pair<double, std::string>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> pq;
    pq.push({0.0, src_dev->id});

    while (!pq.empty()) {
        const auto [d, current] = pq.top();
        pq.pop();
        if (d > dist[current]) {
            continue;
        }
        if (current == dst_dev->id) {
            break;
        }
        const auto out_it = outgoing_cache_.find(current);
        if (out_it == outgoing_cache_.end()) {
            continue;
        }
        for (const auto* link : out_it->second) {
            const double weight = link_cost_seconds(*link);
            const double candidate = d + weight;
            const auto next_it = dist.find(link->dst);
            if (next_it == dist.end()) {
                continue;
            }
            const double old = next_it->second;
            const bool better = candidate < old;
            const bool tie_break = (candidate == old &&
                                    parent_link.find(link->dst) != parent_link.end() &&
                                    link->id < parent_link[link->dst]);
            if (better || tie_break) {
                next_it->second = candidate;
                parent_node[link->dst] = current;
                parent_link[link->dst] = link->id;
                pq.push({candidate, link->dst});
            }
        }
    }

    if (parent_node.find(dst_dev->id) == parent_node.end()) {
        return {};
    }

    std::vector<std::string> route;
    for (std::string cur = dst_dev->id; cur != src_dev->id;) {
        const auto link_it = parent_link.find(cur);
        const auto node_it = parent_node.find(cur);
        if (link_it == parent_link.end() || node_it == parent_node.end()) {
            return {};
        }
        route.push_back(link_it->second);
        cur = node_it->second;
    }
    std::reverse(route.begin(), route.end());
    return route;
}

double HardwareTopology::get_transfer_time(std::string_view src, std::string_view dst, size_t bytes) const {
    return get_transfer_time(src, dst, bytes, "native");
}

double HardwareTopology::get_transfer_time(std::string_view src,
                                           std::string_view dst,
                                           size_t bytes,
                                           std::string_view communication_group) const {
    if (src == dst) {
        return 0.0;
    }
    const std::string group = canonical_communication_group(std::string(communication_group));
    TransferCacheKey key{
        std::string(src),
        std::string(dst),
        static_cast<std::uint64_t>(bytes),
        group,
    };
    const auto cached = transfer_time_cache_.find(key);
    if (cached != transfer_time_cache_.end()) {
        return cached->second;
    }

    const auto route = shortest_route_link_ids(src, dst, bytes);
    double route_total = 0.0;
    if (route.empty()) {
        route_total = std::numeric_limits<double>::infinity();
    } else {
        const std::size_t route_hops = route.size();
        for (const auto& link_id : route) {
            const auto* link = link_by_id(link_id);
            if (link == nullptr) {
                route_total = std::numeric_limits<double>::infinity();
                break;
            }
            const double hop_time = link_transfer_time_seconds(*link, src, dst, bytes, group, route_hops);
            if (!std::isfinite(hop_time)) {
                route_total = std::numeric_limits<double>::infinity();
                break;
            }
            route_total += hop_time;
        }
    }
    const double total = std::isfinite(route_total) ? route_total + endpoint_extra_latency_seconds(src, dst)
                                                    : route_total;
    transfer_time_cache_.emplace(std::move(key), total);
    return total;
}

HardwareTopology::LinkTransferTiming HardwareTopology::link_transfer_timing_seconds(
    const Link& link,
    std::string_view src,
    std::string_view dst,
    size_t bytes,
    std::string_view communication_group,
    std::size_t route_hops) const {
    if (link.bw_gbps <= 0.0) {
        return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    }
    if (!link.hardware_route.empty() && route_hops < link.hardware_route.size()) {
        route_hops = link.hardware_route.size();
    }
    if (route_hops == 0) {
        route_hops = link.hardware_route.empty() ? 1 : link.hardware_route.size();
    }

    const auto* src_dev = device(src);
    const auto* dst_dev = device(dst);
    const auto endpoint_pair = endpoint_pair_lookup(src_dev, dst_dev);
    const std::string group = canonical_communication_group(std::string(communication_group));
    CommunicationAdjustment adjustment;

    const auto apply_for_hardware_id = [&](const std::string& hardware_link_id) {
        const auto* scale = communication_scale_for_hardware_link(
            link.hardware_route_communication_scales,
            hardware_link_id,
            group,
            endpoint_pair,
            route_hops);
        if (scale != nullptr) {
            adjustment.apply(*scale);
        }
    };

    if (link.hardware_route.empty()) {
        apply_for_hardware_id(link.id);
        constexpr std::string_view reverse_suffix = "_rev";
        if (!adjustment.matched_any_scale && link.id.ends_with(reverse_suffix)) {
            apply_for_hardware_id(link.id.substr(0, link.id.size() - reverse_suffix.size()));
        }
    } else {
        for (const auto& hardware_link_id : link.hardware_route) {
            apply_for_hardware_id(hardware_link_id);
        }
        if (!adjustment.matched_any_scale) {
            apply_for_hardware_id(link.id);
        }
    }

    const double payload = static_cast<double>(bytes);
    const double bandwidth_scale = adjustment.has_bandwidth_scale ? adjustment.bandwidth_scale : 1.0;
    if (payload > 0.0 && bandwidth_scale <= 0.0) {
        return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    }
    const double latency_s = link.latency_ms / 1000.0;
    const double serialize_s =
        payload > 0.0 ? payload / (link.bw_gbps * bandwidth_scale * 1e9) : 0.0;
    const double link_busy_s = latency_s + serialize_s;
    return {link_busy_s + adjustment.extra_latency_s, link_busy_s};
}

double HardwareTopology::link_transfer_time_seconds(const Link& link,
                                                    std::string_view src,
                                                    std::string_view dst,
                                                    size_t bytes,
                                                    std::string_view communication_group,
                                                    std::size_t route_hops) const {
    return link_transfer_timing_seconds(link, src, dst, bytes, communication_group, route_hops).elapsed_s;
}

double HardwareTopology::endpoint_extra_latency_seconds(std::string_view src, std::string_view dst) const {
    if (src == dst) {
        return 0.0;
    }
    const auto* src_dev = device(src);
    const auto* dst_dev = device(dst);
    if (src_dev == nullptr || dst_dev == nullptr) {
        return 0.0;
    }

    const bool src_gpu = is_gpu_endpoint(src_dev);
    const bool dst_gpu = is_gpu_endpoint(dst_dev);
    const bool src_cpu = is_cpu_endpoint(src_dev);
    const bool dst_cpu = is_cpu_endpoint(dst_dev);
    const bool intra = is_intra_machine_pair(src_dev, dst_dev);

    double latency_us = 0.0;
    if (src_gpu && dst_gpu) {
        latency_us = intra ? endpoint_extra_latency_config_.gpu_gpu_intra_machine_us
                           : endpoint_extra_latency_config_.gpu_gpu_inter_machine_us;
    } else if ((src_gpu && dst_cpu) || (src_cpu && dst_gpu)) {
        latency_us = intra ? endpoint_extra_latency_config_.gpu_cpu_intra_machine_us
                           : endpoint_extra_latency_config_.gpu_cpu_inter_machine_us;
    } else if (src_cpu && dst_cpu) {
        latency_us = intra ? endpoint_extra_latency_config_.cpu_cpu_intra_machine_us
                           : endpoint_extra_latency_config_.cpu_cpu_inter_machine_us;
    }
    return latency_us / 1e6;
}

double HardwareTopology::average_endpoint_extra_latency_seconds() const {
    const auto& endpoints = compute_devices();
    if (endpoints.size() < 2) {
        return 0.0;
    }
    double total = 0.0;
    std::size_t count = 0;
    for (const auto* src : endpoints) {
        for (const auto* dst : endpoints) {
            if (src->id == dst->id) {
                continue;
            }
            total += endpoint_extra_latency_seconds(src->id, dst->id);
            count += 1;
        }
    }
    return count == 0 ? 0.0 : total / static_cast<double>(count);
}

void HardwareTopology::set_endpoint_extra_latency_config(EndpointExtraLatencyConfig config) {
    endpoint_extra_latency_config_ = config;
    transfer_time_cache_.clear();
}

const HardwareTopology::EndpointExtraLatencyConfig& HardwareTopology::endpoint_extra_latency_config() const {
    return endpoint_extra_latency_config_;
}

void HardwareTopology::set_cpu_operator_scales(OperatorCostScaleTable scales) {
    cpu_operator_scales_ = std::move(scales);
    for (auto& kv : devices_) {
        apply_operator_cost_scales(kv.second);
    }
}

const OperatorCostScaleTable& HardwareTopology::cpu_operator_scales() const {
    return cpu_operator_scales_;
}

void HardwareTopology::set_gpu_operator_scales(OperatorCostScaleTable scales) {
    gpu_operator_scales_ = std::move(scales);
    for (auto& kv : devices_) {
        apply_operator_cost_scales(kv.second);
    }
}

const OperatorCostScaleTable& HardwareTopology::gpu_operator_scales() const {
    return gpu_operator_scales_;
}

void HardwareTopology::invalidate_caches() {
    devices_cache_valid_ = false;
    links_cache_valid_ = false;
    transfer_time_cache_.clear();
}

void HardwareTopology::rebuild_device_cache() const {
    devices_cache_.clear();
    compute_devices_cache_.clear();
    devices_cache_.reserve(devices_.size());
    compute_devices_cache_.reserve(devices_.size());
    for (const auto& kv : devices_) {
        devices_cache_.push_back(&kv.second);
        if (kv.second.compute_capable) {
            compute_devices_cache_.push_back(&kv.second);
        }
    }
    std::sort(devices_cache_.begin(), devices_cache_.end(), [](const Device* a, const Device* b) {
        return a->id < b->id;
    });
    std::sort(compute_devices_cache_.begin(), compute_devices_cache_.end(), [](const Device* a, const Device* b) {
        return a->id < b->id;
    });
    devices_cache_valid_ = true;
}

void HardwareTopology::rebuild_link_caches() const {
    links_cache_ = links_;
    std::sort(links_cache_.begin(), links_cache_.end(), [](const Link& a, const Link& b) { return a.id < b.id; });

    outgoing_cache_.clear();
    outgoing_cache_.reserve(devices_.size());
    direct_link_cache_.clear();
    direct_link_cache_.reserve(links_.size());
    link_by_id_cache_.clear();
    link_by_id_cache_.reserve(links_.size());

    for (const auto& link : links_) {
        direct_link_cache_.emplace(direct_link_key(link.src, link.dst), &link);
        if (link.bw_gbps <= 0.0) {
            continue;
        }
        outgoing_cache_[link.src].push_back(&link);
        link_by_id_cache_[link.id] = &link;
    }
    for (auto& kv : outgoing_cache_) {
        auto& outgoing = kv.second;
        std::sort(outgoing.begin(), outgoing.end(), [](const Link* a, const Link* b) {
            if (a->dst == b->dst) {
                return a->id < b->id;
            }
            return a->dst < b->dst;
        });
    }
    links_cache_valid_ = true;
}

void HardwareTopology::apply_operator_cost_scales(Device& device) const {
    if (is_cpu_endpoint(&device)) {
        device.operator_cost_scales = cpu_operator_scales_;
    } else if (is_gpu_endpoint(&device)) {
        device.operator_cost_scales = gpu_operator_scales_;
    } else {
        device.operator_cost_scales.clear();
    }
    if (!device.has_calibration_operator_cost_scales) {
        return;
    }
    for (const auto& kv : device.calibration_operator_cost_scales) {
        device.operator_cost_scales[kv.first] = kv.second;
    }
}

double HardwareTopology::shortest_route_cost_seconds(std::string_view src, std::string_view dst, std::size_t bytes) const {
    if (src == dst) {
        return 0.0;
    }
    const auto* src_dev = device(src);
    const auto* dst_dev = device(dst);
    if (src_dev == nullptr || dst_dev == nullptr) {
        return std::numeric_limits<double>::infinity();
    }
    if (!links_cache_valid_) {
        rebuild_link_caches();
    }

    const auto link_cost_seconds = [&](const Link& link) {
        const auto route_hops = link.hardware_route.empty() ? std::size_t{1} : link.hardware_route.size();
        return link_transfer_time_seconds(link, link.src, link.dst, bytes, "native", route_hops);
    };

    const double kInf = std::numeric_limits<double>::infinity();
    std::unordered_map<std::string, double> dist;
    dist.reserve(devices_.size());
    for (const auto* device_ptr : devices()) {
        dist.emplace(device_ptr->id, kInf);
    }
    dist[src_dev->id] = 0.0;

    using QueueItem = std::pair<double, std::string>;
    std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> pq;
    pq.push({0.0, src_dev->id});

    while (!pq.empty()) {
        const auto [d, current] = pq.top();
        pq.pop();
        if (d > dist[current]) {
            continue;
        }
        if (current == dst_dev->id) {
            return d;
        }
        const auto out_it = outgoing_cache_.find(current);
        if (out_it == outgoing_cache_.end()) {
            continue;
        }
        for (const auto* link : out_it->second) {
            const double candidate = d + link_cost_seconds(*link);
            auto next_it = dist.find(link->dst);
            if (next_it == dist.end() || candidate >= next_it->second) {
                continue;
            }
            next_it->second = candidate;
            pq.push({candidate, link->dst});
        }
    }

    return std::numeric_limits<double>::infinity();
}

}  // namespace hardware_topology
