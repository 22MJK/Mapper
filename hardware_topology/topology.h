#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hardware_topology {

struct OperatorCostScale {
    double bandwidth_scale{1.0};
    double flops_scale{1.0};
    double launch_overhead_us{0.0};
    struct Segment {
        std::optional<double> max_num_ops;
        double bandwidth_scale{1.0};
        double flops_scale{1.0};
        double launch_overhead_us{0.0};
    };
    std::vector<Segment> segments;
};

using OperatorCostScaleTable = std::unordered_map<std::string, OperatorCostScale>;

struct CommunicationCostScale {
    std::optional<double> bandwidth_scale;
    double extra_latency_us{0.0};
};

using CommunicationRouteHopScaleTable = std::unordered_map<std::size_t, CommunicationCostScale>;
using CommunicationEndpointPairScaleTable = std::unordered_map<std::string, CommunicationRouteHopScaleTable>;
using CommunicationGroupScaleTable = std::unordered_map<std::string, CommunicationEndpointPairScaleTable>;
using CommunicationHardwareLinkScaleTable = std::unordered_map<std::string, CommunicationGroupScaleTable>;

struct Device {
    std::string id;
    std::string name;
    std::string type;
    std::string parent;
    double peak_gflops{0.0};
    double mem_bw_gbps{0.0};
    double mem_latency_ms{0.0};
    int max_concurrent{1};
    bool compute_capable{true};
    std::unordered_set<std::string> features;
    std::unordered_set<std::string> supported_precisions;
    OperatorCostScaleTable operator_cost_scales;
    OperatorCostScaleTable calibration_operator_cost_scales;
    bool has_calibration_operator_cost_scales{false};
};

// Directed link (src -> dst).
struct Link {
    std::string id;
    std::string src;
    std::string dst;
    double bw_gbps{0.0};
    double latency_ms{0.0};
    std::vector<std::string> hardware_route;
    CommunicationHardwareLinkScaleTable hardware_route_communication_scales;
};

class HardwareTopology {
public:
    void set_time_unit(std::string time_unit);
    const std::string& time_unit() const;

    void add_device(Device device);
    void add_link(Link link);

    const Device* device(std::string_view id) const;
    const std::vector<const Device*>& devices() const;
    const std::vector<const Device*>& compute_devices() const;
    const std::vector<Link>& links() const;

    // Direct-link queries (directed).
    std::optional<double> bw_gbps(std::string_view src, std::string_view dst) const;
    std::optional<double> latency_ms(std::string_view src, std::string_view dst) const;
    std::optional<std::string> link_id(std::string_view src, std::string_view dst) const;
    const Link* link_by_id(std::string_view id) const;

    // Returns minimum-time route as link-id list via Dijkstra.
    // Cost model per link: latency + serialization(bytes / bw).
    // Empty means "no route found" (or src==dst).
    std::vector<std::string> shortest_route_link_ids(std::string_view src,
                                                     std::string_view dst,
                                                     size_t bytes = 0) const;

    // Transfer time in seconds. If no route exists, returns +inf.
    double get_transfer_time(std::string_view src, std::string_view dst, size_t bytes) const;
    double get_transfer_time(std::string_view src,
                             std::string_view dst,
                             size_t bytes,
                             std::string_view communication_group) const;
    struct LinkTransferTiming {
        double elapsed_s{0.0};
        double link_busy_s{0.0};
    };
    // elapsed_s includes calibrated per-hop extra latency; link_busy_s does not.
    LinkTransferTiming link_transfer_timing_seconds(const Link& link,
                                                    std::string_view src,
                                                    std::string_view dst,
                                                    size_t bytes,
                                                    std::string_view communication_group = "native",
                                                    std::size_t route_hops = 0) const;
    double link_transfer_time_seconds(const Link& link,
                                      std::string_view src,
                                      std::string_view dst,
                                      size_t bytes,
                                      std::string_view communication_group = "native",
                                      std::size_t route_hops = 0) const;
    double endpoint_extra_latency_seconds(std::string_view src, std::string_view dst) const;
    double average_endpoint_extra_latency_seconds() const;

    struct EndpointExtraLatencyConfig {
        double gpu_gpu_intra_machine_us{0.0};
        double gpu_gpu_inter_machine_us{0.0};
        double gpu_cpu_intra_machine_us{0.0};
        double gpu_cpu_inter_machine_us{0.0};
        double cpu_cpu_intra_machine_us{0.0};
        double cpu_cpu_inter_machine_us{0.0};
    };
    void set_endpoint_extra_latency_config(EndpointExtraLatencyConfig config);
    const EndpointExtraLatencyConfig& endpoint_extra_latency_config() const;
    void set_cpu_operator_scales(OperatorCostScaleTable scales);
    const OperatorCostScaleTable& cpu_operator_scales() const;
    void set_gpu_operator_scales(OperatorCostScaleTable scales);
    const OperatorCostScaleTable& gpu_operator_scales() const;

private:
    struct TransferCacheKey {
        std::string src;
        std::string dst;
        std::uint64_t bytes{0};
        std::string communication_group;

        bool operator==(const TransferCacheKey& other) const {
            return src == other.src && dst == other.dst && bytes == other.bytes &&
                   communication_group == other.communication_group;
        }
    };

    struct TransferCacheKeyHash {
        std::size_t operator()(const TransferCacheKey& key) const;
    };

    void invalidate_caches();
    void rebuild_device_cache() const;
    void rebuild_link_caches() const;
    void apply_operator_cost_scales(Device& device) const;
    double shortest_route_cost_seconds(std::string_view src, std::string_view dst, std::size_t bytes) const;

    std::string time_unit_{"s"};
    EndpointExtraLatencyConfig endpoint_extra_latency_config_;
    OperatorCostScaleTable cpu_operator_scales_;
    OperatorCostScaleTable gpu_operator_scales_;
    std::unordered_map<std::string, Device> devices_;
    std::vector<Link> links_;
    mutable bool devices_cache_valid_{false};
    mutable bool links_cache_valid_{false};
    mutable std::vector<const Device*> devices_cache_;
    mutable std::vector<const Device*> compute_devices_cache_;
    mutable std::vector<Link> links_cache_;
    mutable std::unordered_map<std::string, std::vector<const Link*>> outgoing_cache_;
    mutable std::unordered_map<std::string, const Link*> direct_link_cache_;
    mutable std::unordered_map<std::string, const Link*> link_by_id_cache_;
    mutable std::unordered_map<TransferCacheKey, double, TransferCacheKeyHash> transfer_time_cache_;
};

}  // namespace hardware_topology
