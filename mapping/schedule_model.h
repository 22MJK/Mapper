#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "hardware_topology/topology.h"
#include "mapping/graph.h"

namespace mapping {

struct MappingPlan;

std::string canonical_comm_kind(std::string value);
bool is_collective_kind(const std::string& kind);

double estimate_collective_time_seconds(const std::string& comm_kind,
                                        double bytes,
                                        std::size_t participants,
                                        const hardware_topology::HardwareTopology& topology);

double estimate_communication_time_seconds(const TaskEdge& edge,
                                           const hardware_topology::HardwareTopology& topology,
                                           const std::string& src,
                                           const std::string& dst);

double estimate_average_communication_time_seconds(
    const TaskEdge& edge,
    const std::vector<const hardware_topology::Device*>& devices,
    const hardware_topology::HardwareTopology& topology);

double schedule_p2p_transfer_seconds(const hardware_topology::HardwareTopology& topology,
                                     const std::string& src,
                                     const std::string& dst,
                                     double bytes,
                                     double earliest_start,
                                     std::unordered_map<std::string, double>& comm_available,
                                     std::unordered_map<std::string, double>& link_available);

double estimate_makespan_seconds(const TaskGraph& graph,
                                 const MappingPlan& plan,
                                 const hardware_topology::HardwareTopology& topology);

}  // namespace mapping
