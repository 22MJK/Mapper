#pragma once

#include <string>

#include "hardware_topology/topology.h"
#include "mapping/graph.h"
#include "mapping/mapper.h"

namespace taskflow {

class TaskflowWriter {
public:
    static void write(const std::string& path,
                      const std::string& time_unit,
                      const mapping::TaskGraph& graph,
                      const mapping::MappingPlan& mapping_plan,
                      const hardware_topology::HardwareTopology& topology);
    static void write_chakra_et(const std::string& et_prefix,
                                const std::string& time_unit,
                                const mapping::TaskGraph& graph,
                                const mapping::MappingPlan& mapping_plan,
                                const hardware_topology::HardwareTopology& topology);
    static void write_outputs(const std::string& json_path,
                              const std::string& et_prefix,
                              const std::string& time_unit,
                              const mapping::TaskGraph& graph,
                              const mapping::MappingPlan& mapping_plan,
                              const hardware_topology::HardwareTopology& topology,
                              bool emit_json,
                              bool emit_chakra_et);
};

}  // namespace taskflow
