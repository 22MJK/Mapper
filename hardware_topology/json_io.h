#pragma once

#include <string>
#include <cmath>
#include "hardware_topology/topology.h"

namespace hardware_topology {

// Returns true on success; on failure fills error (if provided).
bool load_from_json(const std::string& path, HardwareTopology& out, std::string* error = nullptr);

}  // namespace hardware_topology
