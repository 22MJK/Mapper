#pragma once

#include <string>

#include "workload/workload.h"

namespace workload {

// Returns true on success; on failure fills error (if provided).
bool load_from_json(const std::string& path, Workload& out, std::string* error = nullptr);

}  // namespace workload
