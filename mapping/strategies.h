#pragma once

#include <stdexcept>
#include <vector>

#include "mapping/graph.h"

namespace mapping {

class PartitionStrategy {
public:
    virtual ~PartitionStrategy() = default;
    virtual std::vector<std::vector<Task>> partition(const TaskGraph& graph, int parts) const = 0;
};

class LayerPartition : public PartitionStrategy {
public:
    std::vector<std::vector<Task>> partition(const TaskGraph& graph, int parts) const override;
};

}  // namespace mapping
