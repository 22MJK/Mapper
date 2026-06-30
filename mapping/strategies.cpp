#include "mapping/strategies.h"

namespace mapping {

std::vector<std::vector<Task>> LayerPartition::partition(const TaskGraph& graph, int parts) const {
    if (parts <= 0) {
        throw std::runtime_error("Parts must be >= 1");
    }
    const auto& ordered = graph.topological_order();
    std::vector<std::vector<Task>> result(parts);
    const int base = static_cast<int>(ordered.size()) / parts;
    int remainder = static_cast<int>(ordered.size()) % parts;
    int cursor = 0;
    for (int part = 0; part < parts; ++part) {
        int size = base + (remainder > 0 ? 1 : 0);
        if (size == 0 || cursor >= static_cast<int>(ordered.size())) {
            break;  // no more tasks to distribute
        }
        result[part].insert(result[part].end(), ordered.begin() + cursor, ordered.begin() + cursor + size);
        cursor += size;
        if (remainder > 0) {
            --remainder;
        }
    }
    return result;
}

}  // namespace mapping
