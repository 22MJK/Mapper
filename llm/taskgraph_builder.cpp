#include "llm/taskgraph_builder.h"

#include "taskflow/json.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace llm {
namespace {

constexpr long double kMaxDouble = static_cast<long double>(std::numeric_limits<double>::max());

std::string lower(std::string value) {
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

bool is_gpu_device(const hardware_topology::Device* device) {
    return device != nullptr && lower(device->type) == "gpu";
}

std::vector<const hardware_topology::Device*> select_llm_devices(
    const hardware_topology::HardwareTopology& topology) {
    std::vector<const hardware_topology::Device*> devices;
    for (const auto* device : topology.compute_devices()) {
        if (is_gpu_device(device)) {
            devices.push_back(device);
        }
    }
    if (devices.empty()) {
        devices = topology.compute_devices();
    }
    std::sort(devices.begin(), devices.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->id < rhs->id;
    });
    return devices;
}

std::string device_group_label(const std::vector<const hardware_topology::Device*>& group) {
    std::vector<std::string> ids;
    ids.reserve(group.size());
    for (const auto* device : group) {
        ids.push_back(device == nullptr ? "" : device->id);
    }
    std::sort(ids.begin(), ids.end());
    std::ostringstream out;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << ids[i];
    }
    return out.str();
}

struct DeviceLocality {
    std::string parent;
    std::string nearest_neighbor;
    double nearest_bw_gbps{0.0};
    double nearest_latency_ms{std::numeric_limits<double>::infinity()};
};

std::vector<DeviceLocality> build_device_localities(
    const hardware_topology::HardwareTopology& topology,
    const std::vector<const hardware_topology::Device*>& devices) {
    std::unordered_map<std::string, std::size_t> index_by_id;
    index_by_id.reserve(devices.size());
    std::vector<DeviceLocality> localities(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        index_by_id.emplace(devices[i]->id, i);
        localities[i].parent = devices[i]->parent;
    }

    auto consider_link = [&](const std::string& endpoint, const std::string& neighbor, double bw, double latency_ms) {
        const auto it = index_by_id.find(endpoint);
        if (it == index_by_id.end()) {
            return;
        }
        auto& locality = localities[it->second];
        if (bw > locality.nearest_bw_gbps ||
            (bw == locality.nearest_bw_gbps &&
             (latency_ms < locality.nearest_latency_ms ||
              (latency_ms == locality.nearest_latency_ms && neighbor < locality.nearest_neighbor)))) {
            locality.nearest_neighbor = neighbor;
            locality.nearest_bw_gbps = bw;
            locality.nearest_latency_ms = latency_ms;
        }
    };

    for (const auto& link : topology.links()) {
        consider_link(link.src, link.dst, link.bw_gbps, link.latency_ms);
        consider_link(link.dst, link.src, link.bw_gbps, link.latency_ms);
    }
    return localities;
}

double direct_link_distance(const hardware_topology::HardwareTopology& topology,
                            const hardware_topology::Device* lhs,
                            const hardware_topology::Device* rhs) {
    const auto forward_bw = topology.bw_gbps(lhs->id, rhs->id);
    const auto backward_bw = topology.bw_gbps(rhs->id, lhs->id);
    const auto forward_latency = topology.latency_ms(lhs->id, rhs->id);
    const auto backward_latency = topology.latency_ms(rhs->id, lhs->id);
    const double bw = std::max(forward_bw.value_or(0.0), backward_bw.value_or(0.0));
    if (bw <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    const double latency = std::min(forward_latency.value_or(std::numeric_limits<double>::infinity()),
                                    backward_latency.value_or(std::numeric_limits<double>::infinity()));
    return 0.25 + latency / 1000.0 + 1.0 / bw;
}

double device_pair_distance(const hardware_topology::HardwareTopology& topology,
                            const hardware_topology::Device* lhs,
                            const hardware_topology::Device* rhs,
                            const DeviceLocality& lhs_locality,
                            const DeviceLocality& rhs_locality) {
    if (lhs == nullptr || rhs == nullptr) {
        return std::numeric_limits<double>::infinity();
    }
    if (lhs->id == rhs->id) {
        return 0.0;
    }
    const double direct = direct_link_distance(topology, lhs, rhs);
    if (std::isfinite(direct)) {
        return direct;
    }
    if (!lhs_locality.nearest_neighbor.empty() &&
        lhs_locality.nearest_neighbor == rhs_locality.nearest_neighbor) {
        const double bw = std::min(lhs_locality.nearest_bw_gbps, rhs_locality.nearest_bw_gbps);
        const double latency = std::max(lhs_locality.nearest_latency_ms, rhs_locality.nearest_latency_ms);
        return 1.0 + latency / 1000.0 + (bw > 0.0 ? 1.0 / bw : 1.0);
    }
    if (!lhs_locality.parent.empty() && lhs_locality.parent == rhs_locality.parent) {
        return 10.0;
    }
    return 100.0;
}

std::string device_locality_sort_key(const hardware_topology::Device* device,
                                     const DeviceLocality& locality) {
    std::ostringstream out;
    out << (locality.parent.empty() ? "~" : locality.parent) << "|"
        << (locality.nearest_neighbor.empty() ? "~" : locality.nearest_neighbor) << "|"
        << (device == nullptr ? "" : device->id);
    return out.str();
}

std::vector<std::vector<double>> build_device_distance_matrix(
    const hardware_topology::HardwareTopology& topology,
    const std::vector<const hardware_topology::Device*>& devices,
    const std::vector<DeviceLocality>& localities) {
    std::vector<std::vector<double>> distances(devices.size(), std::vector<double>(devices.size(), 0.0));
    for (std::size_t i = 0; i < devices.size(); ++i) {
        for (std::size_t j = i + 1; j < devices.size(); ++j) {
            const double distance = device_pair_distance(topology, devices[i], devices[j], localities[i], localities[j]);
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }
    return distances;
}

double group_pair_distance(const std::vector<int>& lhs,
                           const std::vector<int>& rhs,
                           const std::vector<std::vector<double>>& distances) {
    if (lhs.empty() || rhs.empty()) {
        return std::numeric_limits<double>::infinity();
    }
    double total = 0.0;
    std::size_t count = 0;
    for (const int a : lhs) {
        for (const int b : rhs) {
            total += distances[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)];
            ++count;
        }
    }
    return count == 0 ? std::numeric_limits<double>::infinity() : total / static_cast<double>(count);
}

double group_internal_distance(const std::vector<int>& group,
                               const std::vector<std::vector<double>>& distances) {
    if (group.size() <= 1) {
        return 0.0;
    }
    double total = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < group.size(); ++i) {
        for (std::size_t j = i + 1; j < group.size(); ++j) {
            total += distances[static_cast<std::size_t>(group[i])][static_cast<std::size_t>(group[j])];
            ++count;
        }
    }
    return total / static_cast<double>(count);
}

std::vector<std::vector<int>> build_topology_groups(
    const std::vector<const hardware_topology::Device*>& devices,
    const std::vector<DeviceLocality>& localities,
    int tp) {
    const std::size_t group_size = static_cast<std::size_t>(std::max(1, tp));
    std::vector<int> ordered_indices;
    ordered_indices.reserve(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        ordered_indices.push_back(static_cast<int>(i));
    }
    std::sort(ordered_indices.begin(), ordered_indices.end(), [&](int lhs, int rhs) {
        const auto lhs_key =
            device_locality_sort_key(devices[static_cast<std::size_t>(lhs)], localities[static_cast<std::size_t>(lhs)]);
        const auto rhs_key =
            device_locality_sort_key(devices[static_cast<std::size_t>(rhs)], localities[static_cast<std::size_t>(rhs)]);
        return lhs_key < rhs_key;
    });

    std::vector<std::vector<int>> groups;
    for (std::size_t offset = 0; offset < ordered_indices.size(); offset += group_size) {
        std::vector<int> group;
        const std::size_t end = std::min(ordered_indices.size(), offset + group_size);
        group.reserve(end - offset);
        for (std::size_t i = offset; i < end; ++i) {
            group.push_back(ordered_indices[i]);
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

std::vector<int> order_topology_groups(const std::vector<std::vector<int>>& groups,
                                       const std::vector<const hardware_topology::Device*>& devices,
                                       const std::vector<std::vector<double>>& distances) {
    std::vector<int> remaining;
    remaining.reserve(groups.size());
    for (std::size_t i = 0; i < groups.size(); ++i) {
        remaining.push_back(static_cast<int>(i));
    }

    std::vector<int> order;
    order.reserve(groups.size());
    while (!remaining.empty()) {
        int best = -1;
        double best_score = std::numeric_limits<double>::infinity();
        std::string best_label;
        for (const int candidate : remaining) {
            const double score = order.empty()
                                     ? group_internal_distance(groups[static_cast<std::size_t>(candidate)], distances)
                                     : group_pair_distance(groups[static_cast<std::size_t>(order.back())],
                                                           groups[static_cast<std::size_t>(candidate)],
                                                           distances);
            std::vector<const hardware_topology::Device*> candidate_devices;
            candidate_devices.reserve(groups[static_cast<std::size_t>(candidate)].size());
            for (const int index : groups[static_cast<std::size_t>(candidate)]) {
                candidate_devices.push_back(devices[static_cast<std::size_t>(index)]);
            }
            const auto label = device_group_label(candidate_devices);
            if (best < 0 || score < best_score || (score == best_score && label < best_label)) {
                best = candidate;
                best_score = score;
                best_label = label;
            }
        }
        order.push_back(best);
        remaining.erase(std::remove(remaining.begin(), remaining.end(), best), remaining.end());
    }
    return order;
}

std::vector<int> align_group_to_previous_stage(const std::vector<int>& group,
                                               const std::vector<int>& previous,
                                               const std::vector<const hardware_topology::Device*>& devices,
                                               const std::vector<std::vector<double>>& distances) {
    if (previous.empty() || previous.size() != group.size()) {
        auto sorted = group;
        std::sort(sorted.begin(), sorted.end(), [&](int lhs, int rhs) {
            return devices[static_cast<std::size_t>(lhs)]->id < devices[static_cast<std::size_t>(rhs)]->id;
        });
        return sorted;
    }

    std::vector<int> remaining = group;
    std::vector<int> aligned;
    aligned.reserve(group.size());
    for (const int reference : previous) {
        auto best_it = remaining.end();
        double best_distance = std::numeric_limits<double>::infinity();
        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
            const double distance = distances[static_cast<std::size_t>(reference)][static_cast<std::size_t>(*it)];
            if (distance < best_distance ||
                (distance == best_distance &&
                 (best_it == remaining.end() ||
                  devices[static_cast<std::size_t>(*it)]->id <
                      devices[static_cast<std::size_t>(*best_it)]->id))) {
                best_distance = distance;
                best_it = it;
            }
        }
        if (best_it == remaining.end()) {
            break;
        }
        aligned.push_back(*best_it);
        remaining.erase(best_it);
    }
    std::sort(remaining.begin(), remaining.end(), [&](int lhs, int rhs) {
        return devices[static_cast<std::size_t>(lhs)]->id < devices[static_cast<std::size_t>(rhs)]->id;
    });
    aligned.insert(aligned.end(), remaining.begin(), remaining.end());
    return aligned;
}

std::vector<const hardware_topology::Device*> select_topology_aware_llm_devices(
    const hardware_topology::HardwareTopology& topology,
    const LlmParallelConfig& parallel) {
    auto devices = select_llm_devices(topology);
    if (devices.size() <= 1 || parallel.tp <= 0 || parallel.pp <= 0 || parallel.dp <= 0) {
        return devices;
    }

    const auto localities = build_device_localities(topology, devices);
    const auto distances = build_device_distance_matrix(topology, devices, localities);
    const auto groups = build_topology_groups(devices, localities, parallel.tp);
    const auto group_order = order_topology_groups(groups, devices, distances);

    std::vector<const hardware_topology::Device*> ordered;
    ordered.reserve(devices.size());
    std::unordered_set<int> emitted;
    emitted.reserve(devices.size());
    std::vector<int> previous_stage_group;
    const int pp = std::max(1, parallel.pp);
    for (std::size_t pos = 0; pos < group_order.size(); ++pos) {
        const auto& group = groups[static_cast<std::size_t>(group_order[pos])];
        const bool first_stage_in_replica = (static_cast<int>(pos % static_cast<std::size_t>(pp)) == 0);
        const auto aligned = first_stage_in_replica
                                 ? align_group_to_previous_stage(group, {}, devices, distances)
                                 : align_group_to_previous_stage(group, previous_stage_group, devices, distances);
        for (const int index : aligned) {
            ordered.push_back(devices[static_cast<std::size_t>(index)]);
            emitted.insert(index);
        }
        previous_stage_group = aligned;
        if (static_cast<int>((pos + 1) % static_cast<std::size_t>(pp)) == 0) {
            previous_stage_group.clear();
        }
    }

    for (std::size_t i = 0; i < devices.size(); ++i) {
        if (emitted.count(static_cast<int>(i)) == 0) {
            ordered.push_back(devices[i]);
        }
    }
    return ordered;
}

double clamp_double(long double value) {
    if (!(value > 0.0L)) {
        return 0.0;
    }
    if (value > kMaxDouble) {
        return std::numeric_limits<double>::max();
    }
    return static_cast<double>(value);
}

std::uint64_t bytes_for_elements(long double elements, std::uint64_t dtype_bytes) {
    if (!(elements > 0.0L)) {
        return 0;
    }
    const long double bytes = elements * static_cast<long double>(dtype_bytes);
    if (bytes >= static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    return static_cast<std::uint64_t>(std::llround(bytes));
}

double bytes_double(long double elements, std::uint64_t dtype_bytes) {
    return clamp_double(elements * static_cast<long double>(dtype_bytes));
}

std::vector<std::vector<int>> build_stage_layers(int layers, int pp) {
    std::vector<std::vector<int>> stages(static_cast<std::size_t>(pp));
    for (int layer = 0; layer < layers; ++layer) {
        const int stage = std::min(pp - 1, (layer * pp) / std::max(1, layers));
        stages[static_cast<std::size_t>(stage)].push_back(layer);
    }
    return stages;
}

std::vector<int> layer_to_stage(const std::vector<std::vector<int>>& stages, int layers) {
    std::vector<int> out(static_cast<std::size_t>(layers), 0);
    for (std::size_t stage = 0; stage < stages.size(); ++stage) {
        for (const int layer : stages[stage]) {
            if (layer >= 0 && layer < layers) {
                out[static_cast<std::size_t>(layer)] = static_cast<int>(stage);
            }
        }
    }
    return out;
}

std::vector<int> default_stage_for_layer(int layers, int pp) {
    return layer_to_stage(build_stage_layers(layers, pp), layers);
}

std::vector<int> resolve_stage_for_layer(const LlmParallelConfig& parallel, int layers) {
    if (parallel.stage_for_layer.empty()) {
        return default_stage_for_layer(layers, parallel.pp);
    }
    return parallel.stage_for_layer;
}

std::string stage_partition_summary(const std::vector<int>& stage_for_layer, int pp) {
    if (stage_for_layer.empty()) {
        return "[]";
    }
    std::ostringstream out;
    out << "[";
    for (int stage = 0; stage < pp; ++stage) {
        if (stage != 0) {
            out << "|";
        }
        int first = -1;
        int last = -1;
        for (std::size_t layer = 0; layer < stage_for_layer.size(); ++layer) {
            if (stage_for_layer[layer] == stage) {
                if (first < 0) {
                    first = static_cast<int>(layer);
                }
                last = static_cast<int>(layer);
            }
        }
        if (first < 0) {
            out << "-";
        } else if (first == last) {
            out << first;
        } else {
            out << first << "-" << last;
        }
    }
    out << "]";
    return out.str();
}

bool is_linear_attention_type(const std::string& type) {
    return type == "linear_attention" || type == "linear";
}

bool is_sliding_attention_type(const std::string& type) {
    return type == "sliding_attention" || type == "sliding_window_attention" || type == "local_attention";
}

std::string layer_attention_type(const LlmModelConfig& model, int layer) {
    if (layer >= 0 && static_cast<std::size_t>(layer) < model.layer_types.size()) {
        return model.layer_types[static_cast<std::size_t>(layer)];
    }
    return "full_attention";
}

bool has_linear_attention_layers(const LlmModelConfig& model) {
    return std::any_of(model.layer_types.begin(), model.layer_types.end(), is_linear_attention_type);
}

bool is_strict_collective_kind(const std::string& kind) {
    return kind == "all_reduce" || kind == "all_to_all" || kind == "all_gather" || kind == "reduce_scatter";
}

bool is_llm_comm_kind_allowed(const std::string& kind) {
    return kind.empty() || kind == "p2p" || is_strict_collective_kind(kind);
}

std::string strict_collective_list() {
    return "all_reduce, all_to_all, all_gather, reduce_scatter";
}

std::string layer_type_summary(const LlmModelConfig& model) {
    if (model.layer_types.empty()) {
        return "full_attention=" + std::to_string(model.num_layers);
    }
    int full = 0;
    int sliding = 0;
    int linear = 0;
    int other = 0;
    for (const auto& type : model.layer_types) {
        if (is_linear_attention_type(type)) {
            linear += 1;
        } else if (is_sliding_attention_type(type)) {
            sliding += 1;
        } else if (type == "full_attention" || type == "attention") {
            full += 1;
        } else {
            other += 1;
        }
    }
    std::ostringstream out;
    out << "full_attention=" << full << ", sliding_attention=" << sliding
        << ", linear_attention=" << linear << ", other=" << other;
    return out.str();
}

long double shard_heads(int heads, int tp) {
    if (heads <= 0) {
        return 0.0L;
    }
    if (tp <= 1) {
        return static_cast<long double>(heads);
    }
    if (heads % tp == 0) {
        return static_cast<long double>(heads / tp);
    }
    return heads < tp ? static_cast<long double>(heads) : static_cast<long double>(heads) / static_cast<long double>(tp);
}

long double shard_dim(long double dim, int tp) {
    return dim / static_cast<long double>(std::max(1, tp));
}

void validate_model_and_plan(const LlmModelConfig& model,
                             const LlmRequestConfig& request,
                             const LlmParallelConfig& parallel,
                             const hardware_topology::HardwareTopology& topology,
                             std::vector<std::string>& diagnostics) {
    if (parallel.tp <= 0 || parallel.pp <= 0 || parallel.cp <= 0 || parallel.dp <= 0) {
        throw std::runtime_error("LLM parallel degrees must be positive");
    }
    if (parallel.cp != 1) {
        throw std::runtime_error("LLM CP is reserved but not implemented in the first version; use --llm-cp 1");
    }
    if (model.hidden_size % parallel.tp != 0) {
        throw std::runtime_error("hidden_size must be divisible by llm TP degree");
    }
    if (model.num_attention_heads % parallel.tp != 0) {
        throw std::runtime_error("num_attention_heads must be divisible by llm TP degree");
    }
    if (model.intermediate_size % parallel.tp != 0) {
        throw std::runtime_error("intermediate_size must be divisible by llm TP degree");
    }
    if (model.num_kv_heads % parallel.tp != 0) {
        if (model.num_kv_heads < parallel.tp) {
            diagnostics.push_back("KV heads are fewer than TP shards; first version models KV heads as replicated.");
        } else {
            throw std::runtime_error("num_kv_heads must be divisible by TP, unless num_kv_heads < TP for replication");
        }
    }
    if (parallel.pp > model.num_layers) {
        throw std::runtime_error("llm PP degree cannot exceed num_layers");
    }
    if (!parallel.stage_for_layer.empty()) {
        if (parallel.stage_for_layer.size() != static_cast<std::size_t>(model.num_layers)) {
            throw std::runtime_error("LLM stage_for_layer length must match num_hidden_layers");
        }
        std::vector<int> stage_counts(static_cast<std::size_t>(parallel.pp), 0);
        int previous_stage = 0;
        for (std::size_t layer = 0; layer < parallel.stage_for_layer.size(); ++layer) {
            const int stage = parallel.stage_for_layer[layer];
            if (stage < 0 || stage >= parallel.pp) {
                throw std::runtime_error("LLM stage_for_layer contains a stage outside [0, pp)");
            }
            if (layer != 0 && stage < previous_stage) {
                throw std::runtime_error("LLM stage_for_layer must be non-decreasing for contiguous PP stages");
            }
            previous_stage = stage;
            ++stage_counts[static_cast<std::size_t>(stage)];
        }
        for (int stage = 0; stage < parallel.pp; ++stage) {
            if (stage_counts[static_cast<std::size_t>(stage)] == 0) {
                throw std::runtime_error("LLM stage_for_layer must assign at least one layer to every PP stage");
            }
        }
    }
    if (!model.layer_types.empty() && model.layer_types.size() != static_cast<std::size_t>(model.num_layers)) {
        throw std::runtime_error("LLM layer_types length must match num_hidden_layers");
    }
    if (has_linear_attention_layers(model)) {
        if (model.linear_num_key_heads <= 0 || model.linear_num_value_heads <= 0 ||
            model.linear_key_head_dim <= 0 || model.linear_value_head_dim <= 0) {
            throw std::runtime_error("linear_attention layers require linear key/value head counts and dimensions");
        }
        if (model.linear_num_key_heads >= parallel.tp && model.linear_num_key_heads % parallel.tp != 0) {
            throw std::runtime_error("linear_num_key_heads must be divisible by llm TP degree when not replicated");
        }
        if (model.linear_num_value_heads >= parallel.tp && model.linear_num_value_heads % parallel.tp != 0) {
            throw std::runtime_error("linear_num_value_heads must be divisible by llm TP degree when not replicated");
        }
    }
    if (model.is_moe) {
        if (model.num_experts <= 0 || model.experts_per_token <= 0 ||
            model.experts_per_token > model.num_experts || model.moe_intermediate_size <= 0) {
            throw std::runtime_error("MoE models require num_experts, experts_per_token, and moe_intermediate_size");
        }
    }
    if (request.prefill_batch_size <= 0 || request.prompt_len < 0 || request.decode_batch_size <= 0 ||
        request.decode_steps < 0 || request.avg_context_len < 0) {
        throw std::runtime_error("Invalid LLM request dimensions");
    }
    if (request.prompt_len == 0 && request.decode_steps == 0) {
        throw std::runtime_error("LLM request must include prefill tokens or decode steps");
    }

    const auto devices = select_llm_devices(topology);
    const std::size_t devices_per_replica = static_cast<std::size_t>(parallel.tp * parallel.pp);
    const std::size_t required = devices_per_replica * static_cast<std::size_t>(parallel.dp);
    if (devices.size() < required) {
        throw std::runtime_error("Not enough compute devices for LLM DP replicas: need " +
                                 std::to_string(required) + ", got " + std::to_string(devices.size()));
    }
    if (has_linear_attention_layers(model)) {
        diagnostics.push_back("Detected HF hybrid layer_types; linear_attention layers use linear-time attention cost.");
    }
    if (model.is_moe) {
        diagnostics.push_back("Detected MoE decoder; expert routing is modeled with all_to_all when TP > 1.");
    }
}

struct PhaseShape {
    std::string name;
    bool decode{false};
    int batch{1};
    int seq{1};
    int steps{1};
    int context{1};
};

struct OpCost {
    double flops{0.0};
    double bytes{0.0};
};

OpCost linear_cost(long double tokens,
                   long double in_dim,
                   long double out_dim,
                   int shards,
                   std::uint64_t dtype_bytes) {
    const long double shard = static_cast<long double>(std::max(1, shards));
    const long double flops = 2.0L * tokens * in_dim * out_dim / shard;
    const long double bytes = (tokens * in_dim + in_dim * out_dim / shard + tokens * out_dim / shard) *
                              static_cast<long double>(dtype_bytes);
    return {clamp_double(flops), clamp_double(bytes)};
}

OpCost linear_cost_for_output_shard(long double tokens,
                                    long double in_dim,
                                    long double out_dim_shard,
                                    std::uint64_t dtype_bytes) {
    const long double flops = 2.0L * tokens * in_dim * out_dim_shard;
    const long double bytes =
        (tokens * in_dim + in_dim * out_dim_shard + tokens * out_dim_shard) *
        static_cast<long double>(dtype_bytes);
    return {clamp_double(flops), clamp_double(bytes)};
}

OpCost moe_expert_cost(long double tokens,
                       long double hidden,
                       long double intermediate,
                       int experts_per_token,
                       int shards,
                       std::uint64_t dtype_bytes) {
    const long double routed_tokens =
        tokens * static_cast<long double>(std::max(1, experts_per_token)) /
        static_cast<long double>(std::max(1, shards));
    const long double gate_up_out = 2.0L * intermediate;
    const long double flops =
        2.0L * routed_tokens * hidden * gate_up_out +
        2.0L * routed_tokens * intermediate * hidden;
    const long double bytes =
        (routed_tokens * hidden + hidden * gate_up_out + routed_tokens * gate_up_out +
         intermediate * hidden + routed_tokens * hidden) *
        static_cast<long double>(dtype_bytes);
    return {clamp_double(flops), clamp_double(bytes)};
}

std::int64_t shape_dim(long double value) {
    if (!(value > 0.0L)) {
        return 0;
    }
    if (value >= static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        return std::numeric_limits<std::int64_t>::max();
    }
    return static_cast<std::int64_t>(std::llround(value));
}

mapping::TaskInput dense_input(std::string tensor_id,
                               double bytes,
                               std::initializer_list<long double> dims,
                               const std::string& dtype = "fp32") {
    mapping::TaskInput input;
    input.tensor_id = std::move(tensor_id);
    input.tensor_bytes = bytes;
    input.storage_format = "dense";
    input.dtype = dtype;
    input.shape.reserve(dims.size());
    for (const auto dim : dims) {
        input.shape.push_back(shape_dim(dim));
    }
    return input;
}

mapping::TaskInput dense_weight_input(std::string tensor_id,
                                      long double rows,
                                      long double cols,
                                      std::uint64_t dtype_bytes,
                                      const std::string& dtype = "fp32") {
    return dense_input(std::move(tensor_id), bytes_double(rows * cols, dtype_bytes), {rows, cols}, dtype);
}

mapping::Task make_task(std::string name,
                        std::string subtype,
                        const OpCost& cost,
                        const std::string& device_id,
                        std::vector<mapping::TaskInput> inputs = {}) {
    mapping::Task task;
    task.name = std::move(name);
    task.type = "compute";
    task.subtype = std::move(subtype);
    task.compute_flops = cost.flops;
    task.memory_bytes = cost.bytes;
    task.access_pattern = "dense";
    task.input_data = std::move(inputs);
    task.features.insert("dense_linear_algebra");
    task.features.insert("streaming_memory");
    task.features.insert("massive_parallelism");
    if (!device_id.empty()) {
        task.tags.insert("device:" + device_id);
    }
    return task;
}

std::string shard_name(const std::string& phase, int layer, const std::string& op, int tp) {
    std::ostringstream out;
    out << phase << ".layer_" << layer << "." << op << ".tp" << tp;
    return out.str();
}

std::string weight_name(int layer, const std::string& op, int tp) {
    std::ostringstream out;
    out << "layer_" << layer << "." << op << "_weight.tp" << tp;
    return out.str();
}

std::string root_name(const std::string& phase, const std::string& op, int tp) {
    std::ostringstream out;
    out << phase << "." << op << ".tp" << tp;
    return out.str();
}

void add_edges_from_all(mapping::TaskGraph& graph,
                        const std::vector<std::string>& srcs,
                        const std::string& dst,
                        double bytes,
                        const std::string& tensor_id,
                        const std::string& comm_kind,
                        std::size_t comm_participants = 0,
                        const std::string& comm_group = {},
                        const std::string& dtype = {}) {
    if (!is_llm_comm_kind_allowed(comm_kind)) {
        throw std::runtime_error("Unsupported LLM communication kind '" + comm_kind +
                                 "'; collective comm_kind must be one of: " + strict_collective_list());
    }
    for (const auto& src : srcs) {
        graph.add_edge(src, dst, bytes, tensor_id, comm_kind, "dense", comm_participants, comm_group, dtype);
    }
}

void add_rank_collective_edges(mapping::TaskGraph& graph,
                               const std::vector<std::string>& srcs,
                               const std::vector<std::string>& dsts,
                               double bytes,
                               const std::string& tensor_id,
                               const std::string& comm_kind,
                               std::size_t comm_participants,
                               const std::string& comm_group,
                               const std::string& dtype = {}) {
    if (srcs.size() != dsts.size()) {
        throw std::runtime_error("Invalid LLM collective group for " + tensor_id +
                                 ": source and destination shard counts differ");
    }
    if (!is_llm_comm_kind_allowed(comm_kind)) {
        throw std::runtime_error("Unsupported LLM communication kind '" + comm_kind +
                                 "'; collective comm_kind must be one of: " + strict_collective_list());
    }
    for (std::size_t rank = 0; rank < srcs.size(); ++rank) {
        graph.add_edge(srcs[rank],
                       dsts[rank],
                       bytes,
                       tensor_id,
                       comm_kind,
                       "dense",
                       comm_participants,
                       comm_group,
                       dtype);
    }
}

void require_tp_shards(const std::vector<std::string>& shards, int tp, const std::string& label) {
    const auto expected = static_cast<std::size_t>(tp);
    if (tp <= 0 || shards.size() != expected) {
        throw std::runtime_error("Invalid LLM TP semantic group for " + label + ": expected " +
                                 std::to_string(std::max(0, tp)) + " shards, got " +
                                 std::to_string(shards.size()));
    }
}

void require_collective_group(const std::string& comm_group, const std::string& label) {
    if (comm_group.empty()) {
        throw std::runtime_error("Missing LLM collective group for " + label);
    }
}

void add_tp_allreduce_edges(mapping::TaskGraph& graph,
                            const std::vector<std::string>& srcs,
                            const std::vector<std::string>& dsts,
                            double bytes,
                            const std::string& tensor_id,
                            int tp,
                            const std::string& comm_group,
                            const std::string& dtype = {}) {
    require_tp_shards(srcs, tp, tensor_id + " sources");
    require_tp_shards(dsts, tp, tensor_id + " destinations");
    if (tp == 1) {
        graph.add_edge(srcs.front(), dsts.front(), bytes, tensor_id, "p2p", "dense", 0, {}, dtype);
        return;
    }
    require_collective_group(comm_group, tensor_id);
    add_rank_collective_edges(graph,
                              srcs,
                              dsts,
                              bytes,
                              tensor_id,
                              "all_reduce",
                              static_cast<std::size_t>(tp),
                              comm_group,
                              dtype);
}

void add_tp_allgather_edges(mapping::TaskGraph& graph,
                            const std::vector<std::string>& srcs,
                            const std::string& dst,
                            double bytes,
                            const std::string& tensor_id,
                            int tp,
                            const std::string& comm_group,
                            const std::string& dtype = {}) {
    require_tp_shards(srcs, tp, tensor_id + " sources");
    if (tp == 1) {
        graph.add_edge(srcs.front(), dst, bytes, tensor_id, "p2p", "dense", 0, {}, dtype);
        return;
    }
    require_collective_group(comm_group, tensor_id);
    add_edges_from_all(
        graph, srcs, dst, bytes, tensor_id, "all_gather", static_cast<std::size_t>(tp), comm_group, dtype);
}

void add_tp_alltoall_edges(mapping::TaskGraph& graph,
                           const std::vector<std::string>& srcs,
                           const std::vector<std::string>& dsts,
                           double bytes,
                           const std::string& tensor_id,
                           int tp,
                           const std::string& comm_group,
                           const std::string& dtype = {}) {
    require_tp_shards(srcs, tp, tensor_id + " sources");
    require_tp_shards(dsts, tp, tensor_id + " destinations");
    if (tp == 1) {
        graph.add_edge(srcs.front(), dsts.front(), bytes, tensor_id, "p2p", "dense", 0, {}, dtype);
        return;
    }
    require_collective_group(comm_group, tensor_id);
    add_rank_collective_edges(graph,
                              srcs,
                              dsts,
                              bytes,
                              tensor_id,
                              "all_to_all",
                              static_cast<std::size_t>(tp),
                              comm_group,
                              dtype);
}

void add_tp_same_rank_edges(mapping::TaskGraph& graph,
                            const std::vector<std::string>& srcs,
                            const std::vector<std::string>& dsts,
                            double bytes,
                            const std::string& tensor_id,
                            int tp,
                            const std::string& dtype = {}) {
    require_tp_shards(srcs, tp, tensor_id + " sources");
    require_tp_shards(dsts, tp, tensor_id + " destinations");
    for (int rank = 0; rank < tp; ++rank) {
        graph.add_edge(srcs[static_cast<std::size_t>(rank)],
                       dsts[static_cast<std::size_t>(rank)],
                       bytes,
                       tensor_id,
                       "p2p",
                       "dense",
                       0,
                       {},
                       dtype);
    }
}

void add_tp_stage_boundary_edges(mapping::TaskGraph& graph,
                                 const std::vector<std::string>& srcs,
                                 const std::vector<std::string>& dsts,
                                 double bytes,
                                 const std::string& tensor_id,
                                 int tp,
                                 const std::vector<const hardware_topology::Device*>& devices,
                                 int src_stage,
                                 bool pin_shards_to_devices,
                                 const std::string& phase_name,
                                 int layer,
                                 const std::string& dtype = {}) {
    require_tp_shards(srcs, tp, tensor_id + " boundary sources");
    require_tp_shards(dsts, tp, tensor_id + " boundary destinations");
    if (tp == 1) {
        graph.add_edge(srcs.front(), dsts.front(), bytes, tensor_id, "p2p", "dense", 0, {}, dtype);
        return;
    }

    std::vector<std::string> transfer;
    transfer.reserve(static_cast<std::size_t>(tp));
    const int src_device_base = src_stage * tp;
    for (int rank = 0; rank < tp; ++rank) {
        const auto name = phase_name + ".layer_" + std::to_string(layer) +
                          ".pp_boundary_transfer.tp" + std::to_string(rank);
        const auto& source_device = devices[static_cast<std::size_t>(src_device_base + rank)]->id;
        graph.add_task(make_task(name,
                                 "copy",
                                 {0.0, bytes},
                                 pin_shards_to_devices ? source_device : "",
                                 {dense_input(tensor_id + ".source_stage_hidden", bytes, {}, dtype)}));
        transfer.push_back(name);
    }

    add_tp_allreduce_edges(graph,
                           srcs,
                           transfer,
                           bytes,
                           tensor_id + ".source_stage_hidden",
                           tp,
                           phase_name + ".layer_" + std::to_string(layer) + ".pp_boundary_all_reduce",
                           dtype);
    add_tp_same_rank_edges(graph, transfer, dsts, bytes, tensor_id, tp, dtype);
}

struct CollectiveSemanticGroup {
    std::string kind;
    std::string group;
    std::size_t participants{0};
    std::unordered_set<std::string> sources;
    std::unordered_set<std::string> destinations;
    std::unordered_set<std::string> byte_values;
    std::size_t edge_count{0};
};

std::string collective_semantic_key(const mapping::TaskEdge& edge) {
    return edge.comm_kind + "|" + edge.comm_group;
}

void validate_collective_group_semantics(const CollectiveSemanticGroup& group) {
    if (!is_strict_collective_kind(group.kind)) {
        throw std::runtime_error("Unsupported LLM collective kind '" + group.kind +
                                 "'; allowed collectives are: " + strict_collective_list());
    }
    if (group.group.empty()) {
        throw std::runtime_error("LLM collective edge is missing comm_group for " + group.kind);
    }
    if (group.participants <= 1) {
        throw std::runtime_error("LLM collective group '" + group.group + "' must have more than one participant");
    }
    if (group.byte_values.size() != 1) {
        throw std::runtime_error("LLM collective group '" + group.group + "' has inconsistent tensor_bytes");
    }

    const auto p = group.participants;
    if (group.sources.size() != p) {
        throw std::runtime_error("LLM collective group '" + group.group + "' expected " + std::to_string(p) +
                                 " source shards, got " + std::to_string(group.sources.size()));
    }

    if (group.kind == "all_reduce" || group.kind == "reduce_scatter" || group.kind == "all_to_all") {
        if (group.destinations.size() != p) {
            throw std::runtime_error("LLM collective group '" + group.group + "' expected " + std::to_string(p) +
                                     " destination shards, got " + std::to_string(group.destinations.size()));
        }
        if (group.edge_count < p) {
            throw std::runtime_error("LLM collective group '" + group.group +
                                     "' has too few dependency edges to cover all participant shards");
        }
        return;
    }

    if (group.kind == "all_gather") {
        if (group.destinations.empty() || group.destinations.size() > p) {
            throw std::runtime_error("LLM all_gather group '" + group.group +
                                     "' must have between 1 and participant-count destinations");
        }
        if (group.edge_count < p) {
            throw std::runtime_error("LLM all_gather group '" + group.group +
                                     "' has too few dependency edges to cover all source shards");
        }
    }
}

void validate_llm_collective_semantics(const mapping::TaskGraph& graph) {
    std::unordered_map<std::string, CollectiveSemanticGroup> groups;
    for (const auto& task : graph.topological_order()) {
        for (const auto& edge : graph.successors(task.name)) {
            const std::string kind = edge.comm_kind.empty() ? "p2p" : edge.comm_kind;
            if (kind == "p2p") {
                continue;
            }
            if (!is_strict_collective_kind(kind)) {
                throw std::runtime_error("Unsupported LLM communication kind '" + kind +
                                         "'; collective comm_kind must be one of: " + strict_collective_list());
            }
            auto& group = groups[collective_semantic_key(edge)];
            if (group.edge_count == 0) {
                group.kind = kind;
                group.group = edge.comm_group;
                group.participants = edge.comm_participants;
            } else if (group.participants != edge.comm_participants) {
                throw std::runtime_error("LLM collective group '" + edge.comm_group +
                                         "' has inconsistent comm_participants");
            }
            group.sources.insert(edge.src);
            group.destinations.insert(edge.dst);
            group.byte_values.insert(std::to_string(edge.tensor_bytes));
            group.edge_count += 1;
        }
    }
    for (const auto& entry : groups) {
        validate_collective_group_semantics(entry.second);
    }
}

std::vector<std::string> add_phase(mapping::TaskGraph& graph,
                                   const LlmModelConfig& model,
                                   const LlmRequestConfig& request,
                                   const LlmParallelConfig& parallel,
                                   const std::vector<const hardware_topology::Device*>& devices,
                                   const std::vector<int>& stage_for_layer,
                                   const PhaseShape& phase,
                                   const std::vector<std::string>& phase_deps) {
    (void)request;
    const auto dtype_bytes = dtype_size_bytes(model.param_dtype);
    const auto dtype = dtype_name(model.param_dtype);
    const long double tokens =
        phase.decode ? static_cast<long double>(phase.batch) * static_cast<long double>(phase.steps)
                     : static_cast<long double>(phase.batch) * static_cast<long double>(phase.seq);
    const long double hidden = static_cast<long double>(model.hidden_size);
    const long double inter = static_cast<long double>(model.intermediate_size);
    const long double heads = static_cast<long double>(model.num_attention_heads);
    const long double kv_heads = static_cast<long double>(model.num_kv_heads);
    const long double head_dim = static_cast<long double>(model.head_dim);
    const bool replicate_kv_heads = model.num_kv_heads % parallel.tp != 0;
    const long double q_heads_per_shard = heads / static_cast<long double>(parallel.tp);
    const long double kv_heads_per_shard =
        replicate_kv_heads ? kv_heads : kv_heads / static_cast<long double>(parallel.tp);
    const double hidden_full_bytes = bytes_double(tokens * hidden, dtype_bytes);
    const double hidden_shard_bytes = bytes_double(tokens * shard_dim(hidden, parallel.tp), dtype_bytes);
    const double mlp_intermediate_bytes = bytes_double(tokens * shard_dim(inter, parallel.tp), dtype_bytes);

    std::vector<std::string> prev;
    prev.reserve(static_cast<std::size_t>(parallel.tp));
    for (int tp = 0; tp < parallel.tp; ++tp) {
        const auto& device = devices[static_cast<std::size_t>(tp)]->id;
        const auto name = root_name(phase.name, "embedding", tp);
        const OpCost cost{0.0, hidden_full_bytes};
        std::vector<mapping::TaskInput> inputs{dense_input("tokens", hidden_full_bytes, {tokens, hidden}, dtype)};
        if (!phase_deps.empty()) {
            inputs.push_back(dense_input(
                phase.name + ".phase_dep", hidden_shard_bytes, {tokens, shard_dim(hidden, parallel.tp)}, dtype));
        }
        mapping::Task task = make_task(name,
                                       "copy",
                                       cost,
                                       parallel.pin_shards_to_devices ? device : "",
                                       std::move(inputs));
        graph.add_task(std::move(task));
        if (!phase_deps.empty()) {
            add_edges_from_all(
                graph, phase_deps, name, hidden_shard_bytes, phase.name + ".phase_dep", "p2p", 0, {}, dtype);
        }
        prev.push_back(name);
    }

    for (int layer = 0; layer < model.num_layers; ++layer) {
        const int stage = stage_for_layer[static_cast<std::size_t>(layer)];
        const int device_base = stage * parallel.tp;
        std::vector<std::string> qkv;
        std::vector<std::string> attn;
        std::vector<std::string> o_proj;
        std::vector<std::string> mlp_up;
        std::vector<std::string> mlp_down;
        std::vector<std::string> moe_router;
        std::vector<std::string> moe_dispatch;
        std::vector<std::string> moe_expert;
        std::vector<std::string> moe_combine;
        qkv.reserve(static_cast<std::size_t>(parallel.tp));
        attn.reserve(static_cast<std::size_t>(parallel.tp));
        o_proj.reserve(static_cast<std::size_t>(parallel.tp));
        mlp_up.reserve(static_cast<std::size_t>(parallel.tp));
        mlp_down.reserve(static_cast<std::size_t>(parallel.tp));
        moe_router.reserve(static_cast<std::size_t>(parallel.tp));
        moe_dispatch.reserve(static_cast<std::size_t>(parallel.tp));
        moe_expert.reserve(static_cast<std::size_t>(parallel.tp));
        moe_combine.reserve(static_cast<std::size_t>(parallel.tp));

        const auto attention_type = layer_attention_type(model, layer);
        const bool linear_attention = is_linear_attention_type(attention_type);
        const bool sliding_attention = is_sliding_attention_type(attention_type);
        const long double q_dim_per_shard = q_heads_per_shard * head_dim;
        const long double key_dim_per_shard =
            linear_attention
                ? shard_heads(model.linear_num_key_heads, parallel.tp) *
                      static_cast<long double>(model.linear_key_head_dim)
                : kv_heads_per_shard * head_dim;
        const long double value_dim_per_shard =
            linear_attention
                ? shard_heads(model.linear_num_value_heads, parallel.tp) *
                      static_cast<long double>(model.linear_value_head_dim)
                : kv_heads_per_shard * head_dim;
        const long double qkv_out_per_shard = q_dim_per_shard + key_dim_per_shard + value_dim_per_shard;
        const long double attn_out_dim_per_shard = linear_attention ? value_dim_per_shard : q_dim_per_shard;
        const long double o_proj_input_dim = linear_attention
                                                 ? static_cast<long double>(model.linear_num_value_heads) *
                                                       static_cast<long double>(model.linear_value_head_dim)
                                                 : heads * head_dim;
        const double qkv_bytes = bytes_double(tokens * qkv_out_per_shard, dtype_bytes);
        const double attn_out_bytes = bytes_double(tokens * attn_out_dim_per_shard, dtype_bytes);
        const bool moe_layer = model.is_moe;
        const long double moe_inter = static_cast<long double>(
            model.moe_intermediate_size > 0 ? model.moe_intermediate_size : model.intermediate_size);
        const int experts_per_token = std::max(1, model.experts_per_token);
        const double router_logits_bytes =
            bytes_double(tokens * static_cast<long double>(std::max(1, model.num_experts)), dtype_bytes);
        const double routed_hidden_bytes =
            bytes_double(tokens * hidden * static_cast<long double>(experts_per_token), dtype_bytes);
        const double local_routed_hidden_bytes =
            bytes_double(tokens * hidden * static_cast<long double>(experts_per_token) /
                         static_cast<long double>(std::max(1, parallel.tp)),
                         dtype_bytes);
        const long double local_routed_tokens =
            tokens * static_cast<long double>(experts_per_token) /
            static_cast<long double>(std::max(1, parallel.tp));

        for (int tp = 0; tp < parallel.tp; ++tp) {
            const auto& device = devices[static_cast<std::size_t>(device_base + tp)]->id;
            const std::string device_id = parallel.pin_shards_to_devices ? device : "";

            const auto qkv_name = shard_name(phase.name, layer, "qkv_proj", tp);
            graph.add_task(make_task(qkv_name,
                                     "gemm",
                                     linear_cost_for_output_shard(tokens, hidden, qkv_out_per_shard, dtype_bytes),
                                     device_id,
                                     {dense_input(phase.name + ".hidden", hidden_full_bytes, {tokens, hidden}, dtype),
                                      dense_weight_input(weight_name(layer, "qkv", tp),
                                                         hidden,
                                                         qkv_out_per_shard,
                                                         dtype_bytes,
                                                         dtype)}));
            qkv.push_back(qkv_name);

            long double attn_flops = 0.0L;
            long double attn_bytes = 0.0L;
            const long double effective_context =
                sliding_attention && model.sliding_window > 0
                    ? std::min(static_cast<long double>(std::max(1, phase.context)),
                               static_cast<long double>(model.sliding_window))
                    : static_cast<long double>(std::max(1, phase.context));
            const long double effective_seq_context =
                sliding_attention && model.sliding_window > 0
                    ? std::min(static_cast<long double>(std::max(1, phase.seq)),
                               static_cast<long double>(model.sliding_window))
                    : static_cast<long double>(std::max(1, phase.seq));
            if (linear_attention) {
                const long double kernel = static_cast<long double>(std::max(1, model.linear_conv_kernel_dim));
                attn_flops = 2.0L * tokens * (q_dim_per_shard + key_dim_per_shard + value_dim_per_shard) +
                             2.0L * tokens * key_dim_per_shard * value_dim_per_shard /
                                 std::max(1.0L, q_heads_per_shard) +
                             tokens * (key_dim_per_shard + value_dim_per_shard) * kernel;
                attn_bytes = tokens * (qkv_out_per_shard + attn_out_dim_per_shard) *
                             static_cast<long double>(dtype_bytes);
            } else if (phase.decode) {
                attn_flops = 4.0L * static_cast<long double>(phase.batch) *
                             static_cast<long double>(std::max(1, phase.steps)) * q_heads_per_shard *
                             effective_context * head_dim /
                             1.0L;
                attn_bytes =
                    static_cast<long double>(phase.batch) * static_cast<long double>(std::max(1, phase.steps)) *
                    effective_context * 2.0L * kv_heads_per_shard * head_dim *
                    static_cast<long double>(dtype_bytes);
            } else {
                attn_flops = 4.0L * static_cast<long double>(phase.batch) * q_heads_per_shard *
                             static_cast<long double>(phase.seq) * effective_seq_context * head_dim /
                             1.0L;
                attn_bytes = tokens * qkv_out_per_shard * static_cast<long double>(dtype_bytes);
            }
            const std::string attn_op = linear_attention
                                             ? (phase.decode ? "linear_attention_decode" : "linear_attention_prefill")
                                             : (sliding_attention
                                                    ? (phase.decode ? "sliding_attention_decode"
                                                                    : "sliding_attention_prefill")
                                                    : (phase.decode ? "attention_decode" : "attention_prefill"));
            const auto attn_name = shard_name(phase.name, layer, attn_op, tp);
            // Attention is modeled as a representative dense GEMM. A GEMM needs
            // two matrix operands, so provide a second (score/context) matrix in
            // addition to the qkv activation. Its inner dimension is sized so the
            // GEMM flops (2 * rows * cols * inner) match the analytical attention
            // flops, keeping replay self-consistent with the cost model.
            const long double attn_lhs_rows = tokens;
            const long double attn_lhs_cols = qkv_out_per_shard;
            const long double attn_rhs_cols =
                (attn_lhs_rows > 0.0L && attn_lhs_cols > 0.0L)
                    ? std::max(1.0L, attn_flops / (2.0L * attn_lhs_rows * attn_lhs_cols))
                    : std::max(1.0L, attn_out_dim_per_shard);
            const double attn_rhs_bytes = bytes_double(attn_lhs_cols * attn_rhs_cols, dtype_bytes);
            graph.add_task(make_task(
                attn_name,
                "gemm",
                {clamp_double(attn_flops), clamp_double(attn_bytes)},
                device_id,
                {dense_input(phase.name + ".qkv", qkv_bytes, {attn_lhs_rows, attn_lhs_cols}, dtype),
                 dense_input(attn_name + ".context", attn_rhs_bytes, {attn_lhs_cols, attn_rhs_cols}, dtype)}));
            graph.add_edge(qkv_name, attn_name, qkv_bytes, phase.name + ".qkv", "p2p", "dense", 0, {}, dtype);
            attn.push_back(attn_name);

            const auto o_name = shard_name(phase.name, layer, "o_proj", tp);
            graph.add_task(make_task(
                o_name,
                "gemm",
                linear_cost(tokens, o_proj_input_dim, hidden, parallel.tp, dtype_bytes),
                device_id,
                {dense_input(phase.name + ".attn_out", attn_out_bytes, {tokens, attn_out_dim_per_shard}, dtype),
                 dense_weight_input(weight_name(layer, "o_proj", tp),
                                    attn_out_dim_per_shard,
                                    hidden,
                                    dtype_bytes,
                                    dtype)}));
            graph.add_edge(attn_name, o_name, attn_out_bytes, phase.name + ".attn_out", "p2p", "dense", 0, {}, dtype);
            o_proj.push_back(o_name);

            if (moe_layer) {
                const auto router_name = shard_name(phase.name, layer, "moe_router", tp);
                graph.add_task(make_task(
                    router_name,
                    "gemm",
                    linear_cost(tokens, hidden, static_cast<long double>(model.num_experts), 1, dtype_bytes),
                    device_id,
                    {dense_input(phase.name + ".post_attn", hidden_full_bytes, {tokens, hidden}, dtype),
                     dense_weight_input(weight_name(layer, "moe_router", tp),
                                        hidden,
                                        static_cast<long double>(model.num_experts),
                                        dtype_bytes,
                                        dtype)}));
                moe_router.push_back(router_name);

                const auto dispatch_name = shard_name(phase.name, layer, "moe_dispatch", tp);
                graph.add_task(make_task(
                    dispatch_name,
                    "copy",
                    {0.0, routed_hidden_bytes},
                    device_id,
                    {dense_input(phase.name + ".router_logits",
                                 router_logits_bytes,
                                 {tokens, static_cast<long double>(model.num_experts)},
                                 dtype),
                     dense_input(phase.name + ".post_attn", hidden_full_bytes, {tokens, hidden}, dtype)}));
                graph.add_edge(router_name,
                               dispatch_name,
                               router_logits_bytes,
                               phase.name + ".router_logits",
                               "p2p",
                               "dense",
                               0,
                               {},
                               dtype);
                moe_dispatch.push_back(dispatch_name);

                const auto expert_name = shard_name(phase.name, layer, "moe_expert_ffn", tp);
                graph.add_task(make_task(
                    expert_name,
                    "gemm",
                    moe_expert_cost(tokens, hidden, moe_inter, experts_per_token, parallel.tp, dtype_bytes),
                    device_id,
                    {dense_input(phase.name + ".moe_dispatched",
                                 local_routed_hidden_bytes,
                                 {local_routed_tokens, hidden},
                                 dtype),
                     dense_weight_input(weight_name(layer, "moe_expert_gate_up", tp),
                                        hidden,
                                        2.0L * moe_inter,
                                        dtype_bytes,
                                        dtype),
                     dense_weight_input(weight_name(layer, "moe_expert_down", tp),
                                        moe_inter,
                                        hidden,
                                        dtype_bytes,
                                        dtype)}));
                moe_expert.push_back(expert_name);

                const auto combine_name = shard_name(phase.name, layer, "moe_combine", tp);
                graph.add_task(make_task(
                    combine_name,
                    "copy",
                    {0.0, hidden_full_bytes},
                    device_id,
                    {dense_input(phase.name + ".moe_expert_out",
                                 local_routed_hidden_bytes,
                                 {local_routed_tokens, hidden},
                                 dtype)}));
                moe_combine.push_back(combine_name);
            } else {
                const auto up_name = shard_name(phase.name, layer, "mlp_gate_up", tp);
                const long double mlp_up_outputs = model.use_gated_mlp ? 2.0L * inter : inter;
                const OpCost up_cost = linear_cost(tokens, hidden, mlp_up_outputs, parallel.tp, dtype_bytes);
                graph.add_task(make_task(up_name,
                                         "gemm",
                                         up_cost,
                                         device_id,
                                         {dense_input(phase.name + ".post_attn", hidden_full_bytes, {tokens, hidden}, dtype),
                                          dense_weight_input(weight_name(layer, "mlp_gate_up", tp),
                                                             hidden,
                                                             shard_dim(mlp_up_outputs, parallel.tp),
                                                             dtype_bytes,
                                                             dtype)}));
                mlp_up.push_back(up_name);

                const auto down_name = shard_name(phase.name, layer, "mlp_down", tp);
                graph.add_task(make_task(
                    down_name,
                    "gemm",
                    linear_cost(tokens, inter, hidden, parallel.tp, dtype_bytes),
                    device_id,
                    {dense_input(phase.name + ".mlp_intermediate",
                                 mlp_intermediate_bytes,
                                 {tokens, shard_dim(inter, parallel.tp)},
                                 dtype),
                     dense_weight_input(weight_name(layer, "mlp_down", tp),
                                        shard_dim(inter, parallel.tp),
                                        hidden,
                                        dtype_bytes,
                                        dtype)}));
                graph.add_edge(up_name,
                               down_name,
                               mlp_intermediate_bytes,
                               phase.name + ".mlp_intermediate",
                               "p2p",
                               "dense",
                               0,
                               {},
                               dtype);
                mlp_down.push_back(down_name);
            }
        }
        if (layer == 0) {
            add_tp_same_rank_edges(graph, prev, qkv, hidden_full_bytes, phase.name + ".hidden", parallel.tp, dtype);
        } else if (parallel.tp > 1 &&
                   stage_for_layer[static_cast<std::size_t>(layer - 1)] != stage) {
            add_tp_stage_boundary_edges(graph,
                                        prev,
                                        qkv,
                                        hidden_full_bytes,
                                        phase.name + ".hidden",
                                        parallel.tp,
                                        devices,
                                        stage_for_layer[static_cast<std::size_t>(layer - 1)],
                                        parallel.pin_shards_to_devices,
                                        phase.name,
                                        layer,
                                        dtype);
        } else {
            add_tp_allreduce_edges(graph,
                                   prev,
                                   qkv,
                                   hidden_full_bytes,
                                   phase.name + ".hidden",
                                   parallel.tp,
                                   phase.name + ".layer_" + std::to_string(layer) + ".hidden_all_reduce",
                                   dtype);
        }
        if (moe_layer) {
            add_tp_allreduce_edges(graph,
                                   o_proj,
                                   moe_router,
                                   hidden_full_bytes,
                                   phase.name + ".post_attn",
                                   parallel.tp,
                                   phase.name + ".layer_" + std::to_string(layer) + ".post_attn_all_reduce",
                                   dtype);
            add_tp_alltoall_edges(graph,
                                  moe_dispatch,
                                  moe_expert,
                                  routed_hidden_bytes,
                                  phase.name + ".moe_dispatch",
                                  parallel.tp,
                                  phase.name + ".layer_" + std::to_string(layer) + ".moe_dispatch_all_to_all",
                                  dtype);
            add_tp_alltoall_edges(graph,
                                  moe_expert,
                                  moe_combine,
                                  routed_hidden_bytes,
                                  phase.name + ".moe_combine",
                                  parallel.tp,
                                  phase.name + ".layer_" + std::to_string(layer) + ".moe_combine_all_to_all",
                                  dtype);
            prev = std::move(moe_combine);
        } else {
            add_tp_allreduce_edges(graph,
                                   o_proj,
                                   mlp_up,
                                   hidden_full_bytes,
                                   phase.name + ".post_attn",
                                   parallel.tp,
                                   phase.name + ".layer_" + std::to_string(layer) + ".post_attn_all_reduce",
                                   dtype);
            prev = std::move(mlp_down);
        }
    }

    std::vector<std::string> logits;
    logits.reserve(static_cast<std::size_t>(parallel.tp));
    const int last_stage = std::max(0, parallel.pp - 1);
    const int device_base = last_stage * parallel.tp;
    for (int tp = 0; tp < parallel.tp; ++tp) {
        const auto& device = devices[static_cast<std::size_t>(device_base + tp)]->id;
        const std::string device_id = parallel.pin_shards_to_devices ? device : "";
        const auto name = root_name(phase.name, "logits", tp);
        graph.add_task(make_task(
            name,
            "gemm",
            linear_cost(tokens, hidden, model.vocab_size, parallel.tp, dtype_bytes),
            device_id,
            {dense_input(phase.name + ".final_hidden", hidden_full_bytes, {tokens, hidden}, dtype),
             dense_weight_input(weight_name(model.num_layers, "lm_head", tp),
                                hidden,
                                shard_dim(static_cast<long double>(model.vocab_size), parallel.tp),
                                dtype_bytes,
                                dtype)}));
        logits.push_back(name);
    }
    add_tp_allreduce_edges(graph,
                           prev,
                           logits,
                           hidden_full_bytes,
                           phase.name + ".final_hidden",
                           parallel.tp,
                           phase.name + ".final_hidden_all_reduce",
                           dtype);

    const auto sampling_name = phase.name + ".sampling";
    const auto& sampling_device = devices[static_cast<std::size_t>(device_base)]->id;
    const double logits_bytes = clamp_double(tokens * static_cast<long double>(model.vocab_size) *
                                             static_cast<long double>(dtype_bytes));
    graph.add_task(make_task(sampling_name,
                             "copy",
                             {0.0,
                              static_cast<double>(bytes_for_elements(
                                  tokens * static_cast<long double>(model.vocab_size), dtype_bytes))},
                             parallel.pin_shards_to_devices ? sampling_device : "",
                             {dense_input(phase.name + ".logits",
                                          logits_bytes,
                                          {tokens, static_cast<long double>(model.vocab_size)},
                                          dtype)}));
    add_tp_allgather_edges(graph,
                           logits,
                           sampling_name,
                           logits_bytes,
                           phase.name + ".logits",
                           parallel.tp,
                           phase.name + ".logits_all_gather",
                           dtype);
    return {sampling_name};
}

}  // namespace

LlmTaskGraphBuildResult build_task_graph(const LlmModelConfig& model,
                                         const LlmRequestConfig& request,
                                         const LlmParallelConfig& parallel,
                                         const hardware_topology::HardwareTopology& topology) {
    LlmTaskGraphBuildResult result;
    result.model = model;
    result.request = request;
    result.parallel = parallel;
    validate_model_and_plan(model, request, parallel, topology, result.diagnostics);

    auto devices = select_topology_aware_llm_devices(topology, parallel);
    const auto stage_for_layer = resolve_stage_for_layer(parallel, model.num_layers);
    if (parallel.pp > 1) {
        result.parallel.stage_for_layer = stage_for_layer;
    }

    const std::size_t devices_per_replica = static_cast<std::size_t>(parallel.tp * parallel.pp);
    for (int dp = 0; dp < parallel.dp; ++dp) {
        const auto replica_offset = static_cast<std::size_t>(dp) * devices_per_replica;
        std::vector<const hardware_topology::Device*> replica_devices;
        replica_devices.reserve(devices_per_replica);
        for (std::size_t i = 0; i < devices_per_replica; ++i) {
            replica_devices.push_back(devices[replica_offset + i]);
        }

        const std::string prefix = parallel.dp == 1 ? "" : "dp" + std::to_string(dp) + ".";
        std::vector<std::string> phase_deps;
        if (request.prompt_len > 0) {
            phase_deps = add_phase(result.graph,
                                   model,
                                   request,
                                   parallel,
                                   replica_devices,
                                   stage_for_layer,
                                   PhaseShape{prefix + "prefill", false, request.prefill_batch_size,
                                              request.prompt_len, 1, request.prompt_len},
                                   {});
        }
        if (request.decode_steps > 0) {
            const int context = request.avg_context_len > 0 ? request.avg_context_len : request.prompt_len;
            add_phase(result.graph,
                      model,
                      request,
                      parallel,
                      replica_devices,
                      stage_for_layer,
                      PhaseShape{prefix + "decode", true, request.decode_batch_size, 1, request.decode_steps, context},
                      phase_deps);
        }
    }

    validate_llm_collective_semantics(result.graph);

    result.diagnostics.push_back("Generated LLM TaskGraph with TP=" + std::to_string(parallel.tp) +
                                 ", PP=" + std::to_string(parallel.pp) + ", CP=" +
                                 std::to_string(parallel.cp) + ", DP replicas=" +
                                 std::to_string(parallel.dp) + ".");
    result.diagnostics.push_back("PP stage partition: " +
                                 stage_partition_summary(stage_for_layer, parallel.pp) + ".");
    result.diagnostics.push_back("Layer type summary: " + layer_type_summary(model) + ".");
    return result;
}

LlmTaskGraphBuildResult build_task_graph_from_config(const std::string& config_path,
                                                     const LlmRequestConfig& request,
                                                     const LlmParallelConfig& parallel,
                                                     const hardware_topology::HardwareTopology& topology,
                                                     const std::string& model_size) {
    LlmModelConfig model;
    std::string error;
    if (!load_model_config_from_json(config_path, model, &error, model_size)) {
        throw std::runtime_error(error);
    }
    return build_task_graph(model, request, parallel, topology);
}

void write_task_graph_json(const std::string& path,
                           const mapping::TaskGraph& graph,
                           const LlmModelConfig& model,
                           const LlmRequestConfig& request,
                           const LlmParallelConfig& parallel,
                           bool omit_device_tags) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open LLM TaskGraph dump: " + path);
    }
    out << "{\n";
    out << "  \"kind\": \"llm_taskgraph_v1\",\n";
    out << "  \"model\": {\n";
    out << "    \"model_type\": ";
    taskflow::json::write_string(out, model.model_type);
    if (!model.model_name.empty()) {
        out << ",\n    \"model_name\": ";
        taskflow::json::write_string(out, model.model_name);
    }
    if (!model.model_size.empty()) {
        out << ",\n    \"model_size\": ";
        taskflow::json::write_string(out, model.model_size);
    }
    out << ",\n    \"num_layers\": " << model.num_layers;
    out << ",\n    \"hidden_size\": " << model.hidden_size;
    out << ",\n    \"intermediate_size\": " << model.intermediate_size;
    out << ",\n    \"num_attention_heads\": " << model.num_attention_heads;
    out << ",\n    \"num_kv_heads\": " << model.num_kv_heads;
    out << ",\n    \"head_dim\": " << model.head_dim;
    out << ",\n    \"vocab_size\": " << model.vocab_size;
    out << ",\n    \"max_position_embeddings\": " << model.max_position_embeddings;
    out << ",\n    \"dtype\": ";
    taskflow::json::write_string(out, dtype_name(model.param_dtype));
    out << ",\n    \"layer_type_summary\": ";
    taskflow::json::write_string(out, layer_type_summary(model));
    if (!model.layer_types.empty()) {
        out << ",\n    \"layer_types\": [";
        for (std::size_t i = 0; i < model.layer_types.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            taskflow::json::write_string(out, model.layer_types[i]);
        }
        out << "]";
    }
    if (has_linear_attention_layers(model)) {
        out << ",\n    \"linear_attention\": {";
        out << "\"linear_num_key_heads\": " << model.linear_num_key_heads;
        out << ", \"linear_num_value_heads\": " << model.linear_num_value_heads;
        out << ", \"linear_key_head_dim\": " << model.linear_key_head_dim;
        out << ", \"linear_value_head_dim\": " << model.linear_value_head_dim;
        out << ", \"linear_conv_kernel_dim\": " << model.linear_conv_kernel_dim;
        out << "}";
    }
    if (model.is_moe) {
        out << ",\n    \"moe\": {";
        out << "\"num_experts\": " << model.num_experts;
        out << ", \"experts_per_token\": " << model.experts_per_token;
        out << ", \"moe_intermediate_size\": " << model.moe_intermediate_size;
        out << "}";
    }
    out << "\n  },\n";
    out << "  \"request\": {\n";
    out << "    \"prefill_batch_size\": " << request.prefill_batch_size;
    out << ",\n    \"prompt_len\": " << request.prompt_len;
    out << ",\n    \"decode_batch_size\": " << request.decode_batch_size;
    out << ",\n    \"decode_steps\": " << request.decode_steps;
    out << ",\n    \"avg_context_len\": " << request.avg_context_len << "\n  },\n";
    out << "  \"parallel\": {\n";
    out << "    \"tp\": " << parallel.tp << ", \"pp\": " << parallel.pp << ", \"cp\": " << parallel.cp
        << ", \"dp\": " << parallel.dp;
    if (!parallel.stage_for_layer.empty()) {
        out << ",\n    \"stage_for_layer\": [";
        for (std::size_t i = 0; i < parallel.stage_for_layer.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << parallel.stage_for_layer[i];
        }
        out << "]";
    }
    out << "\n  },\n";
    out << "  \"tasks\": [\n";
    const auto& tasks = graph.topological_order();
    auto write_string_set = [&](const auto& values) {
        std::vector<std::string> sorted(values.begin(), values.end());
        std::sort(sorted.begin(), sorted.end());
        out << "[";
        for (std::size_t j = 0; j < sorted.size(); ++j) {
            if (j != 0) {
                out << ", ";
            }
            taskflow::json::write_string(out, sorted[j]);
        }
        out << "]";
    };
    auto write_task_tags = [&](const auto& values) {
        std::vector<std::string> filtered;
        filtered.reserve(values.size());
        for (const auto& value : values) {
            if (omit_device_tags && value.rfind("device:", 0) == 0) {
                continue;
            }
            filtered.push_back(value);
        }
        std::sort(filtered.begin(), filtered.end());
        out << "[";
        for (std::size_t j = 0; j < filtered.size(); ++j) {
            if (j != 0) {
                out << ", ";
            }
            taskflow::json::write_string(out, filtered[j]);
        }
        out << "]";
    };
    for (std::size_t i = 0; i < tasks.size(); ++i) {
        const auto& task = tasks[i];
        out << "    {\"name\": ";
        taskflow::json::write_string(out, task.name);
        out << ", \"subtype\": ";
        taskflow::json::write_string(out, task.subtype);
        out << ", \"compute_flops\": ";
        taskflow::json::write_double(out, task.compute_flops);
        out << ", \"memory_bytes\": ";
        taskflow::json::write_double(out, task.memory_bytes);
        if (!task.features.empty()) {
            out << ", \"features\": ";
            write_string_set(task.features);
        }
        bool has_exported_tags = false;
        for (const auto& tag : task.tags) {
            if (!omit_device_tags || tag.rfind("device:", 0) != 0) {
                has_exported_tags = true;
                break;
            }
        }
        if (has_exported_tags) {
            out << ", \"tags\": ";
            write_task_tags(task.tags);
        }
        out << "}" << (i + 1 == tasks.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"edges\": [\n";
    bool first = true;
    for (const auto& task : tasks) {
        for (const auto& edge : graph.successors(task.name)) {
            if (!first) {
                out << ",\n";
            }
            first = false;
            out << "    {\"src\": ";
            taskflow::json::write_string(out, edge.src);
            out << ", \"dst\": ";
            taskflow::json::write_string(out, edge.dst);
            out << ", \"tensor_bytes\": ";
            taskflow::json::write_double(out, edge.tensor_bytes);
            if (!edge.tensor_id.empty()) {
                out << ", \"tensor_id\": ";
                taskflow::json::write_string(out, edge.tensor_id);
            }
            out << ", \"comm_kind\": ";
            taskflow::json::write_string(out, edge.comm_kind.empty() ? "p2p" : edge.comm_kind);
            if (edge.comm_participants > 0) {
                out << ", \"comm_participants\": " << edge.comm_participants;
            }
            if (!edge.comm_group.empty()) {
                out << ", \"comm_group\": ";
                taskflow::json::write_string(out, edge.comm_group);
            }
            out << "}";
        }
    }
    out << "\n  ]\n";
    out << "}\n";
}

}  // namespace llm
