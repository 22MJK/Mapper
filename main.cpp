#include "hardware_topology/topology.h"
#include "mapper/mapper.h"
#include "hardware_topology/json_io.h"
#include "workload/json_io.h"
#include "workload/workload.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <sys/wait.h>
#endif

namespace {

int parse_int_arg(const std::string& arg, const std::string& prefix, int default_value) {
    if (arg.rfind(prefix, 0) != 0) {
        return default_value;
    }
    const auto value_str = arg.substr(prefix.size());
    return std::atoi(value_str.c_str());
}

std::string parse_string_arg(const std::string& arg, const std::string& prefix, const std::string& default_value) {
    if (arg.rfind(prefix, 0) != 0) {
        return default_value;
    }
    return arg.substr(prefix.size());
}

std::string shell_escape(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (char ch : value) {
        if (ch == '"' || ch == '\\') {
            out << '\\';
        }
        out << ch;
    }
    out << '"';
    return out.str();
}

std::string format_top_counts(const std::vector<mapper::NamedCount>& items, std::size_t limit) {
    if (items.empty() || limit == 0) {
        return "N/A";
    }
    const std::size_t n = std::min(limit, items.size());
    std::ostringstream out;
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << items[i].name << ":" << items[i].count;
    }
    return out.str();
}

std::string format_top_bytes(const std::vector<mapper::NamedBytes>& items, std::size_t limit) {
    if (items.empty() || limit == 0) {
        return "N/A";
    }
    std::vector<mapper::NamedBytes> filtered;
    filtered.reserve(items.size());
    for (const auto& item : items) {
        if (item.bytes > 0) {
            filtered.push_back(item);
        }
    }
    if (filtered.empty()) {
        return "N/A";
    }
    const std::size_t n = std::min(limit, filtered.size());
    std::ostringstream out;
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << filtered[i].name << ":" << filtered[i].bytes << "B";
    }
    return out.str();
}

void print_schedule_summary(const mapper::RunResult& result, const std::string& taskflow_path) {
    const std::string quoted_taskflow = shell_escape(taskflow_path);
    double cross_ratio = 0.0;
    if (result.total_edge_bytes > 0) {
        cross_ratio = static_cast<double>(result.cross_device_edge_bytes) /
                      static_cast<double>(result.total_edge_bytes);
    }

    std::cout << "Schedule summary:\n";
    std::cout << "  graph: tasks=" << result.task_count
              << ", edges=" << result.edge_count
              << ", dag_depth=" << result.dag_depth
              << ", sources=" << result.source_count
              << ", sinks=" << result.sink_count << "\n";
    std::cout << "  communication: transfer_bytes=" << result.total_edge_bytes << "B"
              << ", cross_device_transfer_bytes=" << result.cross_device_edge_bytes << "B"
              << " (" << result.cross_device_edge_count << " edges, "
              << std::fixed << std::setprecision(2) << (cross_ratio * 100.0) << "%)\n";
    std::cout << "  top task subtypes: " << format_top_counts(result.task_subtype_counts, 6) << "\n";
    std::cout << "  tasks per device: " << format_top_counts(result.device_task_counts, 8) << "\n";
    std::cout << "  top comm kinds by bytes: " << format_top_bytes(result.comm_kind_bytes, 6) << "\n";
    std::cout << "  suggested views:\n";
    std::cout << "    python3 visualize/taskflow_viz.py --input " << quoted_taskflow
              << " --abstract --abstract-by device --output taskflow_device.svg\n";
    std::cout << "    python3 visualize/taskflow_viz.py --input " << quoted_taskflow
              << " --abstract --abstract-by iter --output taskflow_iter.svg\n";
    std::cout << "    python3 visualize/taskflow_viz.py --input " << quoted_taskflow
              << " --abstract --abstract-by op --output taskflow_op.svg\n";
    std::cout << "    python3 visualize/taskflow_viz.py --input " << quoted_taskflow
              << " --format mermaid --group-by-device --output taskflow.mmd\n";
}

void try_generate_taskflow_svg(const std::string& taskflow_path,
                               int viz_max_tasks,
                               int viz_max_edges,
                               bool viz_force,
                               const std::string& viz_summary_path) {
    namespace fs = std::filesystem;
    const fs::path script = fs::path("visualize") / "taskflow_viz.py";
    if (!fs::exists(script)) {
        std::cerr << "Warning: visualize/taskflow_viz.py not found; skip visualization.\n";
        return;
    }
    const fs::path taskflow_file(taskflow_path);
    fs::path out_path = taskflow_file;
    out_path.replace_extension(".svg");
    std::string cmd = "python3 ";
    cmd += shell_escape(script.string());
    cmd += " --input ";
    cmd += shell_escape(taskflow_file.string());
    cmd += " --output ";
    cmd += shell_escape(out_path.string());
    cmd += " --max-nodes ";
    cmd += std::to_string(viz_max_tasks);
    cmd += " --max-edges ";
    cmd += std::to_string(viz_max_edges);
    cmd += " --quiet-skip-summary";
    if (viz_force) {
        cmd += " --force-render";
    }
    if (!viz_summary_path.empty()) {
        cmd += " --summary ";
        cmd += shell_escape(viz_summary_path);
    }
    const int rc = std::system(cmd.c_str());
    int exit_code = rc;
#ifndef _WIN32
    if (rc != -1 && WIFEXITED(rc)) {
        exit_code = WEXITSTATUS(rc);
    }
#endif
    if (exit_code == 3) {
        std::cerr << "Visualization skipped for large taskflow."
                  << " You can force it with --viz-force";
        if (!viz_summary_path.empty()) {
            std::cerr << " (summary: " << viz_summary_path << ")";
        }
        std::cerr << ".\n";
        return;
    }
    if (exit_code != 0) {
        std::cerr << "Warning: taskflow visualization failed (exit " << exit_code << ").\n";
    }
}

}  // namespace

int main(int argc, char** argv) {
    int parts = 0;
    std::string time_unit = "s";
    bool time_unit_set = false;
    std::string taskflow_path = "taskflow.json";
    std::string out_format = "json";
    std::string et_prefix;
    std::string hardware_path;
    std::string workload_path;
    std::string mapper_name = "aeft";
    std::string parallel_mode = "none";
    std::string llm_config_path;
    std::string llm_size;
    std::string llm_dump_taskgraph_path;
    std::string workload_dump_taskgraph_path;
    int llm_prefill_batch_size = 1;
    int llm_prompt_len = 2048;
    int llm_decode_batch_size = 1;
    int llm_decode_steps = 0;
    int llm_avg_context_len = 2048;
    int llm_tp = 1;
    int llm_pp = 1;
    int llm_cp = 1;
    int llm_dp = 1;
    bool llm_auto_parallel = false;
    bool llm_rank_parallel = false;
    bool workload_rank_parallel = false;
    bool force_exhaustive = false;
    bool enable_viz = false;
    bool viz_force = false;
    int viz_max_tasks = 2500;
    int viz_max_edges = 10000;
    std::string viz_summary_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        parts = parse_int_arg(arg, "--parts=", parts);
        if (arg == "--parts" && i + 1 < argc) {
            parts = std::atoi(argv[++i]);
            continue;
        }
        if (arg.rfind("--time_unit=", 0) == 0) {
            time_unit = parse_string_arg(arg, "--time_unit=", time_unit);
            time_unit_set = true;
            continue;
        }
        if (arg == "--time_unit" && i + 1 < argc) {
            time_unit = argv[++i];
            time_unit_set = true;
            continue;
        }
        taskflow_path = parse_string_arg(arg, "--out=", taskflow_path);
        taskflow_path = parse_string_arg(arg, "--output=", taskflow_path);
        if (arg == "--out" && i + 1 < argc) {
            taskflow_path = argv[++i];
            continue;
        }
        if (arg == "--output" && i + 1 < argc) {
            taskflow_path = argv[++i];
            continue;
        }
        out_format = parse_string_arg(arg, "--out-format=", out_format);
        out_format = parse_string_arg(arg, "--output-format=", out_format);
        if ((arg == "--out-format" || arg == "--output-format") && i + 1 < argc) {
            out_format = argv[++i];
            continue;
        }
        et_prefix = parse_string_arg(arg, "--et-prefix=", et_prefix);
        if (arg == "--et-prefix" && i + 1 < argc) {
            et_prefix = argv[++i];
            continue;
        }
        hardware_path = parse_string_arg(arg, "--hardware=", hardware_path);
        if (arg == "--hardware" && i + 1 < argc) {
            hardware_path = argv[++i];
            continue;
        }
        workload_path = parse_string_arg(arg, "--workload=", workload_path);
        if (arg == "--workload" && i + 1 < argc) {
            workload_path = argv[++i];
            continue;
        }
        mapper_name = parse_string_arg(arg, "--mapper=", mapper_name);
        if (arg == "--mapper" && i + 1 < argc) {
            mapper_name = argv[++i];
            continue;
        }
        parallel_mode = parse_string_arg(arg, "--parallel=", parallel_mode);
        if (arg == "--parallel" && i + 1 < argc) {
            parallel_mode = argv[++i];
            continue;
        }
        llm_config_path = parse_string_arg(arg, "--llm-config=", llm_config_path);
        llm_config_path = parse_string_arg(arg, "--llm-model=", llm_config_path);
        if ((arg == "--llm-config" || arg == "--llm-model") && i + 1 < argc) {
            llm_config_path = argv[++i];
            continue;
        }
        llm_size = parse_string_arg(arg, "--llm-size=", llm_size);
        if (arg == "--llm-size" && i + 1 < argc) {
            llm_size = argv[++i];
            continue;
        }
        llm_prefill_batch_size = parse_int_arg(arg, "--llm-prefill-batch=", llm_prefill_batch_size);
        if (arg == "--llm-prefill-batch" && i + 1 < argc) {
            llm_prefill_batch_size = std::atoi(argv[++i]);
            continue;
        }
        llm_prompt_len = parse_int_arg(arg, "--llm-prompt-len=", llm_prompt_len);
        if (arg == "--llm-prompt-len" && i + 1 < argc) {
            llm_prompt_len = std::atoi(argv[++i]);
            continue;
        }
        llm_decode_batch_size = parse_int_arg(arg, "--llm-decode-batch=", llm_decode_batch_size);
        if (arg == "--llm-decode-batch" && i + 1 < argc) {
            llm_decode_batch_size = std::atoi(argv[++i]);
            continue;
        }
        llm_decode_steps = parse_int_arg(arg, "--llm-decode-steps=", llm_decode_steps);
        if (arg == "--llm-decode-steps" && i + 1 < argc) {
            llm_decode_steps = std::atoi(argv[++i]);
            continue;
        }
        llm_avg_context_len = parse_int_arg(arg, "--llm-avg-context-len=", llm_avg_context_len);
        if (arg == "--llm-avg-context-len" && i + 1 < argc) {
            llm_avg_context_len = std::atoi(argv[++i]);
            continue;
        }
        llm_tp = parse_int_arg(arg, "--llm-tp=", llm_tp);
        if (arg == "--llm-tp" && i + 1 < argc) {
            llm_tp = std::atoi(argv[++i]);
            continue;
        }
        llm_pp = parse_int_arg(arg, "--llm-pp=", llm_pp);
        if (arg == "--llm-pp" && i + 1 < argc) {
            llm_pp = std::atoi(argv[++i]);
            continue;
        }
        llm_cp = parse_int_arg(arg, "--llm-cp=", llm_cp);
        if (arg == "--llm-cp" && i + 1 < argc) {
            llm_cp = std::atoi(argv[++i]);
            continue;
        }
        llm_dp = parse_int_arg(arg, "--llm-dp=", llm_dp);
        if (arg == "--llm-dp" && i + 1 < argc) {
            llm_dp = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--llm-auto-parallel" || arg == "--llm-auto") {
            llm_auto_parallel = true;
            continue;
        }
        if (arg == "--llm-rank-parallel" || arg == "--llm-topology-parallel" ||
            arg == "--llm-use-all-gpus") {
            llm_rank_parallel = true;
            continue;
        }
        if (arg == "--workload-rank-parallel" || arg == "--workload-use-all-gpus") {
            workload_rank_parallel = true;
            continue;
        }
        llm_dump_taskgraph_path = parse_string_arg(arg, "--llm-dump-taskgraph=", llm_dump_taskgraph_path);
        if (arg == "--llm-dump-taskgraph" && i + 1 < argc) {
            llm_dump_taskgraph_path = argv[++i];
            continue;
        }
        workload_dump_taskgraph_path =
            parse_string_arg(arg, "--dump-workload=", workload_dump_taskgraph_path);
        workload_dump_taskgraph_path =
            parse_string_arg(arg, "--workload-dump-taskgraph=", workload_dump_taskgraph_path);
        if ((arg == "--dump-workload" || arg == "--workload-dump-taskgraph") && i + 1 < argc) {
            workload_dump_taskgraph_path = argv[++i];
            continue;
        }
        viz_max_tasks = parse_int_arg(arg, "--viz-max-tasks=", viz_max_tasks);
        if (arg == "--viz-max-tasks" && i + 1 < argc) {
            viz_max_tasks = std::atoi(argv[++i]);
            continue;
        }
        viz_max_edges = parse_int_arg(arg, "--viz-max-edges=", viz_max_edges);
        if (arg == "--viz-max-edges" && i + 1 < argc) {
            viz_max_edges = std::atoi(argv[++i]);
            continue;
        }
        viz_summary_path = parse_string_arg(arg, "--viz-summary=", viz_summary_path);
        if (arg == "--viz-summary" && i + 1 < argc) {
            viz_summary_path = argv[++i];
            continue;
        }
        if (arg == "--viz-force") {
            enable_viz = true;
            viz_force = true;
            continue;
        }
        if (arg == "--viz") {
            enable_viz = true;
            continue;
        }
        if (arg == "--no-viz") {
            enable_viz = false;
            continue;
        }
        if (arg == "--force-exhaustive") {
            force_exhaustive = true;
            continue;
        }
    }
    if (hardware_path.empty() || (workload_path.empty() && llm_config_path.empty())) {
        std::cerr << "Usage: mapper_demo --hardware=PATH (--workload=PATH | --llm-config=PATH) [--parts=P] [--time_unit=UNIT] "
                     "[--mapper=heft|aeft|peft|peft_lc|hoft|greedy|exhaustive|exhaustive_bb] [--parallel=none|matrix|matrix_parallel|auto|llm] [--out=PATH|--output=PATH] "
                     "[--out-format=json|chakra-et|both] [--et-prefix=PREFIX] [--workload-rank-parallel] [--dump-workload=PATH] [--llm-size=SIZE] [--llm-auto-parallel|--llm-use-all-gpus] [--llm-tp=N] [--llm-pp=N] [--llm-cp=N] [--llm-dp=N] [--llm-prompt-len=N] [--llm-decode-steps=N] "
                     "[--force-exhaustive] [--viz|--no-viz] [--viz-max-tasks=N] [--viz-max-edges=N] [--viz-force] [--viz-summary=PATH]\n";
        return 2;
    }
    if (out_format != "json" && out_format != "chakra-et" && out_format != "both") {
        std::cerr << "--out-format must be one of: json, chakra-et, both.\n";
        return 2;
    }
    if ((out_format == "chakra-et" || out_format == "both") && et_prefix.empty()) {
        et_prefix = taskflow_path;
    }
    if (!workload_path.empty() && !llm_config_path.empty()) {
        std::cerr << "Specify only one of --workload or --llm-config. "
                     "--workload is already materialized; --llm-config is the unsplit LLM source for TP/PP/DP planning.\n";
        return 2;
    }
    if (llm_auto_parallel && llm_config_path.empty()) {
        std::cerr << "--llm-auto-parallel requires --llm-config; it cannot search TP/PP/DP on an already materialized workload.\n";
        return 2;
    }
    if (llm_rank_parallel && llm_config_path.empty()) {
        std::cerr << "--llm-rank-parallel requires --llm-config; it needs the unsplit LLM source to shard by topology ranks.\n";
        return 2;
    }
    if (workload_rank_parallel && workload_path.empty()) {
        std::cerr << "--workload-rank-parallel requires --workload; it operates after workload parsing.\n";
        return 2;
    }
    if (workload_rank_parallel && !llm_config_path.empty()) {
        std::cerr << "--workload-rank-parallel operates on --workload input; use --llm-rank-parallel with --llm-config.\n";
        return 2;
    }
    if (llm_auto_parallel && llm_rank_parallel) {
        std::cerr << "Specify only one of --llm-auto-parallel or --llm-rank-parallel.\n";
        return 2;
    }
    if (viz_max_tasks < 0) {
        viz_max_tasks = 0;
    }
    if (viz_max_edges < 0) {
        viz_max_edges = 0;
    }

    hardware_topology::HardwareTopology topology;
    std::string error;
    if (!hardware_topology::load_from_json(hardware_path, topology, &error)) {
        std::cerr << "Failed to load hardware topology: " << error << "\n";
        return 2;
    }
    if (!time_unit_set) {
        time_unit = topology.time_unit();
    }

    workload::Workload workload("workload", {}, {}, {}, {}, {});
    if (!workload_path.empty()) {
        if (!workload::load_from_json(workload_path, workload, &error)) {
            std::cerr << "Failed to load workload: " << error << "\n";
            return 2;
        }
    }

    mapper::Options options;
    options.parts = parts;
    options.time_unit = time_unit;
    options.mapper = mapper_name;
    options.parallel = !llm_config_path.empty() && parallel_mode == "none" ? "llm" : parallel_mode;
    options.force_exhaustive = force_exhaustive;
    options.llm_config_path = llm_config_path;
    options.llm_size = llm_size;
    options.llm_prefill_batch_size = llm_prefill_batch_size;
    options.llm_prompt_len = llm_prompt_len;
    options.llm_decode_batch_size = llm_decode_batch_size;
    options.llm_decode_steps = llm_decode_steps;
    options.llm_avg_context_len = llm_avg_context_len;
    options.llm_tp = llm_tp;
    options.llm_pp = llm_pp;
    options.llm_cp = llm_cp;
    options.llm_dp = llm_dp;
    options.llm_auto_parallel = llm_auto_parallel;
    options.llm_rank_parallel = llm_rank_parallel;
    options.llm_dump_taskgraph_path = llm_dump_taskgraph_path;
    options.workload_rank_parallel = workload_rank_parallel;
    options.workload_dump_taskgraph_path = workload_dump_taskgraph_path;
    options.output_format = out_format;
    options.et_prefix = et_prefix;
    const auto mapper_start = std::chrono::steady_clock::now();
    mapper::RunResult result;
    try {
        result = mapper::write_taskflow(topology, workload, taskflow_path, options);
    } catch (const std::exception& ex) {
        std::cerr << "Failed to generate taskflow: " << ex.what() << "\n";
        return 1;
    }
    const auto mapper_end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> mapper_runtime = mapper_end - mapper_start;
    if (out_format == "json") {
        std::cout << "Wrote " << taskflow_path << "\n";
    } else if (out_format == "chakra-et") {
        std::cout << "Wrote " << et_prefix << ".*.et\n";
    } else {
        std::cout << "Wrote " << taskflow_path << " and " << et_prefix << ".*.et\n";
    }
    std::cout << "Estimated makespan: " << std::fixed << std::setprecision(9)
              << result.estimated_makespan_s << " s"
              << " (parallel=" << result.selected_parallel
              << ", tasks=" << result.task_count << ")\n";
    std::cout << "Mapper runtime (excluding visualization): "
              << std::fixed << std::setprecision(6)
              << mapper_runtime.count() << " s\n";
    print_schedule_summary(result, taskflow_path);
    std::cout.flush();
    if (enable_viz) {
        try_generate_taskflow_svg(taskflow_path, viz_max_tasks, viz_max_edges, viz_force, viz_summary_path);
    }
    return 0;
}
