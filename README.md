# Mapper

Mapper is a C++20 task-to-device mapping tool for heterogeneous systems. It reads
a hardware topology and a workload description, estimates compute and
communication cost, assigns tasks to devices, and writes taskflow JSON or
per-rank Chakra ET traces.

Mapper supports two input paths:

- Workload JSON: map an existing task DAG.
- LLM config JSON: generate an LLM task graph from model and request shapes, then
  map it to the target topology.

## Build

Use `make`:

```bash
make
```

Or build with CMake:

```bash
cmake -S . -B build-cmake
cmake --build build-cmake
```

## Usage

Create an output directory first:

```bash
mkdir -p output
```

### Map A Workload DAG

Use `--workload` when the task graph already exists:

```bash
./mapper_demo \
  --hardware=../tools/examples/topologies/tc_2gpu_1cpu_switch_memory_pool.json \
  --workload=examples/cg_iteration_workload.json \
  --mapper=aeft \
  --parallel=auto \
  --out=output/taskflow.json
```

### Generate And Map An LLM Task Graph

Use `--llm-config` when Mapper should generate the task graph from a model
configuration:

```bash
./mapper_demo \
  --hardware=../tools/examples/topologies/tc_2gpu_1cpu_switch_memory_pool.json \
  --llm-config=examples/qwenconfig.json \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-tp=2 \
  --llm-pp=1 \
  --llm-dp=1 \
  --mapper=hoft \
  --out=output/llm_taskflow.json
```

For configs that contain multiple model sizes, add `--llm-size`:

```bash
./mapper_demo \
  --hardware=../tools/examples/topologies/tc_8xA800_NVLink_2cpu_upi.json \
  --llm-config=examples/llama_config.json \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-auto-parallel \
  --mapper=hoft \
  --out=output/llm_taskflow.json
```

## Mapping Modes

Choose the mapper with `--mapper`:

| Mapper | Notes |
| --- | --- |
| `heft` | Classic HEFT priority list with insertion-based EFT placement. |
| `aeft` | Average Earliest Finish Time with IOCT-style lookahead. |
| `peft` | Predict Earliest Finish Time using optimistic cost tables. |
| `peft_lc` | Lower-cost PEFT variant with bounded placement candidates and communication-cost caching. |
| `hoft` | Heterogeneous Optimistic Finish Time. |
| `greedy` | Simple baseline placement. |
| `exhaustive`, `exhaustive_bb` | Exact search variants for small graphs. |

## Parallelism Helpers

### Workload Rank Parallel

For ordinary workload DAGs without built-in LLM parallel semantics, Mapper can
split GPU-compatible tasks by GPU rank:

```bash
./mapper_demo \
  --hardware=../tools/examples/topologies/tc_8xA800_NVLink_2cpu_upi.json \
  --workload=examples/cg_iteration_workload.json \
  --workload-rank-parallel \
  --mapper=aeft \
  --dump-workload=output/rank_workload.json \
  --out=output/rank_taskflow.json
```

`--dump-workload` writes the transformed workload with logical rank shards such
as `@r0`, `@r1`, and so on.

### Use All GPUs For LLMs

For LLM configs, `--llm-use-all-gpus` chooses a TP/PP/DP plan that matches the
number of GPUs in the topology:

```bash
./mapper_demo \
  --hardware=../tools/examples/topologies/tc_8xA800_NVLink_2cpu_upi.json \
  --llm-config=examples/qwenconfig.json \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-use-all-gpus \
  --mapper=peft_lc \
  --out=output/llm_all_gpus_taskflow.json
```

This path generates rank-aware LLM tasks and GPU bindings directly. Add
`--llm-dump-taskgraph=PATH` when you want to inspect the generated intermediate
task graph.

Notes:

- `--workload` and `--llm-config` are mutually exclusive.
- `--workload-rank-parallel` is for ordinary workload DAGs.
- `--llm-auto-parallel` and `--llm-use-all-gpus` require `--llm-config`.
- `--llm-cp` currently must be `1`.

## Visualize A Taskflow

Render a device-level SVG:

```bash
python3 visualize/taskflow_viz.py \
  --input output/taskflow.json \
  --abstract \
  --abstract-by device \
  --output output/taskflow_device.svg
```

`mapper_demo` can also try to invoke the visualizer after mapping:

```bash
./mapper_demo \
  --hardware=../tools/examples/topologies/tc_2gpu_1cpu_switch_memory_pool.json \
  --workload=examples/cg_iteration_workload.json \
  --out=output/taskflow.json \
  --viz
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `main.cpp` | CLI entry point. |
| `mapper/` | High-level mapping workflow and run summaries. |
| `mapping/` | Task graph, cost model, schedule model, and mapper implementations. |
| `llm/` | LLM config parser and LLM task graph builder. |
| `hardware_topology/` | Topology model and JSON parser. |
| `workload/` | Workload JSON parser. |
| `taskflow/` | Chakra-ET-style taskflow writer. |
| `visualize/` | Taskflow and workload visualization scripts. |

## License

Copyright (c) 2026 Yuchen Fan, Minghong Sun, Jikui Ma, and Shunyu Mao.
Released under the [MIT License](LICENSE).
