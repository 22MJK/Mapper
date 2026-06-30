# Mapper

Mapper is a C++20 task-to-device mapping tool for heterogeneous systems. It reads
a hardware topology and a workload description, estimates compute and
communication cost, assigns tasks to devices, and writes a Chakra-ET-style
`taskflow.json`.

It supports two input paths:

- Workload JSON: use an existing task DAG.
- LLM config JSON: generate an LLM task graph from model and request shapes, then
  map it to the target topology.

## Quick Start

Build with `make`:

```bash
make
mkdir -p output
```

Run the small CG example:

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --workload=examples/cg_iteration_workload.json \
  --mapper=aeft \
  --parallel=auto \
  --out=output/cg_taskflow.json
```

The command writes `output/cg_taskflow.json` and prints a short schedule summary:

- estimated makespan
- mapper runtime
- DAG depth
- cross-device communication
- tasks per device
- top operator types

You can also build with CMake:

```bash
cmake -S . -B build-cmake
cmake --build build-cmake
```

## Inputs

Mapper needs one hardware file and one workload source.

### Hardware Topology

`--hardware` points to a topology JSON. Example files:

- `examples/realsys.json`
- `examples/tc_8xA800_NVLink_2cpu_upi.json`

The topology describes devices, links, bandwidth, latency, and device groups.

### Workload DAG

Use `--workload` when the task graph already exists:

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --workload=examples/supernodal_cholesky_demo.json \
  --mapper=peft_lc \
  --out=output/supernodal_taskflow.json
```

Workload JSON files describe tasks, tensors, dependencies, operator types, and
data sizes. See `examples/` for compact, runnable inputs.

### LLM Config

Use `--llm-config` when you want Mapper to generate the LLM task graph:

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --llm-config=examples/llama_config.json \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-tp=2 \
  --llm-pp=1 \
  --llm-dp=1 \
  --mapper=hoft \
  --out=output/llama_taskflow.json
```

For model families with multiple sizes in one config, pass `--llm-size`:

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --llm-config=examples/gpt3_config.json \
  --llm-size=13B \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-auto-parallel \
  --mapper=hoft \
  --out=output/gpt3_13b_taskflow.json
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
  --hardware=examples/realsys.json \
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
  --hardware=examples/realsys.json \
  --llm-config=examples/llama_config.json \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-use-all-gpus \
  --mapper=peft_lc \
  --out=output/llama_all_gpus_taskflow.json
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
  --input output/cg_taskflow.json \
  --abstract \
  --abstract-by device \
  --output output/taskflow_device.svg
```

`mapper_demo` can also try to invoke the visualizer after mapping:

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --workload=examples/cg_iteration_workload.json \
  --out=output/cg_taskflow.json \
  --viz
```

## Example LLM Requests

The repository includes config examples for common model families:

| Model | Config | Suggested request labels |
| --- | --- | --- |
| Qwen | `examples/qwenconfig.json` | shortest / longest |
| Llama | `examples/llama_config.json` | shortest / longest |
| GPT-3 13B | `examples/gpt3_config.json` + `--llm-size=13B` | shortest / longest |
| Gemma | `examples/gemma_config.json` | shortest / longest |
| Mixtral | `examples/mixtral_config.json` | shortest / longest |

Request shapes used in experiments:

| Label | Prompt length | Decode steps | Batch |
| --- | ---: | ---: | ---: |
| shortest | 128 | 16 | 1 |
| longest | 8192 | 128 | 1 |

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
| `examples/` | Small topology, workload, and LLM config examples. |
| `visualize/` | Taskflow and workload visualization scripts. |
| `workload_sparse/` | Sparse workload generators and profiling helper source. |
