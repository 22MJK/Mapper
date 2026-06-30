# mapper

`mapper` 是一个 C++20 的任务映射工具：输入硬件拓扑和任务描述，估算计算/通信开销，把任务分配到设备上，最后输出 Chakra-ET 风格的 `taskflow.json`。

它现在支持两类输入：

- 普通 workload：已经写好的固定 DAG。
- LLM config：从模型 config 和 request 形状生成 LLM TaskGraph，再调度成 taskflow。

## Build

```bash
make
```

或：

```bash
cmake -S . -B build-cmake
cmake --build build-cmake
```

## 普通 Workload

普通 workload 是已经物化好的任务图，只做映射和调度。

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --workload=examples/cg_iteration_workload.json \
  --mapper=aeft \
  --parallel=auto \
  --out=taskflow.json
```

可用 mapper：

```text
heft, aeft, peft, peft_lc, hoft, greedy, exhaustive, exhaustive_bb
```

`heft` 是原始 HEFT priority-list + insertion-based EFT 版本；为了生成合法 taskflow，仍遵守任务的硬设备可行性约束。`aeft` 是 AEFT（Average Earliest Finish Time）版本，使用 IOCT lookahead 生成任务优先级，并在处理器选择时用当前 EFT 与后继任务的平均预测完成时间进行权衡。
`peft_lc` 是 PEFT 的低开销变体：在较大的设备候选空间下使用 shape-aware communication-cost cache、拓扑邻近候选和少量 continuation-best 候选来限制 OCT/placement 评估，把 PEFT 中按边枚举所有设备对的主项替换为有界候选集合；当设备候选空间过小或缺少正大小 P2P 通信边时回退到 exact PEFT。

## Workload 层 rank parallel

如果输入已经是未切分的 workload，从 workload 层切分：

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --workload=examples/cg_iteration_workload.json \
  --workload-rank-parallel \
  --mapper=aeft \
  --dump-workload=output/your_rank_workload.json \
  --out=output/your_rank_taskflow.json
```

`--workload-rank-parallel` 会在 workload parse 之后，根据当前拓扑里的 GPU rank 数把 GPU-compatible task 切成 `@r0`, `@r1`, ... 分片；本次调度内部会按 rank 绑定到对应 GPU。`--dump-workload` 导出的切分 workload 不写物理 `device:GPUx` tag，只保留逻辑分片。

## LLM config 工作流

从未切分的 config 进入：

```text
--llm-config -> 生成/选择 LLM TaskGraph -> 调度 -> taskflow
```

这里有三种用法。

手动指定并行方案：

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --llm-config=examples/llama_config.json \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-tp=2 \
  --llm-pp=2 \
  --llm-dp=1 \
  --mapper=hoft \
  --llm-dump-taskgraph=output/llama_taskgraph.json \
  --out=output/llama_taskflow.json
```

自动枚举 TP/PP/DP：

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --llm-config=examples/gpt3_config.json \
  --llm-size=13B \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-auto-parallel \
  --mapper=hoft \
  --llm-dump-taskgraph=output/gpt3_13b_taskgraph.json \
  --out=output/gpt3_13b_taskflow.json
```

按 topology GPU 数自动选择并行方案，端到端生成 taskflow：

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --llm-config=examples/llama_config.json \
  --llm-prompt-len=128 \
  --llm-decode-steps=16 \
  --llm-use-all-gpus \
  --mapper=peft_lc \
  --out=output/llama_topology_taskflow.json
```

`--llm-use-all-gpus`（同 `--llm-rank-parallel`）会根据当前拓扑中的 GPU 数量选择满足 `tp * pp * dp == GPU 数` 的方案；规则会先尝试保持 `dp=1` 并选择最大的可行 TP，若模型层数或 head 数导致无法用满 GPU，则轻量回退到 DP 副本并行。该路径不对每个候选重复跑 mapper/makespan 估计；LLM rank 到 GPU 的绑定使用低开销 topology-aware ordering，只看直连链路、最近交换节点和 parent 等拓扑签名，不估计计算代价。生成的内部图包含对应的 TP 集合通信和 `device:GPUx` 绑定，最终输出时会直接复用这些绑定，避免再对全 pinned 图运行 mapper。只有在需要排查或复现实验时，才额外加 `--llm-dump-taskgraph=PATH` 导出中间图。

注意：

- `--workload` 和 `--llm-config` 二选一。
- `--workload` 是固定 DAG；LLM 场景优先用 `--llm-config --llm-use-all-gpus` 端到端生成 taskflow，避免额外中间 workload。
- `--workload-rank-parallel` 只用于没有 LLM 并行语义的普通 workload。
- `--llm-auto-parallel` 和 `--llm-use-all-gpus` 只能和 `--llm-config` 一起用。
- `--llm-cp` 当前必须是 `1`。

## LLM request 示例

仓库保留了常用模型的 config 示例。实验批量运行时，可以用下列 request 形状组合这些 config：

| 模型 | config | request |
| --- | --- | --- |
| Qwen | `examples/qwenconfig.json` | shortest / longest |
| Llama | `examples/llama_config.json` | shortest / longest |
| GPT-3 13B | `examples/gpt3_config.json` + `--llm-size=13B` | shortest / longest |
| Gemma | `examples/gemma_config.json` | shortest / longest |
| Mixtral | `examples/mixtral_config.json` | shortest / longest |

request 定义：

| profile | prompt_len | decode_steps | batch |
| --- | ---: | ---: | ---: |
| shortest | 128 | 16 | 1 |
| longest | 8192 | 128 | 1 |

## 输出

`mapper_demo` 会写出 `taskflow.json`，并在终端打印：

- makespan
- DAG 深度
- 跨设备通信量
- 设备任务分布
- 热点算子

如果使用 `--llm-dump-taskgraph=PATH`，还会额外写出 LLM 前端生成或自动选择后的 TaskGraph，方便检查 TP/PP/DP 是否符合预期。

## 可视化

```bash
python3 visualize/taskflow_viz.py \
  --input taskflow.json \
  --abstract \
  --abstract-by device \
  --output taskflow_device.svg
```

也可以让 `mapper_demo` 自动尝试渲染：

```bash
./mapper_demo \
  --hardware=examples/realsys.json \
  --workload=examples/cg_iteration_workload.json \
  --out=taskflow.json \
  --viz
```

## 目录

- `main.cpp`: CLI 入口。
- `mapper/`: 调度入口、并行规划、运行摘要。
- `mapping/`: TaskGraph、cost model、mapper 实现。
- `llm/`: LLM config 解析和 TaskGraph 构建。
- `taskflow/`: taskflow writer。
- `workload/`: 普通 workload parser。
- `examples/`: 示例硬件、workload、LLM config。
- `visualize/`: taskflow/workload 可视化脚本。

大规模 workload、profiling 原始数据、论文构建产物和实验输出不随源码仓库分发；
这些本地数据目录已写入 `.gitignore`。
