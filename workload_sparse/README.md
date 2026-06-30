# Sparse Workload Generators

从稀疏 SPD 矩阵生成工作负载 JSON（tasks + tensors），用于性能仿真与调度建模。主入口基于 CHOLMOD Algorithm 887 Left-looking 超节点法生成 supernodal sparse Cholesky workload；`generators/` 下提供额外 workload 生成器。

> **参考论文**：Chen Y, Davis T A, et al. "Algorithm 887: CHOLMOD, Supernodal Sparse Cholesky Factorization." *ACM TOMS*, 35(3), 2008.

---

## 目录结构

| 路径 | 作用 |
|---|---|
| `main.py` | Supernodal Cholesky 命令行入口 |
| `generators/cg_workload_generator.py` | Conjugate Gradient (CG) workload 生成器 |
| `matrix_provider.py` | 获取矩阵（本地 `.mtx` 或 SuiteSparse 在线下载） |
| `matrix_analyzer.py` | 符号分析：排序→消去树→L 模式→超节点划分→nu 估计 |
| `workload_builder.py` | 生成任务图：符号分析 + 数值分解（tstrf/gessm）+ 三角求解（ssssm） |
| `json_writer.py` | 写出 JSON |
| `docs/README_demo10.md` | 10×10 示例矩阵的完整推导（含数值详解） |
| `docs/README_ALL_FLOPs_Bytes.md` | 算子 FLOPs/Bytes 建模说明 |
| `tools/` | 验证与算法演示脚本 |
| `matrix_data/` | 本地矩阵数据 |
| `fitting_tools/` | 算子画像和拟合工具 |
| `vtune/` | CPU TRSV/VTune 相关实验脚本 |
| `workload_supernodal_pdf.json` | 默认样例输出文件 |

`matrix_data/`、profiling 原始输出和本机编译产物不随源码仓库分发。使用公开矩阵时，
可以通过 `--matrix` 在线下载，或把 `.mtx` 文件放到本地 `matrix_data/` 后用 `--local-mtx` 指定。

---

## 运行方式

### Supernodal Cholesky

```bash
python main.py --local-mtx matrix_data/demo10/demo10.mtx --ordering amd --dtype fp64
```

**常用参数：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--local-mtx` | — | 本地 `.mtx` 路径 |
| `--matrix` | — | SuiteSparse 矩阵名（在线下载） |
| `--ordering` | `amd` | 排序方式（fallback: SuperLU COLAMD → RCM） |
| `--max-supernode-size` | 64 | 超节点最大宽度 |
| `--overlap-threshold` | 0.85 | L 列模式 Jaccard 合并阈值 |
| `--max-fill-ratio` | 0.35 | 松弛合并允许的最大填充率 |
| `--dtype` | `fp64` | 数据类型（fp32/fp64/int32/int64） |
| `--out` | `workload_supernodal_pdf.json` | 输出路径 |

### Conjugate Gradient (CG)

```bash
python generators/cg_workload_generator.py \
  --local-mtx matrix_data/bcsstk08/bcsstk08.mtx \
  --iterations 300 \
  --dtype fp32 \
  --out ../examples/cg_solver_bcsstk08_n1074_nnz12960_iter300.json
```

输出格式与 supernodal 一致：顶层 `name` / `tensors` / `tasks`，依赖由 tensor `producer` 推导，task 写 `bytes_read` / `bytes_written`。

默认 `dot` 是全局归约 task；分布式 partial dot 可加 `--dot-collective allreduce`。

计算量和访存量模型：

| task | FLOPs | bytes_read | bytes_written |
|---|---:|---:|---:|
| `spmv` | `2 * nnz(A)` | `CSR(A) + nnz(A) * sizeof(dtype)` | `n * sizeof(dtype)` |
| `sptrsv` | `2 * nnz(L)` | `CSR(L) + n * sizeof(dtype)` | `n * sizeof(dtype)` |
| `dot_pdotq` | `2 * n` | `2 * n * sizeof(dtype)` | `sizeof(dtype)` |
| `dot_rnorm` | `2 * n` | `n * sizeof(dtype)` | `sizeof(dtype)` |
| `scal` | `1` | `2 * sizeof(dtype)` | `sizeof(dtype)` |
| `axpy` | `2 * n` | `2 * n * sizeof(dtype) + sizeof(dtype)` | `n * sizeof(dtype)` |

使用 `--matrix` 下载矩阵需要 `ssgetpy`；本地 `.mtx` 不需要。没有 `scipy` 时会直接解析 Matrix Market。

---

## 整体流程

```
输入 .mtx
  │
  ▼
[matrix_provider]  读取/下载矩阵
  │
  ▼
[matrix_analyzer]  符号分析
  ├─ AMD/COLAMD 排序 → 重排矩阵 A_perm
  ├─ 消去树构建（路径压缩）
  ├─ 行子树算法 → 精确 L 列模式
  ├─ 超节点划分（Jaccard 合并 + 松弛合并）
  └─ 超节点树 + nu/nnz 估计
  │
  ▼
[workload_builder]  生成任务图
  ├─ 阶段一：符号分析（7 tasks）
  ├─ 阶段二：数值分解 — tstrf + gessm（每超节点）
  └─ 阶段三：三角求解 — ssssm（1 task）
  │
  ▼
[json_writer]  输出 workload JSON
```

---

## CHOLMOD Left-looking 算法概述

### 超节点树

消去树的最后一列映射为超节点父子关系：`sn_children[parent] = [child_0, child_1, ...]`。

超节点按自然顺序（列索引递增）处理。数值更新依赖不只看直接父子边，而是由真实 L 列模式中的 `sn_panel_rows` 决定。

### 数值分解算子

对每个超节点 $k$（列 $S_k$，宽 $n_s$，面板高 $n_u$）：

| 算子 | 公式 | 说明 |
|---|---|---|
| **tstrf** | $F = \frac{n_s^3}{3} + n_u \cdot n_s^2$ | POTRF（对角块）+ TRSM（面板）合并 |
| **gessm** | $F = o_{s\to k} \cdot n_{s,s}^2$ | 源超节点面板对目标超节点前沿作用域的 SYRK 外积更新 |

- **tstrf** 的输入：逻辑前沿 `front_k`、符号模式 `lnz_pattern`，以及所有实际更新到该超节点前沿的 `update_{s}_to_{k}`。`front_k` 按 row-wise 访问建模，对齐 tstrf 的 strided 算子画像。
- **gessm** 的输入：源超节点面板切片 `panel_slice_{s}_to_{k}`；输出：`update_{s}_to_{k}`（$o_{s\to k} \times o_{s\to k}$ 稠密对称）
- $o_{s\to k}=|\text{panel\_rows}(s)\cap(S_k\cup\text{panel\_rows}(k))|$。这里使用 L 模式中的真实稀疏面板行，不假设面板行连续；$o_{s\to k}=0$ 时跳过该 gessm。

### 三角求解算子

| 算子 | 公式 | 说明 |
|---|---|---|
| **ssssm** | $F = 4 \cdot nnz(L)$ | 稀疏前代 + 回代合并（乘加分计，×2） |

---

## 任务汇总

### 阶段一：符号分析（7 tasks）

| ID | 任务 | 算子 | FLOPs |
|---|---|---|---|
| 0 | sn_ordering | `amd_ordering` | $2\cdot nnz + n\log_2 n$ |
| 1 | sn_etree | `elimination_tree` | $nnz + n$ |
| 2 | sn_postorder | `postorder` | $n$ |
| 3 | sn_colcount | `column_counts` | $nnz + n$ |
| 4 | sn_partition | `supernode_partition` | $nnz$ |
| 5 | sn_relax | `supernode_relax` | $nnz$ |
| 6 | sn_lnz_pattern | `symbolic_pattern` | $nnz(L)$ |

### 阶段二：数值分解（每超节点）

| 算子 | FLOPs | 输入 | 输出 |
|---|---|---|---|
| `tstrf` | $n_s^3/3 + n_u n_s^2$ | `front_k` + `lnz_pattern` + $\sum$ update | `l11_k` + `panel_k` + `panel_slice_{k}_to_*` |
| `gessm` | $o_{s\to k} \cdot n_{s,s}^2$ | `panel_slice_{s}_to_{k}` | `update_{s}_to_{k}` |

### 阶段三：三角求解（1 task）

| 算子 | FLOPs | 输入 | 输出 |
|---|---|---|---|
| `ssssm` | $4 \cdot nnz(L)$ | `l11_*` + `panel_*` + `b` | `y`, `x` |

---

## 输出 JSON 结构

```json
{
  "name": "supernodal_<matrix>",
  "tensors": [
    { "id": "A",     "size_bytes": 408, "producer": null, "dtype": "fp64", "shape": [10,10] },
    { "id": "front_0","size_bytes": 128, "producer": null, "dtype": "fp64", "shape": [4,4] },
    { "id": "panel_0","size_bytes": 24,  "producer": 7,    "dtype": "fp64", "shape": [3,1] },
    { "id": "panel_slice_0_to_2", "size_bytes": 16, "producer": 7, "dtype": "fp64", "shape": [2,1] },
    { "id": "update_0_to_2", "size_bytes": 32, "producer": 13, "dtype": "fp64", "shape": [2,2] }
  ],
  "tasks": [
    { "id": 7,  "name": "sn_tstrf_0",       "op": "tstrf",  "compute_flops": 3.33, "ns": 1, "nu": 3 },
    { "id": 13, "name": "sn_gessm_0_to_2",  "op": "gessm",  "compute_flops": 2.0,  "overlap_rows": 2 },
    { "id": 19, "name": "sn_ssssm",         "op": "ssssm",  "compute_flops": 128.0 }
  ]
}
```

- 依赖关系由 tensor 的 `producer` 字段推导（不显式使用 edges）。
- 每个 task 包含 `bytes_read` / `bytes_written` 用于通信建模。

---

## 任务依赖关系图

```
阶段一（符号分析）：
  A ──→ [0] amd_ordering ──→ perm
  A + perm ──→ [1] etree ──→ elim_tree
  elim_tree ──→ [2] postorder ──→ postorder
  A + elim_tree + postorder ──→ [3] colcount ──→ colcounts
  A + elim_tree + colcounts ──→ [4] partition ──→ basic_supernodes
  A + basic_supernodes ──→ [5] relax ──→ relaxed_supernodes
  relaxed_supernodes ──→ [6] lnz_pattern ──→ lnz_pattern

阶段二（数值分解 — CHOLMOD Left-looking）：
  lnz_pattern + front_0 ──→ tstrf_0 ──→ panel_0 + panel_slice_0_to_*
  lnz_pattern + front_1 + update_0_to_1 ──→ tstrf_1 ──→ panel_1 + panel_slice_1_to_*
  panel_slice_0_to_2 ──→ gessm_0_to_2 ──→ update_0_to_2 ──→ tstrf_2
  panel_slice_1_to_3 ──→ gessm_1_to_3 ──→ update_1_to_3 ──→ tstrf_3
  panel_slice_3_to_4 ──→ gessm_3_to_4 ──→ update_3_to_4 ──→ tstrf_4

阶段三（三角求解）：
  l11_* + panel_* + b ──→ ssssm ──→ y, x
```

---

## 核心数学符号

| 符号 | 来源 | 含义 |
|---|---|---|
| $n$ | `meta.rows` | 矩阵维度 |
| $nnz$ | `meta.nnz` | A 的非零元数 |
| $nnz(L)$ | `meta.nnz_l` | L 因子的非零元数 |
| $N_{sn}$ | `len(meta.supernodes)` | 超节点总数 |
| $n_s$ | `len(supernodes[k])` | 超节点 k 的列数 |
| $n_u$ | `meta.sn_nu[k]` | 超节点 k 的面板高度 |
| `panel_rows(k)` | `meta.sn_panel_rows[k]` | 超节点 k 在 L 模式中的真实稀疏面板行 |
| $o_{s\to k}$ | 运行时计算 | 源面板行 ∩ 目标前沿作用域的重叠行数 |
| $S$ | `DTYPE_BYTES[dtype]` | 浮点数字节数（fp64=8） |
| $I$ | 4 | 索引字节数（int32） |

---

## 张量定义

### 全局张量

| ID | 形状 | 类型 | 说明 |
|---|---|---|---|
| `A` | $[n,n]$ | fp64, sparse_csr | 输入 SPD 矩阵 |
| `b` | $[n]$ | fp64 | 右端向量 |
| `perm` | $[n]$ | int32 | 排序置换 |
| `elim_tree` | $[n]$ | int32 | 消去树 parent 数组 |
| `postorder` | $[n]$ | int32 | 后序遍历 |
| `colcounts` | $[n]$ | int32 | 列非零计数 |
| `basic_supernodes` | $[N_{sn}]$ | int64 | 基础超节点 |
| `relaxed_supernodes` | $[N_{sn}]$ | int64 | 松弛合并后超节点 |
| `lnz_pattern` | $[nnz(L)]$ | int32 | L 符号模式 |
| `y`, `x` | $[n]$ | fp64 | 三角求解中间/最终结果 |

### 每超节点 $k$ 的张量

| ID | 形状 | 生产者 | 说明 |
|---|---|---|---|
| `front_k` | $[n_s+n_u, n_s+n_u]$ | null | tstrf 的逻辑前沿输入，row-wise 访问 |
| `l11_k` | $[n_s, n_s]$ | tstrf_k | 对角块 |
| `panel_k` | $[n_u, n_s]$ | tstrf_k | 面板，是 ssssm 的 L 因子输入之一 |
| `panel_slice_{s}_to_{k}` | $[o, n_s]$ | tstrf_s | 源面板中实际传给 gessm 的重叠切片 |
| `update_{s}_to_{k}` | $[o, o]$ | gessm_s→k | 源超节点面板外积更新 |

---

## 运行示例

```powershell
# demo10（10×10 小矩阵，含详细推导见 docs/README_demo10.md）
python main.py --local-mtx matrix_data/demo10/demo10.mtx --ordering amd --dtype fp64

# 1138_bus（电力系统矩阵，n=1138）
python main.py --local-mtx matrix_data/1138_bus/1138_bus.mtx --ordering amd --dtype fp64

# bcsstk08（结构工程矩阵，n=1074）
python main.py --local-mtx matrix_data/bcsstk08/bcsstk08.mtx --ordering amd --dtype fp64
```
