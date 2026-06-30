import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from matrix_analyzer import MatrixMeta

"""
Workload Builder — 基于 CHOLMOD (Algorithm 887, ACM TOMS 2008) Left-looking 超节点法

建模依据：
  符号分析阶段 → CHOLMOD §2（AMD排序 → 消去树 → 后序遍历 → 行子树算法 → 超节点划分 → 松弛合并）
  数值分解阶段 → CHOLMOD Left-looking（tstrf: POTRF+TRSM / gessm: 源超节点面板SYRK更新）
  三角求解阶段 → ssssm (稀疏前代+回代)

与之前多波前风格的关键区别：
  - 不再显式构造多波前 assemble 链；每个 tstrf 使用逻辑 front 张量表达前沿读流量
  - 源超节点贡献通过 gessm (SYRK) 自动流向后续超节点前沿
  - POTRF + TRSM 合并为 tstrf 算子
  - 前代+回代合并为 ssssm 算子
  - FLOPs / Bytes 公式遵循 docs/README_ALL_FLOPs_Bytes.md
"""

DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
    "int32": 4,
    "int64": 8,
}


def build_supernodal_workload(meta: MatrixMeta, dtype: str) -> Dict[str, Any]:
    n = meta.rows
    nnz = meta.nnz
    nnz_l = meta.nnz_l
    word_bytes = DTYPE_BYTES[dtype]

    tasks: List[Dict[str, Any]] = []
    tensors: List[Dict[str, Any]] = []

    def add_tensor(tid: str, size_bytes: int, producer: Optional[int], **kwargs: Any) -> None:
        tensors.append({
            "id": tid,
            "size_bytes": int(max(0, size_bytes)),
            "producer": producer,
            **kwargs,
        })

    def add_task(task: Dict[str, Any]) -> None:
        tasks.append(task)

    # ═══════════════════════════════════════════
    # 输入张量
    # ═══════════════════════════════════════════
    matrix_size_bytes = max(1, nnz) * (word_bytes + 4)     # value + col_idx per nonzero
    vector_size_bytes = n * word_bytes

    add_tensor("A", matrix_size_bytes, None,
               dtype=dtype, shape=[n, n], num_elements=nnz, access_pattern="sparse_csr")
    add_tensor("b", vector_size_bytes, None,
               dtype=dtype, shape=[n], num_elements=n, access_pattern="dense")

    task_id = 0

    # ═══════════════════════════════════════════
    # 阶段一：符号分析（论文 §3.1）
    # Bytes 建模：读所有输入张量，写所有输出张量
    #   A (CSR/CSC): (n+1)·I + nnz·(I+S)
    #   整数向量: n·I
    #   int64 向量: N_sn·8
    # ═══════════════════════════════════════════

    I = DTYPE_BYTES["int32"]  # 索引字节数
    S = word_bytes            # 浮点数字节数
    A_bytes = float((n + 1) * I + nnz * (I + S))  # 稀疏矩阵 A 的存储大小
    int_vec_bytes = float(n * I)                    # int32 向量
    sn_vec_bytes = float(max(1, len(meta.supernodes)) * 8)  # int64 超节点向量

    # (1) AMD 填充减少排序 — §3.1.1
    # 读 A, 写 perm
    add_task({
        "id": task_id, "name": "sn_ordering", "op": "amd_ordering",
        "type": "compute", "subtype": "symbolic",
        "compute_flops": float(nnz * 2.0 + n * math.log2(max(2.0, float(n)))),
        "bytes_read": A_bytes,
        "bytes_written": int_vec_bytes,
        "inputs": [{"tensor": "A", "role": "matrix"}],
        "outputs": ["perm"],
    })
    add_tensor("perm", n * 4, task_id, dtype="int32", shape=[n], num_elements=n, access_pattern="dense")
    task_id += 1

    # (2) 消去树构建 — §3.1.2 (前半)
    # 读 A + perm, 写 elim_tree
    add_task({
        "id": task_id, "name": "sn_etree", "op": "elimination_tree",
        "type": "compute", "subtype": "symbolic",
        "compute_flops": float(nnz + n),
        "bytes_read": A_bytes + int_vec_bytes,
        "bytes_written": int_vec_bytes,
        "inputs": [{"tensor": "A", "role": "matrix"}, {"tensor": "perm", "role": "permutation"}],
        "outputs": ["elim_tree"],
    })
    add_tensor("elim_tree", n * 4, task_id, dtype="int32", shape=[n], num_elements=n, access_pattern="dense")
    task_id += 1

    # (3) 后序遍历 — §3.1.2 (后半)
    # 读 elim_tree, 写 postorder
    add_task({
        "id": task_id, "name": "sn_postorder", "op": "postorder",
        "type": "compute", "subtype": "symbolic",
        "compute_flops": float(n),
        "bytes_read": int_vec_bytes,
        "bytes_written": int_vec_bytes,
        "inputs": [{"tensor": "elim_tree", "role": "tree"}],
        "outputs": ["postorder"],
    })
    add_tensor("postorder", n * 4, task_id, dtype="int32", shape=[n], num_elements=n, access_pattern="dense")
    task_id += 1

    # (4) 列非零计数 — §3.1.3
    # 读 A + elim_tree + postorder, 写 colcounts
    add_task({
        "id": task_id, "name": "sn_colcount", "op": "column_counts",
        "type": "compute", "subtype": "symbolic",
        "compute_flops": float(nnz + n),
        "bytes_read": A_bytes + int_vec_bytes + int_vec_bytes,
        "bytes_written": int_vec_bytes,
        "inputs": [
            {"tensor": "A", "role": "matrix"},
            {"tensor": "elim_tree", "role": "tree"},
            {"tensor": "postorder", "role": "postorder"},
        ],
        "outputs": ["colcounts"],
    })
    add_tensor("colcounts", n * 4, task_id, dtype="int32", shape=[n], num_elements=n, access_pattern="dense")
    task_id += 1

    # (5) 基础超节点划分 — §3.2.1
    # 读 A + elim_tree + colcounts, 写 basic_supernodes
    add_task({
        "id": task_id, "name": "sn_partition", "op": "supernode_partition",
        "type": "compute", "subtype": "symbolic",
        "compute_flops": float(nnz),
        "bytes_read": A_bytes + int_vec_bytes + int_vec_bytes,
        "bytes_written": sn_vec_bytes,
        "inputs": [
            {"tensor": "A", "role": "column_patterns"},
            {"tensor": "elim_tree", "role": "tree"},
            {"tensor": "colcounts", "role": "column_counts"},
        ],
        "outputs": ["basic_supernodes"],
    })
    add_tensor("basic_supernodes", max(1, len(meta.supernodes)) * 8, task_id,
               dtype="int64", shape=[len(meta.supernodes)], num_elements=len(meta.supernodes),
               access_pattern="dense")
    task_id += 1

    # (6) 超节点松弛合并 — §3.2.2
    # 读 A + basic_supernodes, 写 relaxed_supernodes
    add_task({
        "id": task_id, "name": "sn_relax", "op": "supernode_relax",
        "type": "compute", "subtype": "symbolic",
        "compute_flops": float(nnz),
        "bytes_read": A_bytes + sn_vec_bytes,
        "bytes_written": sn_vec_bytes,
        "inputs": [
            {"tensor": "A", "role": "column_patterns"},
            {"tensor": "basic_supernodes", "role": "supernodes"},
        ],
        "outputs": ["relaxed_supernodes"],
    })
    add_tensor("relaxed_supernodes", max(1, len(meta.supernodes)) * 8, task_id,
               dtype="int64", shape=[len(meta.supernodes)], num_elements=len(meta.supernodes),
               access_pattern="dense")
    task_id += 1

    # L 符号模式 — §3.1.4
    # 读 relaxed_supernodes, 写 lnz_pattern
    lnz_bytes = float(max(1, nnz_l) * I)
    add_task({
        "id": task_id, "name": "sn_lnz_pattern", "op": "symbolic_pattern",
        "type": "compute", "subtype": "symbolic",
        "compute_flops": float(nnz_l),
        "bytes_read": sn_vec_bytes,
        "bytes_written": lnz_bytes,
        "inputs": [{"tensor": "relaxed_supernodes", "role": "supernodes"}],
        "outputs": ["lnz_pattern"],
    })
    add_tensor("lnz_pattern", max(1, nnz_l) * 4, task_id, dtype="int32",
               shape=[nnz_l], num_elements=nnz_l, access_pattern="dense")
    task_id += 1

    # ═══════════════════════════════════════════
    # 阶段二：数值分解 — CHOLMOD Left-looking 超节点法
    #
    # 每个超节点执行：
    #   tstrf: POTRF(ns×ns 对角块) + TRSM(nu×ns 面板)  合并算子
    #   FLOPs = ns³/3 + nu·ns²
    #   Bytes = (ns+nu)²·S (读前沿) + ns²·S (写L11) + nu·ns·S (写面板)
    #
    # 每个 source→target 贡献执行：
    #   gessm: 源超节点面板 SYRK 外积更新
    #   overlap_rows = source panel rows ∩ target front scope(cols + panel rows)
    #   FLOPs = overlap_rows · child_ns²
    #   Bytes = (ov·ns + ov²)·S
    #
    # 依赖关系（Left-looking）：
    #   - tstrf(k) 依赖所有实际更新到 k 的 gessm 输出
    #   - gessm(source→target) 依赖 tstrf(source) 的 panel
    # ═══════════════════════════════════════════

    # S = word_bytes (浮点数字节数), I = 索引字节数（已在阶段一定义）
    # ═══════════════════════════════════════════

    # ── 创建所有 tstrf 任务（按超节点自然顺序） ──
    # 先声明面板张量，因为 gessm 任务需要引用它们
    front_tensors: Dict[int, str] = {}  # sn_id → logical frontal tensor id
    panel_tensors: Dict[int, str] = {}  # sn_id → panel tensor id
    panel_slice_tensors: Dict[Tuple[int, int], str] = {}  # (source, target) → logical slice tensor id
    factor_block_tensors: List[str] = []

    def panel_rows_for(sn_id: int) -> List[int]:
        explicit_rows = getattr(meta, "sn_panel_rows", None)
        if explicit_rows is not None and sn_id < len(explicit_rows):
            return list(explicit_rows[sn_id])
        # 兼容旧 MatrixMeta：旧版本只保存 nu，无法表达稀疏行号，只能回退到连续近似。
        last_col = meta.supernodes[sn_id][-1]
        return list(range(last_col + 1, last_col + 1 + meta.sn_nu[sn_id]))

    gessm_updates_by_target: Dict[int, List[Dict[str, Any]]] = {
        sn_id: [] for sn_id in range(len(meta.supernodes))
    }
    gessm_updates_by_source: Dict[int, List[Dict[str, Any]]] = {
        sn_id: [] for sn_id in range(len(meta.supernodes))
    }

    # Fast equivalent of:
    #   panel_rows(source) ∩ (target_cols ∪ panel_rows(target))
    # The previous source×target scan is quadratic in the number of supernodes.
    col_to_sn = [-1] * n
    for sn_id, cols in enumerate(meta.supernodes):
        for col in cols:
            if 0 <= col < n:
                col_to_sn[col] = sn_id

    panel_row_to_targets: Dict[int, List[int]] = defaultdict(list)
    for target_id in range(len(meta.supernodes)):
        for row in panel_rows_for(target_id):
            panel_row_to_targets[row].append(target_id)

    for source_id, source_cols in enumerate(meta.supernodes):
        target_to_rows: Dict[int, Set[int]] = defaultdict(set)
        for row in panel_rows_for(source_id):
            if 0 <= row < len(col_to_sn):
                target_id = col_to_sn[row]
                if target_id > source_id:
                    target_to_rows[target_id].add(row)
            for target_id in panel_row_to_targets.get(row, []):
                if target_id > source_id:
                    target_to_rows[target_id].add(row)

        for target_id in sorted(target_to_rows):
            overlap_row_ids = sorted(target_to_rows[target_id])
            if not overlap_row_ids:
                continue
            gessm_updates_by_target[target_id].append({
                "source": source_id,
                "target": target_id,
                "overlap_rows": len(overlap_row_ids),
                "overlap_row_ids": overlap_row_ids,
                "source_ns": len(source_cols),
            })
            gessm_updates_by_source[source_id].append({
                "source": source_id,
                "target": target_id,
                "overlap_rows": len(overlap_row_ids),
                "overlap_row_ids": overlap_row_ids,
                "source_ns": len(source_cols),
            })

    for sn_id, cols in enumerate(meta.supernodes):
        ns = len(cols)
        nu = meta.sn_nu[sn_id]
        front_id = f"front_{sn_id}"
        panel_id = f"panel_{sn_id}"
        front_tensors[sn_id] = front_id
        panel_tensors[sn_id] = panel_id
        # 逻辑前沿张量：表达 tstrf 读取的稠密前沿 F，不再让 tstrf 直接把整块稀疏 A
        # 作为输入，从而保持访问模式与 tstrf 算子画像一致。
        front_numel = (ns + nu) * (ns + nu)
        add_tensor(front_id, front_numel * S, None,
                   dtype=dtype, shape=[ns + nu, ns + nu], num_elements=front_numel,
                   access_pattern="dense")
        # 预先声明面板张量（由 tstrf 产出）
        add_tensor(panel_id, nu * ns * S, None,
                   dtype=dtype, shape=[nu, ns], num_elements=nu * ns, access_pattern="dense")

    for sn_id, cols in enumerate(meta.supernodes):
        ns = len(cols)
        nu = meta.sn_nu[sn_id]

        # 收集所有实际更新到当前超节点的 gessm 贡献张量 ID
        gessm_inputs: List[Dict[str, str]] = []
        for update in gessm_updates_by_target[sn_id]:
            source = update["source"]
            update_id = f"update_{source}_to_{sn_id}"
            gessm_inputs.append({"tensor": update_id, "role": "child_update", "access": "dense"})

        # ── tstrf: POTRF + TRSM 合并 ──
        # compute_FLOPs = ns³/3 + nu·ns²  (来源: docs/README_ALL_FLOPs_Bytes.md)
        tstrf_flops = float(ns ** 3 / 3.0 + nu * ns * ns)

        # Bytes: (ns+nu)²·S 读 + ns²·S 写(L11) + nu·ns·S 写(panel)
        tstrf_bytes_read = float((ns + nu) * (ns + nu) * S)
        tstrf_bytes_write = float((ns * ns + nu * ns) * S)

        l11_id = f"l11_{sn_id}"
        front_id = front_tensors[sn_id]
        panel_id = panel_tensors[sn_id]
        slice_outputs: List[str] = []
        for update in gessm_updates_by_source[sn_id]:
            target = update["target"]
            slice_id = f"panel_slice_{sn_id}_to_{target}"
            panel_slice_tensors[(sn_id, target)] = slice_id
            slice_outputs.append(slice_id)

        # tstrf 输入：逻辑前沿 F + L 符号模式 + 所有 source→target gessm 贡献。
        # front 使用 row_wise override，使 mapper 侧访问画像落到 strided，而不是被全局稀疏 A 误判为 sparse。
        tstrf_inputs = [
            {"tensor": front_id, "role": "frontal_matrix", "access": "row_wise"},
            {"tensor": "lnz_pattern", "role": "symbolic_pattern", "access": "dense"},
        ] + gessm_inputs

        add_task({
            "id": task_id,
            "name": f"sn_tstrf_{sn_id}",
            "op": "tstrf",
            "type": "compute", "subtype": "left_looking_factor",
            "compute_flops": tstrf_flops,
            "bytes_read": tstrf_bytes_read,
            "bytes_written": tstrf_bytes_write,
            "ns": ns, "nu": nu,
            "inputs": tstrf_inputs,
            "outputs": [l11_id, panel_id] + slice_outputs,
        })
        current_task_id = task_id
        task_id += 1

        # L11 对角块张量
        add_tensor(l11_id, ns * ns * S, current_task_id,
                   dtype=dtype, shape=[ns, ns], num_elements=ns * ns, access_pattern="dense")
        factor_block_tensors.append(l11_id)
        # 更新面板张量的 producer
        # (面板已在上面预声明，这里更新其 producer 为当前 tstrf 任务)
        for t in tensors:
            if t["id"] == panel_id:
                t["producer"] = current_task_id
                break
        if nu * ns > 0:
            factor_block_tensors.append(panel_id)
        # gessm 只读取源 panel 与目标前沿作用域相交的切片。显式建模 slice tensor，
        # 让通信边和 input_tensor_sizes 使用 ov×ns，而不是整块 panel。
        for update in gessm_updates_by_source[sn_id]:
            target = update["target"]
            overlap_rows = update["overlap_rows"]
            slice_id = panel_slice_tensors[(sn_id, target)]
            add_tensor(slice_id, overlap_rows * ns * S, current_task_id,
                       dtype=dtype, shape=[overlap_rows, ns], num_elements=overlap_rows * ns,
                       access_pattern="dense")

    # ── 创建所有 gessm 任务（source→target 的 SYRK 更新） ──
    for target_id in range(len(meta.supernodes)):
        for update in gessm_updates_by_target[target_id]:
            source = update["source"]
            source_ns = update["source_ns"]
            overlap_rows = update["overlap_rows"]
            overlap_row_ids = update["overlap_row_ids"]

            source_panel_id = panel_slice_tensors[(source, target_id)]
            update_id = f"update_{source}_to_{target_id}"

            # gessm FLOPs = overlap_rows · child_ns²  (来源: docs/README_ALL_FLOPs_Bytes.md)
            gessm_flops = float(overlap_rows * source_ns * source_ns)

            # Bytes: ov·ns·S 读(子面板) + ov²·S 写(更新矩阵)
            gessm_bytes_read = float(overlap_rows * source_ns * S)
            gessm_bytes_write = float(overlap_rows * overlap_rows * S)

            add_task({
                "id": task_id,
                "name": f"sn_gessm_{source}_to_{target_id}",
                "op": "gessm",
                "type": "compute", "subtype": "child_panel_update",
                "compute_flops": gessm_flops,
                "bytes_read": gessm_bytes_read,
                "bytes_written": gessm_bytes_write,
                "overlap_rows": overlap_rows,
                "overlap_row_ids": overlap_row_ids,
                "ns_child": source_ns,
                "source_sn": source,
                "target_sn": target_id,
                "inputs": [{"tensor": source_panel_id, "role": "child_panel_slice", "access": "dense"}],
                "outputs": [update_id],
            })
            task_id += 1

            # 更新矩阵张量 (ov × ov 稠密对称)
            add_tensor(update_id, overlap_rows * overlap_rows * S, task_id - 1,
                       dtype=dtype, shape=[overlap_rows, overlap_rows],
                       num_elements=overlap_rows * overlap_rows, access_pattern="dense")

    # ═══════════════════════════════════════════
    # 阶段三：三角求解 — ssssm (稀疏前代+回代)
    #
    # ssssm: 解 Lx = b 其中 L 为稀疏下三角 (CSC)
    # compute_FLOPs = 4·nnz(L)   (来源: docs/README_ALL_FLOPs_Bytes.md)
    # Bytes = 2[(n+1)·I + nnz·(I+S)] + 3n·S
    # ═══════════════════════════════════════════

    ssssm_flops = float(4 * nnz_l)
    # L 结构读取两次（前代+回代）：col_ptr + row_idx + values
    ssssm_bytes_read = float(
        2 * ((n + 1) * I + nnz_l * (I + S))  # L 结构读两次
        + n * S                                # b 读
    )
    ssssm_bytes_write = float(
        2 * n * S                              # y 写 + x 写
    )

    ssssm_inputs = [
        {"tensor": tensor_id, "role": "factor_block", "access": "sparse_csc"}
        for tensor_id in factor_block_tensors
    ]
    ssssm_inputs.append({"tensor": "b", "role": "rhs", "access": "dense"})

    add_task({
        "id": task_id, "name": "sn_ssssm", "op": "ssssm",
        "type": "compute", "subtype": "sparse_triangular_solve",
        "compute_flops": ssssm_flops,
        "bytes_read": ssssm_bytes_read,
        "bytes_written": ssssm_bytes_write,
        "inputs": ssssm_inputs,
        "outputs": ["y", "x"],
    })
    add_tensor("y", vector_size_bytes, task_id, dtype=dtype,
               shape=[n], num_elements=n, access_pattern="dense")
    add_tensor("x", vector_size_bytes, task_id, dtype=dtype,
               shape=[n], num_elements=n, access_pattern="dense")

    return {
        "name": f"supernodal_{meta.name}",
        "tensors": tensors,
        "tasks": tasks,
    }
