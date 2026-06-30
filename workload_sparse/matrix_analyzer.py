from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.io import mmread
    from scipy.sparse import csc_matrix, issparse, triu, tril
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    HAS_SCIPY = True
except ImportError:
    mmread = None
    csc_matrix = None
    issparse = None
    triu = None
    tril = None
    reverse_cuthill_mckee = None
    HAS_SCIPY = False

# 可选依赖：pip install scikit-sparse
try:
    from sksparse.cholmod import cholesky
    HAS_CHOLMOD = True
except ImportError:
    cholesky = None
    HAS_CHOLMOD = False

# 备选 AMD：通过 SuperLU 获取列近似最小度排序（无需 scikit-sparse）
try:
    from scipy.sparse.linalg import splu
    HAS_SPLU = True
except ImportError:
    splu = None
    HAS_SPLU = False


def _amd_via_superlu(A: csc_matrix) -> np.ndarray:
    """通过 SuperLU 的 COLAMD (COLumn Approximate Minimum Degree) 获取排序。
    这是 AMD 的近亲，效果接近但不需要 scikit-sparse。"""
    n = A.shape[0]
    # 构造 n×n 稠密单位矩阵的稀疏版本来触发列排序（仅符号分解）
    # 直接对 A 做符号 LU，提取列置换
    try:
        lu = splu(A, options=dict(
            ColPerm='COLAMD',          # 列近似最小度
            SymmetricMode=True,        # 对称模式
            DiagPivotThresh=0.0,       # 不对角元做特殊处理
        ))
        perm = lu.perm_c.astype(np.int64)
        return perm
    except Exception:
        # 如果 SuperLU 失败，回退到自然序
        return np.arange(n, dtype=np.int64)


@dataclass
class MatrixMeta:
    # 分析结果的元信息容器 (CHOLMOD Left-looking 风格)
    name: str
    rows: int
    cols: int
    nnz: int
    nnz_l: int
    supernodes: List[List[int]]
    sn_parent: List[int]
    sn_children: Dict[int, List[int]]
    sn_nu: List[int]            # 每个超节点的面板行数（L 列模式确定）
    sn_nnz: List[int]           # 每个超节点在 L 中的非零元数（对角+面板）
    sn_panel_rows: List[List[int]]  # 每个超节点的真实面板行号（稀疏，不假设连续）
    ordering: str
    symbolic_source: str


def load_csc_spd(path: str) -> csc_matrix:
    # 读取矩阵并规范为 CSC 稀疏格式
    A = mmread(path)
    if not issparse(A):
        A = csc_matrix(A)
    else:
        A = A.tocsc()
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square for Cholesky.")
    # 取上三角并构造对称矩阵（按对角线避免重复）
    U = triu(A, format="csc")
    A_sym = (U + U.T) - csc_matrix(
        (U.diagonal(), (np.arange(U.shape[0]), np.arange(U.shape[0]))),
        shape=U.shape,
    )
    return A_sym.tocsc()


def factor_pattern_with_fallback(
    A: csc_matrix,
    ordering: str,
) -> Tuple[np.ndarray, csc_matrix, csc_matrix, str]:
    """获取排序与排序后的矩阵。优先级：CHOLMOD > SuperLU COLAMD > RCM"""
    # 1) 优先用 CHOLMOD 获取 AMD 排序 + L 因子
    if HAS_CHOLMOD:
        factor = cholesky(A, ordering_method=ordering)
        perm = factor.P().astype(np.int64)
        A_perm = A[perm, :][:, perm].tocsc()
        L = factor.L().tocsc()
        return perm, A_perm, L, "cholmod"

    # 2) CHOLMOD 不可用 → 尝试 SuperLU COLAMD（近似 AMD）
    if HAS_SPLU and ordering in ("amd", "colamd", "metis"):
        perm = _amd_via_superlu(A)
        A_perm = A[perm, :][:, perm].tocsc()
        L = tril(A_perm, format="csc")
        return perm, A_perm, L, "superlu_colamd"

    # 3) 最终回退：RCM
    perm = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    A_perm = A[perm, :][:, perm].tocsc()
    L = tril(A_perm, format="csc")
    return perm, A_perm, L, "rcm_tril_proxy"


def estimate_etree_from_upper(A_csc: csc_matrix) -> List[int]:
    # 基于上三角非零结构估计消去树
    n = A_csc.shape[0]
    parent = [-1] * n
    root = [n] * n
    A_csr = A_csc.tocsr()
    for i in range(n):
        root[i] = n
        start, end = A_csr.indptr[i], A_csr.indptr[i + 1]
        for idx in range(start, end):
            j = int(A_csr.indices[idx])
            if j >= i:
                continue
            while root[j] < i:
                next_j = root[j]
                root[j] = i
                j = next_j
            root[j] = i
            parent[j] = i
    return parent


def build_basic_supernodes_from_L(
    L_cols: List[set],
    parent: List[int],
    max_width: int,
    overlap_threshold: float,
) -> List[List[int]]:
    """基于 L 因子列模式与消去树相邻关系的超节点合并（CHOLMOD 标准做法）。

    CHOLMOD 基于 L 的符号模式（而非 A 的原始模式）划分超节点。
    条件：parent[j-1]==j 且 Jaccard(L_cols[j-1], L_cols[j]) ≥ threshold。
    """
    n = len(L_cols)
    if n == 0:
        return []

    supernodes: List[List[int]] = []
    current = [0]
    for j in range(1, n):
        # 仅当 j-1 的父节点为 j 时才考虑合并
        cond_tree = parent[j - 1] == j
        if not cond_tree:
            supernodes.append(current)
            current = [j]
            continue
        if len(current) >= max_width:
            supernodes.append(current)
            current = [j]
            continue
        # 通过 L 列模式的重叠度判断是否合并
        pj = L_cols[j - 1]
        pk = L_cols[j]
        if not pj or not pk:
            supernodes.append(current)
            current = [j]
            continue
        inter = len(pj & pk)
        jaccard = inter / max(len(pj), len(pk))
        if jaccard >= overlap_threshold:
            current.append(j)
        else:
            supernodes.append(current)
            current = [j]

    if current:
        supernodes.append(current)
    return supernodes


def relax_supernodes_from_L(
    L_cols: List[set],
    parent: List[int],
    supernodes: List[List[int]],
    max_width: int,
    max_fill_ratio: float,
) -> List[List[int]]:
    """基于 L 模式的松弛合并：沿消去树链合并，填充率 ≤ max_fill_ratio。

    CHOLMOD 的松弛合并不放松消去树拓扑约束——
    仅在 parent[block_last] == next_first 的同一条树链上合并。
    """
    if not supernodes:
        return []

    # 建立列到超节点的映射
    n = len(L_cols)
    col_to_sn = [-1] * n
    for sn_id, cols in enumerate(supernodes):
        for c in cols:
            col_to_sn[c] = sn_id

    relaxed: List[List[int]] = []
    i = 0
    while i < len(supernodes):
        block = list(supernodes[i])
        i += 1
        while i < len(supernodes):
            # 消去树约束：parent[block_last] == next_first
            block_last = block[-1]
            next_first = supernodes[i][0]
            if parent[block_last] != next_first:
                break

            candidate = block + supernodes[i]
            if len(candidate) > max_width:
                break

            # 基于 L 模式估计填充率
            row_set: set = set()
            for col in candidate:
                row_set.update(L_cols[col])
            block_rows = len(row_set)
            dense_size = len(candidate) * max(1, block_rows)
            nnz_est = sum(len(L_cols[col]) for col in candidate)
            if dense_size <= 0:
                break
            logical_zeros = dense_size - nnz_est
            fill_ratio = logical_zeros / float(dense_size)
            if fill_ratio > max_fill_ratio:
                break
            block = candidate
            i += 1
        relaxed.append(block)
    return relaxed


def build_supernode_tree(
    supernodes: List[List[int]],
    parent: List[int],
) -> Tuple[List[int], Dict[int, List[int]]]:
    # 将列级消去树映射到超节点级的父子关系
    if not supernodes:
        return [], {}

    n = max([max(sn) for sn in supernodes]) + 1
    col_to_sn = [-1] * n
    for sn_id, cols in enumerate(supernodes):
        for c in cols:
            col_to_sn[c] = sn_id

    sn_parent = [-1] * len(supernodes)
    sn_children: Dict[int, List[int]] = {i: [] for i in range(len(supernodes))}

    for sn_id, cols in enumerate(supernodes):
        last_col = cols[-1]
        p = parent[last_col]
        if p >= 0:
            p_sn = col_to_sn[p]
            sn_parent[sn_id] = p_sn
            if p_sn != sn_id:
                sn_children[p_sn].append(sn_id)

    return sn_parent, sn_children


def compute_L_pattern(A_csc: csc_matrix, parent: List[int]) -> Tuple[List[set], List[set], int]:
    """行子树算法精确计算 L 因子的非零模式（论文 §3.1.4）。

    路径引理: L[k][j] ≠ 0 (j ≤ k) ⇔ G(A) 中存在路径 j → ... → k,
              且所有中间节点编号 < k。
    行子树 T_k = {j ≤ k | L[k][j] ≠ 0} 由以下方法求得:
      对每行 k:
        种子 S_k = {k} ∪ {j < k | A[k][j] ≠ 0}
        对每个种子 j, 沿 parent 向上走直到 ≥ k,
        所有经过的节点 < k 都属于 T_k。

    Returns:
        L_rows: L_rows[k] = {c | L[k][c] ≠ 0, c ≤ k}
        L_cols: L_cols[c] = {r | L[r][c] ≠ 0, r ≥ c}
        nnz_l:  L 因子的总非零元数
    """
    n = A_csc.shape[0]
    A_csr = A_csc.tocsr()
    L_rows: List[set] = [set() for _ in range(n)]

    for k in range(n):
        # 种子: A 第 k 行的下三角非零列 (j < k)
        start, end = A_csr.indptr[k], A_csr.indptr[k + 1]
        seeds = [int(A_csr.indices[idx]) for idx in range(start, end)
                 if A_csr.indices[idx] < k]

        # 从每个种子沿消去树向上走, 收集 j < k 的节点
        for s in seeds:
            p = s
            while p >= 0 and p < k and p not in L_rows[k]:
                L_rows[k].add(p)
                p = parent[p]

        # 对角元
        L_rows[k].add(k)

    # 转列视角 (仅下三角, r ≥ c)
    L_cols: List[set] = [set() for _ in range(n)]
    for r in range(n):
        for c in L_rows[r]:
            L_cols[c].add(r)

    nnz_l = sum(len(row) for row in L_rows)
    return L_rows, L_cols, nnz_l


def estimate_sn_nu_from_L(
    L_cols: List[set],
    supernodes: List[List[int]],
) -> Tuple[List[int], List[int], List[List[int]]]:
    """使用精确 L 列模式计算每个超节点的 nu 和 sn_nnz。

    CHOLMOD Left-looking 中：
      nu = L 中行 > last_col 且列在该超节点内的非零行数
      sn_nnz = L 中该超节点列范围内的非零元总数

    Returns:
        sn_nu:         每个超节点在数值分解中的面板高度（精确）
        sn_nnz:        每个超节点在 L 因子中的非零元总数（精确）
        sn_panel_rows: 每个超节点的真实面板行号（精确，稀疏行集合）
    """
    sn_nu: List[int] = []
    sn_nnz: List[int] = []
    sn_panel_rows: List[List[int]] = []

    for cols in supernodes:
        last_col = cols[-1]

        # 精确 nu：L 中行 > last_col 且列在该超节点内的非零行数
        panel_rows: set = set()
        for c in cols:
            for r in L_cols[c]:
                if r > last_col:
                    panel_rows.add(r)
        sorted_panel_rows = sorted(panel_rows)
        nu = len(sorted_panel_rows)

        # 精确 sn_nnz：L 中该超节点列范围内的非零元总数
        sn_nnz_val = sum(len(L_cols[c]) for c in cols)

        sn_nu.append(nu)
        sn_nnz.append(sn_nnz_val)
        sn_panel_rows.append(sorted_panel_rows)

    return sn_nu, sn_nnz, sn_panel_rows


def _load_symmetric_rows_from_mtx(path: str) -> Tuple[List[set], int]:
    """Pure-Python MatrixMarket coordinate reader used when scipy is absent."""
    header = ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                header = stripped
                break
        if not header.lower().startswith("%%matrixmarket"):
            raise ValueError(f"Unsupported MatrixMarket header in {path}")

        parts = header.split()
        if len(parts) < 5 or parts[1].lower() != "matrix" or parts[2].lower() != "coordinate":
            raise ValueError("Only MatrixMarket coordinate matrices are supported.")
        field = parts[3].lower()
        symmetry = parts[4].lower()
        is_pattern = field == "pattern"
        is_symmetric = symmetry in ("symmetric", "hermitian", "skew-symmetric")

        dims = None
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            dims = stripped.split()
            break
        if dims is None or len(dims) < 3:
            raise ValueError(f"Missing MatrixMarket dimensions in {path}")

        nrows, ncols = int(dims[0]), int(dims[1])
        if nrows != ncols:
            raise ValueError("Matrix must be square for Cholesky.")

        rows: List[set] = [set() for _ in range(nrows)]
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("%"):
                continue
            cols = stripped.split()
            if len(cols) < 2:
                continue
            i = int(cols[0]) - 1
            j = int(cols[1]) - 1
            if is_pattern:
                value_is_zero = False
            elif len(cols) >= 3:
                try:
                    value_is_zero = float(cols[2]) == 0.0
                except ValueError:
                    value_is_zero = False
            else:
                value_is_zero = False
            if value_is_zero:
                continue

            if is_symmetric:
                rows[i].add(j)
                rows[j].add(i)
            else:
                # Match load_csc_spd(): keep the upper triangle and mirror it.
                if i <= j:
                    rows[i].add(j)
                    rows[j].add(i)

    return rows, sum(len(row) for row in rows)


def _reverse_cuthill_mckee_rows(rows: List[set]) -> List[int]:
    n = len(rows)
    if n == 0:
        return []

    degree = [len(row) - (1 if i in row else 0) for i, row in enumerate(rows)]
    visited = [False] * n
    order: List[int] = []

    for seed in sorted(range(n), key=lambda idx: (degree[idx], idx)):
        if visited[seed]:
            continue
        queue: deque[int] = deque([seed])
        visited[seed] = True
        while queue:
            u = queue.popleft()
            order.append(u)
            nbrs = [v for v in rows[u] if v != u and not visited[v]]
            nbrs.sort(key=lambda idx: (degree[idx], idx))
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)

    return list(reversed(order))


def _permute_rows(rows: List[set], perm: List[int]) -> List[set]:
    n = len(rows)
    inv = [0] * n
    for new_idx, old_idx in enumerate(perm):
        inv[old_idx] = new_idx

    permuted: List[set] = [set() for _ in range(n)]
    for old_i, old_cols in enumerate(rows):
        new_i = inv[old_i]
        dst = permuted[new_i]
        for old_j in old_cols:
            dst.add(inv[old_j])
    return permuted


def _estimate_etree_from_rows(rows: List[set]) -> List[int]:
    n = len(rows)
    parent = [-1] * n
    root = [n] * n
    for i in range(n):
        root[i] = n
        for j0 in rows[i]:
            if j0 >= i:
                continue
            j = int(j0)
            while root[j] < i:
                next_j = root[j]
                root[j] = i
                j = next_j
            root[j] = i
            parent[j] = i
    return parent


def _compute_L_pattern_from_rows(rows: List[set], parent: List[int]) -> Tuple[List[set], List[set], int]:
    n = len(rows)
    L_rows: List[set] = [set() for _ in range(n)]

    for k in range(n):
        for s0 in rows[k]:
            if s0 >= k:
                continue
            p = int(s0)
            while p >= 0 and p < k and p not in L_rows[k]:
                L_rows[k].add(p)
                p = parent[p]
        L_rows[k].add(k)

    L_cols: List[set] = [set() for _ in range(n)]
    for r, cols in enumerate(L_rows):
        for c in cols:
            L_cols[c].add(r)

    nnz_l = sum(len(row) for row in L_rows)
    return L_rows, L_cols, nnz_l


def matrix_analyzer_pure_python(
    name: str,
    mtx_path: str,
    ordering: str,
    max_supernode_size: int,
    overlap_threshold: float,
    max_fill_ratio: float,
) -> MatrixMeta:
    rows, nnz = _load_symmetric_rows_from_mtx(mtx_path)
    n = len(rows)

    if ordering == "natural":
        perm = list(range(n))
        symbolic_source = "pure_python_natural"
    else:
        perm = _reverse_cuthill_mckee_rows(rows)
        symbolic_source = "pure_python_rcm"
    rows = _permute_rows(rows, perm)

    parent = _estimate_etree_from_rows(rows)
    _, L_cols, nnz_l = _compute_L_pattern_from_rows(rows, parent)
    basic = build_basic_supernodes_from_L(L_cols, parent, max_supernode_size, overlap_threshold)
    relaxed = relax_supernodes_from_L(L_cols, parent, basic, max_supernode_size, max_fill_ratio)
    sn_parent, sn_children = build_supernode_tree(relaxed, parent)
    sn_nu, sn_nnz, sn_panel_rows = estimate_sn_nu_from_L(L_cols, relaxed)

    return MatrixMeta(
        name=name,
        rows=n,
        cols=n,
        nnz=nnz,
        nnz_l=nnz_l,
        supernodes=relaxed,
        sn_parent=sn_parent,
        sn_children=sn_children,
        sn_nu=sn_nu,
        sn_nnz=sn_nnz,
        sn_panel_rows=sn_panel_rows,
        ordering=ordering,
        symbolic_source=symbolic_source,
    )


def matrix_analyzer(
    name: str,
    mtx_path: str,
    ordering: str,
    max_supernode_size: int,
    overlap_threshold: float,
    max_fill_ratio: float,
) -> MatrixMeta:
    """端到端 CHOLMOD 符号分析流程：

    1. 读取矩阵 → 重排 (AMD/RCM)
    2. 构建消去树 (elimination tree)
    3. 行子树算法 → 精确 L 列模式
    4. 基于 L 列模式划分基础超节点
    5. 松弛合并（沿消去树链）
    6. 构建超节点树 + 计算 nu/nnz
    """
    if not HAS_SCIPY:
        return matrix_analyzer_pure_python(
            name=name,
            mtx_path=mtx_path,
            ordering=ordering,
            max_supernode_size=max_supernode_size,
            overlap_threshold=overlap_threshold,
            max_fill_ratio=max_fill_ratio,
        )

    A = load_csc_spd(mtx_path)
    perm, A_perm, _, symbolic_source = factor_pattern_with_fallback(A, ordering)

    # 基于重排后的矩阵估计消去树
    parent = estimate_etree_from_upper(A_perm)

    # ★ CHOLMOD 关键：先计算 L 模式，再基于 L 模式划分超节点
    L_rows, L_cols, nnz_l = compute_L_pattern(A_perm, parent)

    # 基于 L 列模式构建基础超节点
    basic = build_basic_supernodes_from_L(L_cols, parent, max_supernode_size, overlap_threshold)

    # 基于 L 列模式进行松弛合并（沿消去树链）
    relaxed = relax_supernodes_from_L(L_cols, parent, basic, max_supernode_size, max_fill_ratio)

    # 构建超节点树
    sn_parent, sn_children = build_supernode_tree(relaxed, parent)

    # 从 L 列模式计算每超节点的 nu、nnz 和真实 panel rows
    sn_nu, sn_nnz, sn_panel_rows = estimate_sn_nu_from_L(L_cols, relaxed)

    return MatrixMeta(
        name=name,
        rows=int(A_perm.shape[0]),
        cols=int(A_perm.shape[1]),
        nnz=int(A_perm.nnz),
        nnz_l=nnz_l,
        supernodes=relaxed,
        sn_parent=sn_parent,
        sn_children=sn_children,
        sn_nu=sn_nu,
        sn_nnz=sn_nnz,
        sn_panel_rows=sn_panel_rows,
        ordering=ordering,
        symbolic_source=symbolic_source,
    )
