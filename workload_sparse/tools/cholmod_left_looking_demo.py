"""
CHOLMOD Left-looking 超节点 Cholesky 分解 — Python 实现
========================================================
基于 Algorithm 887 (Chen, Davis, Hager, Rajamanickam, 2008)
在同一个 10×10 示例矩阵上演示完整流程。

核心公式 (Left-looking):
  L(*, S_k) = A(*, S_k) - sum_{d < S_k[0], L(*,d)!=0} L(*, d) * L(S_k, d)^T
  然后对超节点 S_k 的对角块 POTRF、面板 TRSM
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


# ═══════════════════════════════════════════════════════════
# 0. 构造 10×10 SPD 矩阵
# ═══════════════════════════════════════════════════════════

def build_example_matrix():
    n = 10
    A = np.zeros((n, n))
    # 对角线
    for i in range(n):
        A[i, i] = 4.0
    # 三对角边
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0
    # 跨块边
    A[0, 6] = 1.0; A[6, 0] = 1.0   # 块0→2
    A[3, 8] = 1.0; A[8, 3] = 1.0   # 块1→3
    A[6, 9] = 1.0; A[9, 6] = 1.0   # 块2→根
    return A


# ═══════════════════════════════════════════════════════════
# 1. 符号分析 (Symbolic Analysis)
# ═══════════════════════════════════════════════════════════

def symbolic_analysis(A_perm):
    """
    输入: A_perm — 排序后的 SPD 矩阵 (dense numpy)
    输出: parent, L_cols, supernodes, sn_parent, sn_children, sn_panel_rows
    """
    n = A_perm.shape[0]

    # ── 1a. 构建消去树 (Elimination Tree) ──
    # 路径压缩算法 (Liu, 1990)
    parent = [-1] * n
    root = [n] * n
    for i in range(n):
        for j in range(i):
            if A_perm[i, j] != 0:
                cur = j
                while root[cur] < i:
                    nxt = root[cur]
                    root[cur] = i
                    cur = nxt
                root[cur] = i
                parent[cur] = i
    print(f"消去树 parent: {parent}")

    # ── 1b. 行子树算法 — 确定 L 的精确非零模式 ──
    # 路径引理: L[k][j]≠0 ⇔ G(A)中存在路径 j→...→k, 中间节点<k
    L_cols = [set() for _ in range(n)]  # L_cols[c] = {r ≥ c | L[r][c] ≠ 0}
    for k in range(n):
        seeds = {k}
        for j in range(k):
            if A_perm[k, j] != 0:
                seeds.add(j)
        for s in seeds:
            p = s
            while p >= 0 and p <= k:
                L_cols[p].add(k)
                if p == k:
                    break
                p = parent[p]

    print("L 列模式:")
    for c in range(n):
        print(f"  L_col[{c}] = {sorted(L_cols[c])}")

    nnz_l = sum(len(c) for c in L_cols)
    print(f"nnz(L) = {nnz_l}")

    # ── 1c. 超节点划分 ──
    # 基础超节点: parent[j-1]==j 且 L 列模式 Jaccard ≥ 0.85
    basic_sn = []
    cur = [0]
    for j in range(1, n):
        cond_parent = (parent[j - 1] == j)
        pj, pk = L_cols[j - 1], L_cols[j]
        inter = len(pj & pk)
        jaccard = inter / max(len(pj), len(pk))
        if cond_parent and jaccard >= 0.85 and len(cur) < 64:
            cur.append(j)
        else:
            basic_sn.append(cur)
            cur = [j]
    if cur:
        basic_sn.append(cur)

    # 松弛合并: 沿树链, fill ≤ 0.35
    supernodes = []
    i = 0
    while i < len(basic_sn):
        block = list(basic_sn[i])
        i += 1
        while i < len(basic_sn):
            last = block[-1]
            first_next = basic_sn[i][0]
            if parent[last] != first_next:  # ★ 不在同一条树链上 → 不合并
                break
            candidate = block + basic_sn[i]
            if len(candidate) > 64:
                break
            all_rows = set()
            for c in candidate:
                all_rows |= L_cols[c]
            dense_sz = len(candidate) * len(all_rows)
            nnz_sz = sum(len(L_cols[c]) for c in candidate)
            if dense_sz <= 0:
                break
            fill = (dense_sz - nnz_sz) / dense_sz
            if fill > 0.35:
                break
            block = candidate
            i += 1
        supernodes.append(block)

    print(f"\n超节点 ({len(supernodes)} 个):")
    for sid, cols in enumerate(supernodes):
        print(f"  SN#{sid}: cols={cols}, ns={len(cols)}")

    # ── 1d. 超节点树 ──
    col_to_sn = [-1] * n
    for sid, cols in enumerate(supernodes):
        for c in cols:
            col_to_sn[c] = sid

    sn_parent = [-1] * len(supernodes)
    sn_children = {i: [] for i in range(len(supernodes))}
    for sid, cols in enumerate(supernodes):
        last = cols[-1]
        p = parent[last]
        if p >= 0:
            p_sn = col_to_sn[p]
            sn_parent[sid] = p_sn
            sn_children[p_sn].append(sid)

    # ── 1e. 每个超节点的面板行集合 ──
    sn_panel_rows = []
    for cols in supernodes:
        last = cols[-1]
        pr = set()
        for c in cols:
            for r in L_cols[c]:
                if r > last:
                    pr.add(r)
        sn_panel_rows.append(sorted(pr))

    print("\n超节点树及面板行:")
    for sid, cols in enumerate(supernodes):
        print(f"  SN#{sid}: cols={cols}, parent={sn_parent[sid]}, "
              f"children={sn_children[sid]}, panel_rows={sn_panel_rows[sid]}")

    return parent, L_cols, supernodes, sn_parent, sn_children, sn_panel_rows


# ═══════════════════════════════════════════════════════════
# 2. 数值分解 (Numerical Factorization) — Left-looking
# ═══════════════════════════════════════════════════════════

def left_looking_supernodal_cholesky(A_perm, supernodes, sn_parent, sn_children,
                                      sn_panel_rows, L_cols):
    """
    CHOLMOD Left-looking 超节点 Cholesky 分解

    关键: 没有显式的前沿矩阵! 直接在 L 存储上操作.
    对每个超节点 S_k (后序遍历):
      1. Left-looking 更新: L(*, S_k) -= sum_{d < first} L(*,d) * L(S_k, d)^T
      2. POTRF: L_diag = chol(L[S_k, S_k])
      3. TRSM:  L_panel = L[panel_rows, S_k] * L_diag^{-T}
    """
    n = A_perm.shape[0]
    num_sn = len(supernodes)

    # L 存储: 初始化为 A 的下三角部分
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            L[i, j] = A_perm[i, j]

    np.set_printoptions(precision=4, suppress=True, linewidth=200)

    # ── 后序遍历超节点 (按列索引递增即自然后序) ──
    for sn_id, cols in enumerate(supernodes):
        ns = len(cols)
        first = cols[0]
        last = cols[-1]
        panel_rows = sn_panel_rows[sn_id]
        nu = len(panel_rows)

        print(f"\n{'='*60}")
        print(f"SN#{sn_id}: cols={cols}, ns={ns}, nu={nu}, panel_rows={panel_rows}")
        print(f"{'='*60}")

        # ────────────────────────────────────────
        # Step 1: Left-looking 更新
        # ────────────────────────────────────────
        # 对所有 d < first, 若 L(r,d)≠0 且 L(cols,d)≠0:
        #   L(r, cols) -= L(r, d) * L(cols, d)    ← 外积形式的散射
        #
        # 更高效的做法: 只遍历 L_cols 中实际有贡献的列
        # 这里为了清晰, 扫描所有 d < first
        #
        # 代码结构 (CHOLMOD 中的 scatter 操作):
        #   for each descendant column d that updates this supernode:
        #     for each row r with L(r,d)!=0:
        #       if r in {rows of current supernode}:
        #         for each col c in current supernode with L(c,d)!=0:
        #           L(r, c) -= L(r, d) * L(c, d)

        for d in range(first):
            # 只考虑对 cols 中某些列有非零贡献的 d
            cols_with_L_cd = [c for c in cols if abs(L[c, d]) > 1e-14]
            if not cols_with_L_cd:
                continue

            # row_set: 当前超节点的所有行 (对角 + 面板)
            row_set = set(cols) | set(panel_rows)

            # 对 L 中列 d 的每个非零行 r
            for r in sorted(L_cols[d]):
                if r not in row_set:
                    continue
                if abs(L[r, d]) < 1e-14:
                    continue

                L_rd = L[r, d]
                for c in cols_with_L_cd:
                    L_cd = L[c, d]
                    if abs(L_cd) < 1e-14:
                        continue
                    L[r, c] -= L_rd * L_cd

        # 打印更新后的结果
        if nu > 0:
            print(f"  Left-looking 更新后 (scope 行):")
            scope = sorted(set(cols) | set(panel_rows))
            for r in scope:
                vals = [f"L[{r},{c}]={L[r,c]:.4f}" for c in cols
                        if abs(L[r,c]) > 1e-14]
                if vals:
                    print(f"    {', '.join(vals)}")

        # ────────────────────────────────────────
        # Step 2: POTRF — 稠密 Cholesky 对角块
        # ────────────────────────────────────────
        # 等效于 LAPACK dpotrf('L', ns, L[first:last+1, first:last+1])
        diag = L[first:last + 1, first:last + 1]
        for k in range(ns):
            rk = first + k
            L[rk, rk] = np.sqrt(L[rk, rk])
            for i in range(k + 1, ns):
                ri = first + i
                L[ri, rk] /= L[rk, rk]
                for j in range(k + 1, i + 1):
                    rj = first + j
                    L[ri, rj] -= L[ri, rk] * L[rj, rk]

        print(f"  POTRF 后对角块:")
        for r in range(first, last + 1):
            vals = [f"{L[r,c]:.4f}" for c in range(first, r + 1)
                    if abs(L[r,c]) > 1e-14]
            print(f"    行{r}: " + "  ".join(vals))

        # ────────────────────────────────────────
        # Step 3: TRSM — 三角求解面板
        # ────────────────────────────────────────
        # 等效于 LAPACK dtrsm('R','L','T','N', nu, ns, 1.0,
        #                     L_diag, ns, L_panel, nu)
        if nu > 0:
            L_diag = L[first:last + 1, first:last + 1]  # ns×ns 下三角
            for pi, r in enumerate(panel_rows):
                for k in range(ns):
                    ck = first + k
                    L[r, ck] /= L_diag[k, k]
                    for j in range(k + 1, ns):
                        cj = first + j
                        L[r, cj] -= L[r, ck] * L_diag[j, k]

            print(f"  TRSM 后面板 (行 {panel_rows}):")
            for r in panel_rows:
                vals = [f"L[{r},{c}]={L[r,c]:.4f}" for c in cols
                        if abs(L[r,c]) > 1e-14]
                print(f"    {', '.join(vals)}")

    return L


# ═══════════════════════════════════════════════════════════
# 3. 主流程
# ═══════════════════════════════════════════════════════════

def main():
    A = build_example_matrix()
    n = 10

    # ── 排序 ──
    # AMD 排序 (此处手写 AMD 结果; 实际中调用 CHOLMOD 或 AMD 库)
    perm_amd = np.array([0, 1, 2, 4, 5, 3, 7, 6, 8, 9], dtype=int)
    P = np.zeros((n, n), dtype=int)
    for i, p in enumerate(perm_amd):
        P[i, p] = 1
    A_perm = P @ A @ P.T

    print("原始 perm:", perm_amd)
    print("A_perm:\n", A_perm.astype(int))

    # ── 符号分析 ──
    parent, L_cols, supernodes, sn_parent, sn_children, sn_panel_rows = \
        symbolic_analysis(A_perm)

    # ── 数值分解 ──
    L = left_looking_supernodal_cholesky(
        A_perm, supernodes, sn_parent, sn_children, sn_panel_rows, L_cols
    )

    # ── 验证 ──
    print(f"\n{'='*60}")
    print("验证: max|L@L.T - A_perm| =", np.max(np.abs(L @ L.T - A_perm)))
    print(f"nnz(L) = {np.count_nonzero(np.abs(np.tril(L)) > 1e-14)}")
    print(f"{'='*60}")

    print("\n最终 L 因子:")
    for i in range(n):
        vals = [f"{L[i,j]:.4f}" if abs(L[i,j]) > 1e-14 else "0" for j in range(i+1)]
        print(f"  行{i}: " + "  ".join(vals))


if __name__ == "__main__":
    main()
