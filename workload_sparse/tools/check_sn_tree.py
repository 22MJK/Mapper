"""检查实际生成的超节点前沿更新边 vs JSON 中的 gessm 依赖边"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matrix_analyzer import matrix_analyzer

# 1. 运行实际分析
meta = matrix_analyzer('demo10', str(ROOT / 'matrix_data/demo10/demo10.mtx'), 'amd', 64, 0.85, 0.35)

print("=" * 60)
print("【实际 matrix_analyzer 输出】")
print("=" * 60)
for i, cols in enumerate(meta.supernodes):
    print(
        f"  SN#{i}: cols={cols}, ns={len(cols)}, nu={meta.sn_nu[i]}, "
        f"nnz={meta.sn_nnz[i]}, panel_rows={meta.sn_panel_rows[i]}"
    )

print(f"\nsn_parent: {meta.sn_parent}")
print(f"sn_children: {meta.sn_children}")
print(f"nnz(L) = {meta.nnz_l}")
print(f"symbolic_source = {meta.symbolic_source}")

# 2. 基于真实 panel rows 和目标前沿作用域生成应有 gessm 边：
print("\n【应由真实 panel rows 生成的 gessm 边】")
gessm_edges = {}
for source, source_cols in enumerate(meta.supernodes):
    source_panel = set(meta.sn_panel_rows[source])
    for target, target_cols in enumerate(meta.supernodes):
        if target <= source:
            continue
        target_scope = list(target_cols) + meta.sn_panel_rows[target]
        overlap = tuple(row for row in target_scope if row in source_panel)
        if not overlap:
            continue
        gessm_edges[(source, target)] = overlap
        print(f"  gessm_{source}_to_{target}: overlap_rows={len(overlap)}, rows={list(overlap)}")

# 3. 读取 JSON 中的实际 gessm 边
print("\n" + "=" * 60)
print("【JSON workload 中的实际 gessm 边】")
print("=" * 60)
with open(ROOT / "workload_supernodal_pdf.json") as f:
    wl = json.load(f)

json_edges = {}
for t in wl["tasks"]:
    if t["op"] == "gessm":
        name = t["name"]
        # sn_gessm_X_to_Y
        parts = name.replace("sn_gessm_", "").split("_to_")
        child_id = int(parts[0])
        parent_id = int(parts[1])
        json_edges[(child_id, parent_id)] = tuple(t.get("overlap_row_ids", []))
        print(
            f"  {name}: source={child_id} -> target={parent_id}, "
            f"overlap_rows={t.get('overlap_rows')}, rows={t.get('overlap_row_ids')}"
        )

# 4. 对比
print("\n" + "=" * 60)
print("【差异对比】")
print("=" * 60)
expected_keys = set(gessm_edges)
json_keys = set(json_edges)
only_in_code = expected_keys - json_keys
only_in_json = json_keys - expected_keys
common = expected_keys & json_keys
row_mismatch = {
    edge: (gessm_edges[edge], json_edges[edge])
    for edge in common
    if gessm_edges[edge] != json_edges[edge]
}

print(f"  共同边: {common if common else '(无)'}")
print(f"  仅期望边有, JSON 缺: {only_in_code if only_in_code else '(无)'}")
print(f"  仅 JSON 有, 期望边缺: {only_in_json if only_in_json else '(无)'}")
print(f"  overlap row 不一致: {row_mismatch if row_mismatch else '(无)'}")

# 5. 比较 nu 值
print("\n" + "=" * 60)
print("【nu 对比：实际 vs JSON】")
print("=" * 60)
for t in wl["tasks"]:
    if t["op"] == "tstrf":
        sn_id = int(t["name"].replace("sn_tstrf_", ""))
        json_nu = t.get("nu", "N/A")
        code_nu = meta.sn_nu[sn_id] if sn_id < len(meta.sn_nu) else "N/A"
        match = "✅" if json_nu == code_nu else "❌"
        print(f"  SN#{sn_id}: JSON nu={json_nu}, code nu={code_nu} {match}")
