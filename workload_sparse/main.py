import argparse
import os

from json_writer import write_workload
from matrix_analyzer import matrix_analyzer
from matrix_provider import matrix_provider
from workload_builder import DTYPE_BYTES, build_supernodal_workload


def main() -> None:
    ap = argparse.ArgumentParser("Supernodal workload generator (PDF method)")
    ap.add_argument("--matrix", type=str, default=None, help="SuiteSparse matrix name (SPD)")
    ap.add_argument("--local-mtx", type=str, default=None, help="Use local .mtx instead of downloading")
    ap.add_argument("--ordering", type=str, default="amd", help="amd|metis|natural|nesdis")
    ap.add_argument("--max-supernode-size", type=int, default=64)
    ap.add_argument("--overlap-threshold", type=float, default=0.85)
    ap.add_argument("--max-fill-ratio", type=float, default=0.35)
    ap.add_argument("--dtype", type=str, default="fp16", choices=sorted(DTYPE_BYTES.keys()))
    ap.add_argument("--out", type=str, default="workload_supernodal_pdf.json")
    ap.add_argument("--dest-dir", type=str, default="./matrix_data")
    ap.add_argument("--n-min", type=int, default=10, help="Min matrix dimension n for SuiteSparse search")
    ap.add_argument("--n-max", type=int, default=100, help="Max matrix dimension n for SuiteSparse search")
    ap.add_argument("--nnz-min", type=int, default=10, help="Min nnz for SuiteSparse search")
    ap.add_argument("--nnz-max", type=int, default=100, help="Max nnz for SuiteSparse search")
    args = ap.parse_args()

    name, mtx_path = matrix_provider(
        args.matrix,
        args.local_mtx,
        args.dest_dir,
        args.n_min,
        args.n_max,
        args.nnz_min,
        args.nnz_max,
    )
    meta = matrix_analyzer(
        name=name,
        mtx_path=mtx_path,
        ordering=args.ordering,
        max_supernode_size=args.max_supernode_size,
        overlap_threshold=args.overlap_threshold,
        max_fill_ratio=args.max_fill_ratio,
    )
    workload = build_supernodal_workload(meta=meta, dtype=args.dtype)
    write_workload(args.out, workload)

    print(f"[Write] {os.path.abspath(args.out)}")
    total_tasks = len(workload["tasks"])
    total_tensors = len(workload["tensors"])

    # 汇总各类 FLOPs（CHOLMOD Left-looking 算子）
    symbolic_ops = {
        "amd_ordering",
        "elimination_tree",
        "postorder",
        "column_counts",
        "supernode_partition",
        "supernode_relax",
        "symbolic_pattern",
    }
    symbolic_total = sum(t["compute_flops"] for t in workload["tasks"] if t["op"] in symbolic_ops)
    tstrf_total = sum(t["compute_flops"] for t in workload["tasks"] if t["op"] == "tstrf")
    gessm_total = sum(t["compute_flops"] for t in workload["tasks"] if t["op"] == "gessm")
    ssssm_total = sum(t["compute_flops"] for t in workload["tasks"] if t["op"] == "ssssm")

    print(
        "[Summary] "
        f"n={meta.rows}, nnz(A)={meta.nnz}, nnz(L)={meta.nnz_l}, "
        f"supernodes={len(meta.supernodes)}, "
        f"tasks={total_tasks}, tensors={total_tensors}, "
        f"ordering={meta.ordering}"
    )
    print(
        f"[FLOPs]  symbolic={symbolic_total:.2e}, "
        f"tstrf(POTRF+TRSM)={tstrf_total:.2e}, "
        f"gessm(child→parent)={gessm_total:.2e}, "
        f"ssssm(fwd+bwd)={ssssm_total:.2e}"
    )


if __name__ == "__main__":
    main()
