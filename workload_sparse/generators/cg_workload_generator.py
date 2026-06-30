import os
import json
import sys
import argparse
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple

try:
    import ssgetpy
except ImportError:
    ssgetpy = None

try:
    from scipy.io import mmread
except ImportError:
    mmread = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_DATA_DIR = ROOT / "matrix_data"

DTYPE_BYTES = {
    "fp16": 2,
    "bf16": 2,
    "fp32": 4,
    "fp64": 8,
}
INDEX_BYTES = 4


# ============================================================================
# 配置与统计
# ============================================================================

@dataclass
class CGConfig:
    num_iterations: int = 1
    tolerance: float = 1e-6
    max_iterations: int = 1000
    dtype: str = "fp16"
    dot_collective: str = "none"

    def validate(self) -> Tuple[bool, str]:
        if self.num_iterations < 1:
            return False, "迭代次数必须 >= 1"
        if self.num_iterations > self.max_iterations:
            return False, f"迭代次数不能超过 {self.max_iterations}"
        if not (0.0 < self.tolerance < 1.0):
            return False, "精度 tolerance 必须在 (0, 1) 之间"
        if self.dtype not in DTYPE_BYTES:
            return False, f"dtype 必须是 {sorted(DTYPE_BYTES)} 之一"
        if self.dot_collective not in {"none", "allreduce"}:
            return False, "dot_collective 必须是 ['none', 'allreduce'] 之一"
        return True, "配置有效"


@dataclass
class MatrixStats:
    name: str
    rows: int
    cols: int
    nnz: int
    is_square: bool
    dtype: str
    bytes: Dict[str, int]
    flops: Dict[str, float]


# ============================================================================
# 1) Matrix Provider
# ============================================================================

class MatrixProvider:
    @staticmethod
    def fetch_matrix(matrix_name: Optional[str], local_mtx: Optional[str], dest_dir: str) -> str:
        if local_mtx:
            if not os.path.exists(local_mtx):
                raise FileNotFoundError(local_mtx)
            print(f"[1/4] Matrix Provider: 使用本地矩阵 {os.path.abspath(local_mtx)}")
            return local_mtx

        if ssgetpy is None:
            raise RuntimeError("ssgetpy not available; install it or use --local-mtx.")

        os.makedirs(dest_dir, exist_ok=True)

        print("[1/4] Matrix Provider: 搜索/下载 SuiteSparse SPD 矩阵...")

        if matrix_name:
            results = ssgetpy.search(name=matrix_name, isspd=True)
        else:
            results = ssgetpy.search(isspd=True, nzbounds=(10000, 100000), limit=1)

        if not results:
            raise ValueError("未找到符合条件的 SPD 矩阵")

        m = results[0]
        print(f"       目标矩阵: {m.name} (group: {m.group})")
        print(f"       维度: {m.rows}×{m.cols}, nnz={m.nnz}")

        extracted_path, _ = m.download(destpath=dest_dir, extract=True)
        mtx_path = os.path.join(extracted_path, f"{m.name}.mtx")
        if not os.path.exists(mtx_path):
            raise FileNotFoundError(f"未找到 .mtx 文件: {mtx_path}")

        print(f"       .mtx: {mtx_path}")
        return mtx_path


# ============================================================================
# 2) Matrix Analyzer
# ============================================================================

class MatrixAnalyzer:
    @staticmethod
    def _matrix_market_shape_nnz(mtx_file_path: str) -> Tuple[int, int, int]:
        opener = gzip.open if mtx_file_path.endswith(".gz") else open
        with opener(mtx_file_path, "rt", encoding="utf-8", errors="replace") as f:
            header = f.readline().strip().lower().split()
            if len(header) < 5 or header[:3] != ["%%matrixmarket", "matrix", "coordinate"]:
                raise RuntimeError("Only Matrix Market coordinate files are supported without scipy.")

            symmetry = header[4]
            dims = None
            diag_entries = 0
            stored_entries = 0

            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                parts = stripped.split()
                dims = (int(parts[0]), int(parts[1]), int(parts[2]))
                break

            if dims is None:
                raise RuntimeError(f"Missing Matrix Market dimensions in {mtx_file_path}")

            rows, cols, declared_entries = dims
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("%"):
                    continue
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                row = int(parts[0])
                col = int(parts[1])
                stored_entries += 1
                if row == col:
                    diag_entries += 1

            # Prefer the declared count if the file ended exactly as expected, but keep
            # the parsed diagonal count to expand symmetric storage like scipy.mmread.
            entries = declared_entries if declared_entries == stored_entries else stored_entries
            if symmetry in {"symmetric", "hermitian"}:
                nnz = diag_entries + 2 * (entries - diag_entries)
            elif symmetry == "skew-symmetric":
                nnz = 2 * entries
            else:
                nnz = entries

            return rows, cols, int(nnz)

    @staticmethod
    def analyze(mtx_file_path: str, dtype: str) -> MatrixStats:
        print("[2/4] Matrix Analyzer: 读取矩阵并估算 bytes/FLOPs...")

        if mmread is not None:
            A = mmread(mtx_file_path)
            rows, cols = A.shape
            nnz = A.nnz
        else:
            rows, cols, nnz = MatrixAnalyzer._matrix_market_shape_nnz(mtx_file_path)

        # 这里仅判断“是否方阵”，不做昂贵的 A == A.T 检查
        is_square = (rows == cols)

        # bytes 估算按 CSR：values + col_idx + row_ptr
        value_bytes = DTYPE_BYTES[dtype]
        matrix_bytes = nnz * value_bytes + nnz * INDEX_BYTES + (rows + 1) * INDEX_BYTES
        vector_bytes = rows * value_bytes
        scalar_bytes = value_bytes

        # FLOPs 估算（经典近似）
        spmv_flops = 2 * nnz
        dot_flops = 2 * rows
        axpy_flops = 2 * rows
        scal_flops = 1.0

        name = os.path.basename(mtx_file_path)
        if name.endswith(".mtx.gz"):
            name = name[:-7]
        elif name.endswith(".mtx"):
            name = name[:-4]

        stats = MatrixStats(
            name=name,
            rows=rows,
            cols=cols,
            nnz=nnz,
            is_square=is_square,
            dtype=dtype,
            bytes={"matrix": int(matrix_bytes), "vector": int(vector_bytes), "scalar": int(scalar_bytes)},
            flops={
                "spmv": float(spmv_flops),
                "dot": float(dot_flops),
                "axpy": float(axpy_flops),
                "scal": float(scal_flops),
            },
        )

        print(f"       维度: {rows}×{cols}, nnz={nnz}, is_square={is_square}")
        print(f"       FLOPs: spmv={spmv_flops}, dot={dot_flops}, axpy={axpy_flops}")
        return stats


# ============================================================================
# 3) Workload Builder (CG)
# ============================================================================

class WorkloadBuilder:
    @staticmethod
    def build_cg_workload(stats: MatrixStats, cfg: CGConfig, emit_metadata: bool) -> Dict[str, Any]:
        print("[3/4] Workload Builder: 生成 CG workload DAG...")

        b = stats.bytes
        f = stats.flops
        n = stats.rows
        value_bytes = DTYPE_BYTES[cfg.dtype]

        tensors: List[Dict[str, Any]] = []
        tasks: List[Dict[str, Any]] = []
        next_task_id = 0

        distributed_dot = cfg.dot_collective == "allreduce"
        dot_placement_hint = {"parallelism": "matrix_parallel"} if distributed_dot else None

        def add_tensor(
            tid: str,
            size_bytes: int,
            producer: Optional[int] = None,
            access_pattern: str = "dense",
            shape: Optional[List[int]] = None,
            num_elements: Optional[int] = None,
            dtype: Optional[str] = None,
            collective_hint: Optional[Dict[str, Any]] = None,
        ) -> None:
            t: Dict[str, Any] = {
                "id": tid,
                "size_bytes": int(max(0, size_bytes)),
                "producer": None if producer is None else int(producer),
                "dtype": dtype or cfg.dtype,
            }
            if shape is not None:
                t["shape"] = shape
            if num_elements is not None:
                t["num_elements"] = int(num_elements)
            t["access_pattern"] = access_pattern
            if collective_hint is not None:
                t["collective_hint"] = collective_hint
            tensors.append(t)

        def add_vector(tid: str, producer: Optional[int] = None) -> None:
            add_tensor(
                tid,
                b["vector"],
                producer=producer,
                shape=[n],
                num_elements=n,
            )

        def add_scalar(
            tid: str,
            producer: Optional[int] = None,
            collective: bool = False,
        ) -> None:
            add_tensor(
                tid,
                b["scalar"],
                producer=producer,
                shape=[1],
                num_elements=1,
                collective_hint={"type": "allreduce", "op": "sum"} if collective else None,
            )

        def add_task(
            name: str,
            op: str,
            flops: float,
            inputs: List[str],
            outputs: List[str],
            bytes_read: int,
            bytes_written: int,
            placement_hint: Optional[Dict[str, Any]] = None,
        ) -> int:
            nonlocal next_task_id
            tid = next_task_id
            next_task_id += 1
            task: Dict[str, Any] = {
                "id": tid,
                "name": name,
                "op": op,
                "compute_flops": float(flops),
                "bytes_read": int(bytes_read),
                "bytes_written": int(bytes_written),
                "inputs": [{"tensor": x} for x in inputs],
                "outputs": outputs,  # 规范允许字符串数组
            }
            if placement_hint is not None:
                task["placement_hint"] = placement_hint
            tasks.append(task)
            return tid

        # -------- 外部输入/状态输入 tensors --------
        add_tensor(
            "A",
            b["matrix"],
            producer=None,
            access_pattern="sparse_csr",
            shape=[stats.rows, stats.cols],
            num_elements=stats.nnz,
        )
        add_vector("x")
        add_vector("r")
        add_vector("p")
        add_scalar("r_old")

        spmv_bytes_read = b["matrix"] + stats.nnz * value_bytes
        spmv_bytes_written = b["vector"]
        dot_xy_bytes_read = 2 * b["vector"]
        dot_self_bytes_read = b["vector"]
        dot_bytes_written = b["scalar"]
        scalar_div_bytes_read = 2 * b["scalar"]
        scalar_div_bytes_written = b["scalar"]
        axpy_bytes_read = 2 * b["vector"] + b["scalar"]
        axpy_bytes_written = b["vector"]

        # -------- 迭代展开（静态图：num_iterations 决定任务数量）--------
        prev = {"x": "x", "r": "r", "p": "p", "r_old": "r_old"}
        last_outputs = None

        for k in range(cfg.num_iterations):
            suf = f"_iter{k}"

            # q = A * p
            q = f"q{suf}"
            add_vector(q)
            t0 = add_task(
                f"spmv{suf}",
                "spmv",
                f["spmv"],
                ["A", prev["p"]],
                [q],
                spmv_bytes_read,
                spmv_bytes_written,
            )
            tensors[-1]["producer"] = t0

            # p_dot_q = p^T q
            p_dot_q = f"p_dot_q{suf}"
            add_scalar(p_dot_q, collective=distributed_dot)
            t1 = add_task(
                f"dot_pdotq{suf}",
                "dot",
                f["dot"],
                [prev["p"], q],
                [p_dot_q],
                dot_xy_bytes_read,
                dot_bytes_written,
                dot_placement_hint,
            )
            tensors[-1]["producer"] = t1

            # alpha = r_old / p_dot_q
            alpha = f"alpha{suf}"
            add_scalar(alpha)
            t2 = add_task(
                f"scalar_div_alpha{suf}",
                "scal",
                f["scal"],
                [prev["r_old"], p_dot_q],
                [alpha],
                scalar_div_bytes_read,
                scalar_div_bytes_written,
            )
            tensors[-1]["producer"] = t2

            # x_next = x + alpha * p
            x_next = f"x_next{suf}"
            add_vector(x_next)
            t3 = add_task(
                f"axpy_xupdate{suf}",
                "axpy",
                f["axpy"],
                [prev["x"], prev["p"], alpha],
                [x_next],
                axpy_bytes_read,
                axpy_bytes_written,
            )
            tensors[-1]["producer"] = t3

            # r_next = r - alpha * q
            r_next = f"r_next{suf}"
            add_vector(r_next)
            t4 = add_task(
                f"axpy_rupdate{suf}",
                "axpy",
                f["axpy"],
                [prev["r"], q, alpha],
                [r_next],
                axpy_bytes_read,
                axpy_bytes_written,
            )
            tensors[-1]["producer"] = t4

            # r_new = r_next^T r_next
            r_new = f"r_new{suf}"
            add_scalar(r_new, collective=distributed_dot)
            t5 = add_task(
                f"dot_rnorm{suf}",
                "dot",
                f["dot"],
                [r_next, r_next],
                [r_new],
                dot_self_bytes_read,
                dot_bytes_written,
                dot_placement_hint,
            )
            tensors[-1]["producer"] = t5

            # beta = r_new / r_old
            beta = f"beta{suf}"
            add_scalar(beta)
            t6 = add_task(
                f"scalar_div_beta{suf}",
                "scal",
                f["scal"],
                [r_new, prev["r_old"]],
                [beta],
                scalar_div_bytes_read,
                scalar_div_bytes_written,
            )
            tensors[-1]["producer"] = t6

            # p_next = r_next + beta * p
            p_next = f"p_next{suf}"
            add_vector(p_next)
            t7 = add_task(
                f"axpy_pupdate{suf}",
                "axpy",
                f["axpy"],
                [r_next, prev["p"], beta],
                [p_next],
                axpy_bytes_read,
                axpy_bytes_written,
            )
            tensors[-1]["producer"] = t7

            prev = {"x": x_next, "r": r_next, "p": p_next, "r_old": r_new}
            last_outputs = (x_next, r_next, p_next, r_new)

        assert last_outputs is not None

        workload: Dict[str, Any] = {
            "name": f"cg_solver_{stats.name}_iter{cfg.num_iterations}_tol{cfg.tolerance:.0e}",
            "tensors": tensors,
            "tasks": tasks,
        }

        if emit_metadata:
            workload["metadata"] = {
                "algorithm": "CG",
                "num_iterations": cfg.num_iterations,
                "tolerance": cfg.tolerance,
                "matrix": stats.name,
                "rows": stats.rows,
                "cols": stats.cols,
                "nnz": stats.nnz,
                "is_square": stats.is_square,
                "dtype": cfg.dtype,
                "dot_collective": cfg.dot_collective,
                "memory_model": {
                    "spmv": "read CSR A + one RHS value per nonzero, write q",
                    "dot": "read two vectors, or one vector for self-dot, write scalar",
                    "axpy": "read two vectors and one scalar, write one vector",
                    "scal": "read two scalars, write one scalar",
                },
                "note": "tolerance is metadata only (static DAG does not early-stop)",
            }

        WorkloadValidator.validate(workload, cfg.num_iterations)

        total_flops = sum(t["compute_flops"] for t in tasks)
        total_memory = sum(WorkloadValidator.task_memory_bytes(t) for t in tasks)
        print(f"       tasks={len(tasks)} (={cfg.num_iterations}×8), tensors={len(tensors)}")
        print(f"       total_compute_flops={total_flops:.2e}, total_memory_bytes={total_memory:.2e}")
        return workload


# ============================================================================
# 4) Workload Validator
# ============================================================================

class WorkloadValidator:
    @staticmethod
    def task_memory_bytes(task: Dict[str, Any]) -> float:
        memory_bytes = float(task.get("memory_bytes", 0.0))
        if memory_bytes > 0.0:
            return memory_bytes
        return float(task.get("bytes_read", 0.0)) + float(task.get("bytes_written", 0.0))

    @staticmethod
    def validate(workload: Dict[str, Any], num_iterations: int) -> None:
        tensors = workload.get("tensors", [])
        tasks = workload.get("tasks", [])

        if len(tasks) != num_iterations * 8:
            raise ValueError(f"CG task 数量错误: expected {num_iterations * 8}, got {len(tasks)}")

        tensor_ids: Set[str] = set()
        tensor_by_id: Dict[str, Dict[str, Any]] = {}
        for tensor in tensors:
            tid = tensor.get("id")
            if not isinstance(tid, str) or not tid:
                raise ValueError("tensor 缺少有效 id")
            if tid in tensor_ids:
                raise ValueError(f"重复 tensor id: {tid}")
            tensor_ids.add(tid)
            tensor_by_id[tid] = tensor

        task_ids: Set[int] = set()
        task_by_id: Dict[int, Dict[str, Any]] = {}
        task_names: Set[str] = set()
        for task in tasks:
            tid = task.get("id")
            name = task.get("name")
            if not isinstance(tid, int):
                raise ValueError(f"task 缺少有效 id: {task}")
            if tid in task_ids:
                raise ValueError(f"重复 task id: {tid}")
            if not isinstance(name, str) or not name:
                raise ValueError(f"task 缺少有效 name: {task}")
            if name in task_names:
                raise ValueError(f"重复 task name: {name}")
            task_ids.add(tid)
            task_by_id[tid] = task
            task_names.add(name)

            compute_flops = float(task.get("compute_flops", 0.0))
            bytes_read = float(task.get("bytes_read", 0.0))
            bytes_written = float(task.get("bytes_written", 0.0))
            memory_bytes = float(task.get("memory_bytes", 0.0))
            if compute_flops <= 0.0:
                raise ValueError(f"task compute_flops 必须为正: {name}")
            effective_memory = WorkloadValidator.task_memory_bytes(task)
            if bytes_read < 0.0 or bytes_written < 0.0 or effective_memory <= 0.0:
                raise ValueError(f"task 访存量字段非法: {name}")
            if memory_bytes > 0.0 and abs(memory_bytes - (bytes_read + bytes_written)) > 1e-9:
                raise ValueError(f"task memory_bytes 与 bytes_read/bytes_written 不一致: {name}")

        for tensor in tensors:
            producer = tensor.get("producer")
            if producer is None:
                continue
            if producer not in task_by_id:
                raise ValueError(f"tensor producer 不存在: {tensor['id']} -> {producer}")

        edges: Dict[int, Set[int]] = {task_id: set() for task_id in task_ids}
        for task in tasks:
            task_id = task["id"]
            for output_id in task.get("outputs", []):
                if output_id not in tensor_by_id:
                    raise ValueError(f"task output tensor 未定义: {task['name']} -> {output_id}")
                if tensor_by_id[output_id].get("producer") != task_id:
                    raise ValueError(f"task output producer 不匹配: {task['name']} -> {output_id}")

            for item in task.get("inputs", []):
                input_id = item.get("tensor")
                if input_id not in tensor_by_id:
                    raise ValueError(f"task input tensor 未定义: {task['name']} <- {input_id}")
                producer = tensor_by_id[input_id].get("producer")
                if producer is None or producer == task_id:
                    continue
                if producer > task_id:
                    raise ValueError(f"producer 出现在 consumer 之后: {producer} -> {task_id}")
                edges[producer].add(task_id)

        visiting: Set[int] = set()
        visited: Set[int] = set()

        def visit(task_id: int) -> None:
            if task_id in visited:
                return
            if task_id in visiting:
                raise ValueError(f"CG workload 依赖成环，task id={task_id}")
            visiting.add(task_id)
            for dst in edges[task_id]:
                visit(dst)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in sorted(task_ids):
            visit(task_id)


# ============================================================================
# 5) JSON Writer
# ============================================================================

class JSONWriter:
    @staticmethod
    def write(data: Dict[str, Any], path: str) -> None:
        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[4/4] JSON Writer:  写入 {os.path.abspath(path)}")


# ============================================================================
# main
# ============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="CG Workload JSON Generator (minimal)")
    parser.add_argument("--matrix", type=str, default=None, help="SuiteSparse 矩阵名（默认自动选一个 SPD）")
    parser.add_argument("--local-mtx", type=str, default=None, help="使用本地 .mtx 文件而不是下载")
    parser.add_argument("--dest-dir", type=str, default=str(DEFAULT_MATRIX_DATA_DIR), help="SuiteSparse 下载目录")
    parser.add_argument("--iterations", type=int, default=300, help="CG 迭代次数（默认 300，范围 1-1000）")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="收敛精度阈值（默认 1e-6，仅写入元数据）")
    parser.add_argument("--dtype", type=str, default="fp16", choices=sorted(DTYPE_BYTES), help="数值 dtype")
    parser.add_argument(
        "--dot-collective",
        type=str,
        default="none",
        choices=["none", "allreduce"],
        help="dot 归约通信建模：none=全局 dot task 内完成；allreduce=分片 dot 输出用 allreduce 汇总",
    )
    parser.add_argument("--emit-metadata", action="store_true", help="在 workload 顶层输出非 schema metadata（默认不输出）")
    parser.add_argument("--out", type=str, default="workload.json", help="输出 workload JSON 路径")

    args = parser.parse_args(argv)

    cfg = CGConfig(
        num_iterations=args.iterations,
        tolerance=args.tolerance,
        dtype=args.dtype,
        dot_collective=args.dot_collective,
    )
    ok, msg = cfg.validate()
    if not ok:
        print(f" 配置错误: {msg}")
        sys.exit(1)

    print(
        f"✓ 配置验证: {msg}  "
        f"(iterations={cfg.num_iterations}, tolerance={cfg.tolerance:.2e}, dot_collective={cfg.dot_collective})"
    )

    mtx = MatrixProvider.fetch_matrix(args.matrix, args.local_mtx, args.dest_dir)
    stats = MatrixAnalyzer.analyze(mtx, cfg.dtype)
    workload = WorkloadBuilder.build_cg_workload(stats, cfg, emit_metadata=args.emit_metadata)
    JSONWriter.write(workload, args.out)


if __name__ == "__main__":
    main()
