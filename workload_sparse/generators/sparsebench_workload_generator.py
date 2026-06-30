#!/usr/bin/env python3
"""Generate SparseBench-style mapper workload JSON.

SparseBench combines sparse iterative methods (CG/GMRES), sparse storage
schemes (regular seven-point/DIA/CRS), and preconditioners
(none/Jacobi/ILU/BJAC).  The mapper does not have SparseBench-specific
operators, so this generator lowers those cases onto the existing operator
catalog:

  * CG/GMRES iteration bodies: spmv, dot, nrm2, axpy, scal/copy
  * Jacobi apply: scal over a diagonal-inverse vector
  * BJAC apply: forward/backward triangular-style block solves
  * ILU apply: two triangular solves (L and U)
  * DIA SpMV: encoded as spmv with row-wise access, backed by CSR storage
    metadata so the current parser/replay path still preserves nnz.

The output is a static unrolled DAG.  Tolerance and restart metadata do not
cause early stopping.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


DTYPE_BYTES = {
    "fp32": 4,
    "fp64": 8,
}
INDEX_BYTES = 4

METHODS = {"cg", "gmres"}
STORAGE_FORMATS = {"crs", "dia", "regular"}
PRECONDITIONERS = {"none", "jacobi", "ilu", "bjac"}
DOT_COLLECTIVES = {"none", "allreduce"}
NETLIB_DEFAULT_SIZES = (12, 14, 16, 18, 20, 24, 28, 32, 36, 38)
OFFICIAL_TOOL_NAMES = ("bench_gen", "bench_sym", "crs_gen", "crs_sym", "reg_gen", "reg_sym")
OFFICIAL_METHODS = ("cg", "gmres")
OFFICIAL_SYM_CODES = {"1": "cg", "0": "gmres"}
OFFICIAL_STORAGE_CODES = {"1": "regular", "2": "crs"}
OFFICIAL_PREC_CODES = {"0": "none", "1": "jacobi", "2": "ilu", "3": "bjac"}
OFFICIAL_TEST_POINTS = (
    ("cg", "regular", "none"),
    ("cg", "regular", "ilu"),
    ("cg", "regular", "bjac"),
    ("cg", "crs", "none"),
    ("cg", "crs", "ilu"),
    ("gmres", "regular", "none"),
    ("gmres", "regular", "ilu"),
    ("gmres", "regular", "bjac"),
    ("gmres", "crs", "none"),
    ("gmres", "crs", "ilu"),
)


@dataclass(frozen=True)
class SparseBenchConfig:
    method: str
    storage: str
    preconditioner: str
    n: int
    iterations: int
    restart: int
    nnz_per_row: int
    num_diagonals: int
    dtype: str
    dot_collective: str
    emit_metadata: bool
    matrix_nnz_override: Optional[int] = None
    matrix_size_bytes_override: Optional[int] = None
    official_size: Optional[int] = None
    official_matrix_file: Optional[str] = None
    official_symmetry: Optional[str] = None

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)}")
        if self.storage not in STORAGE_FORMATS:
            raise ValueError(f"storage must be one of {sorted(STORAGE_FORMATS)}")
        if self.preconditioner not in PRECONDITIONERS:
            raise ValueError(f"preconditioner must be one of {sorted(PRECONDITIONERS)}")
        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.restart <= 0:
            raise ValueError("restart must be > 0")
        if self.nnz_per_row <= 0:
            raise ValueError("nnz_per_row must be > 0")
        if self.num_diagonals <= 0:
            raise ValueError("num_diagonals must be > 0")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"dtype must be one of {sorted(DTYPE_BYTES)}")
        if self.dot_collective not in DOT_COLLECTIVES:
            raise ValueError(f"dot_collective must be one of {sorted(DOT_COLLECTIVES)}")

    @property
    def value_bytes(self) -> int:
        return DTYPE_BYTES[self.dtype]

    @property
    def matrix_nnz(self) -> int:
        if self.matrix_nnz_override is not None:
            return int(self.matrix_nnz_override)
        if self.storage in {"dia", "regular"}:
            return int(self.n * self.num_diagonals)
        return int(self.n * self.nnz_per_row)

    @property
    def matrix_size_bytes(self) -> int:
        if self.matrix_size_bytes_override is not None:
            return int(self.matrix_size_bytes_override)
        nnz = self.matrix_nnz
        if self.storage in {"dia", "regular"}:
            # Values for each diagonal plus one int offset per diagonal.
            return int(nnz * self.value_bytes + self.num_diagonals * INDEX_BYTES)
        # CSR/CRS: values + col_idx + row_ptr.
        return int(nnz * (self.value_bytes + INDEX_BYTES) + (self.n + 1) * INDEX_BYTES)

    @property
    def vector_size_bytes(self) -> int:
        return int(self.n * self.value_bytes)

    @property
    def scalar_size_bytes(self) -> int:
        return int(self.value_bytes)


@dataclass(frozen=True)
class OfficialMatrixMeta:
    path: Path
    storage: str
    symmetry: str
    size: int
    n: int
    nnz: int
    storage_values: int
    matrix_size_bytes: int


@dataclass(frozen=True)
class OfficialTestPlan:
    sizes: Sequence[int]
    test_points: Sequence[tuple[str, str, str]]
    source: str


def official_matrix_filename(storage: str, size: int, symmetry: str) -> str:
    prefix = "regmat" if storage == "regular" else "crsmat"
    suffix = "s" if symmetry == "symmetric" else "u"
    return f"{prefix}{int(size):03d}{suffix}"


def parse_shell_words(value: str) -> List[str]:
    return [part for part in shlex.split(value, comments=False, posix=True) if part]


def parse_shell_assignment(script_text: str, name: str) -> Optional[str]:
    pattern = re.compile(rf"^\s*{re.escape(name)}=(?P<value>.+?)\s*(?:#.*)?$", re.MULTILINE)
    match = pattern.search(script_text)
    if not match:
        return None
    value = match.group("value").strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    return value


def parse_for_loop_words(script_text: str, variable: str) -> List[str]:
    pattern = re.compile(rf"for\s+{re.escape(variable)}\s+in\s+(?P<value>[^;\n]+)\s*;")
    match = pattern.search(script_text)
    if not match:
        return []
    raw = match.group("value").strip()
    if raw.startswith("$"):
        assigned = parse_shell_assignment(script_text, raw[1:])
        raw = assigned if assigned is not None else ""
    return parse_shell_words(raw)


def parse_prec_sets(script_text: str) -> Dict[str, List[str]]:
    one_line = re.search(
        r'if\s+\[\s+\$stor\s+-eq\s+1\s+\]\s*;\s*then\s+PRECS="(?P<regular>[^"]+)"\s*;\s*else\s+PRECS="(?P<crs>[^"]+)"',
        script_text,
    )
    if one_line:
        return {
            "1": parse_shell_words(one_line.group("regular")),
            "2": parse_shell_words(one_line.group("crs")),
        }

    # Fallback to Netlib 0.9.7 defaults if the script structure changes beyond
    # the simple parser. The source is still reported in the manifest.
    return {"1": ["0", "2", "3"], "2": ["0", "2"]}


def parse_official_test_plan(sparsebench_root: str) -> OfficialTestPlan:
    root = Path(sparsebench_root).resolve()
    test_script = root / "Test"
    if not test_script.exists():
        raise ValueError(f"official SparseBench Test script not found: {test_script}")
    script_text = test_script.read_text(encoding="utf-8", errors="replace")

    raw_sizes = parse_shell_assignment(script_text, "SIZES")
    sizes = [int(value) for value in parse_shell_words(raw_sizes or " ".join(str(size) for size in NETLIB_DEFAULT_SIZES))]
    if not sizes:
        sizes = list(NETLIB_DEFAULT_SIZES)

    sym_codes = parse_for_loop_words(script_text, "sym") or ["1", "0"]
    storage_codes = parse_for_loop_words(script_text, "stor") or ["1", "2"]
    prec_sets = parse_prec_sets(script_text)

    test_points: List[tuple[str, str, str]] = []
    for sym in sym_codes:
        method = OFFICIAL_SYM_CODES.get(sym)
        if method is None:
            continue
        for storage_code in storage_codes:
            storage = OFFICIAL_STORAGE_CODES.get(storage_code)
            if storage is None:
                continue
            for prec_code in prec_sets.get(storage_code, []):
                preconditioner = OFFICIAL_PREC_CODES.get(prec_code)
                if preconditioner is None:
                    continue
                test_points.append((method, storage, preconditioner))

    if not test_points:
        test_points = list(OFFICIAL_TEST_POINTS)
    return OfficialTestPlan(sizes=tuple(sizes), test_points=tuple(test_points), source=str(test_script))


def regular_logical_nnz(n1: int, n2: int, n3: int) -> int:
    diag = n1 * n2 * n3
    x_edges = (n1 - 1) * n2 * n3
    y_edges = n1 * (n2 - 1) * n3
    z_edges = n1 * n2 * (n3 - 1)
    return int(diag + 2 * (x_edges + y_edges + z_edges))


def parse_official_regular_matrix(path: Path, dtype: str, symmetry: str) -> OfficialMatrixMeta:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().split()
        if len(first) < 3:
            raise ValueError(f"invalid regular matrix header: {path}")
        n1, n2, n3 = (int(first[0]), int(first[1]), int(first[2]))
        n = n1 * n2 * n3
        first_values = handle.readline().split()
        if not first_values:
            raise ValueError(f"regular matrix has no values: {path}")
        stored_diagonals = len(first_values)
    if stored_diagonals not in {4, 7}:
        raise ValueError(f"unsupported regular diagonal count {stored_diagonals} in {path}")
    storage_values = n * stored_diagonals
    return OfficialMatrixMeta(
        path=path,
        storage="regular",
        symmetry=symmetry,
        size=n1,
        n=n,
        nnz=regular_logical_nnz(n1, n2, n3),
        storage_values=storage_values,
        matrix_size_bytes=storage_values * DTYPE_BYTES[dtype],
    )


def parse_official_crs_matrix(path: Path, dtype: str, symmetry: str) -> OfficialMatrixMeta:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        first = handle.readline().split()
        if len(first) < 2:
            raise ValueError(f"invalid CRS matrix header: {path}")
        n = int(first[0])
        nnz = int(first[1])
    size = round(n ** (1.0 / 3.0))
    return OfficialMatrixMeta(
        path=path,
        storage="crs",
        symmetry=symmetry,
        size=size,
        n=n,
        nnz=nnz,
        storage_values=nnz,
        matrix_size_bytes=nnz * (DTYPE_BYTES[dtype] + INDEX_BYTES) + (n + 1) * INDEX_BYTES,
    )


def parse_official_matrix(path: Path, storage: str, dtype: str, symmetry: str) -> OfficialMatrixMeta:
    if storage == "regular":
        return parse_official_regular_matrix(path, dtype, symmetry)
    if storage == "crs":
        return parse_official_crs_matrix(path, dtype, symmetry)
    raise ValueError(f"official matrix files only cover regular/crs storage, got {storage}")


def config_with_official_matrix(base: SparseBenchConfig, cfg: SparseBenchConfig, meta: OfficialMatrixMeta) -> SparseBenchConfig:
    return SparseBenchConfig(
        method=cfg.method,
        storage=cfg.storage,
        preconditioner=cfg.preconditioner,
        n=meta.n,
        iterations=base.iterations,
        restart=base.restart,
        nnz_per_row=max(1, int(round(meta.nnz / max(1, meta.n)))),
        num_diagonals=7 if meta.storage == "regular" else base.num_diagonals,
        dtype=base.dtype,
        dot_collective=base.dot_collective,
        emit_metadata=base.emit_metadata,
        matrix_nnz_override=meta.nnz,
        matrix_size_bytes_override=meta.matrix_size_bytes,
        official_size=meta.size,
        official_matrix_file=str(meta.path),
        official_symmetry=meta.symmetry,
    )


class WorkloadBuilder:
    def __init__(self, cfg: SparseBenchConfig) -> None:
        self.cfg = cfg
        self.tensors: List[Dict[str, Any]] = []
        self.tasks: List[Dict[str, Any]] = []
        self._task_id = 0
        self._tensor_ids: Set[str] = set()

    def add_tensor(
        self,
        tensor_id: str,
        size_bytes: int,
        *,
        producer: Optional[int] = None,
        dtype: Optional[str] = None,
        shape: Optional[Sequence[int]] = None,
        num_elements: Optional[int] = None,
        storage_format: Optional[str] = None,
        access_pattern: Optional[str] = None,
        collective_hint: Optional[Dict[str, str]] = None,
        replication: Optional[Dict[str, str]] = None,
    ) -> str:
        if tensor_id in self._tensor_ids:
            raise ValueError(f"duplicate tensor id: {tensor_id}")
        self._tensor_ids.add(tensor_id)
        payload: Dict[str, Any] = {
            "id": tensor_id,
            "name": tensor_id,
            "dtype": dtype or self.cfg.dtype,
            "size_bytes": int(max(0, size_bytes)),
            "producer": producer,
        }
        if shape is not None:
            payload["shape"] = [int(v) for v in shape]
        if num_elements is not None:
            payload["num_elements"] = int(num_elements)
        if storage_format is not None:
            payload["storage_format"] = storage_format
        if access_pattern is not None:
            payload["access_pattern"] = access_pattern
        if collective_hint is not None:
            payload["collective_hint"] = collective_hint
        if replication is not None:
            payload["replication"] = replication
        self.tensors.append(payload)
        return tensor_id

    def add_vector(self, tensor_id: str, *, producer: Optional[int] = None) -> str:
        return self.add_tensor(
            tensor_id,
            self.cfg.vector_size_bytes,
            producer=producer,
            shape=[self.cfg.n],
            num_elements=self.cfg.n,
            access_pattern="dense",
        )

    def add_scalar(
        self,
        tensor_id: str,
        *,
        producer: Optional[int] = None,
        collective: bool = False,
    ) -> str:
        return self.add_tensor(
            tensor_id,
            self.cfg.scalar_size_bytes,
            producer=producer,
            shape=[1],
            num_elements=1,
            access_pattern="dense",
            collective_hint={"type": "allreduce", "op": "sum"} if collective else None,
            replication={"mode": "broadcast"} if collective else None,
        )

    def add_task(
        self,
        name: str,
        op: str,
        compute_flops: float,
        inputs: Sequence[str | Dict[str, str]],
        outputs: Sequence[str],
        *,
        bytes_read: float,
        bytes_written: float,
        features: Optional[Iterable[str]] = None,
    ) -> int:
        task_id = self._task_id
        self._task_id += 1
        normalized_inputs: List[Dict[str, str]] = []
        for item in inputs:
            if isinstance(item, str):
                normalized_inputs.append({"tensor": item})
            else:
                normalized_inputs.append(dict(item))
        task: Dict[str, Any] = {
            "id": task_id,
            "name": name,
            "op": op,
            "compute_flops": float(max(1.0, compute_flops)),
            "bytes_read": int(max(0.0, bytes_read)),
            "bytes_written": int(max(0.0, bytes_written)),
            "inputs": normalized_inputs,
            "outputs": list(outputs),
        }
        if features:
            task["features"] = sorted(set(features))
        self.tasks.append(task)
        return task_id

    def mark_producer(self, tensor_id: str, producer: int) -> None:
        for tensor in self.tensors:
            if tensor["id"] == tensor_id:
                tensor["producer"] = producer
                return
        raise ValueError(f"tensor not found: {tensor_id}")

    def add_matrix_and_state(self) -> None:
        if self.cfg.storage == "crs":
            matrix_storage = "csr"
            matrix_access = "sparse_csr"
        else:
            # Netlib's regular storage is seven-point diagonal/stencil storage,
            # and DIA is the same access family for our lowered model.  Neither
            # is a first-class parser storage format today.  Keep CSR storage so
            # nnz survives taskflow/replay, but mark access as row-wise so mapper
            # cost/features do not pretend this is plain CRS.
            matrix_storage = "csr"
            matrix_access = "row-wise"

        self.add_tensor(
            "A",
            self.cfg.matrix_size_bytes,
            dtype=self.cfg.dtype,
            shape=[self.cfg.n, self.cfg.n],
            num_elements=self.cfg.matrix_nnz,
            storage_format=matrix_storage,
            access_pattern=matrix_access,
        )
        self.add_vector("x")
        self.add_vector("r")

        if self.cfg.method == "cg":
            self.add_vector("p")
            self.add_scalar("rz_old")
        else:
            self.add_scalar("beta0")
            self.add_vector("v0")

        if self.cfg.preconditioner in {"jacobi", "bjac"}:
            self.add_vector("M_diag_inv")
        elif self.cfg.preconditioner == "ilu":
            triangular_bytes = self.cfg.matrix_size_bytes
            self.add_tensor(
                "M_L",
                triangular_bytes,
                shape=[self.cfg.n, self.cfg.n],
                num_elements=max(1, self.cfg.matrix_nnz // 2),
                storage_format="csr",
                access_pattern="sparse_csr",
            )
            self.add_tensor(
                "M_U",
                triangular_bytes,
                shape=[self.cfg.n, self.cfg.n],
                num_elements=max(1, self.cfg.matrix_nnz // 2),
                storage_format="csr",
                access_pattern="sparse_csr",
            )

    def dot_collective(self) -> bool:
        return self.cfg.dot_collective == "allreduce"

    def spmv(self, name: str, vector: str, output: str) -> str:
        self.add_vector(output)
        matrix_input = {"tensor": "A", "role": "matrix"}
        if self.cfg.storage in {"dia", "regular"}:
            matrix_input["access"] = "row-wise"
        task_id = self.add_task(
            name,
            "spmv",
            2.0 * self.cfg.matrix_nnz,
            [matrix_input, {"tensor": vector, "role": "x"}],
            [output],
            bytes_read=self.cfg.matrix_size_bytes + self.cfg.vector_size_bytes,
            bytes_written=self.cfg.vector_size_bytes,
        )
        self.mark_producer(output, task_id)
        return output

    def dot(self, name: str, left: str, right: str, output: str) -> str:
        self.add_scalar(output, collective=self.dot_collective())
        task_id = self.add_task(
            name,
            "dot",
            2.0 * self.cfg.n,
            [left, right],
            [output],
            bytes_read=2.0 * self.cfg.vector_size_bytes,
            bytes_written=self.cfg.scalar_size_bytes,
        )
        self.mark_producer(output, task_id)
        return output

    def nrm2(self, name: str, vector: str, output: str) -> str:
        self.add_scalar(output, collective=self.dot_collective())
        task_id = self.add_task(
            name,
            "nrm2",
            2.0 * self.cfg.n,
            [vector],
            [output],
            bytes_read=self.cfg.vector_size_bytes,
            bytes_written=self.cfg.scalar_size_bytes,
        )
        self.mark_producer(output, task_id)
        return output

    def scalar(self, name: str, inputs: Sequence[str], output: str) -> str:
        self.add_scalar(output)
        task_id = self.add_task(
            name,
            "scal",
            8.0,
            list(inputs),
            [output],
            bytes_read=len(inputs) * self.cfg.scalar_size_bytes,
            bytes_written=self.cfg.scalar_size_bytes,
        )
        self.mark_producer(output, task_id)
        return output

    def axpy(self, name: str, base: str, direction: str, scalar: str, output: str) -> str:
        self.add_vector(output)
        task_id = self.add_task(
            name,
            "axpy",
            2.0 * self.cfg.n,
            [base, direction, scalar],
            [output],
            bytes_read=2.0 * self.cfg.vector_size_bytes + self.cfg.scalar_size_bytes,
            bytes_written=self.cfg.vector_size_bytes,
        )
        self.mark_producer(output, task_id)
        return output

    def scale_vector(self, name: str, vector: str, output: str, extra_input: Optional[str] = None) -> str:
        self.add_vector(output)
        inputs = [vector]
        if extra_input is not None:
            inputs.append(extra_input)
        task_id = self.add_task(
            name,
            "scal",
            1.0 * self.cfg.n,
            inputs,
            [output],
            bytes_read=len(inputs) * self.cfg.vector_size_bytes,
            bytes_written=self.cfg.vector_size_bytes,
        )
        self.mark_producer(output, task_id)
        return output

    def copy_vector(self, name: str, vector: str, output: str) -> str:
        self.add_vector(output)
        task_id = self.add_task(
            name,
            "copy",
            1.0 * self.cfg.n,
            [vector],
            [output],
            bytes_read=self.cfg.vector_size_bytes,
            bytes_written=self.cfg.vector_size_bytes,
        )
        self.mark_producer(output, task_id)
        return output

    def apply_preconditioner(self, suffix: str, residual_or_basis: str) -> str:
        if self.cfg.preconditioner == "none":
            return residual_or_basis
        if self.cfg.preconditioner == "jacobi":
            return self.scale_vector(
                f"jacobi_apply{suffix}",
                residual_or_basis,
                f"z{suffix}",
                extra_input="M_diag_inv",
            )
        if self.cfg.preconditioner == "bjac":
            y = self.add_vector(f"bjac_y{suffix}")
            forward_id = self.add_task(
                f"bjac_forward_solve{suffix}",
                "trsv",
                2.0 * self.cfg.matrix_nnz,
                [
                    {"tensor": "A", "role": "regular_block_factor", "access": "row-wise"},
                    {"tensor": residual_or_basis, "role": "rhs"},
                    {"tensor": "M_diag_inv", "role": "diagonal_inverse"},
                ],
                [y],
                bytes_read=self.cfg.matrix_size_bytes + 2.0 * self.cfg.vector_size_bytes,
                bytes_written=self.cfg.vector_size_bytes,
                features={"sparse_linear_algebra", "irregular_access", "triangular_dependency"},
            )
            self.mark_producer(y, forward_id)

            z = self.add_vector(f"z{suffix}")
            backward_id = self.add_task(
                f"bjac_backward_solve{suffix}",
                "trsv",
                2.0 * self.cfg.matrix_nnz,
                [
                    {"tensor": "A", "role": "regular_block_factor", "access": "row-wise"},
                    {"tensor": y, "role": "rhs"},
                    {"tensor": "M_diag_inv", "role": "diagonal_inverse"},
                ],
                [z],
                bytes_read=self.cfg.matrix_size_bytes + 2.0 * self.cfg.vector_size_bytes,
                bytes_written=self.cfg.vector_size_bytes,
                features={"sparse_linear_algebra", "irregular_access", "triangular_dependency"},
            )
            self.mark_producer(z, backward_id)
            return z

        y = self.add_vector(f"ilu_y{suffix}")
        lower_id = self.add_task(
            f"ilu_lower_solve{suffix}",
            "trsv",
            2.0 * self.cfg.matrix_nnz,
            [
                {"tensor": "M_L", "role": "lower_factor", "access": "sparse_csr"},
                {"tensor": residual_or_basis, "role": "rhs"},
            ],
            [y],
            bytes_read=self.cfg.matrix_size_bytes + self.cfg.vector_size_bytes,
            bytes_written=self.cfg.vector_size_bytes,
            features={"sparse_linear_algebra", "irregular_access", "triangular_dependency"},
        )
        self.mark_producer(y, lower_id)

        z = self.add_vector(f"z{suffix}")
        upper_id = self.add_task(
            f"ilu_upper_solve{suffix}",
            "trsv",
            2.0 * self.cfg.matrix_nnz,
            [
                {"tensor": "M_U", "role": "upper_factor", "access": "sparse_csr"},
                {"tensor": y, "role": "rhs"},
            ],
            [z],
            bytes_read=self.cfg.matrix_size_bytes + self.cfg.vector_size_bytes,
            bytes_written=self.cfg.vector_size_bytes,
            features={"sparse_linear_algebra", "irregular_access", "triangular_dependency"},
        )
        self.mark_producer(z, upper_id)
        return z

    def build_cg(self) -> Dict[str, Any]:
        self.add_matrix_and_state()
        prev_x = "x"
        prev_r = "r"
        prev_p = "p"
        prev_rz = "rz_old"

        for k in range(self.cfg.iterations):
            suffix = f"_iter{k}"
            ap = self.spmv(f"spmv_Ap{suffix}", prev_p, f"Ap{suffix}")
            p_ap = self.dot(f"dot_pAp{suffix}", prev_p, ap, f"pAp{suffix}")
            alpha = self.scalar(f"scalar_alpha{suffix}", [prev_rz, p_ap], f"alpha{suffix}")
            x_next = self.axpy(f"axpy_x{suffix}", prev_x, prev_p, alpha, f"x{suffix}_next")
            r_next = self.axpy(f"axpy_r{suffix}", prev_r, ap, alpha, f"r{suffix}_next")
            z_next = self.apply_preconditioner(suffix, r_next)
            rz_next = self.dot(f"dot_rz{suffix}", r_next, z_next, f"rz{suffix}_next")
            beta = self.scalar(f"scalar_beta{suffix}", [rz_next, prev_rz], f"beta{suffix}")
            p_next = self.axpy(f"axpy_p{suffix}", z_next, prev_p, beta, f"p{suffix}_next")

            prev_x = x_next
            prev_r = r_next
            prev_p = p_next
            prev_rz = rz_next

        return self.finish(
            name=f"sparsebench_{self.cfg.method}_{self.cfg.storage}_{self.cfg.preconditioner}_n{self.cfg.n}_iter{self.cfg.iterations}",
            iteration_inputs=["A", "x", "r", "p", "rz_old"],
            iteration_outputs=[prev_x, prev_r, prev_p, prev_rz],
        )

    def build_gmres(self) -> Dict[str, Any]:
        self.add_matrix_and_state()
        prev_x = "x"
        first_basis = "v0"
        cycle_count = self.cfg.iterations

        for cycle in range(cycle_count):
            basis: List[str] = [first_basis]
            z_basis: List[str] = []
            h_scalars: List[str] = []
            suffix_cycle = f"_cycle{cycle}"

            for j in range(self.cfg.restart):
                suffix = f"{suffix_cycle}_j{j}"
                z_j = self.apply_preconditioner(suffix, basis[j])
                z_basis.append(z_j)
                w = self.spmv(f"spmv_arnoldi{suffix}", z_j, f"w{suffix}_0")

                for i in range(j + 1):
                    h_ij = self.dot(f"dot_h{suffix}_i{i}", w, basis[i], f"h{suffix}_i{i}")
                    h_scalars.append(h_ij)
                    w_next = self.axpy(f"axpy_orthogonalize{suffix}_i{i}", w, basis[i], h_ij, f"w{suffix}_{i + 1}")
                    w = w_next

                h_next = self.nrm2(f"nrm2_h{suffix}_next", w, f"h{suffix}_next")
                h_scalars.append(h_next)
                inv_h = self.scalar(f"scalar_inv_h{suffix}", [h_next], f"inv_h{suffix}")
                v_next = self.scale_vector(f"scal_basis{suffix}", w, f"v{suffix}_next", extra_input=inv_h)
                basis.append(v_next)

            y_coeffs = [self.add_scalar(f"y{suffix_cycle}_{j}") for j in range(self.cfg.restart)]
            solve_id = self.add_task(
                f"trsv_hessenberg{suffix_cycle}",
                "trsv",
                max(1.0, float(self.cfg.restart * self.cfg.restart)),
                [*h_scalars, "beta0"],
                y_coeffs,
                bytes_read=(len(h_scalars) + 1) * self.cfg.scalar_size_bytes,
                bytes_written=self.cfg.restart * self.cfg.value_bytes,
            )
            for coeff in y_coeffs:
                self.mark_producer(coeff, solve_id)

            x_acc = prev_x
            for j, (z_j, coeff) in enumerate(zip(z_basis, y_coeffs)):
                x_acc = self.axpy(f"axpy_gmres_update{suffix_cycle}_j{j}", x_acc, z_j, coeff, f"x{suffix_cycle}_upd{j}")

            # Compute a fresh residual and next restart vector.  This keeps cycles
            # connected without modelling convergence/early stop.
            ax = self.spmv(f"spmv_residual{suffix_cycle}", x_acc, f"Ax{suffix_cycle}")
            r_next = self.axpy(f"axpy_residual{suffix_cycle}", "r", ax, "beta0", f"r{suffix_cycle}_next")
            beta = self.nrm2(f"nrm2_residual{suffix_cycle}", r_next, f"beta{suffix_cycle}_next")
            inv_beta = self.scalar(f"scalar_inv_beta{suffix_cycle}", [beta], f"inv_beta{suffix_cycle}")
            first_basis = self.scale_vector(f"scal_restart_basis{suffix_cycle}", r_next, f"v0{suffix_cycle}_next", extra_input=inv_beta)
            prev_x = x_acc

        return self.finish(
            name=f"sparsebench_{self.cfg.method}{self.cfg.restart}_{self.cfg.storage}_{self.cfg.preconditioner}_n{self.cfg.n}_cycles{cycle_count}",
            iteration_inputs=["A", "x", "r", "beta0", "v0"],
            iteration_outputs=[prev_x, first_basis],
        )

    def finish(self, *, name: str, iteration_inputs: Sequence[str], iteration_outputs: Sequence[str]) -> Dict[str, Any]:
        workload: Dict[str, Any] = {
            "schema": "mapper.workload.v2",
            "name": name,
            "version": 2,
            "device_groups": [{"id": "all_devices", "members": "all"}],
            "iteration_inputs": list(iteration_inputs),
            "iteration_outputs": list(iteration_outputs),
            "tensors": self.tensors,
            "tasks": self.tasks,
        }
        if self.cfg.emit_metadata:
            workload["metadata"] = {
                "benchmark": "SparseBench",
                "method": self.cfg.method,
                "storage": self.cfg.storage,
                "preconditioner": self.cfg.preconditioner,
                "n": self.cfg.n,
                "iterations": self.cfg.iterations,
                "restart": self.cfg.restart if self.cfg.method == "gmres" else None,
                "nnz": self.cfg.matrix_nnz,
                "dtype": self.cfg.dtype,
                "dot_collective": self.cfg.dot_collective,
                "official_size": self.cfg.official_size,
                "official_matrix_file": self.cfg.official_matrix_file,
                "official_symmetry": self.cfg.official_symmetry,
                "lowering": {
                    "crs": "spmv with CSR storage/access",
                    "regular": "Netlib regular seven-point storage lowered to spmv with CSR-compatible nnz metadata and row-wise access",
                    "dia": "spmv with CSR-compatible nnz metadata and row-wise access; DIA is not a first-class parser storage format yet",
                    "jacobi": "scal over M_diag_inv and the input vector",
                    "ilu": "two sparse-access trsv tasks for lower and upper triangular applies; ILU factorization setup is not included",
                    "bjac": "regular-storage block Jacobi solve lowered to forward/backward trsv tasks over row-wise regular blocks",
                },
                "note": "Static DAG; tolerance/convergence and GMRES restart control are metadata/modeling choices, not runtime branches.",
            }
        WorkloadValidator.validate(workload)
        return workload


class WorkloadValidator:
    @staticmethod
    def validate(workload: Dict[str, Any]) -> None:
        tensors = workload.get("tensors", [])
        tasks = workload.get("tasks", [])
        tensor_by_id: Dict[str, Dict[str, Any]] = {}
        task_by_id: Dict[int, Dict[str, Any]] = {}

        for tensor in tensors:
            tid = tensor.get("id")
            if not isinstance(tid, str) or not tid:
                raise ValueError(f"invalid tensor id: {tensor}")
            if tid in tensor_by_id:
                raise ValueError(f"duplicate tensor id: {tid}")
            if int(tensor.get("size_bytes", -1)) < 0:
                raise ValueError(f"negative tensor size: {tid}")
            tensor_by_id[tid] = tensor

        task_names: Set[str] = set()
        for task in tasks:
            task_id = task.get("id")
            name = task.get("name")
            if not isinstance(task_id, int):
                raise ValueError(f"invalid task id: {task}")
            if task_id in task_by_id:
                raise ValueError(f"duplicate task id: {task_id}")
            if not isinstance(name, str) or not name:
                raise ValueError(f"invalid task name: {task}")
            if name in task_names:
                raise ValueError(f"duplicate task name: {name}")
            task_names.add(name)
            task_by_id[task_id] = task
            if float(task.get("compute_flops", 0.0)) <= 0.0:
                raise ValueError(f"non-positive compute_flops: {name}")
            if float(task.get("bytes_read", 0.0)) + float(task.get("bytes_written", 0.0)) <= 0.0:
                raise ValueError(f"non-positive task memory bytes: {name}")
            for item in task.get("inputs", []):
                tensor_id = item.get("tensor")
                if tensor_id not in tensor_by_id:
                    raise ValueError(f"task input tensor not found: {name} <- {tensor_id}")
            for output_id in task.get("outputs", []):
                if output_id not in tensor_by_id:
                    raise ValueError(f"task output tensor not found: {name} -> {output_id}")
                if tensor_by_id[output_id].get("producer") != task_id:
                    raise ValueError(f"producer mismatch: {name} -> {output_id}")

        for tensor in tensors:
            producer = tensor.get("producer")
            if producer is None:
                continue
            if producer not in task_by_id:
                raise ValueError(f"tensor producer task not found: {tensor['id']} -> {producer}")
            if tensor["id"] not in task_by_id[producer].get("outputs", []):
                raise ValueError(f"producer task does not list tensor output: {tensor['id']} -> {producer}")


def write_json(data: Dict[str, Any], path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_workload(cfg: SparseBenchConfig) -> Dict[str, Any]:
    cfg.validate()
    builder = WorkloadBuilder(cfg)
    if cfg.method == "cg":
        return builder.build_cg()
    return builder.build_gmres()


def workload_filename(cfg: SparseBenchConfig) -> str:
    method = cfg.method if cfg.method == "cg" else f"gmres{cfg.restart}"
    iter_part = "iter" if cfg.method == "cg" else "cycles"
    return f"sparsebench_{method}_{cfg.storage}_{cfg.preconditioner}_n{cfg.n}_{iter_part}{cfg.iterations}.json"


def netlib_default_configs(
    base: SparseBenchConfig,
    test_points: Optional[Sequence[tuple[str, str, str]]] = None,
) -> List[SparseBenchConfig]:
    configs: List[SparseBenchConfig] = []
    for method, storage, preconditioner in test_points or OFFICIAL_TEST_POINTS:
        configs.append(
            SparseBenchConfig(
                method=method,
                storage=storage,
                preconditioner=preconditioner,
                n=base.n,
                iterations=base.iterations,
                restart=base.restart,
                nnz_per_row=base.nnz_per_row,
                num_diagonals=base.num_diagonals,
                dtype=base.dtype,
                dot_collective=base.dot_collective,
                emit_metadata=base.emit_metadata,
            )
        )
    return configs


def official_eight_configs(base: SparseBenchConfig) -> List[SparseBenchConfig]:
    configs: List[SparseBenchConfig] = []
    for method in OFFICIAL_METHODS:
        for storage in ("crs", "regular"):
            for preconditioner in ("jacobi", "ilu"):
                configs.append(
                    SparseBenchConfig(
                        method=method,
                        storage=storage,
                        preconditioner=preconditioner,
                        n=base.n,
                        iterations=base.iterations,
                        restart=base.restart,
                        nnz_per_row=base.nnz_per_row,
                        num_diagonals=base.num_diagonals,
                        dtype=base.dtype,
                        dot_collective=base.dot_collective,
                        emit_metadata=base.emit_metadata,
                    )
                )
    return configs


def missing_official_tools(root: Path) -> List[str]:
    missing: List[str] = []
    for name in OFFICIAL_TOOL_NAMES:
        path = root / name
        if not (path.exists() and os.access(path, os.X_OK)):
            missing.append(name)
    return missing


def build_official_tools(
    sparsebench_root: str,
    *,
    mach: str = "codex_sparsebench",
    fc: Optional[str] = None,
    cc: Optional[str] = None,
    f_extra_flags: str = "-std=legacy -fallow-argument-mismatch",
    c_opt_flags: str = "-O -Wno-implicit-function-declaration",
) -> None:
    root = Path(sparsebench_root).resolve()
    if not (root / "Makefile").exists():
        raise ValueError(f"official SparseBench Makefile not found: {root / 'Makefile'}")

    fc = fc or os.environ.get("FC") or "gfortran"
    cc = cc or os.environ.get("CC") or "clang"
    if shutil.which(fc) is None:
        raise ValueError(f"Fortran compiler not found: {fc}. Install gfortran or pass --official-fc")
    if shutil.which(cc) is None:
        raise ValueError(f"C compiler not found: {cc}. Install clang/gcc or pass --official-cc")

    # Netlib SparseBench predates modern macOS defaults:
    # * case-insensitive filesystems make `make install` collide with `Install`, so force the target;
    # * GNU make's default FC is f77, which is usually absent;
    # * clang rejects C89-style implicit declarations unless explicitly allowed.
    cmd = [
        "make",
        "-B",
        f"MACH={mach}",
        "PLAT=default_platform",
        "OPT=reference",
        f"FC={fc}",
        f"CC={cc}",
        f"F_EXTRA_FLAGS={f_extra_flags}",
        f"C_OPT_FLAGS={c_opt_flags}",
        "install",
    ]
    subprocess.run(cmd, cwd=root, check=True)

    missing = missing_official_tools(root)
    if missing:
        raise ValueError(f"official SparseBench build finished but tools are missing: {', '.join(missing)}")


def maybe_generate_official_matrices(
    sparsebench_root: Optional[str],
    sizes: Sequence[int],
    *,
    build_tools: bool = True,
    build_mach: str = "codex_sparsebench",
    fc: Optional[str] = None,
    cc: Optional[str] = None,
) -> None:
    if not sparsebench_root:
        return
    root = Path(sparsebench_root).resolve()
    generate = root / "Scripts" / "generate"
    if not generate.exists():
        raise ValueError(f"official SparseBench Scripts/generate not found: {generate}")
    missing = missing_official_tools(root)
    if missing:
        if not build_tools:
            raise ValueError(f"official SparseBench tools are missing: {', '.join(missing)}")
        build_official_tools(str(root), mach=build_mach, fc=fc, cc=cc)
    cmd = [str(generate), *[str(size) for size in sizes]]
    subprocess.run(cmd, cwd=root, check=True)


def build_suite(
    base: SparseBenchConfig,
    out_dir: str,
    suite: str,
    *,
    official_matrix_dir: Optional[str] = None,
    sizes: Optional[Sequence[int]] = None,
    official_test_plan: Optional[OfficialTestPlan] = None,
) -> Dict[str, Any]:
    if suite == "netlib-default":
        configs = netlib_default_configs(
            base,
            test_points=official_test_plan.test_points if official_test_plan is not None else None,
        )
        description = (
            "Netlib SparseBench Test script default combinations: CG/GMRES over "
            "regular and CRS storage, with none/ILU everywhere plus regular-only BJAC."
        )
    elif suite == "official-eight":
        configs = official_eight_configs(base)
        description = (
            "Compatibility suite for the earlier 2x2x2 method/storage/preconditioner view. "
            "Netlib's default Test script is netlib-default, not this suite."
        )
    else:
        raise ValueError(f"unsupported suite: {suite}")

    entries: List[Dict[str, Any]] = []
    os.makedirs(out_dir, exist_ok=True)
    matrix_dir = Path(official_matrix_dir).resolve() if official_matrix_dir else None
    default_sizes = list(official_test_plan.sizes) if official_test_plan is not None else list(NETLIB_DEFAULT_SIZES)
    suite_sizes = list(sizes or ([] if matrix_dir is None else default_sizes))
    if matrix_dir is None:
        matrix_cases: List[tuple[SparseBenchConfig, Optional[OfficialMatrixMeta]]] = [(cfg, None) for cfg in configs]
    else:
        matrix_cases = []
        for cfg in configs:
            if cfg.storage not in {"regular", "crs"}:
                raise ValueError(f"official matrix files do not cover storage={cfg.storage}")
            symmetry = "symmetric" if cfg.method == "cg" else "unsymmetric"
            for size in suite_sizes:
                matrix_path = matrix_dir / official_matrix_filename(cfg.storage, int(size), symmetry)
                if not matrix_path.exists():
                    raise ValueError(f"official matrix file not found: {matrix_path}")
                meta = parse_official_matrix(matrix_path, cfg.storage, base.dtype, symmetry)
                matrix_cases.append((config_with_official_matrix(base, cfg, meta), meta))

    for cfg, meta in matrix_cases:
        workload = build_workload(cfg)
        filename = workload_filename(cfg)
        if cfg.official_size is not None:
            filename = filename.replace(f"_n{cfg.n}_", f"_size{cfg.official_size}_n{cfg.n}_")
        path = os.path.join(out_dir, filename)
        write_json(workload, path)
        entries.append(
            {
                "method": cfg.method,
                "storage": cfg.storage,
                "preconditioner": cfg.preconditioner,
                "official_size": cfg.official_size,
                "matrix_file": str(meta.path) if meta else None,
                "matrix_n": cfg.n,
                "matrix_nnz": cfg.matrix_nnz,
                "matrix_size_bytes": cfg.matrix_size_bytes,
                "path": filename,
                "tasks": len(workload["tasks"]),
                "tensors": len(workload["tensors"]),
                "approximations": [
                    f"{cfg.storage} SpMV lowered to spmv with row-wise access" if cfg.storage in {"dia", "regular"} else "CRS SpMV uses CSR storage/access",
                    "ILU apply lowered to two sparse-access trsv tasks; ILU(0) setup is not included"
                    if cfg.preconditioner == "ilu"
                    else "BJAC apply lowered to forward/backward trsv over row-wise regular blocks"
                    if cfg.preconditioner == "bjac"
                    else "Jacobi apply lowered to scal over M_diag_inv"
                    if cfg.preconditioner == "jacobi"
                    else "No preconditioner apply task",
                ],
            }
        )

    manifest = {
        "benchmark": "SparseBench",
        "suite": suite,
        "description": description,
        "source_checked": {
            "url": "https://www.netlib.org/benchmark/sparsebench/benchmark.tgz",
            "script": official_test_plan.source if official_test_plan is not None else "SparseBench/Test",
            "preconditioner_codes": {"0": "none", "1": "jacobi", "2": "ilu", "3": "bjac"},
            "default_prec_sets": {"regular": ["none", "ilu", "bjac"], "crs": ["none", "ilu"]},
            "test_points": [
                {"method": method, "storage": storage, "preconditioner": preconditioner}
                for method, storage, preconditioner in (official_test_plan.test_points if official_test_plan is not None else OFFICIAL_TEST_POINTS)
            ],
        },
        "parameters": {
            "n": base.n,
            "iterations": base.iterations,
            "restart": base.restart,
            "nnz_per_row": base.nnz_per_row,
            "num_diagonals": base.num_diagonals,
            "dtype": base.dtype,
            "dot_collective": base.dot_collective,
        },
        "matrix_source": "netlib-generated-files" if matrix_dir is not None else "parameter-estimate",
        "official_matrix_dir": str(matrix_dir) if matrix_dir is not None else None,
        "sizes": suite_sizes if suite_sizes else None,
        "entries": entries,
    }
    write_json(manifest, os.path.join(out_dir, "manifest.json"))
    return manifest


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SparseBench-style mapper workload JSON")
    parser.add_argument(
        "--suite",
        choices=["single", "netlib-default", "official-eight"],
        default="single",
        help="Generate one workload, Netlib's default Test-script suite, or the older 2x2x2 compatibility suite",
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="cg", help="Sparse iterative method")
    parser.add_argument("--storage", choices=sorted(STORAGE_FORMATS), default="crs", help="Sparse matrix storage family")
    parser.add_argument("--preconditioner", choices=sorted(PRECONDITIONERS), default="none", help="Preconditioner model")
    parser.add_argument("--n", type=int, default=100_000, help="Matrix/vector dimension")
    parser.add_argument("--iterations", type=int, default=20, help="CG iterations or GMRES restart cycles")
    parser.add_argument("--restart", type=int, default=20, help="GMRES restart length")
    parser.add_argument("--nnz-per-row", type=int, default=7, help="CRS average nonzeros per row")
    parser.add_argument("--num-diagonals", type=int, default=7, help="regular/DIA diagonal count")
    parser.add_argument("--dtype", choices=sorted(DTYPE_BYTES), default="fp64", help="Floating point dtype")
    parser.add_argument(
        "--dot-collective",
        choices=sorted(DOT_COLLECTIVES),
        default="none",
        help="Emit allreduce hints for dot/nrm2 outputs",
    )
    parser.add_argument(
        "--official-matrix-dir",
        default=None,
        help="Directory containing Netlib-generated regmat*/crsmat* files; when set, suite workloads use exact official N/nnz metadata",
    )
    parser.add_argument(
        "--sparsebench-root",
        default=None,
        help="Netlib SparseBench root; used with --generate-official-matrices and as matrix dir if --official-matrix-dir is omitted",
    )
    parser.add_argument(
        "--generate-official-matrices",
        action="store_true",
        help="Build missing SparseBench tools if needed, then run SparseBench/Scripts/generate for selected --sizes",
    )
    parser.add_argument(
        "--no-build-official-tools",
        action="store_true",
        help="Do not auto-build missing SparseBench Fortran tools before --generate-official-matrices",
    )
    parser.add_argument(
        "--official-build-mach",
        default="codex_sparsebench",
        help="MACH name used for Netlib SparseBench make object directory",
    )
    parser.add_argument(
        "--official-fc",
        default=None,
        help="Fortran compiler for Netlib SparseBench tools; default is FC env or gfortran",
    )
    parser.add_argument(
        "--official-cc",
        default=None,
        help="C compiler for Netlib SparseBench tools; default is CC env or clang",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=None,
        help="Official SparseBench domain side lengths; default is Netlib's 12 14 16 18 20 24 28 32 36 38 when reading official files",
    )
    parser.add_argument("--no-metadata", action="store_true", help="Do not emit SparseBench lowering metadata")
    parser.add_argument("--out", default="sparsebench_workload.json", help="Output workload JSON path")
    parser.add_argument("--out-dir", default="sparsebench_netlib_default", help="Output directory for suite generation")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cfg = SparseBenchConfig(
        method=args.method,
        storage=args.storage,
        preconditioner=args.preconditioner,
        n=args.n,
        iterations=args.iterations,
        restart=args.restart,
        nnz_per_row=args.nnz_per_row,
        num_diagonals=args.num_diagonals,
        dtype=args.dtype,
        dot_collective=args.dot_collective,
        emit_metadata=not args.no_metadata,
    )
    if args.suite != "single":
        official_matrix_dir = args.official_matrix_dir
        official_test_plan: Optional[OfficialTestPlan] = None
        if args.sparsebench_root is not None and args.suite == "netlib-default":
            try:
                official_test_plan = parse_official_test_plan(args.sparsebench_root)
            except ValueError as exc:
                print(f"official Test parsing error: {exc}", file=sys.stderr)
                sys.exit(2)
        if official_matrix_dir is None and args.sparsebench_root is not None:
            official_matrix_dir = args.sparsebench_root
        sizes = args.sizes
        if args.generate_official_matrices:
            if args.sparsebench_root is None:
                print("--generate-official-matrices requires --sparsebench-root", file=sys.stderr)
                sys.exit(2)
            if sizes is None:
                sizes = list(official_test_plan.sizes) if official_test_plan is not None else list(NETLIB_DEFAULT_SIZES)
            try:
                maybe_generate_official_matrices(
                    args.sparsebench_root,
                    sizes,
                    build_tools=not args.no_build_official_tools,
                    build_mach=args.official_build_mach,
                    fc=args.official_fc,
                    cc=args.official_cc,
                )
            except (ValueError, subprocess.CalledProcessError) as exc:
                print(f"official matrix generation error: {exc}", file=sys.stderr)
                sys.exit(2)
            official_matrix_dir = args.sparsebench_root
        try:
            manifest = build_suite(
                cfg,
                args.out_dir,
                args.suite,
                official_matrix_dir=official_matrix_dir,
                sizes=sizes,
                official_test_plan=official_test_plan,
            )
        except ValueError as exc:
            print(f"configuration error: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"wrote SparseBench suite: {os.path.abspath(args.out_dir)}")
        print(f"test_points={len(manifest['entries'])}, manifest={os.path.abspath(os.path.join(args.out_dir, 'manifest.json'))}")
        for entry in manifest["entries"]:
            print(f"  {entry['method']}/{entry['storage']}/{entry['preconditioner']}: {entry['path']}")
        return

    try:
        workload = build_workload(cfg)
    except ValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        sys.exit(2)

    write_json(workload, args.out)
    total_flops = sum(float(task.get("compute_flops", 0.0)) for task in workload["tasks"])
    total_memory = sum(float(task.get("bytes_read", 0.0)) + float(task.get("bytes_written", 0.0)) for task in workload["tasks"])
    print(f"wrote workload: {os.path.abspath(args.out)}")
    print(f"tasks={len(workload['tasks'])}, tensors={len(workload['tensors'])}")
    print(f"total_compute_flops={total_flops:.3e}, total_memory_bytes={total_memory:.3e}")


if __name__ == "__main__":
    main()
