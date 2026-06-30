#!/usr/bin/env python3
"""Prepare Netlib SparseBench matrices and mapper workloads.

This is the reusable official-toolchain path:

  Netlib SparseBench source -> build official tools -> Scripts/generate sizes
  -> parse official Test script -> mapper workload JSON suite.
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable, Optional


TOOL_DIR = Path(__file__).resolve().parent
MAPPER_ROOT = TOOL_DIR.parents[1]
GENERATOR_DIR = MAPPER_ROOT / "workload_sparse" / "generators"
sys.path.insert(0, str(GENERATOR_DIR))

import sparsebench_workload_generator as sparsebench  # noqa: E402


DEFAULT_URL = "https://www.netlib.org/benchmark/sparsebench/benchmark.tgz"
DEFAULT_OUT_DIR = MAPPER_ROOT / "workload_sparse" / "generated" / "sparsebench" / "official_all_sizes"
DEFAULT_WORK_DIR = MAPPER_ROOT / "workload_sparse" / "generated" / "sparsebench" / "_official_work"


def safe_extract(tarball: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball, "r:*") as archive:
        dest_resolved = dest.resolve()
        for member in archive.getmembers():
            target = (dest / member.name).resolve()
            if not str(target).startswith(str(dest_resolved) + os.sep) and target != dest_resolved:
                raise ValueError(f"tarball member escapes destination: {member.name}")
        archive.extractall(dest)


def find_sparsebench_root(search_dir: Path) -> Optional[Path]:
    candidates = []
    for path in search_dir.rglob("Test"):
        root = path.parent
        if (root / "Scripts" / "generate").exists() and (root / "Makefile").exists():
            candidates.append(root)
    if not candidates:
        return None
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def download_tarball(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        handle.write(response.read())
    return dest


def resolve_sparsebench_root(args: argparse.Namespace) -> Path:
    if args.sparsebench_root:
        root = Path(args.sparsebench_root).expanduser().resolve()
        if not (root / "Test").exists():
            raise ValueError(f"SparseBench Test script not found under --sparsebench-root: {root}")
        return root

    work_dir = Path(args.work_dir).expanduser().resolve()
    source_dir = work_dir / "source"
    tarball = Path(args.tarball).expanduser().resolve() if args.tarball else None

    if tarball is None and args.download:
        tarball = download_tarball(args.url, work_dir / "benchmark.tgz")

    if tarball is None:
        existing = find_sparsebench_root(source_dir) if source_dir.exists() else None
        if existing is not None:
            return existing
        raise ValueError("provide --sparsebench-root, --tarball, or --download")

    safe_extract(tarball, source_dir)
    root = find_sparsebench_root(source_dir)
    if root is None:
        raise ValueError(f"unable to find SparseBench root after extracting {tarball}")
    return root


def parse_sizes(raw_sizes: Optional[Iterable[int]], official_sizes: Iterable[int]) -> list[int]:
    sizes = [int(size) for size in raw_sizes or []]
    return sizes or [int(size) for size in official_sizes]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official Netlib SparseBench -> mapper workload flow")
    parser.add_argument("--sparsebench-root", help="Existing Netlib SparseBench source root")
    parser.add_argument("--tarball", help="Existing Netlib benchmark.tgz to extract and use")
    parser.add_argument("--download", action="store_true", help="Download Netlib SparseBench benchmark.tgz")
    parser.add_argument("--url", default=DEFAULT_URL, help="SparseBench tarball URL used with --download")
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="Working directory for downloaded/extracted official source")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for mapper workload suite")
    parser.add_argument("--sizes", type=int, nargs="*", default=None, help="SparseBench domain sizes; defaults to official Test SIZES")
    parser.add_argument("--iterations", type=int, default=20, help="CG iterations or GMRES cycles to unroll")
    parser.add_argument("--restart", type=int, default=20, help="GMRES restart length")
    parser.add_argument("--dtype", choices=sorted(sparsebench.DTYPE_BYTES), default="fp64")
    parser.add_argument("--dot-collective", choices=sorted(sparsebench.DOT_COLLECTIVES), default="allreduce")
    parser.add_argument("--no-build-official-tools", action="store_true", help="Require official tools to already exist")
    parser.add_argument("--official-build-mach", default="codex_sparsebench")
    parser.add_argument("--official-fc", default=None, help="Fortran compiler; default is FC env or gfortran")
    parser.add_argument("--official-cc", default=None, help="C compiler; default is CC env or clang")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        root = resolve_sparsebench_root(args)
        plan = sparsebench.parse_official_test_plan(str(root))
        sizes = parse_sizes(args.sizes, plan.sizes)
        sparsebench.maybe_generate_official_matrices(
            str(root),
            sizes,
            build_tools=not args.no_build_official_tools,
            build_mach=args.official_build_mach,
            fc=args.official_fc,
            cc=args.official_cc,
        )
        cfg = sparsebench.SparseBenchConfig(
            method="cg",
            storage="crs",
            preconditioner="none",
            n=100_000,
            iterations=args.iterations,
            restart=args.restart,
            nnz_per_row=7,
            num_diagonals=7,
            dtype=args.dtype,
            dot_collective=args.dot_collective,
            emit_metadata=True,
        )
        manifest = sparsebench.build_suite(
            cfg,
            args.out_dir,
            "netlib-default",
            official_matrix_dir=str(root),
            sizes=sizes,
            official_test_plan=plan,
        )
    except (ValueError, OSError, tarfile.TarError, urllib.error.URLError) as exc:
        print(f"prepare SparseBench official flow failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    manifest_path = Path(args.out_dir).resolve() / "manifest.json"
    print(f"SparseBench root: {root}")
    print(f"Official Test: {plan.source}")
    print(f"Sizes: {' '.join(str(size) for size in sizes)}")
    print(f"Generated workloads: {len(manifest['entries'])}")
    print(f"Output: {Path(args.out_dir).resolve()}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
