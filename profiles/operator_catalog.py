from __future__ import annotations

from .enums import OperatorKind


SUPPORTED_OPERATOR_KINDS: frozenset[OperatorKind] = frozenset(OperatorKind)
SUPPORTED_OPERATOR_NAMES: frozenset[str] = frozenset(kind.value for kind in OperatorKind)

_ALIASES = {
    "2_norm": OperatorKind.NRM2,
    "2norm": OperatorKind.NRM2,
    "l2_norm": OperatorKind.NRM2,
    "norm": OperatorKind.NRM2,
    "norm2": OperatorKind.NRM2,
    "scale": OperatorKind.SCAL,
    "scalar": OperatorKind.SCAL,
    "scalar_add": OperatorKind.SCAL,
    "scalar_div": OperatorKind.SCAL,
    "scalar_mul": OperatorKind.SCAL,
    "gemv": OperatorKind.MV,
    "matvec": OperatorKind.MV,
    "triangular_solve": OperatorKind.TRSV,
    "sp_trsv": OperatorKind.SPTRSV,
    "sparse_trsv": OperatorKind.SPTRSV,
    "sp_triangular_solve": OperatorKind.SPTRSV,
    "sparse_triangular_solve": OperatorKind.SPTRSV,
    "matmul": OperatorKind.GEMM,
    "mm": OperatorKind.GEMM,
    "bmm": OperatorKind.GEMM,
    "gemm_update": OperatorKind.GEMM,
    "syrk_gemm": OperatorKind.GEMM,
    "schur_update": OperatorKind.GEMM,
    "update": OperatorKind.GEMM,
    "sp_gemm": OperatorKind.SPGEMM,
    "sparse_gemm": OperatorKind.SPGEMM,
    "supernode_factor": OperatorKind.POTRF,
    "factor": OperatorKind.POTRF,
    "amd_ordering": OperatorKind.ORDER,
    "colamd_ordering": OperatorKind.ORDER,
    "metis_ordering": OperatorKind.ORDER,
    "natural_ordering": OperatorKind.ORDER,
    "nesdis_ordering": OperatorKind.ORDER,
    "ordering": OperatorKind.ORDER,
    "symbolic_order": OperatorKind.ORDER,
    "symbolic_etree": OperatorKind.ETREE,
    "elimination_tree": OperatorKind.ETREE,
    "column_counts": OperatorKind.COLCOUNT,
    "symbolic_pattern": OperatorKind.SYMBOLIC,
    "symbolic_dep_build": OperatorKind.SYMBOLIC,
    "symbolic_supernode_detect": OperatorKind.SYMBOLIC,
    "supernode_relax": OperatorKind.SUPERNODE_PARTITION,
    "comm_send_node": OperatorKind.SEND,
    "comm_recv_node": OperatorKind.RECV,
    "comm_coll_node": OperatorKind.ALLREDUCE,
    "all_reduce": OperatorKind.ALLREDUCE,
    "all_gather": OperatorKind.ALLREDUCE,
    "allgather": OperatorKind.ALLREDUCE,
    "reduce_scatter": OperatorKind.ALLREDUCE,
    "reducescatter": OperatorKind.ALLREDUCE,
    "broadcast": OperatorKind.ALLREDUCE,
    "all_to_all": OperatorKind.ALLREDUCE,
    "alltoall": OperatorKind.ALLREDUCE,
}


def normalize_operator_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def canonical_operator_kind(name: str | OperatorKind) -> OperatorKind | None:
    if isinstance(name, OperatorKind):
        return name
    normalized = normalize_operator_name(name)
    if normalized in SUPPORTED_OPERATOR_NAMES:
        return OperatorKind(normalized)
    return _ALIASES.get(normalized)


def canonical_operator_name(name: str | OperatorKind) -> str | None:
    kind = canonical_operator_kind(name)
    return kind.value if kind is not None else None


def require_operator_kind(name: str | OperatorKind) -> OperatorKind:
    kind = canonical_operator_kind(name)
    if kind is None:
        supported = ", ".join(sorted(SUPPORTED_OPERATOR_NAMES))
        raise ValueError(f"Unsupported operator '{name}'. Supported operators: {supported}")
    return kind


def is_supported_operator(name: str | OperatorKind) -> bool:
    return canonical_operator_kind(name) is not None
