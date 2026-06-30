from __future__ import annotations

from .enums import AccessPattern, OpCategory, OperatorKind, ParallelismType, Precision, Residency
from .operator_catalog import require_operator_kind
from .operator_profile import OperatorProfile, TensorShape


def precision_nbytes(precision: Precision) -> int:
    return {
        Precision.FP64: 8,
        Precision.FP32: 4,
        Precision.TF32: 4,
        Precision.FP16: 2,
        Precision.BF16: 2,
        Precision.INT8: 1,
        Precision.INT32: 4,
        Precision.UNKNOWN: 4,
    }[precision]


def _shape(dims: tuple[int | None, ...] | list[int | None]) -> TensorShape:
    return TensorShape(tuple(dims))


def _numel(shape: TensorShape) -> int:
    return shape.numel or 0


def _vector_profile(
    kind: OperatorKind,
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision,
    inputs: int,
    flops_per_element: float,
    input_residency: Residency,
) -> OperatorProfile:
    tensor_shape = _shape(shape)
    nbytes = precision_nbytes(precision)
    numel = _numel(tensor_shape)
    return OperatorProfile(
        name=name,
        op=kind,
        category=OpCategory.VECTOR,
        input_shapes=[tensor_shape] * inputs,
        output_shapes=[tensor_shape],
        precision=precision,
        flops=numel * flops_per_element,
        memory_read_bytes=numel * inputs * nbytes,
        memory_write_bytes=numel * nbytes,
        working_set_bytes=numel * (inputs + 1) * nbytes,
        access_pattern=AccessPattern.CONTIGUOUS,
        parallelism=ParallelismType.HIGH if numel >= 65536 else ParallelismType.LOW,
        is_fusable=True,
        input_residency=input_residency,
    )


def profile_scal(
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    return _vector_profile(OperatorKind.SCAL, name, shape, precision, 1, 1.0, input_residency)


def profile_copy(
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    return _vector_profile(OperatorKind.COPY, name, shape, precision, 1, 0.0, input_residency)


def profile_axpy(
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    return _vector_profile(OperatorKind.AXPY, name, shape, precision, 2, 2.0, input_residency)


def _reduction_profile(
    kind: OperatorKind,
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision,
    flops_per_element: float,
    input_residency: Residency,
) -> OperatorProfile:
    tensor_shape = _shape(shape)
    nbytes = precision_nbytes(precision)
    numel = _numel(tensor_shape)
    return OperatorProfile(
        name=name,
        op=kind,
        category=OpCategory.REDUCTION,
        input_shapes=[tensor_shape],
        output_shapes=[TensorShape((1,))],
        precision=precision,
        flops=numel * flops_per_element,
        memory_read_bytes=numel * nbytes,
        memory_write_bytes=nbytes,
        working_set_bytes=(numel + 1) * nbytes,
        access_pattern=AccessPattern.CONTIGUOUS,
        parallelism=ParallelismType.HIGH if numel >= 65536 else ParallelismType.MEDIUM,
        input_residency=input_residency,
    )


def profile_dot(
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    tensor_shape = _shape(shape)
    op = _reduction_profile(OperatorKind.DOT, name, shape, precision, 2.0, input_residency)
    nbytes = precision_nbytes(precision)
    op.input_shapes = [tensor_shape, tensor_shape]
    op.memory_read_bytes = 2 * _numel(tensor_shape) * nbytes
    op.working_set_bytes = op.memory_read_bytes + op.memory_write_bytes
    return op


def profile_nrm2(
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    return _reduction_profile(OperatorKind.NRM2, name, shape, precision, 2.0, input_residency)


def profile_mv(
    name: str,
    matrix_shape: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    matrix = _shape(matrix_shape)
    nbytes = precision_nbytes(precision)
    m = matrix.dims[-2] if len(matrix.dims) >= 2 else None
    n = matrix.dims[-1] if len(matrix.dims) >= 2 else None
    dynamic = matrix.is_dynamic or m is None or n is None
    out_shape = TensorShape((m,)) if m is not None else TensorShape(())
    flops = 0.0 if dynamic else float(2 * int(m) * int(n))
    vector_bytes = (int(n) if n is not None else 0) * nbytes
    output_bytes = _numel(out_shape) * nbytes
    read_bytes = _numel(matrix) * nbytes + vector_bytes + output_bytes
    write_bytes = _numel(out_shape) * nbytes
    return OperatorProfile(
        name=name,
        op=OperatorKind.MV,
        category=OpCategory.MATVEC,
        input_shapes=[matrix, TensorShape((n,)) if n is not None else TensorShape(())],
        output_shapes=[out_shape],
        precision=precision,
        flops=flops,
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.CONTIGUOUS,
        parallelism=ParallelismType.HIGH,
        input_residency=input_residency,
    )


def profile_spmv(
    name: str,
    rows: int,
    cols: int,
    nnz: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    index_bytes = precision_nbytes(Precision.INT32)
    read_bytes = nnz * (nbytes + index_bytes) + (rows + 1) * index_bytes + cols * nbytes + rows * nbytes
    write_bytes = rows * nbytes
    return OperatorProfile(
        name=name,
        op=OperatorKind.SPMV,
        category=OpCategory.MATVEC,
        input_shapes=[TensorShape((rows, cols)), TensorShape((cols,))],
        output_shapes=[TensorShape((rows,))],
        precision=precision,
        flops=float(2 * nnz),
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.SPARSE,
        parallelism=ParallelismType.HIGH if nnz >= 65536 else ParallelismType.MEDIUM,
        input_residency=input_residency,
        attrs={"nnz": nnz},
    )


def profile_trsv(
    name: str,
    n: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    return OperatorProfile(
        name=name,
        op=OperatorKind.TRSV,
        category=OpCategory.TRIANGULAR_SOLVE,
        input_shapes=[TensorShape((n, n)), TensorShape((n,))],
        output_shapes=[TensorShape((n,))],
        precision=precision,
        flops=float(n * n),
        memory_read_bytes=(n * n + n) * nbytes,
        memory_write_bytes=n * nbytes,
        working_set_bytes=(n * n + 2 * n) * nbytes,
        access_pattern=AccessPattern.STRIDED,
        parallelism=ParallelismType.LOW,
        has_data_dependency=True,
        input_residency=input_residency,
    )


def profile_sptrsv(
    name: str,
    n: int,
    nnz: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    index_bytes = precision_nbytes(Precision.INT32)
    nnz = max(n, min(int(nnz), n * (n + 1) // 2))
    read_bytes = nnz * (nbytes + index_bytes) + (n + 1) * index_bytes + n * nbytes
    write_bytes = n * nbytes
    return OperatorProfile(
        name=name,
        op=OperatorKind.SPTRSV,
        category=OpCategory.TRIANGULAR_SOLVE,
        input_shapes=[TensorShape((n, n)), TensorShape((n,))],
        output_shapes=[TensorShape((n,))],
        precision=precision,
        flops=float(2 * nnz),
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.SPARSE,
        parallelism=ParallelismType.LOW,
        has_data_dependency=True,
        input_residency=input_residency,
        attrs={"nnz": nnz},
    )


def profile_trsm(
    name: str,
    n: int,
    nrhs: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    rhs_elements = n * nrhs
    return OperatorProfile(
        name=name,
        op=OperatorKind.TRSM,
        category=OpCategory.TRIANGULAR_SOLVE,
        input_shapes=[TensorShape((n, n)), TensorShape((n, nrhs))],
        output_shapes=[TensorShape((n, nrhs))],
        precision=precision,
        flops=float(n * n * nrhs),
        memory_read_bytes=(n * n + rhs_elements) * nbytes,
        memory_write_bytes=rhs_elements * nbytes,
        working_set_bytes=(n * n + 2 * rhs_elements) * nbytes,
        access_pattern=AccessPattern.STRIDED,
        parallelism=ParallelismType.MEDIUM if nrhs >= 32 else ParallelismType.LOW,
        has_data_dependency=True,
        input_residency=input_residency,
    )


def profile_gemm(
    name: str,
    lhs: tuple[int | None, ...] | list[int | None],
    rhs: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    lhs_shape = _shape(lhs)
    rhs_shape = _shape(rhs)
    nbytes = precision_nbytes(precision)
    dynamic = lhs_shape.is_dynamic or rhs_shape.is_dynamic
    m = lhs_shape.dims[-2] if len(lhs_shape.dims) >= 2 else None
    k = lhs_shape.dims[-1] if lhs_shape.dims else None
    n = rhs_shape.dims[-1] if len(rhs_shape.dims) >= 2 else None
    batch = TensorShape(lhs_shape.dims[:-2]).numel or 1
    out_shape = TensorShape(tuple((*lhs_shape.dims[:-2], m, n)) if len(lhs_shape.dims) >= 2 else ())
    out_numel = _numel(out_shape)
    flops = 0.0 if dynamic or None in (m, n, k) else float(2 * batch * int(m) * int(n) * int(k))
    read_bytes = (_numel(lhs_shape) + _numel(rhs_shape) + out_numel) * nbytes
    write_bytes = out_numel * nbytes
    return OperatorProfile(
        name=name,
        op=OperatorKind.GEMM,
        category=OpCategory.MATMUL,
        input_shapes=[lhs_shape, rhs_shape],
        output_shapes=[out_shape],
        precision=precision,
        flops=flops,
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.CONTIGUOUS,
        parallelism=ParallelismType.MASSIVE if flops >= 1e9 else ParallelismType.HIGH,
        input_residency=input_residency,
    )


def profile_spgemm(
    name: str,
    rows: int,
    inner: int,
    cols: int,
    a_nnz: int,
    b_nnz: int,
    c_nnz: int,
    flops: float | None = None,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    index_bytes = precision_nbytes(Precision.INT32)
    read_bytes = 2 * (cols + 1) * index_bytes + (a_nnz + b_nnz) * (nbytes + index_bytes)
    write_bytes = c_nnz * (nbytes + index_bytes)
    return OperatorProfile(
        name=name,
        op=OperatorKind.SPGEMM,
        category=OpCategory.MATMUL,
        input_shapes=[TensorShape((rows, inner)), TensorShape((inner, cols))],
        output_shapes=[TensorShape((rows, cols))],
        precision=precision,
        flops=float(flops if flops is not None else 2 * max(a_nnz, b_nnz)),
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.SPARSE,
        parallelism=ParallelismType.HIGH,
        input_residency=input_residency,
        attrs={"a_nnz": a_nnz, "b_nnz": b_nnz, "c_nnz": c_nnz},
    )


def profile_transpose(
    name: str,
    shape: tuple[int | None, ...] | list[int | None],
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    tensor_shape = _shape(shape)
    nbytes = precision_nbytes(precision)
    numel = _numel(tensor_shape)
    out_shape = TensorShape(tuple(reversed(tensor_shape.dims)))
    return OperatorProfile(
        name=name,
        op=OperatorKind.TRANSPOSE,
        category=OpCategory.MATRIX_TRANSFORM,
        input_shapes=[tensor_shape],
        output_shapes=[out_shape],
        precision=precision,
        flops=0.0,
        memory_read_bytes=numel * nbytes,
        memory_write_bytes=numel * nbytes,
        working_set_bytes=2 * numel * nbytes,
        access_pattern=AccessPattern.STRIDED,
        parallelism=ParallelismType.HIGH if numel >= 65536 else ParallelismType.MEDIUM,
        input_residency=input_residency,
    )


def _factor_profile(
    kind: OperatorKind,
    name: str,
    n: int,
    flops: float,
    memory_bytes: float,
    precision: Precision,
    input_residency: Residency,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    return OperatorProfile(
        name=name,
        op=kind,
        category=OpCategory.FACTORIZATION,
        input_shapes=[TensorShape((n, n))],
        output_shapes=[TensorShape((n, n))],
        precision=precision,
        flops=flops,
        memory_read_bytes=memory_bytes,
        memory_write_bytes=0.0,
        working_set_bytes=memory_bytes,
        access_pattern=AccessPattern.STRIDED,
        parallelism=ParallelismType.MEDIUM if n >= 512 else ParallelismType.LOW,
        has_data_dependency=True,
        input_residency=input_residency,
    )


def profile_potrf(
    name: str,
    n: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    return _factor_profile(
        OperatorKind.POTRF,
        name,
        n,
        n**3 / 3.0,
        n * n * nbytes,
        precision,
        input_residency,
    )


def profile_getrf(
    name: str,
    n: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    return _factor_profile(
        OperatorKind.GETRF,
        name,
        n,
        2.0 * n**3 / 3.0,
        2 * n * n * nbytes,
        precision,
        input_residency,
    )


def profile_geqrf(
    name: str,
    m: int,
    n: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    matrix_bytes = 2 * m * n * nbytes
    flops = 2.0 * n * n * (m - n / 3.0)
    return OperatorProfile(
        name=name,
        op=OperatorKind.GEQRF,
        category=OpCategory.FACTORIZATION,
        input_shapes=[TensorShape((m, n))],
        output_shapes=[TensorShape((m, n))],
        precision=precision,
        flops=flops,
        memory_read_bytes=matrix_bytes,
        memory_write_bytes=0.0,
        working_set_bytes=matrix_bytes,
        access_pattern=AccessPattern.STRIDED,
        parallelism=ParallelismType.MEDIUM if max(m, n) >= 512 else ParallelismType.LOW,
        has_data_dependency=True,
        input_residency=input_residency,
        attrs={"m": m, "n": n},
    )


def profile_tstrf(
    name: str,
    ns: int,
    nu: int = 0,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    read_bytes = (ns + nu) * (ns + nu) * nbytes
    write_bytes = (ns * ns + nu * ns) * nbytes
    return OperatorProfile(
        name=name,
        op=OperatorKind.TSTRF,
        category=OpCategory.FACTORIZATION,
        input_shapes=[TensorShape((ns + nu, ns + nu))],
        output_shapes=[TensorShape((ns, ns)), TensorShape((nu, ns))],
        precision=precision,
        flops=ns**3 / 3.0 + nu * ns * ns,
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.STRIDED,
        parallelism=ParallelismType.MEDIUM if ns >= 128 or nu >= 128 else ParallelismType.LOW,
        has_data_dependency=True,
        input_residency=input_residency,
        attrs={"ns": ns, "nu": nu},
    )


def profile_gessm(
    name: str,
    overlap_rows: int,
    ns: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    read_bytes = overlap_rows * ns * nbytes
    write_bytes = overlap_rows * overlap_rows * nbytes
    return OperatorProfile(
        name=name,
        op=OperatorKind.GESSM,
        category=OpCategory.FACTORIZATION,
        input_shapes=[TensorShape((overlap_rows, ns))],
        output_shapes=[TensorShape((overlap_rows, overlap_rows))],
        precision=precision,
        flops=float(overlap_rows * ns * ns),
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.CONTIGUOUS,
        parallelism=ParallelismType.HIGH if overlap_rows * ns >= 65536 else ParallelismType.MEDIUM,
        input_residency=input_residency,
        attrs={"overlap_rows": overlap_rows, "ns": ns},
    )


def profile_ssssm(
    name: str,
    n: int,
    nnz_l: int,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    nbytes = precision_nbytes(precision)
    index_bytes = precision_nbytes(Precision.INT32)
    l_bytes = (n + 1) * index_bytes + nnz_l * (index_bytes + nbytes)
    read_bytes = 2 * l_bytes + n * nbytes
    write_bytes = 2 * n * nbytes
    return OperatorProfile(
        name=name,
        op=OperatorKind.SSSSM,
        category=OpCategory.TRIANGULAR_SOLVE,
        input_shapes=[TensorShape((n, n)), TensorShape((n,))],
        output_shapes=[TensorShape((n,)), TensorShape((n,))],
        precision=precision,
        flops=float(4 * nnz_l),
        memory_read_bytes=read_bytes,
        memory_write_bytes=write_bytes,
        working_set_bytes=read_bytes + write_bytes,
        access_pattern=AccessPattern.SPARSE,
        parallelism=ParallelismType.MEDIUM,
        has_data_dependency=True,
        input_residency=input_residency,
        attrs={"n": n, "nnz_l": nnz_l},
    )


def profile_assemble(
    name: str,
    bytes_touched: int,
    flops: float = 0.0,
    precision: Precision = Precision.FP32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    return OperatorProfile(
        name=name,
        op=OperatorKind.ASSEMBLE,
        category=OpCategory.ASSEMBLY,
        input_shapes=[TensorShape((bytes_touched,))],
        output_shapes=[TensorShape((bytes_touched,))],
        precision=precision,
        flops=flops,
        memory_read_bytes=bytes_touched,
        memory_write_bytes=bytes_touched,
        working_set_bytes=2 * bytes_touched,
        access_pattern=AccessPattern.SPARSE,
        parallelism=ParallelismType.MEDIUM,
        has_data_dependency=True,
        input_residency=input_residency,
    )


def profile_symbolic(
    name: str,
    kind: str | OperatorKind = OperatorKind.SYMBOLIC,
    work_items: int = 1,
    precision: Precision = Precision.INT32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    op_kind = require_operator_kind(kind)
    if op_kind not in {
        OperatorKind.SYMBOLIC,
        OperatorKind.ORDER,
        OperatorKind.ETREE,
        OperatorKind.COLCOUNT,
        OperatorKind.POSTORDER,
        OperatorKind.SUPERNODE_PARTITION,
    }:
        raise ValueError(f"{op_kind.value} is not a symbolic DAG support operator")
    nbytes = precision_nbytes(precision)
    return OperatorProfile(
        name=name,
        op=op_kind,
        category=OpCategory.SYMBOLIC,
        input_shapes=[TensorShape((work_items,))],
        output_shapes=[TensorShape((work_items,))],
        precision=precision,
        flops=float(work_items),
        memory_read_bytes=work_items * nbytes,
        memory_write_bytes=work_items * nbytes,
        working_set_bytes=2 * work_items * nbytes,
        access_pattern=AccessPattern.RANDOM,
        parallelism=ParallelismType.LOW,
        has_control_flow=True,
        has_data_dependency=True,
        latency_sensitive=True,
        input_residency=input_residency,
    )


def profile_comm(
    name: str,
    kind: str | OperatorKind,
    bytes_transferred: int,
    precision: Precision = Precision.INT32,
    input_residency: Residency = Residency.UNKNOWN,
) -> OperatorProfile:
    op_kind = require_operator_kind(kind)
    if op_kind not in {OperatorKind.SEND, OperatorKind.RECV, OperatorKind.ALLREDUCE}:
        raise ValueError(f"{op_kind.value} is not a communication operator")
    return OperatorProfile(
        name=name,
        op=op_kind,
        category=OpCategory.COMMUNICATION,
        input_shapes=[TensorShape((bytes_transferred,))],
        output_shapes=[TensorShape((bytes_transferred,))],
        precision=precision,
        flops=0.0,
        memory_read_bytes=bytes_transferred,
        memory_write_bytes=bytes_transferred,
        working_set_bytes=2 * bytes_transferred,
        access_pattern=AccessPattern.CONTIGUOUS,
        parallelism=ParallelismType.MEDIUM,
        input_residency=input_residency,
        attrs={"bytes_transferred": bytes_transferred},
    )
