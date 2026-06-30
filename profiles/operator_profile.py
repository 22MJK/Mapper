from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, Iterable, TypeVar

from .enums import (
    AccessPattern,
    OpCategory,
    OperatorFeature,
    OperatorKind,
    ParallelismType,
    Precision,
    Residency,
)
from .operator_catalog import require_operator_kind


EnumT = TypeVar("EnumT")


def _enum_set(enum_type: type[EnumT], values: Iterable[EnumT | str] | None) -> frozenset[EnumT]:
    if values is None:
        return frozenset()
    return frozenset(value if isinstance(value, enum_type) else enum_type(value) for value in values)


@dataclass(frozen=True)
class TensorShape:
    dims: tuple[int | None, ...]

    @property
    def numel(self) -> int | None:
        if self.is_dynamic:
            return None
        return reduce(mul, self.dims, 1)

    @property
    def is_dynamic(self) -> bool:
        return any(dim is None or dim < 0 for dim in self.dims)


@dataclass
class OperatorProfile:
    name: str
    op: OperatorKind
    category: OpCategory
    input_shapes: list[TensorShape]
    output_shapes: list[TensorShape]
    precision: Precision = Precision.FP32
    flops: float = 0.0
    memory_read_bytes: float = 0.0
    memory_write_bytes: float = 0.0
    working_set_bytes: float = 0.0
    access_pattern: AccessPattern = AccessPattern.UNKNOWN
    parallelism: ParallelismType = ParallelismType.MEDIUM
    has_control_flow: bool = False
    has_dynamic_shape: bool = False
    has_data_dependency: bool = False
    is_inplace: bool = False
    is_fusable: bool = False
    batch_size: int | None = None
    input_residency: Residency = Residency.UNKNOWN
    output_residency_hint: Residency = Residency.UNKNOWN
    latency_sensitive: bool = False
    features: frozenset[OperatorFeature] = field(default_factory=frozenset)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.op = require_operator_kind(self.op)
        if not self.has_dynamic_shape:
            self.has_dynamic_shape = any(
                shape.is_dynamic for shape in self.input_shapes + self.output_shapes
            )
        explicit = _enum_set(OperatorFeature, self.features)
        self.features = frozenset({*explicit, *infer_operator_features(self)})

    @property
    def memory_bytes(self) -> float:
        return self.memory_read_bytes + self.memory_write_bytes

    @property
    def arithmetic_intensity(self) -> float:
        if self.memory_bytes <= 0:
            return float("inf") if self.flops > 0 else 0.0
        return self.flops / self.memory_bytes


def infer_operator_features(op: OperatorProfile) -> frozenset[OperatorFeature]:
    features: set[OperatorFeature] = set()
    if op.category == OpCategory.VECTOR:
        features.add(OperatorFeature.ELEMENTWISE)
    elif op.category == OpCategory.REDUCTION:
        features.add(OperatorFeature.REDUCTION)
    elif op.category == OpCategory.MATVEC:
        if op.access_pattern == AccessPattern.SPARSE:
            features.add(OperatorFeature.SPARSE_LINEAR_ALGEBRA)
        else:
            features.add(OperatorFeature.DENSE_LINEAR_ALGEBRA)
    elif op.category == OpCategory.TRIANGULAR_SOLVE:
        if op.access_pattern == AccessPattern.SPARSE or op.op == OperatorKind.SPTRSV:
            features.add(OperatorFeature.SPARSE_LINEAR_ALGEBRA)
        else:
            features.add(OperatorFeature.DENSE_LINEAR_ALGEBRA)
        features.update(
            {
                OperatorFeature.TRIANGULAR_DEPENDENCY,
                OperatorFeature.DATA_DEPENDENCY,
            }
        )
    elif op.category == OpCategory.MATMUL:
        if op.access_pattern == AccessPattern.SPARSE:
            features.add(OperatorFeature.SPARSE_LINEAR_ALGEBRA)
        else:
            features.add(OperatorFeature.DENSE_LINEAR_ALGEBRA)
    elif op.category == OpCategory.MATRIX_TRANSFORM:
        features.add(OperatorFeature.MATRIX_TRANSFORM)
    elif op.category == OpCategory.FACTORIZATION:
        features.update({OperatorFeature.FACTORIZATION, OperatorFeature.DENSE_LINEAR_ALGEBRA})
    elif op.category == OpCategory.ASSEMBLY:
        features.add(OperatorFeature.ASSEMBLY)
    elif op.category == OpCategory.SYMBOLIC:
        features.add(OperatorFeature.SYMBOLIC)
        features.add(OperatorFeature.GPU_UNSUPPORTED)
    elif op.category == OpCategory.COMMUNICATION:
        features.add(OperatorFeature.COMMUNICATION)

    if op.access_pattern == AccessPattern.CONTIGUOUS:
        features.update({OperatorFeature.STREAMING_MEMORY, OperatorFeature.COALESCED_ACCESS})
    elif op.access_pattern == AccessPattern.STRIDED:
        features.add(OperatorFeature.STRIDED_ACCESS)
    elif op.access_pattern in {AccessPattern.RANDOM, AccessPattern.SPARSE}:
        features.add(OperatorFeature.IRREGULAR_ACCESS)

    if op.parallelism in {ParallelismType.SERIAL, ParallelismType.LOW}:
        features.add(OperatorFeature.LOW_PARALLELISM)
    elif op.parallelism == ParallelismType.HIGH:
        features.add(OperatorFeature.HIGH_PARALLELISM)
    elif op.parallelism == ParallelismType.MASSIVE:
        features.add(OperatorFeature.MASSIVE_PARALLELISM)

    if op.has_control_flow:
        features.add(OperatorFeature.CONTROL_FLOW)
    if op.has_data_dependency:
        features.add(OperatorFeature.DATA_DEPENDENCY)
    if op.has_dynamic_shape:
        features.add(OperatorFeature.DYNAMIC_SHAPE)
    if op.latency_sensitive:
        features.add(OperatorFeature.LATENCY_SENSITIVE)
    if op.is_inplace:
        features.add(OperatorFeature.IN_PLACE)
    if op.is_fusable:
        features.add(OperatorFeature.FUSABLE)

    working_set = op.working_set_bytes or op.memory_bytes
    if working_set > 0:
        if working_set < 256 * 1024:
            features.add(OperatorFeature.SMALL_WORKING_SET)
        elif working_set >= 64 * 1024 * 1024:
            features.add(OperatorFeature.LARGE_WORKING_SET)

    if op.memory_bytes > 0:
        if op.arithmetic_intensity < 1.0:
            features.add(OperatorFeature.MEMORY_BOUND)
        elif op.arithmetic_intensity >= 8.0:
            features.add(OperatorFeature.COMPUTE_INTENSIVE)
    elif op.flops > 0:
        features.add(OperatorFeature.COMPUTE_INTENSIVE)

    return frozenset(features)
