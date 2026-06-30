from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, TypeVar

from .enums import DeviceFeature, DeviceType, ExecutionType, OperatorFeature, Precision


EnumT = TypeVar("EnumT")


def _enum_set(enum_type: type[EnumT], values: Iterable[EnumT | str] | None) -> frozenset[EnumT]:
    if values is None:
        return frozenset()
    return frozenset(value if isinstance(value, enum_type) else enum_type(value) for value in values)


@dataclass(frozen=True)
class CPUProfile:
    features: frozenset[DeviceFeature] = field(default_factory=frozenset)
    supported_precisions: frozenset[Precision] = field(default_factory=frozenset)
    preferred_operator_features: frozenset[OperatorFeature] = field(default_factory=frozenset)
    discouraged_operator_features: frozenset[OperatorFeature] = field(default_factory=frozenset)
    attrs: dict[str, bool | int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _enum_set(DeviceFeature, self.features))
        object.__setattr__(
            self, "supported_precisions", _enum_set(Precision, self.supported_precisions)
        )
        object.__setattr__(
            self,
            "preferred_operator_features",
            _enum_set(OperatorFeature, self.preferred_operator_features),
        )
        object.__setattr__(
            self,
            "discouraged_operator_features",
            _enum_set(OperatorFeature, self.discouraged_operator_features),
        )


@dataclass(frozen=True)
class GPUProfile:
    features: frozenset[DeviceFeature] = field(default_factory=frozenset)
    supported_precisions: frozenset[Precision] = field(default_factory=frozenset)
    preferred_operator_features: frozenset[OperatorFeature] = field(default_factory=frozenset)
    discouraged_operator_features: frozenset[OperatorFeature] = field(default_factory=frozenset)
    attrs: dict[str, bool | int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", _enum_set(DeviceFeature, self.features))
        object.__setattr__(
            self, "supported_precisions", _enum_set(Precision, self.supported_precisions)
        )
        object.__setattr__(
            self,
            "preferred_operator_features",
            _enum_set(OperatorFeature, self.preferred_operator_features),
        )
        object.__setattr__(
            self,
            "discouraged_operator_features",
            _enum_set(OperatorFeature, self.discouraged_operator_features),
        )


@dataclass(frozen=True)
class DeviceProfile:
    name: str
    device_type: DeviceType
    execution_type: ExecutionType
    features: frozenset[DeviceFeature] = field(default_factory=frozenset)
    supported_precisions: frozenset[Precision] = field(default_factory=frozenset)
    preferred_operator_features: frozenset[OperatorFeature] = field(default_factory=frozenset)
    discouraged_operator_features: frozenset[OperatorFeature] = field(default_factory=frozenset)
    cpu: CPUProfile | None = None
    gpu: GPUProfile | None = None
    attrs: dict[str, bool | int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_type", DeviceType(self.device_type))
        object.__setattr__(self, "execution_type", ExecutionType(self.execution_type))

        features = set(_enum_set(DeviceFeature, self.features))
        precisions = set(_enum_set(Precision, self.supported_precisions))
        preferred = set(_enum_set(OperatorFeature, self.preferred_operator_features))
        discouraged = set(_enum_set(OperatorFeature, self.discouraged_operator_features))

        if self.cpu is not None:
            features.update(self.cpu.features)
            precisions.update(self.cpu.supported_precisions)
            preferred.update(self.cpu.preferred_operator_features)
            discouraged.update(self.cpu.discouraged_operator_features)
        if self.gpu is not None:
            features.update(self.gpu.features)
            precisions.update(self.gpu.supported_precisions)
            preferred.update(self.gpu.preferred_operator_features)
            discouraged.update(self.gpu.discouraged_operator_features)

        object.__setattr__(self, "features", frozenset(features))
        object.__setattr__(self, "supported_precisions", frozenset(precisions))
        object.__setattr__(self, "preferred_operator_features", frozenset(preferred))
        object.__setattr__(self, "discouraged_operator_features", frozenset(discouraged))

    def supports_precision(self, precision: Precision) -> bool:
        precision = Precision(precision)
        return not self.supported_precisions or precision in self.supported_precisions

    def match_score(
        self,
        operator_features: Iterable[OperatorFeature | str],
        precision: Precision = Precision.UNKNOWN,
    ) -> float:
        features = _enum_set(OperatorFeature, operator_features)
        score = 0.0
        score += 2.0 * len(features & self.preferred_operator_features)
        score -= 2.0 * len(features & self.discouraged_operator_features)
        if precision != Precision.UNKNOWN and not self.supports_precision(precision):
            score -= 4.0
        return score
