from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from math import inf, isfinite
from pathlib import Path
from typing import Mapping

from profiles.builtin_devices import GENERIC_CPU, GENERIC_GPU
from profiles.device_profile import DeviceProfile
from profiles.enums import (
    AccessPattern,
    DeviceFeature,
    DeviceType,
    OpCategory,
    OperatorFeature,
    ParallelismType,
    Precision,
    Residency,
)
from profiles.operator_catalog import canonical_operator_name, normalize_operator_name
from profiles.operator_profile import OperatorProfile


EPSILON = 1e-12


@dataclass(frozen=True)
class TopologyDevice:
    name: str
    device_type: DeviceType
    peak_flops_by_precision: Mapping[Precision | str, float]
    memory_bandwidth_bytes_per_s: float
    memory_capacity_bytes: int = 0
    transfer_bandwidth_bytes_per_s: float = 0.0
    transfer_latency_s: float = 0.0
    max_concurrent: int = 1
    parallel_units: int = 1
    threads_per_unit: int = 1
    profile: DeviceProfile | None = None
    attrs: Mapping[str, bool | int | float | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        device_type = DeviceType(self.device_type)
        object.__setattr__(self, "device_type", device_type)
        object.__setattr__(
            self,
            "peak_flops_by_precision",
            {Precision(key): float(value) for key, value in self.peak_flops_by_precision.items()},
        )
        if self.profile is None:
            object.__setattr__(self, "profile", default_device_profile(device_type))

    def peak_flops(self, precision: Precision) -> float:
        precision = Precision(precision)
        peaks = self.peak_flops_by_precision
        if precision in peaks:
            return peaks[precision]
        if Precision.UNKNOWN in peaks:
            return peaks[Precision.UNKNOWN]
        return peaks.get(Precision.FP32, 0.0)


@dataclass(frozen=True)
class PlacementDecision:
    device: TopologyDevice
    latency_s: float
    cpu_latency_s: float
    gpu_latency_s: float
    reason: str


@dataclass(frozen=True)
class OperatorScaleSegment:
    max_num_ops: float | None = None
    bandwidth_scale: float = 1.0
    flops_scale: float = 1.0
    launch_overhead_us: float = 0.0


@dataclass(frozen=True)
class OperatorScale:
    bandwidth_scale: float = 1.0
    flops_scale: float = 1.0
    launch_overhead_us: float = 0.0
    segments: tuple[OperatorScaleSegment, ...] = ()

    def select(self, num_ops: float) -> "OperatorScale":
        for segment in self.segments:
            if segment.max_num_ops is None or num_ops <= segment.max_num_ops:
                return OperatorScale(
                    bandwidth_scale=segment.bandwidth_scale,
                    flops_scale=segment.flops_scale,
                    launch_overhead_us=segment.launch_overhead_us,
                )
        return OperatorScale(
            bandwidth_scale=self.bandwidth_scale,
            flops_scale=self.flops_scale,
            launch_overhead_us=self.launch_overhead_us,
        )


def default_device_profile(device_type: DeviceType) -> DeviceProfile:
    if device_type == DeviceType.CPU:
        return GENERIC_CPU
    if device_type == DeviceType.GPU:
        return GENERIC_GPU
    raise ValueError(f"Unsupported device type: {device_type}")


def estimate_latency(op: OperatorProfile, device: TopologyDevice) -> float:
    if device.device_type == DeviceType.CPU:
        return estimate_cpu_latency(op, device)
    if device.device_type == DeviceType.GPU:
        return estimate_gpu_latency(op, device)
    raise ValueError(f"Unsupported device type: {device.device_type}")


def estimate_cpu_latency(op: OperatorProfile, device: TopologyDevice) -> float:
    if device.device_type != DeviceType.CPU:
        raise ValueError("CPU latency requires a CPU topology device")
    if OperatorFeature.CPU_UNSUPPORTED in op.features:
        return inf
    scale = _operator_scale(op, DeviceType.CPU)
    compute_s = _compute_seconds(
        op,
        device,
        estimate_cpu_vector_efficiency(op, device)
        * estimate_cpu_thread_efficiency(op, device)
        * scale.flops_scale,
    )
    memory_s = _memory_seconds(
        op,
        device,
        estimate_cpu_cache_efficiency(op, device) * scale.bandwidth_scale,
    )
    return (
        max(compute_s, memory_s)
        + estimate_transfer_time(op, device)
        + estimate_cpu_mismatch_penalty(op, device)
    ) * feature_affinity_multiplier(op, device) + scale.launch_overhead_us / 1e6


def estimate_gpu_latency(op: OperatorProfile, device: TopologyDevice) -> float:
    if device.device_type != DeviceType.GPU:
        raise ValueError("GPU latency requires a GPU topology device")
    if OperatorFeature.GPU_UNSUPPORTED in op.features:
        return inf
    scale = _operator_scale(op, DeviceType.GPU)
    compute_s = _compute_seconds(
        op,
        device,
        estimate_gpu_occupancy(op, device) * estimate_gpu_simt_efficiency(op, device) * scale.flops_scale,
    )
    memory_s = _memory_seconds(
        op,
        device,
        estimate_gpu_coalescing_efficiency(op, device) * scale.bandwidth_scale,
    )
    return (
        max(compute_s, memory_s)
        + estimate_transfer_time(op, device)
        + estimate_gpu_mismatch_penalty(op, device)
    ) * feature_affinity_multiplier(op, device) + scale.launch_overhead_us / 1e6


def estimate_cpu_vector_efficiency(op: OperatorProfile, device: TopologyDevice) -> float:
    del op, device
    return 1.0


def estimate_cpu_thread_efficiency(op: OperatorProfile, device: TopologyDevice) -> float:
    del op, device
    return 1.0


def estimate_cpu_cache_efficiency(op: OperatorProfile, device: TopologyDevice) -> float:
    del op, device
    return 1.0


def estimate_gpu_occupancy(op: OperatorProfile, device: TopologyDevice) -> float:
    profile = device.profile or GENERIC_GPU
    if DeviceFeature.SIMT not in profile.features:
        return 0.1
    work_items = _work_items(op)
    full_wave = max(1, device.parallel_units * device.threads_per_unit)
    occupancy = min(1.0, work_items / full_wave)
    if op.parallelism == ParallelismType.MASSIVE:
        occupancy = max(occupancy, 0.85)
    elif op.parallelism == ParallelismType.HIGH:
        occupancy = max(occupancy, 0.55)
    elif op.parallelism == ParallelismType.MEDIUM:
        occupancy = max(occupancy, 0.25)
    return max(0.05, occupancy)


def estimate_gpu_simt_efficiency(op: OperatorProfile, device: TopologyDevice) -> float:
    profile = device.profile or GENERIC_GPU
    efficiency = 0.95 if DeviceFeature.WARP_EXECUTION in profile.features else 0.65
    if OperatorFeature.CONTROL_FLOW in op.features or op.category == OpCategory.SYMBOLIC:
        efficiency *= 0.2
    if OperatorFeature.DATA_DEPENDENCY in op.features:
        efficiency *= 0.65
    if OperatorFeature.DYNAMIC_SHAPE in op.features:
        efficiency *= 0.75
    if op.parallelism in {ParallelismType.SERIAL, ParallelismType.LOW}:
        efficiency *= 0.55
    return max(0.05, efficiency)


def estimate_gpu_coalescing_efficiency(op: OperatorProfile, device: TopologyDevice) -> float:
    profile = device.profile or GENERIC_GPU
    contiguous_eff = 0.95 if DeviceFeature.COALESCED_MEMORY in profile.features else 0.75
    if op.access_pattern == AccessPattern.CONTIGUOUS:
        return contiguous_eff
    if op.access_pattern == AccessPattern.STRIDED:
        return 0.65
    if op.access_pattern == AccessPattern.RANDOM:
        return 0.25
    if op.access_pattern == AccessPattern.SPARSE:
        return 0.18
    return 0.5


def estimate_transfer_time(op: OperatorProfile, device: TopologyDevice) -> float:
    if op.input_residency in {Residency.UNKNOWN, Residency.BOTH}:
        return 0.0
    if device.device_type == DeviceType.CPU and op.input_residency == Residency.CPU:
        return 0.0
    if device.device_type == DeviceType.GPU and op.input_residency == Residency.GPU:
        return 0.0
    bandwidth = device.transfer_bandwidth_bytes_per_s
    if bandwidth <= 0:
        return 0.0
    return device.transfer_latency_s + op.memory_read_bytes / bandwidth


def estimate_mismatch_penalty(op: OperatorProfile, device: TopologyDevice) -> float:
    if device.device_type == DeviceType.CPU:
        return estimate_cpu_mismatch_penalty(op, device)
    if device.device_type == DeviceType.GPU:
        return estimate_gpu_mismatch_penalty(op, device)
    return 0.0


def estimate_cpu_mismatch_penalty(op: OperatorProfile, device: TopologyDevice) -> float:
    del op, device
    return 0.0


def estimate_gpu_mismatch_penalty(op: OperatorProfile, device: TopologyDevice) -> float:
    del op, device
    return 0.0


def feature_affinity_multiplier(op: OperatorProfile, device: TopologyDevice) -> float:
    profile = device.profile or default_device_profile(device.device_type)
    score = profile.match_score(op.features, op.precision)
    return min(1.45, max(0.65, 1.0 - 0.04 * score))


def choose_device(
    op: OperatorProfile,
    cpu: TopologyDevice,
    gpu: TopologyDevice,
) -> PlacementDecision:
    cpu_latency = estimate_cpu_latency(op, cpu)
    gpu_latency = estimate_gpu_latency(op, gpu)
    if cpu_latency <= gpu_latency:
        device = cpu
        latency = cpu_latency
    else:
        device = gpu
        latency = gpu_latency
    return PlacementDecision(
        device=device,
        latency_s=latency,
        cpu_latency_s=cpu_latency,
        gpu_latency_s=gpu_latency,
        reason=build_reason(op, device, cpu_latency, gpu_latency),
    )


def build_reason(
    op: OperatorProfile,
    device: TopologyDevice,
    cpu_latency_s: float | None = None,
    gpu_latency_s: float | None = None,
) -> str:
    profile = device.profile or default_device_profile(device.device_type)
    parts = [f"{device.name} selected for {op.category.value}"]
    if cpu_latency_s is not None and gpu_latency_s is not None:
        parts.append(f"cpu={cpu_latency_s * 1e6:.2f}us")
        parts.append(f"gpu={gpu_latency_s * 1e6:.2f}us")
    score = profile.match_score(op.features, op.precision)
    parts.append(f"feature_score={score:.1f}")
    if device.device_type == DeviceType.CPU:
        if OperatorFeature.CONTROL_FLOW in op.features:
            parts.append("control-flow friendly")
        if OperatorFeature.IRREGULAR_ACCESS in op.features:
            parts.append("irregular-memory friendly")
        if OperatorFeature.SMALL_WORKING_SET in op.features:
            parts.append("small working set")
    else:
        if OperatorFeature.DENSE_LINEAR_ALGEBRA in op.features:
            parts.append("dense linear algebra")
        if OperatorFeature.MEMORY_BOUND in op.features:
            parts.append("memory-bandwidth dominated")
        if op.parallelism in {ParallelismType.HIGH, ParallelismType.MASSIVE}:
            parts.append("SIMT parallelism")
    return "; ".join(parts)


def _compute_seconds(op: OperatorProfile, device: TopologyDevice, efficiency: float) -> float:
    peak = device.peak_flops(op.precision)
    if op.flops > 0 and peak <= 0:
        return inf
    return op.flops / max(peak * efficiency, EPSILON)


def _memory_seconds(op: OperatorProfile, device: TopologyDevice, efficiency: float) -> float:
    bandwidth = device.memory_bandwidth_bytes_per_s
    if op.memory_bytes > 0 and bandwidth <= 0:
        return inf
    return op.memory_bytes / max(bandwidth * efficiency, EPSILON)


def _operator_scale(op: OperatorProfile, device_type: DeviceType) -> OperatorScale:
    name = canonical_operator_name(op.op) or normalize_operator_name(str(op.op))
    return _load_operator_scales(device_type).get(name, OperatorScale()).select(op.flops)


@lru_cache(maxsize=1)
def _load_operator_scale_roots() -> tuple[Mapping[str, OperatorScale], Mapping[str, OperatorScale]]:
    config_path = _find_global_simulator_config()
    if config_path is None:
        return {}, {}
    with config_path.open() as f:
        root = json.load(f)

    cpu_scales = _parse_operator_scales(
        root.get("cpu-operator-scales", root.get("cpu_operator_scales", {})),
        "cpu-operator-scales",
        config_path,
    )
    gpu_scales = _parse_operator_scales(
        root.get("gpu-operator-scales", root.get("gpu_operator_scales", {})),
        "gpu-operator-scales",
        config_path,
    )
    return cpu_scales, gpu_scales


def _load_operator_scales(device_type: DeviceType) -> Mapping[str, OperatorScale]:
    cpu_scales, gpu_scales = _load_operator_scale_roots()
    if device_type == DeviceType.CPU:
        return cpu_scales
    if device_type == DeviceType.GPU:
        return gpu_scales
    return {}


def _parse_operator_scales(
    scale_root: object,
    field_name: str,
    config_path: Path,
) -> Mapping[str, OperatorScale]:
    if scale_root is None:
        return {}
    if not isinstance(scale_root, dict):
        raise ValueError(f"{field_name} in {config_path} must be an object")

    scales: dict[str, OperatorScale] = {}
    for op_name, item in scale_root.items():
        if not isinstance(item, dict):
            raise ValueError(f"{field_name}.{op_name} in {config_path} must be an object")
        bandwidth_scale = _validated_scale(
            item.get("bandwidth_scale", 1.0),
            f"bandwidth_scale for {op_name}",
            config_path,
        )
        flops_scale = _validated_scale(
            item.get("flops_scale", 1.0),
            f"flops_scale for {op_name}",
            config_path,
        )
        launch_overhead_us = _validated_overhead_us(
            item.get("launch_overhead_us", 0.0),
            f"launch_overhead_us for {op_name}",
            config_path,
        )
        raw_segments = item.get("segments", [])
        if raw_segments is None:
            raw_segments = []
        if not isinstance(raw_segments, list):
            raise ValueError(f"segments for {op_name} in {config_path} must be a list")
        segments: list[OperatorScaleSegment] = []
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                raise ValueError(f"segments for {op_name} in {config_path} must contain objects")
            max_num_ops_raw = raw_segment.get("max_num_ops")
            max_num_ops = None
            if max_num_ops_raw is not None:
                max_num_ops = _validated_scale(
                    max_num_ops_raw,
                    f"max_num_ops for {op_name}",
                    config_path,
                )
            segments.append(
                OperatorScaleSegment(
                    max_num_ops=max_num_ops,
                    bandwidth_scale=_validated_scale(
                        raw_segment.get("bandwidth_scale", bandwidth_scale),
                        f"bandwidth_scale for {op_name} segment",
                        config_path,
                    ),
                    flops_scale=_validated_scale(
                        raw_segment.get("flops_scale", flops_scale),
                        f"flops_scale for {op_name} segment",
                        config_path,
                    ),
                    launch_overhead_us=_validated_overhead_us(
                        raw_segment.get("launch_overhead_us", launch_overhead_us),
                        f"launch_overhead_us for {op_name} segment",
                        config_path,
                    ),
                )
            )
        segments.sort(key=lambda segment: inf if segment.max_num_ops is None else segment.max_num_ops)
        scales[normalize_operator_name(str(op_name))] = OperatorScale(
            bandwidth_scale=bandwidth_scale,
            flops_scale=flops_scale,
            launch_overhead_us=launch_overhead_us,
            segments=tuple(segments),
        )
    return scales


def _validated_scale(value: object, field_name: str, config_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} in {config_path} must be numeric")
    scale = float(value)
    if not isfinite(scale) or scale <= 0:
        raise ValueError(f"{field_name} in {config_path} must be a positive finite number")
    return scale


def _validated_overhead_us(value: object, field_name: str, config_path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} in {config_path} must be numeric")
    overhead = float(value)
    if not isfinite(overhead) or overhead < 0:
        raise ValueError(f"{field_name} in {config_path} must be a non-negative finite number")
    return overhead


def _find_global_simulator_config() -> Path | None:
    for start in (Path.cwd(), Path(__file__)):
        found = _find_global_simulator_config_from(start)
        if found is not None:
            return found
    return None


def _find_global_simulator_config_from(start: Path) -> Path | None:
    current = start if start.is_dir() else start.parent
    try:
        current = current.resolve()
    except OSError:
        current = current.absolute()
    while True:
        repo_candidate = current / "simulator" / "simulator_global_config.json"
        if repo_candidate.exists():
            return repo_candidate
        simulator_candidate = current / "simulator_global_config.json"
        if simulator_candidate.exists():
            return simulator_candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def _work_items(op: OperatorProfile) -> int:
    for shape in op.output_shapes:
        if shape.numel is not None:
            return max(1, shape.numel)
    for shape in op.input_shapes:
        if shape.numel is not None:
            return max(1, shape.numel)
    return 1
