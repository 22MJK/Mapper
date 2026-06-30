from .device_profile import CPUProfile, DeviceProfile, GPUProfile
from .enums import DeviceFeature, DeviceType, ExecutionType, OperatorFeature, Precision


COMMON_NUMERIC_PRECISIONS = frozenset(
    {
        Precision.FP64,
        Precision.FP32,
        Precision.TF32,
        Precision.FP16,
        Precision.BF16,
        Precision.INT8,
        Precision.INT32,
    }
)


GENERIC_CPU = DeviceProfile(
    name="generic_cpu",
    device_type=DeviceType.CPU,
    execution_type=ExecutionType.THREAD_POOL,
    cpu=CPUProfile(
        features=frozenset(
            {
                DeviceFeature.SCALAR_CORES,
                DeviceFeature.SIMD,
                DeviceFeature.CACHE_HIERARCHY,
                DeviceFeature.LOW_LATENCY_LAUNCH,
                DeviceFeature.BRANCHING,
                DeviceFeature.IRREGULAR_MEMORY,
            }
        ),
        supported_precisions=COMMON_NUMERIC_PRECISIONS,
        preferred_operator_features=frozenset(
            {
                OperatorFeature.CONTROL_FLOW,
                OperatorFeature.DATA_DEPENDENCY,
                OperatorFeature.IRREGULAR_ACCESS,
                OperatorFeature.LATENCY_SENSITIVE,
                OperatorFeature.LOW_PARALLELISM,
                OperatorFeature.SMALL_WORKING_SET,
                OperatorFeature.SYMBOLIC,
            }
        ),
        discouraged_operator_features=frozenset(
            {
                OperatorFeature.MASSIVE_PARALLELISM,
                OperatorFeature.DENSE_LINEAR_ALGEBRA,
                OperatorFeature.COMPUTE_INTENSIVE,
            }
        ),
    ),
)


GENERIC_GPU = DeviceProfile(
    name="generic_gpu",
    device_type=DeviceType.GPU,
    execution_type=ExecutionType.KERNEL_SIMT,
    gpu=GPUProfile(
        features=frozenset(
            {
                DeviceFeature.SIMT,
                DeviceFeature.WARP_EXECUTION,
                DeviceFeature.HIGH_BANDWIDTH_MEMORY,
                DeviceFeature.COALESCED_MEMORY,
                DeviceFeature.MASSIVE_PARALLELISM,
                DeviceFeature.DEVICE_MEMORY,
                DeviceFeature.DMA,
                DeviceFeature.COLLECTIVES,
            }
        ),
        supported_precisions=COMMON_NUMERIC_PRECISIONS,
        preferred_operator_features=frozenset(
            {
                OperatorFeature.MASSIVE_PARALLELISM,
                OperatorFeature.HIGH_PARALLELISM,
                OperatorFeature.DENSE_LINEAR_ALGEBRA,
                OperatorFeature.COMPUTE_INTENSIVE,
                OperatorFeature.STREAMING_MEMORY,
                OperatorFeature.COALESCED_ACCESS,
            }
        ),
        discouraged_operator_features=frozenset(
            {
                OperatorFeature.CONTROL_FLOW,
                OperatorFeature.DATA_DEPENDENCY,
                OperatorFeature.DYNAMIC_SHAPE,
                OperatorFeature.IRREGULAR_ACCESS,
                OperatorFeature.LATENCY_SENSITIVE,
                OperatorFeature.SMALL_WORKING_SET,
                OperatorFeature.SYMBOLIC,
            }
        ),
    ),
)
