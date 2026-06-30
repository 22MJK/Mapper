from enum import Enum


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class ExecutionType(str, Enum):
    THREAD_POOL = "thread_pool"
    KERNEL_SIMT = "kernel_simt"


class DeviceFeature(str, Enum):
    SCALAR_CORES = "scalar_cores"
    SIMD = "simd"
    CACHE_HIERARCHY = "cache_hierarchy"
    LOW_LATENCY_LAUNCH = "low_latency_launch"
    BRANCHING = "branching"
    IRREGULAR_MEMORY = "irregular_memory"
    SIMT = "simt"
    WARP_EXECUTION = "warp_execution"
    TENSOR_CORES = "tensor_cores"
    HIGH_BANDWIDTH_MEMORY = "high_bandwidth_memory"
    COALESCED_MEMORY = "coalesced_memory"
    MASSIVE_PARALLELISM = "massive_parallelism"
    DEVICE_MEMORY = "device_memory"
    DMA = "dma"
    COLLECTIVES = "collectives"


class AccessPattern(str, Enum):
    CONTIGUOUS = "contiguous"
    STRIDED = "strided"
    RANDOM = "random"
    SPARSE = "sparse"
    UNKNOWN = "unknown"


class ParallelismType(str, Enum):
    SERIAL = "serial"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MASSIVE = "massive"


class OpCategory(str, Enum):
    VECTOR = "vector"
    REDUCTION = "reduction"
    MATVEC = "matvec"
    TRIANGULAR_SOLVE = "triangular_solve"
    MATMUL = "matmul"
    MATRIX_TRANSFORM = "matrix_transform"
    FACTORIZATION = "factorization"
    ASSEMBLY = "assembly"
    SYMBOLIC = "symbolic"
    COMMUNICATION = "communication"


class OperatorFeature(str, Enum):
    ELEMENTWISE = "elementwise"
    REDUCTION = "reduction"
    DENSE_LINEAR_ALGEBRA = "dense_linear_algebra"
    SPARSE_LINEAR_ALGEBRA = "sparse_linear_algebra"
    TRIANGULAR_DEPENDENCY = "triangular_dependency"
    FACTORIZATION = "factorization"
    MATRIX_TRANSFORM = "matrix_transform"
    ASSEMBLY = "assembly"
    SYMBOLIC = "symbolic"
    COMMUNICATION = "communication"
    STREAMING_MEMORY = "streaming_memory"
    COALESCED_ACCESS = "coalesced_access"
    STRIDED_ACCESS = "strided_access"
    IRREGULAR_ACCESS = "irregular_access"
    CONTROL_FLOW = "control_flow"
    DATA_DEPENDENCY = "data_dependency"
    DYNAMIC_SHAPE = "dynamic_shape"
    LOW_PARALLELISM = "low_parallelism"
    HIGH_PARALLELISM = "high_parallelism"
    MASSIVE_PARALLELISM = "massive_parallelism"
    LATENCY_SENSITIVE = "latency_sensitive"
    SMALL_WORKING_SET = "small_working_set"
    LARGE_WORKING_SET = "large_working_set"
    MEMORY_BOUND = "memory_bound"
    COMPUTE_INTENSIVE = "compute_intensive"
    IN_PLACE = "in_place"
    FUSABLE = "fusable"
    CPU_UNSUPPORTED = "cpu_unsupported"
    GPU_UNSUPPORTED = "gpu_unsupported"


class OperatorKind(str, Enum):
    SCAL = "scal"
    NOOP = "noop"
    COPY = "copy"
    AXPY = "axpy"
    DOT = "dot"
    NRM2 = "nrm2"
    SPMV = "spmv"
    SPTRSV = "sptrsv"
    MV = "mv"
    TRSV = "trsv"
    TRSM = "trsm"
    GEMM = "gemm"
    SPGEMM = "spgemm"
    TRANSPOSE = "transpose"
    POTRF = "potrf"
    GEQRF = "geqrf"
    ASSEMBLE = "assemble"
    SYMBOLIC = "symbolic"
    ORDER = "order"
    ETREE = "etree"
    COLCOUNT = "colcount"
    POSTORDER = "postorder"
    SUPERNODE_PARTITION = "supernode_partition"
    SEND = "send"
    RECV = "recv"
    ALLREDUCE = "allreduce"
    GETRF = "getrf"
    TSTRF = "tstrf"
    GESSM = "gessm"
    SSSSM = "ssssm"


class Residency(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    BOTH = "both"
    UNKNOWN = "unknown"


class Precision(str, Enum):
    FP64 = "fp64"
    FP32 = "fp32"
    TF32 = "tf32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT32 = "int32"
    UNKNOWN = "unknown"
