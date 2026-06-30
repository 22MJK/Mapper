# fitting_tools — BLAS/LAPACK GPU DRAM 实测拟合工具包

> 基于 RTX 4060 Laptop (SM 8.9) + CUDA 12.6 + Nsight Compute 2024.3.0

## 文件说明

| 文件 | 用途 |
|---|---|
| `fit_measured_data.py` | **主脚本**: 读取 ncu CSV, 拟合 Bytes 公式, 输出 k×THEORY |
| `benchmark_all.cu` | Level-1/2/3 BLAS 统一 benchmark |
| `benchmark_extra.cu` | 稀疏/LAPACK/转置/带状 benchmark (需 cuSPARSE+cuSOLVER) |
| `benchmark_trsv.cu` | TRSV 专用 (已集成到 benchmark_all) |
| `benchmark_blas.cu` | SYRK/TRSM 专用 (已集成到 benchmark_all) |
| `benchmark_gemm.cu` | GEMM 专用 (已集成到 benchmark_all) |
| `run_7sizes.bat` | 一键剖析全部 L2+L3 (7组规模) |
| `run_extra.bat` | 剖析稀疏/LAPACK/转置 (7组规模) |
| `run_final.bat` | 补剖析 SPTRSV+SPGEMM+TDENSE |
| `cuda_env.ps1` | PowerShell 中初始化 CUDA+MSVC 环境 |
| `README_ALL_FLOPs_Bytes.md` | 完整 FLOPs/Bytes 公式 + GPU 拟合系数 |

## 仓库内容约定

本目录只跟踪可复现的源码、脚本和说明文档。以下内容是本地生成或外部依赖，
不随源码仓库分发：

- `benchmark_*.exe`, `*.lib`, `*.exp`, `*.obj`: 本机编译产物。
- `ncu_*.csv`: Nsight Compute 原始 profiling 输出，可能包含本机路径和进程信息。
- `openblas/`: 预编译 OpenBLAS 包。需要 CPU BLAS/LAPACK 时，请使用系统包管理器、
  vcpkg/Conan，或在本机单独安装。

## 使用流程

```powershell
# 1. 编译所有 benchmark
nvcc -O3 -lcublas benchmark_all.cu -o benchmark_all.exe -arch=sm_89
nvcc -O3 -lcublas benchmark_trsv.cu -o benchmark_trsv.exe -arch=sm_89
nvcc -O3 -lcublas benchmark_blas.cu -o benchmark_blas.exe -arch=sm_89
nvcc -O3 -lcublas -lcusparse -lcusolver benchmark_extra.cu -o benchmark_extra.exe -arch=sm_89

# 2. 剖析 (需管理员权限)
右键 run_7sizes.bat → 以管理员身份运行
右键 run_extra.bat  → 以管理员身份运行
右键 run_final.bat  → 以管理员身份运行

# 3. 拟合
python fit_measured_data.py
```

## 拟合模型

DRAM_GPU = k × DRAM_README (MAPE 最小化)
k 值详见 `README_ALL_FLOPs_Bytes.md` 各表的 "BytesGPU拟合" 列。
