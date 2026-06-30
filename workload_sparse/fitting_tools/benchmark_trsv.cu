#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>    // NVTX v2 (CUDA 自带) 用于定位 Kernel

// 错误检查宏
#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }
#define CUBLAS_CHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS Error: " << err << std::endl; exit(-1); }

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <Matrix_Size_N>" << std::endl;
        return -1;
    }
    
    // 从命令行参数获取矩阵规模 N
    int n = std::stoi(argv[1]);
    int lda = n;
    int incx = 1;

    // 1. 在主机（CPU）上分配并初始化一个稠密上三角矩阵和向量
    std::vector<float> h_A(n * n, 0.0f);
    std::vector<float> h_x(n, 1.0f);
    
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            h_A[i * n + j] = (i == j) ? 2.0f : 0.1f; // 保证对角线非零
        }
    }

    // 2. 分配设备（GPU）显存
    float *d_A, *d_x;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));

    // 3. 将数据拷贝到 GPU
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // 4. 初始化 cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cudaEvent timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ====== 核心热点区域：使用 NVTX 标记 ======
    nvtxRangePushA("cuBLAS_TRSV_Range");
    CUDA_CHECK(cudaEventRecord(start));

    // 执行上三角 TRSV 算子: A * x = b, 结果直接覆盖到 d_x 中
    CUBLAS_CHECK(cublasStrsv(
        handle, 
        CUBLAS_FILL_MODE_UPPER, 
        CUBLAS_OP_N, 
        CUBLAS_DIAG_NON_UNIT, 
        n, d_A, lda, d_x, incx
    ));

    CUDA_CHECK(cudaEventRecord(stop));
    // 必须同步，确保 GPU 计算在 NVTX 范围结束前完成
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop(); 
    // ==========================================

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 清理资源
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_x));

    std::cout << "TRSV_TIME_MS " << ms << std::endl;
    std::cout << "SUCCESS: TRSV for N=" << n << " executed." << std::endl;
    return 0;
}
