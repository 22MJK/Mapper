#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHECK(err) if (err != cudaSuccess) { std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; exit(-1); }
#define CUBLAS_CHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS Error: " << err << std::endl; exit(-1); }

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <op> <params...>" << std::endl;
        std::cerr << "  syrk  <N> <K>" << std::endl;
        std::cerr << "  trsm  <N> <M>" << std::endl;
        return -1;
    }

    std::string op = argv[1];

    if (op == "syrk") {
        // ── SYRK: C = A * A^T (lower, N×N from A(N×K)) ──
        int N = std::stoi(argv[2]);
        int K = std::stoi(argv[3]);
        int lda = K, ldc = N;
        float alpha = 1.0f, beta = 0.0f;

        std::vector<float> h_A(N * K, 1.0f);
        std::vector<float> h_C(N * N, 0.0f);

        float *d_A, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, N * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_C, N * N * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * K * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_C, h_C.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));

        // warm-up
        CUBLAS_CHECK(cublasSsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K, &alpha, d_A, lda, &beta, d_C, ldc));
        CUDA_CHECK(cudaDeviceSynchronize());

        nvtxRangePushA("cuBLAS_SYRK_Range");
        CUBLAS_CHECK(cublasSsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K, &alpha, d_A, lda, &beta, d_C, ldc));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUBLAS_CHECK(cublasDestroy(h));
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_C));
        std::cout << "SUCCESS: SYRK N=" << N << " K=" << K << std::endl;
    }
    else if (op == "trsm") {
        // ── TRSM: solve op(T)*X = B (T: N×N三角, B: N×M) ──
        int N = std::stoi(argv[2]);
        int M = std::stoi(argv[3]);
        int lda = N, ldb = N;
        float alpha = 1.0f;

        std::vector<float> h_A(N * N, 0.0f);
        std::vector<float> h_B(N * M, 1.0f);
        for (int i = 0; i < N; i++)
            for (int j = 0; j <= i; j++)
                h_A[i * N + j] = (i == j) ? 2.0f : 0.1f;

        float *d_A, *d_B;
        CUDA_CHECK(cudaMalloc(&d_A, N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_B, N * M * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), N * M * sizeof(float), cudaMemcpyHostToDevice));

        cublasHandle_t h;
        CUBLAS_CHECK(cublasCreate(&h));

        // warm-up
        CUBLAS_CHECK(cublasStrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                  CUBLAS_DIAG_NON_UNIT, N, M, &alpha, d_A, lda, d_B, ldb));
        CUDA_CHECK(cudaDeviceSynchronize());

        nvtxRangePushA("cuBLAS_TRSM_Range");
        CUBLAS_CHECK(cublasStrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                  CUBLAS_DIAG_NON_UNIT, N, M, &alpha, d_A, lda, d_B, ldb));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvtxRangePop();

        CUBLAS_CHECK(cublasDestroy(h));
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B));
        std::cout << "SUCCESS: TRSM N=" << N << " M=" << M << std::endl;
    }
    else {
        std::cerr << "Unknown op: " << op << std::endl;
        return -1;
    }
    return 0;
}
