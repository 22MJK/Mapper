/*
 * benchmark_all.cu — 统一 BLAS/LAPACK benchmark
 * 覆盖 Level-1/2/3 + LAPACK 全部算子
 * 
 * 用法: benchmark_all.exe <op> <params...>
 * 
 * Level-1:  scal <N>  | copy <N>  | axpy <N>  | dot <N>
 *           nrm2 <N>  | rot <N>   | rotm <N>
 * Level-2:  gemv <M> <N> | symv <N> | trmv <N> | ger <M> <N> | syr <N>
 * Level-3:  gemm <M> <N> <K> | symm <M> <N> | syrk <N> <K>
 *           syr2k <N> <K> | trmm <M> <N>
 * (trsv + trsm already in benchmark_blas.cu / benchmark_trsv.cu)
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHK(e) if(e!=cudaSuccess){std::cerr<<"CUDA:"<<cudaGetErrorString(e)<<"\n";exit(-1);}
#define CUB_CHK(e)  if(e!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLAS err:"<<e<<"\n";exit(-1);}

// ═══════════════════════════════════════
// Helpers
// ═══════════════════════════════════════
static cublasHandle_t g_handle;
static void init() { CUB_CHK(cublasCreate(&g_handle)); }
static void fini() { CUB_CHK(cublasDestroy(g_handle)); }

static void warmup_sync() { CUDA_CHK(cudaDeviceSynchronize()); }

// ═══════════════════════════════════════
// Level-1 BLAS
// ═══════════════════════════════════════
void bench_scal(int N) {
    std::vector<float> x(N, 1.0f);
    float *dx; CUDA_CHK(cudaMalloc(&dx, N*4));
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    float alpha = 2.0f;

    nvtxRangePushA("L1_scal");
    CUB_CHK(cublasSscal(g_handle, N, &alpha, dx, 1));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dx));
    std::cout << "SUCCESS: scal N=" << N << std::endl;
}

void bench_copy(int N) {
    float *dx, *dy;
    CUDA_CHK(cudaMalloc(&dx, N*4)); CUDA_CHK(cudaMalloc(&dy, N*4));
    std::vector<float> x(N, 1.0f);
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));

    nvtxRangePushA("L1_copy");
    CUB_CHK(cublasScopy(g_handle, N, dx, 1, dy, 1));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: copy N=" << N << std::endl;
}

void bench_axpy(int N) {
    float *dx, *dy;
    CUDA_CHK(cudaMalloc(&dx, N*4)); CUDA_CHK(cudaMalloc(&dy, N*4));
    std::vector<float> x(N, 1.0f), y(N, 2.0f);
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy, y.data(), N*4, cudaMemcpyHostToDevice));
    float alpha = 2.0f;

    nvtxRangePushA("L1_axpy");
    CUB_CHK(cublasSaxpy(g_handle, N, &alpha, dx, 1, dy, 1));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: axpy N=" << N << std::endl;
}

void bench_dot(int N) {
    float *dx, *dy, result;
    CUDA_CHK(cudaMalloc(&dx, N*4)); CUDA_CHK(cudaMalloc(&dy, N*4));
    std::vector<float> x(N, 1.0f), y(N, 2.0f);
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy, y.data(), N*4, cudaMemcpyHostToDevice));

    nvtxRangePushA("L1_dot");
    CUB_CHK(cublasSdot(g_handle, N, dx, 1, dy, 1, &result));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: dot N=" << N << std::endl;
}

void bench_nrm2(int N) {
    float *dx, result;
    CUDA_CHK(cudaMalloc(&dx, N*4));
    std::vector<float> x(N, 3.0f);
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));

    nvtxRangePushA("L1_nrm2");
    CUB_CHK(cublasSnrm2(g_handle, N, dx, 1, &result));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dx));
    std::cout << "SUCCESS: nrm2 N=" << N << std::endl;
}

void bench_rot(int N) {
    float *dx, *dy;
    CUDA_CHK(cudaMalloc(&dx, N*4)); CUDA_CHK(cudaMalloc(&dy, N*4));
    std::vector<float> x(N, 1.0f), y(N, 2.0f);
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy, y.data(), N*4, cudaMemcpyHostToDevice));
    float c = 0.5f, s = 0.866f;

    nvtxRangePushA("L1_rot");
    CUB_CHK(cublasSrot(g_handle, N, dx, 1, dy, 1, &c, &s));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: rot N=" << N << std::endl;
}

void bench_rotm(int N) {
    float *dx, *dy;
    CUDA_CHK(cudaMalloc(&dx, N*4)); CUDA_CHK(cudaMalloc(&dy, N*4));
    std::vector<float> x(N, 1.0f), y(N, 2.0f);
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy, y.data(), N*4, cudaMemcpyHostToDevice));
    float param[5] = {1.0f, 0.5f, -0.5f, 0.5f, 0.0f};

    nvtxRangePushA("L1_rotm");
    CUB_CHK(cublasSrotm(g_handle, N, dx, 1, dy, 1, param));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: rotm N=" << N << std::endl;
}

// ═══════════════════════════════════════
// Level-2 BLAS
// ═══════════════════════════════════════
void bench_gemv(int M, int N) {
    float *dA, *dx, *dy;
    CUDA_CHK(cudaMalloc(&dA, M*N*4)); CUDA_CHK(cudaMalloc(&dx, N*4)); CUDA_CHK(cudaMalloc(&dy, M*4));
    std::vector<float> A(M*N, 0.5f), x(N, 1.0f), y(M, 0.0f);
    CUDA_CHK(cudaMemcpy(dA, A.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy, y.data(), M*4, cudaMemcpyHostToDevice));
    float a=1.0f, b=0.0f;

    // warmup
    CUB_CHK(cublasSgemv(g_handle, CUBLAS_OP_N, M, N, &a, dA, M, dx, 1, &b, dy, 1));
    warmup_sync();

    nvtxRangePushA("L2_gemv");
    CUB_CHK(cublasSgemv(g_handle, CUBLAS_OP_N, M, N, &a, dA, M, dx, 1, &b, dy, 1));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: gemv M=" << M << " N=" << N << std::endl;
}

void bench_symv(int N) {
    float *dA, *dx, *dy;
    CUDA_CHK(cudaMalloc(&dA, N*N*4)); CUDA_CHK(cudaMalloc(&dx, N*4)); CUDA_CHK(cudaMalloc(&dy, N*4));
    std::vector<float> A(N*N, 0.0f), x(N, 1.0f), y(N, 0.0f);
    for (int i=0;i<N;i++) { A[i*N+i]=2.0f; for (int j=0;j<i;j++) A[i*N+j]=A[j*N+i]=0.1f; }
    CUDA_CHK(cudaMemcpy(dA, A.data(), N*N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy, y.data(), N*4, cudaMemcpyHostToDevice));
    float a=1.0f, b=0.0f;

    CUB_CHK(cublasSsymv(g_handle, CUBLAS_FILL_MODE_LOWER, N, &a, dA, N, dx, 1, &b, dy, 1));
    warmup_sync();

    nvtxRangePushA("L2_symv");
    CUB_CHK(cublasSsymv(g_handle, CUBLAS_FILL_MODE_LOWER, N, &a, dA, N, dx, 1, &b, dy, 1));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: symv N=" << N << std::endl;
}

void bench_trmv(int N) {
    float *dA, *dx;
    CUDA_CHK(cudaMalloc(&dA, N*N*4)); CUDA_CHK(cudaMalloc(&dx, N*4));
    std::vector<float> A(N*N, 0.0f), x(N, 1.0f);
    for (int i=0;i<N;i++) for (int j=0;j<=i;j++) A[i*N+j]=(i==j)?2.0f:0.1f;
    CUDA_CHK(cudaMemcpy(dA, A.data(), N*N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));

    CUB_CHK(cublasStrmv(g_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, dA, N, dx, 1));
    warmup_sync();

    nvtxRangePushA("L2_trmv");
    CUB_CHK(cublasStrmv(g_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, N, dA, N, dx, 1));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dx));
    std::cout << "SUCCESS: trmv N=" << N << std::endl;
}

void bench_ger(int M, int N) {
    float *dA, *dx, *dy;
    CUDA_CHK(cudaMalloc(&dA, M*N*4)); CUDA_CHK(cudaMalloc(&dx, M*4)); CUDA_CHK(cudaMalloc(&dy, N*4));
    std::vector<float> A(M*N, 0.0f), x(M, 1.0f), y(N, 1.0f);
    CUDA_CHK(cudaMemcpy(dA, A.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx, x.data(), M*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy, y.data(), N*4, cudaMemcpyHostToDevice));
    float a=1.0f;

    CUB_CHK(cublasSger(g_handle, M, N, &a, dx, 1, dy, 1, dA, M));
    warmup_sync();

    nvtxRangePushA("L2_ger");
    CUB_CHK(cublasSger(g_handle, M, N, &a, dx, 1, dy, 1, dA, M));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout << "SUCCESS: ger M=" << M << " N=" << N << std::endl;
}

void bench_syr(int N) {
    float *dA, *dx;
    CUDA_CHK(cudaMalloc(&dA, N*N*4)); CUDA_CHK(cudaMalloc(&dx, N*4));
    std::vector<float> A(N*N, 0.0f), x(N, 1.0f);
    CUDA_CHK(cudaMemcpy(dA, A.data(), N*N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx, x.data(), N*4, cudaMemcpyHostToDevice));
    float a=1.0f;

    CUB_CHK(cublasSsyr(g_handle, CUBLAS_FILL_MODE_LOWER, N, &a, dx, 1, dA, N));
    warmup_sync();

    nvtxRangePushA("L2_syr");
    CUB_CHK(cublasSsyr(g_handle, CUBLAS_FILL_MODE_LOWER, N, &a, dx, 1, dA, N));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dx));
    std::cout << "SUCCESS: syr N=" << N << std::endl;
}

// ═══════════════════════════════════════
// Level-3 BLAS (补充 gemm/syrk/trsm 之外的)
// ═══════════════════════════════════════
void bench_symm(int M, int N) {
    float *dA, *dB, *dC;
    CUDA_CHK(cudaMalloc(&dA, M*M*4)); CUDA_CHK(cudaMalloc(&dB, M*N*4)); CUDA_CHK(cudaMalloc(&dC, M*N*4));
    std::vector<float> A(M*M, 0.0f), B(M*N, 1.0f), C(M*N, 0.0f);
    for (int i=0;i<M;i++) for (int j=0;j<=i;j++) A[i*M+j]=(i==j)?2.0f:0.1f;
    CUDA_CHK(cudaMemcpy(dA, A.data(), M*M*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, B.data(), M*N*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dC, C.data(), M*N*4, cudaMemcpyHostToDevice));
    float a=1.0f, b=0.0f;

    CUB_CHK(cublasSsymm(g_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, M, N, &a, dA, M, dB, M, &b, dC, M));
    warmup_sync();

    nvtxRangePushA("L3_symm");
    CUB_CHK(cublasSsymm(g_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, M, N, &a, dA, M, dB, M, &b, dC, M));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB)); CUDA_CHK(cudaFree(dC));
    std::cout << "SUCCESS: symm M=" << M << " N=" << N << std::endl;
}

void bench_syr2k(int N, int K) {
    float *dA, *dB, *dC;
    CUDA_CHK(cudaMalloc(&dA, N*K*4)); CUDA_CHK(cudaMalloc(&dB, N*K*4)); CUDA_CHK(cudaMalloc(&dC, N*N*4));
    std::vector<float> A(N*K, 1.0f), B(N*K, 0.5f), C(N*N, 0.0f);
    CUDA_CHK(cudaMemcpy(dA, A.data(), N*K*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, B.data(), N*K*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dC, C.data(), N*N*4, cudaMemcpyHostToDevice));
    float a=1.0f, b=0.0f;

    CUB_CHK(cublasSsyr2k(g_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K, &a, dA, K, dB, K, &b, dC, N));
    warmup_sync();

    nvtxRangePushA("L3_syr2k");
    CUB_CHK(cublasSsyr2k(g_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K, &a, dA, K, dB, K, &b, dC, N));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB)); CUDA_CHK(cudaFree(dC));
    std::cout << "SUCCESS: syr2k N=" << N << " K=" << K << std::endl;
}

void bench_trmm(int M, int N) {
    float *dA, *dB;
    CUDA_CHK(cudaMalloc(&dA, M*M*4)); CUDA_CHK(cudaMalloc(&dB, M*N*4));
    std::vector<float> A(M*M, 0.0f), B(M*N, 1.0f);
    for (int i=0;i<M;i++) for (int j=0;j<=i;j++) A[i*M+j]=(i==j)?2.0f:0.1f;
    CUDA_CHK(cudaMemcpy(dA, A.data(), M*M*4, cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dB, B.data(), M*N*4, cudaMemcpyHostToDevice));
    float a=1.0f;

    CUB_CHK(cublasStrmm(g_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &a, dA, M, dB, M, dB, M));
    warmup_sync();

    nvtxRangePushA("L3_trmm");
    CUB_CHK(cublasStrmm(g_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &a, dA, M, dB, M, dB, M));
    warmup_sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB));
    std::cout << "SUCCESS: trmm M=" << M << " N=" << N << std::endl;
}

// ═══════════════════════════════════════
// Main
// ═══════════════════════════════════════
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: benchmark_all.exe <op> <params...>\n"
                  << "L1: scal N | copy N | axpy N | dot N | nrm2 N | rot N | rotm N\n"
                  << "L2: gemv M N | symv N | trmv N | ger M N | syr N\n"
                  << "L3: gemm M N K | symm M N | syrk N K | syr2k N K | trmm M N\n";
        return 1;
    }

    init();
    std::string op = argv[1];

    if (op == "scal") bench_scal(std::stoi(argv[2]));
    else if (op == "copy") bench_copy(std::stoi(argv[2]));
    else if (op == "axpy") bench_axpy(std::stoi(argv[2]));
    else if (op == "dot")  bench_dot(std::stoi(argv[2]));
    else if (op == "nrm2") bench_nrm2(std::stoi(argv[2]));
    else if (op == "rot")  bench_rot(std::stoi(argv[2]));
    else if (op == "rotm") bench_rotm(std::stoi(argv[2]));
    else if (op == "gemv") bench_gemv(std::stoi(argv[2]), std::stoi(argv[3]));
    else if (op == "symv") bench_symv(std::stoi(argv[2]));
    else if (op == "trmv") bench_trmv(std::stoi(argv[2]));
    else if (op == "ger")  bench_ger(std::stoi(argv[2]), std::stoi(argv[3]));
    else if (op == "syr")  bench_syr(std::stoi(argv[2]));
    else if (op == "gemm") {
        int M=std::stoi(argv[2]), N=std::stoi(argv[3]), K=std::stoi(argv[4]);
        float *dA,*dB,*dC; float a=1,b=0;
        CUDA_CHK(cudaMalloc(&dA,M*K*4)); CUDA_CHK(cudaMalloc(&dB,K*N*4)); CUDA_CHK(cudaMalloc(&dC,M*N*4));
        std::vector<float> A(M*K,1), B(K*N,1), C(M*N,0);
        CUDA_CHK(cudaMemcpy(dA,A.data(),M*K*4,cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(dB,B.data(),K*N*4,cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(dC,C.data(),M*N*4,cudaMemcpyHostToDevice));
        CUB_CHK(cublasSgemm(g_handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,N,dA,K,&b,dC,N));
        warmup_sync();
        nvtxRangePushA("L3_gemm");
        CUB_CHK(cublasSgemm(g_handle,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB,N,dA,K,&b,dC,N));
        warmup_sync(); nvtxRangePop();
        CUDA_CHK(cudaFree(dA));CUDA_CHK(cudaFree(dB));CUDA_CHK(cudaFree(dC));
        std::cout<<"SUCCESS: gemm M="<<M<<" N="<<N<<" K="<<K<<std::endl;
    }
    else if (op == "symm") bench_symm(std::stoi(argv[2]), std::stoi(argv[3]));
    else if (op == "syrk") {
        int N=std::stoi(argv[2]), K=std::stoi(argv[3]);
        float *dA,*dC; float a=1,b=0;
        CUDA_CHK(cudaMalloc(&dA,N*K*4)); CUDA_CHK(cudaMalloc(&dC,N*N*4));
        std::vector<float> A(N*K,1), C(N*N,0);
        CUDA_CHK(cudaMemcpy(dA,A.data(),N*K*4,cudaMemcpyHostToDevice));
        CUDA_CHK(cudaMemcpy(dC,C.data(),N*N*4,cudaMemcpyHostToDevice));
        CUB_CHK(cublasSsyrk(g_handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,N,K,&a,dA,K,&b,dC,N));
        warmup_sync();
        nvtxRangePushA("L3_syrk");
        CUB_CHK(cublasSsyrk(g_handle,CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_N,N,K,&a,dA,K,&b,dC,N));
        warmup_sync(); nvtxRangePop();
        CUDA_CHK(cudaFree(dA));CUDA_CHK(cudaFree(dC));
        std::cout<<"SUCCESS: syrk N="<<N<<" K="<<K<<std::endl;
    }
    else if (op == "syr2k") bench_syr2k(std::stoi(argv[2]), std::stoi(argv[3]));
    else if (op == "trmm") bench_trmm(std::stoi(argv[2]), std::stoi(argv[3]));
    else { std::cerr << "Unknown op: " << op << std::endl; fini(); return 1; }

    fini();
    return 0;
}
