/*
 * benchmark_extra.cu — 稀疏 + LAPACK + 带状 + 转置 全算子 benchmark
 * 编译: nvcc -O3 -lcublas -lcusparse -lcusolver benchmark_extra.cu -o benchmark_extra.exe -arch=sm_89
 */
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <nvtx3/nvToolsExt.h>

#define CUDA_CHK(e) if(e!=cudaSuccess){std::cerr<<"CUDA:"<<cudaGetErrorString(e)<<"\n";exit(-1);}
#define CUB_CHK(e)  if(e!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLAS:"<<e<<"\n";exit(-1);}
#define CUSP_CHK(e) if(e!=CUSPARSE_STATUS_SUCCESS){std::cerr<<"cuSPARSE:"<<e<<"\n";exit(-1);}
#define CUSOL_CHK(e) if(e!=CUSOLVER_STATUS_SUCCESS){std::cerr<<"cuSOLVER:"<<e<<"\n";exit(-1);}

static cublasHandle_t blas_h;
static cusparseHandle_t sp_h;
static cusolverDnHandle_t sol_h;

static void init_all() {
    CUB_CHK(cublasCreate(&blas_h));
    CUSP_CHK(cusparseCreate(&sp_h));
    CUSOL_CHK(cusolverDnCreate(&sol_h));
}
static void fini_all() {
    cublasDestroy(blas_h);
    cusparseDestroy(sp_h);
    cusolverDnDestroy(sol_h);
}
static void sync() { CUDA_CHK(cudaDeviceSynchronize()); }

// ═══════════════════════════════════════
// gbmv (带状 GEMV)
// ═══════════════════════════════════════
void bench_gbmv(int N) {
    int M=N, KL=N/4, KU=N/4;
    int lda = KL+KU+1;
    std::vector<float> A(lda*N, 0.5f), x(N, 1.0f), y(M, 0.0f);
    float *dA, *dx, *dy;
    CUDA_CHK(cudaMalloc(&dA,lda*N*4)); CUDA_CHK(cudaMalloc(&dx,N*4)); CUDA_CHK(cudaMalloc(&dy,M*4));
    CUDA_CHK(cudaMemcpy(dA,A.data(),lda*N*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx,x.data(),N*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy,y.data(),M*4,cudaMemcpyHostToDevice));
    float a=1,b=0;
    CUB_CHK(cublasSgbmv(blas_h,CUBLAS_OP_N,M,N,KL,KU,&a,dA,lda,dx,1,&b,dy,1)); sync();
    nvtxRangePushA("L2_gbmv");
    CUB_CHK(cublasSgbmv(blas_h,CUBLAS_OP_N,M,N,KL,KU,&a,dA,lda,dx,1,&b,dy,1)); sync();
    nvtxRangePop();
    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout<<"SUCCESS: gbmv M="<<M<<" N="<<N<<" KL="<<KL<<" KU="<<KU<<std::endl;
}

// ═══════════════════════════════════════
// SpMV (CSR 稀疏矩阵-向量乘)
// ═══════════════════════════════════════
void bench_spmv(int N, int nnz_per_row) {
    int nnz = N * nnz_per_row;
    std::vector<int> h_ptr(N+1), h_idx(nnz);
    std::vector<float> h_val(nnz, 1.0f), h_x(N, 1.0f), h_y(N, 0.0f);
    for(int i=0;i<=N;i++) h_ptr[i]=i*nnz_per_row;
    for(int i=0;i<N;i++) for(int j=0;j<nnz_per_row;j++) h_idx[i*nnz_per_row+j]=(i+j)%N;

    int *d_ptr,*d_idx; float *d_val,*dx,*dy;
    CUDA_CHK(cudaMalloc(&d_ptr,(N+1)*4)); CUDA_CHK(cudaMalloc(&d_idx,nnz*4));
    CUDA_CHK(cudaMalloc(&d_val,nnz*4)); CUDA_CHK(cudaMalloc(&dx,N*4)); CUDA_CHK(cudaMalloc(&dy,N*4));
    CUDA_CHK(cudaMemcpy(d_ptr,h_ptr.data(),(N+1)*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_idx,h_idx.data(),nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_val,h_val.data(),nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx,h_x.data(),N*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dy,h_y.data(),N*4,cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    CUSP_CHK(cusparseCreateCsr(&matA,N,N,nnz,d_ptr,d_idx,d_val,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
    CUSP_CHK(cusparseCreateDnVec(&vecX,N,dx,CUDA_R_32F));
    CUSP_CHK(cusparseCreateDnVec(&vecY,N,dy,CUDA_R_32F));

    float a=1,b=0; size_t bufSize;
    CUSP_CHK(cusparseSpMV_bufferSize(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a,matA,vecX,&b,vecY,CUDA_R_32F,CUSPARSE_SPMV_ALG_DEFAULT,&bufSize));
    void* dBuf; CUDA_CHK(cudaMalloc(&dBuf,bufSize));

    CUSP_CHK(cusparseSpMV(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a,matA,vecX,&b,vecY,CUDA_R_32F,CUSPARSE_SPMV_ALG_DEFAULT,dBuf)); sync();
    nvtxRangePushA("SP_SpMV");
    CUSP_CHK(cusparseSpMV(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a,matA,vecX,&b,vecY,CUDA_R_32F,CUSPARSE_SPMV_ALG_DEFAULT,dBuf)); sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dBuf));
    cusparseDestroySpMat(matA); cusparseDestroyDnVec(vecX); cusparseDestroyDnVec(vecY);
    CUDA_CHK(cudaFree(d_ptr)); CUDA_CHK(cudaFree(d_idx)); CUDA_CHK(cudaFree(d_val));
    CUDA_CHK(cudaFree(dx)); CUDA_CHK(cudaFree(dy));
    std::cout<<"SUCCESS: SpMV N="<<N<<" nnz="<<nnz<<std::endl;
}

// ═══════════════════════════════════════
// sparse trsv (CSC 稀疏三角求解)
// ═══════════════════════════════════════
void bench_sparse_trsv(int N, int nnz_per_col) {
    int nnz_max = N * nnz_per_col;
    std::vector<int> h_ptr(N+1), h_idx(nnz_max);
    std::vector<float> h_val(nnz_max, 1.0f), h_x(N, 1.0f);
    // Build CSR lower-triangular: row i has entries at cols j <= i
    for(int i=0;i<=N;i++){ h_ptr[i]=0; for(int j=0;j<nnz_per_col;j++) if(i+j<N) h_ptr[i]++; }
    for(int i=1;i<=N;i++) h_ptr[i]+=h_ptr[i-1];
    for(int i=0;i<N;i++){
        int pos=h_ptr[i];
        for(int j=0;j<nnz_per_col&&i+j<N;j++){
            h_idx[pos]=i+j; h_val[pos]=(i+j==i)?2.0f:0.1f; pos++;
        }
    }

    int actual_nnz = h_ptr[N];
    int *d_ptr,*d_idx; float *d_val,*dx;
    CUDA_CHK(cudaMalloc(&d_ptr,(N+1)*4)); CUDA_CHK(cudaMalloc(&d_idx,actual_nnz*4));
    CUDA_CHK(cudaMalloc(&d_val,actual_nnz*4)); CUDA_CHK(cudaMalloc(&dx,N*4));
    CUDA_CHK(cudaMemcpy(d_ptr,h_ptr.data(),(N+1)*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_idx,h_idx.data(),actual_nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_val,h_val.data(),actual_nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(dx,h_x.data(),N*4,cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matL;
    cusparseDnVecDescr_t vecX;
    // Use CSR (required by cuSPARSE SpSV)
    CUSP_CHK(cusparseCreateCsr(&matL,N,N,actual_nnz,d_ptr,d_idx,d_val,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
    CUSP_CHK(cusparseCreateDnVec(&vecX,N,dx,CUDA_R_32F));

    float a=1;
    cusparseSpSVDescr_t svDescr;
    CUSP_CHK(cusparseSpSV_createDescr(&svDescr));

    size_t bufSize;
    CUSP_CHK(cusparseSpSV_bufferSize(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a,matL,vecX,vecX,CUDA_R_32F,CUSPARSE_SPSV_ALG_DEFAULT,svDescr,&bufSize));
    void* dBuf; CUDA_CHK(cudaMalloc(&dBuf,bufSize));

    CUSP_CHK(cusparseSpSV_analysis(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a,matL,vecX,vecX,CUDA_R_32F,CUSPARSE_SPSV_ALG_DEFAULT,svDescr,dBuf));
    CUSP_CHK(cusparseSpSV_solve(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a,matL,vecX,vecX,CUDA_R_32F,CUSPARSE_SPSV_ALG_DEFAULT,svDescr)); sync();

    nvtxRangePushA("SP_spsv");
    CUSP_CHK(cusparseSpSV_solve(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        &a,matL,vecX,vecX,CUDA_R_32F,CUSPARSE_SPSV_ALG_DEFAULT,svDescr)); sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dBuf));
    cusparseSpSV_destroyDescr(svDescr);
    cusparseDestroySpMat(matL); cusparseDestroyDnVec(vecX);
    CUDA_CHK(cudaFree(d_ptr)); CUDA_CHK(cudaFree(d_idx)); CUDA_CHK(cudaFree(d_val)); CUDA_CHK(cudaFree(dx));
    std::cout<<"SUCCESS: sparse_trsv N="<<N<<" nnz="<<actual_nnz<<std::endl;
}

// ═══════════════════════════════════════
// spgemm (CSR*CSR sparse matrix multiply)
// ═══════════════════════════════════════
void bench_spgemm(int N, int nnz_per_row) {
    int nnz = N * nnz_per_row;
    std::vector<int> h_ptr(N+1), h_idx(nnz);
    std::vector<float> h_val(nnz, 0.5f);
    for(int i=0;i<=N;i++) h_ptr[i]=i*nnz_per_row;
    for(int i=0;i<N;i++) for(int j=0;j<nnz_per_row;j++) h_idx[i*nnz_per_row+j]=(i+j)%N;

    int *d_ptrA,*d_idxA,*d_ptrB,*d_idxB,*d_ptrC,*d_idxC;
    float *d_valA,*d_valB,*d_valC;
    CUDA_CHK(cudaMalloc(&d_ptrA,(N+1)*4)); CUDA_CHK(cudaMalloc(&d_idxA,nnz*4)); CUDA_CHK(cudaMalloc(&d_valA,nnz*4));
    CUDA_CHK(cudaMalloc(&d_ptrB,(N+1)*4)); CUDA_CHK(cudaMalloc(&d_idxB,nnz*4)); CUDA_CHK(cudaMalloc(&d_valB,nnz*4));
    CUDA_CHK(cudaMemcpy(d_ptrA,h_ptr.data(),(N+1)*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_idxA,h_idx.data(),nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_valA,h_val.data(),nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_ptrB,h_ptr.data(),(N+1)*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_idxB,h_idx.data(),nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_valB,h_val.data(),nnz*4,cudaMemcpyHostToDevice));

    cusparseSpMatDescr_t matA,matB,matC;
    CUSP_CHK(cusparseCreateCsr(&matA,N,N,nnz,d_ptrA,d_idxA,d_valA,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
    CUSP_CHK(cusparseCreateCsr(&matB,N,N,nnz,d_ptrB,d_idxB,d_valB,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));
    // empty matC
    CUDA_CHK(cudaMalloc(&d_ptrC,(N+1)*4)); CUDA_CHK(cudaMalloc(&d_idxC,1)); CUDA_CHK(cudaMalloc(&d_valC,1));
    CUSP_CHK(cusparseCreateCsr(&matC,N,N,0,d_ptrC,d_idxC,d_valC,
        CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));

    float a=1,b=0;
    cusparseSpGEMMDescr_t gemmDesc;
    CUSP_CHK(cusparseSpGEMM_createDescr(&gemmDesc));

    size_t buf1,buf2;
    CUSP_CHK(cusparseSpGEMM_workEstimation(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,&a,matA,matB,&b,matC,
        CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,gemmDesc,&buf1,nullptr));
    void* dBuf1; CUDA_CHK(cudaMalloc(&dBuf1,buf1));
    CUSP_CHK(cusparseSpGEMM_workEstimation(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,&a,matA,matB,&b,matC,
        CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,gemmDesc,&buf1,dBuf1));

    CUSP_CHK(cusparseSpGEMM_compute(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,&a,matA,matB,&b,matC,
        CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,gemmDesc,&buf2,nullptr));
    void* dBuf2; CUDA_CHK(cudaMalloc(&dBuf2,buf2));

    nvtxRangePushA("SP_spgemm");
    CUSP_CHK(cusparseSpGEMM_compute(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,&a,matA,matB,&b,matC,
        CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,gemmDesc,&buf2,dBuf2));
    CUSP_CHK(cusparseSpGEMM_copy(sp_h,CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,&a,matA,matB,&b,matC,
        CUDA_R_32F,CUSPARSE_SPGEMM_DEFAULT,gemmDesc));
    sync(); nvtxRangePop();

    CUDA_CHK(cudaFree(dBuf1)); CUDA_CHK(cudaFree(dBuf2));
    cusparseSpGEMM_destroyDescr(gemmDesc);
    cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
    CUDA_CHK(cudaFree(d_ptrA)); CUDA_CHK(cudaFree(d_idxA)); CUDA_CHK(cudaFree(d_valA));
    CUDA_CHK(cudaFree(d_ptrB)); CUDA_CHK(cudaFree(d_idxB)); CUDA_CHK(cudaFree(d_valB));
    CUDA_CHK(cudaFree(d_ptrC)); CUDA_CHK(cudaFree(d_idxC)); CUDA_CHK(cudaFree(d_valC));
    std::cout<<"SUCCESS: spgemm N="<<N<<std::endl;
}

// ═══════════════════════════════════════
// dense transpose
// ═══════════════════════════════════════
void bench_transpose_dense(int M, int N) {
    float *dA, *dB;
    CUDA_CHK(cudaMalloc(&dA,M*N*4)); CUDA_CHK(cudaMalloc(&dB,M*N*4));
    std::vector<float> A(M*N,1.0f);
    CUDA_CHK(cudaMemcpy(dA,A.data(),M*N*4,cudaMemcpyHostToDevice));
    float a=1,b=0;

    CUB_CHK(cublasSgeam(blas_h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,&a,dA,M,&b,nullptr,N,dB,N)); sync();
    nvtxRangePushA("T_dense");
    CUB_CHK(cublasSgeam(blas_h,CUBLAS_OP_T,CUBLAS_OP_N,N,M,&a,dA,M,&b,nullptr,N,dB,N)); sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dB));
    std::cout<<"SUCCESS: transpose_dense M="<<M<<" N="<<N<<std::endl;
}

// ═══════════════════════════════════════
// sparse transpose (CSR->CSC)
// ═══════════════════════════════════════
void bench_transpose_sparse(int N, int nnz_per_row) {
    int nnz = N * nnz_per_row;
    std::vector<int> h_ptr(N+1), h_idx(nnz);
    std::vector<float> h_val(nnz, 1.0f);
    for(int i=0;i<=N;i++) h_ptr[i]=i*nnz_per_row;
    for(int i=0;i<N;i++) for(int j=0;j<nnz_per_row;j++) h_idx[i*nnz_per_row+j]=(i+j)%N;

    int *d_ptr,*d_idx,*d_ptrT,*d_idxT;
    float *d_val,*d_valT;
    CUDA_CHK(cudaMalloc(&d_ptr,(N+1)*4)); CUDA_CHK(cudaMalloc(&d_idx,nnz*4)); CUDA_CHK(cudaMalloc(&d_val,nnz*4));
    CUDA_CHK(cudaMalloc(&d_ptrT,(N+1)*4)); CUDA_CHK(cudaMalloc(&d_idxT,nnz*4)); CUDA_CHK(cudaMalloc(&d_valT,nnz*4));
    CUDA_CHK(cudaMemcpy(d_ptr,h_ptr.data(),(N+1)*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_idx,h_idx.data(),nnz*4,cudaMemcpyHostToDevice));
    CUDA_CHK(cudaMemcpy(d_val,h_val.data(),nnz*4,cudaMemcpyHostToDevice));

    size_t bufSize;
    CUSP_CHK(cusparseCsr2cscEx2_bufferSize(sp_h,N,N,nnz,d_val,d_ptr,d_idx,
        d_valT,d_ptrT,d_idxT,CUDA_R_32F,CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,&bufSize));
    void* dBuf; CUDA_CHK(cudaMalloc(&dBuf,bufSize));

    nvtxRangePushA("T_sparse");
    CUSP_CHK(cusparseCsr2cscEx2(sp_h,N,N,nnz,d_val,d_ptr,d_idx,
        d_valT,d_ptrT,d_idxT,CUDA_R_32F,CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,CUSPARSE_CSR2CSC_ALG1,dBuf)); sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dBuf));
    CUDA_CHK(cudaFree(d_ptr)); CUDA_CHK(cudaFree(d_idx)); CUDA_CHK(cudaFree(d_val));
    CUDA_CHK(cudaFree(d_ptrT)); CUDA_CHK(cudaFree(d_idxT)); CUDA_CHK(cudaFree(d_valT));
    std::cout<<"SUCCESS: transpose_sparse N="<<N<<" nnz="<<nnz<<std::endl;
}

// ═══════════════════════════════════════
// LAPACK: potrf / getrf / geqrf (cuSOLVER)
// ═══════════════════════════════════════
void bench_potrf(int N) {
    float *dA; int lda=N, Lwork;
    CUDA_CHK(cudaMalloc(&dA,N*N*4));
    std::vector<float> A(N*N,0.0f);
    for(int i=0;i<N;i++){A[i*N+i]=N*1.0f; for(int j=0;j<i;j++) A[i*N+j]=A[j*N+i]=0.5f;}
    CUDA_CHK(cudaMemcpy(dA,A.data(),N*N*4,cudaMemcpyHostToDevice));

    CUSOL_CHK(cusolverDnSpotrf_bufferSize(sol_h,CUBLAS_FILL_MODE_LOWER,N,dA,lda,&Lwork));
    int* dInfo; float* dWork;
    CUDA_CHK(cudaMalloc(&dInfo,4)); CUDA_CHK(cudaMalloc(&dWork,Lwork*4));

    CUSOL_CHK(cusolverDnSpotrf(sol_h,CUBLAS_FILL_MODE_LOWER,N,dA,lda,dWork,Lwork,dInfo)); sync();
    nvtxRangePushA("LAP_potrf");
    CUSOL_CHK(cusolverDnSpotrf(sol_h,CUBLAS_FILL_MODE_LOWER,N,dA,lda,dWork,Lwork,dInfo)); sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dInfo)); CUDA_CHK(cudaFree(dWork));
    std::cout<<"SUCCESS: potrf N="<<N<<std::endl;
}

void bench_getrf(int N) {
    float *dA; int lda=N, Lwork;
    CUDA_CHK(cudaMalloc(&dA,N*N*4));
    std::vector<float> A(N*N,0.0f);
    for(int i=0;i<N;i++){A[i*N+i]=2.0f; for(int j=0;j<N;j++) if(i!=j)A[i*N+j]=0.1f;}
    CUDA_CHK(cudaMemcpy(dA,A.data(),N*N*4,cudaMemcpyHostToDevice));

    CUSOL_CHK(cusolverDnSgetrf_bufferSize(sol_h,N,N,dA,lda,&Lwork));
    int* dIpiv,*dInfo; float* dWork;
    CUDA_CHK(cudaMalloc(&dIpiv,N*4)); CUDA_CHK(cudaMalloc(&dInfo,4)); CUDA_CHK(cudaMalloc(&dWork,Lwork*4));

    CUSOL_CHK(cusolverDnSgetrf(sol_h,N,N,dA,lda,dWork,dIpiv,dInfo)); sync();
    nvtxRangePushA("LAP_getrf");
    CUSOL_CHK(cusolverDnSgetrf(sol_h,N,N,dA,lda,dWork,dIpiv,dInfo)); sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dIpiv)); CUDA_CHK(cudaFree(dInfo)); CUDA_CHK(cudaFree(dWork));
    std::cout<<"SUCCESS: getrf N="<<N<<std::endl;
}

void bench_geqrf(int M, int N) {
    float *dA,*dTau; int lda=M, Lwork;
    CUDA_CHK(cudaMalloc(&dA,M*N*4)); CUDA_CHK(cudaMalloc(&dTau,std::min(M,N)*4));
    std::vector<float> A(M*N,0.0f);
    for(int i=0;i<M*N;i++) A[i]=(i%(M+1)==0)?2.0f:0.1f;
    CUDA_CHK(cudaMemcpy(dA,A.data(),M*N*4,cudaMemcpyHostToDevice));

    CUSOL_CHK(cusolverDnSgeqrf_bufferSize(sol_h,M,N,dA,lda,&Lwork));
    int* dInfo; float* dWork;
    CUDA_CHK(cudaMalloc(&dInfo,4)); CUDA_CHK(cudaMalloc(&dWork,Lwork*4));

    CUSOL_CHK(cusolverDnSgeqrf(sol_h,M,N,dA,lda,dTau,dWork,Lwork,dInfo)); sync();
    nvtxRangePushA("LAP_geqrf");
    CUSOL_CHK(cusolverDnSgeqrf(sol_h,M,N,dA,lda,dTau,dWork,Lwork,dInfo)); sync();
    nvtxRangePop();

    CUDA_CHK(cudaFree(dA)); CUDA_CHK(cudaFree(dTau)); CUDA_CHK(cudaFree(dInfo)); CUDA_CHK(cudaFree(dWork));
    std::cout<<"SUCCESS: geqrf M="<<M<<" N="<<N<<std::endl;
}

// ═══════════════════════════════════════
int main(int argc, char* argv[]) {
    if(argc<2){
        std::cerr<<"Usage: benchmark_extra.exe <op> <params...>\n"
                 <<"  gbmv N | spmv N nnz_per_row | sptrsv N nnz_per_col\n"
                 <<"  spgemm N nnz_per_row | t_dense M N | t_sparse N nnz_per_row\n"
                 <<"  potrf N | getrf N | geqrf M N\n";
        return 1;
    }
    init_all();
    std::string op=argv[1];

    if(op=="gbmv") bench_gbmv(std::stoi(argv[2]));
    else if(op=="spmv") bench_spmv(std::stoi(argv[2]),std::stoi(argv[3]));
    else if(op=="sptrsv") bench_sparse_trsv(std::stoi(argv[2]),std::stoi(argv[3]));
    else if(op=="spgemm") bench_spgemm(std::stoi(argv[2]),std::stoi(argv[3]));
    else if(op=="t_dense") bench_transpose_dense(std::stoi(argv[2]),std::stoi(argv[3]));
    else if(op=="t_sparse") bench_transpose_sparse(std::stoi(argv[2]),std::stoi(argv[3]));
    else if(op=="potrf") bench_potrf(std::stoi(argv[2]));
    else if(op=="getrf") bench_getrf(std::stoi(argv[2]));
    else if(op=="geqrf") bench_geqrf(std::stoi(argv[2]),std::stoi(argv[3]));
    else {std::cerr<<"Unknown op:"<<op<<"\n"; fini_all(); return 1;}

    fini_all();
    return 0;
}
