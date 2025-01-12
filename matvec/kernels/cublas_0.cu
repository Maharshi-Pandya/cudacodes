#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.cuh"

/*
CuBLAS matrix vector multiplication for the baseline scores.
We simply run the Sgemv function that cuBLAS provides.
*/
void run_kernel_cublas_sgemv(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    // create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Sgemv: y = (alpha * A * x) + (beta * y)
    float alpha = 1.0f, beta = 0.0f;
    cudaEventRecord(start);
    cublasSgemv(handle, CUBLAS_OP_N, M, N, &alpha, matd, M, vecd, 1, &beta, resd, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    float gflops = compute_gflops(M, N, ms);
    printf(">> CuBLAS sgemv execution time: %f ms\n", ms);
    printf(">> CuBLAS sgemv (GFLOPS): %f\n", gflops);
    printf(">> Theoretical max (GFLOPS): %f\n", THEORETICAL_MAX_GFLOPS);
    printf(">> Maximum memory bandwidth: %f GB/s\n", THEORETICAL_MAX_MEMORY_BANDWIDTH);
    printf(">> CuBLAS sgemv achieves %f %% of peak GFLOPS\n", compute_peak_gflops(gflops));
    printf(">> CuBLAS sgemv achieves %f %% of peak Memory Bandwidth\n", compute_peak_memory_bandwidth(M, N, ms));

    cublasDestroy(handle);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
