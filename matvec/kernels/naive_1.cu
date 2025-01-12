#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "utils.cuh"

/*
Naive Sgemv kernel

- Each thread calculates one element of the output vector
- The row index is calculated using block index and thread index
- Uses linearized indexing
*/
__global__ void naive_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        for (int col = 0; col < N; col++) {
            sum += matd[row * N + col] * vecd[col];
        }
        resd[row] = sum;
    }
}

/*
Runs the naive Sgemv kernel.
*/
void run_kernel_naive_sgemv(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N) {
    dim3 block_size(1024);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    naive_sgemv_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float gflops = compute_gflops(M, N, ms);
    printf(">> Naive sgemv execution time: %f ms\n", ms);
    printf(">> Naive sgemv (GFLOPS): %f\n", gflops);
    printf(">> Theoretical max (GFLOPS): %f\n", THEORETICAL_MAX_GFLOPS);
    printf(">> Maximum memory bandwidth: %f GB/s\n", THEORETICAL_MAX_MEMORY_BANDWIDTH);
    printf(">> Naive sgemv achieves %f %% of peak GFLOPS\n", compute_peak_gflops(gflops));
    printf(">> Naive sgemv achieves %f %% of peak Memory Bandwidth\n", compute_peak_memory_bandwidth(M, N, ms));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}
