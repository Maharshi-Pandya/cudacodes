#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "blocktiling_5.cuh"
#include "cuda_utils.cuh"
#include "naive_0.cuh"
#include "online_1.cuh"
#include "sharedmem_2.cuh"
#include "shfl_3.cuh"
#include "vectorized_4.cuh"

/*
Helper function to generate a clamped random number sampled from a
normal distribution with mean 0 and std 1
*/
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

int main() {
    int M = 4096;
    int N = 4096;
    int matsize = M * N;
    int totalsize = matsize * sizeof(float);

    // allocate and initialize host matrix
    float* mat = (float*)malloc(totalsize);
    float* res = (float*)malloc(totalsize);
    for (int i = 0; i < matsize; i++) {
        mat[i] = random_normal_clamped(-10, 10);
    }

    float *matd, *resd;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&matd, totalsize));
    CUDA_CHECK(cudaMalloc(&resd, totalsize));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(matd, mat, totalsize, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    run_kernel_4(matd, resd, M, N);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(res, resd, totalsize, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    free(mat);
    free(res);
    cudaFree(matd);
    cudaFree(resd);
}