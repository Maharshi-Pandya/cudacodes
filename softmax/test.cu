#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"
#include "naive.cuh"

#define M_PI 3.14159265f

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
    int M = 1024;
    int N = 32768;
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

    run_kernel_0(matd, resd, M, N);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(res, resd, totalsize, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // correctness check on the first row
    // the output should be 1.0 (or a number very close to it)
    // TODO: add full correctness check
    float sum = 0.f;
    for (int i = 0; i < N; i++) {
        sum += res[i];
    }
    printf("\nSum of the 1st row of softmax result: %f\n", sum);

    free(mat);
    free(res);
    cudaFree(matd);
    cudaFree(resd);
}