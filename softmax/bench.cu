#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.cuh"
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

/*
Benchmarks a kernel for different sizes
*/
void benchmark_kernel_for_sizes(int minN, int maxN) {
    FILE *exec_time_file = fopen("benchmarks/exec_time_ms_cuda.txt", "w");

    if (exec_time_file == NULL) {
        perror("Error opening the file for GFLOPS.\n");
    }

    for (int N = minN; N < maxN; N *= 2) {
        int M = 1024;  // matrix size (M, N)

        printf("------------ Running CUDA softmax benchmark for MxN = (%d, %d) -------------\n", M, N);

        int matsize = M * N;
        int totalsize = matsize * sizeof(float);

        // allocate and initialize host matrix
        float *mat = (float *)malloc(totalsize);
        float *res = (float *)malloc(totalsize);
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

        // run softmax kernel
        ms = run_kernel_4(matd, resd, M, N);

        fprintf(exec_time_file, "%d %f\n", M, ms);

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

    fclose(exec_time_file);
}

int main() {
    benchmark_kernel_for_sizes(2048, 262144);
}