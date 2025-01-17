#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "coalesced_warp_2.cuh"
#include "coalesced_warpblock_3.cuh"
#include "cublas_0.cuh"
#include "naive_1.cuh"
#include "utils.cuh"
#include "vectorized_4.cuh"

/*
Benchmarks a kernel against cuBLAS for different sizes
*/
void benchmark_kernel_for_sizes(int minM, int maxM, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
    FILE *gflops_file = fopen("benchmarks/kernel_4_vs_cublas-gflops.txt", "w");
    FILE *memory_file = fopen("benchmarks/kernel_4_vs_cublas-memory.txt", "w");

    if (gflops_file == NULL) {
        perror("Error opening the file for GFLOPS.\n");
    }
    if (memory_file == NULL) {
        perror("Error opening the file for Memory Bandwidth.\n");
    }

    for (int M = minM; M <= maxM; M *= 2) {
        int N = 2 * M;  // matrix size (M, N)

        printf("------------ Running benchmark for M = %d ---------------\n", M);

        size_t matsize = M * N;  // (M, N)
        size_t vecsize = N;      // (N, 1)
        size_t mat_totalsize = matsize * sizeof(float);
        size_t vec_totalsize = vecsize * sizeof(float);

        // allocate host
        float *mat = (float *)malloc(mat_totalsize);
        float *vec = (float *)malloc(vec_totalsize);
        float *res = (float *)malloc(M * sizeof(float));

        for (size_t i = 0; i < matsize; i++) {
            mat[i] = random_normal_clamped(-10.f, 10.f);
            // hacky way to init the vector as well
            if (i < vecsize) {
                vec[i] = random_normal_clamped(-10.f, 10.f);
            }
        }

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        float ms = 0.0f;

        // allocate device
        float *matd, *vecd, *resd;
        cudaEventRecord(start);
        CUDA_CHECK(cudaMalloc((void **)&matd, mat_totalsize));
        CUDA_CHECK(cudaMalloc((void **)&vecd, vec_totalsize));
        CUDA_CHECK(cudaMalloc((void **)&resd, M * sizeof(float)));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf(">> GPU allocation time: %f ms\n", ms);

        // copy host to device
        cudaEventRecord(start);
        CUDA_CHECK(cudaMemcpy(matd, mat, mat_totalsize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vecd, vec, vec_totalsize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(resd, res, M * sizeof(float), cudaMemcpyHostToDevice));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf(">> Host to device transfer time: %f ms\n", ms);

        // run cuBLAS kernel and write results to file
        float mscub = run_kernel_cublas_sgemv(matd, vecd, resd, M, N, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
        float gflopscub = compute_gflops(M, N, mscub);
        float mem_bandcub = compute_peak_memory_bandwidth(M, N, mscub, THEORETICAL_MAX_MEMORY_BANDWIDTH);

        // run custom kernel and write results to file
        ms = run_kernel_vectorized_sgmev(matd, vecd, resd, M, N, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
        float gflops = compute_gflops(M, N, ms);
        float mem_band = compute_peak_memory_bandwidth(M, N, ms, THEORETICAL_MAX_MEMORY_BANDWIDTH);

        fprintf(gflops_file, "%d %f %f\n", M, gflops, gflopscub);
        fprintf(memory_file, "%d %f %f\n", M, mem_band, mem_bandcub);

        // copy device to host
        cudaEventRecord(start);
        CUDA_CHECK(cudaMemcpy(res, resd, M * sizeof(float), cudaMemcpyDeviceToHost));
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        printf(">> Device to host transfer time: %f ms\n", ms);

        // cleanup
        cudaFree(matd);
        cudaFree(vecd);
        cudaFree(resd);
        free(mat);
        free(vec);
        free(res);
    }

    fclose(gflops_file);
    fclose(memory_file);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cudaCores = prop.multiProcessorCount * 128;
    float clockGHz = prop.clockRate / 1e6;

    float THEORETICAL_MAX_GFLOPS = cudaCores * clockGHz * 2;
    float THEORETICAL_MAX_MEMORY_BANDWIDTH = (2 * prop.memoryClockRate * prop.memoryBusWidth) / (8.0 * 1e6);

    benchmark_kernel_for_sizes(4096, 4096, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);
}