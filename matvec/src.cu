#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "coalesced_warp_2.cuh"
#include "coalesced_warpblock_3.cuh"
#include "cublas_0.cuh"
#include "naive_1.cuh"
#include "utils.cuh"

int main() {
    int M = 4096;
    int N = 4096;

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

    run_kernel_coalesced_warpblock_sgmev(matd, vecd, resd, M, N);

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