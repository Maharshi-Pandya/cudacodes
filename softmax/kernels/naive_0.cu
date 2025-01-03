#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

/*
This kernel implements a naive softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
One thread processes one entire row, and thus this kernel will be the slowest
since we aren't exploiting parallelism capabilities of GPUs that much.
We are only parallelizing over the rows.
*/
__global__ void softmax_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        // max
        float m = -1 * INFINITY;
        // norm factor
        float L = 0.0f;

        // 3 passes (not optimal)
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            m = max(m, matd[i]);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            L += expf(matd[i] - m);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(matd[i] - m) / L;
        }
    }
}

/*
Runs the naive softmax kernel: `id = 0`
*/
void run_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(1024);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_0<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}