#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
One thread processes one entire row, but instead of 3 passes we do only 2 passes.
This is possible due to the property of exponentials.
We are parallelizing over the rows.
*/
__global__ void softmax_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float m = -1 * INFINITY;
        float L = 0.0f;

        // compute max and norm factor in one pass only
        // by exploiting the property of exponentials
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            float curr = matd[i];
            if (curr > m) {
                L = L * expf(m - curr);
                m = curr;
            }
            L += expf(curr - m);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(matd[i] - m) / L;
        }
    }
}

/*
Runs the online softmax kernel: `id = 1`
*/
void run_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(1024);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_1<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}