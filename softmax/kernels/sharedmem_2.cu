#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
In this, we handle each row with a block where the threads within one block work together
to process one row (max and norm factor). Each thread will process some elements
and will contains its local max and local norm in shared memory. Then, we perform reduction
operations to compute the final max and norm factor. Also, we compute maxes and norms
in one pass itself.
*/
__global__ void softmax_kernel_2(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // edge condition (we don't process further)
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // compute local max and norm for each thread
    // and then finally have a sync barrier before moving on
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // each thread will have its own local max
    // we store it in the tid of the shared memory
    smem[tid] = local_max;
    __syncthreads();

    // block-level reduction in O(log(N)) time over all threads
    // is faster than linear reduction over all threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        // sync barrier before next iteration to ensure correctness
        __syncthreads();
    }

    // the first element after max reduction from all threads
    // will contain the global max for the row
    float row_max = smem[0];
    __syncthreads();

    // each thread will have its own local norm
    // we will store the corrected local norm in the shared memory
    // again, exploits property of exponentials
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    // sum reduction similar to above for global norm factor
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

/*
Runs the online softmax kernel: `id = 2`
*/
void run_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(1024);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_2<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}