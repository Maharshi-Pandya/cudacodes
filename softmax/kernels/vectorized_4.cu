#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
Instead of accessing shared memory and having sync barrier overhead, we will use warp-level primitives (then
block-level) for performing max and sum reductions. The benefit is: it is faster than shared
memory access and also does not need syncing since each warp (group of 32 threads) execute
an instuction parallely on GPU so no chance of race conditions.

We will also use vectorized loads and stores.
*/
__global__ void softmax_kernel_4(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    assert(N % 4 == 0);

    // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    // number of threads in a warp
    unsigned int warp_size = 32;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // cast as float4
    int n_float4s = N / 4;
    float4* input_row_vec = reinterpret_cast<float4*>(input_row);
    float4* output_row_vec = reinterpret_cast<float4*>(output_row);

    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];
        // Process each component
        if (elem.x > local_max) {
            local_norm *= expf(local_max - elem.x);
            local_max = elem.x;
        }
        local_norm += expf(elem.x - local_max);

        if (elem.y > local_max) {
            local_norm *= expf(local_max - elem.y);
            local_max = elem.y;
        }
        local_norm += expf(elem.y - local_max);

        if (elem.z > local_max) {
            local_norm *= expf(local_max - elem.z);
            local_max = elem.z;
        }
        local_norm += expf(elem.z - local_max);

        if (elem.w > local_max) {
            local_norm *= expf(local_max - elem.w);
            local_max = elem.w;
        }
        local_norm += expf(elem.w - local_max);
    }
    __syncthreads();

    // each thread will have its own local max
    // we store it in shared memory for reduction
    // smem[tid] = local_max;
    // __syncthreads();

    // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
    // note: if there are 256 threads in one block (8 warps of 32 threads each)
    // the following for loop reduces the value in all the 8 warps
    // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
    // float val = smem[tid];
    float val = local_max;
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // when blockDim is greater than 32, we need to do a block level reduction
    // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
    // the global max will be stored in the first warp
    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            // which warp are we at?
            // store the value in its first thread index
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        // first warp will do global reduction only
        // this is possible because we stored the values in the shared memory
        // so the threads in the first warp will read from it and then reduce
        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        // this is for when the number of threads in a block are not
        // greater than the warp size, in that case we already reduced
        // so we can store the value
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    // we got the global row max now
    float row_max = smem[0];
    __syncthreads();

    // each thread will have its own local_norm
    // we will store the corrected local_norm in the shared memory
    // smem[tid] = local_norm * expf(local_max - row_max);
    // __syncthreads();

    // same reduction algorithm as above, but instead of max reduction
    // we do a sum reduction i.e. we accumulate the values
    // val = smem[tid];
    val = local_norm * expf(local_max - row_max);
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        // first warp will do global reduction
        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : 0.0f;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    float row_norm = smem[0];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];
        elem.x = expf(elem.x - row_max) / row_norm;
        elem.y = expf(elem.y - row_max) / row_norm;
        elem.z = expf(elem.z - row_max) / row_norm;
        elem.w = expf(elem.w - row_max) / row_norm;

        output_row_vec[i] = elem;
    }
}

/*
Runs the online softmax kernel: `id = 3`
*/
void run_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(1024);
    dim3 grid_size(M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_4<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}