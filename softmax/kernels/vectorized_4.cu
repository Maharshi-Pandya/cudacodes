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
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // cast as float4
    int n_float4s = N / 4;
    float4* input_row_vec = reinterpret_cast<float4*>(input_row);
    float4* output_row_vec = reinterpret_cast<float4*>(output_row);
    float maxval = -INFINITY;

    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];

        maxval = fmaxf(maxval, elem.x);
        maxval = fmaxf(maxval, elem.y);
        maxval = fmaxf(maxval, elem.z);
        maxval = fmaxf(maxval, elem.w);
        if (maxval > local_max) {
            local_norm *= __expf(local_max - maxval);
            local_max = maxval;
        }
        local_norm += __expf(elem.x - maxval);
        local_norm += __expf(elem.y - maxval);
        local_norm += __expf(elem.z - maxval);
        local_norm += __expf(elem.w - maxval);
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
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    // when blockDim is greater than 32, we need to do a block level reduction
    // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
    // the global max will be stored in the first warp
    if (blockDim.x > warpSize) {
        if (tid % warpSize == 0) {
            // which warp are we at?
            // store the value in its first thread index
            smem[tid / warpSize] = val;
        }
        __syncthreads();

        // first warp will do global reduction only
        // this is possible because we stored the values in the shared memory
        // so the threads in the first warp will read from it and then reduce
        if (tid < warpSize) {
            val = (tid < CEIL_DIV(blockDim.x, warpSize)) ? smem[tid] : -INFINITY;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
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
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (blockDim.x > warpSize) {
        if (tid % warpSize == 0) {
            smem[tid / warpSize] = val;
        }
        __syncthreads();

        // first warp will do global reduction
        if (tid < warpSize) {
            val = (tid < CEIL_DIV(blockDim.x, warpSize)) ? smem[tid] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
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
    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];
        elem.x = __expf(elem.x - row_max) / row_norm;
        elem.y = __expf(elem.y - row_max) / row_norm;
        elem.z = __expf(elem.z - row_max) / row_norm;
        elem.w = __expf(elem.w - row_max) / row_norm;

        output_row_vec[i] = elem;
    }
}

/*
Runs the online softmax kernel: `id = 4`
*/
float run_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    dim3 block_size(1024);
    dim3 grid_size(M);

    int warp_size = 32;
    size_t smem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_4<<<grid_size, block_size, smem_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}