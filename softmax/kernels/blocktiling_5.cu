#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include "cuda_utils.cuh"

const int TILE_SIZE = 4;


/*
Takes in an array of size `TILE_SIZE` and reduces it as warp-wide sum.
The first element in the array will contain the reduced sum.
*/
__device__ __forceinline__ float warpReduceSum(float val) {
    for(int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/*
Takes in an array of size `TILE_SIZE` and reduces it warp-wide max.
The first element in the array will contain the reduced max.
*/
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
}

// __device__ __forceinline__ void blockReduceSum(volatile float* arr, float* smem, int ty, int tx, int blockDimX) {
//     warpReduceSum(arr);

//     if (blockDimX <= warpSize) {
//         // for small block sizes, single warp reduction is sufficient
//         return;
//     }

//     int cols = blockDimX / warpSize;
//     int lane = tx % warpSize;
//     int wid = tx / warpSize;

//     if (lane == 0) {
//         #pragma unroll
//         smem[ty*cols + wid] = arr[wid];
//     }
//     __syncthreads();

//     if (tx < warpSize) {
//         #pragma unroll
//         for(int i = 0; i < TILE_SIZE; i++) {
//             arr[i] = smem[i*cols + tx];
//         }
//         warpReduceSum(arr);
//     }
// }

/*
How this works:
Instead of having one block calculate only one row of the output matrix, one block
will compute `TILE_SIZE` number of rows. This way we will have fewer blocks, and more
computations per block. 2D threads will process elements. `tx` will process elements and
`ty` will process rows in a block

We will need partial block-wide reduction with width `tx` threads participating in the reduction
for each row's maximum and norm value.
*/
__global__ void softmax_kernel_5(float* __restrict__ xd, float* __restrict__ resd, int M, int N) {
    int bx = blockDim.x;

    // ty equals TILE_SIZE
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // result matrix's row
    int row = (bx * TILE_SIZE + ty);
    if (row >= M) return;

    // one for each row
    float local_maxs[TILE_SIZE] = {-INFINITY};
    float local_norms[TILE_SIZE] = {0.f};
    float x[TILE_SIZE] = {0.f};

    for (int i = tx; i < N; i += bx) {
        #pragma unroll
        for (int j = 0; j < TILE_SIZE; j++) {
            x[j] = xd[row * N + i];
            if (x[j] > local_maxs[j]) {
                local_norms[j] *= expf(local_maxs[j] - x[j]);
                local_maxs[j] = x[j];
            }
            local_norms[j] += expf(x[j] - local_maxs[j]);
        }
    }
    __syncthreads();

    for(int tile = 0; tile < TILE_SIZE; tile++) {
        float lm = local_maxs[tile];
        local_maxs[tile] = warpReduceMax(lm);
        local_norms[tile] *= expf(lm - local_maxs[tile]);
        local_norms[tile] = warpReduceSum(local_norms[tile]);
    }

    // finally, compute softmax
    for (int i = tx; i < N; i += bx)
        for (int tile = 0; tile < TILE_SIZE; tile++)
            resd[row * N + i] = expf(xd[row * N + i] - local_maxs[tile]) / local_norms[tile];
}

/*
Runs the online softmax kernel: `id = 5`
*/
void run_kernel_5(float* __restrict__ matd, float* __restrict__ resd, int M, int N) {
    // grid size and block size for this kernel
    // change as necessary
    int num_threads_x = 32;

    dim3 block_size(TILE_SIZE, num_threads_x);
    dim3 grid_size(CEIL_DIV(M, TILE_SIZE));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.f;

    CUDA_CHECK(cudaEventRecord(start));
    softmax_kernel_5<<<grid_size, block_size>>>(matd, resd, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf(">> Kernel execution time: %f ms\n", ms);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}