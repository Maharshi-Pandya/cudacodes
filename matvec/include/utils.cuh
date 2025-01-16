#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define M_PI 3.14159265f

float random_normal_clamped(float min, float max);

float compute_gflops(int M, int N, float ms);

float compute_peak_gflops(float gflops, float THEORETICAL_MAX_GFLOPS);

float compute_peak_memory_bandwidth(int M, int N, float ms, float THEORETICAL_MAX_MEMORY_BANDWIDTH);

void print_kernel_essentials(int M, int N, float ms, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH);

/*
Reduction functions on device. These will be inline:
The compiler will replace the call with the code instead of calling the function (overhead)
*/
/*
Utility warp level sum reduction with shuffle instructions
*/

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__device__ __forceinline__ void blockReduceSum(float val, float *smem, int tid, int blockDimX) {
    // 1. do warpReduce sum
    val = warpReduceSum(val);

    // 2. do blockReduce sum
    if (blockDimX > warpSize) {
        int lane = tid % warpSize;
        int wid = tid / warpSize;
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();

        if (tid < warpSize) {
            val = tid < CEIL_DIV(blockDimX, warpSize) ? smem[tid] : 0.0f;
            val = warpReduceSum(val);
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    // __syncthreads();
    // sync not needed because only thread 0 reads from smem[0]
}

#endif  // CUDA_UTILS_CUH