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
#define THEORETICAL_MAX_GFLOPS 2138.0f          // GFLOPS
#define THEORETICAL_MAX_MEMORY_BANDWIDTH 112.1  // GB per second

float random_normal_clamped(float min, float max);

float compute_gflops(int M, int N, float ms);

float compute_peak_gflops(float gflops);

float compute_peak_memory_bandwidth(int M, int N, float ms);

#endif  // CUDA_UTILS_CUH