#include <stdio.h>

#include "utils.cuh"

/*
Helper function to generate a clamped random number sampled from a
normal distribution with mean 0 and std 1
*/
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

float compute_gflops(int M, int N, float ms) {
    return (2 * M * N) / (ms * 1e6);
}

float compute_peak_gflops(float gflops, float THEORETICAL_MAX_GFLOPS) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    return (gflops / THEORETICAL_MAX_GFLOPS) * 100;
}

float compute_peak_memory_bandwidth(int M, int N, float ms, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    size_t totalFloats = (size_t)(M * N + N + M);
    float totalBytes = (float)totalFloats * sizeof(float);

    float secs = ms / 1000.0f;
    float gbPerSec = (totalBytes / secs) / 1.0e9;

    return (gbPerSec / THEORETICAL_MAX_MEMORY_BANDWIDTH) * 100;
}

void print_kernel_essentials(int M, int N, float ms, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
    float gflops = compute_gflops(M, N, ms);
    printf(">> Execution time: %f ms\n", ms);
    printf(">> Achieved (GFLOPS): %f\n", gflops);
    printf(">> Theoretical max (GFLOPS): %f\n", THEORETICAL_MAX_GFLOPS);
    printf(">> Maximum memory bandwidth: %f GB/s\n", THEORETICAL_MAX_MEMORY_BANDWIDTH);
    printf(">> Achieves %f %% of peak GFLOPS\n", compute_peak_gflops(gflops, THEORETICAL_MAX_GFLOPS));
    printf(">> Achieves %f %% of peak Memory Bandwidth\n", compute_peak_memory_bandwidth(M, N, ms, THEORETICAL_MAX_MEMORY_BANDWIDTH));
}
