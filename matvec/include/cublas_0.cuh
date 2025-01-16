#ifndef CUBLAS_SGEMV
#define CUBLAS_SGEMV

float run_kernel_cublas_sgemv(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH);

#endif  // CUBLAS_SGEMV
