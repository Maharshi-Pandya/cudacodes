#ifndef NAIVE_SGEMV
#define NAIVE_SGEMV

__global__ void naive_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

void run_kernel_naive_sgemv(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

#endif  // NAIVE_SGEMV
