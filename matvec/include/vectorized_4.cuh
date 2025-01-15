#ifndef VECTORIZED_SGEMV
#define VECTORIZED_SGEMV

__global__ void vectorized_sgemv_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

void run_kernel_vectorized_sgmev(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

#endif  // VECTORIZED_SGEMV
