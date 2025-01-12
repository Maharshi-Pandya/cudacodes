#ifndef CUBLAS_SGEMV
#define CUBLAS_SGEMV

void run_kernel_cublas_sgemv(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

#endif  // CUBLAS_SGEMV
