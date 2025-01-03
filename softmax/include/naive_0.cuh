#ifndef NAIVE_SOFTMAX
#define NAIVE_SOFTMAX

__global__ void softmax_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

void run_kernel_0(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

#endif  // NAIVE_SOFTMAX