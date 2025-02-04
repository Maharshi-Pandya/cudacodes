#ifndef VECTORIZED_SOFTMAX
#define VECTORIZED_SOFTMAX

__global__ void softmax_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

float run_kernel_4(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

#endif  // VECTORIZED_SOFTMAX