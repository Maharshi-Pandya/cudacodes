#ifndef SHFL_SOFTMAX
#define SHFL_SOFTMAX

__global__ void softmax_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

void run_kernel_3(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

#endif  // SHFL_SOFTMAX