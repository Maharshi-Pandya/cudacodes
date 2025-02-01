#ifndef BLOCKTILING_SOFTMAX
#define BLOCKTILING_SOFTMAX

__global__ void softmax_kernel_5(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

void run_kernel_5(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

#endif  // BLOCKTILING_SOFTMAX