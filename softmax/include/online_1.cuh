#ifndef ONLINE_SOFTMAX
#define ONLINE_SOFTMAX

__global__ void softmax_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

void run_kernel_1(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

#endif  // ONLINE_SOFTMAX