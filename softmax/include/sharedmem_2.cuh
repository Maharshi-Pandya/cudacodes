#ifndef SHAREDMEM_SOFTMAX
#define SHAREDMEM_SOFTMAX

__global__ void softmax_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

void run_kernel_2(float* __restrict__ matd, float* __restrict__ resd, int M, int N);

#endif  // SHAREDMEM_SOFTMAX