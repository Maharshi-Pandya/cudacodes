#ifndef COALWARPBLOCK_SGEMV
#define COALWARPBLOCK_SGEMV

__global__ void coalesced_warpblock_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

void run_kernel_coalesced_warpblock_sgmev(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

#endif  // COALWARPBLOCK_SGEMV
