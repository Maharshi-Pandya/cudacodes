#ifndef COALWARPBLOCK_SGEMV
#define COALWARPBLOCK_SGEMV

__global__ void coalesced_warpblock_sgmev_kernel(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N);

float run_kernel_coalesced_warpblock_sgmev(float* __restrict__ matd, float* __restrict__ vecd, float* __restrict__ resd, int M, int N, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH);

#endif  // COALWARPBLOCK_SGEMV
