#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int dev_count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&dev_count);
    cudaGetDeviceProperties(&prop, 0);

    printf(">> CUDA enabled devices in the system: %d\n\n", dev_count);

    printf(">> Max grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf(">> Max block size: %d\n\n", prop.maxThreadsPerBlock);

    printf(">> Number of SMs: %d\n", prop.multiProcessorCount);
    printf(">> Clock rate of the SMs (in kHz): %d\n", prop.clockRate);

    printf(">> Max threads dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf(">> Max threads per SM: %d\n\n", prop.maxThreadsPerMultiProcessor);

    printf(">> Registers available per block: %d\n", prop.regsPerBlock);
    printf(">> Registers available per SM: %d\n\n", prop.regsPerMultiprocessor);

    printf(">> Warp size (threads per warp): %d\n\n", prop.warpSize);
    printf(">> Shared memory size per block: %zd bytes\n", prop.sharedMemPerBlock);
    printf(">> Shared memory size per SM: %zd bytes\n\n", prop.sharedMemPerMultiprocessor);

    printf(">> L2 cache size: %d bytes\n", prop.l2CacheSize);
}