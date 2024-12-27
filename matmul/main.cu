// Tiled Matrix Multiplication (TGEMM) kernel

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define TILE_SIZE 16

/*
Tiled GEMM kernel:

- Each block calculates a "tile" of the output matrix C
    > Here the indices for C, that each block (bx, by) computes would be:
    row = by * TILE_SIZE + ty;
    col = bx * TILE_SIZE + tx;

- Each block will loop over the tiles in the common dimension.

- The threads within each block loads the elements in shared memory
    > Thread (tx, ty) will load the corresponding elements from A and B
    shared_A[ty][tx] =  A[row * K + (tile_num * TILE_SIZE + tx)]
    shared_B[ty][tx] = B[(tile_num * TILE_SIZE + ty) * N + col]

    Note: from A, the same row is loaded and from B the same column is loaded

- Then they accumulate the dot product in a variable for the common dimension
- So block (bx, by) has completed computing the tile (bx, by) of C.
*/
__global__ void tiled_gemm_kernel(float* __restrict__ Ad, float* __restrict__ Bd, float* __restrict__ Cd, int M, int N, int K) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int by = blockIdx.y;
    int bx = blockIdx.x;

    // indices of C[row, col]
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // tile that will be loaded by THIS block
    __shared__ float a_smem[TILE_SIZE][TILE_SIZE];
    __shared__ float b_smem[TILE_SIZE][TILE_SIZE];

    // final dot product sum
    float acc = 0.f;

    // THIS block will loop over the tiles in common dimension
    for (int tile_num = 0; tile_num < CEIL_DIV(K, TILE_SIZE); tile_num++) {
        int offset = tile_num * TILE_SIZE;

        // out of bounds check
        // same row, different column for A
        if (row < M && (offset + tx) < K)
            a_smem[ty][tx] = Ad[row * M + offset + tx];
        else
            a_smem[ty][tx] = 0.f;

        // different row, same column for B
        if ((offset + ty) < K && col < N)
            b_smem[ty][tx] = Bd[(offset + ty) * N + col];
        else
            b_smem[ty][tx] = 0.f;
        __syncthreads();

        // dot product and accumulate
        for (int i = 0; i < TILE_SIZE; i++) {
            acc += a_smem[ty][i] * b_smem[i][tx];
        }
        __syncthreads();
    }

    // write the final output after looping over all tiles
    if (row < M && col < N) {
        Cd[row * N + col] = acc;
    }
}