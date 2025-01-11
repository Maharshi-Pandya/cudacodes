// Matrix Multiplication (xGEMM) kernels
// Note: this file might change often as i learn more about CUDA and kernels in general

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <time.h>

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
#define M_PI 3.14159265358979323846f
#define TILE_SIZE 32

#define BM 64
#define BK 8
#define BN 64
#define COARSE_FACTOR 8

/*
Naive xGEMM kernel:

- 2D blocks, 2D threads
- Each thread calculates one element of the output matrix C
- No shared memory, only global memory access
*/
__global__ void naive_xgemm_kernel(float* __restrict__ Ad, float* __restrict__ Bd, float* __restrict__ Cd, int M, int N, int K) {
    // for coalesced memory access:
    // maps rows to y-direction, and cols to x-direction
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += Ad[row * K + k] * Bd[k * N + col];
        }
        Cd[row * N + col] = acc;
    }
}

/*
Tiled xGEMM kernel:

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
__global__ void tiled_xgemm_kernel(float* __restrict__ Ad, float* __restrict__ Bd, float* __restrict__ Cd, int M, int N, int K) {
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
            a_smem[ty][tx] = Ad[row * K + offset + tx];
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

/*
Tiled xGEMM kernel with + 1D blocktiling

- Each thread calculates more than one element (one column of output matrix C)
- Tiles of A has shape (BM, BK) and tile of B has shape (BK, BN)
- Threads process COARSE_FACTOR rows at a time
*/
__global__ void tiled_xgemm_1d_coarse_kernel(float* __restrict__ Ad, float* __restrict__ Bd, float* __restrict__ Cd, int M, int N, int K) {
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // for within each tile + for loading B's tile
    int ty = threadIdx.x / BN;
    int tx = threadIdx.x % BN;

    // for loading A's tile
    int aty = threadIdx.x / BK;
    int atx = threadIdx.x % BK;

    // working on C[row, col]
    int row = by * BM + (ty * COARSE_FACTOR);
    int col = bx * BN + tx;

    // shared memory for A and B for computing tiles
    __shared__ float a_smem[BM * BK];
    __shared__ float b_smem[BK * BN];

    float acc[COARSE_FACTOR] = {0.f};

    for (int tile = 0; tile < K; tile += BK) {
        // load tiles into shared memory for both A and B
        if ((by * BM + aty) < M && (tile + atx) < K)
            a_smem[aty * BK + atx] = Ad[(by * BM + aty) * K + (tile + atx)];
        else
            a_smem[aty * BK + atx] = 0.f;
        if ((tile + ty) < K && (bx * BN + tx) < N)
            b_smem[ty * BN + tx] = Bd[(tile + ty) * N + (bx * BN + tx)];
        else
            b_smem[ty * BN + tx] = 0.f;
        __syncthreads();

        // inner loop:
        // each thread computes 8 elements
        for (int k = 0; k < BK; k++) {
            float b_reg = b_smem[k * BN + tx];
            for (int c = 0; c < COARSE_FACTOR; c++)
                acc[c] += a_smem[(ty * COARSE_FACTOR + c) * BK + k] * b_reg;
        }
        __syncthreads();
    }

    for (int c = 0; c < COARSE_FACTOR; c++) {
        if ((row + c) < M && col < N)
            Cd[(row + c) * N + col] = acc[c];
    }
}

void gemm_cpu_naive(float* A, float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.f;
            for (int k = 0; k < K; k++) {
                sum += (A[i * K + k] * B[k * N + j]);
            }
            C[i * N + j] = sum;
        }
    }
}

/*
Helper function to generate a clamped random number sampled from a
normal distribution with mean 0 and std 1
*/
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;

    int a_size = M * K;
    int b_size = K * N;
    int c_size = M * N;

    printf("Shape A: (%d, %d)\n", M, K);
    printf("Shape B: (%d, %d)\n", K, N);
    printf("Shape C: (%d, %d)\n", M, N);

    float* A = (float*)malloc(a_size * sizeof(float));
    float* B = (float*)malloc(b_size * sizeof(float));
    float* C = (float*)malloc(c_size * sizeof(float));
    float* C_cpu = (float*)malloc(c_size * sizeof(float));

    // init the matrices with random values
    for (int i = 0; i < a_size; i++) {
        A[i] = random_normal_clamped(-10, 10);
    }
    for (int i = 0; i < b_size; i++) {
        B[i] = random_normal_clamped(-10, 10);
    }
    for (int i = 0; i < b_size; i++) {
        C[i] = 0.f;
    }

    float *Ad, *Bd, *Cd;

    // uncomment the below lines for tiled_xgemm without coalesce
    // dim3 block_size(TILE_SIZE, TILE_SIZE);
    // dim3 grid_size(CEIL_DIV(N, block_size.x), CEIL_DIV(M, block_size.y));

    dim3 block_size(BM * BN / COARSE_FACTOR);
    dim3 grid_size(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&Ad, a_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Bd, b_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Cd, c_size * sizeof(float)));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(Ad, A, a_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Bd, B, b_size * sizeof(float), cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    tiled_xgemm_1d_coarse_kernel<<<grid_size, block_size>>>(Ad, Bd, Cd, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(C, Cd, c_size * sizeof(float), cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    printf("\n>> Running GEMM on CPU...\n");
    clock_t ts = clock();
    gemm_cpu_naive(A, B, C_cpu, M, N, K);
    clock_t te = clock();
    printf(">> Done\n");

    float elapsed_time = (te - ts) * 1000 / CLOCKS_PER_SEC;
    printf("Elapsed time: %.6f ms\n", elapsed_time);

    // check if results match within an error tolerance (eps)
    bool match = true;
    float eps = 0.0001;
    for (int i = 0; i < c_size; i++) {
        if (fabs(C_cpu[i] - C[i]) > eps) {
            match = false;
            break;
        }
    }
    printf("\n>> Results match for CPU and GPU? ");
    printf("%s\n", match ? "true" : "false");
}