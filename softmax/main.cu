#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

/*
This kernel implements a naive softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
One thread processes one entire row, and thus this kernel will be the slowest
since we aren't exploiting parallelism capabilities of GPUs that much.
We are only parallelizing over the rows (one block processes one row).
*/
__global__ void naive_softmax_kernel(float* xd, float* resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        // max
        float m = -1 * INFINITY;
        // norm factor
        float L = 0.0f;

        // 3 passes (not optimal)
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            m = max(m, xd[i]);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            L += expf(xd[i] - m);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(xd[i] - m) / L;
        }
    }
}

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
One thread processes one entire row, but instead of 3 passes we do only 2 passes.
This is possible due to the property of exponentials.
We are parallelizing over the rows (one block processes one row).
*/
__global__ void online_softmax_kernel(float* xd, float* resd, int M, int N) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float m = -1 * INFINITY;
        float L = 0.0f;

        // compute max and norm factor in one pass only
        // by exploiting the property of exponentials
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            float curr = xd[i];
            if (curr > m) {
                L = L * expf(m - curr);
                m = curr;
            }
            L += expf(curr - m);
        }
        for (int col = 0; col < N; col++) {
            int i = row * N + col;
            resd[i] = expf(xd[i] - m) / L;
        }
    }
}

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
In this, we handle each row with a block where the threads within one block work together
to process one row (max and norm factor). Each thread will process some elements
and will contains its local max and local norm in shared memory. Then, we perform reduction
operations to compute the final max and norm factor. Also, we compute maxes and norms
in one pass itself.
*/
__global__ void optimized_softmax_kernel(float* xd, float* resd, int M, int N) {
    // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // edge condition (we don't process further)
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // compute local max and norm for each thread
    // and then finally have a sync barrier before moving on
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // each thread will have its own local max
    // we store it in the tid of the shared memory
    smem[tid] = local_max;
    __syncthreads();

    // block-level reduction in O(log(N)) time over all threads
    // is faster than linear reduction over all threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = max(smem[tid], smem[tid + stride]);
        }
        // sync barrier before next iteration to ensure correctness
        __syncthreads();
    }

    // the first element after max reduction from all threads
    // will contain the global max for the row
    float row_max = smem[0];
    __syncthreads();

    // each thread will have its own local norm
    // we will store the corrected local norm in the shared memory
    // again, exploits property of exponentials
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    // sum reduction similar to above for global norm factor
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
This one is largely similar to the above kernel. The difference is instead of accessing
shared memory and having sync barrier overhead, we will use warp-level primitives (then
block-level) for performing max and sum reductions. The benefit is: it is faster than shared
memory access and also does not need syncing since each warp (group of 32 threads) execute
an instuction parallely on GPU so no chance of race conditions.
*/
__global__ void optimized_softmax_kernel_2(float* xd, float* resd, int M, int N) {
    // max and norm reduction will happen in shared memory (static)
    __shared__ float smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    // number of threads in a warp
    unsigned int warp_size = 32;
    if (row >= M) return;

    float* input_row = xd + row * N;
    float* output_row = resd + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // same as above kernel
    for (int i = tid; i < N; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // each thread will have its own local max
    // we store it in shared memory for reduction
    smem[tid] = local_max;
    __syncthreads();

    // warp level reduction using XOR shuffle ('exchanges' the values in the threads)
    // note: if there are 256 threads in one block (8 warps of 32 threads each)
    // the following for loop reduces the value in all the 8 warps
    // the 8 warps contain the 8 maximum values of the 32 threads that reside in those warps
    float val = smem[tid];
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }

    // when blockDim is greater than 32, we need to do a block level reduction
    // AFTER warp level reductions since we have the 8 maximum values that needs to be reduced again
    // the global max will be stored in the first warp
    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            // which warp are we at?
            // store the value in its first thread index
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        // first warp will do global reduction only
        // this is possible because we stored the values in the shared memory
        // so the threads in the first warp will read from it and then reduce
        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : -INFINITY;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        // this is for when the number of threads in a block are not
        // greater than the warp size, in that case we already reduced
        // so we can store the value
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    // we got the global row max now
    float row_max = smem[0];
    __syncthreads();

    // each thread will have its own local_norm
    // we will store the corrected local_norm in the shared memory
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    // same reduction algorithm as above, but instead of max reduction
    // we do a sum reduction i.e. we accumulate the values
    val = smem[tid];
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }

    if (blockDim.x > warp_size) {
        if (tid % warp_size == 0) {
            smem[tid / warp_size] = val;
        }
        __syncthreads();

        // first warp will do global reduction
        if (tid < warp_size) {
            val = (tid < CEIL_DIV(blockDim.x, warp_size)) ? smem[tid] : 0.0f;
            for (int offset = warp_size / 2; offset > 0; offset /= 2) {
                val += __shfl_xor_sync(0xffffffff, val, offset);
            }
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();

    float row_norm = smem[0];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < N; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
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
    int N = 32768;
    int matsize = M * N;
    int totalsize = matsize * sizeof(float);

    // allocate and initialize host matrix
    float* mat = (float*)malloc(totalsize);
    float* res = (float*)malloc(totalsize);
    for (int i = 0; i < matsize; i++) {
        mat[i] = random_normal_clamped(-10, 10);
    }

    // arrays to allocate on device ends with 'd'
    float *xd, *resd;
    dim3 block_size(1024);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    // below code calculates the time elapsed for
    // each cuda operation performed such as GPU allocation,
    // copying from host to device, kernel execution time etc...

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&xd, totalsize));
    CUDA_CHECK(cudaMalloc(&resd, totalsize));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(xd, mat, totalsize, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    optimized_softmax_kernel_2<<<grid_size, block_size>>>(xd, resd, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(res, resd, totalsize, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // correctness check on the first row
    // the output should be 1.0 (or a number very close to it)
    // TODO: add full correctness check
    float sum = 0.f;
    for (int i = 0; i < N; i++) {
        sum += res[i];
    }
    printf("\nSum of the 1st row of softmax result: %f\n", sum);

    free(mat);
    free(res);
    cudaFree(xd);
    cudaFree(resd);
}