#include <assert.h>
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <torch/extension.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))

struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return a + b;
    }
    __device__ __forceinline__ float identity() const {
        return 0.0f;
    }
};

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
    __device__ __forceinline__ float identity() const {
        return -INFINITY;
    }
};

template <typename Op>
__device__ __forceinline__ float warpReduce(float val, Op op) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val = op(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    return val;
}

template <typename Op>
__device__ __forceinline__ void blockReduce(float val, float *smem, int tid, int blockDimX, Op op) {
    // 1. do warpReduce sum
    val = warpReduce(val, op);

    // 2. do blockReduce sum
    if (blockDimX > warpSize) {
        int lane = tid % warpSize;
        int wid = tid / warpSize;
        if (lane == 0) {
            smem[wid] = val;
        }
        __syncthreads();

        if (tid < warpSize) {
            val = tid < CEIL_DIV(blockDimX, warpSize) ? smem[tid] : op.identity();
            val = warpReduce(val, op);
            if (tid == 0) smem[0] = val;
        }
    } else {
        if (tid == 0) smem[0] = val;
    }
    __syncthreads();
}

__global__ void scale_kernel_inplace(float *X, float scale_factor, int bs, int nh, int sl, int ed, int n_elements) {
    assert(n_elements % 4 == 0);

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int n_float4s = n_elements / 4;

    float4 *inputs = reinterpret_cast<float4 *>(X);

    if (idx < n_float4s) {
        float4 elem = inputs[idx];
        elem.x *= scale_factor;
        elem.y *= scale_factor;
        elem.z *= scale_factor;
        elem.w *= scale_factor;

        inputs[idx] = elem;
    }
}

__global__ void softmax_kernel_inplace(float *__restrict__ X, int M, int N) {
    assert(N % 4 == 0);

    // max and norm reduction will happen in shared memory (static)
    extern __shared__ float smem[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= M) return;

    float *input_row = X + row * N;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // cast as float4
    int n_float4s = N / 4;
    float4 *input_row_vec = reinterpret_cast<float4 *>(input_row);

    float maxval = -INFINITY;
    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];

        maxval = fmaxf(maxval, elem.x);
        maxval = fmaxf(maxval, elem.y);
        maxval = fmaxf(maxval, elem.z);
        maxval = fmaxf(maxval, elem.w);
        if (maxval > local_max) {
            local_norm *= __expf(local_max - maxval);
            local_max = maxval;
        }
        local_norm += __expf(elem.x - maxval);
        local_norm += __expf(elem.y - maxval);
        local_norm += __expf(elem.z - maxval);
        local_norm += __expf(elem.w - maxval);
    }

    blockReduce(local_max, smem, tid, blockDim.x, MaxOp());
    float row_max = smem[0];

    float adjusted = local_norm * expf(local_max - row_max);
    blockReduce(adjusted, smem, tid, blockDim.x, SumOp());
    float row_norm = smem[0];

    // finally, compute softmax
    #pragma unroll
    for (int i = tid; i < n_float4s; i += blockDim.x) {
        float4 elem = input_row_vec[i];
        elem.x = __expf(elem.x - row_max) / row_norm;
        elem.y = __expf(elem.y - row_max) / row_norm;
        elem.z = __expf(elem.z - row_max) / row_norm;
        elem.w = __expf(elem.w - row_max) / row_norm;

        input_row_vec[i] = elem;
    }
}

torch::Tensor attention_forward(uint64_t handle, torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    cublasHandle_t cu_handle = reinterpret_cast<cublasHandle_t>(handle);
    cublasSetMathMode(cu_handle, CUBLAS_TF32_TENSOR_OP_MATH);

    int bs = Q.size(0);
    int nh = Q.size(1);
    int sl = Q.size(2);
    int ed = Q.size(3);

    int n_elements = bs * nh * sl * ed;
    torch::Tensor pre = torch::empty({bs, nh, sl, sl}, torch::TensorOptions().device(torch::kCUDA));
    torch::Tensor out = torch::zeros_like(Q);

    const float alpha = (1 / sqrt(ed));
    const float beta = 0.0f;
    cublasSgemmStridedBatched(
        cu_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        sl, sl, ed,
        &alpha,
        K.data_ptr<float>(), ed, sl * ed,
        Q.data_ptr<float>(), ed, sl * ed,
        &beta,
        pre.data_ptr<float>(), sl, sl * sl,
        bs * nh);

    // softmax
    dim3 block_dim(128);
    dim3 grid_dim_softmax(bs * nh * sl);
    size_t smem_size = CEIL_DIV(block_dim.x, 32) * sizeof(float);
    softmax_kernel_inplace<<<grid_dim_softmax, block_dim, smem_size>>>(pre.data_ptr<float>(), bs * nh * sl, sl);

    // update alpha here for no scaling
    const float alpha_sv = 1.0f;
    cublasSgemmStridedBatched(
        cu_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        ed, sl, sl,
        &alpha_sv,
        V.data_ptr<float>(), ed, sl * ed,
        pre.data_ptr<float>(), sl, sl * sl,
        &beta,
        out.data_ptr<float>(), ed, sl * ed,
        bs * nh);

    return out;
}