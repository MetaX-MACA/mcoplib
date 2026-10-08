#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <cstring>
#include <cstdlib>
#include "utils.h"
#include "moe_step4_weighted_topk_gather.h"

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<typename T, int N>
__global__ void step4_weighted_topk_gather_kernel_opt(
    const T* __restrict__ input, const float* __restrict__ weight, T* __restrict__ output,
    int64_t K, int64_t H)
{
    const int64_t token_id = blockIdx.y;
    const int64_t tx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * N;
    if(tx >= H) return;

    const T*     ptr_in  = input  + token_id * K * H + tx;
    T*           ptr_out = output + token_id * H + tx;
    const float* ptr_w   = weight + token_id * K;

    float acc[N];
    #pragma unroll
    for(int i = 0; i < N; i++) acc[i] = 0.0f;

    // full K reduction in-thread; K is small (=8) so this is a few FMAs.
    for(int64_t k = 0; k < K; k++) {
        const float wk = ptr_w[k];
        T reg[N];
        copy<sizeof(T)*N>((void*)(ptr_in + k * H), (void*)reg);
        #pragma unroll
        for(int i = 0; i < N; i++) acc[i] += (float)reg[i] * wk;
    }

    T reg_dst[N];
    #pragma unroll
    for(int i = 0; i < N; i++) reg_dst[i] = (T)acc[i];
    copy<sizeof(T)*N>((void*)reg_dst, (void*)ptr_out);
}

// (threadIdx.x, threadIdx.y)
template<typename T, int N, int NUM_THREADS_X, int NUM_THREADS_Y>
__global__ void step4_weighted_topk_gather_kernel(const T* input, const float* weight, T* output, int64_t K, int64_t H)
{
    int64_t token_id = blockIdx.y;
    int64_t tx = (blockIdx.x * blockDim.x + threadIdx.x) * N;
    if(tx >= H) return;
    const T* ptr_block_input = input + token_id * K * H + tx;
    T* ptr_block_output = output + token_id * H + tx;
    const float* ptr_block_weight = weight + token_id * K;
    int64_t stride = blockDim.y;
    
    float acc[N] = {0.0f};
    for(int64_t tid = threadIdx.y; tid < K; tid += stride) {
        float w = ptr_block_weight[tid];
        const T* ptr_local_input = ptr_block_input + tid * H;
        T reg_local[N];
        copy<sizeof(T)*N>((void*)ptr_local_input, (void*)reg_local);
        #pragma unroll N
        for(int i = 0; i < N; i++) {
            acc[i] += (float)reg_local[i] * w;
        }
    }
    
    __shared__ float sm_acc[NUM_THREADS_Y][NUM_THREADS_X*N];
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        sm_acc[threadIdx.y][threadIdx.x * N + i] = acc[i];
    }
    __syncthreads();
    for(int step = blockDim.y >> 1; step >= 1; step = step >> 1) {
        if(threadIdx.y < step) {
            for(int i = 0; i < N; i++) {
                sm_acc[threadIdx.y][threadIdx.x * N + i] += sm_acc[threadIdx.y + step][threadIdx.x * N + i];
            }
        }
        __syncthreads();
    }
    if(threadIdx.y == 0) {
        T reg_dst[N];
        #pragma unroll N
        for(int i = 0; i < N; i++) {
            reg_dst[i] = (T)sm_acc[0][threadIdx.x * N + i];
        }
        copy<sizeof(T)*N>((void*)reg_dst, ptr_block_output);
    }
}

template<typename T>
void launch_step4_weight_topk_gather(
    cudaStream_t stream,
    const T* input,
    const float* weight,
    T* output,
    int64_t Token,
    int64_t K,
    int64_t H
)
{
    constexpr int N = 16 / sizeof(T);
    constexpr int BX = 256;
    if(K <= 8) {
        if((H % N) == 0) {
            dim3 block(BX, 1, 1);
            dim3 grid((H / N + BX - 1) / BX, Token, 1);
            step4_weighted_topk_gather_kernel_opt<T, N><<<grid, block, 0, stream>>>(input, weight, output, K, H);
        } else {
            dim3 block(BX, 1, 1);
            dim3 grid((H + BX - 1) / BX, Token, 1);
            step4_weighted_topk_gather_kernel_opt<T, 1><<<grid, block, 0, stream>>>(input, weight, output, K, H);
        }
        return;
    }
    dim3 blockSize(8, 64, 1);
    dim3 GridSize((H + blockSize.x - 1) / blockSize.x, Token, 1);
    if((H % N) == 0) {
        GridSize.x = (GridSize.x + N - 1) / N;
        step4_weighted_topk_gather_kernel<T, N, 8, 64><<<GridSize, blockSize, 0, stream>>>(input, weight, output, K, H);
    } else {
        step4_weighted_topk_gather_kernel<T, 1, 8, 64><<<GridSize, blockSize, 0, stream>>>(input, weight, output, K, H);
    }
}

void step4_weighted_topk_gather(
    at::Tensor input,               // [T, K, H]
    at::Tensor router_weight,       // [T, K]
    at::Tensor output               // [T, H]
)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(router_weight.is_cuda(), "router_weight must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3d tensor");
    TORCH_CHECK(router_weight.dim() == 2, "router_weight must be 2d tensor");
    TORCH_CHECK(output.dim() == 2, "output must be 2d tensor");
    TORCH_CHECK(router_weight.scalar_type() == at::ScalarType::Float);

    int64_t H = input.size(-1);
    int64_t T = input.size(0);
    int64_t K = input.size(1);
    if(T == 0) return;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    switch(input.scalar_type()) {
        case at::ScalarType::BFloat16:
            launch_step4_weight_topk_gather<bfloat16>(stream, 
              reinterpret_cast<bfloat16*>(input.data_ptr<at::BFloat16>()),
              reinterpret_cast<float*>(router_weight.data_ptr<float>()),
              reinterpret_cast<bfloat16*>(output.data_ptr<at::BFloat16>()),
              T, K, H);
        break;
        default:
            TORCH_CHECK(false, "step4_weighted_topk_gather unsupported input dtype: ",
                  input.scalar_type());
    }
}

