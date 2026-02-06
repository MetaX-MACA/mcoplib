// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

template<typename T>
float __device__ __forceinline__ convert(T value) {
    return static_cast<float>(value);
}

template<>
float __device__ __forceinline__ convert(bfloat16 value) {
    return __bfloat162float(value);
}

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
__device__ __forceinline__ void copy_data(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;
    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

template<>
__device__ __forceinline__ void copy_data<32>(const void* local, void* data)
{
    const int8_t* in = static_cast<const int8_t*>(local);
    int8_t* out = static_cast<int8_t*>(data);
    *(float4*)out = *(float4*)in;
    *(float4*)(out + 16) = *(float4*)(in + 16);
}

template<typename T, typename VT, typename VT1, int NUM_VT, int NUM_THREADS> 
__global__ void silu_and_mul_quant(const T* input, const float* smooth_scale, int8_t* output, float* scale, 
    const int32_t* expert_token_start, const int32_t* expert_token_count, int64_t hidden_size)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    int stride = NUM_THREADS * N;
    int const expert_id = blockIdx.x;
    int64_t hidden_size2 = hidden_size << 1;
    int gridDim_y = gridDim.y;
    int block_count = expert_token_count[expert_id];
    int block_start = expert_token_start[expert_id];
    const float * ptr_smooth_scale = smooth_scale + expert_id * hidden_size;
    using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    __shared__ float block_absmax_val;

    for(int bk = blockIdx.y; bk < block_count; bk += gridDim_y) {
        const T* ptr_input0 = input + (block_start + bk) * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        int8_t * ptr_output = output + (block_start + bk) * hidden_size;
        float absmax_val = 0.0f;
        float reg_i[NUM_VT][N];
        for(int i = tid*N, j = 0; i < hidden_size; i += stride, j++) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input0 + i);
            vsrc1 = *(VT*)(ptr_input1 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            float reg_smooth_scale[N];
            copy_data<sizeof(float)*N>((void*)(ptr_smooth_scale + i), (void*)reg_smooth_scale);
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = convert(ptr_local0[k]);
                float val1 = convert(ptr_local1[k]);
                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid * reg_smooth_scale[k];
                reg_i[j][k] = gate_up;
                absmax_val = max(absmax_val, abs(gate_up));
            }
        }
        __syncthreads();
        float const block_absmax_val_maybe =
            BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);

        if (tid == 0) {
            block_absmax_val = block_absmax_val_maybe;
            scale[(block_start + bk)] = block_absmax_val * 0.0078740157;
        }
        __syncthreads();
        float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
        for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
            VT1 vdst;
            int8_t* ptr_dst = (int8_t*)&vdst;
            #pragma unroll N
            for(int j = 0; j < N; ++j) {
                ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
            }
            *(VT1*)(ptr_output + i) = vdst;
        }   
    }
}

template<typename T, typename VT, typename VT1, int NUM_VT, int NUM_THREADS> 
__global__ void silu_and_mul_sm_quant(const T* input, const float* smooth_scale, int8_t* output, float* scale, 
    const int32_t* expert_token_start, const int32_t* expert_token_count, int64_t hidden_size)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    int stride = NUM_THREADS * N;
    int const expert_id = blockIdx.x;
    int64_t hidden_size2 = hidden_size << 1;
    float absmax_val = 0.0f;
    float reg_i[4][N];
    __shared__ float sm_gate[4096];
    int hidden_size1 = stride * 4;
    int gridDim_y = gridDim.y;
    int block_count = expert_token_count[expert_id];
    int block_start = expert_token_start[expert_id];
    const float * ptr_smooth_scale = smooth_scale + expert_id * hidden_size;
    using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
    __shared__ typename BlockReduce::TempStorage reduceStorage;
    __shared__ float block_absmax_val;
    for(int bk = blockIdx.y; bk < block_count; bk += gridDim_y) {
        const T* ptr_input0 = input + (block_start + bk) * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        int8_t * ptr_output = output + (block_start + bk) * hidden_size;
        for(int i = tid*N, j = 0; i < hidden_size1; i += stride, j++) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input0 + i);
            vsrc1 = *(VT*)(ptr_input1 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            float reg_smooth_scale[N];
            copy_data<sizeof(float)*N>((void*)(ptr_smooth_scale + i), (void*)reg_smooth_scale);
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = convert(ptr_local0[k]);
                float val1 = convert(ptr_local1[k]);
                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid * reg_smooth_scale[k];
                reg_i[j][k] = gate_up;
                absmax_val = max(absmax_val, abs(gate_up));
            }
        }
        const T* ptr_input2 = ptr_input0 + hidden_size1;
        const T* ptr_input3 = ptr_input1 + hidden_size1;
        ptr_smooth_scale = ptr_smooth_scale + hidden_size1;
        int remain_hidden_size = hidden_size - hidden_size1;
        for(int i = tid*N; i < remain_hidden_size; i += stride) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input2 + i);
            vsrc1 = *(VT*)(ptr_input3 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            float* ptr_sm = sm_gate + i;
            float reg_smooth_scale[N];
            copy_data<sizeof(float)*N>((void*)(ptr_smooth_scale + i), (void*)reg_smooth_scale);
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = convert(ptr_local0[k]);
                float val1 = convert(ptr_local1[k]);
                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid * reg_smooth_scale[k];
                ptr_sm[k] = gate_up;
                absmax_val = max(absmax_val, abs(gate_up));
            }
        }

        float const block_absmax_val_maybe =
            BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);
        
        if (tid == 0) {
            block_absmax_val = block_absmax_val_maybe;
            scale[(block_start + bk)] = block_absmax_val * 0.0078740157;
        }
        __syncthreads();
        float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
        for (int i = tid*N, k = 0; i < hidden_size1; i += stride, k++) {
            VT1 vdst;
            int8_t* ptr_dst = (int8_t*)&vdst;
            #pragma unroll N
            for(int j = 0; j < N; ++j) {
                ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
            }
            *(VT1*)(ptr_output + i) = vdst;
        }

        ptr_output = ptr_output + hidden_size1;
        for(int i = tid*N; i < remain_hidden_size; i += stride) {
            VT1 vdst;
            int8_t* ptr_dst = (int8_t*)&vdst;
            float* ptr_sm = sm_gate + i;
            #pragma unroll N
            for(int j = 0; j < N; ++j) {
                ptr_dst[j] = float_to_int8_rn(ptr_sm[j] * tmp_scale);
            }
            *(VT1*)(ptr_output + i) = vdst;
        }
    }
}

template<typename T>
void launch_silu_mul_quant_no_mask(const T* input,const float* smooth_scale, int8_t* output, float* scale, const int32_t* expert_token_start , const int32_t* expert_token_count,  int total_experts_num, int64_t hidden_size,cudaStream_t stream) {
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int64_t inner_hidden_size = hidden_size / 2;
    constexpr int blocksize = 512;
    int N = sizeof(float4) / sizeof(T);
    dim3 grid(total_experts_num, (sm_count + total_experts_num - 1) / total_experts_num, 1);
    if(N == 8&&(inner_hidden_size & (N - 1)) == 0) {
        int base = blocksize * N;
        if(inner_hidden_size <= base) {
            silu_and_mul_quant<T, float4, float2, 1, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base*2) {
            silu_and_mul_quant<T, float4, float2, 2, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base * 3) {
            silu_and_mul_quant<T, float4, float2, 3, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base * 4) {
            silu_and_mul_quant<T, float4, float2, 4, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base*4 + 4096) {
            silu_and_mul_sm_quant<T, float4, float2, 4,blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else {
            TORCH_CHECK(false, "silu_and_mul_quant not support this hidden_size\n");
        }
    } else if(N == 4 && (inner_hidden_size & (N - 1)) == 0) {
        int base = blocksize * N;
        if(inner_hidden_size <= base) {
            silu_and_mul_quant<T, float4, float, 1, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base*2) {
            silu_and_mul_quant<T, float4, float, 2, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base * 3) {
            silu_and_mul_quant<T, float4, float, 3, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base * 4) {
            silu_and_mul_quant<T, float4, float, 4, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else if(inner_hidden_size <= base * 8) {
            silu_and_mul_quant<T, float4, float, 8, blocksize><<<grid, blocksize,0,stream>>>(input, smooth_scale, output, scale, expert_token_start, expert_token_count, inner_hidden_size);
        } else {
            TORCH_CHECK(false, "silu_and_mul_quant not support this hidden_size\n");
        }
    } else {
        TORCH_CHECK(false, "silu_and_mul_quant not support this hidden_size, becasuse it's not aligned\n");
    }
}

void moe_swiglu_dynamic_quantize(at::Tensor scatter_tokens, at::Tensor smooth_scale, at::Tensor experts_tokens_start, at::Tensor experts_tokens_count,
    at::Tensor& y, at::Tensor& per_tokens_scale, int total_experts_num)
{
	DEBUG_TRACE_PARAMS(scatter_tokens, smooth_scale, experts_tokens_start, experts_tokens_count, y, per_tokens_scale, total_experts_num);
	DEBUG_DUMP_PARAMS(scatter_tokens, smooth_scale, experts_tokens_start, experts_tokens_count, y, per_tokens_scale, total_experts_num);
    CHECK_DEVICE(scatter_tokens);
    CHECK_DEVICE(smooth_scale);
    CHECK_DEVICE(experts_tokens_start);
    CHECK_DEVICE(experts_tokens_count);
    CHECK_DEVICE(y);
    CHECK_DEVICE(per_tokens_scale);
    
    int64_t const hidden_size = scatter_tokens.size(-1);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if(scatter_tokens.dtype() == at::ScalarType::Half) {
        launch_silu_mul_quant_no_mask<half>(reinterpret_cast<const half*>(scatter_tokens.data_ptr<at::Half>()), reinterpret_cast<const float*>(smooth_scale.data_ptr<float>()), y.data_ptr<int8_t>(), 
            per_tokens_scale.data_ptr<float>(), (const int32_t*)experts_tokens_start.data_ptr<int32_t>(), (const int32_t*)experts_tokens_count.data_ptr<int32_t>(), total_experts_num, hidden_size, stream);
    } else if(scatter_tokens.dtype() == at::ScalarType::BFloat16) {
        launch_silu_mul_quant_no_mask<bfloat16>(reinterpret_cast<bfloat16*>(scatter_tokens.data_ptr<at::BFloat16>()), reinterpret_cast<const float*>(smooth_scale.data_ptr<float>()), y.data_ptr<int8_t>(), 
            per_tokens_scale.data_ptr<float>(), (const int32_t*)experts_tokens_start.data_ptr<int32_t>(), (const int32_t*)experts_tokens_count.data_ptr<int32_t>(), total_experts_num, hidden_size, stream);
    }else {
        TORCH_CHECK(false, "Only float16, bfloat16 are supported");
    }
}