// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <maca_fp8.h>
#include "../dispatch_utils.h"

#ifndef USE_ROCM
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
#endif

#define _USE_C600_

template<int N> 
__device__  __forceinline__ void copy(void * src, void* dst){
    int8_t* ptr_src = (int8_t*)src;
    int8_t* ptr_dst = (int8_t*)dst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
        ptr_dst[i] = ptr_src[i];
    }
}

template<>
__device__ __forceinline__ void copy<16>(void* src, void* dst) {
    float4 *ptr_src = (float4*)src;
    float4* ptr_dst = (float4*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<8>(void* src, void* dst) {
    float2 *ptr_src = (float2*)src;
    float2* ptr_dst = (float2*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<4>(void* src, void* dst) {
    float *ptr_src = (float*)src;
    float* ptr_dst = (float*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<2>(void* src, void* dst) {
    half *ptr_src = (half*)src;
    half* ptr_dst = (half*)dst;
    *ptr_dst = *ptr_src;
}

template<>
__device__ __forceinline__ void copy<1>(void* src, void* dst) {
    int8_t *ptr_src = (int8_t*)src;
    int8_t* ptr_dst = (int8_t*)dst;
    *ptr_dst = *ptr_src;
}

template<typename T>
__device__ T __forceinline__ convertT(float value) {
    return T(value);
}

template<>
__device__ half __forceinline__ convertT(float value) {
    return __float2half(value);
}

template<>
__device__ maca_bfloat16 __forceinline__ convertT(float value) {
    return __float2bfloat16(value);
}

// template<>
// __device__ __maca_fp8_e4m3 __forceinline__ converT(float value) {
//     return 
// }

template<typename T>
__device__ float __forceinline__ convertTofloat(T value) {
    return float(value);
}

template<>
__device__ float __forceinline__ convertTofloat(half value) {
    return __half2float(value);
}

template<>
__device__ float __forceinline__ convertTofloat(maca_bfloat16 value) {
    return __bfloat162float(value);
}

typedef __NATIVE_VECTOR__(4, float) v4f32;
typedef __NATIVE_VECTOR__(4, _Float16) v4f16;

//SRC_T bfloat or half DST_T __maca_fp8_e4m3
__global__ void per_token_cast_to_f8_kernel(const maca_bfloat16* input, __maca_fp8_e4m3* dst_quant, float* scale, int num_elems) {
    float reg_in[8];
    int num_threads = blockDim.x;
    int thread_offset = (blockIdx.x * num_threads + threadIdx.x) * 8;
    float abs_max = 1e-4f;
    if(thread_offset < num_elems) {
        float4 tmp = *(float4*)(input + thread_offset);
        maca_bfloat16* ptr_tmp = (maca_bfloat16*)&tmp;
        #pragma unroll 8
        for(int i = 0; i < 8; i++) {
            reg_in[i] = convertTofloat<maca_bfloat16>(ptr_tmp[i]);
            abs_max = max(fabs(reg_in[i]), abs_max);
        }
    }

    for(int i = 8; i >= 1; i= i >> 1) {
        abs_max = max(__shfl_down_sync_16(0xffffffffffffffff, abs_max, i), abs_max);
    }

    __shared__ float sm_max[64];
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
        sm_max[group_id] = abs_max;
        if(thread_offset < num_elems) {
            int scale_offset = thread_offset >> 7;

            scale[scale_offset] = abs_max / 448;
        }
    }
    __syncthreads();

    if(thread_offset < num_elems) {
        __maca_fp8_e4m3 reg_dst[8];
        
        float div = 448.0 / sm_max[group_id];
        #pragma unroll 8
        for(int i = 0; i < 8; i++) {
            reg_in[i] = reg_in[i] * div;
        }
#ifdef _USE_C600_
        v4f16 reg_tmp0, reg_tmp1;
        #pragma unroll 4
        for(int i = 0; i < 4; i++) {
            reg_tmp0[i] = _Float16(reg_in[i]);
            reg_tmp1[i] = _Float16(reg_in[i + 4]);
        }
        *(uint32_t*)reg_dst = __builtin_mxc_cvt_pk4_f16tof8(reg_tmp0);
        *(uint32_t*)(reg_dst + 4) = __builtin_mxc_cvt_pk4_f16tof8(reg_tmp1);
#else
        *(uint32_t*)reg_dst = __builtin_mxc_cvt_pk4_f32tof8(*(v4f32*)reg_in);
        *(uint32_t*)(reg_dst + 4) = __builtin_mxc_cvt_pk4_f32tof8(*((v4f32*)(reg_in + 4)));
#endif
        copy<sizeof(__maca_fp8_e4m3) * 8>((void*)reg_dst, (void*)(dst_quant + thread_offset));
    }
}

void per_token_cast_to_fp8(
    torch::Tensor& out,
    torch::Tensor& scale,   
    torch::Tensor const& input)
{
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(scale.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    int64_t const hidden_size = input.size(-1);
    TORCH_CHECK((hidden_size % 128) == 0);
    int64_t num_elems = input.numel();
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if(out.dtype() == torch::kFloat8_e4m3fn && input.dtype() == at::ScalarType::BFloat16) {
        int block_size = 512;
        int64_t gridSize = (num_elems + block_size * 8 - 1) / (block_size * 8);
        auto input_buffer = reinterpret_cast<maca_bfloat16*>(input.data_ptr<at::BFloat16>());
        auto out_buffer = reinterpret_cast<__maca_fp8_e4m3 *>(out.data_ptr<at::Float8_e4m3fn>());
        auto scale_buffer = reinterpret_cast<float*>(scale.data_ptr());
        per_token_cast_to_f8_kernel<<<gridSize, block_size, 0, stream>>>((const maca_bfloat16*)input_buffer, out_buffer, scale_buffer, num_elems);
        return;
    } else {
            TORCH_CHECK(0,
                "per_token_cast_to_fp8 doesn't support this type");
    }
}