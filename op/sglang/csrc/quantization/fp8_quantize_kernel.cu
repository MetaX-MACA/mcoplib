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
typedef __NATIVE_VECTOR__(4, float)        v4f32;
typedef __NATIVE_VECTOR__(4, _Float16)     v4f16;
typedef __NATIVE_VECTOR__(4, unsigned int) v4u32;
typedef __NATIVE_VECTOR__(2, unsigned int) v2u32;

template <typename T, int VEC, bool TRANS_SCATTER, bool NT>
__global__ __launch_bounds__(256)
void per_token_cast_to_f8_kernel(const T* __restrict__ input,
                                 __maca_fp8_e4m3* __restrict__ dst_quant,
                                 float* __restrict__ scale,
                                 int num_elems, int m, int G)
{
    constexpr int WORDS = VEC / 2;
    constexpr int LANES = 128 / VEC;

    const int tid = threadIdx.x;
    const int off = (blockIdx.x * blockDim.x + tid) * VEC;
    if (off >= num_elems) return;

    unsigned int w[WORDS];
    if (VEC == 16) {
        if (NT) {
            *reinterpret_cast<v4u32*>(w)     = __builtin_nontemporal_load(reinterpret_cast<const v4u32*>(input + off));
            *reinterpret_cast<v4u32*>(w + 4) = __builtin_nontemporal_load(reinterpret_cast<const v4u32*>(input + off + 8));
        } else {
            *reinterpret_cast<v4u32*>(w)     = *reinterpret_cast<const v4u32*>(input + off);
            *reinterpret_cast<v4u32*>(w + 4) = *reinterpret_cast<const v4u32*>(input + off + 8);
        }
    } else if (VEC == 8) {
        if (NT) *reinterpret_cast<v4u32*>(w) = __builtin_nontemporal_load(reinterpret_cast<const v4u32*>(input + off));
        else    *reinterpret_cast<v4u32*>(w) = *reinterpret_cast<const v4u32*>(input + off);
    } else {
        if (NT) *reinterpret_cast<v2u32*>(w) = __builtin_nontemporal_load(reinterpret_cast<const v2u32*>(input + off));
        else    *reinterpret_cast<v2u32*>(w) = *reinterpret_cast<const v2u32*>(input + off);
    }

    const T* p = reinterpret_cast<const T*>(w);
    float f[VEC];
    float abs_max = 1e-4f;
    #pragma unroll
    for (int i = 0; i < VEC; ++i) { f[i] = convertTofloat<T>(p[i]); abs_max = fmaxf(abs_max, fabsf(f[i])); }

    #pragma unroll
    for (int s = LANES >> 1; s >= 1; s >>= 1)
        abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffffffffffffULL, abs_max, s));

    if ((tid & (LANES - 1)) == 0) {
        const int g = off >> 7;
        scale[TRANS_SCATTER ? (g % G) * m + g / G : g] = abs_max * (1.0f / 448.0f);
    }

    const float div = __fdividef(448.0f, abs_max);
    #pragma unroll
    for (int i = 0; i < VEC; ++i) f[i] *= div;

    __maca_fp8_e4m3 d[VEC];
    #pragma unroll
    for (int k = 0; k < VEC / 4; ++k) {
#ifdef _USE_C600_
        v4f16 t;
        #pragma unroll
        for (int i = 0; i < 4; ++i) t[i] = _Float16(f[k * 4 + i]);
        *reinterpret_cast<uint32_t*>(d + k * 4) = __builtin_mxc_cvt_pk4_f16tof8(t);
#else
        *reinterpret_cast<uint32_t*>(d + k * 4) =
            __builtin_mxc_cvt_pk4_f32tof8(*reinterpret_cast<v4f32*>(f + k * 4));
#endif
    }

    auto* dp = dst_quant + off;
    if (VEC == 16) {
        if (NT) __builtin_nontemporal_store(*reinterpret_cast<const v4u32*>(d), reinterpret_cast<v4u32*>(dp));
        else    *reinterpret_cast<v4u32*>(dp) = *reinterpret_cast<const v4u32*>(d);
    } else if (VEC == 8) {
        if (NT) __builtin_nontemporal_store(*reinterpret_cast<const v2u32*>(d), reinterpret_cast<v2u32*>(dp));
        else    *reinterpret_cast<v2u32*>(dp) = *reinterpret_cast<const v2u32*>(d);
    } else {
        if (NT) __builtin_nontemporal_store(*reinterpret_cast<const unsigned int*>(d), reinterpret_cast<unsigned int*>(dp));
        else    *reinterpret_cast<unsigned int*>(dp) = *reinterpret_cast<const unsigned int*>(d);
    }
}

__global__ __launch_bounds__(256)
void transpose_scale_kernel(const float* __restrict__ src, float* __restrict__ dst, int m, int G)
{
    __shared__ float tile[32][33];
    const int bx = blockIdx.x * 32, by = blockIdx.y * 32;
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int r = by + threadIdx.y + j, c = bx + threadIdx.x;
        if (r < m && c < G) tile[threadIdx.y + j][threadIdx.x] = src[(long)r * G + c];
    }
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < 32; j += 8) {
        int r = bx + threadIdx.y + j, c = by + threadIdx.x;
        if (r < G && c < m) dst[(long)r * m + c] = tile[threadIdx.x][threadIdx.y + j];
    }
}

void per_token_cast_to_fp8(torch::Tensor& out, torch::Tensor& scale,
                           torch::Tensor const& input, bool trans_scale)
{
    TORCH_CHECK(input.is_contiguous() && out.is_contiguous());
    TORCH_CHECK(out.dtype() == torch::kFloat8_e4m3fn);
    const int64_t m = input.size(0), n = input.size(1);
    TORCH_CHECK((n % 128) == 0);
    const int64_t G = n / 128, num_elems = input.numel();
    const bool is_bf16 = input.dtype() == at::ScalarType::BFloat16;
    TORCH_CHECK(is_bf16 || input.dtype() == at::ScalarType::Half,
                "per_token_cast_to_fp8 doesn't support this type");

    const auto stream = at::cuda::getCurrentCUDAStream();
    #ifdef _USE_C600_
        constexpr int BLK = 256;
    #else
        constexpr int BLK = 128;
    #endif
    constexpr int MIN_BLOCKS = 32 * 8;
    auto out_buf = reinterpret_cast<__maca_fp8_e4m3*>(out.data_ptr<at::Float8_e4m3fn>());
    float* sptr = reinterpret_cast<float*>(scale.data_ptr());
    void* in_ptr = input.data_ptr();

    const bool small = !trans_scale && m <= 256;

    int vec = 16;
    if (num_elems < (int64_t)BLK * 16 * MIN_BLOCKS) vec = 8;
    if (num_elems < (int64_t)BLK *  8 * MIN_BLOCKS) vec = 4;

    const bool use_ws = trans_scale && G >= 8 && m >= 32;
    torch::Tensor ws;
    float* sbuf = sptr;
    if (use_ws) {
        ws = torch::empty({m, G}, input.options().dtype(torch::kFloat32));
        sbuf = ws.data_ptr<float>();
    }
    const bool scatter = trans_scale && !use_ws;

#define LAUNCH(T, V, S, NT)                                                   \
    per_token_cast_to_f8_kernel<T, V, S, NT>                                  \
        <<<(num_elems + (int64_t)BLK * (V) - 1) / ((int64_t)BLK * (V)),       \
           BLK, 0, stream>>>(                                                 \
            (const T*)in_ptr, out_buf, sbuf, (int)num_elems, (int)m, (int)G)
#define DISPATCH(T, S) do {                                                   \
    if (small)          LAUNCH(T, 8, S, false);                               \
    else if (vec == 16) LAUNCH(T, 16, S, true);                               \
    else if (vec == 8)  LAUNCH(T, 8, S, true);                                \
    else                LAUNCH(T, 4, S, true); } while (0)

    if (is_bf16) { if (scatter) DISPATCH(maca_bfloat16, true); else DISPATCH(maca_bfloat16, false); }
    else         { if (scatter) DISPATCH(half, true);          else DISPATCH(half, false); }

    if (use_ws)
        transpose_scale_kernel<<<dim3((G + 31) / 32, (m + 31) / 32), dim3(32, 8), 0, stream>>>(
            sbuf, sptr, (int)m, (int)G);
}