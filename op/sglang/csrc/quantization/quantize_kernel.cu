// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#include "../dispatch_utils.h"

#ifndef USE_ROCM
  #include <cub/util_type.cuh>
  #include <cub/cub.cuh>
#else
  #include <hipcub/util_type.hpp>
  #include <hipcub/hipcub.hpp>
#endif

#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

struct DivModFast {
    DivModFast(int d = 1)
    {
        d_ = (d == 0) ? 1 : d;
        for (l_ = 0;; ++l_) {
            if ((1U << l_) >= d_)
                break;
        }
        uint64_t one = 1;
        uint64_t m   = ((one << 32) * ((one << l_) - d_)) / d_ + 1;
        m_           = static_cast<uint32_t>(m);
    }

    __device__ __inline__ int div(int idx) const
    {
        uint32_t tm = __umulhi(m_, idx); // get high 32-bit of the product
        return (tm + idx) >> l_;
    }

    __device__ __inline__ int mod(int idx) const
    {
        return idx - d_ * div(idx);
    }

    __device__ __inline__ void divmod(int idx, int &quo, int &rem)
    {
        quo = div(idx);
        rem = idx - quo * d_;
    }
    
    uint32_t d_; // divisor
    uint32_t l_; // ceil(log2(d_))
    uint32_t m_; // m' in the papaer
};

constexpr static uint32_t seil = 0x03020706u;
typedef __NATIVE_VECTOR__(2, float) v2f;
#define CVT_B0TOF32(q, out) out = __builtin_mxc_b0_cast_to_f32(q);
#define CVT_B1TOF32(q, out) out = __builtin_mxc_b1_cast_to_f32(q);
#define CVT_B2TOF32(q, out) out = __builtin_mxc_b2_cast_to_f32(q);
#define CVT_B3TOF32(q, out) out = __builtin_mxc_b3_cast_to_f32(q);

__device__ __forceinline__ void f32x2_cvt_bf16x2(uint32_t& dst, float src[2]) {
    uint32_t tmp[2];
    tmp[0] = __builtin_mxc_ubfe(*(reinterpret_cast<uint32_t*>(src)), 16, 1);
    tmp[0] = tmp[0] + *reinterpret_cast<uint32_t*>(src);
    tmp[0] = (uint32_t)0x7fff + tmp[0];
    tmp[1] = __builtin_mxc_ubfe(*(reinterpret_cast<uint32_t*>(src + 1)), 16, 1);
    tmp[1] = tmp[1] + *(reinterpret_cast<uint32_t*>(src + 1));
    tmp[1] = (uint32_t)0x7fff + tmp[1];
    dst = __builtin_mxc_byte_perm(tmp[0], tmp[1], seil);
}

template<class scalar_t>
__device__ __forceinline__ void awq_dequant_4bits(const uint32_t& p, scalar_t (&out)[8], scalar_t (&scale)[8], const uint32_t& scale_zero) {
    if constexpr(std::is_same_v<scalar_t, __maca_bfloat16>) {
        v2f v2z,v2scale, v2neg;
        v2neg[0] = -1.0f; v2neg[1] = -1.0f;
        v2z[0] = 0.0f; v2z[1] = 0.0f;
        v2f v2zero;
        float tmp[2];
        int p0 = p & 0x0f0f0f0f;
        int z0 = scale_zero & 0x0f0f0f0f;
        CVT_B0TOF32(p0, tmp[0]);
        CVT_B2TOF32(p0, tmp[1]);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[0]; v2scale[1] = (float)scale[1];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)out), tmp);

        CVT_B1TOF32(p0, tmp[0]);
        CVT_B3TOF32(p0, tmp[1]);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[4]; v2scale[1] = (float)scale[5];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 4)), tmp);

        p0 = (p >> 4) & 0x0f0f0f0f;
        z0 = (scale_zero >> 4) & 0x0f0f0f0f;
        CVT_B0TOF32(p0, tmp[0]);
        CVT_B2TOF32(p0, tmp[1]);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[2]; v2scale[1] = (float)scale[3];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 2)), tmp);

        CVT_B1TOF32(p0, tmp[0]);
        CVT_B3TOF32(p0, tmp[1]);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);

        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(v2zero, v2neg, *((v2f*)tmp));
        v2scale[0] = (float)scale[6]; v2scale[1] = (float)scale[7];
        *((v2f*)tmp) = __builtin_mxc_pk_fma_f32(*((v2f*)tmp), v2scale, v2z);
        f32x2_cvt_bf16x2(*((uint32_t*)(out + 6)), tmp);
    } else {
        v2f a0, v2z, v2scale, v2zero, v2neg;
        v2z[0] = 0; v2z[1] = 0;
        v2neg[0] = -1.0f; v2neg[1] = -1.0f;
        int p0 = p & 0x0f0f0f0f;
        int z0 = scale_zero & 0x0f0f0f0f;
        CVT_B0TOF32(p0, a0.x);
        CVT_B2TOF32(p0, a0.y);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[0]; v2scale[1] = (float)scale[1];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[0] = (scalar_t)a0.x;
        out[1] = (scalar_t)a0.y;

        CVT_B1TOF32(p0, a0.x);
        CVT_B3TOF32(p0, a0.y);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[4]; v2scale[1] = (float)scale[5];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[4] = (scalar_t)a0.x;
        out[5] = (scalar_t)a0.y;

        p0 = (p >> 4) & 0x0f0f0f0f;
        z0 = (scale_zero >> 4) & 0x0f0f0f0f;
        CVT_B0TOF32(p0, a0.x);
        CVT_B2TOF32(p0, a0.y);
        CVT_B0TOF32(z0, v2zero[0]);
        CVT_B2TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[2]; v2scale[1] = (float)scale[3];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[2] = (scalar_t)a0.x;
        out[3] = (scalar_t)a0.y;

        CVT_B1TOF32(p0, a0.x);
        CVT_B3TOF32(p0, a0.y);
        CVT_B1TOF32(z0, v2zero[0]);
        CVT_B3TOF32(z0, v2zero[1]);
        a0 = __builtin_mxc_pk_fma_f32(v2zero, v2neg, a0);
        v2scale[0] = (float)scale[6]; v2scale[1] = (float)scale[7];
        a0 = __builtin_mxc_pk_fma_f32(a0, v2scale, v2z);
        out[6] = (scalar_t)a0.x;
        out[7] = (scalar_t)a0.y;
    }
}

template<typename T>
__global__ void dequantize_weights_opt(int* __restrict__ B, T* __restrict__ scaling_factors,
                       int* __restrict__ zeros, T* __restrict__ C, int G, int length, int blocksize, int num_elems, DivModFast length_fast) {
    constexpr int N = 8;
    T B_loaded_scale[8];
    T B_shared[8];
    int tid = blockIdx.x * blocksize + threadIdx.x;
    if(tid >= num_elems) return;
    // int row = tid / length;
    // int col = tid % length;
    int row, col;
    length_fast.divmod(tid, row, col);
    int group_row = row / G;
    int group_offset = group_row * length + col;
    int offset = row * length + col;
    uint32_t* ptr_zeros = (uint32_t*)(zeros + group_offset);
    uint32_t* ptr_B = (uint32_t*)(B + offset);
    T* ptr_scale = scaling_factors + group_offset * N;
    T* ptr_C = C + offset * N;
    uint32_t zeros_loaded = *(uint32_t*)ptr_zeros;
    uint32_t B_loaded = *(uint32_t*)ptr_B;
    *(uint4*)(B_loaded_scale) = *(uint4*)(ptr_scale);
    awq_dequant_4bits<T>(B_loaded,B_shared, B_loaded_scale, zeros_loaded);
    *(float4*)(ptr_C) = *(float4*)(B_shared);
}


torch::Tensor mx_awq_dequantize(torch::Tensor _kernel,
                             torch::Tensor _scaling_factors,
                             torch::Tensor _zeros, int64_t split_k_iters,
                             int64_t thx, int64_t thy) {
  DEBUG_TRACE_PARAMS(_kernel, _scaling_factors, _zeros, split_k_iters, thx, thy);
  DEBUG_DUMP_PARAMS(_kernel, _scaling_factors, _zeros, split_k_iters, thx, thy);
  int in_c = _kernel.size(0);
  int qout_c = _kernel.size(1);
  int out_c = qout_c * 8;
  int G = in_c / _scaling_factors.size(0);

  auto options = torch::TensorOptions()
                     .dtype(_scaling_factors.dtype())
                     .device(_scaling_factors.device());
  at::Tensor _de_kernel = torch::empty({in_c, out_c}, options);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto kernel = reinterpret_cast<int*>(_kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<int*>(_zeros.data_ptr<int>());

  int blocksize = 512;
  int num_elems = in_c * qout_c;
  int gridsize = (num_elems + blocksize - 1) / blocksize;
  if(_scaling_factors.dtype() == at::ScalarType::Half) {
    auto de_kernel = reinterpret_cast<half*>(_de_kernel.data_ptr<at::Half>());
    auto scaling_factors =
        reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
    dequantize_weights_opt<__half><<<gridsize, blocksize, 0 , stream>>>(kernel, scaling_factors, zeros, de_kernel, G, qout_c, blocksize, num_elems, DivModFast(qout_c));
  } else if(_scaling_factors.dtype() == at::ScalarType::BFloat16) {
    auto de_kernel = reinterpret_cast<maca_bfloat16*>(_de_kernel.data_ptr<at::BFloat16>());
    auto scaling_factors =
        reinterpret_cast<maca_bfloat16*>(_scaling_factors.data_ptr<at::BFloat16>());
    dequantize_weights_opt<maca_bfloat16><<<gridsize, blocksize, 0, stream>>>(kernel, scaling_factors, zeros, de_kernel, G, qout_c, blocksize, num_elems, DivModFast(qout_c));
  } else {
    TORCH_CHECK(0,
                "awq_dequantize doesn't support this type");
  }
  return _de_kernel;
}
