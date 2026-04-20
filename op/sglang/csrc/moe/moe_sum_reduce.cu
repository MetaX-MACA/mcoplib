// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <ATen/OpMathType.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <iostream>
#include <type_traits>

#include "cutlass/array.h"
#include "utils.h"

template <typename T>
using opmath_t = at::opmath_type<T>;

template <typename T>
__device__ __forceinline__ opmath_t<T> to_acc(T x) {
  return static_cast<opmath_t<T>>(x);
}

template <typename T>
__device__ __forceinline__ T from_acc(opmath_t<T> x) {
  return static_cast<T>(x);
}

template <>
__device__ __forceinline__ opmath_t<at::Half> to_acc<at::Half>(at::Half x) {
  return __half2float(__nv_half(x));
}
template <>
__device__ __forceinline__ at::Half from_acc<at::Half>(opmath_t<at::Half> x) {
  return __float2half_rn(x);
}

template <>
__device__ __forceinline__ opmath_t<at::BFloat16> to_acc<at::BFloat16>(at::BFloat16 x) {
  return __bfloat162float(__nv_bfloat16(x));
}
template <>
__device__ __forceinline__ at::BFloat16 from_acc<at::BFloat16>(opmath_t<at::BFloat16> x) {
  return __float2bfloat16_rn(x);
}

template <typename T>
__device__ __forceinline__ T ldg_cg(const T* p) {
  return __ldg(p);
}

// =============================================================================
// Vectorized Memory Access Helper (16-byte aligned for 128-bit loads)
// =============================================================================

union alignas(16) BF16Vec8 {
  uint4 u128;
  __nv_bfloat16 nv_bf16[8];
  at::BFloat16 bf16[8];

  __device__ __forceinline__ void load(const at::BFloat16* __restrict__ ptr) {
    u128 = *reinterpret_cast<const uint4*>(ptr);
  }

  __device__ __forceinline__ void store(at::BFloat16* __restrict__ ptr) const {
    *reinterpret_cast<uint4*>(ptr) = u128;
  }
};

// =============================================================================
// Optimized Kernel with Vectorized Loads (Compile-time TOPK)
// =============================================================================

template <int WARPS_PER_BLOCK, int TOPK>
__launch_bounds__(WARPS_PER_BLOCK * 32)
__global__ void moe_sum_reduce_optimized_kernel(
    const at::BFloat16* __restrict__ x,
    at::BFloat16* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  //
  // Parallelism Strategy:
  // - Each warp processes one token
  // - Each lane processes 16 elements of hidden_dim
  // - Grid-stride loop handles hidden_dim > 32 * 16
  //

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int64_t t = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;

  if (t >= token_num) return;

  constexpr int VEC_SIZE = 16;  // Process 16 BF16 elements per iteration
  const int64_t n_chunks = hidden_dim / VEC_SIZE;

  // Grid-stride loop over hidden_dim chunks
  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * 32 + lane_id;
       chunk < n_chunks;
       chunk += static_cast<int64_t>(gridDim.x) * 32) {

    const int64_t d = chunk * VEC_SIZE;
    const int64_t base = t * stride_token + d;

    // Scalar accumulators (better register allocation than arrays)
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    float a8 = 0.0f, a9 = 0.0f, a10 = 0.0f, a11 = 0.0f;
    float a12 = 0.0f, a13 = 0.0f, a14 = 0.0f, a15 = 0.0f;

    // TOPK reduction loop (fully unrolled at compile time)
    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      const int64_t offset_k = base + static_cast<int64_t>(k) * stride_topk;

      // Vectorized load: 16 BF16 = 32 bytes = 2 x uint4
      BF16Vec8 v0, v1;
      v0.load(x + offset_k);
      v1.load(x + offset_k + 8);

      // Accumulate using __nv_bfloat16 for proper type conversion
      a0 += __bfloat162float(v0.nv_bf16[0]);
      a1 += __bfloat162float(v0.nv_bf16[1]);
      a2 += __bfloat162float(v0.nv_bf16[2]);
      a3 += __bfloat162float(v0.nv_bf16[3]);
      a4 += __bfloat162float(v0.nv_bf16[4]);
      a5 += __bfloat162float(v0.nv_bf16[5]);
      a6 += __bfloat162float(v0.nv_bf16[6]);
      a7 += __bfloat162float(v0.nv_bf16[7]);
      a8 += __bfloat162float(v1.nv_bf16[0]);
      a9 += __bfloat162float(v1.nv_bf16[1]);
      a10 += __bfloat162float(v1.nv_bf16[2]);
      a11 += __bfloat162float(v1.nv_bf16[3]);
      a12 += __bfloat162float(v1.nv_bf16[4]);
      a13 += __bfloat162float(v1.nv_bf16[5]);
      a14 += __bfloat162float(v1.nv_bf16[6]);
      a15 += __bfloat162float(v1.nv_bf16[7]);
    }

    // Apply scale factor
    a0 *= scale; a1 *= scale; a2 *= scale; a3 *= scale;
    a4 *= scale; a5 *= scale; a6 *= scale; a7 *= scale;
    a8 *= scale; a9 *= scale; a10 *= scale; a11 *= scale;
    a12 *= scale; a13 *= scale; a14 *= scale; a15 *= scale;

    // Convert back to BF16 and store
    const int64_t dst = t * out_stride_token + d;
    BF16Vec8 out0, out1;

    out0.nv_bf16[0] = __float2bfloat16_rn(a0);
    out0.nv_bf16[1] = __float2bfloat16_rn(a1);
    out0.nv_bf16[2] = __float2bfloat16_rn(a2);
    out0.nv_bf16[3] = __float2bfloat16_rn(a3);
    out0.nv_bf16[4] = __float2bfloat16_rn(a4);
    out0.nv_bf16[5] = __float2bfloat16_rn(a5);
    out0.nv_bf16[6] = __float2bfloat16_rn(a6);
    out0.nv_bf16[7] = __float2bfloat16_rn(a7);

    out1.nv_bf16[0] = __float2bfloat16_rn(a8);
    out1.nv_bf16[1] = __float2bfloat16_rn(a9);
    out1.nv_bf16[2] = __float2bfloat16_rn(a10);
    out1.nv_bf16[3] = __float2bfloat16_rn(a11);
    out1.nv_bf16[4] = __float2bfloat16_rn(a12);
    out1.nv_bf16[5] = __float2bfloat16_rn(a13);
    out1.nv_bf16[6] = __float2bfloat16_rn(a14);
    out1.nv_bf16[7] = __float2bfloat16_rn(a15);

    out0.store(y + dst);
    out1.store(y + dst + 8);
  }
}

// =============================================================================
// Optimized Dynamic TOPK Kernel (runtime topk with manual unrolling)
// =============================================================================

template <int WARPS_PER_BLOCK>
__launch_bounds__(WARPS_PER_BLOCK * 32)
__global__ void moe_sum_reduce_dynamic_kernel(
    const at::BFloat16* __restrict__ x,
    at::BFloat16* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t topk_num,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  //
  // Optimized dynamic kernel with:
  // 1. Scalar accumulators for better register allocation
  // 2. Manual loop unrolling hint
  // 3. Vectorized memory access
  //

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int64_t t = static_cast<int64_t>(blockIdx.y) * WARPS_PER_BLOCK + warp_id;

  if (t >= token_num) return;

  constexpr int VEC_SIZE = 16;
  const int64_t n_chunks = hidden_dim / VEC_SIZE;

  for (int64_t chunk = static_cast<int64_t>(blockIdx.x) * 32 + lane_id;
       chunk < n_chunks;
       chunk += static_cast<int64_t>(gridDim.x) * 32) {

    const int64_t d = chunk * VEC_SIZE;
    const int64_t base = t * stride_token + d;

    // Use scalar accumulators for better performance
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
    float a8 = 0.0f, a9 = 0.0f, a10 = 0.0f, a11 = 0.0f;
    float a12 = 0.0f, a13 = 0.0f, a14 = 0.0f, a15 = 0.0f;

    // Runtime topk loop with hint for partial unrolling
    #pragma unroll 4
    for (int k = 0; k < topk_num; ++k) {
      const int64_t offset_k = base + static_cast<int64_t>(k) * stride_topk;

      BF16Vec8 v0, v1;
      v0.load(x + offset_k);
      v1.load(x + offset_k + 8);

      a0 += __bfloat162float(v0.nv_bf16[0]);
      a1 += __bfloat162float(v0.nv_bf16[1]);
      a2 += __bfloat162float(v0.nv_bf16[2]);
      a3 += __bfloat162float(v0.nv_bf16[3]);
      a4 += __bfloat162float(v0.nv_bf16[4]);
      a5 += __bfloat162float(v0.nv_bf16[5]);
      a6 += __bfloat162float(v0.nv_bf16[6]);
      a7 += __bfloat162float(v0.nv_bf16[7]);
      a8 += __bfloat162float(v1.nv_bf16[0]);
      a9 += __bfloat162float(v1.nv_bf16[1]);
      a10 += __bfloat162float(v1.nv_bf16[2]);
      a11 += __bfloat162float(v1.nv_bf16[3]);
      a12 += __bfloat162float(v1.nv_bf16[4]);
      a13 += __bfloat162float(v1.nv_bf16[5]);
      a14 += __bfloat162float(v1.nv_bf16[6]);
      a15 += __bfloat162float(v1.nv_bf16[7]);
    }

    // Apply scale
    a0 *= scale; a1 *= scale; a2 *= scale; a3 *= scale;
    a4 *= scale; a5 *= scale; a6 *= scale; a7 *= scale;
    a8 *= scale; a9 *= scale; a10 *= scale; a11 *= scale;
    a12 *= scale; a13 *= scale; a14 *= scale; a15 *= scale;

    // Store result
    const int64_t dst = t * out_stride_token + d;
    BF16Vec8 out0, out1;

    out0.nv_bf16[0] = __float2bfloat16_rn(a0);
    out0.nv_bf16[1] = __float2bfloat16_rn(a1);
    out0.nv_bf16[2] = __float2bfloat16_rn(a2);
    out0.nv_bf16[3] = __float2bfloat16_rn(a3);
    out0.nv_bf16[4] = __float2bfloat16_rn(a4);
    out0.nv_bf16[5] = __float2bfloat16_rn(a5);
    out0.nv_bf16[6] = __float2bfloat16_rn(a6);
    out0.nv_bf16[7] = __float2bfloat16_rn(a7);

    out1.nv_bf16[0] = __float2bfloat16_rn(a8);
    out1.nv_bf16[1] = __float2bfloat16_rn(a9);
    out1.nv_bf16[2] = __float2bfloat16_rn(a10);
    out1.nv_bf16[3] = __float2bfloat16_rn(a11);
    out1.nv_bf16[4] = __float2bfloat16_rn(a12);
    out1.nv_bf16[5] = __float2bfloat16_rn(a13);
    out1.nv_bf16[6] = __float2bfloat16_rn(a14);
    out1.nv_bf16[7] = __float2bfloat16_rn(a15);

    out0.store(y + dst);
    out1.store(y + dst + 8);
  }
}

// =============================================================================
// Small Token Kernel (token_num <= 128) with Vectorized Loads
// =============================================================================

template <int TOPK>
__launch_bounds__(256)
__global__ void moe_sum_reduce_small_token_bf16_kernel(
    const at::BFloat16* __restrict__ x,
    at::BFloat16* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const float scale) {
  //
  // Small token kernel with vectorized loads for BF16
  // Uses shared memory for better performance on small batches
  //

  constexpr int VEC_SIZE = 8;  // 8 BF16 per vector load

  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x * VEC_SIZE + threadIdx.x * VEC_SIZE;
         d < hidden_dim;
         d += blockDim.x * gridDim.x * VEC_SIZE) {

      const int64_t base = t * stride_token + d;

      // Accumulators for 8 elements
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;

      #pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        const int64_t offset_k = base + static_cast<int64_t>(k) * stride_topk;

        BF16Vec8 v;
        v.load(x + offset_k);

        a0 += __bfloat162float(v.nv_bf16[0]);
        a1 += __bfloat162float(v.nv_bf16[1]);
        a2 += __bfloat162float(v.nv_bf16[2]);
        a3 += __bfloat162float(v.nv_bf16[3]);
        a4 += __bfloat162float(v.nv_bf16[4]);
        a5 += __bfloat162float(v.nv_bf16[5]);
        a6 += __bfloat162float(v.nv_bf16[6]);
        a7 += __bfloat162float(v.nv_bf16[7]);
      }

      a0 *= scale; a1 *= scale; a2 *= scale; a3 *= scale;
      a4 *= scale; a5 *= scale; a6 *= scale; a7 *= scale;

      BF16Vec8 out;
      out.nv_bf16[0] = __float2bfloat16_rn(a0);
      out.nv_bf16[1] = __float2bfloat16_rn(a1);
      out.nv_bf16[2] = __float2bfloat16_rn(a2);
      out.nv_bf16[3] = __float2bfloat16_rn(a3);
      out.nv_bf16[4] = __float2bfloat16_rn(a4);
      out.nv_bf16[5] = __float2bfloat16_rn(a5);
      out.nv_bf16[6] = __float2bfloat16_rn(a6);
      out.nv_bf16[7] = __float2bfloat16_rn(a7);

      out.store(y + t * out_stride_token + d);
    }
  }
}

template <typename scalar_t, int TOPK>
__global__ void moe_sum_reduce_small_token_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const opmath_t<scalar_t> scale) {

  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);

      #pragma unroll
      for (int k = 0; k < TOPK; ++k) {
        acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
      }

      acc *= scale;
      y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
    }
  }
}

template <typename scalar_t>
__global__ void moe_sum_reduce_small_token_general_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const opmath_t<scalar_t> scale) {

  for (int t = blockIdx.y; t < token_num; t += gridDim.y) {
    for (int d = blockIdx.x * blockDim.x + threadIdx.x; d < hidden_dim; d += blockDim.x * gridDim.x) {
      const int64_t base = t * stride_token + d;
      opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);

      #pragma unroll 4
      for (int k = 0; k < topk_num; ++k) {
        acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
      }

      acc *= scale;
      y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
    }
  }
}

// =============================================================================
// Warp-per-Token Kernel for General Types (token_num > 128, non-BF16)
// =============================================================================

template <typename scalar_t, int TOPK, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const opmath_t<scalar_t> scale) {

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
    const int64_t base = t * stride_token + d;

    #pragma unroll
    for (int k = 0; k < TOPK; ++k) {
      acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
    }
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
  }
}

template <typename scalar_t, int WARPS_PER_BLOCK>
__global__ void moe_sum_reduce_warp_token_general_kernel(
    const scalar_t* __restrict__ x,
    scalar_t* __restrict__ y,
    const int64_t token_num,
    const int64_t hidden_dim,
    const int64_t stride_token,
    const int64_t stride_topk,
    const int64_t out_stride_token,
    const int topk_num,
    const opmath_t<scalar_t> scale) {

  const int warp_id = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int64_t t = (int64_t)blockIdx.y * WARPS_PER_BLOCK + warp_id;
  if (t >= token_num) return;

  for (int64_t d = (int64_t)blockIdx.x * 32 + lane; d < hidden_dim; d += (int64_t)gridDim.x * 32) {
    opmath_t<scalar_t> acc = opmath_t<scalar_t>(0);
    const int64_t base = t * stride_token + d;

    #pragma unroll 4
    for (int k = 0; k < topk_num; ++k) {
      acc += to_acc<scalar_t>(x[base + (int64_t)k * stride_topk]);
    }
    acc *= scale;
    y[t * out_stride_token + d] = from_acc<scalar_t>(acc);
  }
}

// =============================================================================
// Host Dispatch Function
// =============================================================================

void moe_sum_reduce(at::Tensor& input, at::Tensor& output, double routed_scaling_factor) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  TORCH_CHECK(input.dim() == 3, "input must be a 3D tensor like [token_num, topk_num, hidden_dim]");
  TORCH_CHECK(output.dim() == 2, "output must be [token_num, hidden_dim]");
  TORCH_CHECK(input.size(0) == output.size(0), "token dim mismatch");
  TORCH_CHECK(input.size(2) == output.size(1), "hidden_dim mismatch");

  TORCH_CHECK(input.is_contiguous(), "expect input to be contiguous");
  TORCH_CHECK(output.is_contiguous(), "expect output to be contiguous");

  const int64_t token_num = input.size(0);
  const int64_t topk_num = input.size(1);
  const int64_t hidden_dim = input.size(2);

  const int64_t in_stride_token = input.stride(0);
  const int64_t in_stride_topk = input.stride(1);
  const int64_t out_stride_token = output.stride(0);

  auto stream = at::cuda::getCurrentCUDAStream();

  // Fast path for BF16 with vectorized loads
  const bool use_bf16_optimized = (input.scalar_type() == at::kBFloat16) &&
                                   (hidden_dim % 8 == 0);

  if (use_bf16_optimized) {
    const float scale = static_cast<float>(routed_scaling_factor);

    if (token_num > 128) {
      // Warp-per-token strategy for large batches
      constexpr int WARPS_PER_BLOCK = 8;
      constexpr int THREADS = WARPS_PER_BLOCK * 32;

      const int64_t n_chunks = hidden_dim / 16;
      int64_t grid_x = (n_chunks + 32 - 1) / 32;
      if (grid_x > 65535) grid_x = 65535;

      int64_t grid_y = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
      if (grid_y > 65535) grid_y = 65535;

      dim3 block(THREADS);
      dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

      // Dispatch based on topk_num (add support for common values including 9)
      #define LAUNCH_OPT_KERNEL(TOPK) \
        moe_sum_reduce_optimized_kernel<WARPS_PER_BLOCK, TOPK><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const at::BFloat16*>(input.data_ptr<at::BFloat16>()), \
            reinterpret_cast<at::BFloat16*>(output.data_ptr<at::BFloat16>()), \
            token_num, hidden_dim, \
            in_stride_token, in_stride_topk, out_stride_token, scale);

      switch (topk_num) {
        case 1:  LAUNCH_OPT_KERNEL(1);  break;
        case 2:  LAUNCH_OPT_KERNEL(2);  break;
        case 3:  LAUNCH_OPT_KERNEL(3);  break;
        case 4:  LAUNCH_OPT_KERNEL(4);  break;
        case 6:  LAUNCH_OPT_KERNEL(6);  break;
        case 8:  LAUNCH_OPT_KERNEL(8);  break;
        case 9:  LAUNCH_OPT_KERNEL(9);  break;  // DeepSeek uses topk=9
        case 12: LAUNCH_OPT_KERNEL(12); break;
        case 16: LAUNCH_OPT_KERNEL(16); break;
        case 32: LAUNCH_OPT_KERNEL(32); break;
        default:
          moe_sum_reduce_dynamic_kernel<WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
              reinterpret_cast<const at::BFloat16*>(input.data_ptr<at::BFloat16>()),
              reinterpret_cast<at::BFloat16*>(output.data_ptr<at::BFloat16>()),
              token_num, hidden_dim, topk_num,
              in_stride_token, in_stride_topk, out_stride_token, scale);
          break;
      }
      #undef LAUNCH_OPT_KERNEL

    } else {
      // Small token: block-per-token strategy with vectorized loads
      const int block_size = 256;
      int64_t grid_x = (hidden_dim + block_size * 8 - 1) / (block_size * 8);
      grid_x = grid_x > 65535 ? 65535 : grid_x;
      int64_t grid_y = token_num < 65535 ? token_num : 65535;

      dim3 block(block_size);
      dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

      #define LAUNCH_SMALL_BF16_KERNEL(TOPK) \
        moe_sum_reduce_small_token_bf16_kernel<TOPK><<<grid, block, 0, stream>>>( \
            reinterpret_cast<const at::BFloat16*>(input.data_ptr<at::BFloat16>()), \
            reinterpret_cast<at::BFloat16*>(output.data_ptr<at::BFloat16>()), \
            token_num, hidden_dim, \
            in_stride_token, in_stride_topk, out_stride_token, scale);

      switch (topk_num) {
        case 1:  LAUNCH_SMALL_BF16_KERNEL(1);  break;
        case 2:  LAUNCH_SMALL_BF16_KERNEL(2);  break;
        case 3:  LAUNCH_SMALL_BF16_KERNEL(3);  break;
        case 4:  LAUNCH_SMALL_BF16_KERNEL(4);  break;
        case 6:  LAUNCH_SMALL_BF16_KERNEL(6);  break;
        case 8:  LAUNCH_SMALL_BF16_KERNEL(8);  break;
        case 9:  LAUNCH_SMALL_BF16_KERNEL(9);  break;
        case 12: LAUNCH_SMALL_BF16_KERNEL(12); break;
        case 16: LAUNCH_SMALL_BF16_KERNEL(16); break;
        default:
          // Fallback to general kernel
          AT_DISPATCH_FLOATING_TYPES_AND2(
              at::kHalf, at::kBFloat16, input.scalar_type(), "moe_sum_reduce_small_token", [&] {
                using scalar_t_ = scalar_t;
                using acc_t_ = opmath_t<scalar_t_>;
                const acc_t_ scale = static_cast<acc_t_>(routed_scaling_factor);
                moe_sum_reduce_small_token_general_kernel<scalar_t_><<<grid, block, 0, stream>>>(
                    input.data_ptr<scalar_t_>(),
                    output.data_ptr<scalar_t_>(),
                    token_num, hidden_dim,
                    in_stride_token, in_stride_topk, out_stride_token,
                    static_cast<int>(topk_num), scale);
              });
          break;
      }
      #undef LAUNCH_SMALL_BF16_KERNEL
    }

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel launch failed");
    return;
  }

  // Fallback for other data types (FP16, etc.)
  const bool per_token_use_one_warp = (token_num > 128);

  if (!per_token_use_one_warp) {
    // Small token: block-per-token strategy
    const int block_size = 256;
    int64_t grid_x = (hidden_dim + block_size - 1) / block_size;
    grid_x = grid_x > 65535 ? 65535 : grid_x;
    int64_t grid_y = token_num < 65535 ? token_num : 65535;

    dim3 block(block_size);
    dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y));

    #define LAUNCH_SMALL_TOKEN_KERNEL(TOPK) \
      moe_sum_reduce_small_token_kernel<scalar_t_, TOPK><<<grid, block, 0, stream>>>( \
          input.data_ptr<scalar_t_>(), \
          output.data_ptr<scalar_t_>(), \
          token_num, hidden_dim, \
          in_stride_token, in_stride_topk, out_stride_token, scale);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "moe_sum_reduce_cuda_small_token", [&] {
          using scalar_t_ = scalar_t;
          using acc_t_ = opmath_t<scalar_t_>;
          const acc_t_ scale = static_cast<acc_t_>(routed_scaling_factor);

          switch (topk_num) {
            case 1:  LAUNCH_SMALL_TOKEN_KERNEL(1);  break;
            case 2:  LAUNCH_SMALL_TOKEN_KERNEL(2);  break;
            case 4:  LAUNCH_SMALL_TOKEN_KERNEL(4);  break;
            case 8:  LAUNCH_SMALL_TOKEN_KERNEL(8);  break;
            case 9:  LAUNCH_SMALL_TOKEN_KERNEL(9);  break;
            case 16: LAUNCH_SMALL_TOKEN_KERNEL(16); break;
            default:
              moe_sum_reduce_small_token_general_kernel<scalar_t_><<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t_>(),
                  output.data_ptr<scalar_t_>(),
                  token_num, hidden_dim,
                  in_stride_token, in_stride_topk, out_stride_token,
                  static_cast<int>(topk_num), scale);
              break;
          }
        });
    #undef LAUNCH_SMALL_TOKEN_KERNEL

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (small-token) launch failed");

  } else {
    // Warp-per-token strategy
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;

    int64_t gx = (hidden_dim + 32 - 1) / 32;
    gx = gx > 65535 ? 65535 : gx;

    int64_t gy = (token_num + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gy = gy > 65535 ? 65535 : gy;

    dim3 block(THREADS);
    dim3 grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));

    #define LAUNCH_WARP_PER_TOKEN_KERNEL(TOPK) \
      moe_sum_reduce_warp_token_kernel<scalar_t_, TOPK, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>( \
          input.data_ptr<scalar_t_>(), \
          output.data_ptr<scalar_t_>(), \
          token_num, hidden_dim, \
          in_stride_token, in_stride_topk, out_stride_token, scale);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf, at::kBFloat16, input.scalar_type(), "moe_sum_reduce_cuda_large_token", [&] {
          using scalar_t_ = scalar_t;
          using acc_t_ = opmath_t<scalar_t_>;
          const acc_t_ scale = static_cast<acc_t_>(routed_scaling_factor);

          switch (topk_num) {
            case 1:  LAUNCH_WARP_PER_TOKEN_KERNEL(1);  break;
            case 2:  LAUNCH_WARP_PER_TOKEN_KERNEL(2);  break;
            case 4:  LAUNCH_WARP_PER_TOKEN_KERNEL(4);  break;
            case 8:  LAUNCH_WARP_PER_TOKEN_KERNEL(8);  break;
            case 9:  LAUNCH_WARP_PER_TOKEN_KERNEL(9);  break;
            case 16: LAUNCH_WARP_PER_TOKEN_KERNEL(16); break;
            default:
              moe_sum_reduce_warp_token_general_kernel<scalar_t_, WARPS_PER_BLOCK><<<grid, block, 0, stream>>>(
                  input.data_ptr<scalar_t_>(),
                  output.data_ptr<scalar_t_>(),
                  token_num, hidden_dim,
                  in_stride_token, in_stride_topk, out_stride_token,
                  static_cast<int>(topk_num), scale);
              break;
          }
        });
    #undef LAUNCH_WARP_PER_TOKEN_KERNEL

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "moe_sum_reduce CUDA kernel (warp-token) launch failed");
  }
}