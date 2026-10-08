/*
 * 
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cmath>
#include "cuda_vec_utils.cuh"
#include "dispatch_utils.h"
#include "mcoplib_sgl_common.cuh"
namespace vllm {

template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          bool act_first,
          bool HAS_CLAMP>
__device__ __forceinline__ scalar_t compute(const scalar_t& x,
                                            const scalar_t& y,
                                            const float limit,
                                            const float alpha,
                                            const float beta,
                                            const bool step4) {
  if constexpr (act_first) {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      if (step4) {
        up = (scalar_t)fmaxf(fminf((float)up, limit), -limit);
      } else {
        gate = (scalar_t)fminf((float)gate, limit);
        up = (scalar_t)fmaxf(fminf((float)up, limit), -limit);
      }
    }

    const float activated = (float)ACT_FN(gate, alpha);
    return (scalar_t)((step4 ? fminf(activated, limit) : activated) *
                      ((float)up + beta));
  } else {
    scalar_t gate = x;
    scalar_t up = y;
    if constexpr (HAS_CLAMP) {
      gate = (scalar_t)fmaxf(fminf((float)gate, limit), -limit);
      up = (scalar_t)fminf((float)up, limit);
    }
    
    return (scalar_t)(((float)gate + beta) * ACT_FN(up, alpha));
  }
}

template <typename packed_t,
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first,
          bool HAS_CLAMP>
__device__ __forceinline__ packed_t packed_compute(
    const packed_t& x,
    const packed_t& y,
    const float limit,
    const float alpha,
    const float beta,
    const bool step4) {
  if constexpr (act_first) {
    packed_t gate = x;
    packed_t up = y;
    float2 u = cast_to_float2(up);

    if constexpr (HAS_CLAMP) {
      if (step4) {
        u.x = fmaxf(fminf(u.x, limit), -limit);
        u.y = fmaxf(fminf(u.y, limit), -limit);
      } else {
        float2 g = cast_to_float2(gate);
        g.x = fminf(g.x, limit);
        g.y = fminf(g.y, limit);
        gate = cast_to_packed<packed_t>(g);
        u.x = fmaxf(fminf(u.x, limit), -limit);
        u.y = fmaxf(fminf(u.y, limit), -limit);
      }
    }

    float2 activated = cast_to_float2(PACKED_ACT_FN(gate, alpha));
    if (step4) {
      activated.x = fminf(activated.x, limit);
      activated.y = fminf(activated.y, limit);
    }
    activated.x *= (u.x + beta);
    activated.y *= (u.y + beta);

    return cast_to_packed<packed_t>(activated);
  } else {
    packed_t gate = x;
    packed_t up = y;
    float2 g = cast_to_float2(gate);

    if constexpr (HAS_CLAMP) {
      float2 u = cast_to_float2(up);
      g.x = fmaxf(fminf(g.x, limit), -limit);
      g.y = fmaxf(fminf(g.y, limit), -limit);
      u.x = fminf(u.x, limit);
      u.y = fminf(u.y, limit);
      up = cast_to_packed<packed_t>(u);
    }

    float2 activated = cast_to_float2(PACKED_ACT_FN(up, alpha));
    activated.x *= (g.x + beta);
    activated.y *= (g.y + beta);

    return cast_to_packed<packed_t>(activated);
  }
}

// Activation and gating kernel template.
template <typename scalar_t,
          typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float),
          bool act_first,
          bool use_vec,
          bool HAS_CLAMP,
          bool use_256b = false>
__global__ void act_and_mul_kernel(
    scalar_t* __restrict__ out,         // [..., d]
    const scalar_t* __restrict__ input, // [..., 2, d]
    const int d,
    const float limit,
    const float alpha,
    const float beta,
    const bool step4) {
  const scalar_t* x_ptr = input + blockIdx.x * 2 * d;
  const scalar_t* y_ptr = x_ptr + d;
  scalar_t* out_ptr = out + blockIdx.x * d;

  if constexpr (use_vec) {
    using cuda_t = typename CUDATypeConverter<scalar_t>::Type;
    using pvec_t = PackedVec<cuda_t, use_256b>;

    const pvec_t* x_vec = reinterpret_cast<const pvec_t*>(x_ptr);
    const pvec_t* y_vec = reinterpret_cast<const pvec_t*>(y_ptr);
    pvec_t* out_vec = reinterpret_cast<pvec_t*>(out_ptr);

    const int num_vecs = d / 2 / pvec_t::NUM_ELTS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      pvec_t x, y;

      if constexpr (use_256b) {
        ld256(x, &x_vec[i]);
        ld256(y, &y_vec[i]);
      } else {
        ld128(x, &x_vec[i]);
        ld128(y, &y_vec[i]);
      }

#pragma unroll
      for (int j = 0; j < pvec_t::NUM_ELTS; ++j) {
        x.elts[j] =
            packed_compute<packed_t,
                           PACKED_ACT_FN,
                           act_first,
                           HAS_CLAMP>(
                x.elts[j],
                y.elts[j],
                limit,
                alpha,
                beta,
                step4);
      }

      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        st128(x, &out_vec[i]);
      }
    }
  } else {
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);

      out_ptr[idx] =
          compute<scalar_t,
                  ACT_FN,
                  act_first,
                  HAS_CLAMP>(
              x,
              y,
              limit,
              alpha,
              beta,
              step4);
    }
  }
}

template <typename T>
__device__ __forceinline__ T silu_kernel(const T& x, const float alpha) {
  // x * sigmoid(alpha * x)
  return (T)(((float)x) / (1.0f + expf((float)-x * alpha)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_silu_kernel(const packed_t& val,
                                                       const float alpha) {
  // x * sigmoid(alpha * x)
  float2 fval = cast_to_float2(val);
  fval.x = fval.x / (1.0f + expf(-fval.x * alpha));
  fval.y = fval.y / (1.0f + expf(-fval.y * alpha));
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_kernel(const T& x, const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  const float f = (float)x;
  constexpr float ALPHA = M_SQRT1_2;
  return (T)(f * 0.5f * (1.0f + ::erf(f * ALPHA)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_gelu_kernel(const packed_t& val,
                                                       const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'none' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L36-L38
  constexpr float ALPHA = M_SQRT1_2;
  float2 fval = cast_to_float2(val);
  fval.x = fval.x * 0.5f * (1.0f + ::erf(fval.x * ALPHA));
  fval.y = fval.y * 0.5f * (1.0f + ::erf(fval.y * ALPHA));
  return cast_to_packed<packed_t>(fval);
}

template <typename T>
__device__ __forceinline__ T gelu_tanh_kernel(const T& x, const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  const float f = (float)x;
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  return (T)(0.5f * f * (1.0f + ::tanhf(inner)));
}

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_gelu_tanh_kernel(const packed_t& val, const float /*alpha*/) {
  // Equivalent to PyTorch GELU with 'tanh' approximation.
  // Refer to:
  // https://github.com/pytorch/pytorch/blob/8ac9b20d4b090c213799e81acf48a55ea8d437d6/aten/src/ATen/native/cuda/ActivationGeluKernel.cu#L25-L30
  float2 fval = cast_to_float2(val);
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;

  float x_cube = fval.x * fval.x * fval.x;
  float inner = BETA * (fval.x + KAPPA * x_cube);
  fval.x = 0.5f * fval.x * (1.0f + ::tanhf(inner));

  x_cube = fval.y * fval.y * fval.y;
  inner = BETA * (fval.y + KAPPA * x_cube);
  fval.y = 0.5f * fval.y * (1.0f + ::tanhf(inner));
  return cast_to_packed<packed_t>(fval);
}

}  // namespace vllm

// Launch activation and gating kernel.
// Use ACT_FIRST (bool) indicating whether to apply the activation function
// first. HAS_CLAMP (bool) enables pre-activation clamping: gate input is
// clamped (max only) and up input is clamped (both sides) before the
// activation function is applied.
#define LAUNCH_ACTIVATION_GATE_KERNEL(KERNEL, PACKED_KERNEL, ACT_FIRST, \
                                      HAS_CLAMP, LIMIT, ALPHA, BETA, STEP4)     \
auto dtype = input.scalar_type();                                        \
int d = input.size(-1) / 2;                                              \
int64_t num_tokens = input.numel() / input.size(-1);                     \
if (num_tokens == 0) {                                                   \
  return;                                                                \
}                                                                        \
dim3 grid(num_tokens);                                                   \
int cc_major = at::cuda::getCurrentDeviceProperties()->major;            \
int support_vec =                                                        \
    (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)        \
        ? vllm::VecTraits<true>::ARCH_MAX_VEC_SIZE                       \
        : vllm::VecTraits<false>::ARCH_MAX_VEC_SIZE;                     \
int vec_size = support_vec / at::elementSize(dtype);                     \
const bool use_vec = (d % vec_size == 0);                                \
const at::cuda::OptionalCUDAGuard device_guard(device_of(input));        \
const cudaStream_t stream = at::cuda::getCurrentCUDAStream();            \
if (use_vec) {                                                           \
  dim3 block(std::min(d / vec_size, 1024));                              \
  if (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {     \
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {      \
      vllm::act_and_mul_kernel<                                          \
          scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,  \
          KERNEL<scalar_t>,                                              \
          PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>, \
          ACT_FIRST, true, HAS_CLAMP, true><<<grid, block, 0, stream>>>( \
          out.data_ptr<scalar_t>(),                                      \
          input.data_ptr<scalar_t>(),                                    \
          d,                                                             \
          LIMIT,                                                         \
          ALPHA,                                                         \
          BETA,                                                          \
          STEP4);                                                         \
    });                                                                  \
  } else {                                                               \
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {      \
      vllm::act_and_mul_kernel<                                          \
          scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,  \
          KERNEL<scalar_t>,                                              \
          PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>, \
          ACT_FIRST, true, HAS_CLAMP, false><<<grid, block, 0, stream>>>(\
          out.data_ptr<scalar_t>(),                                      \
          input.data_ptr<scalar_t>(),                                    \
          d,                                                             \
          LIMIT,                                                         \
          ALPHA,                                                         \
          BETA, STEP4);                                                         \
    });                                                                  \
  }                                                                      \
} else {                                                                 \
  dim3 block(std::min(d, 1024));                                         \
  VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel", [&] {        \
    vllm::act_and_mul_kernel<                                            \
        scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,    \
        KERNEL<scalar_t>,                                                \
        PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>, \
        ACT_FIRST, false, HAS_CLAMP><<<grid, block, 0, stream>>>(        \
        out.data_ptr<scalar_t>(),                                        \
        input.data_ptr<scalar_t>(),                                      \
        d,                                                               \
        LIMIT,                                                           \
        ALPHA,                                                           \
        BETA, STEP4);                                                           \
  });                                                                    \
}

namespace vllm {

template <typename T>
__device__ __forceinline__ float silu_mul_scalar(float g, float u) {
  // silu(g) * u = g * sigmoid(g) * u = (g / (1 + e^{-g})) * u
  float sig = __builtin_mxc_rcpf(1.0f + __builtin_expf(-g));
  return g * sig * u;
}

template <typename scalar_t, typename cuda_t, int VEC>
__global__ void silu_and_mul_fast_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, 2*d]
    const int d,
    const int num_vecs) {                // d / VEC (whole 128-bit vectors)
  using vec_t = int4;  // 16 bytes = VEC elements (VEC = 16/sizeof(scalar_t))

  const int64_t row = blockIdx.y;
  const cuda_t* __restrict__ gate_ptr =
      reinterpret_cast<const cuda_t*>(input) + row * 2 * (int64_t)d;
  const cuda_t* __restrict__ up_ptr = gate_ptr + d;
  cuda_t* __restrict__ out_ptr =
      reinterpret_cast<cuda_t*>(out) + row * (int64_t)d;

  const vec_t* __restrict__ gate_v = reinterpret_cast<const vec_t*>(gate_ptr);
  const vec_t* __restrict__ up_v = reinterpret_cast<const vec_t*>(up_ptr);
  vec_t* __restrict__ out_v = reinterpret_cast<vec_t*>(out_ptr);

  // Coalesced block-strided loop over the row's whole vectors. grid.x blocks
  // cooperate on one row, so the stride is (blocks-per-row * threads).
  const int stride = gridDim.x * blockDim.x;
  for (int vidx = blockIdx.x * blockDim.x + threadIdx.x; vidx < num_vecs;
       vidx += stride) {
    vec_t vg = __ldg(&gate_v[vidx]);
    vec_t vu = __ldg(&up_v[vidx]);
    cuda_t* g = reinterpret_cast<cuda_t*>(&vg);
    cuda_t* u = reinterpret_cast<cuda_t*>(&vu);
    vec_t vo;
    cuda_t* o = reinterpret_cast<cuda_t*>(&vo);
    if constexpr (std::is_same_v<cuda_t, half>) {
      half2* g2 = reinterpret_cast<half2*>(&vg);
      half2* u2 = reinterpret_cast<half2*>(&vu);
      half2* o2 = reinterpret_cast<half2*>(&vo);
      const half2 neg_log2e = __float2half2_rn(-1.4426950408889634f);
      const half2 one = __float2half2_rn(1.0f);
#pragma unroll
      for (int k = 0; k < VEC / 2; ++k) {
        half2 e = h2exp2(__hmul2(g2[k], neg_log2e));
        half2 sig = h2rcp(__hadd2(one, e));
        o2[k] = __hmul2(g2[k], __hmul2(sig, u2[k]));
      }
    } else {
#pragma unroll
      for (int k = 0; k < VEC; ++k) {
        o[k] = (cuda_t)silu_mul_scalar<cuda_t>((float)g[k], (float)u[k]);
      }
    }
    out_v[vidx] = vo;
  }
}

// Scalar (non-vectorized) variant used when the per-row hidden size `d` is not
// a multiple of the 128-bit vector width, which would make `up_ptr` /
// `out_ptr` misaligned for 16-byte loads (e.g. d = 1537, 257). Still uses the
// 2D (col_tiles, num_tokens) grid so occupancy stays high for small batch.
template <typename scalar_t, typename cuda_t>
__global__ void silu_and_mul_scalar_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, 2*d]
    const int d) {
  const int64_t row = blockIdx.y;
  const cuda_t* __restrict__ gate_ptr =
      reinterpret_cast<const cuda_t*>(input) + row * 2 * (int64_t)d;
  const cuda_t* __restrict__ up_ptr = gate_ptr + d;
  cuda_t* __restrict__ out_ptr =
      reinterpret_cast<cuda_t*>(out) + row * (int64_t)d;

  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d; i += stride) {
    float gf = (float)__ldg(&gate_ptr[i]);
    float uf = (float)__ldg(&up_ptr[i]);
    out_ptr[i] = (cuda_t)silu_mul_scalar<cuda_t>(gf, uf);
  }
}

}  // namespace vllm

void silu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  const int d = input.size(-1) / 2;
  if (d == 1537){ //avoid Performance regression
    LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel,
                                true, false, 0.0f, 1.0f, 0.0f, false);

  } else{
    const int64_t num_tokens = input.numel() / input.size(-1);
    if (num_tokens == 0 || d == 0) {
      return;
    }
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int sm_count = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

    // Tunables (overridable via env for autotuning; sensible defaults baked in).
    static const int ENV_THREADS = [] {
      const char* e = getenv("SILU_THREADS");
      int v = e ? atoi(e) : 64;
      if (v < 64) v = 64;
      if (v > 1024) v = 1024;
      return (v / 64) * 64;  // multiple of warp (64)
    }();
    static const int ENV_WAVES = [] {
      const char* e = getenv("SILU_WAVES");
      int v = e ? atoi(e) : 8;
      if (v < 1) v = 1;
      if (v > 64) v = 64;
      return v;
    }();

    VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_fast", [&] {
      using cuda_t = typename vllm::CUDATypeConverter<scalar_t>::Type;
      constexpr int VEC = 16 / sizeof(scalar_t);  // 8 for bf16/fp16
      // Vector path requires each row's gate/up/out slice to be 16-byte aligned.
      // Given PyTorch's 256-byte base alignment, that holds iff d % VEC == 0.
      const bool use_vec = (d % VEC == 0);
      const int num_vecs = use_vec ? (d / VEC) : 0;
      const int items = use_vec ? num_vecs : d;

      const int target_blocks = sm_count * ENV_WAVES;

      int threads = ENV_THREADS;
      if (items > 0 && items < threads) {
        threads = ((items + 63) / 64) * 64;
        if (threads < 64) threads = 64;
      }

      // Full-row tiling: blocks to cover the row once (1 item/thread).
      int max_tiles = (items + threads - 1) / threads;
      if (max_tiles < 1) max_tiles = 1;
      int col_tiles = max_tiles;

      // For small batch, add column tiles so num_tokens*col_tiles fills all SMs.
      if (num_tokens < (int64_t)target_blocks) {
        int want = (int)((target_blocks + num_tokens - 1) / num_tokens);
        if (want < 1) want = 1;
        if (col_tiles < want) col_tiles = want;
        if (col_tiles > max_tiles) col_tiles = max_tiles;
      }

      dim3 grid(col_tiles, num_tokens);
      dim3 block(threads);
      if (use_vec) {
        vllm::silu_and_mul_fast_kernel<scalar_t, cuda_t, VEC>
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),
                                        input.data_ptr<scalar_t>(), d, num_vecs);
      } else {
        vllm::silu_and_mul_scalar_kernel<scalar_t, cuda_t>
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),
                                        input.data_ptr<scalar_t>(), d);
      }
    });
}
}

void gelu_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::gelu_kernel, vllm::packed_gelu_kernel,
                                true, false, 0.0f, 1.0f, 0.0f, false);
}

void gelu_tanh_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input)  // [..., 2 * d]
{
  LAUNCH_ACTIVATION_GATE_KERNEL(
      vllm::gelu_tanh_kernel, vllm::packed_gelu_tanh_kernel, true, false, 0.0f, 1.0f, 0.0f, false);
}

