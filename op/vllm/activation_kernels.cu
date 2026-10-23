#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "cuda_vec_utils.cuh"
#include "dispatch_utils.h"

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

// silu_and_mul_with_clamp math, all in fp32:
//   legacy: out = silu_a(min(g, limit)) * (clamp(u, +-limit) + beta)
//   step4:  out = min(silu_a(g), limit) * (clamp(u, +-limit) + beta)
// sigmoid via the native mxcc builtins (mirrors silu_mul_scalar above).
__device__ __forceinline__ float silu_clamp_mul_scalar(
    float g, float u, const float limit, const float alpha, const float beta,
    const bool step4) {
  u = fmaxf(fminf(u, limit), -limit);
  if (!step4) {
    g = fminf(g, limit);
  }
  const float sig = __builtin_mxc_rcpf(1.0f + __builtin_expf(-g * alpha));
  if (step4) {
    return fminf(g * sig, limit) * (u + beta);
  }
  return g * sig * (u + beta);
}

// Per-128-bit-vector clamp math shared by the 2D and flat kernels.
// fp16 uses half2 SIMD; bf16/fp32 stay in fp32 scalars (bf16x2 clamps were
// measured as a net regression on MACA).
template <typename cuda_t, int VEC>
__device__ __forceinline__ int4 silu_clamp_mul_vec(
    const int4& vg, const int4& vu, const float limit, const float alpha,
    const float beta, const bool step4) {
  int4 vo;
  if constexpr (std::is_same_v<cuda_t, half>) {
    const half2* g2 = reinterpret_cast<const half2*>(&vg);
    const half2* u2 = reinterpret_cast<const half2*>(&vu);
    half2* o2 = reinterpret_cast<half2*>(&vo);
    const half2 lim2 = __float2half2_rn(limit);
    const half2 nlim2 = __float2half2_rn(-limit);
    // sigmoid(alpha*g) = rcp(1 + 2^(-alpha*log2(e)*g))
    const half2 neg_log2e_alpha =
        __float2half2_rn(-1.4426950408889634f * alpha);
    const half2 one = __float2half2_rn(1.0f);
    const half2 beta2 = __float2half2_rn(beta);
#pragma unroll
    for (int k = 0; k < VEC / 2; ++k) {
      half2 g = g2[k];
      half2 u = __hmax2(__hmin2(u2[k], lim2), nlim2);
      if (!step4) {
        g = __hmin2(g, lim2);
      }
      half2 sig = h2rcp(__hadd2(one, h2exp2(__hmul2(g, neg_log2e_alpha))));
      half2 act = __hmul2(g, sig);
      if (step4) {
        act = __hmin2(act, lim2);
      }
      o2[k] = __hmul2(act, __hadd2(u, beta2));
    }
  } else {
    const cuda_t* g = reinterpret_cast<const cuda_t*>(&vg);
    const cuda_t* u = reinterpret_cast<const cuda_t*>(&vu);
    cuda_t* o = reinterpret_cast<cuda_t*>(&vo);
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      o[k] = (cuda_t)silu_clamp_mul_scalar((float)g[k], (float)u[k], limit,
                                           alpha, beta, step4);
    }
  }
  return vo;
}

// Clamped variant of silu_and_mul_fast_kernel: same 2D (col_tiles, rows)
// grid and 128-bit vector layout.
template <typename scalar_t, typename cuda_t, int VEC>
__global__ void silu_and_mul_clamp_fast_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, 2*d]
    const int d,
    const int num_vecs,                  // d / VEC (whole 128-bit vectors)
    const float limit,
    const float alpha,
    const float beta,
    const bool step4) {
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

  const int stride = gridDim.x * blockDim.x;
  for (int vidx = blockIdx.x * blockDim.x + threadIdx.x; vidx < num_vecs;
       vidx += stride) {
    vec_t vg = __ldg(&gate_v[vidx]);
    vec_t vu = __ldg(&up_v[vidx]);
    out_v[vidx] =
        silu_clamp_mul_vec<cuda_t, VEC>(vg, vu, limit, alpha, beta, step4);
  }
}

// Flat cross-row variant for rows whose vec count is not a multiple of the
// warp width: small d strands whole warps (d=96 -> 12/64 lanes), larger
// non-aligned d strands the tail tile (d=864 -> 108 = 64+44, 44/64 on the
// second tile). Linearize [rows x num_vecs] instead and recover the row with
// one integer division per 128-bit vector -- much cheaper than the idle
// lanes it reclaims.
template <typename scalar_t, typename cuda_t, int VEC>
__global__ void silu_and_mul_clamp_flat_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, 2*d]
    const int d,
    const int num_vecs,                  // d / VEC
    const int total_vecs,                // num_tokens * num_vecs (fits int32)
    const float limit,
    const float alpha,
    const float beta,
    const bool step4) {
  using vec_t = int4;
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= total_vecs) {
    return;
  }
  const int row = v / num_vecs;
  const int col = v - row * num_vecs;
  const cuda_t* __restrict__ gate_ptr =
      reinterpret_cast<const cuda_t*>(input) + (int64_t)row * 2 * d;
  const vec_t* __restrict__ gate_v =
      reinterpret_cast<const vec_t*>(gate_ptr);
  const vec_t* __restrict__ up_v =
      reinterpret_cast<const vec_t*>(gate_ptr + d);
  vec_t* __restrict__ out_v =
      reinterpret_cast<vec_t*>(reinterpret_cast<cuda_t*>(out) +
                               (int64_t)row * d);
  out_v[col] = silu_clamp_mul_vec<cuda_t, VEC>(__ldg(&gate_v[col]),
                                               __ldg(&up_v[col]), limit,
                                               alpha, beta, step4);
}

// Scalar (non-vectorized) clamp variant for d not divisible by the 128-bit
// vector width; keeps the 2D (col_tiles, rows) grid so occupancy stays high.
template <typename scalar_t, typename cuda_t>
__global__ void silu_and_mul_clamp_scalar_kernel(
    scalar_t* __restrict__ out,          // [num_tokens, d]
    const scalar_t* __restrict__ input,  // [num_tokens, 2*d]
    const int d,
    const float limit,
    const float alpha,
    const float beta,
    const bool step4) {
  const int64_t row = blockIdx.y;
  const cuda_t* __restrict__ gate_ptr =
      reinterpret_cast<const cuda_t*>(input) + row * 2 * (int64_t)d;
  const cuda_t* __restrict__ up_ptr = gate_ptr + d;
  cuda_t* __restrict__ out_ptr =
      reinterpret_cast<cuda_t*>(out) + row * (int64_t)d;

  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < d; i += stride) {
    const float gf = (float)__ldg(&gate_ptr[i]);
    const float uf = (float)__ldg(&up_ptr[i]);
    out_ptr[i] =
        (cuda_t)silu_clamp_mul_scalar(gf, uf, limit, alpha, beta, step4);
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

void silu_and_mul_clamp(torch::Tensor& out,
                        torch::Tensor& input,
                        double limit,
                        double alpha /* = 1.0 */,
                        double beta /* = 0.0 */,
                        bool step4 /* = false */) {
  const int d = input.size(-1) / 2;
  const int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0 || d == 0) {
    return;
  }


  const bool over_cap = num_tokens > 65535;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  // Tunables (overridable via env for autotuning; defaults mirror
  // silu_and_mul's SILU_THREADS / SILU_WAVES).
  static const int ENV_THREADS = [] {
    const char* e = getenv("SILU_CLAMP_THREADS");
    int v = e ? atoi(e) : 64;
    if (v < 64) v = 64;
    if (v > 1024) v = 1024;
    return (v / 64) * 64;  // multiple of warp (64)
  }();
  static const int ENV_WAVES = [] {
    const char* e = getenv("SILU_CLAMP_WAVES");
    int v = e ? atoi(e) : 8;
    if (v < 1) v = 1;
    if (v > 64) v = 64;
    return v;
  }();
  static const int ENV_FLAT_THREADS = [] {
    const char* e = getenv("SILU_CLAMP_FLAT_THREADS");
    int v = e ? atoi(e) : 256;
    if (v < 64) v = 64;
    if (v > 1024) v = 1024;
    return (v / 64) * 64;  // multiple of warp (64)
  }();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_clamp_fast",
                               [&] {
    using cuda_t = typename vllm::CUDATypeConverter<scalar_t>::Type;
    constexpr int VEC = 16 / sizeof(scalar_t);  // 8 for bf16/fp16
    // Vector path requires each row's gate/up/out slice to be 16-byte
    // aligned; with PyTorch's 256-byte base alignment that holds iff
    // d % VEC == 0.
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
    if (use_vec && (over_cap || num_vecs < 64 || num_vecs % 64 != 0)) {
      // Rows whose vec count is not warp-aligned (d=96 -> 12; d=864 -> 108 =
      // 64+44): 2D row-tiling strands the tail tile's lanes (44/64 for
      // d=864, 84% utilization) while the flat layout packs ~100% of them;
      // the per-vector int division hides behind memory at bandwidth-bound
      // sizes (proven at d=96) and small T is launch-bound either way.
      // num_tokens <= 65535 here (guard above), so total fits int32.
      const int total_vecs = (int)(num_tokens * num_vecs);
      const int flat_grid =
          (total_vecs + ENV_FLAT_THREADS - 1) / ENV_FLAT_THREADS;
      vllm::silu_and_mul_clamp_flat_kernel<scalar_t, cuda_t, VEC>
          <<<flat_grid, ENV_FLAT_THREADS, 0, stream>>>(
              out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d,
              num_vecs, total_vecs, (float)limit, (float)alpha, (float)beta,
              step4);
    } else if (use_vec) {
      vllm::silu_and_mul_clamp_fast_kernel<scalar_t, cuda_t, VEC>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d,
              num_vecs, (float)limit, (float)alpha, (float)beta, step4);
    } else {
      vllm::silu_and_mul_clamp_scalar_kernel<scalar_t, cuda_t>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d,
              (float)limit, (float)alpha, (float)beta, step4);
    }
  });
}

void mul_and_silu(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., 2 * d]
{
  // The difference between mul_and_silu and silu_and_mul is that mul_and_silu
  // applies the silu to the latter half of the input.
  LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel,
                                false, false, 0.0f, 1.0f, 0.0f, false);
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

namespace vllm {

template <typename T>
__device__ __forceinline__ T fatrelu_kernel(const T& x, const float threshold) {
  const float f = (float)x;
  return (T)(f > threshold ? f : 0.0f);
}

template <typename packed_t>
__device__ __forceinline__ packed_t
packed_fatrelu_kernel(const packed_t& val, const float threshold) {
  float2 fval = cast_to_float2(val);
  fval.x = fval.x > threshold ? fval.x : 0.0f;
  fval.y = fval.y > threshold ? fval.y : 0.0f;
  return cast_to_packed<packed_t>(fval);
}

template <typename scalar_t, typename packed_t,
          scalar_t (*ACT_FN)(const scalar_t&, const float),
          packed_t (*PACKED_ACT_FN)(const packed_t&, const float), bool use_vec,
          bool use_256b = false>
__global__ void act_and_mul_kernel_with_param(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input, const int d,
    const float param) {
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
      for (int j = 0; j < pvec_t::NUM_ELTS; j++) {
        x.elts[j] = packed_mul(PACKED_ACT_FN(x.elts[j], param), y.elts[j]);
      }
      if constexpr (use_256b) {
        st256(x, &out_vec[i]);
      } else {
        st128(x, &out_vec[i]);
      }
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&x_ptr[idx]);
      const scalar_t y = VLLM_LDG(&y_ptr[idx]);
      out_ptr[idx] = ACT_FN(x, param) * y;
    }
  }
}

template <typename T>
__device__ __forceinline__ T swigluoai_and_mul(const T& gate, const T& up,
                                               float alpha, float limit) {
  // Clamp gate to (-inf, limit] and up to [-limit, limit]
  const float g = fminf((float)gate, limit);
  const float u = fmaxf(fminf((float)up, limit), -limit);
  // glu = gate * sigmoid(gate * alpha), then return (up + 1) * glu
  return (T)((u + 1.0f) * g / (1.0f + expf(-g * alpha)));
}

// Interleaved gate/up: input has [gate0, up0, gate1, up1, ...].
template <typename scalar_t,
          scalar_t (*ACT_FN)(const scalar_t&, const scalar_t&, const float,
                             const float)>
__global__ void swigluoai_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2 * d] (interleaved)
    const int d, const float alpha, const float limit) {
  // For interleaved data: input has 2*d elements per token (gate/up pairs)
  // output has d elements per token
  constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
  constexpr int PAIRS = VEC_SIZE / 2;  // Number of gate/up pairs per int4 load
  const int64_t token_idx = blockIdx.x;
  const scalar_t* in_ptr = input + token_idx * 2 * d;
  scalar_t* out_ptr = out + token_idx * d;

  // Check alignment for 128-bit vectorized access on input.
  // For output we use int2 (64-bit) which has 8-byte alignment requirement.
  const bool in_aligned = is_16byte_aligned(in_ptr);
  const bool out_aligned =
      (reinterpret_cast<uintptr_t>(out_ptr) & 7) == 0;  // 8-byte for int2

  if (in_aligned && out_aligned && d >= PAIRS) {
    // Fast path: vectorized loop
    // Each int4 load gives VEC_SIZE elements = PAIRS gate/up pairs
    // Each int2 store writes PAIRS output elements
    const int4* in_vec = reinterpret_cast<const int4*>(in_ptr);
    int2* out_vec = reinterpret_cast<int2*>(out_ptr);
    const int num_vecs = d / PAIRS;
    const int vec_end = num_vecs * PAIRS;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      int4 v = VLLM_LDG(&in_vec[i]);
      int2 r;
      auto* vp = reinterpret_cast<scalar_t*>(&v);
      auto* rp = reinterpret_cast<scalar_t*>(&r);
#pragma unroll
      for (int j = 0; j < PAIRS; j++) {
        rp[j] = ACT_FN(vp[2 * j], vp[2 * j + 1], alpha, limit);
      }
      out_vec[i] = r;
    }
    // Scalar cleanup for remaining elements
    for (int i = vec_end + threadIdx.x; i < d; i += blockDim.x) {
      out_ptr[i] = ACT_FN(VLLM_LDG(&in_ptr[2 * i]),
                          VLLM_LDG(&in_ptr[2 * i + 1]), alpha, limit);
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      // gate = x[..., ::2]  (even indices)
      const scalar_t gate = VLLM_LDG(&in_ptr[2 * idx]);
      // up = x[..., 1::2]   (odd indices)
      const scalar_t up = VLLM_LDG(&in_ptr[2 * idx + 1]);
      out_ptr[idx] = ACT_FN(gate, up, alpha, limit);
    }
  }
}

// SITU (Kimi SituGLU) gated activation. Non-interleaved layout:
// input = [gate(d), up(d)] per token.
//   gate_out = beta * tanh(gate / beta) * sigmoid(gate)
//   up_out   = (linear_beta > 0) ? linear_beta * tanh(up / linear_beta) : up
//   out      = gate_out * up_out
// Compute is done in fp32 and written straight to `out` -- no intermediate
// tensors and no full-tensor fp32 upcast (the pure-torch forward_native
// allocated ~8 fp32 temporaries per call, which blows up MoE profiling).
template <typename scalar_t>
__global__ void situ_and_mul_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., 2, d]
    const int d, const float beta, const float linear_beta) {
  const int64_t row = blockIdx.x;
  const scalar_t* gate_ptr = input + row * 2 * d;
  const scalar_t* up_ptr = gate_ptr + d;
  scalar_t* out_ptr = out + row * d;
  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    const float gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
    const float up_out =
        clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
    out_ptr[idx] = (scalar_t)(gate_out * up_out);
  }
}

template <typename scalar_t>
__global__ void masked_situ_and_mul_kernel(
    scalar_t* __restrict__ out, const scalar_t* __restrict__ input,
    const int* __restrict__ expert_num_tokens, const int max_num_tokens,
    const int d, const float beta, const float linear_beta) {
  const int expert = blockIdx.y;
  const int num_tokens = expert_num_tokens[expert];
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d || num_tokens == 0) {
    return;
  }

  const bool clamp_up = linear_beta > 0.0f;
  const float inv_beta = 1.0f / beta;
  const float inv_linear_beta = clamp_up ? 1.0f / linear_beta : 0.0f;
  const int64_t expert_row = static_cast<int64_t>(expert) * max_num_tokens;
  for (int token = 0; token < num_tokens; ++token) {
    const int64_t row = expert_row + token;
    const scalar_t* gate_ptr = input + row * 2 * d;
    const scalar_t* up_ptr = gate_ptr + d;
    scalar_t* out_ptr = out + row * d;
    const float g = (float)VLLM_LDG(&gate_ptr[idx]);
    const float u = (float)VLLM_LDG(&up_ptr[idx]);
    const float gate_out = beta * tanhf(g * inv_beta) / (1.0f + expf(-g));
    const float up_out =
        clamp_up ? linear_beta * tanhf(u * inv_linear_beta) : u;
    out_ptr[idx] = (scalar_t)(gate_out * up_out);
  }
}

}  // namespace vllm

#define LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(KERNEL, PACKED_KERNEL, PARAM) \
  auto dtype = input.scalar_type();                                            \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  if (num_tokens == 0) {                                                       \
    return;                                                                    \
  }                                                                            \
  dim3 grid(num_tokens);                                                       \
  int cc_major = at::cuda::getCurrentDeviceProperties()->major;                \
  int support_vec =                                                            \
      (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)            \
          ? vllm::VecTraits<true>::ARCH_MAX_VEC_SIZE                           \
          : vllm::VecTraits<false>::ARCH_MAX_VEC_SIZE;                         \
  int vec_size = support_vec / at::elementSize(dtype);                         \
  const bool use_vec = (d % vec_size == 0);                                    \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  if (use_vec) {                                                               \
    dim3 block(std::min(d / vec_size, 1024));                                  \
    if (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {         \
      VLLM_DISPATCH_FLOATING_TYPES(                                            \
          dtype, "act_and_mul_kernel_with_param", [&] {                        \
            vllm::act_and_mul_kernel_with_param<                               \
                scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,  \
                KERNEL<scalar_t>,                                              \
                PACKED_KERNEL<                                                 \
                    typename vllm::PackedTypeConverter<scalar_t>::Type>,       \
                true, true><<<grid, block, 0, stream>>>(                       \
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d,       \
                PARAM);                                                        \
          });                                                                  \
    } else {                                                                   \
      VLLM_DISPATCH_FLOATING_TYPES(                                            \
          dtype, "act_and_mul_kernel_with_param", [&] {                        \
            vllm::act_and_mul_kernel_with_param<                               \
                scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,  \
                KERNEL<scalar_t>,                                              \
                PACKED_KERNEL<                                                 \
                    typename vllm::PackedTypeConverter<scalar_t>::Type>,       \
                true, false><<<grid, block, 0, stream>>>(                      \
                out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d,       \
                PARAM);                                                        \
          });                                                                  \
    }                                                                          \
  } else {                                                                     \
    dim3 block(std::min(d, 1024));                                             \
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "act_and_mul_kernel_with_param", [&] { \
      vllm::act_and_mul_kernel_with_param<                                     \
          scalar_t, typename vllm::PackedTypeConverter<scalar_t>::Type,        \
          KERNEL<scalar_t>,                                                    \
          PACKED_KERNEL<typename vllm::PackedTypeConverter<scalar_t>::Type>,   \
          false><<<grid, block, 0, stream>>>(                                  \
          out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d, PARAM);     \
    });                                                                        \
  }

#define LAUNCH_SIGLUOAI_AND_MUL(KERNEL, ALPHA, LIMIT)                          \
  int d = input.size(-1) / 2;                                                  \
  int64_t num_tokens = input.numel() / input.size(-1);                         \
  dim3 grid(num_tokens);                                                       \
  dim3 block(std::min(d, 1024));                                               \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));            \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                \
  VLLM_DISPATCH_FLOATING_TYPES(                                                \
      input.scalar_type(), "clamp_swiglu_kernel_with_params", [&] {            \
        vllm::swigluoai_and_mul_kernel<scalar_t, KERNEL<scalar_t>>             \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),             \
                                         input.data_ptr<scalar_t>(), d, ALPHA, \
                                         LIMIT);                               \
      });

void fatrelu_and_mul(torch::Tensor& out,    // [..., d],
                     torch::Tensor& input,  // [..., 2 * d]
                     double threshold) {
  LAUNCH_ACTIVATION_GATE_KERNEL_WITH_PARAM(
      vllm::fatrelu_kernel, vllm::packed_fatrelu_kernel, threshold);
}
void swigluoai_and_mul(torch::Tensor& out,    // [..., d]
                       torch::Tensor& input,  // [..., 2 * d]
                       double alpha, double limit) {
  LAUNCH_SIGLUOAI_AND_MUL(vllm::swigluoai_and_mul, alpha, limit);
}

// Kimi SITU gated activation. `linear_beta <= 0` means "unset" (up passed
// through), matching SituAndMul(linear_beta=None) on the Python side.
void situ_and_mul(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input,  // [..., 2 * d]
                  double beta, double linear_beta) {
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  if (num_tokens == 0) {
    return;
  }
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "situ_and_mul_kernel", [&] {
        vllm::situ_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),
            d, (float)beta, (float)linear_beta);
      });
}

void masked_situ_and_mul(torch::Tensor& out,    // [E, T, d]
                         torch::Tensor& input,  // [E, T, 2 * d]
                         const torch::Tensor& expert_num_tokens,
                         double beta, double linear_beta) {
  TORCH_CHECK(out.dim() == 3 && input.dim() == 3 &&
                  out.size(0) == input.size(0) &&
                  out.size(1) == input.size(1) &&
                  out.size(2) * 2 == input.size(2),
              "masked_situ_and_mul: expected out=[E,T,d], input=[E,T,2*d] "
              "(check argument order), got out size ",
              out.sizes(), ", input size ", input.sizes());
  int num_experts = input.size(0);
  int max_num_tokens = input.size(1);
  int d = input.size(2) / 2;
  if (num_experts == 0 || max_num_tokens == 0) {
    return;
  }
  constexpr int block_size = 256;
  dim3 grid((d + block_size - 1) / block_size, num_experts);
  dim3 block(block_size);
  const at::cuda::CUDAGuard device_guard(input.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "masked_situ_and_mul_kernel", [&] {
        vllm::masked_situ_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.mutable_data_ptr<scalar_t>(), input.const_data_ptr<scalar_t>(),
            expert_num_tokens.const_data_ptr<int>(), max_num_tokens, d,
            (float)beta, (float)linear_beta);
      });
}
namespace vllm {

// Element-wise activation kernel template.
template <typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&), bool use_vec,
          bool use_256b = false>
__global__ void activation_kernel(
    scalar_t* __restrict__ out,          // [..., d]
    const scalar_t* __restrict__ input,  // [..., d]
    const int d) {
  const scalar_t* in_ptr = input + blockIdx.x * d;
  scalar_t* out_ptr = out + blockIdx.x * d;

  if constexpr (use_vec) {
    // Fast path: 128-bit/256-bit vectorized loop
    using vec_t = typename VecTraits<use_256b>::vec_t;
    constexpr int ARCH_MAX_VEC_SIZE = VecTraits<use_256b>::ARCH_MAX_VEC_SIZE;
    constexpr int VEC_SIZE = ARCH_MAX_VEC_SIZE / sizeof(scalar_t);
    const vec_t* in_vec = reinterpret_cast<const vec_t*>(in_ptr);
    vec_t* out_vec = reinterpret_cast<vec_t*>(out_ptr);
    const int num_vecs = d / VEC_SIZE;

    for (int i = threadIdx.x; i < num_vecs; i += blockDim.x) {
      vec_t v;
      if constexpr (use_256b) {
        ld256(v, &in_vec[i]);
      } else {
        v = VLLM_LDG(&in_vec[i]);
      }
      auto* vp = reinterpret_cast<scalar_t*>(&v);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; j++) {
        vp[j] = ACT_FN(vp[j]);
      }
      if constexpr (use_256b) {
        st256(v, &out_vec[i]);
      } else {
        out_vec[i] = v;
      }
    }
  } else {
    // Scalar fallback for unaligned data or small d
    for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const scalar_t x = VLLM_LDG(&in_ptr[idx]);
      out_ptr[idx] = ACT_FN(x);
    }
  }
}

}  // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                 \
  auto dtype = input.scalar_type();                                      \
  int d = input.size(-1);                                                \
  int64_t num_tokens = input.numel() / input.size(-1);                   \
  if (num_tokens == 0) {                                                 \
    return;                                                              \
  }                                                                      \
  dim3 grid(num_tokens);                                                 \
  int cc_major = at::cuda::getCurrentDeviceProperties()->major;          \
  int support_vec =                                                      \
      (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)      \
          ? vllm::VecTraits<true>::ARCH_MAX_VEC_SIZE                     \
          : vllm::VecTraits<false>::ARCH_MAX_VEC_SIZE;                   \
  int vec_size = support_vec / at::elementSize(dtype);                   \
  const bool use_vec = (d % vec_size == 0);                              \
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));      \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();          \
  if (use_vec) {                                                         \
    dim3 block(std::min(d / vec_size, 1024));                            \
    if (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128) {   \
      VLLM_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {     \
        vllm::activation_kernel<scalar_t, KERNEL<scalar_t>, true, true>  \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });                                                                \
    } else {                                                             \
      VLLM_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {     \
        vllm::activation_kernel<scalar_t, KERNEL<scalar_t>, true, false> \
            <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),       \
                                         input.data_ptr<scalar_t>(), d); \
      });                                                                \
    }                                                                    \
  } else {                                                               \
    dim3 block(std::min(d, 1024));                                       \
    VLLM_DISPATCH_FLOATING_TYPES(dtype, "activation_kernel", [&] {       \
      vllm::activation_kernel<scalar_t, KERNEL<scalar_t>, false>         \
          <<<grid, block, 0, stream>>>(out.data_ptr<scalar_t>(),         \
                                       input.data_ptr<scalar_t>(), d);   \
    });                                                                  \
  }

namespace vllm {

template <typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_quick_kernel(const T& x) {
  // x * sigmoid(1.702 * x)
  return (T)(((float)x) / (1.0f + expf(-1.702f * (float)x)));
}

template <typename T>
__device__ __forceinline__ T relu_squared_kernel(const T& x) {
  // relu(x)^2 — introduced in https://arxiv.org/abs/2109.08668v2
  const float f = (float)x;
  const float val = f > 0.0f ? f : 0.0f;
  return (T)(val * val);
}

}  // namespace vllm

void gelu_new(torch::Tensor& out,    // [..., d]
              torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(torch::Tensor& out,    // [..., d]
               torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}

void gelu_quick(torch::Tensor& out,    // [..., d]
                torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_quick_kernel);
}

void relu_squared(torch::Tensor& out,    // [..., d]
                  torch::Tensor& input)  // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::relu_squared_kernel);
}
