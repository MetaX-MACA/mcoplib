#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <type_traits>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/all.h>

#include "utils.h"

#define FP8_E4M3_MAX 448

enum MoeFinDtype { MOE_FIN_FP32 = 0, MOE_FIN_FP16 = 1, MOE_FIN_BF16 = 2 , MOE_FIN_FP8};

enum MoeFinStatus {
  MOE_FIN_OK         = 0,
  MOE_FIN_ERR_NULL   = 1,  // 必选指针为空
  MOE_FIN_ERR_MODE   = 2,  // drop_pad_mode ∉ [0,3]
  MOE_FIN_ERR_DIM    = 3,  // 维度非法
  MOE_FIN_ERR_DEPEND = 4,  // x2 无 x1 / bias 无 expertIdx
  MOE_FIN_ERR_DTYPE  = 5,  // dtype 编码非法
};

inline const char* moe_fin_status_str(MoeFinStatus s) {
  switch (s) {
    case MOE_FIN_OK:         return "OK";
    case MOE_FIN_ERR_NULL:   return "ERR_NULL";
    case MOE_FIN_ERR_MODE:   return "ERR_MODE";
    case MOE_FIN_ERR_DIM:    return "ERR_DIM";
    case MOE_FIN_ERR_DEPEND: return "ERR_DEPEND";
    case MOE_FIN_ERR_DTYPE:  return "ERR_DTYPE";
  }
  return "ERR_UNKNOWN";
}

typedef __NATIVE_VECTOR__(4, float)        v4f32;

template <typename T> struct Cvt;
template <> struct Cvt<float> {
  static __device__ __forceinline__ float up(float v) { return v; }
};
template <> struct Cvt<__half> {
  static __device__ __forceinline__ float up(__half v) { return __half2float(v); }
};

template <> struct Cvt<__nv_bfloat16> {
  static __device__ __forceinline__ float up(__nv_bfloat16 v) { return __bfloat162float(v); }
};

template <> struct Cvt<__nv_fp8_e4m3> {
  static __device__ __forceinline__ float up(__nv_fp8_e4m3 v) { return static_cast<float>(v); }
};

// VEC 固定两种取值：16B/sizeof(T)（向量化主路径，要求 H % VEC == 0）或 1（标量兜底）
template <typename T, int VEC> struct ElemIO;

template <> struct ElemIO<float, 4> {
  static __device__ __forceinline__ void ld(const float* p, float* v) {
    const float4 r = *reinterpret_cast<const float4*>(p);
    v[0] = r.x; v[1] = r.y; v[2] = r.z; v[3] = r.w;
  }
  static __device__ __forceinline__ void st(float* p, const float* v) {
    float4 r;
    r.x = v[0]; r.y = v[1]; r.z = v[2]; r.w = v[3];
    *reinterpret_cast<float4*>(p) = r;
  }
};
template <> struct ElemIO<float, 1> {
  static __device__ __forceinline__ void ld(const float* p, float* v) { v[0] = p[0]; }
  static __device__ __forceinline__ void st(float* p, const float* v) { p[0] = v[0]; }
};
template <> struct ElemIO<__half, 8> {
  static __device__ __forceinline__ void ld(const __half* p, float* v) {
    float4 vreg = *reinterpret_cast<const float4*>(p);
    const __half2* q = reinterpret_cast<const __half2*>(&vreg);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 f = __half22float2(q[t]);
      v[2 * t] = f.x; v[2 * t + 1] = f.y;
    }
  }
  static __device__ __forceinline__ void st(__half* p, const float* v) {
    float4 vreg;
    __half2* q = reinterpret_cast<__half2*>(&vreg);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      q[t] = __floats2half2_rn(v[2 * t], v[2 * t + 1]);
    }
  *(float4*)p = vreg;
  }
};

template <> struct ElemIO<__half, 1> {
  static __device__ __forceinline__ void ld(const __half* p, float* v) { v[0] = __half2float(p[0]); }
  static __device__ __forceinline__ void st(__half* p, const float* v) { p[0] = __float2half_rn(v[0]); }
};

template <> struct ElemIO<__nv_bfloat16, 8> {
  static __device__ __forceinline__ void ld(const __nv_bfloat16* p, float* v) {
    float4 vreg = *(float4*)p;
    const __nv_bfloat162* q = reinterpret_cast<const __nv_bfloat162*>(&vreg);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const float2 f = __bfloat1622float2(q[t]);
      v[2 * t] = f.x; v[2 * t + 1] = f.y;
    }
  }
  static __device__ __forceinline__ void st(__nv_bfloat16* p, const float* v) {
    float4 vreg;
    __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(&vreg);
#pragma unroll
    for (int t = 0; t < 4; ++t)
      q[t] = __floats2bfloat162_rn(v[2 * t], v[2 * t + 1]);
    *(float4*)p = vreg;
  }
};
template <> struct ElemIO<__nv_bfloat16, 1> {
  static __device__ __forceinline__ void ld(const __nv_bfloat16* p, float* v) { v[0] = __bfloat162float(p[0]); }
  static __device__ __forceinline__ void st(__nv_bfloat16* p, const float* v) { p[0] = __float2bfloat16_rn(v[0]); }
};

template <> struct ElemIO<__nv_fp8_e4m3, 16> {
  static __device__ __forceinline__ void ld(const __nv_fp8_e4m3* p, float* v) {
    float4 vreg = *reinterpret_cast<const float4*>(p);
    const __nv_fp8_e4m3* q = reinterpret_cast<const __nv_fp8_e4m3*>(p);
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      const float f = static_cast<float>(q[t]);
      v[t] = f;
    }
  }
  static __device__ __forceinline__ void st(__nv_fp8_e4m3* p, const float* v) {
     float4 vreg;
     uint32_t* q = reinterpret_cast<uint32_t*>(&vreg);
     #pragma unroll
     for(int k = 0; k < 4; k++) {
        q[k] = __builtin_mxc_cvt_pk4_f32tof8(*reinterpret_cast<const v4f32*>(v + k * 4));
     }
     *reinterpret_cast<float4*>(p) = vreg;
  }
};
template <> struct ElemIO<__nv_fp8_e4m3, 1> {
  static __device__ __forceinline__ void ld(const __nv_fp8_e4m3* p, float* v) { v[0] = static_cast<float>(p[0]); }
  static __device__ __forceinline__ void st(__nv_fp8_e4m3* p, const float* v) { float val = fmaxf(fminf(v[0], FP8_E4M3_MAX), -FP8_E4M3_MAX);p[0] = static_cast<__nv_fp8_e4m3>(val); }
};

template <typename T, typename TS, int VEC>
__global__ void __launch_bounds__(1024)
moe_finalize_routing_v2_kernel(
    const T* __restrict__ expanded_x,
    const int* __restrict__ expanded_row_idx,
    const T* __restrict__ x1_opt,
    const T* __restrict__ x2_opt,
    const T* __restrict__ bias_opt,
    const TS* __restrict__ scales_opt,
    const int* __restrict__ expert_idx_opt,
    T* __restrict__ out,
    const int64_t num_rows, const int64_t k, const int64_t h,
    const int drop_pad_mode)
{
  const int64_t i = blockIdx.x;
  const int64_t j =
      (int64_t)blockIdx.y * ((int64_t)blockDim.x * VEC) + (int64_t)threadIdx.x * VEC;
  if (j >= h) return;

  const bool row_major = (drop_pad_mode & 2) != 0;
  float acc[VEC];

  if (x1_opt != nullptr) {
    ElemIO<T, VEC>::ld(x1_opt + i * h + j, acc);
    if (x2_opt != nullptr) {
      float t[VEC];
      ElemIO<T, VEC>::ld(x2_opt + i * h + j, t);
#pragma unroll
      for (int v = 0; v < VEC; ++v) acc[v] += t[v];
    }
  } else if (x2_opt != nullptr) {
    ElemIO<T, VEC>::ld(x2_opt + i * h + j, acc);
  } else {
#pragma unroll
    for (int v = 0; v < VEC; ++v) acc[v] = 0.f;
  }

  for (int64_t kk = 0; kk < k; ++kk) {
    const int64_t pos = row_major ? (i * k + kk) : (i + kk * num_rows);
    const int r = expanded_row_idx[pos];
    if (r < 0) continue;  // 丢弃：整项跳过（含 bias）
    const float s = (scales_opt != nullptr) ? Cvt<TS>::up(scales_opt[i * k + kk]) : 1.f;
    const T* xr = expanded_x + (int64_t)r * h + j;

    float xv[VEC];
    ElemIO<T, VEC>::ld(xr, xv);
    if (bias_opt != nullptr) {
      const int e = expert_idx_opt[i * k + kk];
      float bv[VEC];
      ElemIO<T, VEC>::ld(bias_opt + (int64_t)e * h + j, bv);
#pragma unroll
      for (int v = 0; v < VEC; ++v) acc[v] += s * (xv[v] + bv[v]);
    } else {
#pragma unroll
      for (int v = 0; v < VEC; ++v) acc[v] += s * xv[v];
    }
  }
  ElemIO<T, VEC>::st(out + i * h + j, acc);
}

template <typename T, typename TS>
static inline void moe_fin_launch(
    const T* expanded_x, const int* expanded_row_idx, const T* x1, const T* x2, const T* bias,
    const TS* scales, const int* eidx, T* out,
    int64_t num_rows, int64_t k, int64_t h, int mode, cudaStream_t stream) {
  constexpr int TPB = 256;
  constexpr int VMAX = 16 / (int)sizeof(T);
  const dim3 block(TPB);
  if ((h % VMAX) == 0) {
    const int64_t slice = (int64_t)TPB * VMAX;
    const dim3 grid((unsigned)num_rows, (unsigned)((h + slice - 1) / slice));
    moe_finalize_routing_v2_kernel<T, TS, VMAX>
        <<<grid, block, 0, stream>>>(expanded_x, expanded_row_idx, x1, x2, bias, scales, eidx, out,
                                     num_rows, k, h, mode);
  } else {
    const int64_t slice = TPB;  // 标量兜底，支持任意 H
    const dim3 grid((unsigned)num_rows, (unsigned)((h + slice - 1) / slice));
    moe_finalize_routing_v2_kernel<T, TS, 1>
        <<<grid, block, 0, stream>>>(expanded_x, expanded_row_idx, x1, x2, bias, scales, eidx, out,
                                     num_rows, k, h, mode);
  }
}

// 注意：raw 指针接口无法校验 tensor 长度，以下一致性由调用方保证：
//   expanded_x 行数       = (drop_pad_mode&1) ? E*C : num_rows*k
//   expanded_row_idx 长度 = num_rows*k (int32)
//   x1/x2/out             = num_rows*h；bias = E*h；scales/expert_idx = num_rows*k
//   drop-pad 模式 rowIdx ∈ [-1, E*C)；drop-less 模式 rowIdx ∈ [0, num_rows*k)
//   expert_idx ∈ [0, E)
inline MoeFinStatus moe_finalize_routing_v2(
    const void* expanded_x,
    const void* expanded_row_idx,
    const void* x1_optional,        // 可空
    const void* x2_optional,        // 可空，需 x1 非空
    const void* bias_optional,      // 可空，需 expert_idx 非空
    const void* scales_optional,    // 可空，空则 scale≡1（此时 K=k）
    const void* expert_idx_optional,// 可空
    void* out,
    int64_t num_rows, int64_t k, int64_t h,
    int64_t drop_pad_mode,          // 0..3
    int dtype,                      // MoeFinDtype
    int scales_is_fp32,             // scales 是否为 fp32（否则与主 dtype 相同）
    cudaStream_t stream) {
  if (expanded_x == nullptr || expanded_row_idx == nullptr || out == nullptr)
    return MOE_FIN_ERR_NULL;
  if (drop_pad_mode < 0 || drop_pad_mode > 3) return MOE_FIN_ERR_MODE;
  if (num_rows <= 0 || k <= 0 || h <= 0) return MOE_FIN_ERR_DIM;
  if (x2_optional != nullptr && x1_optional == nullptr) return MOE_FIN_ERR_DEPEND;
  if (bias_optional != nullptr && expert_idx_optional == nullptr) return MOE_FIN_ERR_DEPEND;

  const int* eidx = (const int*)expert_idx_optional;
  const int mode = (int)drop_pad_mode;

#define MOE_FIN_CASE(T, TS)                                                     \
  moe_fin_launch<T, TS>((const T*)expanded_x, (const int*)expanded_row_idx, (const T*)x1_optional,      \
                        (const T*)x2_optional, (const T*)bias_optional,         \
                        (const TS*)scales_optional, eidx, (T*)out, num_rows, k, \
                        h, mode, stream)

  if (dtype == MOE_FIN_FP32) {
    MOE_FIN_CASE(float, float);
  } else if (dtype == MOE_FIN_FP16) {
    if (scales_is_fp32 || scales_optional == nullptr)
      MOE_FIN_CASE(__half, float);
    else
      MOE_FIN_CASE(__half, __half);
  } else if (dtype == MOE_FIN_BF16) {
    if (scales_is_fp32 || scales_optional == nullptr)
      MOE_FIN_CASE(__nv_bfloat16, float);
    else
      MOE_FIN_CASE(__nv_bfloat16, __nv_bfloat16);
  } else if (dtype == MOE_FIN_FP8) {
    if (scales_is_fp32 || scales_optional == nullptr)
      MOE_FIN_CASE(__nv_fp8_e4m3, float);
    else
      MOE_FIN_CASE(__nv_fp8_e4m3, __nv_fp8_e4m3);
  } else {
    return MOE_FIN_ERR_DTYPE;
  }
#undef MOE_FIN_CASE
  return MOE_FIN_OK;
}

void moe_finalize_routing(
    at::Tensor& out,
    const at::Tensor& expanded_x,
    const at::Tensor& expanded_row_idx,
    const c10::optional<at::Tensor>& x1 = c10::nullopt,
    const c10::optional<at::Tensor>& x2 = c10::nullopt,
    const c10::optional<at::Tensor>& bias = c10::nullopt,
    const c10::optional<at::Tensor>& scales = c10::nullopt,
    const c10::optional<at::Tensor>& expert_idx = c10::nullopt,
    int64_t drop_pad_mode = 0,
    int64_t k = 1)
{
  TORCH_CHECK(expanded_x.is_contiguous(), "expanded_x is not contiguous");
  TORCH_CHECK(expanded_row_idx.is_contiguous(), "expanded_row_idx is not contiguous");
  TORCH_CHECK(out.is_contiguous(), "out is not contiguous");
  TORCH_CHECK(drop_pad_mode >= 0 && drop_pad_mode <= 3, "dropPadMode must be [0,3]");
  const bool drop_pad = (drop_pad_mode & 1) != 0;
  TORCH_CHECK(expanded_x.dim() == (drop_pad ? 3 : 2),
              "expandedX must be (num_rows*K, H) or (E, C, H)");
  TORCH_CHECK(expanded_row_idx.scalar_type() == at::kInt);
  bool has_x1 = x1.has_value();
  bool has_x2 = x2.has_value();
  bool has_bias = bias.has_value();
  bool has_scales = scales.has_value();
  bool has_expert_idx = expert_idx.has_value();
  // K / num_rows 推导：scales 优先；scales 为空时 K = k（属性）
  const int64_t K = has_scales ? scales->size(1) : k;
  TORCH_CHECK(K >= 1);
  if (has_scales && k > 1) TORCH_CHECK(k == K, "k must match scales.shape[1]");
  const int64_t num_rows = has_scales ? scales->size(0)
                                  : expanded_row_idx.numel() / K;
  const int64_t H = expanded_x.size(-1);
  TORCH_CHECK(num_rows > 0 && H > 0);

  TORCH_CHECK(!x2 || x1, "x2 requires x1");
  TORCH_CHECK(!bias || expert_idx, "bias requires expertIdx");

  TORCH_CHECK(expanded_row_idx.numel() == num_rows * K, "expandedRowIdx must be num_rows*K");
  if (has_scales)
    TORCH_CHECK(scales->dim() == 2 && scales->size(0) == num_rows && scales->size(1) == K);
  if (has_expert_idx)
    TORCH_CHECK(expert_idx->dim() == 2 && expert_idx->size(0) == num_rows && expert_idx->size(1) == K &&
                expert_idx->scalar_type() == at::kInt);
  if (!drop_pad)
    TORCH_CHECK(expanded_x.size(0) == num_rows * K, "expandedX rows must be num_rows*K");
  if (has_x1) TORCH_CHECK(x1->sizes() == (at::IntArrayRef{num_rows, H}));
  if (has_x2) TORCH_CHECK(x2->sizes() == (at::IntArrayRef{num_rows, H}));
  if (has_bias)  TORCH_CHECK(bias->dim() == 2 && bias->size(1) == H);

  const auto sdt = expanded_x.scalar_type();
  TORCH_CHECK(sdt == at::kHalf || sdt == at::kBFloat16 || sdt == at::kFloat || sdt == at::kFloat8_e4m3fn);
  const int dtype = (sdt == at::kFloat) ? MOE_FIN_FP32
                  : (sdt == at::kHalf)  ? MOE_FIN_FP16
                  : (sdt == at::kFloat8_e4m3fn) ? MOE_FIN_FP8 
                                              : MOE_FIN_BF16;
  const bool scales_fp32 = has_scales && scales->scalar_type() == at::kFloat;

  const auto st = moe_finalize_routing_v2(
      expanded_x.data_ptr(), expanded_row_idx.data_ptr(),
      has_x1 ? x1->data_ptr() : nullptr,
      has_x2 ? x2->data_ptr() : nullptr,
      has_bias  ? bias->data_ptr()  : nullptr,
      has_scales ? scales->data_ptr() : nullptr,
      has_expert_idx ? expert_idx->data_ptr() : nullptr,
      out.data_ptr(), num_rows, K, H, drop_pad_mode, dtype,
      scales_fp32 ? 1 : 0, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == MOE_FIN_OK, "moe_finalize_routing_v2: ",
              moe_fin_status_str(st));
}

#undef FP8_E4M3_MAX