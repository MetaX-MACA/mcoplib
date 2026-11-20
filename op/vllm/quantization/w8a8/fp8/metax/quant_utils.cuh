#pragma once

#include "../../../../attention/attention_dtypes.h"
#include <torch/headeronly/core/ScalarType.h>
#include <cuda_fp8.h>
#include <common/maca_fp8.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace vllm {

namespace fp8 {
// Unspecialized conversions are a compile error: the old passthrough
// (`return x;`) silently skipped fp8 encoding for any (Tout, Tin) pair
// without a specialization below (e.g. the torch stable-ABI scalar types),
// corrupting quantized data with no runtime signal.
template <typename>
inline constexpr bool _no_conversion_specialization = false;

template <typename Tout, typename Tin>
__inline__ __device__ Tout vec_conversion(
    const Tin& x, const __nv_fp8_interpretation_t fp8_type = __NV_E4M3) {
  static_assert(_no_conversion_specialization<Tin>,
                "no vec_conversion specialization for this (Tout, Tin) pair");
}

// float -> c10::Float8_e4m3fn
template <>
__inline__ __device__ c10::Float8_e4m3fn
vec_conversion<c10::Float8_e4m3fn, float>(
    const float& a, const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  return static_cast<c10::Float8_e4m3fn>(a);
    #else
  return c10::Float8_e4m3fn(__nv_cvt_float_to_fp8(a, __NV_SATFINITE, fp8_type),
                            c10::Float8_e4m3fn::from_bits());
    #endif
}

/* Scaled and vectorized conversions, for data exchange between high and low
   precision domains Convention of the scale in API, e.g: FP8_data =
   Quantization( High_Precision_data / scale ) s.t. Quantize(HP / scale) => FP8
     Dequant(FP8) * scale =>  HP
 */

template <typename Tout, typename Tin>
__inline__ __device__ Tout scaled_vec_conversion(
    const Tin& x, const float scale, const __nv_fp8_interpretation_t fp8_type) {
  static_assert(
      _no_conversion_specialization<Tin>,
      "no scaled_vec_conversion specialization for this (Tout, Tin) pair");
}

// fp8 byte -> float, decoded in software.
//
// ``__nv_cvt_fp8_to_halfraw`` and ``__nv_cvt_fp8x2_to_halfraw2`` resolve to
// the MACA originals in common/maca_fp8.hpp, whose ``__MACA_ARCH__ >= 890``
// branch feeds the byte's *integer* value into ``__float2half``, shifts that
// 16 bits up and converts the result back to half -- every non-zero code
// overflows to inf. Upstream csrc/quantization/w8a8/fp8/nvidia can use the
// intrinsic because it is a hardware instruction there; decoding by hand is
// what keeps the dequantizing paths correct on this backend (convert_fp8,
// paged attention over an fp8 cache, gather_and_maybe_dequant_cache).
__device__ __forceinline__ float fp8_to_float(
    uint8_t b, const __nv_fp8_interpretation_t fp8_type) {
  bool const e5m2 = (fp8_type == __NV_E5M2);
  int const mant_bits = e5m2 ? 2 : 3;
  int const exp_bits = e5m2 ? 5 : 4;
  int const bias = e5m2 ? 15 : 7;
  int const max_exp = (1 << exp_bits) - 1;
  int const exponent = (b >> mant_bits) & max_exp;
  int const mantissa = b & ((1 << mant_bits) - 1);

  float const kInf = __int_as_float(0x7f800000U);
  float const kNaN = __int_as_float(0x7fffffffU);

  float value;
  if (e5m2 && exponent == max_exp) {
    // E5M2 spends the top exponent on inf (mantissa clear) and NaN.
    value = (mantissa == 0) ? kInf : kNaN;
  } else if (!e5m2 && (b & 0x7F) == 0x7F) {
    // E4M3 has only 0x7F / 0xFF as NaN; its exponent 15 is a normal value
    // (256 .. 448), so it must not be treated like the E5M2 all-ones.
    value = kNaN;
  } else if (exponent == 0) {
    // Subnormal: mantissa * 2**(1 - bias - mant_bits).
    value = static_cast<float>(mantissa) *
            __int_as_float((127 + 1 - bias - mant_bits) << 23);
  } else {
    value = (1.0f + static_cast<float>(mantissa) /
                       static_cast<float>(1 << mant_bits)) *
            __int_as_float((127 + exponent - bias) << 23);
  }
  return (b & 0x80) ? -value : value;
}

// fp8 -> half
template <>
__inline__ __device__ uint16_t scaled_vec_conversion<uint16_t, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  return float_to_half(fp8_to_float(a, fp8_type) * scale);
}

// fp8x2 -> half2
template <>
__inline__ __device__ uint32_t scaled_vec_conversion<uint32_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint16_t u16[2];
    uint32_t u32;
  } tmp;
  tmp.u16[0] = float_to_half(fp8_to_float((uint8_t)a, fp8_type) * scale);
  tmp.u16[1] =
      float_to_half(fp8_to_float((uint8_t)(a >> 8U), fp8_type) * scale);
  return tmp.u32;
}

// fp8x4 -> half2x2
template <>
__inline__ __device__ uint2 scaled_vec_conversion<uint2, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint2 u32x2;
    uint32_t u32[2];
  } tmp;
  tmp.u32[0] =
      scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)a, scale, fp8_type);
  tmp.u32[1] = scaled_vec_conversion<uint32_t, uint16_t>((uint16_t)(a >> 16U),
                                                         scale, fp8_type);
  return tmp.u32x2;
}

// fp8x8 -> half2x4
template <>
__inline__ __device__ uint4
scaled_vec_conversion<uint4, uint2>(const uint2& a, const float scale,
                                    const __nv_fp8_interpretation_t fp8_type) {
  union {
    uint4 u64x2;
    uint2 u64[2];
  } tmp;
  tmp.u64[0] = scaled_vec_conversion<uint2, uint32_t>(a.x, scale, fp8_type);
  tmp.u64[1] = scaled_vec_conversion<uint2, uint32_t>(a.y, scale, fp8_type);
  return tmp.u64x2;
}

// fp8 -> __nv_bfloat16
template <>
__inline__ __device__ __nv_bfloat16
scaled_vec_conversion<__nv_bfloat16, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // Note there is no direct convert function from fp8 to bf16.
  return __float2bfloat16(fp8_to_float(a, fp8_type) * scale);
}

// fp8x2 -> __nv_bfloat162
template <>
__inline__ __device__ __nv_bfloat162
scaled_vec_conversion<__nv_bfloat162, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_bfloat162 res;
  res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale,
                                                        fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U),
                                                        scale, fp8_type);
  return res;
}

// fp8x4 -> bf16_4_t
template <>
__inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t res;
  res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale,
                                                          fp8_type);
  res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U),
                                                          scale, fp8_type);
  return res;
}

// fp8x8 -> bf16_8_t
template <>
__inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  bf16_4_t tmp1, tmp2;
  tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale, fp8_type);
  bf16_8_t res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// fp8 -> float
template <>
__inline__ __device__ float scaled_vec_conversion<float, uint8_t>(
    const uint8_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  return fp8_to_float(a, fp8_type) * scale;
}

// fp8x2 -> float2
template <>
__inline__ __device__ float2 scaled_vec_conversion<float2, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  // fp8x2 -> half2
  uint32_t tmp = scaled_vec_conversion<uint32_t, uint16_t>(a, scale, fp8_type);
  // half2 -> float2
  return half2_to_float2(tmp);
}

// fp8x4 -> float4
template <>
__inline__ __device__ Float4_ scaled_vec_conversion<Float4_, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ res;
  res.x = scaled_vec_conversion<float2, uint16_t>((uint16_t)a, scale, fp8_type);
  res.y = scaled_vec_conversion<float2, uint16_t>((uint16_t)(a >> 16U), scale,
                                                  fp8_type);
  return res;
}

// fp8x8 -> float8
template <>
__inline__ __device__ Float8_ scaled_vec_conversion<Float8_, uint2>(
    const uint2& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp1, tmp2;
  tmp1 = scaled_vec_conversion<Float4_, uint32_t>(a.x, scale, fp8_type);
  tmp2 = scaled_vec_conversion<Float4_, uint32_t>(a.y, scale, fp8_type);
  Float8_ res;
  res.x = tmp1.x;
  res.y = tmp1.y;
  res.z = tmp2.x;
  res.w = tmp2.y;
  return res;
}

// half -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, uint16_t>(
    const uint16_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(half_to_float(a) / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// bf16 -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, __nv_bfloat16>(
    const __nv_bfloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
  assert(false);
    #else
  __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(__bfloat162float(a) / scale,
                                                 __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
    #endif
  __builtin_unreachable();  // Suppress missing return statement warning
}

// torch stable-ABI (headeronly) scalar types delegate to the CUDA-native
// conversions, so stable-ABI kernels dispatched on c10::BFloat16 /
// c10::Half quantize correctly without manual casts.
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, c10::BFloat16>(
    const c10::BFloat16& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  return scaled_vec_conversion<uint8_t, __nv_bfloat16>(
      reinterpret_cast<const __nv_bfloat16&>(a), scale, fp8_type);
}

template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, c10::Half>(
    const c10::Half& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  return scaled_vec_conversion<uint8_t, uint16_t>(
      reinterpret_cast<const uint16_t&>(a), scale, fp8_type);
}

// float -> fp8
template <>
__inline__ __device__ uint8_t scaled_vec_conversion<uint8_t, float>(
    const float& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  __nv_fp8_storage_t res =
      __nv_cvt_float_to_fp8(a / scale, __NV_SATFINITE, fp8_type);
  return (uint8_t)res;
}

// fp8x4 -> float4
template <>
__inline__ __device__ float4 scaled_vec_conversion<float4, uint32_t>(
    const uint32_t& a, const float scale,
    const __nv_fp8_interpretation_t fp8_type) {
  Float4_ tmp = scaled_vec_conversion<Float4_, uint32_t>(a, scale, fp8_type);
  float4 res = make_float4(tmp.x.x, tmp.x.y, tmp.y.x, tmp.y.y);
  return res;
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E4M3);
  } else if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E5M2) {
    return scaled_vec_conversion<Tout, Tin>(x, scale, __NV_E5M2);
  }
}

  // The following macro is used to dispatch the conversion function based on
  // the data type of the key and value cache. The FN is a macro that calls a
  // function with template<typename scalar_t, typename cache_t,
  // Fp8KVCacheDataType kv_dt>.
  #define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                  \
    vllm::Fp8KVCacheDataType KV_CACHE_DTYPE =                                  \
        vllm::get_fp8_kv_cache_data_type(KV_DTYPE);                            \
    if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {                   \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, float, vllm::Fp8KVCacheDataType::kAuto);                     \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);               \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);     \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kFp8E4M3) {         \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);                \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);             \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);        \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else if (KV_CACHE_DTYPE == vllm::Fp8KVCacheDataType::kFp8E5M2) {         \
      if (SRC_DTYPE == torch::headeronly::ScalarType::Float) {                 \
        FN(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);                \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::Half) {           \
        FN(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);             \
      } else if (SRC_DTYPE == torch::headeronly::ScalarType::BFloat16) {       \
        FN(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kFp8E5M2);        \
      } else {                                                                 \
        STD_TORCH_CHECK(false,                                                 \
                        "Unsupported input type of kv cache: ", SRC_DTYPE);    \
      }                                                                        \
    } else {                                                                   \
      STD_TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE); \
    }

    template <typename Tout, typename Tin>
    __inline__ __device__ Tout scaled_vec_conversion(const Tin& x, const float scale) {
      return x;
    }

    template <>
    __inline__ __device__ __nv_bfloat16 scaled_vec_conversion<__nv_bfloat16, uint8_t>(const uint8_t& a, const float scale) {
      __half_raw res = __maca_cvt_fp8_to_halfraw(a, __MACA_E4M3);
      float tmp = half_to_float(res.x);
      return __float2bfloat16(tmp * scale);
    }

    template <>
    __inline__ __device__ __nv_bfloat162 scaled_vec_conversion<__nv_bfloat162, uint16_t>(const uint16_t& a, const float scale) {
      __nv_bfloat162 res;
      res.x = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)a, scale);
      res.y = scaled_vec_conversion<__nv_bfloat16, uint8_t>((uint8_t)(a >> 8U), scale);
      return res;
    }

    template <>
    __inline__ __device__ bf16_4_t scaled_vec_conversion<bf16_4_t, uint32_t>(const uint32_t& a, const float scale) {
      bf16_4_t res;
      res.x = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)a, scale);
      res.y = scaled_vec_conversion<__nv_bfloat162, uint16_t>((uint16_t)(a >> 16U), scale);
      return res;
    }

    template <>
    __inline__ __device__ bf16_8_t scaled_vec_conversion<bf16_8_t, uint2>(const uint2& a, const float scale) {
      bf16_4_t tmp1, tmp2;
      tmp1 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.x, scale);
      tmp2 = scaled_vec_conversion<bf16_4_t, uint32_t>(a.y, scale);
      bf16_8_t res;
      res.x = tmp1.x;
      res.y = tmp1.y;
      res.z = tmp2.x;
      res.w = tmp2.y;
      return res;
    }
}  // namespace fp8
}  // namespace vllm
