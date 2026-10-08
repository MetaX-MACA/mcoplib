#pragma once

#include "../../../../attention/attention_dtypes.h"
#include <torch/headeronly/core/ScalarType.h>
#include <common/maca_fp8.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <type_traits>

namespace vllm {

namespace fp8 {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout convert(const Tin& x) {
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
}

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert(const Tin& x, const float scale) {
  assert(false);
  __builtin_unreachable();  // Suppress missing return statement warning
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
