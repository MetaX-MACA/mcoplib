#include "moe_init_routing_v4_mx4.h"

#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cmath>

namespace {

// Keep this TU-local encoder identical to retained Q29. Moving the old
// definition to a shared header would also change the retained device TU.
__device__ __forceinline__ uint8_t e2m1_bf16_rne(c10::BFloat16 value) {
  uint32_t bits = value.x;
  uint32_t magnitude = bits & 0x7fffU;
  uint32_t sign = (bits >> 12) & 8U;
  uint32_t subnormal = static_cast<uint32_t>(magnitude > 0x3e80U) +
                       static_cast<uint32_t>(magnitude >= 0x3f40U);
  uint32_t normal = (magnitude + 0x1fU + ((magnitude >> 6) & 1U)
                     - 0x3f00U) >> 6;
  // Saturation also maps infinities and NaNs to magnitude code 7, as e2m1.
  normal = normal < 7U ? normal : 7U;
  return sign | (magnitude < 0x3f80U ? subnormal : normal);
}

template <typename T>
struct alignas(16) Fp8Values8 {
  T v0, v1, v2, v3, v4, v5, v6, v7;
};

// Q41: one lane owns a complete32-value MXFP4 group and returns16 packed
// bytes. Preserve BF16-truncated amax, E8M0 and the Q29 exact E2M1 encoder.
__device__ __forceinline__ uint4 quantize_mx4_fragment32(
    const c10::BFloat16* x, uint8_t& scale) {
  Fp8Values8<c10::BFloat16> first = *reinterpret_cast<const Fp8Values8<c10::BFloat16>*>(x);
  Fp8Values8<c10::BFloat16> second = *reinterpret_cast<const Fp8Values8<c10::BFloat16>*>(x + 8);
  Fp8Values8<c10::BFloat16> third = *reinterpret_cast<const Fp8Values8<c10::BFloat16>*>(x + 16);
  Fp8Values8<c10::BFloat16> fourth = *reinterpret_cast<const Fp8Values8<c10::BFloat16>*>(x + 24);
  float value0 = static_cast<float>(first.v0);
  float value1 = static_cast<float>(first.v1);
  float value2 = static_cast<float>(first.v2);
  float value3 = static_cast<float>(first.v3);
  float value4 = static_cast<float>(first.v4);
  float value5 = static_cast<float>(first.v5);
  float value6 = static_cast<float>(first.v6);
  float value7 = static_cast<float>(first.v7);
  float value8 = static_cast<float>(second.v0);
  float value9 = static_cast<float>(second.v1);
  float value10 = static_cast<float>(second.v2);
  float value11 = static_cast<float>(second.v3);
  float value12 = static_cast<float>(second.v4);
  float value13 = static_cast<float>(second.v5);
  float value14 = static_cast<float>(second.v6);
  float value15 = static_cast<float>(second.v7);
  float value16 = static_cast<float>(third.v0);
  float value17 = static_cast<float>(third.v1);
  float value18 = static_cast<float>(third.v2);
  float value19 = static_cast<float>(third.v3);
  float value20 = static_cast<float>(third.v4);
  float value21 = static_cast<float>(third.v5);
  float value22 = static_cast<float>(third.v6);
  float value23 = static_cast<float>(third.v7);
  float value24 = static_cast<float>(fourth.v0);
  float value25 = static_cast<float>(fourth.v1);
  float value26 = static_cast<float>(fourth.v2);
  float value27 = static_cast<float>(fourth.v3);
  float value28 = static_cast<float>(fourth.v4);
  float value29 = static_cast<float>(fourth.v5);
  float value30 = static_cast<float>(fourth.v6);
  float value31 = static_cast<float>(fourth.v7);
  uint32_t amax = __float_as_uint(value0) & 0x7fff0000U;
  uint32_t bits1 = __float_as_uint(value1) & 0x7fff0000U;
  amax = amax > bits1 ? amax : bits1;
  uint32_t bits2 = __float_as_uint(value2) & 0x7fff0000U;
  amax = amax > bits2 ? amax : bits2;
  uint32_t bits3 = __float_as_uint(value3) & 0x7fff0000U;
  amax = amax > bits3 ? amax : bits3;
  uint32_t bits4 = __float_as_uint(value4) & 0x7fff0000U;
  amax = amax > bits4 ? amax : bits4;
  uint32_t bits5 = __float_as_uint(value5) & 0x7fff0000U;
  amax = amax > bits5 ? amax : bits5;
  uint32_t bits6 = __float_as_uint(value6) & 0x7fff0000U;
  amax = amax > bits6 ? amax : bits6;
  uint32_t bits7 = __float_as_uint(value7) & 0x7fff0000U;
  amax = amax > bits7 ? amax : bits7;
  uint32_t bits8 = __float_as_uint(value8) & 0x7fff0000U;
  amax = amax > bits8 ? amax : bits8;
  uint32_t bits9 = __float_as_uint(value9) & 0x7fff0000U;
  amax = amax > bits9 ? amax : bits9;
  uint32_t bits10 = __float_as_uint(value10) & 0x7fff0000U;
  amax = amax > bits10 ? amax : bits10;
  uint32_t bits11 = __float_as_uint(value11) & 0x7fff0000U;
  amax = amax > bits11 ? amax : bits11;
  uint32_t bits12 = __float_as_uint(value12) & 0x7fff0000U;
  amax = amax > bits12 ? amax : bits12;
  uint32_t bits13 = __float_as_uint(value13) & 0x7fff0000U;
  amax = amax > bits13 ? amax : bits13;
  uint32_t bits14 = __float_as_uint(value14) & 0x7fff0000U;
  amax = amax > bits14 ? amax : bits14;
  uint32_t bits15 = __float_as_uint(value15) & 0x7fff0000U;
  amax = amax > bits15 ? amax : bits15;
  uint32_t bits16 = __float_as_uint(value16) & 0x7fff0000U;
  amax = amax > bits16 ? amax : bits16;
  uint32_t bits17 = __float_as_uint(value17) & 0x7fff0000U;
  amax = amax > bits17 ? amax : bits17;
  uint32_t bits18 = __float_as_uint(value18) & 0x7fff0000U;
  amax = amax > bits18 ? amax : bits18;
  uint32_t bits19 = __float_as_uint(value19) & 0x7fff0000U;
  amax = amax > bits19 ? amax : bits19;
  uint32_t bits20 = __float_as_uint(value20) & 0x7fff0000U;
  amax = amax > bits20 ? amax : bits20;
  uint32_t bits21 = __float_as_uint(value21) & 0x7fff0000U;
  amax = amax > bits21 ? amax : bits21;
  uint32_t bits22 = __float_as_uint(value22) & 0x7fff0000U;
  amax = amax > bits22 ? amax : bits22;
  uint32_t bits23 = __float_as_uint(value23) & 0x7fff0000U;
  amax = amax > bits23 ? amax : bits23;
  uint32_t bits24 = __float_as_uint(value24) & 0x7fff0000U;
  amax = amax > bits24 ? amax : bits24;
  uint32_t bits25 = __float_as_uint(value25) & 0x7fff0000U;
  amax = amax > bits25 ? amax : bits25;
  uint32_t bits26 = __float_as_uint(value26) & 0x7fff0000U;
  amax = amax > bits26 ? amax : bits26;
  uint32_t bits27 = __float_as_uint(value27) & 0x7fff0000U;
  amax = amax > bits27 ? amax : bits27;
  uint32_t bits28 = __float_as_uint(value28) & 0x7fff0000U;
  amax = amax > bits28 ? amax : bits28;
  uint32_t bits29 = __float_as_uint(value29) & 0x7fff0000U;
  amax = amax > bits29 ? amax : bits29;
  uint32_t bits30 = __float_as_uint(value30) & 0x7fff0000U;
  amax = amax > bits30 ? amax : bits30;
  uint32_t bits31 = __float_as_uint(value31) & 0x7fff0000U;
  amax = amax > bits31 ? amax : bits31;
  int max_exp = (amax >> 23) & 255;
  int code = max(max_exp - 2, 0);
  if (max_exp == 255) code = 255;
  scale = static_cast<uint8_t>(code);
  float inverse = code == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, 127 - code);
  uint32_t packed0 = static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value0 * inverse)));
  packed0 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value1 * inverse))) << 4;
  packed0 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value2 * inverse))) << 8;
  packed0 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value3 * inverse))) << 12;
  packed0 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value4 * inverse))) << 16;
  packed0 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value5 * inverse))) << 20;
  packed0 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value6 * inverse))) << 24;
  packed0 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value7 * inverse))) << 28;
  uint32_t packed1 = static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value8 * inverse)));
  packed1 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value9 * inverse))) << 4;
  packed1 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value10 * inverse))) << 8;
  packed1 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value11 * inverse))) << 12;
  packed1 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value12 * inverse))) << 16;
  packed1 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value13 * inverse))) << 20;
  packed1 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value14 * inverse))) << 24;
  packed1 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value15 * inverse))) << 28;
  uint32_t packed2 = static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value16 * inverse)));
  packed2 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value17 * inverse))) << 4;
  packed2 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value18 * inverse))) << 8;
  packed2 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value19 * inverse))) << 12;
  packed2 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value20 * inverse))) << 16;
  packed2 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value21 * inverse))) << 20;
  packed2 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value22 * inverse))) << 24;
  packed2 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value23 * inverse))) << 28;
  uint32_t packed3 = static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value24 * inverse)));
  packed3 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value25 * inverse))) << 4;
  packed3 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value26 * inverse))) << 8;
  packed3 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value27 * inverse))) << 12;
  packed3 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value28 * inverse))) << 16;
  packed3 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value29 * inverse))) << 20;
  packed3 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value30 * inverse))) << 24;
  packed3 |= static_cast<uint32_t>(e2m1_bf16_rne(c10::BFloat16(value31 * inverse))) << 28;
  return uint4{packed0, packed1, packed2, packed3};
}

// Quantize each4096-input tile once, then reuse its two encoded fragments
// and two group scales per lane across topk. No temporary quant buffer access.
__global__ void quantize_scatter_mx4_tokens2(
    const c10::BFloat16* x, uint4* dst, uint8_t* scale_dst, const int* inverse,
    int64_t tokens, int topk, int hidden, int capacity) {
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      const int* ranks = inverse + token * topk;
      const auto* in = x + token * hidden;
      for (int fragment = 0; fragment < hidden; fragment += 4096) {
        uint8_t scale0, scale1;
        uint4 value0 = quantize_mx4_fragment32(in + fragment + lane * 32, scale0);
        uint4 value1 = quantize_mx4_fragment32(in + fragment + 2048 + lane * 32, scale1);
        int group = fragment / 32 + lane;
        for (int k = 0; k < topk; ++k) {
          int row = ranks[k];
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * (hidden / 32) + fragment / 32 + lane;
            __stcs(out, value0);
            __stcs(out + 64, value1);
            auto* scales = scale_dst + static_cast<int64_t>(row) * (hidden / 32) + group;
            scales[0] = scale0;
            scales[64] = scale1;
          }
        }
      }
    }
  }
}

// Q48: cache up to64 inverse ranks only for the two-fragment H8192 path.
__global__ void quantize_scatter_mx4_rankwave_tokens2(
    const c10::BFloat16* x, uint4* dst, uint8_t* scale_dst, const int* inverse,
    int64_t tokens, int topk, int hidden, int capacity) {
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      const int* ranks = inverse + token * topk;
      // One coalesced rank load per token; lane>=topk never reads inverse.
      int cached_rank = -1;
      if (lane < topk) cached_rank = ranks[lane];
      const auto* in = x + token * hidden;
      for (int fragment = 0; fragment < hidden; fragment += 4096) {
        uint8_t scale0, scale1;
        uint4 value0 = quantize_mx4_fragment32(in + fragment + lane * 32, scale0);
        uint4 value1 = quantize_mx4_fragment32(in + fragment + 2048 + lane * 32, scale1);
        int group = fragment / 32 + lane;
        for (int k = 0; k < topk; ++k) {
          // The valid token occupies a full64-lane wave, including K<64.
          int row = __shfl_sync(0xffffffffffffffffULL, cached_rank, k, 64);
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * (hidden / 32) + fragment / 32 + lane;
            __stcs(out, value0);
            __stcs(out + 64, value1);
            auto* scales = scale_dst + static_cast<int64_t>(row) * (hidden / 32) + group;
            scales[0] = scale0;
            scales[64] = scale1;
          }
        }
      }
    }
  }
}

}  // namespace

namespace mcoplib::moe_routing_detail {
void launch_mx4_quantize_scatter(
    const c10::BFloat16* x, void* data, uint8_t* scales, const int* inverse,
    int64_t tokens, int topk, int hidden, int capacity, cudaStream_t stream) {
  int grid = std::min<int64_t>((tokens + 1) / 2, 65535);
  if (hidden == 8192 && topk > 0 && topk <= 64) {
    quantize_scatter_mx4_rankwave_tokens2<<<grid, 128, 0, stream>>>(
        x, static_cast<uint4*>(data), scales, inverse, tokens, topk, hidden, capacity);
  } else {
    quantize_scatter_mx4_tokens2<<<grid, 128, 0, stream>>>(
        x, static_cast<uint4*>(data), scales, inverse, tokens, topk, hidden, capacity);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace mcoplib::moe_routing_detail
