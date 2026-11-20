#include "../include/moe_init_routing_v4.h"
#include "moe_init_routing_v4_mx4.h"

#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <maca_fp8.h>
#include <algorithm>
#include <climits>
#include <cstring>
#include <type_traits>

namespace {

constexpr int kThreads = 256;
constexpr int64_t kAlignment = 256;
constexpr int64_t kInverseSortMinRoutes = 1 << 19;
constexpr int64_t kInverseSortMaxRoutes = 81920 * 30;
int64_t aligned(int64_t n) { return (n + kAlignment - 1) / kAlignment * kAlignment; }

bool low_precision(at::ScalarType t) {
  return t == at::ScalarType::Float8_e4m3fn ||
         t == at::ScalarType::Float8_e5m2 ||
         t == at::ScalarType::Float4_e2m1fn_x2;
}

void tensor_check(const at::Tensor& t, const at::Tensor& x, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.device() == x.device(), name, " must be on x's GPU");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(!t.requires_grad(), name, ": autograd is not implemented");
}

struct Layout {
  int64_t tokens, routes, columns, hidden, capacity, row_bytes, scale_bytes;
  int start, end;
  at::ScalarType dtype, scale_dtype;
  std::vector<int64_t> data_shape, count_shape, scale_shape, weight_shape;
};

Layout layout(const at::Tensor& x, const at::Tensor& ids, int64_t experts,
              int64_t mode, int64_t row_type, int64_t count_type, bool count_flag,
              int64_t active, const std::vector<int64_t>& range,
              const std::optional<at::Tensor>& scale,
              const std::optional<at::Tensor>& weight) {
  TORCH_CHECK(x.is_cuda(), "x must be a GPU tensor");
  tensor_check(x, x, "x");
  tensor_check(ids, x, "expert_idx");
  TORCH_CHECK(x.dim() == 2 && ids.dim() == 2 && x.size(0) == ids.size(0),
              "x and expert_idx must be matrices with equal token counts");
  TORCH_CHECK(ids.scalar_type() == at::kInt, "expert_idx must be int32");
  TORCH_CHECK(experts > 0 && experts <= 10240, "expert_num must be in [1, 10240]");
  TORCH_CHECK(ids.numel() <= INT_MAX, "number of routes exceeds int32 indexing");
  TORCH_CHECK(row_type == 0 || row_type == 1, "row_idx_type must be 0 or 1");
  TORCH_CHECK(count_type == 0 || count_type == 1,
              "expert_tokens_num_type must be 0 (cumsum) or 1 (count); key_value is not implemented");
  TORCH_CHECK(active >= 0, "active_num must be nonnegative; 0 means all routes");
  TORCH_CHECK(range.empty() || range.size() == 2, "active_expert_range must have two entries");
  int start = range.empty() ? 0 : range[0];
  int end = range.empty() ? experts : range[1];
  TORCH_CHECK(range.empty() || (range[0] >= 0 && range[0] <= range[1] && range[1] <= experts),
              "active_expert_range must satisfy 0 <= start <= end <= expert_num");
  TORCH_CHECK(mode == -1 || mode == 2 || mode == 3 || mode == 4 || mode == 5 || mode == 9,
              "supported quant_mode values are -1, 2, 3, 4, 5, 9");
  auto dtype = x.scalar_type();
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf || dtype == at::kFloat ||
              dtype == at::kChar || low_precision(dtype), "unsupported x dtype");
  int64_t hidden = x.size(1) * (dtype == at::ScalarType::Float4_e2m1fn_x2 ? 2 : 1);
  TORCH_CHECK(hidden <= INT_MAX, "hidden size exceeds int32 indexing");
  int64_t cap = active == 0 ? ids.numel() : std::min(active, ids.numel());
  Layout l{x.size(0), ids.numel(), x.size(1), hidden, cap,
           x.size(1) * static_cast<int64_t>(x.element_size()), 0,
           start, end, dtype, at::kFloat,
           {cap, x.size(1)}, {count_flag ? end - start : 0}, {0}, {0}};
  if (mode != -1) {
    TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kHalf,
                "quantization requires bfloat16 or float16 input");
    TORCH_CHECK(!scale.has_value(), "these quantization modes do not accept input scale");
    TORCH_CHECK(mode != 9 || hidden % 2 == 0, "MXFP4 requires even hidden size");
    l.dtype = mode == 9 ? at::ScalarType::Float4_e2m1fn_x2 :
        (mode == 2 || mode == 4 ? at::ScalarType::Float8_e5m2 : at::ScalarType::Float8_e4m3fn);
    l.columns = mode == 9 ? hidden / 2 : hidden;
    l.row_bytes = l.columns;
    l.data_shape = {cap, l.columns};
    if (mode == 4 || mode == 5) {
      l.scale_shape = {cap, (hidden + 127) / 128};
      l.scale_bytes = (hidden + 127) / 128 * sizeof(float);
    } else {
      l.scale_dtype = at::ScalarType::Float8_e8m0fnu;
      l.scale_bytes = (hidden + 63) / 64 * 2;
      l.scale_shape = mode == 9 ? std::vector<int64_t>{cap, (hidden + 63) / 64, 2} :
                                 std::vector<int64_t>{cap, l.scale_bytes};
    }
  }
  if (scale && mode == -1) {
    tensor_check(*scale, x, "scale");
    if (low_precision(dtype)) {
      TORCH_CHECK(scale->scalar_type() == at::ScalarType::Float8_e8m0fnu,
                  "low precision passthrough scale must be float8_e8m0fnu");
      TORCH_CHECK(scale->sizes().vec() == std::vector<int64_t>({l.tokens, (hidden + 63) / 64, 2}),
                  "low precision scale must have shape [tokens, ceil(hidden/64), 2]");
      l.scale_dtype = scale->scalar_type();
      l.scale_shape = {cap, (hidden + 63) / 64, 2};
      l.scale_bytes = (hidden + 63) / 64 * 2;
    } else {
      TORCH_CHECK(scale->scalar_type() == at::kFloat &&
                  scale->sizes().vec() == std::vector<int64_t>({l.tokens}),
                  "passthrough scale must be float32 [tokens]");
      l.scale_shape = {cap};
      l.scale_bytes = sizeof(float);
    }
  }
  if (weight) {
    tensor_check(*weight, x, "topk_weight");
    TORCH_CHECK(weight->scalar_type() == at::kFloat && weight->sizes() == ids.sizes(),
                "topk_weight must be float32 with the shape of expert_idx");
    l.weight_shape = {cap, 1};
  }
  return l;
}

// Only metadata is inspected here: never convert a device scalar to a host
// integer on the graph-safe out path.
void check_active_tensor(const at::Tensor& active, const at::Tensor& x) {
  tensor_check(active, x, "active_num_optional");
  TORCH_CHECK(active.scalar_type() == at::kLong &&
              (active.dim() == 0 || (active.dim() == 1 && active.size(0) == 1)),
              "active_num_optional must be an INT64 scalar or [1] tensor");
}

void set_capacity(Layout& l, int64_t capacity, int64_t mode,
                  bool has_scale, bool has_weight) {
  l.capacity = capacity;
  l.data_shape[0] = capacity;
  if (mode != -1 || has_scale) l.scale_shape[0] = capacity;
  if (has_weight) l.weight_shape[0] = capacity;
}

struct Scratch {
  int64_t array_bytes, bounds_offset, sort_offset, quant_offset, scale_offset, bytes;
  size_t sort_bytes;
  bool inverse_sort;
};

// layout() validates 1 <= experts <= 10240. make_keys() emits only values
// in [0, experts], including the invalid-route sentinel experts itself. Include
// that sentinel even when experts is a power of two; preserve stable tie order.
int routing_sort_end_bit(int64_t experts) {
  return 32 - __builtin_clz(static_cast<unsigned>(experts));
}

bool use_inverse_sort(const at::Tensor& x, const Layout& l) {
#ifdef USE_MACA
  if (l.routes >= kInverseSortMinRoutes && l.routes <= kInverseSortMaxRoutes) {
    const auto* props = at::cuda::getDeviceProperties(x.get_device());
    return props->warpSize == 64 && std::strcmp(props->name, "MetaX C600-U") == 0;
  }
#endif
  return false;
}

// Called only in the nonempty large-route branch. Original IDs form [0,N),
// so use bit_width(N-1), independently of the expert sentinel's key width.
int inverse_sort_end_bit(int64_t routes) {
  return 32 - __builtin_clz(static_cast<unsigned>(routes - 1));
}

Scratch scratch(const Layout& l, int64_t experts, int64_t mode, cudaStream_t stream,
                bool inverse_sort) {
  int64_t routes = l.routes;
  size_t sort_bytes = 0;
  if (routes) {
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, sort_bytes, static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        static_cast<int>(routes), 0, routing_sort_end_bit(experts), stream));
  }
  if (inverse_sort) {
    size_t inverse_bytes = 0;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(
        nullptr, inverse_bytes, static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr),
        static_cast<int>(routes), 0, inverse_sort_end_bit(routes), stream));
    sort_bytes = std::max(sort_bytes, inverse_bytes);
  }
  int64_t array_bytes = aligned(routes * sizeof(int));
  int64_t bounds = 4 * array_bytes;
  int64_t sort = bounds + aligned((experts + 1) * sizeof(int64_t));
  int64_t quant = sort + aligned(sort_bytes);
  int64_t scale = quant + (mode == -1 ? 0 : aligned(l.tokens * l.row_bytes));
  int64_t bytes = scale + (mode == -1 ? 0 : aligned(l.tokens * l.scale_bytes));
  return {array_bytes, bounds, sort, quant, scale, bytes, sort_bytes, inverse_sort};
}

// E2M1 positive levels: 0, .5, 1, 1.5, 2, 3, 4, 6; RNE at each midpoint.
__device__ uint8_t e2m1(float x) {
  float a = fabsf(x);
  int code = a > 0.25f;
  if (a >= 0.75f) code = 2;
  if (a > 1.25f) code = 3;
  if (a >= 1.75f) code = 4;
  if (a > 2.5f) code = 5;
  if (a >= 3.5f) code = 6;
  if (a > 5.0f || isnan(a)) code = 7;
  return code | (signbit(x) ? 8 : 0);
}

// Q29: encode an already rounded BF16 product without expanding it to FP32.
// Below 1, E2M1 has half-unit spacing. Normal values retain one fraction bit;
// adding half-minus-one plus the retained LSB implements ties-to-even.
__device__ __forceinline__ uint8_t e2m1_bf16_rne(at::BFloat16 value) {
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

// Quantize once per input token, then gather the encoded row K times. This is
// valid for these expert-independent quantizers and avoids K repeated reductions.
template <typename T>
__global__ void quantize_groups(const T* x, uint8_t* out, uint8_t* scales,
                                int64_t total_groups, int hidden, int groups,
                                int scale_stride, int mode) {
  using Reduce = cub::BlockReduce<uint32_t, 128>;
  __shared__ typename Reduce::TempStorage reduction;
  __shared__ float inverse;
  __shared__ float packed_values[32];
  int group_size = mode == 4 || mode == 5 ? 128 : 32;
  for (int64_t job = blockIdx.x; job < total_groups; job += gridDim.x) {
    int64_t token = job / groups;
    int group = job % groups;
    int col = group * group_size + threadIdx.x;
    bool valid = threadIdx.x < group_size && col < hidden;
    float value = valid ? static_cast<float>(x[token * hidden + col]) : 0.0f;
    uint32_t bits = __float_as_uint(value) & 0x7fffffffU;
    // MX quantizers derive their scale from BF16 (FP16 is truncated to BF16).
    if (mode != 4 && mode != 5) bits &= 0xffff0000U;
    uint32_t amax = Reduce(reduction).Reduce(bits, cub::Max());
    if (threadIdx.x == 0) {
      int max_exp = (amax >> 23) & 255;
      if (mode == 4 || mode == 5) {
        int emax = mode == 4 ? 15 : 8;
        int code = max_exp - emax + ((amax & 0x7fffffU) > 0x600000U);
        if (code <= 0 || code > 254) code = 1;
        float scale = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float(code << 23);
        inverse = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float((254 - code) << 23);
        reinterpret_cast<float*>(scales + token * scale_stride)[group] = scale;
      } else {
        int emax = mode == 9 ? 2 : (mode == 2 ? 15 : 8);
        int code;
        if (mode == 9) {
          code = max(max_exp - emax, 0);  // MXFP4 uses exponent-only scale.
        } else {
          code = max_exp <= emax ? 1 : max_exp - emax + ((amax & 0x7fffffU) > 0x600000U);
          if (amax == 0) code = 0;
        }
        if (max_exp == 255) code = 255;
        scales[token * scale_stride + group] = static_cast<uint8_t>(code);
        inverse = code == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, 127 - code);
        if (amax == 0 && mode != 9) inverse = 0.0f;
      }
    }
    __syncthreads();
    if (mode == 9) {
      if (threadIdx.x < 32)
        packed_values[threadIdx.x] = static_cast<float>(at::BFloat16(value * inverse));
      __syncthreads();
      if (valid && threadIdx.x % 2 == 0) {
        out[token * (hidden / 2) + col / 2] = e2m1(packed_values[threadIdx.x]) |
            (e2m1(packed_values[threadIdx.x + 1]) << 4);
      }
    } else if (valid) {
      float scaled = value * inverse;
      // BF16 MXFP8 multiplication rounds in BF16 before conversion to FP8.
      if (mode == 2 || mode == 3) {
        if constexpr (std::is_same<T, at::BFloat16>::value)
          scaled = static_cast<float>(at::BFloat16(scaled));
      }
      out[token * hidden + col] = __maca_cvt_float_to_fp8(scaled, __MACA_SATFINITE,
          mode == 2 || mode == 4 ? __MACA_E5M2 : __MACA_E4M3);
    }
    __syncthreads();
  }
}

// Exact FP32 -> E4M3 SATFINITE, round-to-nearest-even. Unlike the SDK's
// general FP64 path, this works directly on FP32 bits. NaNs follow the float
// wrapper's canonical positive 0x7f; infinities saturate and signed zero stays.
__device__ __forceinline__ uint8_t e4m3_satfinite_rne(float value) {
  uint32_t bits = __float_as_uint(value);
  uint32_t magnitude = bits & 0x7fffffffU;
  uint32_t sign = (bits >> 24) & 0x80U;
  if (magnitude > 0x7f800000U) return 0x7fU;
  if (magnitude >= 0x43e00000U) return sign | 0x7eU;  // |value| >= 448
  if (magnitude <= 0x3a800000U) return sign;  // <= half the minimum subnormal
  uint32_t encoded;
  if (magnitude >= 0x3c800000U) {  // normal E4M3: keep three mantissa bits
    encoded = (magnitude + 0x7ffffU + ((magnitude >> 20) & 1U)
               - 0x3c000000U) >> 20;
  } else {
    // A subnormal is an integer multiple of 2^-9. The guards bound shift
    // to [21,24], so every shift and rounding addition is defined in uint32.
    uint32_t shift = 141U - (magnitude >> 23);
    uint32_t significand = (magnitude & 0x7fffffU) | 0x800000U;
    encoded = (significand + ((1U << (shift - 1)) - 1U)
               + ((significand >> shift) & 1U)) >> shift;
  }
  return sign | encoded;
}

// C600-U mode 5: one physical 64-lane wave owns a 128-element group.
// Each lane keeps two values; two independent groups share a 128-thread CTA.
template <typename T>
__global__ void quantize_fp8_warp_groups(const T* x, uint8_t* out,
                                        uint8_t* scales, int64_t total_groups,
                                        int hidden, int groups, int scale_stride) {
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 63;
  int wave = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t job = tile + wave;
    int64_t token = job / groups;
    int group = job % groups;
    int col0 = group * 128 + lane;
    int col1 = col0 + 64;
    bool valid_job = job < total_groups;
    bool valid0 = valid_job && col0 < hidden;
    bool valid1 = valid_job && col1 < hidden;
    float value0 = valid0 ? static_cast<float>(x[token * hidden + col0]) : 0.0f;
    float value1 = valid1 ? static_cast<float>(x[token * hidden + col1]) : 0.0f;
    uint32_t bits0 = __float_as_uint(value0) & 0x7fffffffU;
    uint32_t bits1 = __float_as_uint(value1) & 0x7fffffffU;
    uint32_t amax = bits0 > bits1 ? bits0 : bits1;
#pragma unroll
    for (int offset = 32; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 64);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code = max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
      if (code <= 0 || code > 254) code = 1;
      float scale = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float(code << 23);
      inverse = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float((254 - code) << 23);
      if (valid_job)
        reinterpret_cast<float*>(scales + token * scale_stride)[group] = scale;
    }
    // Padding waves still execute every exchange with all 64 lanes active.
    inverse = __shfl_sync(mask, inverse, 0, 64);
    if (valid0)
      out[token * hidden + col0] = e4m3_satfinite_rne(value0 * inverse);
    if (valid1)
      out[token * hidden + col1] = e4m3_satfinite_rne(value1 * inverse);
  }
}

// Q11: adjacent pairs keep the same Group128 reduction and exact encoder,
// while allowing one 32-bit input load and one packed 16-bit output store.
// Partial groups and unaligned input bases use quantize_fp8_warp_groups.
template <typename T>
struct alignas(4) Fp8Values2 {
  T x, y;
};

template <typename T>
__global__ void quantize_fp8_vectors2(const T* x, uint8_t* out,
                                     uint8_t* scales, int64_t total_groups,
                                     int hidden, int groups, int scale_stride) {
  static_assert(sizeof(Fp8Values2<T>) == 4, "two two-byte inputs");
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 63;
  int wave = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t job = tile + wave;
    // Q12: complete Group128 rows and dense FP32 scales are linear in job.
    // Preserve 64-bit global offsets; only the scalar fallback decodes rows.
    int64_t value_offset = job * 128 + lane * 2;
    bool valid_job = job < total_groups;
    float value0 = 0.0f, value1 = 0.0f;
    if (valid_job) {
      Fp8Values2<T> values = *reinterpret_cast<const Fp8Values2<T>*>(
          x + value_offset);
      value0 = static_cast<float>(values.x);
      value1 = static_cast<float>(values.y);
    }
    uint32_t bits0 = __float_as_uint(value0) & 0x7fffffffU;
    uint32_t bits1 = __float_as_uint(value1) & 0x7fffffffU;
    uint32_t amax = bits0 > bits1 ? bits0 : bits1;
#pragma unroll
    for (int offset = 32; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 64);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code = max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
      if (code <= 0 || code > 254) code = 1;
      float scale = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float(code << 23);
      inverse = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float((254 - code) << 23);
      if (valid_job)
        reinterpret_cast<float*>(scales)[job] = scale;
    }
    // Even the padding wave participates in every full-width exchange.
    inverse = __shfl_sync(mask, inverse, 0, 64);
    if (valid_job) {
      uint16_t packed = static_cast<uint16_t>(e4m3_satfinite_rne(value0 * inverse));
      packed |= static_cast<uint16_t>(e4m3_satfinite_rne(value1 * inverse)) << 8;
      *reinterpret_cast<uint16_t*>(out + value_offset) = packed;
    }
  }
}

// Q14: four adjacent values per lane; two independent 32-lane groups share
// each physical64-lane wave. Full groups retain mode5's FP32 multiply/RNE.
template <typename T>
struct alignas(8) Fp8Values4 {
  T x, y, z, w;
};

template <typename T>
__global__ void quantize_fp8_vectors4(const T* x, uint8_t* out,
                                     uint8_t* scales, int64_t total_groups) {
  static_assert(sizeof(Fp8Values4<T>) == 8, "four two-byte inputs");
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 31;
  int subgroup = threadIdx.x >> 5;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 4;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 4) {
    int64_t job = tile + subgroup;
    int64_t value_offset = job * 128 + lane * 4;
    bool valid_job = job < total_groups;
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    if (valid_job) {
      Fp8Values4<T> values = *reinterpret_cast<const Fp8Values4<T>*>(x + value_offset);
      value0 = static_cast<float>(values.x);
      value1 = static_cast<float>(values.y);
      value2 = static_cast<float>(values.z);
      value3 = static_cast<float>(values.w);
    }
    uint32_t amax = __float_as_uint(value0) & 0x7fffffffU;
    uint32_t bits1 = __float_as_uint(value1) & 0x7fffffffU;
    uint32_t bits2 = __float_as_uint(value2) & 0x7fffffffU;
    uint32_t bits3 = __float_as_uint(value3) & 0x7fffffffU;
    amax = amax > bits1 ? amax : bits1;
    amax = amax > bits2 ? amax : bits2;
    amax = amax > bits3 ? amax : bits3;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 32);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code = max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
      if (code <= 0 || code > 254) code = 1;
      float scale = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float(code << 23);
      inverse = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float((254 - code) << 23);
      if (valid_job) reinterpret_cast<float*>(scales)[job] = scale;
    }
    // Padding subgroups join every exchange; width32 keeps maxima/inverses
    // inside one Group128, with all64 physical lanes active throughout.
    inverse = __shfl_sync(mask, inverse, 0, 32);
    if (valid_job) {
      uint32_t packed = static_cast<uint32_t>(e4m3_satfinite_rne(value0 * inverse));
      packed |= static_cast<uint32_t>(e4m3_satfinite_rne(value1 * inverse)) << 8;
      packed |= static_cast<uint32_t>(e4m3_satfinite_rne(value2 * inverse)) << 16;
      packed |= static_cast<uint32_t>(e4m3_satfinite_rne(value3 * inverse)) << 24;
      *reinterpret_cast<uint32_t*>(out + value_offset) = packed;
    }
  }
}

// Q17: eight adjacent values per lane, eight Group128 subgroups per CTA.
// Every physical lane joins width16 exchanges, including padding subgroups.
template <typename T>
struct alignas(16) Fp8Values8 {
  T v0, v1, v2, v3, v4, v5, v6, v7;
};

template <typename T>
__global__ void quantize_fp8_vectors8(const T* x, uint8_t* out,
                                     uint8_t* scales, int64_t total_groups) {
  static_assert(sizeof(Fp8Values8<T>) == 16, "eight two-byte inputs");
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 15;
  int subgroup = threadIdx.x >> 4;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 8;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 8) {
    int64_t job = tile + subgroup;
    int64_t value_offset = job * 128 + lane * 8;
    bool valid_job = job < total_groups;
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    float value4 = 0.0f, value5 = 0.0f, value6 = 0.0f, value7 = 0.0f;
    if (valid_job) {
      Fp8Values8<T> values = *reinterpret_cast<const Fp8Values8<T>*>(x + value_offset);
      value0 = static_cast<float>(values.v0);
      value1 = static_cast<float>(values.v1);
      value2 = static_cast<float>(values.v2);
      value3 = static_cast<float>(values.v3);
      value4 = static_cast<float>(values.v4);
      value5 = static_cast<float>(values.v5);
      value6 = static_cast<float>(values.v6);
      value7 = static_cast<float>(values.v7);
    }
    uint32_t amax = __float_as_uint(value0) & 0x7fffffffU;
    uint32_t bits1 = __float_as_uint(value1) & 0x7fffffffU;
    amax = amax > bits1 ? amax : bits1;
    uint32_t bits2 = __float_as_uint(value2) & 0x7fffffffU;
    amax = amax > bits2 ? amax : bits2;
    uint32_t bits3 = __float_as_uint(value3) & 0x7fffffffU;
    amax = amax > bits3 ? amax : bits3;
    uint32_t bits4 = __float_as_uint(value4) & 0x7fffffffU;
    amax = amax > bits4 ? amax : bits4;
    uint32_t bits5 = __float_as_uint(value5) & 0x7fffffffU;
    amax = amax > bits5 ? amax : bits5;
    uint32_t bits6 = __float_as_uint(value6) & 0x7fffffffU;
    amax = amax > bits6 ? amax : bits6;
    uint32_t bits7 = __float_as_uint(value7) & 0x7fffffffU;
    amax = amax > bits7 ? amax : bits7;
#pragma unroll
    for (int offset = 8; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 16);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code = max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
      if (code <= 0 || code > 254) code = 1;
      float scale = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float(code << 23);
      inverse = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float((254 - code) << 23);
      if (valid_job) reinterpret_cast<float*>(scales)[job] = scale;
    }
    inverse = __shfl_sync(mask, inverse, 0, 16);
    if (valid_job) {
      uint64_t packed = static_cast<uint64_t>(e4m3_satfinite_rne(value0 * inverse));
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(value1 * inverse)) << 8;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(value2 * inverse)) << 16;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(value3 * inverse)) << 24;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(value4 * inverse)) << 32;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(value5 * inverse)) << 40;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(value6 * inverse)) << 48;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(value7 * inverse)) << 56;
      *reinterpret_cast<uint64_t*>(out + value_offset) = packed;
    }
  }
}

// Q39: eight lanes own a complete128-value group. Sixteen values per lane
// produce one uint4, preserving128bit data stores in direct distribution.
// Keep named values: dynamically indexed arrays may lower to private memory.
__device__ __forceinline__ uint4 quantize_fp8_fragment16(
    const at::BFloat16* x, float& scale) {
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  Fp8Values8<at::BFloat16> first = *reinterpret_cast<const Fp8Values8<at::BFloat16>*>(x);
  Fp8Values8<at::BFloat16> second = *reinterpret_cast<const Fp8Values8<at::BFloat16>*>(x + 8);
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
  uint32_t amax = __float_as_uint(value0) & 0x7fffffffU;
  uint32_t bits1 = __float_as_uint(value1) & 0x7fffffffU;
  amax = amax > bits1 ? amax : bits1;
  uint32_t bits2 = __float_as_uint(value2) & 0x7fffffffU;
  amax = amax > bits2 ? amax : bits2;
  uint32_t bits3 = __float_as_uint(value3) & 0x7fffffffU;
  amax = amax > bits3 ? amax : bits3;
  uint32_t bits4 = __float_as_uint(value4) & 0x7fffffffU;
  amax = amax > bits4 ? amax : bits4;
  uint32_t bits5 = __float_as_uint(value5) & 0x7fffffffU;
  amax = amax > bits5 ? amax : bits5;
  uint32_t bits6 = __float_as_uint(value6) & 0x7fffffffU;
  amax = amax > bits6 ? amax : bits6;
  uint32_t bits7 = __float_as_uint(value7) & 0x7fffffffU;
  amax = amax > bits7 ? amax : bits7;
  uint32_t bits8 = __float_as_uint(value8) & 0x7fffffffU;
  amax = amax > bits8 ? amax : bits8;
  uint32_t bits9 = __float_as_uint(value9) & 0x7fffffffU;
  amax = amax > bits9 ? amax : bits9;
  uint32_t bits10 = __float_as_uint(value10) & 0x7fffffffU;
  amax = amax > bits10 ? amax : bits10;
  uint32_t bits11 = __float_as_uint(value11) & 0x7fffffffU;
  amax = amax > bits11 ? amax : bits11;
  uint32_t bits12 = __float_as_uint(value12) & 0x7fffffffU;
  amax = amax > bits12 ? amax : bits12;
  uint32_t bits13 = __float_as_uint(value13) & 0x7fffffffU;
  amax = amax > bits13 ? amax : bits13;
  uint32_t bits14 = __float_as_uint(value14) & 0x7fffffffU;
  amax = amax > bits14 ? amax : bits14;
  uint32_t bits15 = __float_as_uint(value15) & 0x7fffffffU;
  amax = amax > bits15 ? amax : bits15;
#pragma unroll
  for (int offset = 4; offset; offset >>= 1) {
    uint32_t peer = __shfl_xor_sync(mask, amax, offset, 8);
    amax = amax > peer ? amax : peer;
  }
  scale = 0.0f;
  float inverse = 0.0f;
  if ((threadIdx.x & 7) == 0) {
    int max_exp = (amax >> 23) & 255;
    int code = max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
    if (code <= 0 || code > 254) code = 1;
    scale = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float(code << 23);
    inverse = max_exp == 255 ? __int_as_float(0x7fc00000) : __int_as_float((254 - code) << 23);
  }
  inverse = __shfl_sync(mask, inverse, 0, 8);
  uint32_t packed0 = static_cast<uint32_t>(e4m3_satfinite_rne(value0 * inverse));
  packed0 |= static_cast<uint32_t>(e4m3_satfinite_rne(value1 * inverse)) << 8;
  packed0 |= static_cast<uint32_t>(e4m3_satfinite_rne(value2 * inverse)) << 16;
  packed0 |= static_cast<uint32_t>(e4m3_satfinite_rne(value3 * inverse)) << 24;
  uint32_t packed1 = static_cast<uint32_t>(e4m3_satfinite_rne(value4 * inverse));
  packed1 |= static_cast<uint32_t>(e4m3_satfinite_rne(value5 * inverse)) << 8;
  packed1 |= static_cast<uint32_t>(e4m3_satfinite_rne(value6 * inverse)) << 16;
  packed1 |= static_cast<uint32_t>(e4m3_satfinite_rne(value7 * inverse)) << 24;
  uint32_t packed2 = static_cast<uint32_t>(e4m3_satfinite_rne(value8 * inverse));
  packed2 |= static_cast<uint32_t>(e4m3_satfinite_rne(value9 * inverse)) << 8;
  packed2 |= static_cast<uint32_t>(e4m3_satfinite_rne(value10 * inverse)) << 16;
  packed2 |= static_cast<uint32_t>(e4m3_satfinite_rne(value11 * inverse)) << 24;
  uint32_t packed3 = static_cast<uint32_t>(e4m3_satfinite_rne(value12 * inverse));
  packed3 |= static_cast<uint32_t>(e4m3_satfinite_rne(value13 * inverse)) << 8;
  packed3 |= static_cast<uint32_t>(e4m3_satfinite_rne(value14 * inverse)) << 16;
  packed3 |= static_cast<uint32_t>(e4m3_satfinite_rne(value15 * inverse)) << 24;
  return uint4{packed0, packed1, packed2, packed3};
}

// Q39: quantize each input fragment once, then reuse two encoded uint4s
// across valid topk destinations. Match Q26's2048B/wave/rank-traversal tile.
__global__ void quantize_scatter_fp8_tokens2(
    const at::BFloat16* x, uint4* dst, float* scale_dst, const int* inverse,
    int64_t tokens, int topk, int hidden, int capacity) {
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      const int* ranks = inverse + token * topk;
      const auto* in = x + token * hidden;
      for (int fragment = 0; fragment < hidden; fragment += 2048) {
        float scale0, scale1;
        uint4 value0 = quantize_fp8_fragment16(in + fragment + lane * 16, scale0);
        uint4 value1 = quantize_fp8_fragment16(in + fragment + 1024 + lane * 16, scale1);
        int group = fragment / 128 + (lane >> 3);
        for (int k = 0; k < topk; ++k) {
          int row = ranks[k];
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * (hidden / 16) + fragment / 16 + lane;
            __stcs(out, value0);
            __stcs(out + 64, value1);
            if ((lane & 7) == 0) {
              auto* scales = scale_dst + static_cast<int64_t>(row) * (hidden / 128) + group;
              scales[0] = scale0;
              scales[8] = scale1;
            }
          }
        }
      }
    }
  }
}

// Q40: direct MXFP8 distribution. Two lanes own one complete32-value
// group; preserve BF16-truncated amax, E8M0 scale and BF16 product rounding.
__device__ __forceinline__ uint4 quantize_mx8_fragment16(
    const at::BFloat16* x, uint8_t& scale) {
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  Fp8Values8<at::BFloat16> first = *reinterpret_cast<const Fp8Values8<at::BFloat16>*>(x);
  Fp8Values8<at::BFloat16> second = *reinterpret_cast<const Fp8Values8<at::BFloat16>*>(x + 8);
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
#pragma unroll
  for (int offset = 1; offset; offset >>= 1) {
    uint32_t peer = __shfl_xor_sync(mask, amax, offset, 2);
    amax = amax > peer ? amax : peer;
  }
  scale = 0;
  float inverse = 0.0f;
  if ((threadIdx.x & 1) == 0) {
    int max_exp = (amax >> 23) & 255;
    int code = max_exp <= 8 ? 1 : max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
    if (amax == 0) code = 0;
    if (max_exp == 255) code = 255;
    scale = static_cast<uint8_t>(code);
    inverse = code == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, 127 - code);
    if (amax == 0) inverse = 0.0f;
  }
  inverse = __shfl_sync(mask, inverse, 0, 2);
  float scaled0 = static_cast<float>(at::BFloat16(value0 * inverse));
  float scaled1 = static_cast<float>(at::BFloat16(value1 * inverse));
  float scaled2 = static_cast<float>(at::BFloat16(value2 * inverse));
  float scaled3 = static_cast<float>(at::BFloat16(value3 * inverse));
  float scaled4 = static_cast<float>(at::BFloat16(value4 * inverse));
  float scaled5 = static_cast<float>(at::BFloat16(value5 * inverse));
  float scaled6 = static_cast<float>(at::BFloat16(value6 * inverse));
  float scaled7 = static_cast<float>(at::BFloat16(value7 * inverse));
  float scaled8 = static_cast<float>(at::BFloat16(value8 * inverse));
  float scaled9 = static_cast<float>(at::BFloat16(value9 * inverse));
  float scaled10 = static_cast<float>(at::BFloat16(value10 * inverse));
  float scaled11 = static_cast<float>(at::BFloat16(value11 * inverse));
  float scaled12 = static_cast<float>(at::BFloat16(value12 * inverse));
  float scaled13 = static_cast<float>(at::BFloat16(value13 * inverse));
  float scaled14 = static_cast<float>(at::BFloat16(value14 * inverse));
  float scaled15 = static_cast<float>(at::BFloat16(value15 * inverse));
  uint32_t packed0 = static_cast<uint32_t>(e4m3_satfinite_rne(scaled0));
  packed0 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled1)) << 8;
  packed0 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled2)) << 16;
  packed0 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled3)) << 24;
  uint32_t packed1 = static_cast<uint32_t>(e4m3_satfinite_rne(scaled4));
  packed1 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled5)) << 8;
  packed1 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled6)) << 16;
  packed1 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled7)) << 24;
  uint32_t packed2 = static_cast<uint32_t>(e4m3_satfinite_rne(scaled8));
  packed2 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled9)) << 8;
  packed2 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled10)) << 16;
  packed2 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled11)) << 24;
  uint32_t packed3 = static_cast<uint32_t>(e4m3_satfinite_rne(scaled12));
  packed3 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled13)) << 8;
  packed3 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled14)) << 16;
  packed3 |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled15)) << 24;
  return uint4{packed0, packed1, packed2, packed3};
}

// Two encoded uint4 fragments share the existing inverse-rank traversal.
// Scale owners write raw E8M0 bytes; complete2048B tiles have no scale padding.
__global__ void quantize_scatter_mx8_tokens2(
    const at::BFloat16* x, uint4* dst, uint8_t* scale_dst, const int* inverse,
    int64_t tokens, int topk, int hidden, int capacity) {
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      const int* ranks = inverse + token * topk;
      const auto* in = x + token * hidden;
      for (int fragment = 0; fragment < hidden; fragment += 2048) {
        uint8_t scale0, scale1;
        uint4 value0 = quantize_mx8_fragment16(in + fragment + lane * 16, scale0);
        uint4 value1 = quantize_mx8_fragment16(in + fragment + 1024 + lane * 16, scale1);
        int group = fragment / 32 + (lane >> 1);
        for (int k = 0; k < topk; ++k) {
          int row = ranks[k];
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * (hidden / 16) + fragment / 16 + lane;
            __stcs(out, value0);
            __stcs(out + 64, value1);
            if ((lane & 1) == 0) {
              auto* scales = scale_dst + static_cast<int64_t>(row) * (hidden / 32) + group;
              scales[0] = scale0;
              scales[32] = scale1;
            }
          }
        }
      }
    }
  }
}

// Q49: H4096 MXFP8 reuses one wave-local rank collection across two tiles.
__global__ void quantize_scatter_mx8_rankwave_tokens2(
    const at::BFloat16* x, uint4* dst, uint8_t* scale_dst, const int* inverse,
    int64_t tokens, int topk, int hidden, int capacity) {
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      const int* ranks = inverse + token * topk;
      // Q49: collect ranks once per token and reuse across both fragments.
      int cached_rank = -1;
      if (lane < topk) cached_rank = ranks[lane];
      const auto* in = x + token * hidden;
      for (int fragment = 0; fragment < hidden; fragment += 2048) {
        uint8_t scale0, scale1;
        uint4 value0 = quantize_mx8_fragment16(in + fragment + lane * 16, scale0);
        uint4 value1 = quantize_mx8_fragment16(in + fragment + 1024 + lane * 16, scale1);
        int group = fragment / 32 + (lane >> 1);
        for (int k = 0; k < topk; ++k) {
          // The valid-token branch covers a complete64-lane wave.
          int row = __shfl_sync(0xffffffffffffffffULL, cached_rank, k, 64);
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * (hidden / 16) + fragment / 16 + lane;
            __stcs(out, value0);
            __stcs(out + 64, value1);
            if ((lane & 1) == 0) {
              auto* scales = scale_dst + static_cast<int64_t>(row) * (hidden / 32) + group;
              scales[0] = scale0;
              scales[32] = scale1;
            }
          }
        }
      }
    }
  }
}

template <typename T>
void launch_fp8_warp_groups(const T* x, uint8_t* out, uint8_t* scales,
                            int64_t total_groups, int hidden, int groups,
                            int scale_stride, cudaStream_t stream) {
  if (hidden % 128 == 0 && reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(out) % 8 == 0) {
    int vector_grid = std::min<int64_t>((total_groups + 7) / 8, 65535);
    quantize_fp8_vectors8<<<vector_grid, 128, 0, stream>>>(x, out, scales, total_groups);
    return;
  }
  if (hidden % 128 == 0 && reinterpret_cast<uintptr_t>(x) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(out) % 4 == 0) {
    int vector_grid = std::min<int64_t>((total_groups + 3) / 4, 65535);
    quantize_fp8_vectors4<<<vector_grid, 128, 0, stream>>>(x, out, scales, total_groups);
    return;
  }
  int grid = std::min<int64_t>((total_groups + 1) / 2, 65535);
  if (hidden % 128 == 0 && reinterpret_cast<uintptr_t>(x) % 4 == 0 &&
      reinterpret_cast<uintptr_t>(out) % 2 == 0) {
    quantize_fp8_vectors2<<<grid, 128, 0, stream>>>(
        x, out, scales, total_groups, hidden, groups, scale_stride);
    return;
  }
  quantize_fp8_warp_groups<<<grid, 128, 0, stream>>>(
      x, out, scales, total_groups, hidden, groups, scale_stride);
}

// C600-U: four independent 32-element MX groups per CTA. All 128 lanes
// execute every shuffle, including padding groups at the end of a tile.
// This keeps both halves of each physical 64-lane wave converged while
// width=32 prevents values from crossing a quantization-group boundary.
template <typename T, int Mode>
__global__ void quantize_mx_subgroups(const T* x, uint8_t* out, uint8_t* scales,
                                     int64_t total_groups, int hidden, int groups,
                                     int scale_stride) {
  static_assert(Mode == 3 || Mode == 9, "MX subgroup mode");
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 31;
  int subgroup = threadIdx.x >> 5;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 4;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 4) {
    int64_t job = tile + subgroup;
    int64_t token = job / groups;
    int group = job % groups;
    int col = group * 32 + lane;
    bool valid_job = job < total_groups;
    bool valid = valid_job && col < hidden;
    float value = valid ? static_cast<float>(x[token * hidden + col]) : 0.0f;
    uint32_t amax = __float_as_uint(value) & 0x7fff0000U;
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 32);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code;
      if constexpr (Mode == 9) {
        code = max(max_exp - 2, 0);
      } else {
        code = max_exp <= 8 ? 1 : max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
        if (amax == 0) code = 0;
      }
      if (max_exp == 255) code = 255;
      if (valid_job) scales[token * scale_stride + group] = static_cast<uint8_t>(code);
      inverse = code == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, 127 - code);
      if constexpr (Mode == 3) {
        if (amax == 0) inverse = 0.0f;
      }
    }
    inverse = __shfl_sync(mask, inverse, 0, 32);
    if constexpr (Mode == 9) {
      // Match the original BF16 multiplication rounding before E2M1 RNE.
      unsigned packed = e2m1(static_cast<float>(at::BFloat16(value * inverse)));
      unsigned peer = __shfl_xor_sync(mask, packed, 1, 32);
      if (valid && (lane & 1) == 0)
        out[token * (hidden / 2) + col / 2] = static_cast<uint8_t>(packed | (peer << 4));
    } else if (valid) {
      float scaled = value * inverse;
      if constexpr (std::is_same<T, at::BFloat16>::value)
        scaled = static_cast<float>(at::BFloat16(scaled));
      out[token * hidden + col] = __maca_cvt_float_to_fp8(scaled, __MACA_SATFINITE, __MACA_E4M3);
    }
  }
}

// Four adjacent BF16/Half inputs per lane; only complete 32-element groups
// with aligned row bases enter this path. Named members avoid indexed arrays.
template <typename T>
struct alignas(8) MxValues4 {
  T x, y, z, w;
};

template <typename T, int Mode>
__global__ void quantize_mx_vectors4(const T* x, uint8_t* out, uint8_t* scales,
                                    int64_t total_groups, int hidden, int groups,
                                    int scale_stride) {
  static_assert(Mode == 3 || Mode == 9, "MX vector mode");
  static_assert(sizeof(MxValues4<T>) == 8, "four two-byte inputs");
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 7;
  int subgroup = threadIdx.x >> 3;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 16;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 16) {
    int64_t job = tile + subgroup;
    // Complete groups cover contiguous input/output rows, so addressing does
    // not need to decode job into token/group and then reconstruct the row.
    int64_t value_offset = job * 32 + lane * 4;
    bool valid_job = job < total_groups;
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    if (valid_job) {
      MxValues4<T> values = *reinterpret_cast<const MxValues4<T>*>(
          x + value_offset);
      value0 = static_cast<float>(values.x);
      value1 = static_cast<float>(values.y);
      value2 = static_cast<float>(values.z);
      value3 = static_cast<float>(values.w);
    }
    // Preserve the original BF16-truncated amax, including for Half inputs.
    uint32_t amax = __float_as_uint(value0) & 0x7fff0000U;
    uint32_t bits1 = __float_as_uint(value1) & 0x7fff0000U;
    uint32_t bits2 = __float_as_uint(value2) & 0x7fff0000U;
    uint32_t bits3 = __float_as_uint(value3) & 0x7fff0000U;
    amax = amax > bits1 ? amax : bits1;
    amax = amax > bits2 ? amax : bits2;
    amax = amax > bits3 ? amax : bits3;
#pragma unroll
    for (int offset = 4; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 8);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code;
      if constexpr (Mode == 9) {
        code = max(max_exp - 2, 0);
      } else {
        code = max_exp <= 8 ? 1 : max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
        if (amax == 0) code = 0;
      }
      if (max_exp == 255) code = 255;
      if (valid_job) {
        int64_t scale_offset = job;
        // H%32==0 makes scale_stride equal groups or groups+1. Only odd group
        // counts have one padding byte per row; only the leader needs division.
        if (scale_stride != groups) scale_offset += job / groups;
        scales[scale_offset] = static_cast<uint8_t>(code);
      }
      inverse = code == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, 127 - code);
      if constexpr (Mode == 3) {
        if (amax == 0) inverse = 0.0f;
      }
    }
    // Padding subgroups still participate with all64 physical lanes active;
    // width8 keeps every reduction/broadcast inside one quantization group.
    inverse = __shfl_sync(mask, inverse, 0, 8);
    if (valid_job) {
      float scaled0 = value0 * inverse;
      float scaled1 = value1 * inverse;
      float scaled2 = value2 * inverse;
      float scaled3 = value3 * inverse;
      if constexpr (Mode == 9 || std::is_same<T, at::BFloat16>::value) {
        scaled0 = static_cast<float>(at::BFloat16(scaled0));
        scaled1 = static_cast<float>(at::BFloat16(scaled1));
        scaled2 = static_cast<float>(at::BFloat16(scaled2));
        scaled3 = static_cast<float>(at::BFloat16(scaled3));
      }
      if constexpr (Mode == 9) {
        uint16_t packed = e2m1(scaled0) | (e2m1(scaled1) << 4) |
            (e2m1(scaled2) << 8) | (e2m1(scaled3) << 12);
        *reinterpret_cast<uint16_t*>(out + value_offset / 2) = packed;
      } else {
        // Q5-mode3: reuse the exact FP32-bit SATFINITE/RNE encoder after
        // the existing MX multiply/BF16 rounding. Scale, lane ownership and
        // packing are unchanged; do not assume scaled values are bounded.
        uint32_t packed = static_cast<uint32_t>(e4m3_satfinite_rne(scaled0));
        packed |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled1)) << 8;
        packed |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled2)) << 16;
        packed |= static_cast<uint32_t>(e4m3_satfinite_rne(scaled3)) << 24;
        *reinterpret_cast<uint32_t*>(out + value_offset) = packed;
      }
    }
  }
}

// Q20: eight adjacent inputs per lane, four lanes per complete MXFP4 group.
// All physical lanes join width4 exchanges, including padding groups.
template <typename T>
struct alignas(16) MxValues8 {
  T v0, v1, v2, v3, v4, v5, v6, v7;
};

template <typename T>
__global__ void quantize_mx4_vectors8(const T* x, uint8_t* out, uint8_t* scales,
                                     int64_t total_groups, int groups,
                                     int scale_stride) {
  static_assert(sizeof(MxValues8<T>) == 16, "eight two-byte inputs");
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 3;
  int subgroup = threadIdx.x >> 2;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 32;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 32) {
    int64_t job = tile + subgroup;
    int64_t value_offset = job * 32 + lane * 8;
    bool valid_job = job < total_groups;
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    float value4 = 0.0f, value5 = 0.0f, value6 = 0.0f, value7 = 0.0f;
    if (valid_job) {
      MxValues8<T> values = *reinterpret_cast<const MxValues8<T>*>(x + value_offset);
      value0 = static_cast<float>(values.v0);
      value1 = static_cast<float>(values.v1);
      value2 = static_cast<float>(values.v2);
      value3 = static_cast<float>(values.v3);
      value4 = static_cast<float>(values.v4);
      value5 = static_cast<float>(values.v5);
      value6 = static_cast<float>(values.v6);
      value7 = static_cast<float>(values.v7);
    }
    // Match the original BF16-truncated amax for both BF16 and Half inputs.
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
#pragma unroll
    for (int offset = 2; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 4);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code = max(max_exp - 2, 0);
      if (max_exp == 255) code = 255;
      if (valid_job) {
        int64_t scale_offset = job;
        if (scale_stride != groups) scale_offset += job / groups;
        scales[scale_offset] = static_cast<uint8_t>(code);
      }
      inverse = code == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, 127 - code);
    }
    inverse = __shfl_sync(mask, inverse, 0, 4);
    if (valid_job) {
      // MXFP4 rounds each product to BF16 even when the source is Half.
      at::BFloat16 scaled0(value0 * inverse);
      at::BFloat16 scaled1(value1 * inverse);
      at::BFloat16 scaled2(value2 * inverse);
      at::BFloat16 scaled3(value3 * inverse);
      at::BFloat16 scaled4(value4 * inverse);
      at::BFloat16 scaled5(value5 * inverse);
      at::BFloat16 scaled6(value6 * inverse);
      at::BFloat16 scaled7(value7 * inverse);
      uint32_t packed = static_cast<uint32_t>(e2m1_bf16_rne(scaled0));
      packed |= static_cast<uint32_t>(e2m1_bf16_rne(scaled1)) << 4;
      packed |= static_cast<uint32_t>(e2m1_bf16_rne(scaled2)) << 8;
      packed |= static_cast<uint32_t>(e2m1_bf16_rne(scaled3)) << 12;
      packed |= static_cast<uint32_t>(e2m1_bf16_rne(scaled4)) << 16;
      packed |= static_cast<uint32_t>(e2m1_bf16_rne(scaled5)) << 20;
      packed |= static_cast<uint32_t>(e2m1_bf16_rne(scaled6)) << 24;
      packed |= static_cast<uint32_t>(e2m1_bf16_rne(scaled7)) << 28;
      *reinterpret_cast<uint32_t*>(out + value_offset / 2) = packed;
    }
  }
}

// Q21: eight inputs per lane for MXFP8; preserve source-dependent product rounding.
template <typename T>
__global__ void quantize_mx8_vectors8(const T* x, uint8_t* out, uint8_t* scales,
                                     int64_t total_groups, int groups,
                                     int scale_stride) {
  static_assert(sizeof(MxValues8<T>) == 16, "eight two-byte inputs");
  constexpr unsigned long long mask = 0xffffffffffffffffULL;
  int lane = threadIdx.x & 3;
  int subgroup = threadIdx.x >> 2;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 32;
       tile < total_groups; tile += static_cast<int64_t>(gridDim.x) * 32) {
    int64_t job = tile + subgroup;
    int64_t value_offset = job * 32 + lane * 8;
    bool valid_job = job < total_groups;
    float value0 = 0.0f, value1 = 0.0f, value2 = 0.0f, value3 = 0.0f;
    float value4 = 0.0f, value5 = 0.0f, value6 = 0.0f, value7 = 0.0f;
    if (valid_job) {
      MxValues8<T> values = *reinterpret_cast<const MxValues8<T>*>(x + value_offset);
      value0 = static_cast<float>(values.v0);
      value1 = static_cast<float>(values.v1);
      value2 = static_cast<float>(values.v2);
      value3 = static_cast<float>(values.v3);
      value4 = static_cast<float>(values.v4);
      value5 = static_cast<float>(values.v5);
      value6 = static_cast<float>(values.v6);
      value7 = static_cast<float>(values.v7);
    }
    // Match the original BF16-truncated amax for both BF16 and Half inputs.
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
#pragma unroll
    for (int offset = 2; offset; offset >>= 1) {
      uint32_t peer = __shfl_xor_sync(mask, amax, offset, 4);
      amax = amax > peer ? amax : peer;
    }
    float inverse = 0.0f;
    if (lane == 0) {
      int max_exp = (amax >> 23) & 255;
      int code = max_exp <= 8 ? 1 : max_exp - 8 + ((amax & 0x7fffffU) > 0x600000U);
      if (amax == 0) code = 0;
      if (max_exp == 255) code = 255;
      if (valid_job) {
        int64_t scale_offset = job;
        if (scale_stride != groups) scale_offset += job / groups;
        scales[scale_offset] = static_cast<uint8_t>(code);
      }
      inverse = code == 255 ? __int_as_float(0x7fc00000) : ldexpf(1.0f, 127 - code);
      if (amax == 0) inverse = 0.0f;
    }
    inverse = __shfl_sync(mask, inverse, 0, 4);
    if (valid_job) {
      float scaled0 = value0 * inverse;
      float scaled1 = value1 * inverse;
      float scaled2 = value2 * inverse;
      float scaled3 = value3 * inverse;
      float scaled4 = value4 * inverse;
      float scaled5 = value5 * inverse;
      float scaled6 = value6 * inverse;
      float scaled7 = value7 * inverse;
      if constexpr (std::is_same<T, at::BFloat16>::value) {
        scaled0 = static_cast<float>(at::BFloat16(scaled0));
        scaled1 = static_cast<float>(at::BFloat16(scaled1));
        scaled2 = static_cast<float>(at::BFloat16(scaled2));
        scaled3 = static_cast<float>(at::BFloat16(scaled3));
        scaled4 = static_cast<float>(at::BFloat16(scaled4));
        scaled5 = static_cast<float>(at::BFloat16(scaled5));
        scaled6 = static_cast<float>(at::BFloat16(scaled6));
        scaled7 = static_cast<float>(at::BFloat16(scaled7));
      }
      uint64_t packed = static_cast<uint64_t>(e4m3_satfinite_rne(scaled0));
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(scaled1)) << 8;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(scaled2)) << 16;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(scaled3)) << 24;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(scaled4)) << 32;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(scaled5)) << 40;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(scaled6)) << 48;
      packed |= static_cast<uint64_t>(e4m3_satfinite_rne(scaled7)) << 56;
      *reinterpret_cast<uint64_t*>(out + value_offset) = packed;
    }
  }
}

template <typename T>
void launch_mx_subgroups(const T* x, uint8_t* out, uint8_t* scales,
                         int64_t total_groups, int hidden, int groups,
                         int scale_stride, int mode, cudaStream_t stream) {
  // This launcher is already restricted to C600-U/mode3 or9. Width divisibility
  // ensures complete groups and aligned row strides; no measured-shape list.
  if (mode == 3 && hidden % 32 == 0 &&
      reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(out) % 8 == 0) {
    int grid = std::min<int64_t>((total_groups + 31) / 32, 65535);
    quantize_mx8_vectors8<<<grid, 128, 0, stream>>>(
        x, out, scales, total_groups, groups, scale_stride);
    return;
  }
  if (mode == 9 && hidden % 32 == 0 &&
      reinterpret_cast<uintptr_t>(x) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(out) % 4 == 0) {
    int grid = std::min<int64_t>((total_groups + 31) / 32, 65535);
    quantize_mx4_vectors8<<<grid, 128, 0, stream>>>(
        x, out, scales, total_groups, groups, scale_stride);
    return;
  }
  if (hidden % 32 == 0 && reinterpret_cast<uintptr_t>(x) % 8 == 0 &&
      reinterpret_cast<uintptr_t>(out) % 4 == 0) {
    int grid = std::min<int64_t>((total_groups + 15) / 16, 65535);
    if (mode == 3)
      quantize_mx_vectors4<T, 3><<<grid, 128, 0, stream>>>(
          x, out, scales, total_groups, hidden, groups, scale_stride);
    else
      quantize_mx_vectors4<T, 9><<<grid, 128, 0, stream>>>(
          x, out, scales, total_groups, hidden, groups, scale_stride);
    return;
  }
  int grid = std::min<int64_t>((total_groups + 3) / 4, 65535);
  if (mode == 3)
    quantize_mx_subgroups<T, 3><<<grid, 128, 0, stream>>>(
        x, out, scales, total_groups, hidden, groups, scale_stride);
  else
    quantize_mx_subgroups<T, 9><<<grid, 128, 0, stream>>>(
        x, out, scales, total_groups, hidden, groups, scale_stride);
}

__global__ void make_keys(const int* ids, int* keys, int* positions,
                          int routes, int start, int end, int sentinel) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < routes) {
    int e = ids[i];
    keys[i] = e >= start && e < end ? e : sentinel;
    positions[i] = static_cast<int>(i);
  }
}

__global__ void expert_bounds(const int* keys, int64_t* bounds, int routes, int experts) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e <= experts) {
    int lo = 0, hi = routes;
    while (lo < hi) {
      int mid = lo + (hi - lo) / 2;
      if (keys[mid] < e) lo = mid + 1;
      else hi = mid;
    }
    bounds[e] = lo;
  }
}

__global__ void write_counts(const int64_t* bounds, int64_t* out,
                             int start, int end, int count_type) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < end - start) {
    int e = start + i;
    out[i] = bounds[e + 1] - (count_type == 0 ? bounds[start] : bounds[e]);
  }
}

__global__ void write_indices(const int* sorted, const int64_t* bounds, int* out,
                              const float* weights, float* expanded_weights,
                              int routes, int experts, int capacity, int row_type) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < routes && i < bounds[experts]) {
    int original = sorted[i];
    if (row_type == 1) out[i] = original;
    else out[original] = static_cast<int>(i);
    if (weights && i < capacity) expanded_weights[i] = weights[original];
  }
}

// Control is written on every invocation, including empty inputs and errors.
// No host read, allocation or data-dependent launch count is needed in capture.
__global__ void resolve_active_rows(const int64_t* active, const int64_t* bounds,
                                     int64_t* info, int routes, int experts,
                                     int capacity) {
  int64_t a = *active;
  int64_t available = routes ? bounds[experts] : 0;
  int64_t logical = a < 0 ? -1 : ((a == 0 || a > routes) ? routes : a);
  int64_t status = a < 0 ? 1 : (logical > capacity ? 2 : 0);
  info[0] = logical;
  info[1] = status ? 0 : (logical < available ? logical : available);
  info[2] = available;
  info[3] = status;
}

__global__ void write_indices_active(const int* sorted, const int64_t* info,
                                      int* out, const float* weights,
                                      float* expanded_weights, int routes,
                                      int row_type) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Row indices and counts describe every available route, independently of
  // the requested activeNum or fixed output storage capacity.
  if (i < routes && i < info[2]) {
    int original = sorted[i];
    if (row_type == 1) out[i] = original;
    else out[original] = static_cast<int>(i);
    if (weights && i < info[1]) expanded_weights[i] = weights[original];
  }
}

template <bool Vectorized>
__global__ void gather_rows_active(const uint8_t* src, uint8_t* dst,
                                    const int* sorted, const int64_t* info,
                                    int topk, int64_t row_bytes) {
  int64_t effective = info[1];
  for (int64_t r = blockIdx.x; r < effective; r += gridDim.x) {
    int token = sorted[r] / topk;
    const auto* in = src + static_cast<int64_t>(token) * row_bytes;
    auto* out = dst + r * row_bytes;
    if constexpr (Vectorized) {
      for (int64_t j = threadIdx.x; j < row_bytes / sizeof(uint4); j += blockDim.x)
        reinterpret_cast<uint4*>(out)[j] = reinterpret_cast<const uint4*>(in)[j];
    } else {
      for (int64_t j = threadIdx.x; j < row_bytes; j += blockDim.x) out[j] = in[j];
    }
  }
}

// Non-overwriting SortPairs leaves make_keys' identity values intact. Sorting
// original route IDs with these rank values produces the inverse permutation
// with coalesced output writes. Invalid sorted-tail ranks must map to -1.
__global__ void invalidate_inverse_ranks(int* ranks, const int64_t* bounds,
                                         int routes, int experts) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < routes && i >= bounds[experts]) ranks[i] = -1;
}

// A row is a byte string: this also preserves FP8/FP4 payloads and NaN bits.
template <bool Vectorized>
__global__ void gather_rows(const uint8_t* src, uint8_t* dst, const int* sorted,
                            const int64_t* bounds, int experts, int topk,
                            int64_t row_bytes, int capacity) {
  for (int64_t r = blockIdx.x; r < capacity; r += gridDim.x) {
    if (r >= bounds[experts]) continue;
    int token = sorted[r] / topk;
    auto* in = src + static_cast<int64_t>(token) * row_bytes;
    auto* out = dst + r * row_bytes;
    if constexpr (Vectorized) {
      for (int64_t j = threadIdx.x; j < row_bytes / sizeof(uint4); j += blockDim.x)
        reinterpret_cast<uint4*>(out)[j] = reinterpret_cast<const uint4*>(in)[j];
    } else {
      for (int64_t j = threadIdx.x; j < row_bytes; j += blockDim.x) out[j] = in[j];
    }
  }
}

// Two independent physical waves own two complete data rows. Keep each wave's
// accesses contiguous and preserve byte payloads, invalid/capacity tails and
// 64-bit global addresses; no inter-wave communication or shared memory.
__global__ void gather_data_rows2(const uint4* src, uint4* dst, const int* sorted,
                                  const int64_t* bounds, int experts, int topk,
                                  int64_t vectors_per_row, int capacity) {
  int lane = threadIdx.x & 63;
  int row_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < capacity; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t row = tile + row_in_tile;
    if (row < capacity && row < bounds[experts]) {
      int token = sorted[row] / topk;
      auto* in = src + static_cast<int64_t>(token) * vectors_per_row;
      auto* out = dst + row * vectors_per_row;
      // Q22: stream output stores while preserving reusable input load policy.
      for (int64_t j = lane; j < vectors_per_row; j += 64) {
        uint4 value = in[j];
        __stcs(out + j, value);
      }
    }
  }
}

// Q37: a forward-index wave shares one sorted-row/token decode between data
// and scale. Keep data streaming and consume the scale only after the data loop.
__global__ void gather_data_scales_rows2(const uint4* src, uint4* dst,
                                       const uint4* scale_src, uint4* scale_dst,
                                       const int* sorted, const int64_t* bounds,
                                       int experts, int topk,
                                       int64_t vectors_per_row,
                                       int scale_vectors_per_row, int capacity) {
  int lane = threadIdx.x & 63;
  int row_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < capacity; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t row = tile + row_in_tile;
    if (row < capacity && row < bounds[experts]) {
      int token = sorted[row] / topk;
      auto* in = src + static_cast<int64_t>(token) * vectors_per_row;
      auto* out = dst + row * vectors_per_row;
      for (int64_t j = lane; j < vectors_per_row; j += 64) {
        uint4 value = in[j];
        __stcs(out + j, value);
      }
      if (lane < scale_vectors_per_row)
        scale_dst[row * scale_vectors_per_row + lane] =
            scale_src[static_cast<int64_t>(token) * scale_vectors_per_row + lane];
    }
  }
}

// Q24: each physical wave loads a token fragment once and reuses it for all
// valid destinations in the already-computed original-route -> sorted-row map.
// Destination ranks are unique, including repeated expert IDs within a token.
__global__ void scatter_data_tokens2(const uint4* src, uint4* dst,
                                     const int* inverse, int64_t tokens, int topk,
                                     int64_t vectors_per_row, int capacity) {
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      auto* in = src + token * vectors_per_row;
      const int* ranks = inverse + token * topk;
      // Q26: two fragments share one rank load and destination-row address.
      for (int64_t j = lane; j < vectors_per_row; j += 128) {
        uint4 value0 = in[j];
        bool has_second = j + 64 < vectors_per_row;
        uint4 value1{};
        if (has_second) value1 = in[j + 64];
        for (int k = 0; k < topk; ++k) {
          int row = ranks[k];
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * vectors_per_row + j;
            __stcs(out, value0);
            if (has_second) __stcs(out + 64, value1);
          }
        }
      }
    }
  }
}

// Q34a: fuse scale distribution with just-in-time scale loads.
// Data fragments keep Q26 ownership and streaming stores; scale uses ordinary
// stores and the same unique original-route -> destination-row mapping.
__global__ void scatter_data_scales_tokens2(const uint4* src, uint4* dst,
                                     const uint4* scale_src, uint4* scale_dst,
                                     const int* inverse, int64_t tokens, int topk,
                                     int64_t vectors_per_row, int capacity) {
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      auto* in = src + token * vectors_per_row;
      const int* ranks = inverse + token * topk;
      // Q26: two fragments share one rank load and destination-row address.
      for (int64_t j = lane; j < vectors_per_row; j += 128) {
        uint4 value0 = in[j];
        bool has_second = j + 64 < vectors_per_row;
        uint4 value1{};
        if (has_second) value1 = in[j + 64];
        for (int k = 0; k < topk; ++k) {
          int row = ranks[k];
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * vectors_per_row + j;
            __stcs(out, value0);
            if (has_second) __stcs(out + 64, value1);
            // Q34a: load only when consumed; do not carry scale across branches
            // or topk iterations. Only the first eight data fragments own scale.
            if (j < 8)
              scale_dst[static_cast<int64_t>(row) * 8 + j] = scale_src[token * 8 + j];
          }
        }
      }
    }
  }
}

// Q36: extend fusion to 64/256-byte scale rows. Preserve Q34a runtime data
// stride, paired-fragment tail and immediate scale consumption.
template <int ScaleVectors>
__global__ void scatter_data_scales_tokens2_width(const uint4* src, uint4* dst,
                                     const uint4* scale_src, uint4* scale_dst,
                                     const int* inverse, int64_t tokens, int topk,
                                     int64_t vectors_per_row, int capacity) {
  static_assert(ScaleVectors == 4 || ScaleVectors == 16,
                "additional scale rows are 64 or 256 bytes");
  int lane = threadIdx.x & 63;
  int token_in_tile = threadIdx.x >> 6;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 2;
       tile < tokens; tile += static_cast<int64_t>(gridDim.x) * 2) {
    int64_t token = tile + token_in_tile;
    if (token < tokens) {
      auto* in = src + token * vectors_per_row;
      const int* ranks = inverse + token * topk;
      // Q26: two fragments share one rank load and destination-row address.
      for (int64_t j = lane; j < vectors_per_row; j += 128) {
        uint4 value0 = in[j];
        bool has_second = j + 64 < vectors_per_row;
        uint4 value1{};
        if (has_second) value1 = in[j + 64];
        for (int k = 0; k < topk; ++k) {
          int row = ranks[k];
          if (row >= 0 && row < capacity) {
            auto* out = dst + static_cast<int64_t>(row) * vectors_per_row + j;
            __stcs(out, value0);
            if (has_second) __stcs(out + 64, value1);
            // Q34a: load only when consumed; do not carry scale across branches
            // or topk iterations. Only the first ScaleVectors data fragments own scale.
            if (j < ScaleVectors)
              scale_dst[static_cast<int64_t>(row) * ScaleVectors + j] = scale_src[token * ScaleVectors + j];
          }
        }
      }
    }
  }
}

void launch_gather(const void* src, at::Tensor dst, int64_t bytes,
                   const int* sorted, const int64_t* bounds, int experts,
                   int topk, int64_t capacity, cudaStream_t stream) {
  if (!capacity || !bytes) return;
  int grid = std::min<int64_t>(capacity, 65535);
  auto* from = static_cast<const uint8_t*>(src);
  auto* to = static_cast<uint8_t*>(dst.data_ptr());
  if (bytes % 16 == 0 && reinterpret_cast<uintptr_t>(from) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(to) % 16 == 0)
    gather_rows<true><<<grid, 128, 0, stream>>>(from, to, sorted, bounds, experts, topk, bytes, capacity);
  else
    gather_rows<false><<<grid, 128, 0, stream>>>(from, to, sorted, bounds, experts, topk, bytes, capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_mx_data_gather(const void* src, at::Tensor dst, int64_t bytes,
                            const int* sorted, const int64_t* bounds, int experts,
                            int topk, int64_t capacity, bool two_wave_rows,
                            const int* inverse, int64_t tokens,
                            cudaStream_t stream) {
  if (!capacity || !bytes) return;
  // Q10's16B-only stride guard regressed at4080B in both paired runs.
  // Q10b qualifies a stricter layout class; this is an alignment hypothesis,
  // not a measured win for every stride in the interval.
  // Q13 mode5 and Q16 packed FP4 passthrough use the same layout guard.
  if (two_wave_rows && bytes >= 2048 && bytes <= 4096 && bytes % 128 == 0 &&
      reinterpret_cast<uintptr_t>(src) % 128 == 0 &&
      reinterpret_cast<uintptr_t>(dst.data_ptr()) % 128 == 0) {
    if (inverse) {
      int grid = std::min<int64_t>((tokens + 1) / 2, 65535);
      scatter_data_tokens2<<<grid, 128, 0, stream>>>(
          static_cast<const uint4*>(src), static_cast<uint4*>(dst.data_ptr()),
          inverse, tokens, topk, bytes / 16, capacity);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    int grid = std::min<int64_t>((capacity + 1) / 2, 65535);
    gather_data_rows2<<<grid, 128, 0, stream>>>(
        static_cast<const uint4*>(src), static_cast<uint4*>(dst.data_ptr()),
        sorted, bounds, experts, topk, bytes / 16, capacity);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  launch_gather(src, dst, bytes, sorted, bounds, experts, topk, capacity, stream);
}

// Eight lanes copy one aligned 128-byte scale row; a CTA owns 16 rows.
// Each row remains contiguous within a physical wave, with no shared state.
__global__ void gather_scale_rows_128(const uint4* src, uint4* dst,
                                     const int* sorted, const int64_t* bounds,
                                     int experts, int topk, int capacity) {
  int lane = threadIdx.x & 7;
  int row_in_tile = threadIdx.x >> 3;
  for (int64_t tile = static_cast<int64_t>(blockIdx.x) * 16;
       tile < capacity; tile += static_cast<int64_t>(gridDim.x) * 16) {
    int64_t row = tile + row_in_tile;
    if (row < capacity && row < bounds[experts]) {
      int token = sorted[row] / topk;
      dst[row * 8 + lane] = src[static_cast<int64_t>(token) * 8 + lane];
    }
  }
}

void launch_mx_scale_gather(const void* src, at::Tensor dst, int64_t bytes,
                            const int* sorted, const int64_t* bounds, int experts,
                            int topk, int64_t capacity, bool grouped_scale_rows,
                            cudaStream_t stream) {
  // Q15: mode5's dense FP32 scale row is also a128-byte payload at32 groups.
  // Copy bytes with the existing kernel; no dtype-dependent arithmetic.
  if (!grouped_scale_rows || bytes != 128 ||
      reinterpret_cast<uintptr_t>(src) % 16 != 0 ||
      reinterpret_cast<uintptr_t>(dst.data_ptr()) % 16 != 0) {
    launch_gather(src, dst, bytes, sorted, bounds, experts, topk, capacity, stream);
    return;
  }
  if (!capacity) return;
  int grid = std::min<int64_t>((capacity + 15) / 16, 65535);
  gather_scale_rows_128<<<grid, 128, 0, stream>>>(
      static_cast<const uint4*>(src), static_cast<uint4*>(dst.data_ptr()),
      sorted, bounds, experts, topk, capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_output(const at::Tensor& out, const at::Tensor& x, at::ScalarType dtype,
                   const std::vector<int64_t>& shape, const char* name) {
  tensor_check(out, x, name);
  TORCH_CHECK(out.scalar_type() == dtype && out.sizes().vec() == shape,
              name, " has incorrect dtype or shape");
}

void launch_active_gather(const void* src, at::Tensor dst, int64_t bytes,
                           const int* sorted, const int64_t* info, int topk,
                           int64_t capacity, cudaStream_t stream) {
  if (!capacity || !bytes) return;
  int grid = std::min<int64_t>(capacity, 65535);
  auto* from = static_cast<const uint8_t*>(src);
  auto* to = static_cast<uint8_t*>(dst.data_ptr());
  if (bytes % 16 == 0 && reinterpret_cast<uintptr_t>(from) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(to) % 16 == 0)
    gather_rows_active<true><<<grid, 128, 0, stream>>>(from, to, sorted, info, topk, bytes);
  else
    gather_rows_active<false><<<grid, 128, 0, stream>>>(from, to, sorted, info, topk, bytes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Initial Tensor implementation deliberately uses the qualified generic
// quantize-once/gather pipeline. Legacy int callers retain the Q49 dispatch.
void run_active_out(const at::Tensor& x, const at::Tensor& ids,
                    at::Tensor data, at::Tensor row_idx, at::Tensor counts,
                    at::Tensor scales, at::Tensor weights, at::Tensor workspace,
                    const at::Tensor& active, at::Tensor info,
                    const Layout& l, const Scratch& s, int experts, int mode,
                    int row_type, int count_type,
                    const std::optional<at::Tensor>& scale,
                    const std::optional<at::Tensor>& weight, cudaStream_t stream) {
  auto* base = workspace.data_ptr<uint8_t>();
  auto* keys = reinterpret_cast<int*>(base);
  auto* sorted_keys = reinterpret_cast<int*>(base + s.array_bytes);
  auto* positions = reinterpret_cast<int*>(base + 2 * s.array_bytes);
  auto* sorted = reinterpret_cast<int*>(base + 3 * s.array_bytes);
  auto* bounds = reinterpret_cast<int64_t*>(base + s.bounds_offset);
  auto* control = info.data_ptr<int64_t>();
  int grid = (l.routes + kThreads - 1) / kThreads;
  if (l.routes) {
    make_keys<<<grid, kThreads, 0, stream>>>(ids.data_ptr<int>(), keys, positions,
        l.routes, l.start, l.end, experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    size_t sort_bytes = s.sort_bytes;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(base + s.sort_offset, sort_bytes,
        keys, sorted_keys, positions, sorted, static_cast<int>(l.routes),
        0, routing_sort_end_bit(experts), stream));
    expert_bounds<<<(experts + 1 + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        sorted_keys, bounds, l.routes, experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  resolve_active_rows<<<1, 1, 0, stream>>>(active.data_ptr<int64_t>(), bounds,
      control, l.routes, experts, l.capacity);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (!l.routes) {
    if (counts.numel()) C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr(), 0, counts.nbytes(), stream));
    return;
  }
  if (counts.numel()) {
    write_counts<<<(l.end - l.start + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        bounds, counts.data_ptr<int64_t>(), l.start, l.end, count_type);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  C10_CUDA_CHECK(cudaMemsetAsync(row_idx.data_ptr(), 0xff, row_idx.nbytes(), stream));
  write_indices_active<<<grid, kThreads, 0, stream>>>(sorted, control, row_idx.data_ptr<int>(),
      weight ? weight->data_ptr<float>() : nullptr,
      weight ? weights.data_ptr<float>() : nullptr, l.routes, row_type);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (mode == -1) {
    launch_active_gather(x.data_ptr(), data, l.row_bytes, sorted, control,
        ids.size(1), l.capacity, stream);
    if (scale) launch_active_gather(scale->data_ptr(), scales, l.scale_bytes,
        sorted, control, ids.size(1), l.capacity, stream);
  } else if (l.hidden && l.capacity) {
    auto* quant = base + s.quant_offset;
    auto* quant_scales = base + s.scale_offset;
    int group_size = mode == 4 || mode == 5 ? 128 : 32;
    int64_t groups = (l.hidden + group_size - 1) / group_size;
    int quant_grid = std::min<int64_t>(l.tokens * groups, 65535);
    // Includes the unused byte of an odd MX group count. The generic quantizer
    // overwrites all real scales; padding must be deterministic as in Q49.
    C10_CUDA_CHECK(cudaMemsetAsync(quant_scales, 0, l.tokens * l.scale_bytes, stream));
    if (x.scalar_type() == at::kBFloat16)
      quantize_groups<<<quant_grid, 128, 0, stream>>>(x.data_ptr<at::BFloat16>(), quant,
          quant_scales, l.tokens * groups, l.hidden, groups, l.scale_bytes, mode);
    else
      quantize_groups<<<quant_grid, 128, 0, stream>>>(x.data_ptr<at::Half>(), quant,
          quant_scales, l.tokens * groups, l.hidden, groups, l.scale_bytes, mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_active_gather(quant, data, l.row_bytes, sorted, control,
        ids.size(1), l.capacity, stream);
    launch_active_gather(quant_scales, scales, l.scale_bytes, sorted, control,
        ids.size(1), l.capacity, stream);
  }
}

}  // namespace

int64_t moe_init_routing_v4_workspace_size(const at::Tensor& x, const at::Tensor& ids,
                                          int64_t experts, int64_t mode) {
  auto l = layout(x, ids, experts, mode, 0, 1, true, 0, {}, std::nullopt, std::nullopt);
  const c10::cuda::CUDAGuard guard(x.device());
  return scratch(l, experts, mode, at::cuda::getCurrentCUDAStream(),
                 use_inverse_sort(x, l)).bytes;
}

void moe_init_routing_v4_out(
    const at::Tensor& x, const at::Tensor& ids, at::Tensor data, at::Tensor row_idx,
    at::Tensor counts, at::Tensor scales, at::Tensor weights, at::Tensor workspace,
    int64_t experts, int64_t mode, int64_t row_type, int64_t count_type,
    bool count_flag, int64_t active, const std::vector<int64_t>& range,
    const std::optional<at::Tensor>& scale, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& active_tensor,
    const std::optional<at::Tensor>& active_info) {
  TORCH_CHECK(!active_tensor || active == 0,
              "active_num and active_num_optional cannot both be specified");
  TORCH_CHECK(active_tensor || !active_info,
              "active_info requires active_num_optional");
  Layout l = layout(x, ids, experts, mode, row_type, count_type, count_flag, active, range, scale, weight);
  const c10::cuda::CUDAGuard guard(x.device());
  auto stream = at::cuda::getCurrentCUDAStream();
  Scratch s = scratch(l, experts, mode, stream, use_inverse_sort(x, l));
  if (active_tensor) {
    check_active_tensor(*active_tensor, x);
    TORCH_CHECK(active_info.has_value(),
                "Tensor out calls require active_info (GPU INT64 [4])");
    check_output(*active_info, x, at::kLong, {4}, "active_info");
    TORCH_CHECK(data.dim() == 2 && data.size(0) <= l.routes,
                "Tensor out storage capacity must be in [0, number of routes]");
    set_capacity(l, data.size(0), mode, scale.has_value(), weight.has_value());
  }
  check_output(data, x, l.dtype, l.data_shape, "expanded_x");
  check_output(row_idx, x, at::kInt, {l.routes}, "expanded_row_idx");
  check_output(counts, x, at::kLong, l.count_shape, "expert_tokens");
  check_output(scales, x, l.scale_dtype, l.scale_shape, "expanded_scale");
  check_output(weights, x, at::kFloat, l.weight_shape, "expanded_topk_weight");
  tensor_check(workspace, x, "workspace");
  TORCH_CHECK(workspace.scalar_type() == at::kByte && workspace.dim() == 1 &&
              workspace.numel() >= s.bytes && reinterpret_cast<uintptr_t>(workspace.data_ptr()) % kAlignment == 0,
              "workspace must be an aligned uint8 vector with at least ", s.bytes, " bytes");
  std::vector<at::Tensor> inputs{x, ids};
  if (scale) inputs.push_back(*scale);
  if (weight) inputs.push_back(*weight);
  if (active_tensor) inputs.push_back(*active_tensor);
  std::vector<at::Tensor> outputs{data, row_idx, counts, scales, weights, workspace};
  if (active_info) outputs.push_back(*active_info);
  for (size_t i = 0; i < outputs.size(); ++i) {
    for (const auto& input : inputs) at::assert_no_overlap(outputs[i], input);
    for (size_t j = 0; j < i; ++j) at::assert_no_overlap(outputs[i], outputs[j]);
  }
  if (active_tensor) {
    run_active_out(x, ids, data, row_idx, counts, scales, weights, workspace,
        *active_tensor, *active_info, l, s, experts, mode, row_type, count_type,
        scale, weight, stream);
    return;
  }
  if (!l.routes) {
    if (counts.numel()) C10_CUDA_CHECK(cudaMemsetAsync(counts.data_ptr(), 0, counts.nbytes(), stream));
    return;
  }
  auto* base = workspace.data_ptr<uint8_t>();
  auto* keys = reinterpret_cast<int*>(base);
  auto* sorted_keys = reinterpret_cast<int*>(base + s.array_bytes);
  auto* positions = reinterpret_cast<int*>(base + 2 * s.array_bytes);
  auto* sorted = reinterpret_cast<int*>(base + 3 * s.array_bytes);
  auto* bounds = reinterpret_cast<int64_t*>(base + s.bounds_offset);
  int grid = (l.routes + kThreads - 1) / kThreads;
  make_keys<<<grid, kThreads, 0, stream>>>(ids.data_ptr<int>(), keys, positions, l.routes, l.start, l.end, experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  size_t sort_bytes = s.sort_bytes;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(base + s.sort_offset, sort_bytes,
      keys, sorted_keys, positions, sorted, static_cast<int>(l.routes), 0, routing_sort_end_bit(experts), stream));
  expert_bounds<<<(experts + 1 + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      sorted_keys, bounds, l.routes, experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (counts.numel()) {
    write_counts<<<(l.end - l.start + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        bounds, counts.data_ptr<int64_t>(), l.start, l.end, count_type);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  bool inverse_sort = s.inverse_sort && row_type == 0;
  // Q30: the inverse SortPairs below overwrites every row_idx entry, including
  // invalid ranks. With no weights, neither this initialization nor the
  // intermediate index write has a consumer. Keep the weighted/fallback path.
  if (!inverse_sort || weight.has_value()) {
    C10_CUDA_CHECK(cudaMemsetAsync(row_idx.data_ptr(), 0xff, row_idx.nbytes(), stream));
    write_indices<<<grid, kThreads, 0, stream>>>(sorted, bounds, row_idx.data_ptr<int>(),
        weight ? weight->data_ptr<float>() : nullptr, weight ? weights.data_ptr<float>() : nullptr,
        l.routes, experts, l.capacity, inverse_sort ? 1 : row_type);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (inverse_sort) {
    invalidate_inverse_ranks<<<grid, kThreads, 0, stream>>>(positions, bounds, l.routes, experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // Keys/sorted expert IDs are dead after bounds/counts. Keep the full sorted
    // original-ID array unchanged: data, scale and weight routing still use it.
    size_t inverse_bytes = s.sort_bytes;
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(base + s.sort_offset, inverse_bytes,
        sorted, keys, positions, row_idx.data_ptr<int>(), static_cast<int>(l.routes),
        0, inverse_sort_end_bit(l.routes), stream));
  }
  if (mode == -1) {
    bool fp4_rows2 = false;
    if (x.scalar_type() == at::ScalarType::Float4_e2m1fn_x2) {
      const auto* props = at::cuda::getDeviceProperties(x.get_device());
      fp4_rows2 = props->warpSize == 64 && std::strcmp(props->name, "MetaX C600-U") == 0;
    }
    // Q25: reuse the existing token-major scatter for wide aligned BF16 rows.
    bool bf16_tokens2 = false;
    if (x.scalar_type() == at::kBFloat16 && row_type == 0 && l.capacity &&
        l.row_bytes >= 4096 && l.row_bytes <= 8192 && l.row_bytes % 128 == 0 &&
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 128 == 0 &&
        reinterpret_cast<uintptr_t>(data.data_ptr()) % 128 == 0) {
      const auto* props = at::cuda::getDeviceProperties(x.get_device());
      bf16_tokens2 = props->warpSize == 64 && std::strcmp(props->name, "MetaX C600-U") == 0;
    }
    if (bf16_tokens2) {
      int grid = std::min<int64_t>((l.tokens + 1) / 2, 65535);
      scatter_data_tokens2<<<grid, 128, 0, stream>>>(
          static_cast<const uint4*>(x.data_ptr()), static_cast<uint4*>(data.data_ptr()),
          row_idx.data_ptr<int>(), l.tokens, ids.size(1), l.row_bytes / 16, l.capacity);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (fp4_rows2)
      launch_mx_data_gather(x.data_ptr(), data, l.row_bytes, sorted, bounds, experts,
                            ids.size(1), l.capacity, true,
                            row_type == 0 ? row_idx.data_ptr<int>() : nullptr, l.tokens, stream);
    else
      launch_gather(x.data_ptr(), data, l.row_bytes, sorted, bounds, experts, ids.size(1), l.capacity, stream);
    if (scale) launch_gather(scale->data_ptr(), scales, l.scale_bytes, sorted, bounds,
                             experts, ids.size(1), l.capacity, stream);
  } else if (l.hidden && l.capacity) {
    auto* quant = base + s.quant_offset;
    auto* quant_scales = base + s.scale_offset;
    int group_size = mode == 4 || mode == 5 ? 128 : 32;
    int64_t groups = (l.hidden + group_size - 1) / group_size;
    int quant_grid = std::min<int64_t>(l.tokens * groups, 65535);
    bool mx_subgroups = false;
    bool fp8_warp_groups = false;
    if (mode == 3 || mode == 9 || mode == 5) {
      const auto* props = at::cuda::getDeviceProperties(x.get_device());
      bool c600u = std::strcmp(props->name, "MetaX C600-U") == 0;
      mx_subgroups = c600u && (mode == 3 || mode == 9);
      fp8_warp_groups = c600u && mode == 5;
    }
    // Q41a: isolate MXFP4 device code; retain Q41's complete-tile dispatch.
    bool direct_mx4_scatter = mx_subgroups && mode == 9 && row_type == 0 &&
        x.scalar_type() == at::kBFloat16 &&
        at::cuda::getDeviceProperties(x.get_device())->warpSize == 64 &&
        l.hidden >= 4096 && l.hidden <= 8192 && l.hidden % 4096 == 0 &&
        groups == l.scale_bytes &&
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(data.data_ptr()) % 128 == 0;
    if (direct_mx4_scatter) {
      mcoplib::moe_routing_detail::launch_mx4_quantize_scatter(
          x.data_ptr<at::BFloat16>(), data.data_ptr(),
          static_cast<uint8_t*>(scales.data_ptr()), row_idx.data_ptr<int>(),
          l.tokens, ids.size(1), l.hidden, l.capacity, stream);
      return;
    }
    // Q40: BF16 MXFP8 uses the same complete2048B distribution tiles as Q39.
    bool direct_mx8_scatter = mx_subgroups && mode == 3 && row_type == 0 &&
        x.scalar_type() == at::kBFloat16 &&
        at::cuda::getDeviceProperties(x.get_device())->warpSize == 64 &&
        l.hidden >= 2048 && l.hidden <= 4096 && l.hidden % 2048 == 0 &&
        groups == l.scale_bytes &&
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(data.data_ptr()) % 128 == 0;
    if (direct_mx8_scatter) {
      int direct_grid = std::min<int64_t>((l.tokens + 1) / 2, 65535);
      if (l.hidden == 4096 && ids.size(1) > 0 && ids.size(1) <= 64) {
        quantize_scatter_mx8_rankwave_tokens2<<<direct_grid, 128, 0, stream>>>(
            x.data_ptr<at::BFloat16>(), static_cast<uint4*>(data.data_ptr()),
            static_cast<uint8_t*>(scales.data_ptr()), row_idx.data_ptr<int>(),
            l.tokens, ids.size(1), l.hidden, l.capacity);
      } else {
        quantize_scatter_mx8_tokens2<<<direct_grid, 128, 0, stream>>>(
            x.data_ptr<at::BFloat16>(), static_cast<uint4*>(data.data_ptr()),
            static_cast<uint8_t*>(scales.data_ptr()), row_idx.data_ptr<int>(),
            l.tokens, ids.size(1), l.hidden, l.capacity);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    // Q39: complete2048B encoded tiles match paired-fragment scatter.
    // Keep all other layouts, Half input and forward-index routing on Q37a.
    bool direct_fp8_scatter = fp8_warp_groups && row_type == 0 &&
        x.scalar_type() == at::kBFloat16 &&
        at::cuda::getDeviceProperties(x.get_device())->warpSize == 64 &&
        l.hidden >= 2048 && l.hidden <= 4096 && l.hidden % 2048 == 0 &&
        reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(data.data_ptr()) % 128 == 0 &&
        reinterpret_cast<uintptr_t>(scales.data_ptr()) % alignof(float) == 0;
    if (direct_fp8_scatter) {
      int direct_grid = std::min<int64_t>((l.tokens + 1) / 2, 65535);
      quantize_scatter_fp8_tokens2<<<direct_grid, 128, 0, stream>>>(
          x.data_ptr<at::BFloat16>(), static_cast<uint4*>(data.data_ptr()),
          scales.data_ptr<float>(), row_idx.data_ptr<int>(), l.tokens,
          ids.size(1), l.hidden, l.capacity);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return;
    }
    // Q31: these quantizers overwrite every FP32 scale or MX group byte.
    // An odd number of MX groups leaves one padding byte per token untouched.
    bool scale_fully_written = fp8_warp_groups ||
        (mx_subgroups && groups == l.scale_bytes);
    if (!scale_fully_written)
      C10_CUDA_CHECK(cudaMemsetAsync(quant_scales, 0, l.tokens * l.scale_bytes, stream));
    if (fp8_warp_groups && x.scalar_type() == at::kBFloat16)
      launch_fp8_warp_groups(x.data_ptr<at::BFloat16>(), quant, quant_scales,
          l.tokens * groups, l.hidden, groups, l.scale_bytes, stream);
    else if (fp8_warp_groups)
      launch_fp8_warp_groups(x.data_ptr<at::Half>(), quant, quant_scales,
          l.tokens * groups, l.hidden, groups, l.scale_bytes, stream);
    else if (mx_subgroups && x.scalar_type() == at::kBFloat16)
      launch_mx_subgroups(x.data_ptr<at::BFloat16>(), quant, quant_scales,
          l.tokens * groups, l.hidden, groups, l.scale_bytes, mode, stream);
    else if (mx_subgroups)
      launch_mx_subgroups(x.data_ptr<at::Half>(), quant, quant_scales,
          l.tokens * groups, l.hidden, groups, l.scale_bytes, mode, stream);
    else if (x.scalar_type() == at::kBFloat16)
      quantize_groups<<<quant_grid, 128, 0, stream>>>(x.data_ptr<at::BFloat16>(), quant, quant_scales,
          l.tokens * groups, l.hidden, groups, l.scale_bytes, mode);
    else
      quantize_groups<<<quant_grid, 128, 0, stream>>>(x.data_ptr<at::Half>(), quant, quant_scales,
          l.tokens * groups, l.hidden, groups, l.scale_bytes, mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    bool two_wave_rows = (mx_subgroups || fp8_warp_groups) &&
        at::cuda::getDeviceProperties(x.get_device())->warpSize == 64;
    // Intersect the existing data-scatter layout with complete 64/128/256B scales.
    bool fused_data_scales = two_wave_rows &&
        l.row_bytes >= 2048 && l.row_bytes <= 4096 && l.row_bytes % 128 == 0 &&
        reinterpret_cast<uintptr_t>(quant) % 128 == 0 &&
        reinterpret_cast<uintptr_t>(data.data_ptr()) % 128 == 0 &&
        (l.scale_bytes == 64 || l.scale_bytes == 128 || l.scale_bytes == 256) &&
        reinterpret_cast<uintptr_t>(quant_scales) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(scales.data_ptr()) % 16 == 0;
    if (fused_data_scales && row_type == 1 &&
        (l.scale_bytes == 64 || l.scale_bytes == 256)) {
      int fused_grid = std::min<int64_t>((l.capacity + 1) / 2, 65535);
      gather_data_scales_rows2<<<fused_grid, 128, 0, stream>>>(
          reinterpret_cast<const uint4*>(quant), static_cast<uint4*>(data.data_ptr()),
          reinterpret_cast<const uint4*>(quant_scales),
          static_cast<uint4*>(scales.data_ptr()), sorted, bounds, experts,
          ids.size(1), l.row_bytes / 16, l.scale_bytes / 16, l.capacity);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else if (fused_data_scales && row_type == 0) {
      int fused_grid = std::min<int64_t>((l.tokens + 1) / 2, 65535);
      if (l.scale_bytes == 128) {
        scatter_data_scales_tokens2<<<fused_grid, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(quant), static_cast<uint4*>(data.data_ptr()),
            reinterpret_cast<const uint4*>(quant_scales),
            static_cast<uint4*>(scales.data_ptr()), row_idx.data_ptr<int>(),
            l.tokens, ids.size(1), l.row_bytes / 16, l.capacity);
      } else if (l.scale_bytes == 64) {
        scatter_data_scales_tokens2_width<4><<<fused_grid, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(quant), static_cast<uint4*>(data.data_ptr()),
            reinterpret_cast<const uint4*>(quant_scales),
            static_cast<uint4*>(scales.data_ptr()), row_idx.data_ptr<int>(),
            l.tokens, ids.size(1), l.row_bytes / 16, l.capacity);
      } else {
        scatter_data_scales_tokens2_width<16><<<fused_grid, 128, 0, stream>>>(
            reinterpret_cast<const uint4*>(quant), static_cast<uint4*>(data.data_ptr()),
            reinterpret_cast<const uint4*>(quant_scales),
            static_cast<uint4*>(scales.data_ptr()), row_idx.data_ptr<int>(),
            l.tokens, ids.size(1), l.row_bytes / 16, l.capacity);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    } else {
      launch_mx_data_gather(quant, data, l.row_bytes, sorted, bounds, experts,
          ids.size(1), l.capacity, two_wave_rows,
          row_type == 0 ? row_idx.data_ptr<int>() : nullptr, l.tokens, stream);
      launch_mx_scale_gather(quant_scales, scales, l.scale_bytes, sorted, bounds,
                             experts, ids.size(1), l.capacity,
                             mx_subgroups || fp8_warp_groups, stream);
    }
  }
}

std::vector<at::Tensor> moe_init_routing_v4(
    const at::Tensor& x, const at::Tensor& ids, int64_t experts, int64_t mode,
    int64_t row_type, int64_t count_type, bool count_flag, int64_t active,
    const std::vector<int64_t>& range,
    const std::optional<at::Tensor>& scale, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& active_tensor,
    const std::optional<at::Tensor>& active_info) {
  TORCH_CHECK(!active_tensor || active == 0,
              "active_num and active_num_optional cannot both be specified");
  TORCH_CHECK(active_tensor || !active_info,
              "active_info requires active_num_optional");
  Layout l = layout(x, ids, experts, mode, row_type, count_type, count_flag, active, range, scale, weight);
  std::optional<at::Tensor> info = active_info;
  if (active_tensor) {
    check_active_tensor(*active_tensor, x);
    const c10::cuda::CUDAGuard guard(x.device());
    cudaStreamCaptureStatus capture_status;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream(), &capture_status));
    TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
                "Tensor allocating API requires a host read for exact output shape; "
                "use moe_init_routing_v4_out with fixed storage and active_info in a graph");
    int64_t value = active_tensor->item<int64_t>();
    TORCH_CHECK(value >= 0, "active_num_optional must be nonnegative; 0 means all routes");
    set_capacity(l, value == 0 ? l.routes : std::min(value, l.routes), mode,
        scale.has_value(), weight.has_value());
    if (!info) info = at::empty({4}, x.options().dtype(at::kLong));
  }
  auto data = at::empty(l.data_shape, x.options().dtype(l.dtype));
  auto rows = at::empty({l.routes}, x.options().dtype(at::kInt));
  auto counts = at::empty(l.count_shape, x.options().dtype(at::kLong));
  auto scales = at::empty(l.scale_shape, x.options().dtype(l.scale_dtype));
  auto weights = at::empty(l.weight_shape, x.options().dtype(at::kFloat));
  auto workspace = at::empty({moe_init_routing_v4_workspace_size(x, ids, experts, mode)}, x.options().dtype(at::kByte));
  moe_init_routing_v4_out(x, ids, data, rows, counts, scales, weights, workspace,
      experts, mode, row_type, count_type, count_flag, active, range, scale, weight,
      active_tensor, info);
  return {data, rows, counts, scales, weights};
}
