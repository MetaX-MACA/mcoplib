#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "../include/indexer_norm_rope.h"
#include "../kernel/utils.cuh"
#include "../kernel/utils.h"

#ifndef __shfl_xor_sync_16
#define __shfl_xor_sync_16(mask, value, offset) __shfl_xor_sync(mask, value, offset, 16)
#endif

namespace {
constexpr int kHeadDim = 256;
constexpr int kRotaryDim = 16;  // Half-width; rotate dimensions [0, 32).
constexpr int kThreadsPerHead = 8;
constexpr int kHeadsPerBlock = 16;
constexpr int kElemsPerLane = kHeadDim / kThreadsPerHead;
constexpr int kChunksPerLane = kElemsPerLane / 8;

template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedVector { T data[N]; };

__device__ __forceinline__ float reduce_head_sum(float value) {
#pragma unroll
  for (int offset = kThreadsPerHead / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffffffffffULL, value, offset,
                             kThreadsPerHead);
  }
  return value;
}


enum class KernelMode { kFused, kQOnly, kKOnly };

template <KernelMode kMode, bool kVectorizedRope = false,
          bool kPackedRopeValues = false, bool kDirectQH16Mapping = false,
          bool kDirectQH4FusedMapping = false>
__global__ void indexer_norm_rope_kernel(
    bfloat16 const* __restrict__ q, bfloat16 const* __restrict__ k,
    bfloat16* __restrict__ q_out, bfloat16* __restrict__ k_out,
    bfloat16 const* __restrict__ q_weight,
    bfloat16 const* __restrict__ k_weight,
    bfloat16 const* __restrict__ k_bias,
    bfloat16 const* __restrict__ cos, bfloat16 const* __restrict__ sin,
    int32_t const* __restrict__ positions, int64_t q_stride,
    int64_t k_stride, int64_t q_out_stride, int64_t k_out_stride,
    int64_t cos_stride, int64_t num_tokens, int64_t num_q_heads,
    int64_t num_k_heads, float eps, float q_weight_bias) {
  int const local_head = threadIdx.x / kThreadsPerHead;
  int const lane = threadIdx.x & (kThreadsPerHead - 1);
  constexpr bool kFused = kMode == KernelMode::kFused;
  constexpr bool kQOnly = kMode == KernelMode::kQOnly;
  static_assert(!kDirectQH16Mapping || kQOnly);
  static_assert(!kDirectQH4FusedMapping || kFused);
  static_assert(!(kDirectQH16Mapping && kDirectQH4FusedMapping));
  int64_t token, head;
  bool is_q;
  if constexpr (kDirectQH4FusedMapping) {
    if (local_head >= 15) return;
    token = static_cast<int64_t>(blockIdx.x) * 3 + local_head / 5;
    if (token >= num_tokens) return;
    int const slot = local_head % 5;
    is_q = slot < 4;
    head = is_q ? slot : 0;
  } else if constexpr (kDirectQH16Mapping) {
    token = static_cast<int64_t>(blockIdx.x);
    head = local_head;
    is_q = true;
  } else {
    int64_t const heads_per_token =
        kFused ? num_q_heads + num_k_heads
               : (kQOnly ? num_q_heads : num_k_heads);
    int const heads_per_block = blockDim.x / kThreadsPerHead;
    int64_t const task =
        static_cast<int64_t>(blockIdx.x) * heads_per_block + local_head;
    if (task >= num_tokens * heads_per_token) return;
    token = task / heads_per_token;
    int64_t const slot = task - token * heads_per_token;
    is_q = kFused ? slot < num_q_heads : kQOnly;
    head = kFused ? (is_q ? slot : slot - num_q_heads) : slot;
  }
  bfloat16 const* src = (is_q ? q + token * q_stride : k + token * k_stride)
                          + head * kHeadDim;
  bfloat16* dst = (is_q ? q_out + token * q_out_stride
                        : k_out + token * k_out_stride) + head * kHeadDim;

  // Four explicit 16-byte input transactions per lane (32 BF16 values).
  using Vec16B = AlignedVector<bfloat16, 8>;
  Vec16B input[kChunksPerLane];
#pragma unroll
  for (int chunk = 0; chunk < kChunksPerLane; ++chunk) {
    int const vector_offset = (chunk * kThreadsPerHead + lane) * 8;
    ldg_b128_reg_async(cast_b128(&input[chunk])[0],
        const_cast<bfloat16*>(src + vector_offset), true, true);
  }

  // Overlap the outstanding input reads with independent parameter reads.
  bfloat16 const* const selected_weight = is_q ? q_weight : k_weight;
  Vec16B weight[kChunksPerLane], bias[kChunksPerLane];
#pragma unroll
  for (int chunk = 0; chunk < kChunksPerLane; ++chunk) {
    int const vector_offset = (chunk * kThreadsPerHead + lane) * 8;
    if constexpr (kQOnly) {
      if ((local_head & 7) == 0) {
        ldg_b128_reg_async(cast_b128(&weight[chunk])[0],
            const_cast<bfloat16*>(q_weight + vector_offset), true, true);
      }
    } else {
      ldg_b128_reg_async(cast_b128(&weight[chunk])[0],
          const_cast<bfloat16*>(selected_weight + vector_offset), true, true);
      if (!is_q) {
        bias[chunk] = *reinterpret_cast<Vec16B const*>(k_bias + vector_offset);
      }
    }
  }
  int32_t const position = positions[token];
  // The first four outstanding requests are input. Keep the four later
  // weight requests in flight while the input-only reduction executes.
  if constexpr (kQOnly) {
    if ((local_head & 7) == 0) arrive_gvmcnt(4);
    else arrive_gvmcnt(0);
  } else {
    arrive_gvmcnt(4);
  }
  float values[kElemsPerLane];
  float sum = 0.0f, sum_sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    bfloat16 const value = input[i / 8].data[i & 7];
    values[i] = target_to_float<bfloat16>(value);
    if constexpr (!kQOnly) sum += values[i];
    sum_sq += values[i] * values[i];
  }
  if constexpr (!kQOnly) sum = reduce_head_sum(sum);
  sum_sq = reduce_head_sum(sum_sq);
  float const mean = is_q ? 0.0f : sum / static_cast<float>(kHeadDim);
  // Match the official deployed LayerNorm reduction: E[x^2] - E[x]^2.
  float const variance = sum_sq / static_cast<float>(kHeadDim) -
                         (is_q ? 0.0f : mean * mean);
  float const rstd = rsqrtf(variance + eps);

  // Weight is first consumed by the affine normalization below.
  arrive_gvmcnt(0);
  if constexpr (kQOnly) {
#pragma unroll
    for (int chunk = 0; chunk < kChunksPerLane; ++chunk) {
      int32_t* words = reinterpret_cast<int32_t*>(&weight[chunk]);
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        words[word] = __shfl_sync(0xffffffffffffffffULL, words[word], lane, 64);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    bfloat16 const wi = weight[i / 8].data[i & 7];
    float const w = target_to_float<bfloat16>(wi);
    values[i] = (values[i] - mean) * rstd *
                (w + (is_q ? q_weight_bias : 0.0f));
    if (!is_q) {
      bfloat16 const kbi = bias[i / 8].data[i & 7];
      values[i] += target_to_float<bfloat16>(kbi);
    }
  }

  // Dims [0, 32) occupy lanes 0..3 in chunk 0.
  if constexpr (kVectorizedRope) {
    // Lanes 0 and 1 load the two contiguous halves of the RoPE row;
    // xor-2 supplies the same packed parameters to lanes 2 and 3.
    Vec16B rope_cos{}, rope_sin{};
    if (lane < 2) {
      int const rotary_offset = lane * 8;
      rope_cos = *reinterpret_cast<Vec16B const*>(
          cos + position * cos_stride + rotary_offset);
      rope_sin = *reinterpret_cast<Vec16B const*>(
          sin + position * cos_stride + rotary_offset);
    }
#pragma unroll
    for (int word = 0; word < 4; ++word) {
      int32_t cos_word = reinterpret_cast<int32_t*>(&rope_cos)[word];
      int32_t sin_word = reinterpret_cast<int32_t*>(&rope_sin)[word];
      int32_t const partner_cos_word = __shfl_xor_sync(
          0xffffffffffffffffULL, cos_word, 2, kThreadsPerHead);
      int32_t const partner_sin_word = __shfl_xor_sync(
          0xffffffffffffffffULL, sin_word, 2, kThreadsPerHead);
      if (lane >= 2) {
        cos_word = partner_cos_word;
        sin_word = partner_sin_word;
      }
      bfloat16 const* const cos_pair =
          reinterpret_cast<bfloat16 const*>(&cos_word);
      bfloat16 const* const sin_pair =
          reinterpret_cast<bfloat16 const*>(&sin_word);
      if constexpr (kPackedRopeValues) {
        int const i = word * 2;
        bfloat162 const local_bf16 = convert_2fp32_to_2fp16_rn<bfloat162>(
            make_float2(values[i], values[i + 1]));
        int32_t const local_word =
            *reinterpret_cast<int32_t const*>(&local_bf16);
        int32_t const partner_word = __shfl_xor_sync(
            0xffffffffffffffffULL, local_word, 2, kThreadsPerHead);
        bfloat162 const partner_bf16 =
            *reinterpret_cast<bfloat162 const*>(&partner_word);
        float2 const local = convert_2fp16_to_2fp32(local_bf16);
        float2 const partner = convert_2fp16_to_2fp32(partner_bf16);
        if (lane < 4) {
          float const c0 = target_to_float<bfloat16>(cos_pair[0]);
          float const s0 = target_to_float<bfloat16>(sin_pair[0]);
          float const c1 = target_to_float<bfloat16>(cos_pair[1]);
          float const s1 = target_to_float<bfloat16>(sin_pair[1]);
          values[i] = lane < 2 ? local.x * c0 - partner.x * s0
                               : partner.x * s0 + local.x * c0;
          values[i + 1] = lane < 2 ? local.y * c1 - partner.y * s1
                                   : partner.y * s1 + local.y * c1;
        }
      } else {
#pragma unroll
        for (int item = 0; item < 2; ++item) {
          int const i = word * 2 + item;
          float const local = target_to_float<bfloat16>(
              convert_fp32_to_fp16_rn<bfloat16>(values[i]));
          float const partner = __shfl_xor_sync(
              0xffffffffffffffffULL, local, 2, kThreadsPerHead);
          if (lane < 4) {
            float const c = target_to_float<bfloat16>(cos_pair[item]);
            float const s = target_to_float<bfloat16>(sin_pair[item]);
            values[i] = lane < 2 ? local * c - partner * s
                                 : partner * s + local * c;
          }
        }
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      float const local = target_to_float<bfloat16>(
          convert_fp32_to_fp16_rn<bfloat16>(values[i]));
      float const partner = __shfl_xor_sync(
          0xffffffffffffffffULL, local, 2, kThreadsPerHead);
      if (lane < 4) {
        int const rotary_index = (lane & 1) * 8 + i;
        float const c = target_to_float<bfloat16>(
            cos[position * cos_stride + rotary_index]);
        float const s = target_to_float<bfloat16>(
            sin[position * cos_stride + rotary_index]);
        values[i] = lane < 2 ? local * c - partner * s
                             : partner * s + local * c;
      }
    }
  }

  Vec16B output[kChunksPerLane];
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    output[i / 8].data[i & 7] = convert_fp32_to_fp16_rn<bfloat16>(values[i]);
  }
#pragma unroll
  for (int chunk = 0; chunk < kChunksPerLane; ++chunk) {
    int const vector_offset = (chunk * kThreadsPerHead + lane) * 8;
    *reinterpret_cast<Vec16B*>(dst + vector_offset) = output[chunk];
  }
}

struct LogicalShape { int64_t tokens; int64_t heads; };

LogicalShape validate_input(torch::Tensor const& x, char const* name,
                            int64_t head_dim) {
  TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16,
              name, " must be bfloat16");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.dim() == 2 || x.dim() == 3,
              name, " must be [T, H*D] or [T, H, D]");
  if (x.dim() == 3) {
    TORCH_CHECK(x.size(2) == head_dim, name, ".size(2) must equal head_dim");
    return {x.size(0), x.size(1)};
  }
  TORCH_CHECK(x.size(1) % head_dim == 0,
              name, ".size(1) must be divisible by head_dim");
  return {x.size(0), x.size(1) / head_dim};
}

void validate_common(torch::Tensor const& qw, torch::Tensor const& kw,
                     torch::Tensor const& kb, torch::Tensor const& cos,
                     torch::Tensor const& sin, torch::Tensor const& positions,
                     torch::Device device, int64_t tokens, int64_t head_dim,
                     int64_t rotary_dim, double eps) {
  TORCH_CHECK(head_dim == kHeadDim, "only head_dim=256 is currently supported");
  TORCH_CHECK(rotary_dim == kRotaryDim,
              "only rotary_dim=16 (32 rotated dimensions) is currently supported");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");
  for (auto const* x : {&qw, &kw, &kb, &cos, &sin}) {
    TORCH_CHECK(x->is_cuda() && x->device() == device,
                "weights and cos/sin must be on the input CUDA device");
    TORCH_CHECK(x->scalar_type() == at::ScalarType::BFloat16,
                "weights and cos/sin must be bfloat16");
    TORCH_CHECK(x->is_contiguous(), "weights and cos/sin must be contiguous");
  }
  TORCH_CHECK(qw.dim() == 1 && qw.numel() == head_dim,
              "q_norm_weight must have shape [head_dim]");
  TORCH_CHECK(kw.dim() == 1 && kw.numel() == head_dim,
              "k_norm_weight must have shape [head_dim]");
  TORCH_CHECK(kb.dim() == 1 && kb.numel() == head_dim,
              "k_norm_bias must have shape [head_dim]");
  TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes() &&
                  cos.size(1) == rotary_dim,
              "cos and sin must have shape [max_seq_len, rotary_dim]");
  TORCH_CHECK(positions.is_cuda() && positions.device() == device &&
                  positions.scalar_type() == at::ScalarType::Int &&
                  positions.is_contiguous() && positions.dim() == 1 &&
                  positions.numel() == tokens,
              "positions must be contiguous int32 [T] on the input device");
}

void launch(torch::Tensor const& q, torch::Tensor const& k,
            torch::Tensor& q_out, torch::Tensor& k_out,
            torch::Tensor const& qw, torch::Tensor const& kw,
            torch::Tensor const& kb, torch::Tensor const& cos,
            torch::Tensor const& sin, torch::Tensor const& positions,
            int64_t q_heads, int64_t k_heads, double eps, double q_weight_bias) {
  int64_t const tasks = q.size(0) * (q_heads + k_heads);
  if (tasks == 0) return;
  constexpr int launch_threads = 128;
  constexpr int launch_heads = launch_threads / kThreadsPerHead;
  at::cuda::OptionalCUDAGuard const guard(device_of(q));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream(q.get_device());
  bool const use_vectorized_rope = q_heads == 16 && q.size(0) >= 512;
  bool const use_packed_rope_values = q_heads == 16 && q.size(0) >= 4096;
  auto launch_kernel = [&](auto mode_tag, auto rope_tag, auto values_tag,
                           auto direct_mapping_tag, int64_t mode_tasks) {
    constexpr KernelMode mode = decltype(mode_tag)::value;
    constexpr bool vectorized_rope = decltype(rope_tag)::value;
    constexpr bool packed_rope_values = decltype(values_tag)::value;
    constexpr bool direct_qh16_mapping = decltype(direct_mapping_tag)::value;
    int64_t const blocks = (mode_tasks + launch_heads - 1) / launch_heads;
    TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(), "grid is too large");
    indexer_norm_rope_kernel<mode, vectorized_rope, packed_rope_values,
                                 direct_qh16_mapping, false>
        <<<static_cast<uint32_t>(blocks), launch_threads, 0, stream>>>(
        reinterpret_cast<bfloat16 const*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16*>(q_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16*>(k_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(qw.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(kw.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(kb.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(sin.data_ptr<at::BFloat16>()),
        positions.data_ptr<int32_t>(), q.stride(0), k.stride(0), q_out.stride(0),
        k_out.stride(0), cos.stride(0), q.size(0), q_heads, k_heads,
        static_cast<float>(eps), static_cast<float>(q_weight_bias));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  };
  auto launch_mode = [&](auto mode_tag, int64_t mode_tasks) {
    constexpr KernelMode mode = decltype(mode_tag)::value;
    auto launch_rope = [&](auto direct_mapping_tag) {
      if (use_vectorized_rope) {
        if (use_packed_rope_values) {
          launch_kernel(mode_tag, std::true_type{}, std::true_type{},
                        direct_mapping_tag, mode_tasks);
        } else {
          launch_kernel(mode_tag, std::true_type{}, std::false_type{},
                        direct_mapping_tag, mode_tasks);
        }
      } else {
        launch_kernel(mode_tag, std::false_type{}, std::false_type{},
                      direct_mapping_tag, mode_tasks);
      }
    };
    if constexpr (mode == KernelMode::kQOnly) {
      if (q_heads == kHeadsPerBlock) launch_rope(std::true_type{});
      else launch_rope(std::false_type{});
    } else {
      launch_rope(std::false_type{});
    }
  };
  bool const use_direct_qh4 =
      q_heads == 4 && k_heads == 1 && q.size(0) <= 128;
  if (use_direct_qh4) {
    int64_t const blocks = (q.size(0) + 2) / 3;
    TORCH_CHECK(blocks <= std::numeric_limits<int32_t>::max(),
                "grid is too large");
    indexer_norm_rope_kernel<KernelMode::kFused, false, false, false, true>
        <<<static_cast<uint32_t>(blocks), launch_threads, 0, stream>>>(
        reinterpret_cast<bfloat16 const*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16*>(q_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16*>(k_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(qw.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(kw.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(kb.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16 const*>(sin.data_ptr<at::BFloat16>()),
        positions.data_ptr<int32_t>(), q.stride(0), k.stride(0),
        q_out.stride(0), k_out.stride(0), cos.stride(0), q.size(0),
        q_heads, k_heads, static_cast<float>(eps),
        static_cast<float>(q_weight_bias));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  bool const use_split = q.size(0) >= 4096 && k_heads == 1 &&
                         (q_heads == 4 || q_heads == 16);
  if (use_split) {
    launch_mode(std::integral_constant<KernelMode, KernelMode::kQOnly>{},
                q.size(0) * q_heads);
    launch_mode(std::integral_constant<KernelMode, KernelMode::kKOnly>{},
                q.size(0) * k_heads);
  } else {
    launch_mode(std::integral_constant<KernelMode, KernelMode::kFused>{}, tasks);
  }
}
}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> indexer_norm_rope(
    torch::Tensor const& q, torch::Tensor const& k, torch::Tensor const& z,
    torch::Tensor const& qw, torch::Tensor const& kw, torch::Tensor const& kb,
    torch::Tensor const& cos, torch::Tensor const& sin,
    torch::Tensor const& positions, int64_t head_dim, int64_t rotary_dim,
    double eps, double q_weight_bias) {
  auto const qs = validate_input(q, "index_q", head_dim);
  auto const ks = validate_input(k, "index_k", head_dim);
  auto const zs = validate_input(z, "index_z", head_dim);
  TORCH_CHECK(qs.tokens == ks.tokens && qs.tokens == zs.tokens,
              "Q, K, and Z must have the same token count");
  TORCH_CHECK(qs.heads > 0 && ks.heads > 0 && ks.heads == zs.heads,
              "Q must be nonempty and K/Z must have the same positive head count");
  TORCH_CHECK(q.device() == k.device() && q.device() == z.device(),
              "Q, K, and Z must be on the same device");
  validate_common(qw, kw, kb, cos, sin, positions, q.device(), qs.tokens,
                  head_dim, rotary_dim, eps);
  auto q_out = torch::empty_like(q);
  auto k_out = torch::empty_like(k);
  launch(q, k, q_out, k_out, qw, kw, kb, cos, sin, positions, qs.heads,
         ks.heads, eps, q_weight_bias);
  return {q_out, k_out, z};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> indexer_norm_rope_out(
    torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& z, torch::Tensor q_out, torch::Tensor k_out,
    torch::Tensor const& qw, torch::Tensor const& kw,
    torch::Tensor const& kb, torch::Tensor const& cos,
    torch::Tensor const& sin, torch::Tensor const& positions,
    int64_t head_dim, int64_t rotary_dim, double eps,
    double q_weight_bias) {
  auto const qs = validate_input(q, "index_q", head_dim);
  auto const ks = validate_input(k, "index_k", head_dim);
  auto const zs = validate_input(z, "index_z", head_dim);
  validate_input(q_out, "q_out", head_dim);
  validate_input(k_out, "k_out", head_dim);
  TORCH_CHECK(qs.tokens == ks.tokens && qs.tokens == zs.tokens,
              "Q, K, and Z must have the same token count");
  TORCH_CHECK(qs.heads > 0 && ks.heads > 0 && ks.heads == zs.heads,
              "Q must be nonempty and K/Z must have the same positive head count");
  TORCH_CHECK(q.device() == k.device() && q.device() == z.device(),
              "Q, K, and Z must be on the same device");
  TORCH_CHECK(q_out.device() == q.device() && k_out.device() == q.device(),
              "q_out and k_out must be on the input CUDA device");
  TORCH_CHECK(q_out.sizes() == q.sizes(), "q_out must match index_q shape");
  TORCH_CHECK(k_out.sizes() == k.sizes(), "k_out must match index_k shape");
  TORCH_CHECK(!q_out.is_alias_of(q) && !q_out.is_alias_of(k) &&
                  !q_out.is_alias_of(z) && !k_out.is_alias_of(q) &&
                  !k_out.is_alias_of(k) && !k_out.is_alias_of(z) &&
                  !q_out.is_alias_of(k_out),
              "q_out and k_out must not alias inputs or each other");
  validate_common(qw, kw, kb, cos, sin, positions, q.device(), qs.tokens,
                  head_dim, rotary_dim, eps);
  launch(q, k, q_out, k_out, qw, kw, kb, cos, sin, positions, qs.heads,
         ks.heads, eps, q_weight_bias);
  return {q_out, k_out, z};
}

torch::Tensor indexer_norm_rope_packed_(
    torch::Tensor qkz, torch::Tensor const& qw, torch::Tensor const& kw,
    torch::Tensor const& kb, torch::Tensor const& cos,
    torch::Tensor const& sin, torch::Tensor const& positions,
    int64_t q_heads, int64_t k_heads, int64_t head_dim,
    int64_t rotary_dim, double eps, double q_weight_bias) {
  TORCH_CHECK(qkz.is_cuda() && qkz.is_contiguous() &&
                  qkz.scalar_type() == at::ScalarType::BFloat16 && qkz.dim() == 2,
              "index_qkz must be contiguous CUDA bfloat16 [T, packed_width]");
  TORCH_CHECK(q_heads > 0 && k_heads > 0, "head counts must be positive");
  TORCH_CHECK(qkz.size(1) == (q_heads + 2 * k_heads) * head_dim,
              "invalid packed width");
  validate_common(qw, kw, kb, cos, sin, positions, qkz.device(), qkz.size(0),
                  head_dim, rotary_dim, eps);
  auto q = qkz.narrow(1, 0, q_heads * head_dim);
  auto k = qkz.narrow(1, q_heads * head_dim, k_heads * head_dim);
  launch(q, k, q, k, qw, kw, kb, cos, sin, positions, q_heads, k_heads,
         eps, q_weight_bias);
  return qkz;
}
