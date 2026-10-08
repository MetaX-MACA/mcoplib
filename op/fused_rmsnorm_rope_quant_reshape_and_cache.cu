#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include <maca_fp8.h>
#include "../kernel/utils.h"
#include "../kernel/all_reduce_kernel.cuh"
#include "../include/fused_rmsnorm_rope_quant_reshape_and_cache.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <limits>

namespace {
#ifndef __shfl_xor_sync_16
#define __shfl_xor_sync_16(mask, val, offset) \
  __shfl_xor_sync(mask, val, offset, 16)
#endif

constexpr int KV_BF16 = 0;
constexpr int KV_FP8 = 1;
constexpr int KV_INT8 = 2;

constexpr int kHeadDim = 128;
// 128-bit vectorized access: a bf16 vector holds 8 elements (16 bytes).
constexpr int kVec = 8;
constexpr int kVecsPerHead = kHeadDim / kVec;  // 16
constexpr int kHeadBlocks = kHeadDim / kVec;   // k_cache head_dim/x

// Head-major tile: 16 lanes own one head, block = 256 threads.
constexpr int kTPH = 16;
constexpr int kHeadsPerBlock = 16;
constexpr int kOcc = 4;
constexpr int kRegVec = kVecsPerHead / kTPH;  // 1

// Token-major tile: 4 lanes own one head, one group owns a whole token, so the
// norm heads of that token stream through registers without re-reading HBM.
constexpr int kTokTPH = 4;
constexpr int kTokGroupsPerBlock = 32;
constexpr int kTokOcc = 8;
constexpr int kTokRegVec = kVecsPerHead / kTokTPH;  // 4

constexpr int kTileT = 64;
constexpr int kBlockThreadsB = 512;

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T data[Size];
};

using InputVector = AlignedVector<__maca_bfloat16, kVec>;
using WeightVector = AlignedVector<float, kVec>;

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ uint32_t rc_pack4_fp8_e4m3(float a, float b, float c,
                                                      float d) {
  using v4f32 = float __attribute__((ext_vector_type(4)));
  v4f32 v = {a, b, c, d};
#pragma unroll
  for (int k = 0; k < 4; k++) v[k] = fminf(fmaxf(v[k], -448.f), 448.f);
  return __builtin_mxc_cvt_pk4_f32tof8(v);
}

// fp32 -> cache dtype, with the reciprocal scale already applied by the caller.
template <typename cache_t, int KV>
__device__ __forceinline__ cache_t quant_one(float f) {
  if constexpr (KV == KV_BF16) {
    return __float2bfloat16_rn(f);
  } else if constexpr (KV == KV_INT8) {
    int32_t q = __float2int_rn(f);
    q = min(q, 127);
    q = max(q, -127);
    return static_cast<cache_t>(q);
  } else {
    f = fminf(fmaxf(f, -448.f), 448.f);
    return static_cast<cache_t>(c10::Float8_e4m3fn(f).x);
  }
}

using v4u32 = unsigned int __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void store_nontemporal(void* p, InputVector const& v) {
  __builtin_nontemporal_store(*reinterpret_cast<const v4u32*>(&v),
                              reinterpret_cast<v4u32*>(p));
}

__device__ __forceinline__ int find_batch_bin(
    const int* __restrict__ accum_q_lens, int tid, int batch_size) {
  int lo = 0, hi = batch_size;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (accum_q_lens[mid + 1] > tid) hi = mid;
    else                             lo = mid + 1;
  }
  return lo;
}

// A native 16-lane subgroup contains an integer number of aligned heads for
// both ThreadsPerHead values. XOR offsets below ThreadsPerHead stay inside each
// head, so their reductions execute independently on the native shuffle path.
template <int ThreadsPerHead>
__device__ __forceinline__ float subgroup_all_reduce_sum(float value) {
  static_assert(ThreadsPerHead <= 16 && (ThreadsPerHead & (ThreadsPerHead - 1)) == 0,
                "subgroup reduction requires a power-of-two size <= 16");
  static_assert(16 % ThreadsPerHead == 0,
                "heads must align inside a native 16-lane subgroup");
#pragma unroll
  for (int offset = ThreadsPerHead / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync_16(0xffffffffffffffffULL, value, offset);
  }
  return value;
}

// 8 个 bf16 = 4 个 uint32，与 lane^xor_off 整体交换
__device__ __forceinline__ void exchange_vec(InputVector& v, int xor_off) {
  int* p = reinterpret_cast<int*>(&v);
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    p[k] = __shfl_xor_sync_16(0xffffffffffffffffULL, p[k], xor_off);
  }
}

// Keep the slice in registers as PACKED bf16 (half the register footprint of
// expanded fp32 -> higher occupancy), converting to float only transiently to
// accumulate the sum of squares and, later, to normalize.
__device__ __forceinline__ float accumulate_sum_squares(InputVector const& vec) {
  float sum_squares = 0.0f;
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    float const v = __bfloat162float(vec.data[e]);
    sum_squares += v * v;
  }
  return sum_squares;
}

__device__ __forceinline__ void normalize_vec(float (&out)[kVec],
                                              InputVector const& in,
                                              WeightVector const& weight_vec,
                                              float inv_rms) {
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    out[e] = __bfloat162float(in.data[e]) * inv_rms * weight_vec.data[e];
  }
}

__device__ __forceinline__ InputVector pack_vec(float const (&vf)[kVec]) {
  InputVector out;
#pragma unroll
  for (int e = 0; e < kVec; ++e) out.data[e] = __float2bfloat16_rn(vf[e]);
  return out;
}

// Write one 16-byte slice of a K head into the paged cache.
template <typename cache_t, int KV>
__device__ __forceinline__ void store_k_vec(cache_t* kdst,
                                            float const (&vf)[kVec],
                                            float inv_scale) {
  if constexpr (KV == KV_BF16) {
    *reinterpret_cast<InputVector*>(kdst) = pack_vec(vf);
  } else if constexpr (KV == KV_FP8) {
    uint2 out;
    out.x = rc_pack4_fp8_e4m3(vf[0] * inv_scale, vf[1] * inv_scale,
                              vf[2] * inv_scale, vf[3] * inv_scale);
    out.y = rc_pack4_fp8_e4m3(vf[4] * inv_scale, vf[5] * inv_scale,
                              vf[6] * inv_scale, vf[7] * inv_scale);
    *reinterpret_cast<uint2*>(kdst) = out;
  } else {
    cache_t tmp[kVec];
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      tmp[e] = quant_one<cache_t, KV>(vf[e] * inv_scale);
    }
    *reinterpret_cast<uint2*>(kdst) = *reinterpret_cast<const uint2*>(tmp);
  }
}

template <typename cache_t, int KV, int ROPE_DIM, bool BATCH1>
__global__ void __launch_bounds__(kTokTPH* kTokGroupsPerBlock, kTokOcc)
    fused_qk_norm_rope_kcache_tok_kernel(
        __maca_bfloat16* __restrict__ packed_qkv,
        const float* __restrict__ q_norm_weight,
        const float* __restrict__ k_norm_weight,
        const float* __restrict__ cos_base,
        const float* __restrict__ sin_base,
        const int* __restrict__ accum_q_lens,
        const int* __restrict__ cache_lens,
        cache_t* __restrict__ key_cache,
        const int64_t* __restrict__ slot_mapping,
        const float* __restrict__ k_scale,
        int num_tokens, int batch_size,
        int q_head_num, int kv_head_num, int norm_head_num,
        int total_head_num, int block_size, int rope_offset,
        float eps) {
  constexpr int kRopeVecs = ROPE_DIM / kVec;
  constexpr int kHalfVec = kRopeVecs / 2;
  constexpr int kPairStride = kHalfVec / kTokTPH;
  static_assert(kPairStride >= 1,
                "token-major tile requires both rope halves inside one lane");

  int const group = threadIdx.x / kTokTPH;
  int const lane_id = threadIdx.x % kTokTPH;

  int const token_id = blockIdx.x * kTokGroupsPerBlock + group;
  if (token_id >= num_tokens) {
    return;
  }

  int pos;
  if constexpr (BATCH1) {
    pos = cache_lens[0] + token_id;
  } else {
    int const b = find_batch_bin(accum_q_lens, token_id, batch_size);
    pos = cache_lens[b] + (token_id - accum_q_lens[b]);
  }

  // The rope table row is shared by every head of this token: load it once.
  int const rope_base_r = (rope_offset / kVec) / kTokTPH;
  WeightVector cos_vec[kPairStride];
  WeightVector sin_vec[kPairStride];
  {
    const float* const cp_row = cos_base + static_cast<int64_t>(pos) * ROPE_DIM;
    const float* const sp_row = sin_base + static_cast<int64_t>(pos) * ROPE_DIM;
#pragma unroll
    for (int pr = 0; pr < kPairStride; ++pr) {
      int const tbl_off =
          (lane_id + (rope_base_r + pr) * kTokTPH) * kVec - rope_offset;
      cos_vec[pr] = *reinterpret_cast<WeightVector const*>(cp_row + tbl_off);
      sin_vec[pr] = *reinterpret_cast<WeightVector const*>(sp_row + tbl_off);
    }
  }

  int64_t const token_stride = static_cast<int64_t>(total_head_num) * kHeadDim;
  __maca_bfloat16* const tok_ptr =
      packed_qkv + static_cast<int64_t>(token_id) * token_stride;

  int64_t const slot = slot_mapping[token_id];
  bool const kv_ok = (slot >= 0);
  int64_t const blk = kv_ok ? (slot / block_size) : 0;
  int64_t const blk_off = kv_ok ? (slot % block_size) : 0;
  cache_t* const kbase = key_cache
      + blk * static_cast<int64_t>(kv_head_num) * kHeadBlocks * block_size * kVec
      + blk_off * kVec;

  float inv_scale = 1.0f;
  if constexpr (KV != KV_BF16) inv_scale = __builtin_mxc_rcpf(*k_scale);

  InputVector cur[kTokRegVec], nxt[kTokRegVec];

  auto load_head = [&](int h, InputVector* dst) {
    InputVector const* const head_data = reinterpret_cast<InputVector const*>(
        tok_ptr + static_cast<int64_t>(h) * kHeadDim);
#pragma unroll
    for (int r = 0; r < kTokRegVec; ++r) dst[r] = head_data[lane_id + r * kTokTPH];
  };

  load_head(0, cur);

#pragma unroll 1
  for (int h = 0; h < norm_head_num; ++h) {
    if (h + 1 < norm_head_num) load_head(h + 1, nxt);

    float sum_squares = 0.0f;
#pragma unroll
    for (int r = 0; r < kTokRegVec; ++r) {
      sum_squares += accumulate_sum_squares(cur[r]);
    }
    sum_squares = subgroup_all_reduce_sum<kTokTPH>(sum_squares);
    float const inv_rms = __builtin_mxc_rcpf(
        sqrtf(sum_squares / static_cast<float>(kHeadDim) + eps));

    bool const is_q = (h < q_head_num);
    WeightVector const* const packed_weight =
        reinterpret_cast<WeightVector const*>(is_q ? q_norm_weight : k_norm_weight);

    float vf[kTokRegVec][kVec];
#pragma unroll
    for (int r = 0; r < kTokRegVec; ++r) {
      normalize_vec(vf[r], cur[r], packed_weight[lane_id + r * kTokTPH], inv_rms);
    }

    // NeoX rope: the (x, y) halves both live in this lane's registers.
#pragma unroll
    for (int pr = 0; pr < kPairStride; ++pr) {
      int const r_lo = rope_base_r + pr;
      int const r_hi = r_lo + kPairStride;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float const x = vf[r_lo][e];
        float const y = vf[r_hi][e];
        vf[r_lo][e] = x * cos_vec[pr].data[e] - y * sin_vec[pr].data[e];
        vf[r_hi][e] = y * cos_vec[pr].data[e] + x * sin_vec[pr].data[e];
      }
    }

    if (is_q) {
      __maca_bfloat16* const hp = tok_ptr + static_cast<int64_t>(h) * kHeadDim;
#pragma unroll
      for (int r = 0; r < kTokRegVec; ++r) {
        store_nontemporal(hp + (lane_id + r * kTokTPH) * kVec, pack_vec(vf[r]));
      }
    } else if (kv_ok) {
      cache_t* const khead = kbase + static_cast<int64_t>(h - q_head_num) *
                                         kHeadBlocks * block_size * kVec;
#pragma unroll
      for (int r = 0; r < kTokRegVec; ++r) {
        int const vec_idx = lane_id + r * kTokTPH;
        store_k_vec<cache_t, KV>(
            khead + static_cast<int64_t>(vec_idx) * block_size * kVec, vf[r],
            inv_scale);
      }
    }

#pragma unroll
    for (int r = 0; r < kTokRegVec; ++r) cur[r] = nxt[r];
  }
}

template <typename cache_t, int KV, int ROPE_DIM, bool BATCH1>
__global__ void __launch_bounds__(kTPH* kHeadsPerBlock, kOcc)
    fused_qk_norm_rope_kcache_kernel(
        __maca_bfloat16* __restrict__ packed_qkv,
        const float* __restrict__ q_norm_weight,
        const float* __restrict__ k_norm_weight,
        const float* __restrict__ cos_base,
        const float* __restrict__ sin_base,
        const int* __restrict__ accum_q_lens,
        const int* __restrict__ cache_lens,
        cache_t* __restrict__ key_cache,
        const int64_t* __restrict__ slot_mapping,
        const float* __restrict__ k_scale,
        int num_tokens, int batch_size,
        int q_head_num, int kv_head_num, int norm_head_num,
        int total_head_num, int block_size, int rope_offset,
        int64_t total_heads, float eps) {
  constexpr int kRopeVecs = ROPE_DIM / kVec;
  constexpr int kHalfVec = kRopeVecs / 2;
  // With 16 lanes per head each lane owns exactly one vector, so the rope
  // partner half always lives in another lane: the shuffle path is the only
  // one reachable here.
  static_assert(kHalfVec / kTPH == 0, "head-major tile expects the shuffle rope");

  int const local_head = threadIdx.x / kTPH;
  int const lane_id = threadIdx.x % kTPH;

  int64_t const global_head =
      static_cast<int64_t>(blockIdx.x) * kHeadsPerBlock + local_head;
  bool const active = (global_head < total_heads);

  // Map the flat head index back to (token, head-within-token).
  uint32_t const gh = static_cast<uint32_t>(active ? global_head : 0);
  uint32_t const nhn = static_cast<uint32_t>(norm_head_num);
  uint32_t const tok = gh / nhn;
  int const token_id = static_cast<int>(tok);
  int const head_in_token = static_cast<int>(gh - tok * nhn);

  int pos = 0;
  if constexpr (BATCH1) {
    if (active) pos = cache_lens[0] + token_id;
  } else {
    if (active) {
      int const b = find_batch_bin(accum_q_lens, token_id, batch_size);
      pos = cache_lens[b] + (token_id - accum_q_lens[b]);
    }
  }
  if (!active) {
    return;
  }

  int64_t const token_stride = static_cast<int64_t>(total_head_num) * kHeadDim;
  __maca_bfloat16* const head_ptr = packed_qkv
      + static_cast<int64_t>(token_id) * token_stride
      + static_cast<int64_t>(head_in_token) * kHeadDim;
  InputVector* const head_data = reinterpret_cast<InputVector*>(head_ptr);

  InputVector reg_input[kRegVec];
  float sum_squares = 0.0f;
#pragma unroll
  for (int r = 0; r < kRegVec; ++r) {
    reg_input[r] = head_data[lane_id + r * kTPH];
    sum_squares += accumulate_sum_squares(reg_input[r]);
  }

  sum_squares = subgroup_all_reduce_sum<kTPH>(sum_squares);
  float const inv_rms = __builtin_mxc_rcpf(
      sqrtf(sum_squares / static_cast<float>(kHeadDim) + eps));

  bool const is_q = (head_in_token < q_head_num);
  WeightVector const* const packed_weight =
      reinterpret_cast<WeightVector const*>(is_q ? q_norm_weight : k_norm_weight);

  float vf[kRegVec][kVec];
#pragma unroll
  for (int r = 0; r < kRegVec; ++r) {
    normalize_vec(vf[r], reg_input[r], packed_weight[lane_id + r * kTPH], inv_rms);
  }

  // One vector per lane: fetch the partner half over the native shuffle path.
  {
    InputVector partner = pack_vec(vf[0]);
    exchange_vec(partner, kHalfVec);

    int const rope_lane = lane_id - (rope_offset / kVec);
    bool const in_rope = (rope_lane >= 0) && (rope_lane < kRopeVecs);
    if (in_rope) {
      bool const upper = (rope_lane >= kHalfVec);
      int const tbl_off = (upper ? rope_lane - kHalfVec : rope_lane) * kVec;
      WeightVector const cos_vec = *reinterpret_cast<WeightVector const*>(
          cos_base + static_cast<int64_t>(pos) * ROPE_DIM + tbl_off);
      WeightVector const sin_vec = *reinterpret_cast<WeightVector const*>(
          sin_base + static_cast<int64_t>(pos) * ROPE_DIM + tbl_off);
      float const sgn = upper ? 1.0f : -1.0f;
#pragma unroll
      for (int e = 0; e < kVec; ++e) {
        float const o = __bfloat162float(partner.data[e]);
        vf[0][e] = vf[0][e] * cos_vec.data[e] + sgn * o * sin_vec.data[e];
      }
    }
  }

  if (is_q) {
#pragma unroll
    for (int r = 0; r < kRegVec; ++r) {
      store_nontemporal(head_ptr + (lane_id + r * kTPH) * kVec, pack_vec(vf[r]));
    }
    return;
  }

  int64_t const slot = slot_mapping[token_id];
  if (slot < 0) {
    return;
  }

  cache_t* const kbase = key_cache
      + (slot / block_size) * static_cast<int64_t>(kv_head_num) * kHeadBlocks *
            block_size * kVec
      + static_cast<int64_t>(head_in_token - q_head_num) * kHeadBlocks *
            block_size * kVec
      + (slot % block_size) * kVec;

  float inv_scale = 1.0f;
  if constexpr (KV != KV_BF16) inv_scale = __builtin_mxc_rcpf(*k_scale);

#pragma unroll
  for (int r = 0; r < kRegVec; ++r) {
    int const vec_idx = lane_id + r * kTPH;
    store_k_vec<cache_t, KV>(
        kbase + static_cast<int64_t>(vec_idx) * block_size * kVec, vf[r],
        inv_scale);
  }
}

template <typename cache_t, int KV, int VST>
__global__ void fused_v_reshape_and_cache_kernel(
    const __maca_bfloat16* __restrict__ value,  // packed_qkv + V 段列偏移
    cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int value_stride, const int num_heads, const int head_size,
    const int block_size, const float* __restrict__ v_scale,
    const int num_tokens) {
  constexpr int TILE_T = kTileT;
  constexpr int MAXCH = TILE_T / kVec;

  float const inv_scale = (KV == KV_BF16) ? 1.0f : (1.0f / *v_scale);

  const int head_idx = blockIdx.x;
  const int tok0 = blockIdx.y * TILE_T;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int tile_tokens = min(TILE_T, num_tokens - tok0);
  if (tile_tokens <= 0) return;

  const int SP = head_size + kVec;  // padded to avoid bank conflicts
  extern __shared__ char smem_raw[];
  __maca_bfloat16* smem = reinterpret_cast<__maca_bfloat16*>(smem_raw);

  char* meta_base =
      smem_raw + static_cast<size_t>(TILE_T) * SP * sizeof(__maca_bfloat16);
  int64_t* s_slot = reinterpret_cast<int64_t*>(meta_base);
  int64_t* s_blk = s_slot + TILE_T;
  int64_t* s_off = s_blk + MAXCH;
  int* s_ctg = reinterpret_cast<int*>(s_off + MAXCH);

  const int vecs_per_tok = head_size / kVec;
  const int tok_chunks = (tile_tokens + VST - 1) / VST;

  for (int t = tid; t < tile_tokens; t += nthreads) {
    s_slot[t] = slot_mapping[tok0 + t];
  }

  for (int i = tid; i < tile_tokens * vecs_per_tok; i += nthreads) {
    const int t = i / vecs_per_tok;
    const int dv = i - t * vecs_per_tok;
    const __maca_bfloat16* src = value
        + static_cast<int64_t>(tok0 + t) * value_stride
        + head_idx * head_size + dv * kVec;
    *reinterpret_cast<InputVector*>(&smem[t * SP + dv * kVec]) =
        *reinterpret_cast<const InputVector*>(src);
  }
  __syncthreads();

  // A run of VST consecutive slots lets the transposed store go out as one
  // 16-byte write instead of VST scalar writes.
  for (int c = tid; c < tok_chunks; c += nthreads) {
    const int t0 = c * VST;
    const int cnt = min(VST, tile_tokens - t0);
    const int64_t slot0 = s_slot[t0];
    bool contig = (cnt == VST) && (slot0 >= 0) && ((slot0 % VST) == 0);
    if (contig) {
#pragma unroll
      for (int k = 1; k < VST; ++k) {
        if (s_slot[t0 + k] != slot0 + k) contig = false;
      }
    }
    s_ctg[c] = contig ? 1 : 0;
    s_blk[c] = contig ? (slot0 / block_size) : 0;
    s_off[c] = contig ? (slot0 % block_size) : 0;
  }
  __syncthreads();

  for (int i = tid; i < head_size * tok_chunks; i += nthreads) {
    const int d = i / tok_chunks;
    const int c = i - d * tok_chunks;
    const int t0 = c * VST;

    if (s_ctg[c]) {
      cache_t* dst = value_cache
          + s_blk[c] * static_cast<int64_t>(num_heads) * head_size * block_size
          + head_idx * static_cast<int64_t>(head_size) * block_size
          + d * static_cast<int64_t>(block_size) + s_off[c];

      if constexpr (KV == KV_FP8) {
        float f[VST];
#pragma unroll
        for (int k = 0; k < VST; ++k) {
          f[k] = __bfloat162float(smem[(t0 + k) * SP + d]) * inv_scale;
        }
        if constexpr (VST == 16) {
          uint4 out;
          out.x = rc_pack4_fp8_e4m3(f[0], f[1], f[2], f[3]);
          out.y = rc_pack4_fp8_e4m3(f[4], f[5], f[6], f[7]);
          out.z = rc_pack4_fp8_e4m3(f[8], f[9], f[10], f[11]);
          out.w = rc_pack4_fp8_e4m3(f[12], f[13], f[14], f[15]);
          *reinterpret_cast<uint4*>(dst) = out;
        } else {
          uint2 out;
          out.x = rc_pack4_fp8_e4m3(f[0], f[1], f[2], f[3]);
          out.y = rc_pack4_fp8_e4m3(f[4], f[5], f[6], f[7]);
          *reinterpret_cast<uint2*>(dst) = out;
        }
      } else {
        cache_t tmp[VST];
#pragma unroll
        for (int k = 0; k < VST; ++k) {
          tmp[k] = quant_one<cache_t, KV>(
              __bfloat162float(smem[(t0 + k) * SP + d]) * inv_scale);
        }
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(tmp);
      }
    } else {
      const int cnt = min(VST, tile_tokens - t0);
      for (int k = 0; k < cnt; ++k) {
        const int64_t slot = s_slot[t0 + k];
        if (slot < 0) continue;
        cache_t* dst = value_cache
            + (slot / block_size) * static_cast<int64_t>(num_heads) * head_size *
                  block_size
            + head_idx * static_cast<int64_t>(head_size) * block_size
            + d * static_cast<int64_t>(block_size) + (slot % block_size);
        *dst = quant_one<cache_t, KV>(
            __bfloat162float(smem[(t0 + k) * SP + d]) * inv_scale);
      }
    }
  }
}
}  // namespace

//         hidden_states [T, 4096]
//                 │
//            qkv_proj (GEMM)  -> qkv [T, (64 + 4 + 4) * 128]
//                 │
//       split → q[T, 64, 128]  k[T, 4, 128]  v[T, 4, 128]
//                 │              │              │
//   ① q_norm(128维, fp32权重)  ① k_norm(128维, fp32权重)   （v 不做 norm）
//                 │              │              │
//   ② rope(NeoX, 全128维)      ② rope(NeoX, 全128维)      （v 不做 rope）
//                 │              │              │
//                 │              └──── ③ quant(可选 FP8) ───┤
//                 │                     │                   │
//                 │              ④ reshape_and_cache
//                 │                  写 k_cache          写 v_cache
//                 │                        │
//                 └────── FlashAttention ──┘
void fused_rmsnorm_rope_quant_reshape_and_cache(
    torch::Tensor& packed_qkv,             // bf16 [num_tokens, (Hq + 2*Hkv) * qk_head_dim]
    torch::Tensor const& q_norm_weight,    // fp32 [qk_head_dim]
    torch::Tensor const& k_norm_weight,    // fp32 [qk_head_dim]
    torch::Tensor const& cos,              // fp32 [max_pos, rope_dim]
    torch::Tensor const& sin,              // fp32 [max_pos, rope_dim]
    torch::Tensor const& q_lens,           // 接口兼容，语义由 accum_q_lens 承载
    torch::Tensor const& cache_lens,       // int32 [B]
    torch::Tensor const& accum_q_lens,     // int32 [B+1]
    torch::Tensor& k_cache,                // [nb, Hkv, qk_head_dim/x, block_size, x]
    torch::Tensor& v_cache,                // [nb, Hkv, qk_head_dim, block_size]
    torch::Tensor const& slot_mapping,     // int64 [num_tokens]，负值跳过
    c10::optional<torch::Tensor> k_scale,
    c10::optional<torch::Tensor> v_scale,
    const std::string& kv_cache_dtype,
    int64_t rope_offset, int64_t block_size, double eps) {
  DEBUG_TRACE_PARAMS(packed_qkv, q_norm_weight, k_norm_weight, cos, sin, q_lens, cache_lens, accum_q_lens, k_cache, v_cache, slot_mapping, k_scale, v_scale, kv_cache_dtype, rope_offset, block_size, eps);
  DEBUG_DUMP_PARAMS(packed_qkv, q_norm_weight, k_norm_weight, cos, sin, q_lens, cache_lens, accum_q_lens, k_cache, v_cache, slot_mapping, k_scale, v_scale, kv_cache_dtype, rope_offset, block_size, eps);

  TORCH_CHECK(packed_qkv.is_cuda() && packed_qkv.is_contiguous(),
              "packed_qkv must be a contiguous CUDA tensor");
  TORCH_CHECK(packed_qkv.dim() == 2, "packed_qkv must be [num_tokens, hidden]");
  TORCH_CHECK(packed_qkv.dtype() == at::kBFloat16, "packed_qkv must be bf16");
  TORCH_CHECK(q_norm_weight.dtype() == at::kFloat &&
              k_norm_weight.dtype() == at::kFloat, "norm weights must be fp32");
  TORCH_CHECK(q_norm_weight.is_contiguous() && k_norm_weight.is_contiguous());
  TORCH_CHECK(cos.dtype() == at::kFloat && sin.dtype() == at::kFloat,
              "cos/sin must be fp32");
  TORCH_CHECK(cos.is_contiguous() && sin.is_contiguous());
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos/sin shape mismatch");
  TORCH_CHECK(accum_q_lens.dtype() == at::kInt && cache_lens.dtype() == at::kInt,
              "accum_q_lens / cache_lens must be int32");
  TORCH_CHECK(slot_mapping.dtype() == at::kLong, "slot_mapping must be int64");
  TORCH_CHECK(eps >= 0.0, "eps must be non-negative");

  const int64_t num_tokens  = packed_qkv.size(0);
  const int64_t qk_head_dim = q_norm_weight.numel();
  TORCH_CHECK(qk_head_dim == kHeadDim,
              "fused path requires qk_head_dim == 128, got ", qk_head_dim);
  TORCH_CHECK(k_norm_weight.numel() == qk_head_dim,
              "k_norm_weight must have shape [qk_head_dim]");

  TORCH_CHECK(k_cache.dim() == 5 && v_cache.dim() == 4,
              "k_cache must be 5D and v_cache 4D");
  const int kv_head_num   = static_cast<int>(k_cache.size(1));
  const int x             = static_cast<int>(k_cache.size(4));
  const int h_block_count = static_cast<int>(k_cache.size(2));
  TORCH_CHECK(x == kVec, "fused path requires x == 8, got ", x);
  TORCH_CHECK(h_block_count * x == qk_head_dim, "k_cache head_dim mismatch");
  TORCH_CHECK(k_cache.size(3) == block_size && v_cache.size(3) == block_size,
              "block_size mismatch with cache layout");
  TORCH_CHECK(v_cache.size(1) == kv_head_num && v_cache.size(2) == qk_head_dim,
              "v_cache shape mismatch");
  TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type(),
              "k_cache / v_cache dtype mismatch");

  const int64_t hidden = packed_qkv.size(1);
  TORCH_CHECK(hidden % qk_head_dim == 0, "hidden must be a multiple of head_dim");
  const int total_head_num = static_cast<int>(hidden / qk_head_dim);
  const int q_head_num     = total_head_num - 2 * kv_head_num;
  const int norm_head_num  = q_head_num + kv_head_num;
  TORCH_CHECK(q_head_num > 0, "inconsistent q/kv split: hidden=", hidden,
              " kv_head_num=", kv_head_num, " head_dim=", qk_head_dim);

  const int rope_dim = static_cast<int>(cos.size(-1));
  TORCH_CHECK(rope_dim == 64 || rope_dim == 128,
              "fused path supports rope_dim 64 or 128, got ", rope_dim);
  TORCH_CHECK(rope_offset >= 0 && rope_offset + rope_dim <= qk_head_dim,
              "rope range out of bounds");
  TORCH_CHECK((rope_offset % kVec) == 0, "rope_offset must be 8-aligned");
  TORCH_CHECK(((rope_offset / kVec) % (rope_dim / kVec)) == 0,
              "rope_offset must align to the shuffle-pair stride (", rope_dim,
              " elements)");

  const int batch_size = static_cast<int>(cache_lens.numel());
  TORCH_CHECK(accum_q_lens.numel() == batch_size + 1,
              "accum_q_lens must have B+1 entries (prefix sum of q_lens)");
  TORCH_CHECK(slot_mapping.numel() >= num_tokens,
              "slot_mapping shorter than num_tokens");
  TORCH_CHECK(packed_qkv.device() == k_cache.device() &&
              packed_qkv.device() == v_cache.device(),
              "all tensors must be on the same device");
  (void)q_lens;

  if (kv_cache_dtype == "bf16" || kv_cache_dtype == "bfloat16") {
    TORCH_CHECK(k_cache.dtype() == at::kBFloat16,
                "kv_cache_dtype=bf16 requires bf16 cache");
  } else {
    TORCH_CHECK(k_scale.has_value() && v_scale.has_value(),
                "quantized kv cache requires k_scale / v_scale");
    TORCH_CHECK(k_scale->dtype() == at::kFloat && v_scale->dtype() == at::kFloat,
                "k_scale / v_scale must be fp32");
  }

  if (num_tokens == 0) return;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(packed_qkv));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto* qkv_ptr =
      reinterpret_cast<__maca_bfloat16*>(packed_qkv.data_ptr<at::BFloat16>());
  const float* qw_ptr = q_norm_weight.data_ptr<float>();
  const float* kw_ptr = k_norm_weight.data_ptr<float>();
  const float* cos_ptr = cos.data_ptr<float>();
  const float* sin_ptr = sin.data_ptr<float>();
  const int* accum_ptr = accum_q_lens.data_ptr<int>();
  const int* clen_ptr = cache_lens.data_ptr<int>();
  const int64_t* slot_ptr = slot_mapping.data_ptr<int64_t>();
  const float* kscale_ptr = k_scale.has_value() ? k_scale->data_ptr<float>() : nullptr;
  const float* vscale_ptr = v_scale.has_value() ? v_scale->data_ptr<float>() : nullptr;

  const int64_t total_heads = static_cast<int64_t>(num_tokens) * norm_head_num;
  const bool batch1 = (batch_size == 1);

  // The token-major tile needs the rope pair stride to fall inside one lane.
  const bool tok_major_ok = (rope_offset % (kVec * kTokTPH) == 0) &&
                            ((rope_dim / kVec / 2) % kTokTPH == 0) && (num_tokens >= 4096);

  const int64_t v_col_offset = static_cast<int64_t>(norm_head_num) * qk_head_dim;
  const dim3 gridB(kv_head_num, cdiv(num_tokens, kTileT));
  const size_t smem_bytes =
      static_cast<size_t>(kTileT) * (qk_head_dim + kVec) * packed_qkv.element_size()
      + static_cast<size_t>(kTileT) * sizeof(int64_t)
      + static_cast<size_t>(kTileT / kVec) * (2 * sizeof(int64_t) + sizeof(int))
      + 64;

#define LAUNCH_QK(CACHE_T, KV, RDIM, B1)                                        \
  do {                                                                          \
    if (tok_major_ok) {                                                         \
      fused_qk_norm_rope_kcache_tok_kernel<CACHE_T, KV, RDIM, B1>               \
          <<<dim3(cdiv(num_tokens, kTokGroupsPerBlock)),                        \
             dim3(kTokTPH* kTokGroupsPerBlock), 0, stream>>>(                   \
              qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, accum_ptr, clen_ptr,   \
              reinterpret_cast<CACHE_T*>(k_cache.data_ptr()), slot_ptr,         \
              kscale_ptr, static_cast<int>(num_tokens), batch_size,             \
              q_head_num, kv_head_num, norm_head_num, total_head_num,           \
              static_cast<int>(block_size), static_cast<int>(rope_offset),      \
              static_cast<float>(eps));                                         \
    } else {                                                                    \
      fused_qk_norm_rope_kcache_kernel<CACHE_T, KV, RDIM, B1>                   \
          <<<dim3(cdiv(total_heads, kHeadsPerBlock)),                           \
             dim3(kTPH* kHeadsPerBlock), 0, stream>>>(                          \
              qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, accum_ptr, clen_ptr,   \
              reinterpret_cast<CACHE_T*>(k_cache.data_ptr()), slot_ptr,         \
              kscale_ptr, static_cast<int>(num_tokens), batch_size,             \
              q_head_num, kv_head_num, norm_head_num, total_head_num,           \
              static_cast<int>(block_size), static_cast<int>(rope_offset),      \
              total_heads, static_cast<float>(eps));                            \
    }                                                                           \
  } while (0)

#define DISPATCH_KV(CACHE_T, KV, VST)                                           \
  do {                                                                          \
    if (rope_dim == 128) {                                                      \
      if (batch1) LAUNCH_QK(CACHE_T, KV, 128, true);                            \
      else        LAUNCH_QK(CACHE_T, KV, 128, false);                           \
    } else {                                                                    \
      if (batch1) LAUNCH_QK(CACHE_T, KV, 64, true);                             \
      else        LAUNCH_QK(CACHE_T, KV, 64, false);                            \
    }                                                                           \
    fused_v_reshape_and_cache_kernel<CACHE_T, KV, VST>                          \
        <<<gridB, dim3(kBlockThreadsB), smem_bytes, stream>>>(                  \
            qkv_ptr + v_col_offset,                                             \
            reinterpret_cast<CACHE_T*>(v_cache.data_ptr()), slot_ptr,           \
            static_cast<int>(hidden), kv_head_num,                              \
            static_cast<int>(qk_head_dim), static_cast<int>(block_size),        \
            vscale_ptr, static_cast<int>(num_tokens));                          \
  } while (0)

  if (kv_cache_dtype == "bf16" || kv_cache_dtype == "bfloat16") {
    DISPATCH_KV(__maca_bfloat16, KV_BF16, 8);
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3" ||
             kv_cache_dtype == "float8") {
    if ((num_tokens & 15) == 8 && num_tokens < 256) {
      DISPATCH_KV(__maca_fp8_e4m3, KV_FP8, 8);
    } else {
      DISPATCH_KV(__maca_fp8_e4m3, KV_FP8, 16);
    }
  } else if (kv_cache_dtype == "int8") {
    DISPATCH_KV(int8_t, KV_INT8, 16);
  } else {
    TORCH_CHECK(false, "unsupported kv_cache_dtype: ", kv_cache_dtype);
  }
}
