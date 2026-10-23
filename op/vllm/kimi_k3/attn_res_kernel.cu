/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

// Production AttnRes forward adapted for SM80 (Metax C500).
//
// Algorithm (per token t, N = num_blocks + 1 value rows):
//   values[n] = blocks[n] for n < N-1, prefix for n = N-1
//               (prefix += delta first if delta present)
//   Q[h]      = rms_w[h] * res_w[h]
//   rms_n     = rsqrt(mean(values[n]^2) + eps)
//   logit_n   = rms_n * (values[n] . Q)
//   probs     = softmax(logits over N)         (online softmax)
//   acc       = sum_n probs[n] * values[n]
//   if output_norm:
//       out   = acc * rsqrt(mean(acc^2) + eps_out * s^2) * output_norm_weight
//   else:
//       out   = acc / s
//
// SM80 adaptation notes:
//   - The original Blackwell kernel used TMEM to cache V rows between
//     pass A (sq/dot reductions) and pass B (weighted accumulation). SM80 has
//     no TMEM, so V tiles are kept in registers across the two passes instead.
//   - The original used cp.async.bulk + mbarrier with a separate producer
//     warp. SM80 only exposes cp.async.cg with __pipeline_commit/wait, and
//     the producer/consumer split adds tricky __syncthreads() scoping on
//     SM80 (block-wide barriers require all threads to arrive, so an
//     off-path producer warp would deadlock at consumer-only syncthreads).
//     This adaptation drops the producer warp: BLK=256, all threads
//     cooperatively load V rows via vectorized cp.async, then compute.
//   - Programmatic Launch Completion / cudaGridDependencySynchronize are
//     SM90+ only and are removed.
//   - The original dispatch could request NUM_BUFS up to 8 (NC=4,
//     CHUNK_DEPTH=2) = 112 KB, exceeding C500's 64 KB/SM. We force NC=2 and
//     CHUNK_DEPTH=1 (NUM_BUFS=2 = 28 KB) and disable OUTPUT_NORM_IN_SMEM so
//     the layout fits in 64 KB on every dispatch path.

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>

#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <type_traits>

using bf16_t = __nv_bfloat16;

namespace sm80 {
namespace fwd_prod_v2 {

constexpr int K_TILE = 1024;
constexpr int N_CHUNK_DEFAULT = 4;
// No producer warp on SM80: all 256 threads cooperate on load + compute.
constexpr int BLK = 256;
constexpr int CONSUMER_THREADS = BLK;
constexpr int CONSUMER_WARPS = CONSUMER_THREADS / 32;
constexpr int CONSUMER_GROUPS = 2;  // two 128-thread consumer groups
constexpr int CONSUMER_THREADS_PER_GROUP = CONSUMER_THREADS / CONSUMER_GROUPS;

// Fixed for C500 (sm_80, 64 KB/SM shared memory). NUM_BUFS = CHUNK_DEPTH*NC.
constexpr int NC_FIXED = 2;
constexpr int CHUNK_DEPTH = 1;

__device__ __forceinline__ const bf16_t* residual_addr(
    const bf16_t* block_res, const bf16_t* layer_res, int source, int N,
    int token, int block_stride_m, int block_stride_r, int H) {
  if (source < N - 1) {
    return block_res + static_cast<long long>(token) * block_stride_m +
           source * block_stride_r;
  }
  return layer_res + static_cast<long long>(token) * H;
}

__device__ __forceinline__ float2 float2_add(const float2&a, const float2&b) {
  float2 r;
  r.x=a.x+b.x;
  r.y=a.y+b.y;
  return r;
}

__device__ __forceinline__ float2 float2_mul(const float2&a, const float2&b) {
  float2 r;
  r.x=a.x*b.x;
  r.y=a.y*b.y;
  return r;
}

__device__ __forceinline__ float2 float2_fma(const float2&a, const float2&b, const float2&c) {
  float2 r;
  r.x=fmaf(a.x,b.x,c.x);
  r.y=fmaf(a.y,b.y,c.y);
  return r;
}

template<int NC>
struct FwdSmemPlan {
  float2 ws_stats[CONSUMER_WARPS][NC];
};

// Vectorized copy of one H-element bf16 row (H*2 bytes, multiple of 16)
// from global memory into shared memory. On Metax C500 (MACA / cucc) the
// cp.async PTX path is unavailable, so we fall back to plain vectorized
// int4 (16-byte) loads/stores; the surrounding __syncthreads() guarantees
// cross-thread visibility. All BLK threads cooperate.
__device__ __forceinline__ void cp_async_row(void* smem_dst,
                                             const void* gmem_src, int bytes) {
  int4* d = reinterpret_cast<int4*>(smem_dst);
  const int4* s = reinterpret_cast<const int4*>(gmem_src);
  int n4 = bytes >> 4;  // bytes / 16
  // With H=7168 bf16 -> 14336 bytes -> 896 int4 elements. 256 threads -> 3.5
  // elements each (4 with masking on the last).
  for (int i = threadIdx.x; i < n4; i += blockDim.x) {
    d[i] = s[i];
  }
}

template <int H, int N, int NC = N_CHUNK_DEFAULT, int B = 1,
          bool RELEASE_TMEM = false, bool HAS_DELTA = false,
          bool HAS_OUTPUT_NORM = false, bool OUTPUT_NORM_IN_SMEM = false>
__global__ void __launch_bounds__(BLK, 1) attn_res_fwd_online_v2_kernel(
    const bf16_t* __restrict__ block_res, bf16_t* __restrict__ layer_res,
    const bf16_t* __restrict__ delta, const bf16_t* __restrict__ res_w,
    const bf16_t* __restrict__ rms_w, bf16_t* __restrict__ output, int T,
    int block_stride_m, int block_stride_r, float rms_eps,
    const bf16_t* __restrict__ output_norm_weight, float output_norm_eps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr float LOG2_E = 1.4426950408889634f;
  constexpr int N_CHUNK = NC;
  constexpr int NUM_BUFS = CHUNK_DEPTH * NC;
  constexpr int NHT = H / K_TILE;
  constexpr int SLICES_PER_GROUP =
      (NHT + CONSUMER_GROUPS - 1) / CONSUMER_GROUPS;
  constexpr int VEC = 8;
  constexpr int ACC_PER_THREAD = H == 7168 ? 28 : SLICES_PER_GROUP * VEC;
  static_assert(H >= 4096 && H <= 8192);
  static_assert(H % K_TILE == 0);
  static_assert(CHUNK_DEPTH == 1);
  static_assert(!OUTPUT_NORM_IN_SMEM, "SM80 path reads output_norm from gmem");

  const int tid = threadIdx.x;
  const int wid = tid >> 5;
  const int lane = tid & 31;
  const int TB = T * B;
  const int num_ctas = gridDim.x;
  constexpr int num_chunks = (N + N_CHUNK - 1) / N_CHUNK;

  const int group = (wid >= 4) ? 1 : 0;
  const int ct_in_group = tid & (CONSUMER_THREADS_PER_GROUP - 1);
  const int k_local = ct_in_group * VEC;

  constexpr size_t V_BYTES = (size_t)NUM_BUFS * H * sizeof(bf16_t);
  constexpr size_t DELTA_BYTES =
      HAS_DELTA ? (size_t)CHUNK_DEPTH * H * sizeof(bf16_t) : 0;
  extern __shared__ __align__(16) char smem_raw[];
  bf16_t* v_bufs = reinterpret_cast<bf16_t*>(smem_raw);  // [NUM_BUFS][H]
  bf16_t* delta_bufs = reinterpret_cast<bf16_t*>(smem_raw + V_BYTES);
  FwdSmemPlan<NC>& plan = *reinterpret_cast<FwdSmemPlan<NC>*>(
      smem_raw + V_BYTES + DELTA_BYTES);

  auto slot_of = [](long long gci, int n) {
    return (int)(gci % CHUNK_DEPTH) * N_CHUNK + n;
  };
  auto buf_ptr = [&](int slot) -> bf16_t* { return v_bufs + slot * H; };
  auto delta_buf_ptr = [&](int chunk_slot) -> bf16_t* {
    return delta_bufs + chunk_slot * H;
  };

  // Q = rms_w * res_w, cached in registers across all tokens handled by this
  // CTA. Every thread needs it for the dot reduction.
  float q_cache[ACC_PER_THREAD];
#pragma unroll
  for (int si = 0; si < SLICES_PER_GROUP; si++) {
    if constexpr (H == 7168) {
      if (si == SLICES_PER_GROUP - 1) {
        int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
        int2 rms_v = *reinterpret_cast<const int2*>(rms_w + h_base);
        int2 res_v = *reinterpret_cast<const int2*>(res_w + h_base);
        auto* rms2 = reinterpret_cast<__nv_bfloat162*>(&rms_v);
        auto* res2 = reinterpret_cast<__nv_bfloat162*>(&res_v);
#pragma unroll
        for (int k = 0; k < 2; k++) {
          float2 rf = __bfloat1622float2(rms2[k]);
          float2 sf = __bfloat1622float2(res2[k]);
          q_cache[si * VEC + 2 * k] = rf.x * sf.x;
          q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
        }
        continue;
      }
    }
    int dt = si * CONSUMER_GROUPS + group;
    if (dt >= NHT) continue;
    int h_base = dt * K_TILE + k_local;
    int4 rms_v = *reinterpret_cast<const int4*>(rms_w + h_base);
    int4 res_v = *reinterpret_cast<const int4*>(res_w + h_base);
    auto* rms2 = reinterpret_cast<__nv_bfloat162*>(&rms_v);
    auto* res2 = reinterpret_cast<__nv_bfloat162*>(&res_v);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 rf = __bfloat1622float2(rms2[k]);
      float2 sf = __bfloat1622float2(res2[k]);
      q_cache[si * VEC + 2 * k] = rf.x * sf.x;
      q_cache[si * VEC + 2 * k + 1] = rf.y * sf.y;
    }
  }

  // Persistent loop over (token, batch) work items.
  long long gci = 0;
  for (int tb = blockIdx.x; tb < TB; tb += num_ctas) {
    const int t = tb / B;

    float m_running = -FLT_MAX;
    float s_running = 0.f;
    float acc32[ACC_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ACC_PER_THREAD; i++) {
      acc32[i] = 0.f;
    }
#pragma unroll
    for (int ci = 0; ci < num_chunks; ci++, gci++) {
      int ns = ci * N_CHUNK;
      int an = min(N_CHUNK, N - ns);
      int chunk_slot = (int)(gci % CHUNK_DEPTH);

      // ---- Load all `an` V rows (and delta if applicable) ----
      bool load_delta = false;
      int prefix_n = -1;
      if constexpr (HAS_DELTA) {
        prefix_n = N - 1 - ns;
        if (prefix_n >= 0 && prefix_n < an) {
          load_delta = true;
        }
      }
#pragma unroll
      for (int n = 0; n < N_CHUNK; n++) {
        if (n >= an) continue;
        int slot = slot_of(gci, n);
        const bf16_t* src =
            residual_addr(block_res, layer_res, ns + n, N, t, block_stride_m,
                          block_stride_r, H);
        cp_async_row(buf_ptr(slot), src, H * (int)sizeof(bf16_t));
      }
      if constexpr (HAS_DELTA) {
        if (load_delta) {
          cp_async_row(delta_buf_ptr(chunk_slot), delta + (long long)tb * H,
                       H * (int)sizeof(bf16_t));
        }
      }
      // Plain vectorized loads/stores land in shared memory before the
      // __syncthreads() below; no cp.async commit/wait is needed on MACA.
      __syncthreads();

      // ---- Pass A: sq + dot reductions, cache V in registers ----
      float2 sq_local[N_CHUNK] = {};
      float2 dot_local[N_CHUNK] = {};
      // Note: on SM80 we re-read V from smem in pass B (see pass_B_body) to
      // avoid relying on a register-resident V cache; the smem bufs are still
      // live because CHUNK_DEPTH=1 and no overwrite happens within a chunk.

      auto pass_A_body = [&](auto AN_TOK) {
        constexpr int AN = decltype(AN_TOK)::value;
#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
          if constexpr (H == 7168) {
            if (si == SLICES_PER_GROUP - 1) {
              int h_base =
                  6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
              const float* qv = &q_cache[si * VEC];
#pragma unroll
              for (int n = 0; n < AN; n++) {
                int slot = slot_of(gci, n);
                int2 vp =
                    *reinterpret_cast<const int2*>(buf_ptr(slot) + h_base);
                auto* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
                if constexpr (HAS_DELTA) {
                  if (n == prefix_n) {
                    const bf16_t* delta_ptr =
                        delta_buf_ptr(chunk_slot) + h_base;
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                      auto delta2 = *reinterpret_cast<const __nv_bfloat162*>(
                          delta_ptr + 2 * j);
                      v2[j] = __hadd2(v2[j], delta2);
                    }
                    // Write modified prefix+delta back to both the smem
                    // buffer (so Pass B's re-read sees prefix+delta instead
                    // of the original prefix) and layer_res (in-place update
                    // of the prefix tensor).
                    *reinterpret_cast<int2*>(buf_ptr(slot) + h_base) = vp;
                    *reinterpret_cast<int2*>(layer_res + (long long)tb * H +
                                             h_base) = vp;
                  }
                }
                float2 f[2] = {__bfloat1622float2(v2[0]),
                               __bfloat1622float2(v2[1])};
                sq_local[n] = float2_fma(f[0], f[0], sq_local[n]);
                sq_local[n] = float2_fma(f[1], f[1], sq_local[n]);
                dot_local[n] =
                    float2_fma(f[0], make_float2(qv[0], qv[1]), dot_local[n]);
                dot_local[n] =
                    float2_fma(f[1], make_float2(qv[2], qv[3]), dot_local[n]);
              }
              continue;
            }
          }
          int dt = si * CONSUMER_GROUPS + group;
          if (dt >= NHT) continue;
          int h_base = dt * K_TILE + k_local;
          const float* qv = &q_cache[si * VEC];

#pragma unroll
          for (int n = 0; n < AN; n++) {
            int slot = slot_of(gci, n);
            int4 vp = *reinterpret_cast<const int4*>(buf_ptr(slot) + h_base);
            auto* v2 = reinterpret_cast<__nv_bfloat162*>(&vp);
            if constexpr (HAS_DELTA) {
              if (n == prefix_n) {
                const bf16_t* delta_ptr = delta_buf_ptr(chunk_slot) + h_base;
#pragma unroll
                for (int j = 0; j < VEC / 2; j++) {
                  auto delta2 = *reinterpret_cast<const __nv_bfloat162*>(
                      delta_ptr + 2 * j);
                  v2[j] = __hadd2(v2[j], delta2);
                }
                // See note above: keep smem in sync so Pass B sees prefix+delta.
                *reinterpret_cast<int4*>(buf_ptr(slot) + h_base) = vp;
                *reinterpret_cast<int4*>(layer_res + (long long)tb * H +
                                         h_base) = vp;
              }
            }
            float2 f[4] = {
                __bfloat1622float2(v2[0]), __bfloat1622float2(v2[1]),
                __bfloat1622float2(v2[2]), __bfloat1622float2(v2[3])};
#pragma unroll
            for (int j = 0; j < VEC / 2; j++) {
              sq_local[n] = float2_fma(f[j], f[j], sq_local[n]);
              dot_local[n] = float2_fma(
                  f[j], make_float2(qv[2 * j], qv[2 * j + 1]), dot_local[n]);
            }
          }
        }
      };
      if constexpr (NC == 4) {
        switch (an) {
          case 4: pass_A_body(std::integral_constant<int, 4>{}); break;
          case 3: pass_A_body(std::integral_constant<int, 3>{}); break;
          case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
          case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
          default: __builtin_unreachable();
        }
      } else if constexpr (NC == 3) {
        switch (an) {
          case 3: pass_A_body(std::integral_constant<int, 3>{}); break;
          case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
          case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
          default: __builtin_unreachable();
        }
      } else {
        static_assert(NC == 2);
        switch (an) {
          case 2: pass_A_body(std::integral_constant<int, 2>{}); break;
          case 1: pass_A_body(std::integral_constant<int, 1>{}); break;
          default: __builtin_unreachable();
        }
      }

      // ---- Warp-level reduction of sq_local / dot_local across 32 lanes ----
      float2 reduce_pair[N_CHUNK];
#pragma unroll
      for (int n = 0; n < N_CHUNK; n++) {
        reduce_pair[n] = make_float2(sq_local[n].x + sq_local[n].y,
                                     dot_local[n].x + dot_local[n].y);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
        for (int n = 0; n < N_CHUNK; n++) {
          uint64_t packed = reinterpret_cast<uint64_t&>(reduce_pair[n]);
          packed = __shfl_xor_sync(0xffffffff, packed, offset);
          float2 other = reinterpret_cast<float2&>(packed);
          reduce_pair[n] = float2_add(reduce_pair[n], other);
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int n = 0; n < N_CHUNK; n++) {
          plan.ws_stats[wid][n] = reduce_pair[n];
        }
      }
      __syncthreads();

      // ---- Cross-warp reduction in the lower 128 threads ----
      float local_rsig = 0.f;
      float local_logit = 0.f;
      int stat_n = lane / CONSUMER_WARPS;
      int stat_w = lane % CONSUMER_WARPS;
      float2 totals = {};
      if (stat_n < N_CHUNK) {
        totals = plan.ws_stats[stat_w][stat_n];
      }
#pragma unroll
      for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1) {
        totals.x +=
            __shfl_down_sync(0xffffffff, totals.x, offset, CONSUMER_WARPS);
        totals.y +=
            __shfl_down_sync(0xffffffff, totals.y, offset, CONSUMER_WARPS);
      }
      if (stat_n < N_CHUNK && stat_w == 0) {
        local_rsig = rsqrtf(totals.x / H + rms_eps);
        local_logit = totals.y * local_rsig;
      }
      float logit_n[N_CHUNK];
#pragma unroll
      for (int n = 0; n < N_CHUNK; n++) {
        logit_n[n] = __shfl_sync(0xffffffff, local_logit, n * CONSUMER_WARPS);
      }

      // ---- Online softmax: update running m / s, compute weights ----
      float m_chunk = -FLT_MAX;
#pragma unroll
      for (int n = 0; n < N_CHUNK; n++) {
        if (n < an) m_chunk = fmaxf(m_chunk, logit_n[n]);
      }
      float m_new = fmaxf(m_running, m_chunk);
      float corr = exp2f((m_running - m_new) * LOG2_E);
      float w_n[N_CHUNK] = {};
      float w_sum = 0.f;
#pragma unroll
      for (int n = 0; n < N_CHUNK; n++) {
        if (n < an) {
          w_n[n] = exp2f((logit_n[n] - m_new) * LOG2_E);
          w_sum += w_n[n];
        }
      }

      // ---- Pass B: weighted accumulation from register V cache ----
      auto pass_B_body = [&](auto AN_TOK) {
        constexpr int AN = decltype(AN_TOK)::value;
#pragma unroll
        for (int si = 0; si < SLICES_PER_GROUP; si++) {
          if constexpr (H == 7168) {
            if (si == SLICES_PER_GROUP - 1) {
              int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
              float2 corr2 = make_float2(corr, corr);
              float2 a[2];
#pragma unroll
              for (int j = 0; j < 2; j++) {
                float2 old = make_float2(acc32[si * VEC + 2 * j],
                                         acc32[si * VEC + 2 * j + 1]);
                a[j] = float2_mul(old, corr2);
              }
#pragma unroll
              for (int n = 0; n < AN; n++) {
                int slot = slot_of(gci, n);
                float2 wn = make_float2(w_n[n], w_n[n]);
#pragma unroll
                for (int j = 0; j < 2; j++) {
                  int2 vp = *reinterpret_cast<const int2*>(
                      buf_ptr(slot) + h_base + 2 * j);
                  float2 f = __bfloat1622float2(
                      *reinterpret_cast<const __nv_bfloat162*>(&vp));
                  a[j] = float2_fma(wn, f, a[j]);
                }
              }
#pragma unroll
              for (int j = 0; j < 2; j++) {
                acc32[si * VEC + 2 * j] = a[j].x;
                acc32[si * VEC + 2 * j + 1] = a[j].y;
              }
              continue;
            }
          }
          int dt = si * CONSUMER_GROUPS + group;
          if (dt >= NHT) continue;
          int h_base = dt * K_TILE + k_local;
          float2 corr2 = make_float2(corr, corr);
          float2 a[VEC / 2];
#pragma unroll
          for (int j = 0; j < VEC / 2; j++) {
            float2 old = make_float2(acc32[si * VEC + 2 * j],
                                     acc32[si * VEC + 2 * j + 1]);
            a[j] = float2_mul(old, corr2);
          }
#pragma unroll
          for (int n = 0; n < AN; n++) {
            int slot = slot_of(gci, n);
            float2 wn = make_float2(w_n[n], w_n[n]);
            int4 vp = *reinterpret_cast<const int4*>(buf_ptr(slot) + h_base);
            auto* v2 = reinterpret_cast<const __nv_bfloat162*>(&vp);
#pragma unroll
            for (int j = 0; j < VEC / 2; j++) {
              float2 f = __bfloat1622float2(v2[j]);
              a[j] = float2_fma(wn, f, a[j]);
            }
          }
#pragma unroll
          for (int j = 0; j < VEC / 2; j++) {
            acc32[si * VEC + 2 * j] = a[j].x;
            acc32[si * VEC + 2 * j + 1] = a[j].y;
          }
        }
      };
      if constexpr (NC == 4) {
        switch (an) {
          case 4: pass_B_body(std::integral_constant<int, 4>{}); break;
          case 3: pass_B_body(std::integral_constant<int, 3>{}); break;
          case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
          case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
          default: __builtin_unreachable();
        }
      } else if constexpr (NC == 3) {
        switch (an) {
          case 3: pass_B_body(std::integral_constant<int, 3>{}); break;
          case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
          case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
          default: __builtin_unreachable();
        }
      } else {
        static_assert(NC == 2);
        switch (an) {
          case 2: pass_B_body(std::integral_constant<int, 2>{}); break;
          case 1: pass_B_body(std::integral_constant<int, 1>{}); break;
          default: __builtin_unreachable();
        }
      }

      s_running = s_running * corr + w_sum;
      m_running = m_new;
      // CHUNK_DEPTH=1: v_bufs are reused on the next chunk; consumers must
      // finish reading before the next iteration's cp.async overwrites them.
      __syncthreads();
    }

    // ---- Final output: divide by s_running (or fuse output RMSNorm) ----
    float inv_s = 1.f / s_running;
    bf16_t* out_ptr = output + (long long)tb * H;
    float2 output_sq_pair = {};
    // When output RMSNorm is fused, the softmax denominator cancels:
    //   out = (acc / s) * rsqrt(mean((acc/s)^2) + eps)
    //       = acc * rsqrt(mean(acc^2) + eps * s^2)
#pragma unroll
    for (int si = 0; si < SLICES_PER_GROUP; si++) {
      if constexpr (H == 7168) {
        if (si == SLICES_PER_GROUP - 1) {
          int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
          uint2 packed;
          auto* ov2 = reinterpret_cast<__nv_bfloat162*>(&packed);
          float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
          for (int j = 0; j < 2; j++) {
            float2 old = make_float2(acc32[si * VEC + 2 * j],
                                     acc32[si * VEC + 2 * j + 1]);
            if constexpr (HAS_OUTPUT_NORM) {
              output_sq_pair = float2_fma(old, old, output_sq_pair);
            } else {
              float2 mixed = float2_mul(old, inv2);
              ov2[j] = __float22bfloat162_rn(mixed);
            }
          }
          if constexpr (!HAS_OUTPUT_NORM) {
            *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
          }
          continue;
        }
      }
      int dt = si * CONSUMER_GROUPS + group;
      if (dt >= NHT) continue;
      int h_base = dt * K_TILE + k_local;
      uint4 packed;
      auto* ov2 = reinterpret_cast<__nv_bfloat162*>(&packed);
      float2 inv2 = make_float2(inv_s, inv_s);
#pragma unroll
      for (int j = 0; j < VEC / 2; j++) {
        float2 old =
            make_float2(acc32[si * VEC + 2 * j], acc32[si * VEC + 2 * j + 1]);
        if constexpr (HAS_OUTPUT_NORM) {
          output_sq_pair = float2_fma(old, old, output_sq_pair);
        } else {
          float2 mixed = float2_mul(old, inv2);
          ov2[j] = __float22bfloat162_rn(mixed);
        }
      }
      if constexpr (!HAS_OUTPUT_NORM) {
        *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
      }
    }

    if constexpr (HAS_OUTPUT_NORM) {
      float output_sq = output_sq_pair.x + output_sq_pair.y;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        output_sq += __shfl_xor_sync(0xffffffff, output_sq, offset);
      }
      if (lane == 0) {
        plan.ws_stats[wid][0] = make_float2(output_sq, 0.f);
      }
      __syncthreads();
      float total_sq = lane < CONSUMER_WARPS ? plan.ws_stats[lane][0].x : 0.f;
#pragma unroll
      for (int offset = CONSUMER_WARPS / 2; offset > 0; offset >>= 1) {
        total_sq +=
            __shfl_down_sync(0xffffffff, total_sq, offset, CONSUMER_WARPS);
      }
      if (lane == 0) {
        total_sq =
            rsqrtf(total_sq / H + output_norm_eps * s_running * s_running);
      }
      float output_rsigma = __shfl_sync(0xffffffff, total_sq, 0);
#pragma unroll
      for (int si = 0; si < SLICES_PER_GROUP; si++) {
        if constexpr (H == 7168) {
          if (si == SLICES_PER_GROUP - 1) {
            int h_base = 6 * K_TILE + group * (K_TILE / 2) + ct_in_group * 4;
            uint2 packed;
            auto* values = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
            for (int j = 0; j < 4; j++) {
              float weight = __bfloat162float(output_norm_weight[h_base + j]);
              values[j] = __float2bfloat16(acc32[si * VEC + j] *
                                         output_rsigma * weight);
            }
            *reinterpret_cast<uint2*>(out_ptr + h_base) = packed;
            continue;
          }
        }
        int dt = si * CONSUMER_GROUPS + group;
        if (dt >= NHT) continue;
        int h_base = dt * K_TILE + k_local;
        uint4 packed;
        auto* values = reinterpret_cast<bf16_t*>(&packed);
#pragma unroll
        for (int j = 0; j < VEC; j++) {
          float weight = __bfloat162float(output_norm_weight[h_base + j]);
          values[j] =
              __float2bfloat16(acc32[si * VEC + j] * output_rsigma * weight);
        }
        *reinterpret_cast<uint4*>(out_ptr + h_base) = packed;
      }
      __syncthreads();
    }
  }
#else
  if (threadIdx.x == 0) {
    printf("attn_res_fwd_online_v2_kernel requires sm_80+\n");
  }
#endif
}

template <int H, int N, int NC = N_CHUNK_DEFAULT, bool RELEASE_TMEM = false,
          bool HAS_DELTA = false, bool HAS_OUTPUT_NORM = false,
          bool OUTPUT_NORM_IN_SMEM = false>
static void launch_fwd(const bf16_t* block_residual, bf16_t* layer_residual,
                       const bf16_t* delta, const bf16_t* res_weight,
                       const bf16_t* rms_weight, bf16_t* output, int T,
                       float rms_eps, int num_sm, cudaStream_t stream,
                       const bf16_t* output_norm_weight = nullptr,
                       float output_norm_eps = 0.f, int block_stride_m = 0,
                       int block_stride_r = 0) {
  // SM80 shared-memory budget: NUM_BUFS * H * 2 + (HAS_DELTA ? H * 2 : 0) +
  // sizeof(FwdSmemPlan<NC>). With H=7168, NC=2, CHUNK_DEPTH=1 this is
  // 2*14336 + 14336 + ~256 = ~43 KB, well within C500's 64 KB / SM.
  constexpr size_t smem_size =
      ((size_t)CHUNK_DEPTH * (NC + (HAS_DELTA ? 1 : 0)) * H * sizeof(bf16_t) +
       sizeof(FwdSmemPlan<NC>) + 15) &
      ~size_t(15);

  auto kernel =
      &attn_res_fwd_online_v2_kernel<H, N, NC, 1, RELEASE_TMEM, HAS_DELTA,
                                     HAS_OUTPUT_NORM, OUTPUT_NORM_IN_SMEM>;

  static bool attrs_set = false;
  if (!attrs_set) {
    if (smem_size > 48 * 1024) {
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem_size));
    }
    attrs_set = true;
  }

  // One CTA per SM is enough; the kernel is persistent across (T*B) work
  // items.
  int grid = num_sm;

  kernel<<<grid, BLK, smem_size, stream>>>(
      block_residual, layer_residual, delta, res_weight, rms_weight, output,
      T, block_stride_m, block_stride_r, rms_eps, output_norm_weight,
      output_norm_eps);
}

}  // namespace fwd_prod_v2
}  // namespace sm80

void kimi_k3_attn_res(torch::Tensor& prefix,
                      torch::Tensor const& delta,
                      torch::Tensor const& blocks,
                      torch::Tensor const& norm_weight,
                      torch::Tensor const& qk_weight,
                      torch::Tensor const& output_norm_weight,
                      torch::Tensor& output, int64_t num_blocks,
                      double eps, double output_norm_eps) {
  int const num_tokens = static_cast<int>(prefix.size(0));
  int const device = prefix.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(device);
  cudaDeviceProp const* properties = at::cuda::getCurrentDeviceProperties();

  int64_t const hidden_size = prefix.size(1);
  TORCH_CHECK(prefix.stride(0) == hidden_size &&
                      delta.stride(0) == hidden_size &&
                      output.stride(0) == hidden_size,
                  "Kimi K3 AttnRes requires densely packed rows; got strides ",
                  prefix.stride(0), ", ", delta.stride(0), ", ",
                  output.stride(0), " for hidden_size ", hidden_size);

  using namespace sm80::fwd_prod_v2;
  // HAS_DELTA / HAS_OUTPUT_NORM are selected at runtime based on whether the
  // caller actually supplied those tensors. This keeps the kernel from
  // dereferencing an empty delta tensor (an ATU fault on Metax C500).
  bool const has_delta = delta.numel() > 0;
  bool const has_output_norm = output_norm_weight.numel() > 0;
  // On Metax C500 (sm_80) we always use NC=2, CHUNK_DEPTH=1,
  // OUTPUT_NORM_IN_SMEM=false to keep the shared memory footprint within
  // 64 KB.
  auto const bf_blocks = static_cast<bf16_t const*>(blocks.data_ptr());
  auto const bf_prefix_w = static_cast<bf16_t*>(prefix.data_ptr());
  auto const bf_delta =
      static_cast<bf16_t const*>(has_delta ? delta.data_ptr() : nullptr);
  auto const bf_qk_w = static_cast<bf16_t const*>(qk_weight.data_ptr());
  auto const bf_norm_w = static_cast<bf16_t const*>(norm_weight.data_ptr());
  auto const bf_out_w = static_cast<bf16_t*>(output.data_ptr());
  auto const bf_outnorm_w = static_cast<bf16_t const*>(
      has_output_norm ? output_norm_weight.data_ptr() : nullptr);
  int const N = static_cast<int>(num_blocks) + 1;
  int const num_sm = properties->multiProcessorCount;
  cudaStream_t const stream2 = at::cuda::getCurrentCUDAStream(device);
  int const bsm = static_cast<int>(blocks.stride(0));
  int const bsr = static_cast<int>(blocks.stride(1));

#define LAUNCH(N_VALUE, HAS_D, HAS_ON)                                               \
  launch_fwd<7168, N_VALUE, NC_FIXED, false, HAS_D, HAS_ON, false>(                  \
      bf_blocks, bf_prefix_w, bf_delta, bf_qk_w, bf_norm_w, bf_out_w,       \
      num_tokens, static_cast<float>(eps), num_sm, stream2, bf_outnorm_w,\
      static_cast<float>(output_norm_eps), bsm, bsr)

  switch (num_blocks) {
    case 1:
      if (has_delta && has_output_norm) {
        LAUNCH(2, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(2, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(2, false, true);
      } else {
        LAUNCH(2, false, false);
      }
      break;
    case 2:
      if (has_delta && has_output_norm) {
        LAUNCH(3, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(3, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(3, false, true);
      } else {
        LAUNCH(3, false, false);
      }
      break;
    case 3:
      if (has_delta && has_output_norm) {
        LAUNCH(4, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(4, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(4, false, true);
      } else {
        LAUNCH(4, false, false);
      }
      break;
    case 4:
      if (has_delta && has_output_norm) {
        LAUNCH(5, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(5, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(5, false, true);
      } else {
        LAUNCH(5, false, false);
      }
      break;
    case 5:
      if (has_delta && has_output_norm) {
        LAUNCH(6, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(6, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(6, false, true);
      } else {
        LAUNCH(6, false, false);
      }
      break;
    case 6:
      if (has_delta && has_output_norm) {
        LAUNCH(7, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(7, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(7, false, true);
      } else {
        LAUNCH(7, false, false);
      }
      break;
    case 7:
      if (has_delta && has_output_norm) {
        LAUNCH(8, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(8, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(8, false, true);
      } else {
        LAUNCH(8, false, false);
      }
      break;
    case 8:
      if (has_delta && has_output_norm) {
        LAUNCH(9, true, true);
      } else if (has_delta && !has_output_norm) {
        LAUNCH(9, true, false);
      } else if (!has_delta && has_output_norm) {
        LAUNCH(9, false, true);
      } else {
        LAUNCH(9, false, false);
      }
      break;
    default:
      TORCH_CHECK(false, "Kimi K3 AttnRes: num_blocks must be 1..8, got ", num_blocks);
  }
#undef LAUNCH
  cudaError_t const error = cudaGetLastError();
  TORCH_CHECK(
      error == cudaSuccess,
      "Kimi K3 AttnRes kernel launch failed: ", cudaGetErrorString(error));
}
