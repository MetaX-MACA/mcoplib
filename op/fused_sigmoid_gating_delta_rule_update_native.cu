#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <optional>

#include "../include/fused_sigmoid_gating_delta_rule_update.h"

namespace {

using bf16 = __nv_bfloat16;

constexpr int kHeads = 64;
constexpr int kDim = 128;
constexpr int kSteps = 6;
constexpr int kPrecomputeThreads = 128;
constexpr int kRecurrentThreads = 256;
constexpr unsigned long long kFullMask = 0xffffffffffffffffULL;

__device__ __forceinline__ float sigmoidf_stable(float x) {
  return 1.0f / (1.0f + expf(-x));
}

template <int Width>
__device__ __forceinline__ float group_sum(float value) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    if constexpr (Width == 16) {
      // The C500 16-lane row form is one mov_b32 (row_shl) exchange; the
      // width-qualified shuffle lowers to sm_bperm_b32 plus selector math.
      value += __shfl_down_sync_16(kFullMask, value, offset);
    } else {
      value += __shfl_down_sync(kFullMask, value, offset, Width);
    }
  }
  return value;
}

// Sub-warp reduction over exactly the lanes of one thread group.  C500 dispatch
// is narrower than the 64-lane warp, so naming only the participating lanes
// avoids synchronising the whole warp on every reduction step.
template <int Width>
__device__ __forceinline__ unsigned long long group_mask() {
  unsigned long long kGroupBits = kFullMask;
  if constexpr (Width < 64) {
    kGroupBits = (1ULL << Width) - 1ULL;
  }
  int const subgroup = (static_cast<int>(threadIdx.x) & 63) / Width;
  return kGroupBits << (subgroup * Width);
}

template <int Width>
__device__ __forceinline__ float group_sum_masked(float value,
                                                  unsigned long long mask) {
#pragma unroll
  for (int offset = Width / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, Width);
  }
  return value;
}

// One block handles one (token, head).  This is the native counterpart of
// sigmoid_gating_precompute_kernel: gate decay, beta, and Q/K inverse norms
// are materialized into one flat workspace consumed by the recurrent launch.
__global__ void precompute_kernel(
    const float* __restrict__ A_log, const bf16* __restrict__ a,
    const float* __restrict__ dt_bias, const bf16* __restrict__ b,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    float* __restrict__ workspace, int tokens, int64_t stride_a,
    int64_t stride_b, int64_t stride_q, int64_t stride_k,
    float lower_bound) {
  int const token_head = blockIdx.x;
  int const token = token_head / kHeads;
  int const head = token_head - token * kHeads;
  int const ki = threadIdx.x;

  int64_t const gate_size = static_cast<int64_t>(tokens) * kHeads * kDim;
  int64_t const scalar_size = static_cast<int64_t>(tokens) * kHeads;
  float* const beta_values = workspace + gate_size;
  float* const q_inv_norm = beta_values + scalar_size;
  float* const k_inv_norm = q_inv_norm + scalar_size;

  float const aval = __bfloat162float(
      a[token * stride_a + static_cast<int64_t>(head) * kDim + ki]);
  float const x = aval + dt_bias[head * kDim + ki];
  float const gate =
      lower_bound * sigmoidf_stable(expf(A_log[head]) * x);
  workspace[static_cast<int64_t>(token_head) * kDim + ki] = expf(gate);

  float const qv = __bfloat162float(
      q[token * stride_q + static_cast<int64_t>(head) * kDim + ki]);
  float const kv = __bfloat162float(
      k[token * stride_k + static_cast<int64_t>(head) * kDim + ki]);
  float qsum = qv * qv;
  float ksum = kv * kv;
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    qsum += __shfl_down_sync(kFullMask, qsum, offset, 64);
    ksum += __shfl_down_sync(kFullMask, ksum, offset, 64);
  }

  __shared__ float partial_q[2];
  __shared__ float partial_k[2];
  int const lane = ki & 63;
  int const warp = ki >> 6;
  if (lane == 0) {
    partial_q[warp] = qsum;
    partial_k[warp] = ksum;
  }
  __syncthreads();
  if (ki == 0) {
    q_inv_norm[token_head] = rsqrtf(partial_q[0] + partial_q[1] + 1.0e-6f);
    k_inv_norm[token_head] = rsqrtf(partial_k[0] + partial_k[1] + 1.0e-6f);
    beta_values[token_head] = sigmoidf_stable(
        __bfloat162float(b[token * stride_b + head]));
  }
}

// T12: one 64-lane warp owns a token/head and two K elements per lane.
// Both Q/K halves stay in registers, so their norms need no CTA barrier.
// Keep T6 on precompute_kernel: its smaller grid already matches Triton.
__global__ __launch_bounds__(64)
void precompute_warp_kernel(
    const float* __restrict__ A_log, const bf16* __restrict__ a,
    const float* __restrict__ dt_bias, const bf16* __restrict__ b,
    const bf16* __restrict__ q, const bf16* __restrict__ k,
    float* __restrict__ workspace, int tokens, int64_t stride_a,
    int64_t stride_b, int64_t stride_q, int64_t stride_k,
    float lower_bound) {
  int const token_head = blockIdx.x;
  int const token = token_head / kHeads;
  int const head = token_head % kHeads;
  int const lane = threadIdx.x;
  int64_t const head_base = static_cast<int64_t>(head) * kDim;
  int64_t const gate_size = static_cast<int64_t>(tokens) * kHeads * kDim;
  int64_t const scalar_size = static_cast<int64_t>(tokens) * kHeads;
  float const alpha = expf(A_log[head]);
  float qsum = 0.0f;
  float ksum = 0.0f;
#pragma unroll
  for (int item = 0; item < 2; ++item) {
    int const ki = lane + item * 64;
    float const x = __bfloat162float(a[token * stride_a + head_base + ki]) +
        dt_bias[head_base + ki];
    workspace[static_cast<int64_t>(token_head) * kDim + ki] =
        expf(lower_bound * sigmoidf_stable(alpha * x));
    float const qv = __bfloat162float(q[token * stride_q + head_base + ki]);
    float const kv = __bfloat162float(k[token * stride_k + head_base + ki]);
    qsum += qv * qv;
    ksum += kv * kv;
  }
#pragma unroll
  for (int offset = 32; offset > 0; offset >>= 1) {
    qsum += __shfl_down_sync(kFullMask, qsum, offset, 64);
    ksum += __shfl_down_sync(kFullMask, ksum, offset, 64);
  }
  if (lane == 0) {
    workspace[gate_size + token_head] = sigmoidf_stable(
        __bfloat162float(b[token * stride_b + head]));
    workspace[gate_size + scalar_size + token_head] = rsqrtf(qsum + 1.0e-6f);
    workspace[gate_size + 2 * scalar_size + token_head] = rsqrtf(ksum + 1.0e-6f);
  }
}

void launch_precompute(
    const float* A_log, const bf16* a, const float* dt_bias, const bf16* b,
    const bf16* q, const bf16* k, float* workspace, int tokens,
    int64_t stride_a, int64_t stride_b, int64_t stride_q, int64_t stride_k,
    float lower_bound, cudaStream_t stream) {
  if (tokens == 12) {
    precompute_warp_kernel<<<tokens * kHeads, 64, 0, stream>>>(
        A_log, a, dt_bias, b, q, k, workspace, tokens, stride_a, stride_b,
        stride_q, stride_k, lower_bound);
  } else {
    precompute_kernel<<<tokens * kHeads, kPrecomputeThreads, 0, stream>>>(
        A_log, a, dt_bias, b, q, k, workspace, tokens, stride_a, stride_b,
        stride_q, stride_k, lower_bound);
  }
}

template <int BV, int GroupWidth>
__global__ __launch_bounds__(BV * GroupWidth)
void recurrent_kernel(
    const float* __restrict__ workspace, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, const float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ intermediate_states_buffer,
    const int32_t* __restrict__ intermediate_state_indices,
    int tokens, int64_t stride_h0_source, int64_t stride_q,
    int64_t stride_k, int64_t stride_v, float scale) {
  static_assert(BV * GroupWidth == kRecurrentThreads ||
                    BV * GroupWidth == 2 * kRecurrentThreads,
                "supported recurrent block size");
  constexpr int Items = kDim / GroupWidth;

  int const sequence_head = blockIdx.x;
  int const sequence = sequence_head / kHeads;
  int const head = sequence_head - sequence * kHeads;
  int const value_base = blockIdx.y * BV;
  int const tid = threadIdx.x;
  int const value_lane = tid / GroupWidth;
  int const lane = tid - value_lane * GroupWidth;
  int const value_index = value_base + value_lane;
  int const bos = cu_seqlens[sequence];
  int const state_slot = h0_indices[sequence];
  int const cache_slot = intermediate_state_indices[sequence];

  int64_t const gate_size = static_cast<int64_t>(tokens) * kHeads * kDim;
  int64_t const scalar_size = static_cast<int64_t>(tokens) * kHeads;
  const float* const beta_values = workspace + gate_size;
  const float* const q_inv_norm = beta_values + scalar_size;
  const float* const k_inv_norm = q_inv_norm + scalar_size;

  float state[Items];
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    int const ki = lane + item * GroupWidth;
    int64_t const state_offset =
        static_cast<int64_t>(head) * kDim * kDim +
        static_cast<int64_t>(value_index) * kDim + ki;
    state[item] = state_slot >= 0
        ? h0_source[static_cast<int64_t>(state_slot) * stride_h0_source +
                    state_offset]
        : 0.0f;
  }

  __shared__ float shared_q[kDim];
  __shared__ float shared_k[kDim];
  __shared__ float shared_decay[kDim];

#pragma unroll
  for (int step = 0; step < kSteps; ++step) {
    int const token = bos + step;
    int const token_head = token * kHeads + head;
    if (tid < kDim) {
      shared_q[tid] = __bfloat162float(
          q[token * stride_q + static_cast<int64_t>(head) * kDim + tid]) *
          q_inv_norm[token_head] * scale;
      shared_k[tid] = __bfloat162float(
          k[token * stride_k + static_cast<int64_t>(head) * kDim + tid]) *
          k_inv_norm[token_head];
      shared_decay[tid] = workspace[
          static_cast<int64_t>(token_head) * kDim + tid];
    }
    __syncthreads();

    float projection = 0.0f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      int const ki = lane + item * GroupWidth;
      state[item] *= shared_decay[ki];
      projection += state[item] * shared_k[ki];
    }
    projection = group_sum<GroupWidth>(projection);
    if (lane == 0) {
      float const raw_v = __bfloat162float(
          v[token * stride_v + static_cast<int64_t>(head) * kDim +
            value_index]);
      projection = (raw_v - projection) * beta_values[token_head];
    }
    float const delta =
        __shfl_sync(kFullMask, projection, 0, GroupWidth);

    float output = 0.0f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      int const ki = lane + item * GroupWidth;
      state[item] += delta * shared_k[ki];
      output += state[item] * shared_q[ki];
    }
    output = group_sum<GroupWidth>(output);
    if (lane == 0) {
      o[static_cast<int64_t>(token_head) * kDim + value_index] =
          __float2bfloat16(output);
    }

    if (cache_slot >= 0) {
      int64_t const cache_base =
          (static_cast<int64_t>(cache_slot) * kSteps * kHeads * kDim * kDim) +
          (static_cast<int64_t>(step) * kHeads * kDim * kDim) +
          (static_cast<int64_t>(head) * kDim * kDim) +
          (static_cast<int64_t>(value_index) * kDim);
#pragma unroll
      for (int item = 0; item < Items; ++item) {
        int const ki = lane + item * GroupWidth;
        intermediate_states_buffer[cache_base + ki] = state[item];
      }
    }
    // No thread may overwrite shared_q/shared_k/shared_decay for the next
    // step until every thread has consumed the current panels.  Without this
    // barrier, large grids expose a nondeterministic cross-step race.
    __syncthreads();
  }
}

// T12/BV32 mapping: each 16-lane group owns multiple value rows, while the
// whole CTA stages one common Q/K/decay panel.  This avoids the repeated global
// panel loads of multirow_kernel and reduces the BV32 thread count without
// changing the K-axis ownership or reduction order within a row.
template <int BV, int GroupWidth, int RowsPerGroup>
__global__ __launch_bounds__(GroupWidth * (BV / RowsPerGroup))
void recurrent_shared_multirow_kernel(
    const float* __restrict__ workspace, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, const float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ intermediate_states_buffer,
    const int32_t* __restrict__ intermediate_state_indices,
    int tokens, int64_t stride_h0_source, int64_t stride_q,
    int64_t stride_k, int64_t stride_v, float scale) {
  constexpr int Items = kDim / GroupWidth;
  constexpr int RowGroups = BV / RowsPerGroup;
  constexpr int Threads = GroupWidth * RowGroups;
  static_assert(BV == 32 && GroupWidth == 16 &&
                    (RowsPerGroup == 2 || RowsPerGroup == 4),
                "T12 shared multirow specialization");
  static_assert(Threads >= kDim, "enough threads to stage one panel");

  int const sequence_head = blockIdx.x;
  int const sequence = sequence_head / kHeads;
  int const head = sequence_head - sequence * kHeads;
  int const tid = threadIdx.x;
  int const value_group = tid / GroupWidth;
  int const lane = tid - value_group * GroupWidth;
  int const value_base = blockIdx.y * BV + value_group * RowsPerGroup;
  int const bos = cu_seqlens[sequence];
  int const state_slot = h0_indices[sequence];
  int const cache_slot = intermediate_state_indices[sequence];

  int64_t const gate_size = static_cast<int64_t>(tokens) * kHeads * kDim;
  int64_t const scalar_size = static_cast<int64_t>(tokens) * kHeads;
  const float* const beta_values = workspace + gate_size;
  const float* const q_inv_norm = beta_values + scalar_size;
  const float* const k_inv_norm = q_inv_norm + scalar_size;

  float state[RowsPerGroup][Items];
#pragma unroll
  for (int row = 0; row < RowsPerGroup; ++row) {
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      int const ki = lane + item * GroupWidth;
      int64_t const state_offset =
          static_cast<int64_t>(head) * kDim * kDim +
          static_cast<int64_t>(value_base + row) * kDim + ki;
      state[row][item] = state_slot >= 0
          ? h0_source[static_cast<int64_t>(state_slot) * stride_h0_source +
                      state_offset]
          : 0.0f;
    }
  }

  __shared__ float shared_q[kDim];
  __shared__ float shared_k[kDim];
  __shared__ float shared_decay[kDim];

  int token_head = bos * kHeads + head;
  bf16 const* q_step_ptr = q;
  bf16 const* k_step_ptr = k;
  bf16 const* v_step_ptr = v;
  if constexpr (RowsPerGroup == 2) {
    q_step_ptr = q + static_cast<int64_t>(bos) * stride_q +
                 static_cast<int64_t>(head) * kDim;
    k_step_ptr = k + static_cast<int64_t>(bos) * stride_k +
                 static_cast<int64_t>(head) * kDim;
    v_step_ptr = v + static_cast<int64_t>(bos) * stride_v +
                 static_cast<int64_t>(head) * kDim + value_base;
  }

#pragma unroll
  for (int step = 0; step < kSteps; ++step) {
    int const token = bos + step;
    if constexpr (RowsPerGroup == 2) {
      if (tid < kDim / 2) {
        int const ki = tid * 2;
        float2 const raw_q = __bfloat1622float2(
            *reinterpret_cast<__maca_bfloat162 const*>(
                q_step_ptr + ki));
        float const q_scale = q_inv_norm[token_head] * scale;
        *reinterpret_cast<float2*>(shared_q + ki) =
            make_float2(raw_q.x * q_scale, raw_q.y * q_scale);
      } else if (tid < kDim) {
        int const ki = (tid - kDim / 2) * 2;
        float2 const raw_k = __bfloat1622float2(
            *reinterpret_cast<__maca_bfloat162 const*>(
                k_step_ptr + ki));
        float const k_scale = k_inv_norm[token_head];
        *reinterpret_cast<float2*>(shared_k + ki) =
            make_float2(raw_k.x * k_scale, raw_k.y * k_scale);
      } else {
        int const ki = tid - kDim;
        shared_decay[ki] =
            workspace[static_cast<int64_t>(token_head) * kDim + ki];
      }
    } else if (tid < kDim) {
      shared_q[tid] = __bfloat162float(
          q[token * stride_q + static_cast<int64_t>(head) * kDim + tid]) *
          q_inv_norm[token_head] * scale;
      shared_k[tid] = __bfloat162float(
          k[token * stride_k + static_cast<int64_t>(head) * kDim + tid]) *
          k_inv_norm[token_head];
      shared_decay[tid] =
          workspace[static_cast<int64_t>(token_head) * kDim + tid];
    }
    __syncthreads();

    float projection[RowsPerGroup];
#pragma unroll
    for (int row = 0; row < RowsPerGroup; ++row) {
      projection[row] = 0.0f;
#pragma unroll
      for (int item = 0; item < Items; ++item) {
        int const ki = lane + item * GroupWidth;
        state[row][item] *= shared_decay[ki];
        projection[row] += state[row][item] * shared_k[ki];
      }
      projection[row] = group_sum<GroupWidth>(projection[row]);
    }
    if (lane == 0) {
      float const beta = beta_values[token_head];
      if constexpr (RowsPerGroup == 2) {
        float2 const raw_v = __bfloat1622float2(
            *reinterpret_cast<__maca_bfloat162 const*>(v_step_ptr));
        projection[0] = (raw_v.x - projection[0]) * beta;
        projection[1] = (raw_v.y - projection[1]) * beta;
      } else {
#pragma unroll
        for (int row = 0; row < RowsPerGroup; ++row) {
          float const raw_v = __bfloat162float(
              v[token * stride_v + static_cast<int64_t>(head) * kDim +
                value_base + row]);
          projection[row] = (raw_v - projection[row]) * beta;
        }
      }
    }

    float delta[RowsPerGroup];
#pragma unroll
    for (int row = 0; row < RowsPerGroup; ++row) {
      if constexpr (RowsPerGroup == 2) {
        delta[row] = __shfl_sync_16(kFullMask, projection[row], 0);
      } else {
        delta[row] =
            __shfl_sync(kFullMask, projection[row], 0, GroupWidth);
      }
    }

    float output[RowsPerGroup];
#pragma unroll
    for (int row = 0; row < RowsPerGroup; ++row) {
      output[row] = 0.0f;
#pragma unroll
      for (int item = 0; item < Items; ++item) {
        int const ki = lane + item * GroupWidth;
        state[row][item] += delta[row] * shared_k[ki];
        output[row] += state[row][item] * shared_q[ki];
      }
    }

    if (cache_slot >= 0) {
      if constexpr (RowsPerGroup == 2) {
        int64_t const cache_base =
            (static_cast<int64_t>(cache_slot) * kSteps * kHeads * kDim * kDim) +
            (static_cast<int64_t>(step) * kHeads * kDim * kDim) +
            (static_cast<int64_t>(head) * kDim * kDim) +
            (static_cast<int64_t>(value_base) * kDim) + lane;
        int32_t* const cache_ptr = reinterpret_cast<int32_t*>(
            intermediate_states_buffer + cache_base);
#define STORE_STATE_OFFSET(ROW, ITEM)                                      \
        __builtin_mxc_stg_b32_predicator(                                 \
            cache_ptr,                                                    \
            ((ROW) * kDim + (ITEM) * GroupWidth) * sizeof(float),         \
            __float_as_int(state[(ROW)][(ITEM)]), true, false, false,     \
            true, 1, MACA_ICMP_EQ)
        STORE_STATE_OFFSET(0, 0);
        STORE_STATE_OFFSET(0, 1);
        STORE_STATE_OFFSET(0, 2);
        STORE_STATE_OFFSET(0, 3);
        STORE_STATE_OFFSET(0, 4);
        STORE_STATE_OFFSET(0, 5);
        STORE_STATE_OFFSET(0, 6);
        STORE_STATE_OFFSET(0, 7);
        STORE_STATE_OFFSET(1, 0);
        STORE_STATE_OFFSET(1, 1);
        STORE_STATE_OFFSET(1, 2);
        STORE_STATE_OFFSET(1, 3);
        STORE_STATE_OFFSET(1, 4);
        STORE_STATE_OFFSET(1, 5);
        STORE_STATE_OFFSET(1, 6);
        STORE_STATE_OFFSET(1, 7);
#undef STORE_STATE_OFFSET
      } else {
#pragma unroll
        for (int row = 0; row < RowsPerGroup; ++row) {
          int64_t const cache_base =
              (static_cast<int64_t>(cache_slot) * kSteps * kHeads * kDim *
               kDim) +
              (static_cast<int64_t>(step) * kHeads * kDim * kDim) +
              (static_cast<int64_t>(head) * kDim * kDim) +
              (static_cast<int64_t>(value_base + row) * kDim);
#pragma unroll
          for (int item = 0; item < Items; ++item) {
            int const ki = lane + item * GroupWidth;
            intermediate_states_buffer[cache_base + ki] = state[row][item];
          }
        }
      }
    }

#pragma unroll
    for (int row = 0; row < RowsPerGroup; ++row) {
      output[row] = group_sum<GroupWidth>(output[row]);
    }
    if (lane == 0) {
      if constexpr (RowsPerGroup == 2) {
        bf16* const output_ptr =
            o + static_cast<int64_t>(token_head) * kDim + value_base;
        *reinterpret_cast<__maca_bfloat162*>(output_ptr) =
            __float22bfloat162_rn(make_float2(output[0], output[1]));
      } else {
#pragma unroll
        for (int row = 0; row < RowsPerGroup; ++row) {
          o[static_cast<int64_t>(token_head) * kDim + value_base + row] =
              __float2bfloat16(output[row]);
        }
      }
    }
    __syncthreads();
    token_head += kHeads;
    if constexpr (RowsPerGroup == 2) {
      q_step_ptr += stride_q;
      k_step_ptr += stride_k;
      v_step_ptr += stride_v;
    }
  }
}

// Load VL consecutive bf16 values with the widest vector the tail allows.
template <int VL>
__device__ __forceinline__ void load_bf16_row(bf16 const* __restrict__ source,
                                              float* __restrict__ out) {
  union { uint4 wide; __nv_bfloat16 half[8]; } raw;
  if constexpr (VL == 8) {
    raw.wide = *reinterpret_cast<uint4 const*>(source);
  } else {
    static_assert(VL == 4, "vector variant supports VL of 4 or 8");
    uint2 const half = *reinterpret_cast<uint2 const*>(source);
    raw.wide = make_uint4(half.x, half.y, 0u, 0u);
  }
#pragma unroll
  for (int item = 0; item < VL; ++item) {
    out[item] = __bfloat162float(raw.half[item]);
  }
}

// Register-resident variant of recurrent_kernel.  Each lane owns VL consecutive
// members of the contiguous K axis of one value row, so Q/K/decay and the state
// tile move in 16-byte accesses, the state tile never leaves registers, and the
// launch needs no shared-memory staging and no block barrier.  The arithmetic,
// the reduction depth (log2(GroupWidth) shuffles) and the value-row ownership
// match recurrent_kernel exactly; only the k-to-lane assignment differs, which
// changes the summation order of the K reduction.
template <int BV, int GroupWidth>
__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_vector_kernel(
    const float* __restrict__ workspace, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, const float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ intermediate_states_buffer,
    const int32_t* __restrict__ intermediate_state_indices, int tokens,
    int64_t stride_h0_source, int64_t stride_q, int64_t stride_k,
    int64_t stride_v, float scale) {
  static_assert(BV * GroupWidth == kRecurrentThreads,
                "one thread group is assigned to each value row");
  constexpr int VL = kDim / GroupWidth;
  static_assert(VL == 4 || VL == 8, "vector variant supports VL of 4 or 8");
  static_assert(kDim % VL == 0, "the K axis must tile by VL");

  union FloatArray {
    float f[VL];
    float4 v[VL / 4];
  };

  int const sequence_head = blockIdx.x;
  int const sequence = sequence_head / kHeads;
  int const head = sequence_head - sequence * kHeads;
  int const value_base = blockIdx.y * BV;
  int const tid = threadIdx.x;
  int const value_lane = tid / GroupWidth;
  int const lane = tid - value_lane * GroupWidth;
  int const value_index = value_base + value_lane;
  int const k_base = lane * VL;
  int const bos = cu_seqlens[sequence];
  int const state_slot = h0_indices[sequence];
  int const cache_slot = intermediate_state_indices[sequence];

  int64_t const gate_size = static_cast<int64_t>(tokens) * kHeads * kDim;
  int64_t const scalar_size = static_cast<int64_t>(tokens) * kHeads;
  float const* const beta_values = workspace + gate_size;
  float const* const q_inv_norm = beta_values + scalar_size;
  float const* const k_inv_norm = q_inv_norm + scalar_size;

  FloatArray state;
#pragma unroll
  for (int item = 0; item < VL / 4; ++item) {
    state.v[item] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  int64_t const state_offset = static_cast<int64_t>(head) * kDim * kDim +
      static_cast<int64_t>(value_index) * kDim + k_base;
  if (state_slot >= 0) {
    float4 const* const source = reinterpret_cast<float4 const*>(
        h0_source + static_cast<int64_t>(state_slot) * stride_h0_source +
        state_offset);
#pragma unroll
    for (int item = 0; item < VL / 4; ++item) {
      state.v[item] = source[item];
    }
  }

  for (int step = 0; step < kSteps; ++step) {
    int const token = bos + step;
    int const token_head = token * kHeads + head;

    FloatArray decay, q_values, k_values;
    float4 const* const decay_ptr = reinterpret_cast<float4 const*>(
        workspace + static_cast<int64_t>(token_head) * kDim + k_base);
#pragma unroll
    for (int item = 0; item < VL / 4; ++item) {
      decay.v[item] = decay_ptr[item];
    }
    load_bf16_row<VL>(
        q + token * stride_q + static_cast<int64_t>(head) * kDim + k_base,
        q_values.f);
    load_bf16_row<VL>(
        k + token * stride_k + static_cast<int64_t>(head) * kDim + k_base,
        k_values.f);

    float const q_scale = q_inv_norm[token_head] * scale;
    float const k_scale = k_inv_norm[token_head];
    float const beta = beta_values[token_head];

    float projection = 0.0f;
#pragma unroll
    for (int item = 0; item < VL; ++item) {
      state.f[item] *= decay.f[item];
      k_values.f[item] *= k_scale;
      q_values.f[item] *= q_scale;
      projection += state.f[item] * k_values.f[item];
    }
    projection = group_sum<GroupWidth>(projection);
    if (lane == 0) {
      float const raw_v = __bfloat162float(
          v[token * stride_v + static_cast<int64_t>(head) * kDim +
            value_index]);
      projection = (raw_v - projection) * beta;
    }
    float const delta =
        __shfl_sync(kFullMask, projection, 0, GroupWidth);

    float output = 0.0f;
#pragma unroll
    for (int item = 0; item < VL; ++item) {
      state.f[item] += delta * k_values.f[item];
      output += state.f[item] * q_values.f[item];
    }
    output = group_sum<GroupWidth>(output);
    if (lane == 0) {
      o[static_cast<int64_t>(token_head) * kDim + value_index] =
          __float2bfloat16(output);
    }

    if (cache_slot >= 0) {
      int64_t const cache_base =
          (static_cast<int64_t>(cache_slot) * kSteps * kHeads * kDim * kDim) +
          (static_cast<int64_t>(step) * kHeads * kDim * kDim) +
          (static_cast<int64_t>(head) * kDim * kDim) +
          (static_cast<int64_t>(value_index) * kDim) + k_base;
      float4* const dest =
          reinterpret_cast<float4*>(intermediate_states_buffer + cache_base);
#pragma unroll
      for (int item = 0; item < VL / 4; ++item) {
        dest[item] = state.v[item];
      }
    }
  }
}

// Double-buffered variant of recurrent_kernel: the next step's Q/K/decay panel
// is staged into the spare buffer before the current step's arithmetic, so the
// panel load overlaps the computation and each step needs one barrier instead
// of two.  Arithmetic, K-to-lane assignment, reduction depth and value-row
// ownership are identical to recurrent_kernel.
template <int BV, int GroupWidth>
__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_pipelined_kernel(
    const float* __restrict__ workspace, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, const float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ intermediate_states_buffer,
    const int32_t* __restrict__ intermediate_state_indices, int tokens,
    int64_t stride_h0_source, int64_t stride_q, int64_t stride_k,
    int64_t stride_v, float scale) {
  static_assert(BV * GroupWidth == kRecurrentThreads, "value rows per group");
  constexpr int Items = kDim / GroupWidth;

  int const sequence_head = blockIdx.x;
  int const sequence = sequence_head / kHeads;
  int const head = sequence_head - sequence * kHeads;
  int const value_base = blockIdx.y * BV;
  int const tid = threadIdx.x;
  int const value_lane = tid / GroupWidth;
  int const lane = tid - value_lane * GroupWidth;
  int const value_index = value_base + value_lane;
  int const bos = cu_seqlens[sequence];
  int const state_slot = h0_indices[sequence];
  int const cache_slot = intermediate_state_indices[sequence];

  int64_t const gate_size = static_cast<int64_t>(tokens) * kHeads * kDim;
  int64_t const scalar_size = static_cast<int64_t>(tokens) * kHeads;
  const float* const beta_values = workspace + gate_size;
  const float* const q_inv_norm = beta_values + scalar_size;
  const float* const k_inv_norm = q_inv_norm + scalar_size;

  float state[Items];
#pragma unroll
  for (int item = 0; item < Items; ++item) {
    int const ki = lane + item * GroupWidth;
    int64_t const state_offset =
        static_cast<int64_t>(head) * kDim * kDim +
        static_cast<int64_t>(value_index) * kDim + ki;
    state[item] = state_slot >= 0
        ? h0_source[static_cast<int64_t>(state_slot) * stride_h0_source +
                    state_offset]
        : 0.0f;
  }

  __shared__ float shared_q[2][kDim];
  __shared__ float shared_k[2][kDim];
  __shared__ float shared_decay[2][kDim];
  unsigned long long const lane_mask = group_mask<GroupWidth>();

  auto stage_panel = [&](int step, int buffer) {
    int const token = bos + step;
    int const token_head = token * kHeads + head;
    if (tid < kDim) {
      shared_q[buffer][tid] =
          __bfloat162float(q[token * stride_q +
                             static_cast<int64_t>(head) * kDim + tid]) *
          q_inv_norm[token_head] * scale;
      shared_k[buffer][tid] =
          __bfloat162float(k[token * stride_k +
                             static_cast<int64_t>(head) * kDim + tid]) *
          k_inv_norm[token_head];
      shared_decay[buffer][tid] =
          workspace[static_cast<int64_t>(token_head) * kDim + tid];
    }
  };

  stage_panel(0, 0);
  __syncthreads();

#pragma unroll
  for (int step = 0; step < kSteps; ++step) {
    int const buffer = step & 1;
    if (step + 1 < kSteps) {
      stage_panel(step + 1, buffer ^ 1);
    }
    int const token = bos + step;
    int const token_head = token * kHeads + head;

    float projection = 0.0f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      int const ki = lane + item * GroupWidth;
      state[item] *= shared_decay[buffer][ki];
      projection += state[item] * shared_k[buffer][ki];
    }
    projection = group_sum_masked<GroupWidth>(projection, lane_mask);
    if (lane == 0) {
      float const raw_v = __bfloat162float(
          v[token * stride_v + static_cast<int64_t>(head) * kDim +
            value_index]);
      projection = (raw_v - projection) * beta_values[token_head];
    }
    float const delta = __shfl_sync(lane_mask, projection, 0, GroupWidth);

    float output = 0.0f;
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      int const ki = lane + item * GroupWidth;
      state[item] += delta * shared_k[buffer][ki];
      output += state[item] * shared_q[buffer][ki];
    }
    output = group_sum_masked<GroupWidth>(output, lane_mask);
    if (lane == 0) {
      o[static_cast<int64_t>(token_head) * kDim + value_index] =
          __float2bfloat16(output);
    }
    if (cache_slot >= 0) {
      int64_t const cache_base =
          (static_cast<int64_t>(cache_slot) * kSteps * kHeads * kDim * kDim) +
          (static_cast<int64_t>(step) * kHeads * kDim * kDim) +
          (static_cast<int64_t>(head) * kDim * kDim) +
          (static_cast<int64_t>(value_index) * kDim);
#pragma unroll
      for (int item = 0; item < Items; ++item) {
        intermediate_states_buffer[cache_base + lane + item * GroupWidth] =
            state[item];
      }
    }
    __syncthreads();
  }
}

// Per-element panel traffic is VL-scaled and row-shared; the state stays in
// registers and is written back as vectors.
template <int BV, int GroupWidth, int VR>
__global__ __launch_bounds__(kRecurrentThreads)
void recurrent_multirow_kernel(
    const float* __restrict__ workspace, const bf16* __restrict__ q,
    const bf16* __restrict__ k, const bf16* __restrict__ v,
    bf16* __restrict__ o, const float* __restrict__ h0_source,
    const int32_t* __restrict__ h0_indices,
    const int32_t* __restrict__ cu_seqlens,
    float* __restrict__ intermediate_states_buffer,
    const int32_t* __restrict__ intermediate_state_indices, int tokens,
    int64_t stride_h0_source, int64_t stride_q, int64_t stride_k,
    int64_t stride_v, float scale) {
  constexpr int VL = kDim / GroupWidth;
  constexpr int RowGroups = BV / VR;
  static_assert(GroupWidth * RowGroups == kRecurrentThreads,
                "the block must cover exactly RowGroups value groups");
  static_assert(VL == 4 || VL == 8, "multi-row variant supports VL of 4 or 8");

  union FloatArray {
    float f[VL];
    float4 v[VL / 4];
  };

  int const sequence_head = blockIdx.x;
  int const sequence = sequence_head / kHeads;
  int const head = sequence_head - sequence * kHeads;
  int const value_group = threadIdx.x / GroupWidth;
  int const lane = threadIdx.x - value_group * GroupWidth;
  int const value_base = blockIdx.y * BV + value_group * VR;
  int const k_base = lane * VL;
  int const bos = cu_seqlens[sequence];
  int const state_slot = h0_indices[sequence];
  int const cache_slot = intermediate_state_indices[sequence];

  int64_t const gate_size = static_cast<int64_t>(tokens) * kHeads * kDim;
  int64_t const scalar_size = static_cast<int64_t>(tokens) * kHeads;
  float const* const beta_values = workspace + gate_size;
  float const* const q_inv_norm = beta_values + scalar_size;
  float const* const k_inv_norm = q_inv_norm + scalar_size;

  FloatArray state[VR];
#pragma unroll
  for (int row = 0; row < VR; ++row) {
#pragma unroll
    for (int item = 0; item < VL / 4; ++item) {
      state[row].v[item] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (state_slot >= 0) {
      int64_t const offset =
          static_cast<int64_t>(head) * kDim * kDim +
          static_cast<int64_t>(value_base + row) * kDim + k_base;
      float4 const* const source = reinterpret_cast<float4 const*>(
          h0_source + static_cast<int64_t>(state_slot) * stride_h0_source +
          offset);
#pragma unroll
      for (int item = 0; item < VL / 4; ++item) {
        state[row].v[item] = source[item];
      }
    }
  }

  for (int step = 0; step < kSteps; ++step) {
    int const token = bos + step;
    int const token_head = token * kHeads + head;

    FloatArray decay, q_values, k_values;
    float4 const* const decay_ptr = reinterpret_cast<float4 const*>(
        workspace + static_cast<int64_t>(token_head) * kDim + k_base);
#pragma unroll
    for (int item = 0; item < VL / 4; ++item) {
      decay.v[item] = decay_ptr[item];
    }
    load_bf16_row<VL>(
        q + token * stride_q + static_cast<int64_t>(head) * kDim + k_base,
        q_values.f);
    load_bf16_row<VL>(
        k + token * stride_k + static_cast<int64_t>(head) * kDim + k_base,
        k_values.f);
    float const q_scale = q_inv_norm[token_head] * scale;
    float const k_scale = k_inv_norm[token_head];

#pragma unroll
    for (int item = 0; item < VL; ++item) {
      q_values.f[item] *= q_scale;
      k_values.f[item] *= k_scale;
    }

    float projection[VR];
#pragma unroll
    for (int row = 0; row < VR; ++row) {
      projection[row] = 0.0f;
#pragma unroll
      for (int item = 0; item < VL; ++item) {
        state[row].f[item] *= decay.f[item];
        projection[row] += state[row].f[item] * k_values.f[item];
      }
      projection[row] = group_sum<GroupWidth>(projection[row]);
    }
    if (lane == 0) {
      float const beta = beta_values[token_head];
#pragma unroll
      for (int row = 0; row < VR; ++row) {
        float const raw_v = __bfloat162float(
            v[token * stride_v + static_cast<int64_t>(head) * kDim +
              value_base + row]);
        projection[row] = (raw_v - projection[row]) * beta;
      }
    }
    float delta[VR];
#pragma unroll
    for (int row = 0; row < VR; ++row) {
      delta[row] = __shfl_sync(kFullMask, projection[row], 0, GroupWidth);
    }

    float output[VR];
#pragma unroll
    for (int row = 0; row < VR; ++row) {
      output[row] = 0.0f;
#pragma unroll
      for (int item = 0; item < VL; ++item) {
        state[row].f[item] += delta[row] * k_values.f[item];
        output[row] += state[row].f[item] * q_values.f[item];
      }
      output[row] = group_sum<GroupWidth>(output[row]);
    }
    if (lane == 0) {
#pragma unroll
      for (int row = 0; row < VR; ++row) {
        o[static_cast<int64_t>(token_head) * kDim + value_base + row] =
            __float2bfloat16(output[row]);
      }
    }

    if (cache_slot >= 0) {
#pragma unroll
      for (int row = 0; row < VR; ++row) {
        int64_t const cache_base =
            (static_cast<int64_t>(cache_slot) * kSteps * kHeads * kDim * kDim) +
            (static_cast<int64_t>(step) * kHeads * kDim * kDim) +
            (static_cast<int64_t>(head) * kDim * kDim) +
            (static_cast<int64_t>(value_base + row) * kDim) + k_base;
        float4* const dest =
            reinterpret_cast<float4*>(intermediate_states_buffer + cache_base);
#pragma unroll
        for (int item = 0; item < VL / 4; ++item) {
          dest[item] = state[row].v[item];
        }
      }
    }
  }
}

void check_tensor(torch::Tensor const& tensor, at::ScalarType dtype,
                  char const* name) {
  TORCH_CHECK(tensor.defined() && tensor.is_cuda(), name,
              " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has dtype ",
              tensor.scalar_type(), ", expected ", dtype);
}

struct LaunchArgs {
  torch::Tensor workspace;
  int tokens;
  int sequences;
};

struct RecurrentArgs {
  float const* workspace;
  bf16 const* q;
  bf16 const* k;
  bf16 const* v;
  bf16* o;
  float const* h0_source;
  int32_t const* h0_indices;
  int32_t const* cu_seqlens;
  float* intermediate_states_buffer;
  int32_t const* intermediate_state_indices;
  int tokens;
  int64_t stride_h0_source;
  int64_t stride_q;
  int64_t stride_k;
  int64_t stride_v;
  float scale;
};

template <int BV, int GroupWidth>
void launch_recurrent(dim3 const& grid, cudaStream_t stream,
                      RecurrentArgs const& args) {
  recurrent_kernel<BV, GroupWidth><<<grid, BV * GroupWidth, 0, stream>>>(
      args.workspace, args.q, args.k, args.v, args.o, args.h0_source,
      args.h0_indices, args.cu_seqlens, args.intermediate_states_buffer,
      args.intermediate_state_indices, args.tokens, args.stride_h0_source,
      args.stride_q, args.stride_k, args.stride_v, args.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int BV, int GroupWidth, int RowsPerGroup>
void launch_recurrent_shared_multirow(dim3 const& grid, cudaStream_t stream,
                                      RecurrentArgs const& args) {
  constexpr int Threads = GroupWidth * (BV / RowsPerGroup);
  recurrent_shared_multirow_kernel<BV, GroupWidth, RowsPerGroup>
      <<<grid, Threads, 0, stream>>>(
          args.workspace, args.q, args.k, args.v, args.o, args.h0_source,
          args.h0_indices, args.cu_seqlens, args.intermediate_states_buffer,
          args.intermediate_state_indices, args.tokens, args.stride_h0_source,
          args.stride_q, args.stride_k, args.stride_v, args.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// BV is the number of value rows per block.  The standard tile assigns one
// thread group per row; the T12/BV32 specialization assigns four rows per
// group and launches 128 threads.
void launch_recurrent_for(dim3 const& grid, cudaStream_t stream,
                          RecurrentArgs const& args, int64_t value_block) {
  switch (value_block) {
    case 4:
      launch_recurrent<4, 64>(grid, stream, args);
      break;
    case 8:
      launch_recurrent<8, 32>(grid, stream, args);
      break;
    case 16:
      launch_recurrent<16, 16>(grid, stream, args);
      break;
    case 32:
      // N2 balances four waves against two value rows per lane group while
      // preserving one shared panel load and the 16-lane K reduction order.
      if (args.tokens == 12) {
        launch_recurrent_shared_multirow<32, 16, 2>(grid, stream, args);
      } else {
        launch_recurrent<32, 8>(grid, stream, args);
      }
      break;
    default:
      TORCH_CHECK(false, "value_block must be 4, 8, 16 or 32, got ",
                  value_block);
  }
}

template <int BV, int GroupWidth>
void launch_recurrent_vector(dim3 const& grid, cudaStream_t stream,
                             RecurrentArgs const& args) {
  recurrent_vector_kernel<BV, GroupWidth>
      <<<grid, kRecurrentThreads, 0, stream>>>(
          args.workspace, args.q, args.k, args.v, args.o, args.h0_source,
          args.h0_indices, args.cu_seqlens, args.intermediate_states_buffer,
          args.intermediate_state_indices, args.tokens, args.stride_h0_source,
          args.stride_q, args.stride_k, args.stride_v, args.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The vectorized variant fixes the block at 256 threads, so the group width
// follows from the value block: 16 rows / 16 lanes (VL 8) or 8 rows / 32 lanes
// (VL 4).  Larger value blocks would need VL 16, which the variant rejects.
void launch_recurrent_vector_for(dim3 const& grid, cudaStream_t stream,
                                 RecurrentArgs const& args,
                                 int64_t value_block) {
  if (value_block == 16) {
    launch_recurrent_vector<16, 16>(grid, stream, args);
  } else {
    TORCH_CHECK(value_block == 8,
                "vectorized variant supports value_block 8 or 16, got ",
                value_block);
    launch_recurrent_vector<8, 32>(grid, stream, args);
  }
}

template <int BV, int GroupWidth, int VR>
void launch_recurrent_multirow(dim3 const& grid, cudaStream_t stream,
                               RecurrentArgs const& args) {
  recurrent_multirow_kernel<BV, GroupWidth, VR>
      <<<grid, kRecurrentThreads, 0, stream>>>(
          args.workspace, args.q, args.k, args.v, args.o, args.h0_source,
          args.h0_indices, args.cu_seqlens, args.intermediate_states_buffer,
          args.intermediate_state_indices, args.tokens, args.stride_h0_source,
          args.stride_q, args.stride_k, args.stride_v, args.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Multi-row variant keeps GroupWidth 16 (VL 8) and derives the rows per lane
// from the value block: VR = value_block * GroupWidth / kRecurrentThreads.
template <int BV, int GroupWidth>
void launch_recurrent_pipelined(dim3 const& grid, cudaStream_t stream,
                                RecurrentArgs const& args) {
  recurrent_pipelined_kernel<BV, GroupWidth>
      <<<grid, kRecurrentThreads, 0, stream>>>(
          args.workspace, args.q, args.k, args.v, args.o, args.h0_source,
          args.h0_indices, args.cu_seqlens, args.intermediate_states_buffer,
          args.intermediate_state_indices, args.tokens, args.stride_h0_source,
          args.stride_q, args.stride_k, args.stride_v, args.scale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_recurrent_pipelined_for(dim3 const& grid, cudaStream_t stream,
                                    RecurrentArgs const& args,
                                    int64_t value_block) {
  if (value_block == 16) {
    launch_recurrent_pipelined<16, 16>(grid, stream, args);
  } else {
    TORCH_CHECK(value_block == 32,
                "pipelined variant supports value_block 16 or 32, got ",
                value_block);
    launch_recurrent_pipelined<32, 8>(grid, stream, args);
  }
}

void launch_recurrent_multirow_for(dim3 const& grid, cudaStream_t stream,
                                   RecurrentArgs const& args,
                                   int64_t value_block) {
  constexpr int GroupWidth = 16;
  if (value_block == 16) {
    launch_recurrent_multirow<16, GroupWidth, 1>(grid, stream, args);
  } else if (value_block == 32) {
    launch_recurrent_multirow<32, GroupWidth, 2>(grid, stream, args);
  } else {
    TORCH_CHECK(value_block == 64,
                "multi-row variant supports value_block 16, 32 or 64, got ",
                value_block);
    launch_recurrent_multirow<64, GroupWidth, 4>(grid, stream, args);
  }
}

RecurrentArgs make_recurrent_args(
    torch::Tensor const& workspace, torch::Tensor const& q,
    torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& o,
    torch::Tensor const& h0_source, torch::Tensor const& h0_indices,
    torch::Tensor const& cu_seqlens,
    torch::Tensor const& intermediate_states_buffer,
    torch::Tensor const& intermediate_state_indices, int64_t stride_h0_source,
    int64_t stride_q, int64_t stride_k, int64_t stride_v, float scale,
    int tokens) {
  return RecurrentArgs{
      workspace.data_ptr<float>(),
      reinterpret_cast<bf16 const*>(q.data_ptr()),
      reinterpret_cast<bf16 const*>(k.data_ptr()),
      reinterpret_cast<bf16 const*>(v.data_ptr()),
      reinterpret_cast<bf16*>(o.data_ptr()),
      h0_source.data_ptr<float>(),
      h0_indices.data_ptr<int32_t>(),
      cu_seqlens.data_ptr<int32_t>(),
      intermediate_states_buffer.data_ptr<float>(),
      intermediate_state_indices.data_ptr<int32_t>(),
      tokens,
      stride_h0_source,
      stride_q,
      stride_k,
      stride_v,
      scale};
}

LaunchArgs validate_and_make_workspace(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, double softplus_beta,
    double softplus_threshold, double lower_bound, torch::Tensor const& q,
    torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& b,
    torch::Tensor const& o, torch::Tensor const& h0_source,
    torch::Tensor const& h0_indices, int64_t stride_h0_source,
    torch::Tensor const& cu_seqlens,
    torch::Tensor const& intermediate_states_buffer,
    torch::Tensor const& intermediate_state_indices, int64_t cache_steps,
    std::optional<torch::Tensor> const& retrieve_parent_token_ptr, double scale,
    int64_t T, int64_t stride_a, int64_t stride_q, int64_t stride_k,
    int64_t stride_v, int64_t stride_b, int64_t NP2_T, int64_t B, int64_t H,
    int64_t HV, int64_t K, int64_t V, int64_t BK, bool USE_INITIAL_STATE,
    bool USE_QK_L2NORM_IN_KERNEL, bool IS_VARLEN, bool IS_KDA,
    bool USE_LOWER_BOUND, bool DISABLE_STATE_UPDATE,
    bool CACHE_INTERMEDIATE_STATES, bool HAS_EAGLE_TREE_CUSTOM_ATTN_MASK,
    bool CACHE_RING, std::optional<torch::Tensor> const& workspace_arg) {
  check_tensor(A_log, at::kFloat, "A_log");
  check_tensor(a, at::kBFloat16, "a");
  check_tensor(dt_bias, at::kFloat, "dt_bias");
  check_tensor(q, at::kBFloat16, "q");
  check_tensor(k, at::kBFloat16, "k");
  check_tensor(v, at::kBFloat16, "v");
  check_tensor(b, at::kBFloat16, "b");
  check_tensor(o, at::kBFloat16, "o");
  check_tensor(h0_source, at::kFloat, "h0_source");
  check_tensor(h0_indices, at::kInt, "h0_indices");
  check_tensor(cu_seqlens, at::kInt, "cu_seqlens");
  check_tensor(intermediate_states_buffer, at::kFloat,
               "intermediate_states_buffer");
  check_tensor(intermediate_state_indices, at::kInt,
               "intermediate_state_indices");

  TORCH_CHECK(T == 6 || T == 12,
              "native fused sigmoid gating currently supports T=6 or T=12");
  int const sequences = static_cast<int>(T / kSteps);
  TORCH_CHECK(B == 1 && H == kHeads && HV == kHeads && K == kDim &&
                  V == kDim && BK == kDim,
              "native fused sigmoid gating requires B=1, H=HV=64, K=V=BK=128");
  TORCH_CHECK(softplus_beta == 1.0 && softplus_threshold == 20.0 &&
                  lower_bound == -5.0,
              "native fused sigmoid gating requires beta=1, threshold=20, "
              "lower_bound=-5");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "scale must be finite and positive");
  TORCH_CHECK(stride_a == a.stride(1) && stride_q == q.stride(1) &&
                  stride_k == k.stride(1) && stride_v == v.stride(1) &&
                  stride_b == b.stride(1),
              "explicit input strides must match tensor strides");
  TORCH_CHECK(stride_h0_source == h0_source.stride(0),
              "stride_h0_source must match h0_source.stride(0)");
  TORCH_CHECK(NP2_T >= T && cache_steps == kSteps,
              "native fused sigmoid gating requires NP2_T>=T and cache_steps=6");
  TORCH_CHECK(USE_INITIAL_STATE && USE_QK_L2NORM_IN_KERNEL && IS_VARLEN &&
                  IS_KDA && USE_LOWER_BOUND && DISABLE_STATE_UPDATE &&
                  CACHE_INTERMEDIATE_STATES &&
                  !HAS_EAGLE_TREE_CUSTOM_ATTN_MASK && !CACHE_RING &&
                  !retrieve_parent_token_ptr.has_value(),
              "unsupported feature flags for native target-verify kernel");
  TORCH_CHECK(A_log.numel() == kHeads &&
                  a.numel() == T * kHeads * kDim &&
                  dt_bias.numel() == kHeads * kDim &&
                  q.numel() == T * kHeads * kDim &&
                  k.numel() == T * kHeads * kDim &&
                  v.numel() == T * kHeads * kDim &&
                  b.numel() == T * kHeads &&
                  o.numel() == T * kHeads * kDim,
              "input/output shapes do not match the captured target contract");
  TORCH_CHECK(q.stride(-1) == 1 && q.stride(-2) == kDim &&
                  k.stride(-1) == 1 && k.stride(-2) == kDim &&
                  v.stride(-1) == 1 && v.stride(-2) == kDim,
              "Q/K/V head payloads must be dense (packed token stride is allowed)");
  TORCH_CHECK(o.is_contiguous() && a.is_contiguous() && b.is_contiguous() &&
                  dt_bias.is_contiguous() && A_log.is_contiguous() &&
                  h0_indices.is_contiguous() && cu_seqlens.is_contiguous() &&
                  intermediate_states_buffer.is_contiguous() &&
                  intermediate_state_indices.is_contiguous(),
              "native target-verify tensors other than packed Q/K/V must be contiguous");
  TORCH_CHECK(h0_source.stride(-1) == 1 &&
                  h0_source.stride(-2) == kDim &&
                  h0_source.stride(-3) == kDim * kDim,
              "h0_source state payload must be dense");
  TORCH_CHECK(h0_indices.numel() >= sequences &&
                  cu_seqlens.numel() >= sequences + 1 &&
                  intermediate_state_indices.numel() >= sequences,
              "index tensors are too small for the packed sequences");
  TORCH_CHECK(intermediate_states_buffer.size(1) == kSteps &&
                  intermediate_states_buffer.size(2) == kHeads &&
                  intermediate_states_buffer.size(3) == kDim &&
                  intermediate_states_buffer.size(4) == kDim,
              "intermediate state buffer must have shape [slots,6,64,128,128]");
  auto const device = q.device();
  for (auto const& tensor : {A_log, a, dt_bias, k, v, b, o, h0_source,
                             h0_indices, cu_seqlens,
                             intermediate_states_buffer,
                             intermediate_state_indices}) {
    TORCH_CHECK(tensor.device() == device, "all tensors must share q's device");
  }

  int64_t const workspace_numel =
      static_cast<int64_t>(T) * kHeads * (kDim + 3);
  torch::Tensor workspace;
  if (workspace_arg.has_value()) {
    // Preallocated workspace: identical layout to the allocating path and to
    // the Triton workspace base (gate decay, beta, q inverse norm, k inverse
    // norm, all FP32).
    workspace = *workspace_arg;
    TORCH_CHECK(workspace.defined() && workspace.is_cuda(),
                "workspace must be a CUDA tensor");
    TORCH_CHECK(workspace.scalar_type() == at::kFloat,
                "workspace must be float32");
    TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
    TORCH_CHECK(workspace.device() == device,
                "workspace must share q's device");
    TORCH_CHECK(workspace.numel() >= workspace_numel,
                "workspace needs at least ", workspace_numel, " elements");
  } else {
    workspace = torch::empty({workspace_numel}, q.options().dtype(at::kFloat));
  }
  return {workspace, static_cast<int>(T), sequences};
}

// N1 retains the measured BV16/256-thread optimum.  N2 tests BV32 with 512
// threads so GroupWidth stays 16 and each lane still owns eight state values;
// unlike the rejected BV32/256-thread tile, this does not double lane state.
int64_t production_value_block(int tokens) {
  return tokens == 12 ? 32 : 16;
}

void launch_native(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, torch::Tensor const& q,
    torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& b,
    torch::Tensor const& o, torch::Tensor const& h0_source,
    torch::Tensor const& h0_indices, torch::Tensor const& cu_seqlens,
    torch::Tensor const& intermediate_states_buffer,
    torch::Tensor const& intermediate_state_indices, int64_t stride_h0_source,
    int64_t stride_a, int64_t stride_q, int64_t stride_k, int64_t stride_v,
    int64_t stride_b, float scale, float lower_bound,
    LaunchArgs const& args, int64_t value_block) {
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  launch_precompute(
      A_log.data_ptr<float>(), reinterpret_cast<bf16 const*>(a.data_ptr()),
      dt_bias.data_ptr<float>(), reinterpret_cast<bf16 const*>(b.data_ptr()),
      reinterpret_cast<bf16 const*>(q.data_ptr()),
      reinterpret_cast<bf16 const*>(k.data_ptr()),
      args.workspace.data_ptr<float>(), args.tokens, stride_a, stride_b,
      stride_q, stride_k, lower_bound, stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  RecurrentArgs const recurrent = make_recurrent_args(
      args.workspace, q, k, v, o, h0_source, h0_indices, cu_seqlens,
      intermediate_states_buffer, intermediate_state_indices, stride_h0_source,
      stride_q, stride_k, stride_v, scale, args.tokens);
  int64_t const block =
      value_block == 0 ? production_value_block(args.tokens) : value_block;
  dim3 const grid(args.sequences * kHeads,
                  kDim / static_cast<int>(block));
  launch_recurrent_for(grid, stream, recurrent, block);
}

}  // namespace

void fused_sigmoid_gating_delta_rule_update_native_out(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, double softplus_beta,
    double softplus_threshold, double lower_bound, torch::Tensor const& q,
    torch::Tensor const& k, torch::Tensor const& v, torch::Tensor const& b,
    torch::Tensor o, torch::Tensor const& h0_source,
    torch::Tensor const& h0_indices, int64_t stride_h0_source,
    torch::Tensor const& cu_seqlens, torch::Tensor intermediate_states_buffer,
    torch::Tensor const& intermediate_state_indices, int64_t cache_steps,
    std::optional<torch::Tensor> const& retrieve_parent_token_ptr,
    int64_t stride_retrieve_parent_token_seq,
    int64_t stride_retrieve_parent_token_token, double scale, int64_t T,
    int64_t stride_a, int64_t stride_q, int64_t stride_k, int64_t stride_v,
    int64_t stride_b, int64_t NP2_T, int64_t B, int64_t H, int64_t HV,
    int64_t K, int64_t V, int64_t BK, int64_t BV,
    bool USE_INITIAL_STATE, bool USE_QK_L2NORM_IN_KERNEL, bool IS_VARLEN,
    bool IS_KDA, bool USE_LOWER_BOUND, bool DISABLE_STATE_UPDATE,
    bool CACHE_INTERMEDIATE_STATES, bool HAS_EAGLE_TREE_CUSTOM_ATTN_MASK,
    std::optional<torch::Tensor> const& replayssm_rawv,
    std::optional<torch::Tensor> const& replayssm_rawk,
    std::optional<torch::Tensor> const& replayssm_g,
    std::optional<torch::Tensor> const& replayssm_beta,
    int64_t stride_rawv_slot, int64_t stride_rawk_slot,
    int64_t stride_g_slot, int64_t stride_beta_slot, int64_t MAX_CACHE_LEN,
    bool CACHE_RING, bool SPLIT_N_HV_GRID,
    std::optional<torch::Tensor> const& workspace, int64_t value_block) {
  (void)stride_retrieve_parent_token_seq;
  (void)stride_retrieve_parent_token_token;
  (void)BV;
  (void)replayssm_rawv;
  (void)replayssm_rawk;
  (void)replayssm_g;
  (void)replayssm_beta;
  (void)stride_rawv_slot;
  (void)stride_rawk_slot;
  (void)stride_g_slot;
  (void)stride_beta_slot;
  (void)MAX_CACHE_LEN;
  (void)SPLIT_N_HV_GRID;
  at::cuda::OptionalCUDAGuard const guard(device_of(q));
  auto const args = validate_and_make_workspace(
      A_log, a, dt_bias, softplus_beta, softplus_threshold, lower_bound, q, k,
      v, b, o, h0_source, h0_indices, stride_h0_source, cu_seqlens,
      intermediate_states_buffer, intermediate_state_indices, cache_steps,
      retrieve_parent_token_ptr, scale, T, stride_a, stride_q, stride_k,
      stride_v, stride_b, NP2_T, B, H, HV, K, V, BK, USE_INITIAL_STATE,
      USE_QK_L2NORM_IN_KERNEL, IS_VARLEN, IS_KDA, USE_LOWER_BOUND,
      DISABLE_STATE_UPDATE, CACHE_INTERMEDIATE_STATES,
      HAS_EAGLE_TREE_CUSTOM_ATTN_MASK, CACHE_RING, workspace);
  launch_native(A_log, a, dt_bias, q, k, v, b, o, h0_source, h0_indices,
                cu_seqlens, intermediate_states_buffer,
                intermediate_state_indices, stride_h0_source, stride_a,
                stride_q, stride_k, stride_v, stride_b,
                static_cast<float>(scale), static_cast<float>(lower_bound),
                args, value_block);
}

torch::Tensor fused_sigmoid_gating_delta_rule_update_native(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, double softplus_beta,
    double softplus_threshold, torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& v, torch::Tensor const& b,
    torch::Tensor const& initial_state_source,
    torch::Tensor const& initial_state_indices, std::optional<double> scale,
    bool use_qk_l2norm_in_kernel,
    std::optional<torch::Tensor> const& cu_seqlens, bool is_kda,
    std::optional<double> lower_bound, bool disable_state_update,
    std::optional<torch::Tensor> const& intermediate_states_buffer,
    std::optional<torch::Tensor> const& intermediate_state_indices,
    std::optional<int64_t> cache_steps,
    std::optional<torch::Tensor> const& retrieve_parent_token, bool cache_ring,
    std::optional<torch::Tensor> const& replayssm_rawv,
    std::optional<torch::Tensor> const& replayssm_rawk,
    std::optional<torch::Tensor> const& replayssm_g,
    std::optional<torch::Tensor> const& replayssm_beta) {
  TORCH_CHECK(cu_seqlens.has_value() &&
                  intermediate_states_buffer.has_value() &&
                  intermediate_state_indices.has_value(),
              "native target-verify entry requires sequence and cache tensors");
  int64_t const T = k.size(1);
  auto output5 = torch::empty({1, 1, T, kHeads, kDim}, q.options());
  fused_sigmoid_gating_delta_rule_update_native_out(
      A_log, a, dt_bias, softplus_beta, softplus_threshold,
      lower_bound.value_or(0.0), q, k, v, b, output5, initial_state_source,
      initial_state_indices, initial_state_source.stride(0), *cu_seqlens,
      *intermediate_states_buffer, *intermediate_state_indices,
      cache_steps.value_or(intermediate_states_buffer->size(1)),
      retrieve_parent_token, 0, 0, scale.value_or(1.0 / std::sqrt(kDim)), T,
      a.stride(1), q.stride(1), k.stride(1), v.stride(1), b.stride(1),
      T == 6 ? 8 : 16, 1, kHeads, kHeads, kDim, kDim, kDim, 32, true,
      use_qk_l2norm_in_kernel, true, is_kda, lower_bound.has_value(),
      disable_state_update, true, retrieve_parent_token.has_value(),
      replayssm_rawv, replayssm_rawk, replayssm_g, replayssm_beta, 0, 0, 0, 0,
      0, cache_ring, true, std::nullopt, production_value_block(static_cast<int>(T)));
  return output5.squeeze(0);
}

// Diagnostic entry: materialize only the precompute workspace.  The layout,
// the validate-then-launch contract and the kernel are the production ones;
// this exists so the two production launches can be timed separately.
void fused_sigmoid_gating_delta_rule_update_native_precompute(
    torch::Tensor const& A_log, torch::Tensor const& a,
    torch::Tensor const& dt_bias, torch::Tensor const& b,
    torch::Tensor const& q, torch::Tensor const& k,
    torch::Tensor const& workspace, double lower_bound, int64_t stride_a,
    int64_t stride_b, int64_t stride_q, int64_t stride_k, int64_t T) {
  check_tensor(A_log, at::kFloat, "A_log");
  check_tensor(a, at::kBFloat16, "a");
  check_tensor(dt_bias, at::kFloat, "dt_bias");
  check_tensor(b, at::kBFloat16, "b");
  check_tensor(q, at::kBFloat16, "q");
  check_tensor(k, at::kBFloat16, "k");
  check_tensor(workspace, at::kFloat, "workspace");
  TORCH_CHECK(T == 6 || T == 12, "precompute supports T=6 or T=12");
  TORCH_CHECK(lower_bound == -5.0, "precompute requires lower_bound=-5");
  TORCH_CHECK(stride_a == a.stride(1) && stride_q == q.stride(1) &&
                  stride_k == k.stride(1) && stride_b == b.stride(1),
              "explicit input strides must match tensor strides");
  TORCH_CHECK(workspace.is_contiguous() && workspace.device() == q.device(),
              "workspace must be contiguous and on q's device");
  TORCH_CHECK(workspace.numel() >= T * kHeads * (kDim + 3),
              "workspace is too small for T=", T);
  TORCH_CHECK(A_log.numel() == kHeads && a.numel() == T * kHeads * kDim &&
                  k.numel() == T * kHeads * kDim &&
                  b.numel() == T * kHeads && q.numel() == T * kHeads * kDim,
              "input shapes do not match the captured target contract");
  at::cuda::OptionalCUDAGuard const guard(device_of(q));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  launch_precompute(
      A_log.data_ptr<float>(), reinterpret_cast<bf16 const*>(a.data_ptr()),
      dt_bias.data_ptr<float>(), reinterpret_cast<bf16 const*>(b.data_ptr()),
      reinterpret_cast<bf16 const*>(q.data_ptr()),
      reinterpret_cast<bf16 const*>(k.data_ptr()),
      workspace.data_ptr<float>(), static_cast<int>(T), stride_a, stride_b,
      stride_q, stride_k, static_cast<float>(lower_bound), stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Diagnostic entry: run only the recurrent launch against an already
// materialized workspace.  value_block selects the production tile (16 for T=6,
// 32 for T=12) or the alternative tile for a like-for-like configuration test.
void fused_sigmoid_gating_delta_rule_update_native_recurrent(
    torch::Tensor const& workspace, torch::Tensor const& q,
    torch::Tensor const& k, torch::Tensor const& v, torch::Tensor o,
    torch::Tensor const& h0_source, torch::Tensor const& h0_indices,
    torch::Tensor const& cu_seqlens,
    torch::Tensor intermediate_states_buffer,
    torch::Tensor const& intermediate_state_indices, int64_t stride_h0_source,
    int64_t stride_q, int64_t stride_k, int64_t stride_v, double scale,
    int64_t T, int64_t value_block, int64_t variant) {
  check_tensor(workspace, at::kFloat, "workspace");
  check_tensor(q, at::kBFloat16, "q");
  check_tensor(k, at::kBFloat16, "k");
  check_tensor(v, at::kBFloat16, "v");
  check_tensor(o, at::kBFloat16, "o");
  check_tensor(h0_source, at::kFloat, "h0_source");
  check_tensor(h0_indices, at::kInt, "h0_indices");
  check_tensor(cu_seqlens, at::kInt, "cu_seqlens");
  check_tensor(intermediate_states_buffer, at::kFloat,
               "intermediate_states_buffer");
  check_tensor(intermediate_state_indices, at::kInt,
               "intermediate_state_indices");
  TORCH_CHECK(T == 6 || T == 12, "recurrent supports T=6 or T=12");
  int64_t const block =
      value_block == 0 ? production_value_block(static_cast<int>(T)) : value_block;
  TORCH_CHECK(variant >= 0 && variant <= 3,
              "variant must be 0 (shared staging), 1 (vectorized), 2 "
              "(multi-row) or 3 (pipelined)");
  TORCH_CHECK(std::isfinite(scale) && scale > 0.0,
              "scale must be finite and positive");
  TORCH_CHECK(stride_q == q.stride(1) && stride_k == k.stride(1) &&
                  stride_v == v.stride(1) &&
                  stride_h0_source == h0_source.stride(0),
              "explicit strides must match tensor strides");
  TORCH_CHECK(q.numel() == T * kHeads * kDim && k.numel() == T * kHeads * kDim &&
                  v.numel() == T * kHeads * kDim && o.numel() == T * kHeads * kDim,
              "Q/K/V/output shapes do not match the captured target contract");
  TORCH_CHECK(workspace.is_contiguous() && workspace.numel() >= T * kHeads * (kDim + 3),
              "workspace is too small or not contiguous for T=", T);
  TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
  TORCH_CHECK(intermediate_states_buffer.is_contiguous() &&
                  intermediate_states_buffer.size(1) == kSteps &&
                  intermediate_states_buffer.size(2) == kHeads &&
                  intermediate_states_buffer.size(3) == kDim &&
                  intermediate_states_buffer.size(4) == kDim,
              "intermediate state buffer must have shape [slots,6,64,128,128]");
  at::cuda::OptionalCUDAGuard const guard(device_of(q));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  RecurrentArgs const recurrent = make_recurrent_args(
      workspace, q, k, v, o, h0_source, h0_indices, cu_seqlens,
      intermediate_states_buffer, intermediate_state_indices, stride_h0_source,
      stride_q, stride_k, stride_v, static_cast<float>(scale),
      static_cast<int>(T));
  dim3 const grid((static_cast<int>(T) / kSteps) * kHeads,
                  kDim / static_cast<int>(block));
  if (variant == 1) {
    launch_recurrent_vector_for(grid, stream, recurrent, block);
  } else if (variant == 2) {
    TORCH_CHECK(block == 16 || block == 32 || block == 64,
                "multi-row variant supports value_block 16, 32 or 64, got ",
                block);
    launch_recurrent_multirow_for(grid, stream, recurrent, block);
  } else if (variant == 3) {
    TORCH_CHECK(block == 16 || block == 32,
                "pipelined variant supports value_block 16 or 32, got ", block);
    launch_recurrent_pipelined_for(grid, stream, recurrent, block);
  } else {
    TORCH_CHECK(block == 4 || block == 8 || block == 16 || block == 32,
                "staged variant supports value_block 4, 8, 16 or 32, got ",
                block);
    launch_recurrent_for(grid, stream, recurrent, block);
  }
}
