/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// AOT port of sglang_jit/fused_gemma_qknorm_rope.cuh (MiniMax-M3), C600-U tuned.
//
// Multi-group fused GemmaRMSNorm + partial NeoX RoPE, IN PLACE over `qkv`.
// Up to kMaxGroups=4 norm "groups" are fused into one launch; a group is a
// contiguous run of heads (within the per-token row) that share one norm weight
// and all receive RoPE. Heads outside every group (V / index-V) are untouched.
//
// Per (token, head), head_dim=128, rope_dim=64: read 128 bf16, RMSNorm (fp32
// accum) * (1 + weight), NeoX RoPE on the first 64 dims via a precomputed
// cos_sin_cache indexed by positions[token], pass through dims [64,128), write.
//
// ---- C600-U memory-bound tuning (see c600-optimization-guide.md 案例 F) ----
// The #1 lever for per-head norm is BYTES-PER-THREAD. We use kThreadsPerHead=4
// with 128-bit (float4 = 8 bf16) vectorized loads => 4 float4 rounds/thread =
// 32 elements = 128 B/thread, in a 128-thread block (32 heads/block). This is
// the measured sweet spot: 2 threads/head (256 B) spills registers and regresses.
//
// Key structural win: with kThreadsPerHead=4 and kVec=8, the per-thread element
// stride between rounds is kVec*kThreadsPerHead = 32 = ROPE_DIM/2. So for a lane
// owning dim j (round 0), its NeoX partner dim j+32 is the SAME lane's round-1
// element. RoPE therefore needs NO cross-lane shuffle -- both halves of every
// rotary pair are already in this thread's registers. Reduction over 4 lanes
// uses width-4 __shfl_xor_sync (<=16 => native fast subgroup path, no shared,
// no __syncthreads). rsqrt via the hardware reciprocal __builtin_mxc_rcpf.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cstdint>

namespace sgl_gemma_qknorm_rope {

constexpr int kMaxGroups = 4;

constexpr int kHeadDim = 128;
constexpr int kRopeDim = 64;
constexpr int kHalf = kRopeDim / 2;               // 32
constexpr int kVec = 8;                            // bf16 elems per 128-bit load
constexpr int kThreadsPerHead = 4;                 // 128 B/thread (sweet spot)
constexpr int kRounds = kHeadDim / kVec / kThreadsPerHead;  // 4 float4 rounds/thread
constexpr int kRoundStride = kVec * kThreadsPerHead;        // 32 == kHalf (rope pair local)
constexpr int kHeadsPerBlock = 32;
constexpr int kBlockThreads = kHeadsPerBlock * kThreadsPerHead;  // 128

static_assert(kRoundStride == kHalf, "rope-pair-local layout requires round stride == rope half");
static_assert(kRounds == 4, "expected 4 rounds for head_dim=128");

struct Params {
  __nv_bfloat16* __restrict__ qkv;
  const __nv_bfloat16* __restrict__ weight[kMaxGroups];
  uint32_t group_offset[kMaxGroups];
  uint32_t group_count[kMaxGroups];
  uint32_t num_groups;
  uint32_t total_heads;
  const float* __restrict__ cos_sin_cache;
  const void* __restrict__ positions;  // int32 or int64, selected by PosT
  uint32_t num_tokens;
  int64_t token_stride;
  float eps;
};

union Bf16x8 {
  uint4 raw;
  __nv_bfloat16 h[kVec];
};
union F32x4 {
  uint4 raw;
  float f[4];
};

// One head handled by kThreadsPerHead lanes; each lane owns kRounds float4s.
// Round r, element e -> dim = r*kRoundStride + tph*kVec + e.
//   round 0 -> [0,32)   rope first half
//   round 1 -> [32,64)  rope second half  (partner of round 0, same lane)
//   round 2 -> [64,96)  pass
//   round 3 -> [96,128) pass
template <typename PosT>
__global__ void __launch_bounds__(kBlockThreads, 8) fused_kernel(const Params params) {
  const uint32_t tx = threadIdx.x;
  const uint32_t tph = tx % kThreadsPerHead;
  const uint32_t work_id = (blockIdx.x * kBlockThreads + tx) / kThreadsPerHead;
  const uint32_t total_heads = params.total_heads;
  if (work_id >= params.num_tokens * total_heads) return;

  const uint32_t token_id = work_id / total_heads;
  const uint32_t grouped_head = work_id % total_heads;

  // Resolve group (num_groups <= 4, warp-uniform short scan).
  uint32_t g = 0, base = 0;
#pragma unroll
  for (uint32_t i = 0; i < kMaxGroups; ++i) {
    if (i >= params.num_groups) break;
    const uint32_t cnt = params.group_count[i];
    if (grouped_head < base + cnt) {
      g = i;
      break;
    }
    base += cnt;
  }
  const uint32_t local_head = grouped_head - base;
  const uint32_t head_id = params.group_offset[g] + local_head;

  __nv_bfloat16* __restrict__ input =
      params.qkv + (int64_t)token_id * params.token_stride + (int64_t)head_id * kHeadDim;
  const __nv_bfloat16* __restrict__ weight = params.weight[g];
  const uint32_t col0 = tph * kVec;  // first dim this lane owns in each round

  // Load input as packed bf16 (kept in regs), accumulate sum-of-squares in fp32.
  // Keep the fp32-converted inputs in registers (xf) so we never re-convert them
  // in the norm/rope math below -- bf16<->fp32 conversion is the arithmetic that
  // throttles this op below the memory ceiling (measured: norm-only 1276 GB/s 
  Bf16x8 xin[kRounds];
  float xf[kRounds][kVec];
  float ss = 0.0f;
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const uint32_t off = r * kRoundStride + col0;
    xin[r].raw = *reinterpret_cast<const uint4*>(input + off);
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      const float v = __bfloat162float(xin[r].h[e]);
      xf[r][e] = v;
      ss = fmaf(v, v, ss);
    }
  }
  // Reduce sum-of-squares across the kThreadsPerHead lanes (width 4 => fast path).
#pragma unroll
  for (int off = kThreadsPerHead / 2; off > 0; off >>= 1) {
    ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off, kThreadsPerHead);
  }
  const float inv_rms = __builtin_mxc_rcpf(sqrtf(ss / (float)kHeadDim + params.eps));

  // cos/sin for this lane's rope columns [col0, col0+8): cos at [col0], sin at
  // [col0+32]. Both are contiguous float4x2 loads from the (L2-resident) cache.
  const int64_t rope_idx = static_cast<int64_t>(static_cast<const PosT*>(params.positions)[token_id]);
  const float* cs = params.cos_sin_cache + rope_idx * kRopeDim;
  F32x4 cosv[kVec / 4], sinv[kVec / 4];
#pragma unroll
  for (int q = 0; q < kVec / 4; ++q) {
    cosv[q].raw = __ldg(reinterpret_cast<const uint4*>(cs + col0 + q * 4));
    sinv[q].raw = __ldg(reinterpret_cast<const uint4*>(cs + col0 + kHalf + q * 4));
  }

  // Norm + (rounds 0/1) rope, write back. Rounds 2/3 are norm-only pass-through.
  // weight is reused by every (token, head) sharing this group -> read-only cache.
  Bf16x8 wv[kRounds];
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const uint32_t off = r * kRoundStride + col0;
    wv[r].raw = __ldg(reinterpret_cast<const uint4*>(weight + off));
  }

  Bf16x8 out[kRounds];
  // rounds 0 (first rope half) and 1 (second rope half): fused rotate
#pragma unroll
  for (int e = 0; e < kVec; ++e) {
    const float n0 = xf[0][e] * inv_rms * (1.0f + __bfloat162float(wv[0].h[e]));
    const float n1 = xf[1][e] * inv_rms * (1.0f + __bfloat162float(wv[1].h[e]));
    const float c = cosv[e / 4].f[e % 4];
    const float s = sinv[e / 4].f[e % 4];
    out[0].h[e] = __float2bfloat16(fmaf(n0, c, -n1 * s));
    out[1].h[e] = __float2bfloat16(fmaf(n1, c,  n0 * s));
  }
  // rounds 2,3: norm-only pass-through
#pragma unroll
  for (int r = 2; r < kRounds; ++r) {
#pragma unroll
    for (int e = 0; e < kVec; ++e) {
      const float n = xf[r][e] * inv_rms * (1.0f + __bfloat162float(wv[r].h[e]));
      out[r].h[e] = __float2bfloat16(n);
    }
  }
#pragma unroll
  for (int r = 0; r < kRounds; ++r) {
    const uint32_t off = r * kRoundStride + col0;
    *reinterpret_cast<uint4*>(input + off) = out[r].raw;
  }
}

template <typename PosT>
void launch(const Params& params, cudaStream_t stream) {
  const int64_t heads = (int64_t)params.num_tokens * params.total_heads;
  if (heads == 0) return;
  const uint32_t grid = (uint32_t)((heads + kHeadsPerBlock - 1) / kHeadsPerBlock);
  fused_kernel<PosT><<<grid, kBlockThreads, 0, stream>>>(params);
}

}  // namespace sgl_gemma_qknorm_rope

void fused_gemma_qknorm_rope(
    at::Tensor qkv,
    at::Tensor w0,
    at::Tensor w1,
    at::Tensor w2,
    at::Tensor w3,
    at::Tensor cos_sin_cache,
    at::Tensor positions,
    int64_t off0,
    int64_t cnt0,
    int64_t off1,
    int64_t cnt1,
    int64_t off2,
    int64_t cnt2,
    int64_t off3,
    int64_t cnt3,
    int64_t num_groups,
    double eps) {
  using namespace sgl_gemma_qknorm_rope;
  TORCH_CHECK(qkv.is_cuda(), "qkv must be a CUDA tensor");
  TORCH_CHECK(qkv.dim() == 2, "qkv must be 2D [num_tokens, total_heads*head_dim]");
  TORCH_CHECK(qkv.scalar_type() == at::ScalarType::BFloat16, "qkv must be bfloat16");
  TORCH_CHECK(qkv.stride(1) == 1, "qkv last dim must be contiguous");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float, "cos_sin_cache must be float32");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == kRopeDim, "cos_sin_cache must be [*, 64]");
  TORCH_CHECK(num_groups >= 1 && num_groups <= kMaxGroups, "num_groups must be in [1,4]");
  for (const auto& w : {w0, w1, w2, w3}) {
    TORCH_CHECK(w.scalar_type() == at::ScalarType::BFloat16, "norm weight must be bfloat16");
    TORCH_CHECK(w.numel() == kHeadDim, "norm weight must have 128 elements (head_dim)");
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  Params params{};
  params.qkv = reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr());
  params.cos_sin_cache = cos_sin_cache.data_ptr<float>();
  params.positions = positions.data_ptr();
  params.num_tokens = static_cast<uint32_t>(qkv.size(0));
  params.num_groups = static_cast<uint32_t>(num_groups);
  params.token_stride = qkv.stride(0);
  params.eps = static_cast<float>(eps);

  const at::Tensor weights[kMaxGroups] = {w0, w1, w2, w3};
  const int64_t offsets[kMaxGroups] = {off0, off1, off2, off3};
  const int64_t counts[kMaxGroups] = {cnt0, cnt1, cnt2, cnt3};
  uint32_t total_heads = 0;
  for (int i = 0; i < kMaxGroups; ++i) {
    const int64_t cnt = (i < num_groups) ? counts[i] : 0;
    params.weight[i] = reinterpret_cast<const __nv_bfloat16*>(weights[i].data_ptr());
    params.group_offset[i] = static_cast<uint32_t>(i < num_groups ? offsets[i] : 0);
    params.group_count[i] = static_cast<uint32_t>(cnt);
    total_heads += static_cast<uint32_t>(cnt);
  }
  params.total_heads = total_heads;
  if (total_heads == 0 || params.num_tokens == 0) return;

  TORCH_CHECK(
      positions.numel() >= (int64_t)params.num_tokens, "positions must have at least num_tokens elements");
  if (positions.scalar_type() == at::ScalarType::Long) {
    launch<int64_t>(params, stream);
  } else if (positions.scalar_type() == at::ScalarType::Int) {
    launch<int32_t>(params, stream);
  } else {
    TORCH_CHECK(false, "positions must be int32 or int64");
  }
}
