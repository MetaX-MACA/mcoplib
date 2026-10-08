// Retained CUDA implementation of chunk_gated_delta_rule_fwd_h (K=V=128, BT=64).
//
// Derived from the standalone probe at
// experiments/cuda_optimized/chunk_gated_delta_rule/chunk_gated_delta_rule_probe.cu
// (sha256 214be7c64a04cdeeebc0504909626b4f59582d9a3b90b3aa71203c5bd8fcf1ce),
// records "step 19" in that directory's README. The probe's CPU reference, its
// scalar/MMA variants, and its align/bench drivers are deliberately not carried
// over; what stays is the native kernel, its helpers, and the host-side coverage
// predicate that mirrors the production Triton dispatch.
//
// Instantiated for Pool=true, Gqa x Exp2 x Ragged x StateBf16,
// Streaming=false, Bv=64.
// The Pool=true form is used for every call: with identity slot indices and the
// tensor's own stride(0) it reproduces the dense addressing exactly, and it also
// covers the pooled/sentinel contract and any padded slot pitch.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "../include/chunk_gated_delta_rule_fwd_h.h"

namespace {

// ---- constants and type alias (probe) ----
using bf16 = __nv_bfloat16;

enum NativeDType : int {
  kNativeFloat32 = 0,
  kNativeFloat16 = 1,
  kNativeBFloat16 = 2,
};

int native_dtype_code(at::ScalarType dtype, const char* name) {
  if (dtype == at::kFloat) return kNativeFloat32;
  if (dtype == at::kHalf) return kNativeFloat16;
  if (dtype == at::kBFloat16) return kNativeBFloat16;
  TORCH_CHECK(false, name, " must be float32, float16, or bfloat16, got ",
              dtype);
  return kNativeFloat32;
}

constexpr int kK = 128;
constexpr int kV = 128;
constexpr int kBT = 64;
constexpr int kBV = 64;
constexpr int kThreads = 256;
constexpr int kStateElems = kBV * kK;
constexpr int kResidualElems = kBT * kBV;
constexpr size_t kChunkPlanCacheCapacity = 4;
// Padded leading dimension for the transposed residual panel. kBV + 4 = 68
// gives a 34-word row step and therefore distinct banks across a 16-lane
// group; kBV + 8 = 72 (= 36 words, a multiple of 4) only reached eight.
constexpr int kResStride = kBV + 4;
// Padded row stride for the BF16 snapshot and for the K tile that reuses the
// same buffer. The padding is chosen for shared-memory banks, not for
// alignment: with a half-word step S the lane groups are separated by
// 4*S halves = 2*S words, and the eight-byte operands advance by S/2 words per
// lane. S = 132 gives 2*S = 264 = 8 (mod 32), so the four lane groups tile all
// 32 banks instead of collapsing onto two, and S/2 = 66 = 2 (mod 4) makes
// 66*l mod 32 distinct for l < 16, the floor for a 64-lane eight-byte access.
// 264 bytes is a multiple of 8 but not of 16, so 8-byte accesses are used here.
constexpr int kKStride = kK + 4;
// The projection A operand is staged as a full 64x128 tile whose row step is
// kKStride - the same 132 the snapshot and the K tile use. Read as four
// contiguous halves per lane at w_panel[row*kKStride + ki*16 + chunk*4], the row
// base advances by 66 words, and 66*row mod 32 = 2*row mod 32 takes sixteen
// values over the sixteen rows a warp touches: the read sits at the two-way
// floor with no padding at all. Staging the whole tile also removes the
// per-k-block async staging and its eight waits; the fill is the same plain
// sixteen-byte copy the K tile uses, and it coalesces because sixteen
// consecutive threads cover one row's 256 bytes. What it gives up is the
// overlap: this copy is synchronous, so its global latency is no longer hidden
// behind the projection the way the cp.async stages were.


// ---- allocating-entry metadata cache ----
// Mirrors the four-entry tensor-identity cache used by the Triton entry's
// prepare_chunk_indices/prepare_chunk_offsets/is_tail_free helpers.  Holding a
// strong reference to `source` makes TensorImpl identity safe against address
// reuse; the version additionally invalidates the entry after an in-place
// mutation, which the Python identity-only cache does not detect.
int64_t tensor_version_or_disabled(torch::Tensor const& tensor) {
  auto const& version = tensor.unsafeGetTensorImpl()->version_counter();
  return version.enabled() ? static_cast<int64_t>(version.current_version())
                           : -1;
}

struct ChunkPlan {
  torch::Tensor source;
  int64_t source_version = 0;
  int64_t total_tokens = 0;
  torch::Tensor cu_i32;
  torch::Tensor offsets_i32;
  int64_t sequences = 0;
  int64_t total_chunks = 0;
  int64_t chunks_per_sequence = 0;
  bool ragged = false;
};

class ChunkPlanCache {
 public:
  std::optional<ChunkPlan> lookup(torch::Tensor const& source,
                                  int64_t total_tokens) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t version = tensor_version_or_disabled(source);
    for (size_t i = 0; i < entries_.size(); ++i) {
      ChunkPlan const& entry = entries_[i];
      if (entry.source.is_same(source) &&
          entry.source_version == version &&
          entry.total_tokens == total_tokens) {
        ChunkPlan hit = entry;
        entries_.erase(entries_.begin() + i);
        entries_.push_back(hit);
        return hit;
      }
    }
    return std::nullopt;
  }

  void insert(ChunkPlan plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < entries_.size(); ++i) {
      if (entries_[i].source.is_same(plan.source) &&
          entries_[i].source_version == plan.source_version &&
          entries_[i].total_tokens == plan.total_tokens) {
        entries_.erase(entries_.begin() + i);
        break;
      }
    }
    if (entries_.size() >= kChunkPlanCacheCapacity) {
      entries_.erase(entries_.begin());
    }
    entries_.push_back(std::move(plan));
  }

 private:
  std::mutex mutex_;
  std::vector<ChunkPlan> entries_;
};

ChunkPlanCache& chunk_plan_cache() {
  static ChunkPlanCache cache;
  return cache;
}

ChunkPlan build_chunk_plan(torch::Tensor const& cu, int64_t total_tokens) {
  ChunkPlan plan;
  plan.source = cu;
  plan.source_version = tensor_version_or_disabled(cu);
  plan.total_tokens = total_tokens;
  plan.cu_i32 = cu.to(at::kInt).contiguous();

  auto cu_cpu = cu.to(at::kLong).cpu().contiguous();
  auto const* ptr = cu_cpu.data_ptr<int64_t>();
  std::vector<int64_t> cu_values(ptr, ptr + cu_cpu.numel());
  TORCH_CHECK(cu_values.size() >= 2,
              "cu_seqlens must hold at least one sequence");
  TORCH_CHECK(cu_values.front() == 0,
              "cu_seqlens must start at zero");

  plan.sequences = static_cast<int64_t>(cu_values.size()) - 1;
  std::vector<int64_t> offsets;
  offsets.reserve(cu_values.size());
  offsets.push_back(0);
  int64_t first_length = -1;
  for (int64_t i = 0; i < plan.sequences; ++i) {
    int64_t const length = cu_values[i + 1] - cu_values[i];
    TORCH_CHECK(length >= 0, "cu_seqlens must be non-decreasing");
    if (i == 0) first_length = length;
    plan.ragged =
        plan.ragged || length != first_length || length % kBT != 0;
    offsets.push_back(offsets.back() + (length + kBT - 1) / kBT);
  }
  TORCH_CHECK(cu_values.back() == total_tokens,
              "k holds ", total_tokens,
              " tokens but cu_seqlens end at ", cu_values.back());

  plan.total_chunks = offsets.back();
  plan.chunks_per_sequence =
      first_length < 0 ? 0 : (first_length + kBT - 1) / kBT;
  plan.offsets_i32 = torch::tensor(
      offsets, torch::TensorOptions().device(cu.device()).dtype(torch::kInt32));
  return plan;
}


// ---- index helpers ----
__host__ __device__ inline size_t h_index(int chunk, int head, int v, int k,
                                          int heads) {
  return ((static_cast<size_t>(chunk) * heads + head) * kV + v) * kK + k;
}


// ---- global access helpers ----
template <bool Streaming>
__device__ __forceinline__ bf16 load_u(const bf16* ptr) {
  if constexpr (Streaming) {
    return __ldcs(ptr);
  } else {
    return *ptr;
  }
}

template <bool Streaming>
__device__ __forceinline__ void store_output(bf16* ptr, bf16 value) {
  if constexpr (Streaming) {
    __stcs(ptr, value);
  } else {
    *ptr = value;
  }
}


// ---- fragment types ----
using NativeB64 = __NATIVE_VECTOR__(2, uint32_t);
using NativeF32x4 = __NATIVE_VECTOR__(4, float);

template <bool Streaming>
__device__ __forceinline__ void store_output_vec(bf16* ptr, uint4 value) {
  if constexpr (Streaming) {
    __stcs(reinterpret_cast<uint4*>(ptr), value);
  } else {
    *reinterpret_cast<uint4*>(ptr) = value;
  }
}

__device__ __forceinline__ float load_native_scalar(
    const void* ptr, int64_t index, int dtype) {
  if (dtype == kNativeFloat32) {
    return reinterpret_cast<const float*>(ptr)[index];
  }
  if (dtype == kNativeFloat16) {
    return __half2float(reinterpret_cast<const __half*>(ptr)[index]);
  }
  return __bfloat162float(reinterpret_cast<const bf16*>(ptr)[index]);
}

__device__ __forceinline__ void store_native_scalar(
    void* ptr, int64_t index, int dtype, float value) {
  if (dtype == kNativeFloat32) {
    reinterpret_cast<float*>(ptr)[index] = value;
  } else if (dtype == kNativeFloat16) {
    reinterpret_cast<__half*>(ptr)[index] = __float2half_rn(value);
  } else {
    reinterpret_cast<bf16*>(ptr)[index] = __float2bfloat16(value);
  }
}

__device__ __forceinline__ float round_native_scalar(float value, int dtype) {
  if (dtype == kNativeFloat32) return value;
  if (dtype == kNativeFloat16) {
    return __half2float(__float2half_rn(value));
  }
  return __bfloat162float(__float2bfloat16(value));
}

// Correctness-oriented fallback used outside the optimized BF16 K=V=128
// contract. One block owns one (sequence, head, v) state row. Threads span K,
// while the BT token dimension is processed cooperatively. This keeps the
// recurrence exact without imposing a compile-time input/state dtype matrix.
template <typename IndexT>
__global__ void chunk_delta_generic_kernel(
    const void* __restrict__ k, const void* __restrict__ w,
    const void* __restrict__ u, const void* __restrict__ g,
    const void* __restrict__ gk, void* __restrict__ initial_state,
    void* __restrict__ h, void* __restrict__ v_new, int sequences, int heads,
    int k_heads, int K, int V, int tokens_per_sequence,
    int chunks_per_sequence, const IndexT* __restrict__ state_indices,
    int64_t state_stride, const int* __restrict__ cu_seqlens,
    const int* __restrict__ chunk_offsets, bool ragged, bool has_g,
    bool has_gk, bool save_new_value, bool use_exp2, int k_dtype,
    int w_dtype, int u_dtype, int g_dtype, int gk_dtype, int state_dtype) {
  __shared__ float state_row[256];
  __shared__ float snapshot[256];
  __shared__ float reduction[256];
  __shared__ float residual[kBT];

  const int kk = static_cast<int>(threadIdx.x);
  const int v = static_cast<int>(blockIdx.x);
  const int sequence_head = static_cast<int>(blockIdx.y);
  const int sequence = sequence_head / heads;
  const int head = sequence_head - sequence * heads;
  if (sequence >= sequences || v >= V) return;

  const int64_t slot = static_cast<int64_t>(state_indices[sequence]);
  const bool state_valid = slot >= 0;
  const int64_t state_base =
      (state_valid ? slot : 0) * state_stride +
      (static_cast<int64_t>(head) * V + v) * K;
  if (kk < K) {
    state_row[kk] = state_valid
                        ? load_native_scalar(initial_state, state_base + kk,
                                             state_dtype)
                        : 0.0f;
  }
  __syncthreads();

  const int bos = ragged ? cu_seqlens[sequence]
                         : sequence * tokens_per_sequence;
  const int eos = ragged ? cu_seqlens[sequence + 1]
                         : bos + tokens_per_sequence;
  const int n_chunks = ragged ? (eos - bos + kBT - 1) / kBT
                              : chunks_per_sequence;
  const int boh = ragged ? chunk_offsets[sequence]
                         : sequence * chunks_per_sequence;
  const int k_head = head / (heads / k_heads);

  for (int chunk_local = 0; chunk_local < n_chunks; ++chunk_local) {
    const int token0 = bos + chunk_local * kBT;
    const int rows = std::min(kBT, eos - token0);
    const int chunk = boh + chunk_local;

    if (kk < K) {
      snapshot[kk] = round_native_scalar(state_row[kk], w_dtype);
      const int64_t h_index =
          (((static_cast<int64_t>(chunk) * heads + head) * V + v) * K + kk);
      store_native_scalar(h, h_index, k_dtype, state_row[kk]);
    }
    __syncthreads();

    for (int row = 0; row < rows; ++row) {
      const int token = token0 + row;
      float product = 0.0f;
      if (kk < K) {
        const int64_t w_index =
            (static_cast<int64_t>(token) * heads + head) * K + kk;
        product = load_native_scalar(w, w_index, w_dtype) * snapshot[kk];
      }
      reduction[kk] = product;
      __syncthreads();
      for (int stride = 128; stride > 0; stride >>= 1) {
        if (kk < stride) reduction[kk] += reduction[kk + stride];
        __syncthreads();
      }
      if (kk == 0) {
        const int64_t u_index =
            (static_cast<int64_t>(token) * heads + head) * V + v;
        const float raw_residual =
            load_native_scalar(u, u_index, u_dtype) - reduction[0];
        if (save_new_value) {
          store_native_scalar(v_new, u_index, u_dtype, raw_residual);
        }
        float update_residual = raw_residual;
        if (has_g) {
          const int last_token = token0 + rows - 1;
          const int64_t g_last_index =
              static_cast<int64_t>(last_token) * heads + head;
          const int64_t g_index =
              static_cast<int64_t>(token) * heads + head;
          const float delta = load_native_scalar(g, g_last_index, g_dtype) -
                              load_native_scalar(g, g_index, g_dtype);
          update_residual *= delta <= 0.0f ? __expf(delta) : 0.0f;
        }
        residual[row] = round_native_scalar(update_residual, k_dtype);
      }
      __syncthreads();
    }

    if (kk < K) {
      const int last_token = token0 + rows - 1;
      float decay = 1.0f;
      if (has_g) {
        const int64_t g_index =
            static_cast<int64_t>(last_token) * heads + head;
        decay *= __expf(load_native_scalar(g, g_index, g_dtype));
      }
      if (has_gk) {
        const int64_t gk_index =
            (static_cast<int64_t>(last_token) * heads + head) * K + kk;
        const float gate = load_native_scalar(gk, gk_index, gk_dtype);
        decay *= use_exp2 ? exp2f(gate) : __expf(gate);
      }
      float update = 0.0f;
      for (int row = 0; row < rows; ++row) {
        const int token = token0 + row;
        const int64_t k_index =
            (static_cast<int64_t>(token) * k_heads + k_head) * K + kk;
        update = fmaf(load_native_scalar(k, k_index, k_dtype), residual[row],
                      update);
      }
      state_row[kk] = fmaf(state_row[kk], decay, update);
    }
    __syncthreads();
  }

  if (kk < K && state_valid) {
    store_native_scalar(initial_state, state_base + kk, state_dtype,
                        state_row[kk]);
  }
}


// ---- fragment loads ----
__device__ __forceinline__ NativeB64 load_contiguous_bf16x4(
    const bf16* ptr) {
  return *reinterpret_cast<const NativeB64*>(ptr);
}

__device__ __forceinline__ NativeB64 load_a_col_bf16x4(
    const bf16* ptr, int ldm, int lane) {
  const int row = lane & 15;
  const int col = (lane >> 4) * 4;
  NativeB64 frag;
  uint16_t* raw = reinterpret_cast<uint16_t*>(&frag);
#pragma unroll
  for (int i = 0; i < 4; ++i)
    raw[i] = reinterpret_cast<const uint16_t*>(ptr)[(col + i) * ldm + row];
  return frag;
}


// ---- native kernel ----
// The recurrent state may be fp32 or bf16. The kernel always accumulates in
// fp32: a bf16 state is widened on load and rounded once on the final store,
// which is what both the Triton path and the unit test's reference do.
template <bool StateBf16>
__device__ __forceinline__ NativeF32x4 load_state_vec(const float* st, int off) {
  if constexpr (StateBf16) {
    const NativeB64 raw = *reinterpret_cast<const NativeB64*>(
        reinterpret_cast<const bf16*>(st) + off);
    const uint16_t* raw_h = reinterpret_cast<const uint16_t*>(&raw);
    const bf16* h4 = reinterpret_cast<const bf16*>(raw_h);
    return NativeF32x4{__bfloat162float(h4[0]), __bfloat162float(h4[1]),
                       __bfloat162float(h4[2]), __bfloat162float(h4[3])};
  } else {
    return *reinterpret_cast<const NativeF32x4*>(st + off);
  }
}

template <bool StateBf16>
__device__ __forceinline__ void store_state_vec(float* st, int off,
                                                NativeF32x4 value) {
  if constexpr (StateBf16) {
    alignas(8) bf16 pack[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) pack[r] = __float2bfloat16(value[r]);
    *reinterpret_cast<NativeB64*>(reinterpret_cast<bf16*>(st) + off) =
        *reinterpret_cast<const NativeB64*>(pack);
  } else {
    *reinterpret_cast<NativeF32x4*>(st + off) = value;
  }
}

template <bool FullChunk, bool Streaming, bool Exp2, bool UseG,
          bool SaveNewValue, int Bv>
__device__ __forceinline__ void process_chunk_native(
    const bf16* __restrict__ k, const bf16* __restrict__ w,
    const bf16* __restrict__ u, const float* __restrict__ g,
    const float* __restrict__ gk, bool has_gk,
    bf16* __restrict__ h, bf16* __restrict__ v_new, int heads, int head,
    int k_heads_eff, int k_head, int token0, int chunk, int v_base,
    int row_max, uint32_t tok_kk, int wave_m, int wave_n, int lane,
    bf16* state_bf16, bf16* residual_panel, bf16* w_panel,
    float* decay_panel, NativeF32x4 (&state_reg)[2][2][Bv / 32]) {
  constexpr int kNV = Bv / 32;
  constexpr int kFullLastLocal = kBT - 1;
      // 64-bit addressing is confined to these six per-chunk base pointers.
      // Everything inside the unrolled loops below uses 32-bit element offsets:
      // the ISA showed ~230 of the chunk body's 1193 instructions were pure
      // 64-bit address arithmetic (addc_co_u32/shl_b64/smov_b64) that Triton
      // does not emit.
      const uint32_t tok_k = static_cast<uint32_t>(heads) * kK;
      const uint32_t tok_v = static_cast<uint32_t>(heads) * kV;
      const bf16* __restrict__ u_c =
          u + ((static_cast<size_t>(token0) * heads + head) * kV) + v_base;
      bf16* __restrict__ vn_c = nullptr;
      if constexpr (SaveNewValue) {
        vn_c = v_new +
               ((static_cast<size_t>(token0) * heads + head) * kV) + v_base;
      }
      const bf16* __restrict__ w_c =
          w + (static_cast<size_t>(token0) * heads + head) * kK;
      const bf16* __restrict__ k_c =
          k + (static_cast<size_t>(token0) * k_heads_eff + k_head) * kK;
      const float* __restrict__ g_c = nullptr;
      if constexpr (UseG) {
        g_c = g + static_cast<size_t>(token0) * heads + head;
      }
      const float* __restrict__ gk_c = nullptr;
      if constexpr (!UseG) {
        gk_c = gk +
               (static_cast<size_t>(token0) * heads + head) * kK;
      } else if (has_gk) {
        gk_c = gk +
               (static_cast<size_t>(token0) * heads + head) * kK;
      }

    // Round the register state to BF16 once; the same 4-half word feeds the
    // projection B operand and the h output.  Each thread owns (p, mi, ni),
    // i.e. eight 8-byte words, and no per-element division is needed.
#pragma unroll
    for (int p = 0; p < 2; ++p)
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < kNV; ++ni) {
          const int v_local = wave_n * (Bv / 2) + ni * 16 + (lane & 15);
          const int kk0 = p * 64 + wave_m * 32 + mi * 16 + (lane >> 4) * 4;
          alignas(8) bf16 pack[4];
#pragma unroll
          for (int r = 0; r < 4; ++r)
            pack[r] = __float2bfloat16(state_reg[p][mi][ni][r]);
          const uint2 word = *reinterpret_cast<const uint2*>(pack);
          *reinterpret_cast<uint2*>(state_bf16 + v_local * kKStride + kk0) =
              word;
        }
    // W tile for the projection A operand: all 64 rows and all 128 k at once.
    // Issued before the snapshot barrier so its global loads are in flight while
    // the snapshot stores retire; the barrier is what makes the panel visible to
    // the projection below.
#pragma unroll
    for (int copy = 0; copy < kBT * kK / (kThreads * 8); ++copy) {
      const int vec = threadIdx.x + copy * kThreads;
      const int t = vec >> 4;
      const int kblk = vec & 15;
      const int rowc = FullChunk ? t : (t < row_max ? t : row_max);
      const uint4 word =
          *reinterpret_cast<const uint4*>(w_c + rowc * tok_k + kblk * 8);
      // kKStride is 8-byte but not 16-byte aligned: store the 16 bytes as two
      // 8-byte halves, exactly as the K tile fill below does.
      uint2 halves[2];
      std::memcpy(halves, &word, sizeof(word));
      *reinterpret_cast<uint2*>(w_panel + t * kKStride + kblk * 8) = halves[0];
      *reinterpret_cast<uint2*>(w_panel + t * kKStride + kblk * 8 + 4) =
          halves[1];
    }
    __syncthreads();

    // h holds exactly the same BF16 snapshot. Copy it out of the padded shared
    // panel with 16-byte accesses so that consecutive lanes cover one
    // contiguous h row: issuing the h store straight from the register state
    // would scatter 16 cache lines per warp instruction and measured 15.6%
    // slower on shape 1, independent of the total bytes written.
#pragma unroll
    for (int copy = 0; copy < Bv * 16 / kThreads; ++copy) {
      const int vec = threadIdx.x + copy * kThreads;
      const int v_local = vec >> 4;
      const int kk = (vec & 15) * 8;
      // kKStride is 264 bytes, so this row is only 8-byte aligned: read the two
      // eight-byte halves and issue one sixteen-byte store to h, which is 16-byte
      // aligned whenever kk is a multiple of 8 halves.
      const uint2 lo = *reinterpret_cast<const uint2*>(
          state_bf16 + v_local * kKStride + kk);
      const uint2 hi = *reinterpret_cast<const uint2*>(
          state_bf16 + v_local * kKStride + kk + 4);
      uint4 word;
      word.x = lo.x;
      word.y = lo.y;
      word.z = hi.x;
      word.w = hi.y;
      store_output_vec<Streaming>(
          h + h_index(chunk, head, v_base + v_local, kk, heads), word);
    }

    NativeF32x4 projection[2][kNV];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < kNV; ++ni)
        projection[mi][ni] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
    for (int ki = 0; ki < 8; ++ki) {
      NativeB64 af[2];
      NativeB64 bf[2];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int row = wave_m * 32 + mi * 16 + (lane & 15);
        af[mi] = load_contiguous_bf16x4(
            w_panel + row * kKStride + ki * 16 + (lane >> 4) * 4);
      }
#pragma unroll
      for (int ni = 0; ni < kNV; ++ni) {
        const int row = (lane >> 4) * 4;
        const int col = wave_n * (Bv / 2) + ni * 16 + (lane & 15);
        bf[ni] = load_contiguous_bf16x4(
            state_bf16 + col * kKStride + ki * 16 + row);
      }
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < kNV; ++ni)
          projection[mi][ni] = __builtin_mxc_mma_16x16x16bf16(
              af[mi], bf[ni], projection[mi][ni]);
    }

    // Every wave must stop reading the snapshot before the async K copy
    // overwrites the same panel. barrier_inst is wave-scoped.
    __syncthreads();

    // Snapshot is dead after projection. Refill the same panel with the K tile.
    // The row stride is kKStride, not kK: the update A operand reads four kk at
    // a stride of ldm, and ldm = 128 folds all four wave_m groups onto the same
    // eight banks (measured 6.7% penalty); ldm = kKStride removes it. cp.async
    // cannot be used here because it silently corrupts non-dense BF16
    // destinations on this backend, so this is a plain 16-byte copy.
#pragma unroll
    for (int copy = 0; copy < kBT * 16 / kThreads; ++copy) {
      const int vec = threadIdx.x + copy * kThreads;
      const int t = vec >> 4;
      const int kblk = vec & 15;
      const uint4 word =
          *reinterpret_cast<const uint4*>(
              k_c + (FullChunk ? t : (t < row_max ? t : row_max)) * tok_kk +
                  kblk * 8);
      // kKStride is 8-byte but not 16-byte aligned: store the 16 bytes as two
      // 8-byte halves.
      uint2 halves[2];
      std::memcpy(halves, &word, sizeof(word));
      *reinterpret_cast<uint2*>(state_bf16 + t * kKStride + kblk * 8) =
          halves[0];
      *reinterpret_cast<uint2*>(state_bf16 + t * kKStride + kblk * 8 + 4) =
          halves[1];
    }

    // Scalar-g applies a per-token relative decay to the residual used by the
    // recurrent update. v_new itself remains the ungated residual. Reuse the
    // first BT entries of decay_panel here; they are overwritten with the
    // per-K state decay after every thread has consumed these values.
    if constexpr (UseG) {
      if (threadIdx.x < kBT) {
        const int row = static_cast<int>(threadIdx.x);
        const int rowc = FullChunk ? row : (row < row_max ? row : row_max);
        const int last_local = FullChunk ? kFullLastLocal : row_max;
        const float delta =
            g_c[last_local * heads] - g_c[rowc * heads];
        decay_panel[row] = delta <= 0.0f ? __expf(delta) : 0.0f;
      }
      __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
      for (int ni = 0; ni < kNV; ++ni) {
        const int col = wave_n * (Bv / 2) + ni * 16 + (lane & 15);
        const int row0 = wave_m * 32 + mi * 16 + (lane >> 4) * 4;
        alignas(8) bf16 pack[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const int row = row0 + r;
          const uint32_t out = row * tok_v + col;
          const int rowc = FullChunk ? row : (row < row_max ? row : row_max);
          const float residual =
              __bfloat162float(load_u<Streaming>(u_c + rowc * tok_v + col)) -
              projection[mi][ni][r];
          const bf16 rounded = __float2bfloat16(residual);
          if constexpr (!FullChunk) {
            if (row > row_max) {
              // Outside the sequence the production kernel's masked load makes
              // the residual exactly zero, and its masked store writes nothing.
              pack[r] = __float2bfloat16(0.0f);
              continue;
            }
          }
          if constexpr (SaveNewValue) {
            store_output<Streaming>(vn_c + out, rounded);
          }
          float update_residual = residual;
          if constexpr (UseG) {
            update_residual *= decay_panel[row];
          }
          pack[r] = __float2bfloat16(update_residual);
        }
        *reinterpret_cast<NativeB64*>(residual_panel + col * kResStride + row0) =
            *reinterpret_cast<const NativeB64*>(pack);
      }
    }
    if constexpr (UseG) {
      // Do not overwrite scalar relative-decay values while another wave is
      // still consuming them in the residual epilogue above.
      __syncthreads();
    }
    // All 256 threads need the same 128 `kk` decays. Evaluating them once per
    // `kk` instead of once per (thread, element) removes a 32x redundancy: the
    // ISA showed 16 expf expansions plus four 16-byte gk loads per thread per
    // chunk to produce 128 distinct values.
    if (threadIdx.x < kK) {
      const int last_local = FullChunk ? kFullLastLocal : row_max;
      float decay = 1.0f;
      if constexpr (UseG) {
        decay = __expf(g_c[last_local * heads]);
        if (has_gk) {
          decay *= __expf(gk_c[last_local * tok_k + threadIdx.x]);
        }
      } else {
        const float raw = gk_c[last_local * tok_k + threadIdx.x];
        decay = Exp2 ? exp2f(raw) : __expf(raw);
      }
      decay_panel[threadIdx.x] = decay;
    }
    __syncthreads();

#pragma unroll
    for (int k_phase = 0; k_phase < kK; k_phase += 64) {
      NativeF32x4 update[2][kNV];
#pragma unroll
      for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < kNV; ++ni)
          update[mi][ni] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
      for (int t0 = 0; t0 < kBT; t0 += 16) {
        NativeB64 af[2];
        NativeB64 bf[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const int kk = k_phase + wave_m * 32 + mi * 16;
          af[mi] = load_a_col_bf16x4(
              state_bf16 + t0 * kKStride + kk, kKStride, lane);
        }
#pragma unroll
        for (int ni = 0; ni < kNV; ++ni) {
          const int col = wave_n * (Bv / 2) + ni * 16;
          bf[ni] = *reinterpret_cast<const NativeB64*>(
              residual_panel + (col + (lane & 15)) * kResStride + t0 +
              (lane >> 4) * 4);
        }
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int ni = 0; ni < kNV; ++ni)
            update[mi][ni] = __builtin_mxc_mma_16x16x16bf16(
                af[mi], bf[ni], update[mi][ni]);
      }

      // The lane owns four consecutive kk, so the decay load is a single
      // 16-byte access and the decay is computed once per (phase, mi).  The
      // update MMA already produced the accumulator in the state_reg layout,
      // so the whole read-modify-write is eight FMAs per thread.
      const int p = k_phase >> 6;
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
        const int kk0 = k_phase + wave_m * 32 + mi * 16 + (lane >> 4) * 4;
        const NativeF32x4 decay =
            *reinterpret_cast<const NativeF32x4*>(decay_panel + kk0);
#pragma unroll
        for (int ni = 0; ni < kNV; ++ni) {
#pragma unroll
          for (int r = 0; r < 4; ++r)
            state_reg[p][mi][ni][r] =
                fmaf(state_reg[p][mi][ni][r], decay[r], update[mi][ni][r]);
        }
      }
      // Still required: the next chunk overwrites state_bf16, which this
      // iteration is reading as the update A operand.
      __syncthreads();
    }
}

template <bool Streaming, bool Pool = false, bool Gqa = false, bool Exp2 = false,
          bool UseG = false, bool Ragged = false, bool StateBf16 = false,
          bool SaveNewValue = true, int Bv = 64, typename IndexT = int>
__global__ __launch_bounds__(kThreads) void chunk_delta_native_kernel(
    const bf16* __restrict__ k, const bf16* __restrict__ w,
    const bf16* __restrict__ u, const float* __restrict__ g,
    const float* __restrict__ gk, bool has_gk,
    float* __restrict__ initial_state, bf16* __restrict__ h,
    bf16* __restrict__ v_new, int sequences, int heads,
    int chunks_per_sequence,
    // --- contract extensions; sentinel values reproduce the dense layout ---
    const IndexT* __restrict__ state_indices,  // null => slot == sequence
    long long state_stride,                 // 0   => heads*kV*kK
    int k_heads,                            // 0   => heads (no GQA)
    int use_exp2,                           // Triton USE_EXP2
    const int* __restrict__ cu_seqlens,     // Ragged only: N+1 sequence offsets
    const int* __restrict__ chunk_offsets) { // Ragged only: N global chunk offsets
  // No FP32 state panel: the 64x128 state is register-resident.
  // Per-warp v coverage is Bv/2 (two wave_n values) and each MMA n-tile is 16.
  constexpr int kNV = Bv / 32;
  // This buffer holds the BF16 snapshot (Bv rows, padded stride) and is then
  // reused for the K tile, which is kBT rows x kK halves regardless of Bv.
  // Sizing it by Bv alone overflows the K tile when Bv < kBT.
  constexpr int kStateBf16Elems = Bv * kKStride > kBT * kKStride
                                      ? Bv * kKStride
                                      : kBT * kKStride;
  __shared__ bf16 state_bf16[kStateBf16Elems];             // 16 KiB
  __shared__ bf16 residual_panel[Bv * kResStride];      //  8 KiB
  __shared__ bf16 w_panel[kBT * kKStride];             // 16.5 KiB
  __shared__ float decay_panel[kK];                    // 512 B

  const int v_tile = static_cast<int>(blockIdx.x);
  const int sequence_head = static_cast<int>(blockIdx.y);
  const int sequence = sequence_head / heads;
  const int head = sequence_head - sequence * heads;
  if (sequence >= sequences) return;

  // State-slot addressing. Mirrors the Triton contract: the slot id comes from
  // initial_state_indices, a negative id marks a padded row that is neither
  // read nor written, and stride_init_state is the caller-supplied per-slot
  // pitch (an envelope-strided pool is not H*V*K).
  // In the dense contract the sequence index *is* the slot id and the pitch is
  // heads*kV*kK, so Pool=false folds to the original address expression.
  const int64_t slot =
      Pool ? (state_indices != nullptr
                  ? static_cast<int64_t>(state_indices[sequence])
                  : static_cast<int64_t>(sequence))
           : static_cast<int64_t>(sequence);
  const bool state_valid = !Pool || slot >= 0;
  const long long slot_pitch =
      (Pool && state_stride > 0) ? state_stride
                                 : static_cast<long long>(heads) * kV * kK;
  const long long slot_base =
      (state_valid ? slot : 0) * slot_pitch +
      static_cast<long long>(head) * kV * kK;
  // `state_stride` and `slot_base` are expressed in tensor elements.  The
  // launcher transports a bf16 state through a float* only to share one ABI;
  // pointer arithmetic must therefore happen after restoring the real element
  // type, otherwise every nonzero slot/head offset is doubled.
  const float* __restrict__ st_in;
  float* __restrict__ st_out;
  if constexpr (StateBf16) {
    const bf16* const base_in = reinterpret_cast<const bf16*>(initial_state);
    bf16* const base_out = reinterpret_cast<bf16*>(initial_state);
    st_in = reinterpret_cast<const float*>(base_in + slot_base);
    st_out = reinterpret_cast<float*>(base_out + slot_base);
  } else {
    st_in = initial_state + slot_base;
    st_out = initial_state + slot_base;
  }

  // GQA: k has k_heads heads while w/u/gk/h keep `heads`.
  const int k_heads_eff = Gqa ? (k_heads > 0 ? k_heads : heads) : heads;
  const int k_head = Gqa ? head / (heads / k_heads_eff) : head;
  const uint32_t tok_kk = static_cast<uint32_t>(k_heads_eff) * kK;

  const int v_base = v_tile * Bv;
  // Variable length. With Ragged false and cu_seqlens null these fold to the
  // original sequence*C*BT / sequence*C expressions.
  const int bos = Ragged ? cu_seqlens[sequence]
                         : sequence * chunks_per_sequence * kBT;
  const int eos = Ragged ? cu_seqlens[sequence + 1]
                         : (sequence + 1) * chunks_per_sequence * kBT;
  const int n_chunks = Ragged ? (eos - bos + kBT - 1) / kBT
                              : chunks_per_sequence;
  const int boh = Ragged ? chunk_offsets[sequence]
                         : sequence * chunks_per_sequence;

  const int wave = threadIdx.x >> 6;
  const int lane = threadIdx.x & 63;
  const int wave_m = wave >> 1;
  const int wave_n = wave & 1;

  // The whole 64x128 FP32 state lives in registers.  Each thread owns eight
  // 4-float vectors whose (kk, v) map is exactly the epilogue map of the
  // update MMA, so the state read-modify-write needs no shared memory at all.
  NativeF32x4 state_reg[2][2][kNV];
#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < kNV; ++ni) {
        const int v_local = wave_n * (Bv / 2) + ni * 16 + (lane & 15);
        const int kk0 = p * 64 + wave_m * 32 + mi * 16 + (lane >> 4) * 4;
        state_reg[p][mi][ni] =
            state_valid
                ? load_state_vec<StateBf16>(st_in,
                                            (v_base + v_local) * kK + kk0)
                : NativeF32x4{0.0f, 0.0f, 0.0f, 0.0f};
      }

  for (int chunk_local = 0; chunk_local < n_chunks; ++chunk_local) {
    const int token0 = bos + chunk_local * kBT;
    const int chunk = boh + chunk_local;
    // Highest row of this chunk that is inside the sequence, and the row whose
    // gate defines the decay: both are kBT-1 for a whole chunk, which is what
    // the dense instantiation folds to. Note the min with kBT-1 - without it a
    // whole chunk would take its decay from eos-1-token0, i.e. a row of a much
    // later chunk.
    const int row_max = Ragged ? std::min(kBT - 1, eos - 1 - token0) : kBT - 1;
    // Specialize the body twice. A ragged sequence still consists almost
    // entirely of whole chunks; making `FullChunk` a compile-time value keeps
    // their hot path free of the per-fragment xmask branches and 64-bit address
    // selection needed only by the final partial chunk.
    if constexpr (Ragged) {
      if (row_max == kBT - 1) {
        process_chunk_native<true, Streaming, Exp2, UseG, SaveNewValue, Bv>(
            k, w, u, g, gk, has_gk, h, v_new, heads, head, k_heads_eff,
            k_head, token0, chunk, v_base, row_max, tok_kk, wave_m, wave_n,
            lane, state_bf16, residual_panel, w_panel, decay_panel, state_reg);
      } else {
        process_chunk_native<false, Streaming, Exp2, UseG, SaveNewValue, Bv>(
            k, w, u, g, gk, has_gk, h, v_new, heads, head, k_heads_eff,
            k_head, token0, chunk, v_base, row_max, tok_kk, wave_m, wave_n,
            lane, state_bf16, residual_panel, w_panel, decay_panel, state_reg);
      }
    } else {
      process_chunk_native<true, Streaming, Exp2, UseG, SaveNewValue, Bv>(
          k, w, u, g, gk, has_gk, h, v_new, heads, head, k_heads_eff, k_head,
          token0, chunk, v_base, row_max, tok_kk, wave_m, wave_n, lane,
          state_bf16, residual_panel, w_panel, decay_panel, state_reg);
    }
  }

#pragma unroll
  for (int p = 0; p < 2; ++p)
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
      for (int ni = 0; ni < kNV; ++ni) {
        const int v_local = wave_n * (Bv / 2) + ni * 16 + (lane & 15);
        const int kk0 = p * 64 + wave_m * 32 + mi * 16 + (lane >> 4) * 4;
        if (state_valid) {
          store_state_vec<StateBf16>(st_out, (v_base + v_local) * kK + kk0,
                                     state_reg[p][mi][ni]);
        }
      }
}


// ---- production dispatch mirror (coverage table only) ----
enum class Dispatch {
  kUnsupported,   // host function raises before any launch
  kSingleChunk,   // _chunk_gated_delta_rule_fwd_kernel_h_k128_single_chunk
  kFusedLong,     // _chunk_gated_delta_rule_fwd_kernel_h_k128_fused   (BV=32)
  kTailfree,      // _chunk_gated_delta_rule_fwd_kernel_h_k128_tailfree (BV=64)
  kGeneral,       // chunk_gated_delta_rule_fwd_kernel_h_blockdim64
};

struct Contract {
  int B;              // k.shape[0]
  int T;              // k.shape[1]
  int Hg;             // k.shape[2]
  int K;              // k.shape[3]
  int H;              // u.shape[-2]
  int V;              // u.shape[-1]
  int block_v;        // caller-resolved block_v (default: 64 if K==V==128 else 32)
  int num_warps;
  int num_stages;
  int N;              // logical batch: cu_seqlens ? len(cu_seqlens)-1 : B
  int NT;             // total chunks
  int has_g;          // g is not None
  int has_gk;         // gk is not None
  int has_v_new;      // save_new_value / v_new is not None
  int has_cu_seqlens;
  int tail_free;      // every sequence length is a whole number of BT
};

inline const char* dispatch_name(Dispatch d) {
  switch (d) {
    case Dispatch::kUnsupported: return "unsupported";
    case Dispatch::kSingleChunk: return "single_chunk";
    case Dispatch::kFusedLong:   return "fused_long";
    case Dispatch::kTailfree:    return "tailfree";
    default:                     return "general";
  }
}

inline Dispatch cuda_dispatch(const Contract& c) {
  if (c.K > 256) return Dispatch::kUnsupported;
  if (c.Hg <= 0 || c.H % c.Hg != 0) return Dispatch::kUnsupported;
  const bool fused = c.K == 128 && c.V == 128 && c.N == 1 && c.H == 8 &&
                     c.Hg == 8 && c.block_v == 64 && c.num_warps == 4 &&
                     c.num_stages == 1 && !c.has_g && c.has_gk &&
                     c.has_v_new && c.has_cu_seqlens;
  if (fused && c.NT == 1 && c.T <= kBT) return Dispatch::kSingleChunk;
  if (fused) return Dispatch::kFusedLong;
  const bool tailfree = c.K == 128 && c.V == 128 && c.block_v == 64 &&
                        c.num_warps == 4 && c.num_stages == 1 && !c.has_g &&
                        c.has_gk && c.has_v_new && c.has_cu_seqlens &&
                        !fused && c.tail_free;
  return tailfree ? Dispatch::kTailfree : Dispatch::kGeneral;
}

// `cuda_serves` answers a narrow question: would the CUDA kernel replace the
// branch Triton selected, assuming that branch is the tail-free or fused-long
// one? It is the safe answer for "may I skip the Triton kernel for this call",
// and it is deliberately conservative: a `general` call with K=V=128 and no
// scalar gate is executable by this kernel even though the mirror calls it
// general.
inline bool cuda_serves(Dispatch d) {
  return d == Dispatch::kTailfree || d == Dispatch::kFusedLong;
}

// `cuda_executes` answers the wider question: can the kernel run this call at
// all, with the same semantics, whichever branch Triton picked? This is the
// shape-coverage boundary.
inline bool cuda_executes(const Contract& c) {
  if (c.K <= 0 || c.K > 256 || c.V <= 0) return false;
  if (c.Hg <= 0 || c.H % c.Hg != 0) return false; // host function raises
  if (c.has_cu_seqlens && c.B != 1) return false; // packed varlen layout
  if (c.block_v <= 0 || c.num_warps <= 0 || c.num_stages <= 0) return false;
  // SaveNewValue=false has its own template instance and needs no v_new tensor.
  return true;
}


}  // namespace

namespace {

template <typename IndexT>
void launch_generic(
    const torch::Tensor& k, const torch::Tensor& w,
    const torch::Tensor& u, const torch::Tensor* g,
    const torch::Tensor* gk, const torch::Tensor& initial_state,
    const torch::Tensor& h,
    const torch::Tensor* v_new, const int* cu_seqlens,
    const int* chunk_offsets, bool ragged, int sequences, int heads,
    int chunks_per_sequence, const IndexT* state_indices,
    int64_t state_stride, bool use_exp2, cudaStream_t stream) {
  const int K = static_cast<int>(k.size(3));
  const int V = static_cast<int>(u.size(3));
  const int k_heads = static_cast<int>(k.size(2));
  // Ragged calls read exact boundaries from cu_seqlens. Every non-ragged
  // sequence is a whole number of BT chunks, including packed B=1/N>1.
  const int tokens_per_sequence = chunks_per_sequence * kBT;
  const dim3 grid(V, sequences * heads);
  chunk_delta_generic_kernel<IndexT><<<grid, 256, 0, stream>>>(
      k.data_ptr(), w.data_ptr(), u.data_ptr(),
      g != nullptr ? g->data_ptr() : nullptr,
      gk != nullptr ? gk->data_ptr() : nullptr, initial_state.data_ptr(),
      h.data_ptr(), v_new != nullptr ? v_new->data_ptr() : nullptr,
      sequences, heads, k_heads, K, V, tokens_per_sequence,
      chunks_per_sequence, state_indices, state_stride, cu_seqlens,
      chunk_offsets, ragged, g != nullptr, gk != nullptr, v_new != nullptr,
      use_exp2, native_dtype_code(k.scalar_type(), "k"),
      native_dtype_code(w.scalar_type(), "w"),
      native_dtype_code(u.scalar_type(), "u"),
      g != nullptr ? native_dtype_code(g->scalar_type(), "g")
                   : kNativeFloat32,
      gk != nullptr ? native_dtype_code(gk->scalar_type(), "gk")
                    : kNativeFloat32,
      native_dtype_code(initial_state.scalar_type(), "initial_state"));
}

template <typename IndexT>
void launch_native(const torch::Tensor& k, const torch::Tensor& w,
                   const torch::Tensor& u, const float* g_ptr,
                   const float* gk_ptr, bool has_gk,
                   float* initial_state, bf16* h, bf16* v_new,
                   const int* cu_seqlens, const int* chunk_offsets,
                   bool ragged, bool state_bf16, int sequences, int heads,
                   int chunks_per_sequence, const IndexT* state_indices,
                   long long state_stride, int k_heads, int use_exp2,
                   bool gqa, bool exp2, bool use_scalar_g,
                   bool save_new_value,
                   cudaStream_t stream) {
  const dim3 grid(kV / 64, sequences * heads);
  const bf16* kp = reinterpret_cast<const bf16*>(k.data_ptr());
  const bf16* wp = reinterpret_cast<const bf16*>(w.data_ptr());
  const bf16* up = reinterpret_cast<const bf16*>(u.data_ptr());
#define MCOPLIB_LAUNCH(NV, G, E, UG, R, S)                                     \
  chunk_delta_native_kernel<false, true, G, E, UG, R, S, NV, 64, IndexT>       \
      <<<grid, kThreads, 0, stream>>>(                                          \
          kp, wp, up, g_ptr, gk_ptr, has_gk, initial_state, h, v_new,          \
          sequences, heads, chunks_per_sequence, state_indices, state_stride, \
          k_heads, use_exp2, cu_seqlens, chunk_offsets)
#define MCOPLIB_SWITCH_GK(NV)                                                   \
  do {                                                                         \
    switch (which) {                                                           \
      case 0: MCOPLIB_LAUNCH(NV, false, false, false, false, false); break;    \
      case 1: MCOPLIB_LAUNCH(NV, false, false, false, false, true); break;     \
      case 2: MCOPLIB_LAUNCH(NV, false, false, false, true, false); break;     \
      case 3: MCOPLIB_LAUNCH(NV, false, false, false, true, true); break;      \
      case 4: MCOPLIB_LAUNCH(NV, false, true, false, false, false); break;     \
      case 5: MCOPLIB_LAUNCH(NV, false, true, false, false, true); break;      \
      case 6: MCOPLIB_LAUNCH(NV, false, true, false, true, false); break;      \
      case 7: MCOPLIB_LAUNCH(NV, false, true, false, true, true); break;       \
      case 8: MCOPLIB_LAUNCH(NV, true, false, false, false, false); break;     \
      case 9: MCOPLIB_LAUNCH(NV, true, false, false, false, true); break;      \
      case 10: MCOPLIB_LAUNCH(NV, true, false, false, true, false); break;     \
      case 11: MCOPLIB_LAUNCH(NV, true, false, false, true, true); break;      \
      case 12: MCOPLIB_LAUNCH(NV, true, true, false, false, false); break;     \
      case 13: MCOPLIB_LAUNCH(NV, true, true, false, false, true); break;      \
      case 14: MCOPLIB_LAUNCH(NV, true, true, false, true, false); break;      \
      case 15: MCOPLIB_LAUNCH(NV, true, true, false, true, true); break;       \
      default: TORCH_CHECK(false, "unreachable native dispatch");             \
    }                                                                          \
  } while (false)
#define MCOPLIB_SWITCH_G(NV)                                                    \
  do {                                                                         \
    switch (which_scalar) {                                                    \
      case 0: MCOPLIB_LAUNCH(NV, false, false, true, false, false); break;     \
      case 1: MCOPLIB_LAUNCH(NV, false, false, true, false, true); break;      \
      case 2: MCOPLIB_LAUNCH(NV, false, false, true, true, false); break;      \
      case 3: MCOPLIB_LAUNCH(NV, false, false, true, true, true); break;       \
      case 4: MCOPLIB_LAUNCH(NV, true, false, true, false, false); break;      \
      case 5: MCOPLIB_LAUNCH(NV, true, false, true, false, true); break;       \
      case 6: MCOPLIB_LAUNCH(NV, true, false, true, true, false); break;       \
      case 7: MCOPLIB_LAUNCH(NV, true, false, true, true, true); break;        \
      default: TORCH_CHECK(false, "unreachable scalar-g dispatch");           \
    }                                                                          \
  } while (false)
  const int which = (gqa ? 8 : 0) | (exp2 ? 4 : 0) | (ragged ? 2 : 0) |
                    (state_bf16 ? 1 : 0);
  const int which_scalar =
      (gqa ? 4 : 0) | (ragged ? 2 : 0) | (state_bf16 ? 1 : 0);
  if (use_scalar_g && save_new_value) {
    MCOPLIB_SWITCH_G(true);
  } else if (use_scalar_g) {
    MCOPLIB_SWITCH_G(false);
  } else if (save_new_value) {
    MCOPLIB_SWITCH_GK(true);
  } else {
    MCOPLIB_SWITCH_GK(false);
  }
#undef MCOPLIB_SWITCH_G
#undef MCOPLIB_SWITCH_GK
#undef MCOPLIB_LAUNCH
}

}  // namespace

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
chunk_gated_delta_rule_fwd_h(
    torch::Tensor const& k, torch::Tensor const& w, torch::Tensor const& u,
    std::optional<torch::Tensor> const& g,
    std::optional<torch::Tensor> const& gk,
    std::optional<torch::Tensor> const& initial_state,
    std::optional<torch::Tensor> const& initial_state_indices,
    bool save_new_value,
    std::optional<torch::Tensor> const& cu_seqlens,
    std::optional<torch::Tensor> const& chunk_indices,
    bool use_exp2, std::optional<int64_t> block_v,
    std::optional<int64_t> num_warps, int64_t num_stages) {
  TORCH_CHECK(k.dim() == 4, "k must have shape [B, T, Hg, K]");
  TORCH_CHECK(u.dim() == 4, "u must have shape [B, T, H, V]");
  int64_t const B = k.size(0);
  int64_t const T = k.size(1);
  int64_t const Hg = k.size(2);
  int64_t const K = k.size(3);
  int64_t const H = u.size(2);
  int64_t const V = u.size(3);
  int64_t const resolved_block_v = block_v.value_or(
      K == kK && V == kV ? 64 : 32);
  int64_t const resolved_num_warps = num_warps.value_or(4);

  TORCH_CHECK(K <= 256, "current kernel does not support K > 256");
  TORCH_CHECK(Hg > 0 && H % Hg == 0, "H (", H,
              ") must be divisible by Hg (", Hg, ")");
  TORCH_CHECK(initial_state.has_value() &&
                  initial_state_indices.has_value(),
              "initial_state and initial_state_indices are required because "
              "the production kernel updates the final state in place");
  TORCH_CHECK(!(use_exp2 && g.has_value()),
              "use_exp2 selects the base-2 gate path; it only applies to the "
              "per-channel gk argument and is incompatible with the scalar g "
              "argument");

  TORCH_CHECK_NOT_IMPLEMENTED(
      K > 0 && V > 0,
      "chunk_gated_delta_rule_fwd_h (CUDA): K and V must be positive");
  TORCH_CHECK_NOT_IMPLEMENTED(
      resolved_block_v > 0 && resolved_num_warps > 0 && num_stages > 0,
      "chunk_gated_delta_rule_fwd_h (CUDA): launch tuning values must be "
      "positive");
  TORCH_CHECK_NOT_IMPLEMENTED(
      !cu_seqlens.has_value() || B == 1,
      "chunk_gated_delta_rule_fwd_h (CUDA): cu_seqlens uses the packed B=1 "
      "layout; dense B>1 calls must omit cu_seqlens");

  torch::Tensor state = *initial_state;
  torch::Tensor state_indices = *initial_state_indices;
  TORCH_CHECK(state_indices.scalar_type() == at::kInt ||
                  state_indices.scalar_type() == at::kLong,
              "initial_state_indices must be int32 or int64");
  state_indices = state_indices.contiguous();

  ChunkPlan plan;
  if (cu_seqlens.has_value()) {
    torch::Tensor const& cu = *cu_seqlens;
    TORCH_CHECK(cu.is_cuda() && cu.device() == k.device(),
                "cu_seqlens must be on the same CUDA device as k");
    TORCH_CHECK(cu.dim() == 1 &&
                    (cu.scalar_type() == at::kInt ||
                     cu.scalar_type() == at::kLong),
                "cu_seqlens must be a rank-1 int32 or int64 tensor");
    auto cached = chunk_plan_cache().lookup(cu, T);
    if (cached.has_value()) {
      plan = std::move(*cached);
    } else {
      plan = build_chunk_plan(cu, T);
      chunk_plan_cache().insert(plan);
    }
  } else {
    plan.total_tokens = B * T;
    plan.sequences = B;
    plan.chunks_per_sequence = (T + kBT - 1) / kBT;
    plan.total_chunks = B * plan.chunks_per_sequence;
    plan.ragged = T % kBT != 0;
    std::vector<int64_t> cu_values;
    std::vector<int64_t> offsets;
    cu_values.reserve(B + 1);
    offsets.reserve(B + 1);
    for (int64_t i = 0; i <= B; ++i) {
      cu_values.push_back(i * T);
      offsets.push_back(i * plan.chunks_per_sequence);
    }
    plan.cu_i32 = torch::tensor(
        cu_values,
        torch::TensorOptions().device(k.device()).dtype(torch::kInt32));
    plan.offsets_i32 = torch::tensor(
        offsets,
        torch::TensorOptions().device(k.device()).dtype(torch::kInt32));
  }
  TORCH_CHECK(state_indices.numel() >= plan.sequences,
              "initial_state_indices has ", state_indices.numel(),
              " entries but ", plan.sequences, " sequences are present");
  if (chunk_indices.has_value()) {
    TORCH_CHECK(chunk_indices->dim() >= 1 &&
                    chunk_indices->size(0) == plan.total_chunks,
                "chunk_indices length must match the chunks described by "
                "cu_seqlens");
  }

  const int64_t output_chunks =
      cu_seqlens.has_value() ? plan.total_chunks : plan.chunks_per_sequence;
  auto h = torch::empty({B, output_chunks, H, V, K}, k.options());
  std::optional<torch::Tensor> v_new = std::nullopt;
  if (save_new_value) {
    v_new = torch::empty_like(u);
  }
  chunk_gated_delta_rule_fwd_h_native_out(
      k, w, u, gk, state, state_indices, plan.cu_i32, plan.offsets_i32, h,
      v_new, plan.sequences, plan.chunks_per_sequence, use_exp2, plan.ragged,
      state.scalar_type() == at::kBFloat16, g);
  return {h, v_new};
}

void chunk_gated_delta_rule_fwd_h_native_out(
    torch::Tensor const& k, torch::Tensor const& w, torch::Tensor const& u,
    std::optional<torch::Tensor> const& gk,
    torch::Tensor const& initial_state,
    torch::Tensor const& initial_state_indices,
    torch::Tensor const& cu_seqlens, torch::Tensor const& chunk_offsets,
    torch::Tensor h, std::optional<torch::Tensor> const& v_new,
    int64_t sequences,
    int64_t chunks_per_sequence, bool use_exp2, bool ragged,
    bool state_bf16, std::optional<torch::Tensor> const& g) {
  const bool save_new_value = v_new.has_value();
  const bool has_g = g.has_value();
  const bool has_gk = gk.has_value();
  TORCH_CHECK(!(use_exp2 && has_g),
              "chunk_gated_delta_rule_fwd_h(native): use_exp2 is incompatible with g");
  TORCH_CHECK(k.is_cuda() && w.is_cuda() && u.is_cuda() &&
                  (!has_g || g->is_cuda()) && (!has_gk || gk->is_cuda()) &&
                  initial_state.is_cuda() && initial_state_indices.is_cuda() &&
                  cu_seqlens.is_cuda() && chunk_offsets.is_cuda() &&
                  h.is_cuda() &&
                  (!save_new_value || v_new->is_cuda()),
              "chunk_gated_delta_rule_fwd_h(native): inputs must be CUDA");
  TORCH_CHECK(w.device() == k.device() && u.device() == k.device() &&
                  (!has_g || g->device() == k.device()) &&
                  (!has_gk || gk->device() == k.device()) &&
                  initial_state.device() == k.device() &&
                  initial_state_indices.device() == k.device() &&
                  cu_seqlens.device() == k.device() &&
                  chunk_offsets.device() == k.device() &&
                  h.device() == k.device() &&
                  (!save_new_value || v_new->device() == k.device()),
              "chunk_gated_delta_rule_fwd_h(native): tensors must be on the same device");
  TORCH_CHECK(k.dim() == 4 && w.dim() == 4 && u.dim() == 4,
              "chunk_gated_delta_rule_fwd_h(native): k/w/u must be rank 4");
  TORCH_CHECK(!has_g || g->dim() == 3,
              "chunk_gated_delta_rule_fwd_h(native): g must be rank 3");
  TORCH_CHECK(!has_gk || gk->dim() == 4,
              "chunk_gated_delta_rule_fwd_h(native): gk must be rank 4");
  TORCH_CHECK(w.size(0) == k.size(0) && u.size(0) == k.size(0) &&
                  (!has_g || g->size(0) == k.size(0)) &&
                  (!has_gk || gk->size(0) == k.size(0)),
              "chunk_gated_delta_rule_fwd_h(native): batch dimensions must match");
  TORCH_CHECK(w.size(1) == k.size(1) && u.size(1) == k.size(1) &&
                  (!has_g || g->size(1) == k.size(1)) &&
                  (!has_gk || gk->size(1) == k.size(1)),
              "chunk_gated_delta_rule_fwd_h(native): token dimensions must match");
  TORCH_CHECK(w.size(2) == u.size(2) &&
                  (!has_g || g->size(2) == u.size(2)) &&
                  (!has_gk || (gk->size(2) == u.size(2) &&
                               gk->size(3) == k.size(3))) &&
                  w.size(3) == k.size(3),
              "chunk_gated_delta_rule_fwd_h(native): head/K dimensions must match");
  native_dtype_code(k.scalar_type(), "k");
  native_dtype_code(w.scalar_type(), "w");
  native_dtype_code(u.scalar_type(), "u");
  if (has_g) native_dtype_code(g->scalar_type(), "g");
  if (has_gk) native_dtype_code(gk->scalar_type(), "gk");
  native_dtype_code(initial_state.scalar_type(), "initial_state");
  TORCH_CHECK(h.scalar_type() == k.scalar_type(),
              "chunk_gated_delta_rule_fwd_h(native): h.dtype must match k.dtype");
  TORCH_CHECK(!save_new_value || v_new->scalar_type() == u.scalar_type(),
              "chunk_gated_delta_rule_fwd_h(native): v_new.dtype must match u.dtype");
  TORCH_CHECK(state_bf16 == (initial_state.scalar_type() == at::kBFloat16),
              "chunk_gated_delta_rule_fwd_h(native): state_bf16 disagrees with "
              "initial_state.dtype");
  TORCH_CHECK(initial_state_indices.scalar_type() == at::kInt ||
                  initial_state_indices.scalar_type() == at::kLong,
              "chunk_gated_delta_rule_fwd_h(native): initial_state_indices "
              "must be int32 or int64");
  TORCH_CHECK(k.is_contiguous() && w.is_contiguous() && u.is_contiguous() &&
                  (!has_g || g->is_contiguous()) &&
                  (!has_gk || gk->is_contiguous()) && h.is_contiguous() &&
                  (!save_new_value || v_new->is_contiguous()) &&
                  initial_state_indices.is_contiguous() &&
                  cu_seqlens.is_contiguous() && chunk_offsets.is_contiguous(),
              "chunk_gated_delta_rule_fwd_h(native): tensors must be contiguous");
  TORCH_CHECK(k.size(3) > 0 && k.size(3) <= 256 && u.size(-1) > 0,
              "chunk_gated_delta_rule_fwd_h(native): require 0 < K <= 256 "
              "and V > 0");
  TORCH_CHECK(k.size(2) > 0 && u.size(-2) % k.size(2) == 0,
              "chunk_gated_delta_rule_fwd_h(native): H must be divisible by Hg");
  TORCH_CHECK(initial_state.dim() == 4 &&
                  initial_state.size(1) == u.size(2) &&
                  initial_state.size(2) == u.size(3) &&
                  initial_state.size(3) == k.size(3),
              "chunk_gated_delta_rule_fwd_h(native): initial_state must be "
              "[slots,H,V,K]");
  TORCH_CHECK(!save_new_value || v_new->sizes() == u.sizes(),
              "chunk_gated_delta_rule_fwd_h(native): v_new shape must match u");
  TORCH_CHECK(h.dim() == 5 && h.size(2) == u.size(2) &&
                  h.size(3) == u.size(3) && h.size(4) == k.size(3),
              "chunk_gated_delta_rule_fwd_h(native): invalid h shape");
  TORCH_CHECK(initial_state.stride(-1) == 1 &&
                  initial_state.stride(-2) == k.size(3) &&
                  initial_state.stride(-3) == u.size(3) * k.size(3),
              "chunk_gated_delta_rule_fwd_h(native): the inner (V,K) block of "
              "initial_state must be dense; pass a contiguous tensor or a "
              "slot-strided view whose inner block is dense");
  TORCH_CHECK(initial_state_indices.numel() >= sequences,
              "chunk_gated_delta_rule_fwd_h(native): initial_state_indices too short");

  if (ragged) {
    TORCH_CHECK(cu_seqlens.defined() && cu_seqlens.is_cuda() &&
                    cu_seqlens.scalar_type() == at::kInt &&
                    cu_seqlens.is_contiguous() &&
                    cu_seqlens.numel() >= sequences + 1,
                "chunk_gated_delta_rule_fwd_h(native): ragged calls need int32 "
                "cu_seqlens with sequences + 1 entries");
    TORCH_CHECK(chunk_offsets.defined() && chunk_offsets.is_cuda() &&
                    chunk_offsets.scalar_type() == at::kInt &&
                    chunk_offsets.is_contiguous() &&
                    chunk_offsets.numel() >= sequences + 1,
                "chunk_gated_delta_rule_fwd_h(native): ragged calls need int32 "
                "chunk_offsets with sequences + 1 entries");
  }
  const int* cu_ptr =
      (ragged && cu_seqlens.defined()) ? cu_seqlens.data_ptr<int>() : nullptr;
  const int* choff_ptr = (ragged && chunk_offsets.defined())
                             ? chunk_offsets.data_ptr<int>()
                             : nullptr;

  const int heads = static_cast<int>(u.size(-2));
  const int k_heads = static_cast<int>(k.size(2));
  const long long state_stride = initial_state.stride(0);
  const bool gqa = k_heads != heads;

  at::cuda::OptionalCUDAGuard const guard(device_of(k));
  cudaStream_t const stream = at::cuda::getCurrentCUDAStream();
  const bool optimized_contract =
      k.size(0) == 1 && k.size(3) == kK && u.size(3) == kV;
  // Every optimized instance has either UseG=true or an unconditional gk
  // load. Gate-free calls therefore use the generic CUDA recurrence, whose
  // natural identity decay is exactly the Triton USE_G=USE_GK=false branch.
  const bool optimized_dtype = optimized_contract && (has_g || has_gk) &&
      k.scalar_type() == at::kBFloat16 &&
      w.scalar_type() == at::kBFloat16 &&
      u.scalar_type() == at::kBFloat16 &&
      (!has_g || g->scalar_type() == at::kFloat) &&
      (!has_gk || gk->scalar_type() == at::kFloat) &&
      (initial_state.scalar_type() == at::kFloat ||
       initial_state.scalar_type() == at::kBFloat16);
  if (!optimized_dtype) {
    if (initial_state_indices.scalar_type() == at::kInt) {
      launch_generic<int32_t>(
          k, w, u, has_g ? &*g : nullptr, has_gk ? &*gk : nullptr,
          initial_state, h, save_new_value ? &*v_new : nullptr, cu_ptr,
          choff_ptr, ragged, static_cast<int>(sequences), heads,
          static_cast<int>(chunks_per_sequence),
          initial_state_indices.data_ptr<int32_t>(), state_stride, use_exp2,
          stream);
    } else {
      launch_generic<int64_t>(
          k, w, u, has_g ? &*g : nullptr, has_gk ? &*gk : nullptr,
          initial_state, h, save_new_value ? &*v_new : nullptr, cu_ptr,
          choff_ptr, ragged, static_cast<int>(sequences), heads,
          static_cast<int>(chunks_per_sequence),
          initial_state_indices.data_ptr<int64_t>(), state_stride, use_exp2,
          stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }

  // A bf16 state is passed as a raw pointer and reinterpreted inside the
  // StateBf16 instantiation; data_ptr<T>() would reject the type mismatch.
  float* state_ptr = state_bf16
                         ? reinterpret_cast<float*>(initial_state.data_ptr())
                         : initial_state.data_ptr<float>();
  bf16* const v_new_ptr = save_new_value
                              ? reinterpret_cast<bf16*>(v_new->data_ptr())
                              : nullptr;
  const float* const g_ptr = has_g ? g->data_ptr<float>() : nullptr;
  const float* const gk_ptr = has_gk ? gk->data_ptr<float>() : nullptr;
  if (initial_state_indices.scalar_type() == at::kInt) {
    launch_native<int32_t>(
        k, w, u, g_ptr, gk_ptr, has_gk, state_ptr,
        reinterpret_cast<bf16*>(h.data_ptr()), v_new_ptr, cu_ptr, choff_ptr,
        ragged, state_bf16,
        static_cast<int>(sequences), heads,
        static_cast<int>(chunks_per_sequence),
        initial_state_indices.data_ptr<int32_t>(), state_stride, k_heads,
        use_exp2 ? 1 : 0, gqa, use_exp2, has_g, save_new_value, stream);
  } else {
    launch_native<int64_t>(
        k, w, u, g_ptr, gk_ptr, has_gk, state_ptr,
        reinterpret_cast<bf16*>(h.data_ptr()), v_new_ptr, cu_ptr, choff_ptr,
        ragged, state_bf16,
        static_cast<int>(sequences), heads,
        static_cast<int>(chunks_per_sequence),
        initial_state_indices.data_ptr<int64_t>(), state_stride, k_heads,
        use_exp2 ? 1 : 0, gqa, use_exp2, has_g, save_new_value, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::string chunk_gated_delta_rule_fwd_h_coverage(
    int64_t B, int64_t T, int64_t Hg, int64_t K, int64_t H, int64_t V,
    int64_t block_v, int64_t num_warps, int64_t num_stages, int64_t N,
    int64_t NT, bool has_g, bool has_gk, bool has_v_new, bool has_cu_seqlens,
    bool tail_free) {
  Contract c;
  c.B = static_cast<int>(B);
  c.T = static_cast<int>(T);
  c.Hg = static_cast<int>(Hg);
  c.K = static_cast<int>(K);
  c.H = static_cast<int>(H);
  c.V = static_cast<int>(V);
  c.block_v = static_cast<int>(block_v);
  c.num_warps = static_cast<int>(num_warps);
  c.num_stages = static_cast<int>(num_stages);
  c.N = static_cast<int>(N);
  c.NT = static_cast<int>(NT);
  c.has_g = has_g ? 1 : 0;
  c.has_gk = has_gk ? 1 : 0;
  c.has_v_new = has_v_new ? 1 : 0;
  c.has_cu_seqlens = has_cu_seqlens ? 1 : 0;
  c.tail_free = tail_free ? 1 : 0;
  const Dispatch d = cuda_dispatch(c);
  return std::string(dispatch_name(d)) + " " +
         (cuda_serves(d) ? "1" : "0") + " " + (cuda_executes(c) ? "1" : "0");
}
