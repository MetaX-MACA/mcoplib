#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Optional.h>

#include "cuda_utils.h"
#include "cuda_compat.h"
#include "dispatch_utils.h"
#include "quantization/vectorization_utils.cuh"

#include "quantization/w8a8/fp8/metax/quant_utils.cuh"
#include "concat_mla_q.cuh"
#include <c10/util/Float8_e4m3fn.h>
#include <algorithm>
#include <cassert>
#include <map>
#include <cfloat>
#include <vector>

#if defined(__gfx942__)
constexpr float kFp8ScaleDivisor = 224.f;
#else
constexpr float kFp8ScaleDivisor = 448.f;
#endif

void swap_blocks(torch::Tensor& src, torch::Tensor& dst,
                 const torch::Tensor& block_mapping) {
  torch::Device src_device = src.device();
  torch::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    TORCH_CHECK(src_device.index() == dst_device.index(),
                "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  TORCH_CHECK(block_mapping.device().is_cpu(), "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  // We use the stride instead of numel in case the cache is padded for memory
  // alignment reasons, we assume the blocks data (inclusive of any padding)
  // is contiguous in memory
  const int64_t block_size_in_bytes = src.element_size() * src.stride(0);
  const at::cuda::OptionalCUDAGuard device_guard(
      src_device.is_cuda() ? src_device : dst_device);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = block_mapping[i][0].item<int64_t>();
    int64_t dst_block_number = block_mapping[i][1].item<int64_t>();
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
}

void swap_blocks_batch(const torch::Tensor& src_ptrs,
                       const torch::Tensor& dst_ptrs,
                       const torch::Tensor& sizes,
                       bool is_src_access_order_any) {
  TORCH_CHECK(src_ptrs.device().is_cpu(), "src_ptrs must be on CPU");
  TORCH_CHECK(dst_ptrs.device().is_cpu(), "dst_ptrs must be on CPU");
  TORCH_CHECK(sizes.device().is_cpu(), "sizes must be on CPU");
  TORCH_CHECK(src_ptrs.dtype() == torch::kInt64, "src_ptrs must be int64");
  TORCH_CHECK(dst_ptrs.dtype() == torch::kInt64, "dst_ptrs must be int64");
  TORCH_CHECK(sizes.dtype() == torch::kInt64, "sizes must be int64");

  const int64_t n = src_ptrs.size(0);
  TORCH_CHECK(dst_ptrs.size(0) == n, "dst_ptrs length must match src_ptrs");
  TORCH_CHECK(sizes.size(0) == n, "sizes length must match src_ptrs");

  if (n == 0) return;

  int64_t* src_data = src_ptrs.mutable_data_ptr<int64_t>();
  int64_t* dst_data = dst_ptrs.mutable_data_ptr<int64_t>();
  int64_t* size_data = sizes.mutable_data_ptr<int64_t>();

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Use cuMemcpyBatchAsync / hipMemcpyBatchAsync to submit all copies in a
  // single driver call, amortizing per-copy submission overhead. int64_t
  // and CUdeviceptr/void*/size_t are all 8 bytes on 64-bit platforms, so we
  // reinterpret_cast the tensor data directly to avoid copies.
  static_assert(sizeof(size_t) == sizeof(int64_t));
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  static_assert(sizeof(CUdeviceptr) == sizeof(int64_t));
  // Resolve cuMemcpyBatchAsync at runtime via cuGetProcAddress so that
  // binaries compiled with CUDA 12.8+ still work on older drivers, and
  // we avoid the CUDA 13.0 header remapping (#define to _v2 signature).
  // The function pointer is cached after the first call.
  using BatchFn =
      CUresult (*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t,
                   CUmemcpyAttributes*, size_t*, size_t, size_t*, CUstream);
  static BatchFn batch_fn = []() -> BatchFn {
    CUdriverProcAddressQueryResult sym_status;
    void* fn_ptr = nullptr;
    CUresult res = cuGetProcAddress("cuMemcpyBatchAsync", &fn_ptr, 12080,
                                    CU_GET_PROC_ADDRESS_DEFAULT, &sym_status);
    if (res != CUDA_SUCCESS || fn_ptr == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<BatchFn>(fn_ptr);
  }();

  // cuMemcpyBatchAsync rejects the legacy default stream (handle 0 /
  // cudaStreamLegacy) with CUDA_ERROR_INVALID_VALUE; route it to the per-copy
  // fallback below, which is correct on any stream. Real and per-thread-default
  // streams take the batch fast path.
  const bool usable_stream = stream != nullptr && stream != cudaStreamLegacy;
  if (batch_fn != nullptr && usable_stream) {
    CUmemcpyAttributes attr = {};
    // ANY lets the DMA engine prefetch source bytes out of stream order,
    // which is only safe when no GPU stream is concurrently writing the
    // source.
    attr.srcAccessOrder = is_src_access_order_any
                              ? CU_MEMCPY_SRC_ACCESS_ORDER_ANY
                              : CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    size_t attrs_idx = 0;
    size_t fail_idx = 0;
    CUresult result = batch_fn(reinterpret_cast<CUdeviceptr*>(dst_data),
                               reinterpret_cast<CUdeviceptr*>(src_data),
                               reinterpret_cast<size_t*>(size_data),
                               static_cast<size_t>(n), &attr, &attrs_idx, 1,
                               &fail_idx, static_cast<CUstream>(stream));
    TORCH_CHECK(result == CUDA_SUCCESS, "cuMemcpyBatchAsync failed at index ",
                fail_idx, " with error ", result);
    return;
  }
#elif defined(USE_ROCM) && defined(HIP_VERSION) && HIP_VERSION >= 70100000
  // ROCm 7.1+ exposes hipMemcpyBatchAsync. The 7.2.1 implementation early-
  // returns hipErrorNotSupported whenever numAttrs > 0 (see ROCm/clr @
  // rocm-7.2.1 hipamd/src/hip_memory.cpp:2819-2822), so call with
  // numAttrs=0.
  {
    hipMemcpyAttributes attr = {};
    size_t attrs_idx = 0;
    size_t fail_idx = 0;
    hipError_t result = hipMemcpyBatchAsync(
        reinterpret_cast<void**>(dst_data), reinterpret_cast<void**>(src_data),
        reinterpret_cast<size_t*>(size_data), static_cast<size_t>(n), &attr,
        &attrs_idx, 0, &fail_idx, static_cast<hipStream_t>(stream));
    TORCH_CHECK(result == hipSuccess, "hipMemcpyBatchAsync failed at index ",
                fail_idx, " with error ", result);
    return;
  }
#endif
  {
    // Fallback for CUDA < 12.8, older CUDA drivers, and ROCm < 7.1:
    // individual async copies. cudaMemcpyDefault lets the driver infer
    // direction from pointer types.
    for (int64_t i = 0; i < n; i++) {
      cudaMemcpyAsync(reinterpret_cast<void*>(dst_data[i]),
                      reinterpret_cast<void*>(src_data[i]),
                      static_cast<size_t>(size_data[i]), cudaMemcpyDefault,
                      stream);
    }
  }
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

// Kernel for MLA, which works on a single joint kv_cache
// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_mla_kernel(
    int64_t* cache_ptrs, const int64_t* __restrict__ block_mapping,
    const int mem_footprint_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;
  scalar_t* cache = reinterpret_cast<scalar_t*>(cache_ptrs[layer_idx]);
  int64_t src_block = block_mapping[2 * pair_idx];
  int64_t dst_block = block_mapping[2 * pair_idx + 1];
  int64_t src_offset = src_block * mem_footprint_per_block;
  int64_t dst_offset = dst_block * mem_footprint_per_block;
  for (int i = threadIdx.x; i < mem_footprint_per_block; i += blockDim.x) {
    cache[dst_offset + i] = cache[src_offset + i];
  }
}

}  // namespace vllm

namespace vllm {

// Used to copy/convert one element.
// Quantized paths use the STATIC per-tensor convention shared with the unit
// test: scale = absmax / qmax, stored quant value q = round(x / scale),
// dequant = q * scale.  (qmax: int8 = 127, fp8 e4m3 = 448.)  This mirrors
// dynamic_scaled_int8_quant / per_token_cast_to_fp8 arithmetic but with a
// host-provided per-tensor scale, since the paged cache layout has nowhere to
// store per-token scales.  fp8::scaled_convert is NOT used: its only
// definition on this backend is the assert(false) primary template.
// Hardware fp8 (e4m3) pack: converts 4 float32 -> 4 packed e4m3 bytes in ONE
// instruction (round-to-nearest-even + saturate to +-448), returning the 4
// bytes as a uint32. Verified bit-identical (8192 varied values incl. 0,
// subnormals, +-500 saturation) to the MACA-vetted __cvt_rn_satfinite_e4m3x2_f32
// helper, which is what torch's `.to(float8_e4m3fn)` lowers to on this backend.
// Replaces the per-element software c10::Float8_e4m3fn narrowing that made the
// fp8 store compute-bound (fp8 ~289 vs int8 ~417 GB/s at identical traffic).
__device__ __forceinline__ uint32_t rc_pack4_fp8_e4m3(float a, float b, float c,
                                                       float d) {
  using v4f32 = float __attribute__((ext_vector_type(4)));
  v4f32 v = {a, b, c, d};
#pragma unroll
  for (int k = 0; k < 4; k++) v[k] = fminf(fmaxf(v[k], -448.f), 448.f);
  return __builtin_mxc_cvt_pk4_f32tof8(v);
}

template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  // For the quant paths this holds the RECIPROCAL of the per-tensor scale, so
  // the per-element hot loop is a multiply (x * inv_scale) instead of an fp32
  // divide (x / scale). The reciprocal is computed ONCE per launch (see the
  // kernel prologue), not per element, so it costs nothing extra and removes
  // head_size divides per token per head -- the dominant compute cost that kept
  // the quant paths well below the bf16 transpose ceiling.
  float inv_scale;

  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kInt8) {
      // int8: round-to-nearest-even then saturate to [-127, 127].
      float f = static_cast<float>(src) * inv_scale;
      int32_t q = __float2int_rn(f);
      q = min(q, 127);
      q = max(q, -127);
      dst = static_cast<OutT>(q);
    } else {
      // fp8 e4m3: clamp to representable range then narrow.
      float f = static_cast<float>(src) * inv_scale;
      f = fminf(fmaxf(f, -448.f), 448.f);
      dst = static_cast<OutT>(c10::Float8_e4m3fn(f).x);
    }
  }
};

// Tile of tokens processed per thread-block by the fast SMEM-transpose path.
#define RC_TILE_T 64

// Env-gated probes (compile-time) to isolate the bandwidth limiter. Default 0.
//   RC_PROBE_NO_VALUE : skip the value transpose+store entirely.
//   RC_PROBE_NO_KEY   : skip the key store entirely.
#ifndef RC_PROBE_NO_VALUE
#define RC_PROBE_NO_VALUE 0
#endif
#ifndef RC_PROBE_NO_KEY
#define RC_PROBE_NO_KEY 0
#endif
// RC_FORCE_SLOW: disable the fast SMEM-transpose+vectorized path to measure the
// naive scalar-strided-scatter baseline (the "before" reference). Default 0.
#ifndef RC_FORCE_SLOW
#define RC_FORCE_SLOW 0
#endif

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    cache_t* __restrict__ key_cache,
    cache_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale,
    const int64_t total_work, const int h_block_count,
    const int num_tokens) {
  // Pass the RECIPROCAL of the per-tensor scale so the hot loop multiplies
  // instead of dividing. Computed ONCE here (correctly-rounded 1/scale, cheap
  // enough that the single divide is irrelevant to bandwidth) -- this removes
  // head_size fp32 divides per token per head, the dominant instruction cost
  // that kept the quant paths below the bf16 transpose ceiling. (kAuto ignores
  // the value.)
  float k_scale_val =
      (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : __builtin_mxc_rcpf(*k_scale);
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  float v_scale_val =
      (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : __builtin_mxc_rcpf(*v_scale);
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

  constexpr int VEC = 8;  // bf16 uint4 = 16 bytes
  constexpr int TILE_T = RC_TILE_T;

  // Grid: (num_heads, num_token_tiles). Each block owns one head and a tile of
  // TILE_T tokens. Value is transposed through shared memory so BOTH the global
  // read (contiguous in head_size) and the global write (contiguous in slot for
  // consecutive tokens) are 16-byte coalesced uint4 transactions -- this breaks
  // the value-cache d<->slot transpose that otherwise caps bandwidth at the
  // scattered-2-byte-store rate.
  const int head_idx = blockIdx.x;
  const int tile = blockIdx.y;
  const int tok0 = tile * TILE_T;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int tile_tokens = min(TILE_T, num_tokens - tok0);
  if (tile_tokens <= 0) return;

  // Padded token-major SMEM: smem[t * SP + d], SP multiple of VEC so the
  // 16-byte uint4 loads on the store side stay aligned.
  const int SP = head_size + VEC;
  extern __shared__ char smem_raw[];
  scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
  // Per-tile slot cache + per-chunk metadata, placed after the value tile.
  constexpr int MAXCH = TILE_T / VEC;
  char* meta_base = smem_raw + (size_t)TILE_T * SP * sizeof(scalar_t);
  int64_t* s_slot = reinterpret_cast<int64_t*>(meta_base);
  int64_t* s_blk  = s_slot + TILE_T;
  int64_t* s_off  = s_blk + MAXCH;
  int*     s_ctg  = reinterpret_cast<int*>(s_off + MAXCH);

  // Fast SMEM-transpose path. Value is staged into SMEM as bf16 (scalar_t) and
  // quantized on the coalesced store side, so it works for bf16 (kAuto) AND the
  // fp8/int8 quant dtypes -- the ONLY structural requirement is x == VEC so a
  // key x-block and an 8-token value run each pack into one vector store.
  const bool fast = (sizeof(scalar_t) == 2) && (x == VEC) &&
                    ((head_size % VEC) == 0) && !RC_FORCE_SLOW;

  const int vecs_per_tok = head_size / VEC;
  // Value store packs VST consecutive tokens (fixed d) into ONE 16-byte uint4
  // store along the slot axis: bf16 -> 8 tokens (8*2B), int8/fp8 -> 16 tokens
  // (16*1B). Widening the quant store from uint2 (8B) to uint4 (16B) halves the
  // value store-transaction count. bf16 (VST==VEC==8) is unchanged.
  constexpr int VST = 16 / sizeof(cache_t);
  const int tok_chunks = (tile_tokens + VST - 1) / VST;

  // Cache slot_mapping for this tile into SMEM (once) -- removes the redundant
  // global slot reads the value/key loops otherwise perform.
  for (int t = tid; t < tile_tokens; t += nthreads)
    s_slot[t] = slot_mapping[tok0 + t];

  if (fast) {
#if !RC_PROBE_NO_VALUE
    // VALUE: coalesced uint4 load into token-major SMEM.
    for (int i = tid; i < tile_tokens * vecs_per_tok; i += nthreads) {
      const int t = i / vecs_per_tok;
      const int dv = i - t * vecs_per_tok;
      const int token = tok0 + t;
      const scalar_t* src =
          value + (int64_t)token * value_stride + head_idx * head_size + dv * VEC;
      *reinterpret_cast<uint4*>(&smem[t * SP + dv * VEC]) =
          *reinterpret_cast<const uint4*>(src);
    }
#endif  // !RC_PROBE_NO_VALUE
  }
  __syncthreads();

#if !RC_PROBE_NO_VALUE
  if (fast) {
    // Precompute per-chunk contiguity + block coords once (from SMEM slots).
    // A chunk is VST tokens; slot0 % VST == 0 guarantees the uint4 store lands
    // on a 16-byte boundary and (block_size % VST == 0) the run stays in-block.
    for (int c = tid; c < tok_chunks; c += nthreads) {
      const int t0 = c * VST;
      const int cnt = min(VST, tile_tokens - t0);
      const int64_t slot0 = s_slot[t0];
      bool contig = (cnt == VST) && (slot0 >= 0) && ((slot0 % VST) == 0);
      if (contig) {
#pragma unroll
        for (int k = 1; k < VST; k++)
          if (s_slot[t0 + k] != slot0 + k) { contig = false; }
      }
      s_ctg[c] = contig ? 1 : 0;
      s_blk[c] = contig ? (slot0 / block_size) : 0;
      s_off[c] = contig ? (slot0 % block_size) : 0;
    }
    __syncthreads();

    // VALUE: coalesced store (VST tokens per uint4 at fixed d).
    for (int i = tid; i < head_size * tok_chunks; i += nthreads) {
      const int d = i / tok_chunks;
      const int c = i - d * tok_chunks;
      const int t0 = c * VST;
      if (s_ctg[c]) {
        cache_t* dst =
            value_cache +
            s_blk[c] * (int64_t)num_heads * head_size * block_size +
            head_idx * (int64_t)head_size * block_size +
            d * (int64_t)block_size + s_off[c];
        if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
          // fp8: pack VST(=16) consecutive tokens at fixed d into one uint4 with
          // 4 hardware fp8x4 conversions (4 floats -> uint32 each), replacing 16
          // software c10 narrows -- the fp8 store was compute-bound, not
          // bandwidth-bound. inv_scale folded into the pre-scale multiply.
          const float is = v_op.inv_scale;
          uint4 out;
          out.x = rc_pack4_fp8_e4m3((float)smem[(t0 + 0) * SP + d] * is,
                                    (float)smem[(t0 + 1) * SP + d] * is,
                                    (float)smem[(t0 + 2) * SP + d] * is,
                                    (float)smem[(t0 + 3) * SP + d] * is);
          out.y = rc_pack4_fp8_e4m3((float)smem[(t0 + 4) * SP + d] * is,
                                    (float)smem[(t0 + 5) * SP + d] * is,
                                    (float)smem[(t0 + 6) * SP + d] * is,
                                    (float)smem[(t0 + 7) * SP + d] * is);
          out.z = rc_pack4_fp8_e4m3((float)smem[(t0 + 8) * SP + d] * is,
                                    (float)smem[(t0 + 9) * SP + d] * is,
                                    (float)smem[(t0 + 10) * SP + d] * is,
                                    (float)smem[(t0 + 11) * SP + d] * is);
          out.w = rc_pack4_fp8_e4m3((float)smem[(t0 + 12) * SP + d] * is,
                                    (float)smem[(t0 + 13) * SP + d] * is,
                                    (float)smem[(t0 + 14) * SP + d] * is,
                                    (float)smem[(t0 + 15) * SP + d] * is);
          *reinterpret_cast<uint4*>(dst) = out;
        } else {
          // Quantize (or copy, for bf16) VST consecutive tokens at fixed d into a
          // packed cache_t run, then emit ONE 16-byte uint4 store along slot.
          cache_t tmp[VST];
#pragma unroll
          for (int k = 0; k < VST; k++) v_op(tmp[k], smem[(t0 + k) * SP + d]);
          // VST*sizeof(cache_t) == 16 for both bf16 (8*2B) and int8 (16*1B).
          *reinterpret_cast<uint4*>(dst) =
              *reinterpret_cast<const uint4*>(tmp);
        }
      } else {
        const int cnt = min(VST, tile_tokens - t0);
        for (int k = 0; k < cnt; k++) {
          const int64_t slot = s_slot[t0 + k];
          if (slot < 0) continue;
          const int64_t block_idx = slot / block_size;
          const int64_t block_offset = slot % block_size;
          cache_t* dst =
              value_cache +
              block_idx * (int64_t)num_heads * head_size * block_size +
              head_idx * (int64_t)head_size * block_size +
              d * (int64_t)block_size + block_offset;
          v_op(*dst, smem[(t0 + k) * SP + d]);
        }
      }
    }
  } else {
    // Generic fallback: scalar strided store.
    for (int i = tid; i < tile_tokens * head_size; i += nthreads) {
      const int t = i / head_size;
      const int d = i - t * head_size;
      const int64_t slot = s_slot[t];
      if (slot < 0) continue;
      const int64_t block_idx = slot / block_size;
      const int64_t block_offset = slot % block_size;
      cache_t* dst = value_cache +
                     block_idx * (int64_t)num_heads * head_size * block_size +
                     head_idx * (int64_t)head_size * block_size +
                     d * (int64_t)block_size + block_offset;
      v_op(*dst, value[(int64_t)(tok0 + t) * value_stride + head_idx * head_size + d]);
    }
  }
#endif  // !RC_PROBE_NO_VALUE

  // KEY: direct vectorized copy (no transpose needed).
#if !RC_PROBE_NO_KEY
  for (int i = tid; i < tile_tokens * h_block_count; i += nthreads) {
    const int t = i / h_block_count;
    const int hb = i - t * h_block_count;
    const int64_t slot = s_slot[t];
    if (slot < 0) continue;
    const int64_t block_idx = slot / block_size;
    const int64_t block_offset = slot % block_size;
    const scalar_t* ksrc =
        key + (int64_t)(tok0 + t) * key_stride + head_idx * head_size + hb * x;
    cache_t* kdst =
        key_cache +
        block_idx * (int64_t)num_heads * h_block_count * block_size * x +
        head_idx * (int64_t)h_block_count * block_size * x +
        hb * (int64_t)block_size * x + block_offset * x;
    if (fast) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        // bf16: unchanged single uint4 load + uint4 store.
        *reinterpret_cast<uint4*>(kdst) =
            *reinterpret_cast<const uint4*>(ksrc);
      } else {
        // int8/fp8: quantize the x-block, then one coalesced 8B store.
        if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
          // fp8: VEC(=8) e4m3 bytes = 2 hardware fp8x4 packs into a uint2.
          const float is = k_op.inv_scale;
          uint2 out;
          out.x = rc_pack4_fp8_e4m3((float)ksrc[0] * is, (float)ksrc[1] * is,
                                    (float)ksrc[2] * is, (float)ksrc[3] * is);
          out.y = rc_pack4_fp8_e4m3((float)ksrc[4] * is, (float)ksrc[5] * is,
                                    (float)ksrc[6] * is, (float)ksrc[7] * is);
          *reinterpret_cast<uint2*>(kdst) = out;
        } else {
          cache_t ktmp[VEC];  // x == VEC on the fast path
#pragma unroll
          for (int j = 0; j < VEC; j++) k_op(ktmp[j], ksrc[j]);
          *reinterpret_cast<uint2*>(kdst) =
              *reinterpret_cast<const uint2*>(ktmp);
        }
      }
    } else {
      for (int j = 0; j < x; j++) k_op(kdst[j], ksrc[j]);
    }
  }
#endif  // !RC_PROBE_NO_KEY
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // NHD or HND, shape see comments below
    cache_t* __restrict__ value_cache,   // same above
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n_elems = num_heads * head_size;

  // pointers to the beginning of the source row for this token.
  const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

  // find the start position inside the kv-cache for this token.
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  // this is true for the NHD layout where `head_stride == head_size`
  const bool is_contiguous_heads = (head_stride == head_size);

  float k_scale_val =
      (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : __builtin_mxc_rcpf(*k_scale);
  float v_scale_val =
      (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : __builtin_mxc_rcpf(*v_scale);
  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};
  if (is_contiguous_heads) {
    // NHD layout
    // kv cache: [num_blocks, block_size, num_heads, head_size]
    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x,
                                       blockDim.x, k_op);

    vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems,
                                       threadIdx.x, blockDim.x, v_op);

  } else {
    // HND layout: heads are strided, but each head_size segment is contiguous
    // kv cache: [num_blocks, num_heads, block_size, head_size]
    const int lane = threadIdx.x & 31;     // 0..31 within warp
    const int warp_id = threadIdx.x >> 5;  // warp index within block
    const int warps_per_block = blockDim.x >> 5;

    for (int head = warp_id; head < num_heads; head += warps_per_block) {
      const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
      const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

      cache_t* __restrict__ k_dst_h =
          key_dst + static_cast<int64_t>(head) * head_stride;
      cache_t* __restrict__ v_dst_h =
          value_dst + static_cast<int64_t>(head) * head_stride;

      // within each head, let the 32 threads of the warp perform the vector
      // copy
      vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32,
                                         k_op);

      vectorize_with_alignment<VEC_SIZE>(v_src_h, v_dst_h, head_size, lane, 32,
                                         v_op);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst[dst_idx] = src[src_idx];
      } else {
        dst[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
      }
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

// Grouped variant of concat_and_cache_mla: inserts the context K/V for every
// draft layer in a single launch. Grid is (num_tokens, num_layers); each layer
// reads its own cache base pointer from kv_cache_ptrs (same pointer-array
// pattern as copy_blocks_kernel). bf16 only, so it is a raw 16-bit copy with no
// scaling or quantization; scalar_t is uint16_t for portability.
template <typename scalar_t>
__global__ void concat_and_cache_mla_grouped_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_layers, num_tokens,
                                        // kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_layers, num_tokens, pe_dim]
    const int64_t* __restrict__ kv_cache_ptrs,  // [num_layers]
    const int64_t* __restrict__ slot_mapping,   // [num_layers, num_tokens]
    const int64_t kv_c_layer_stride, const int64_t kv_c_token_stride,
    const int64_t k_pe_layer_stride, const int64_t k_pe_token_stride,
    const int64_t slot_layer_stride, const int64_t block_stride,
    const int64_t entry_stride, const int kv_lora_rank, const int pe_dim,
    const int block_size) {
  const int64_t token_idx = blockIdx.x;
  const int64_t layer_idx = blockIdx.y;
  const int64_t slot_idx =
      slot_mapping[layer_idx * slot_layer_stride + token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  scalar_t* __restrict__ kv_cache =
      reinterpret_cast<scalar_t*>(kv_cache_ptrs[layer_idx]);
  const scalar_t* __restrict__ kv_c_layer =
      kv_c + layer_idx * kv_c_layer_stride;
  const scalar_t* __restrict__ k_pe_layer =
      k_pe + layer_idx * k_pe_layer_stride;

  auto copy = [&](const scalar_t* __restrict__ src, int64_t src_token_stride,
                  int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_token_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      kv_cache[dst_idx] = src[src_idx];
    }
  };

  copy(kv_c_layer, kv_c_token_stride, kv_lora_rank, 0);
  copy(k_pe_layer, k_pe_token_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // For the NoPE part, each tile of 128 elements is handled by half of one warp
  // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
  // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
  // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
  // So in total, we use 3 warps (96 threads) per block.

  // Cast kv_cache to 16_bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
    // Vectorized store of two 16-bit values, performed as one 32-bit store
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
  // Vectorized load of eight 16-bit values, performed as an int4 load
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                              fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                        fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                              fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
  }

  // Compute the scale for the tile
 float tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,                        // dimension of each head
    const int quant_block_size,                // quantization block size
    const int cache_block_size,                // cache block size
    const int cache_block_stride,  // stride for each block in kv_cache

    const bool use_ue8m0  // use ue8m0 scale format
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                threadIdx.y * blockDim.x + threadIdx.x) *
                               VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0 || (head_dim_idx >= head_dim)) {
    return;
  }

  float2 k_val = (reinterpret_cast<const float2*>(
      k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);
  float amax = 0.0f;
  for (int i = 0; i < VEC_SIZE; i++) {
    amax = fmaxf(amax, fabsf(float(k_val_ptr[i])));
  }

  // Reduced amax
  for (int mask = 16; mask > 0; mask /= 2) {
#ifdef USE_ROCM
    amax = fmaxf(amax, __shfl_xor_sync(uint64_t(-1), amax, mask));
#else
    amax = fmaxf(amax, __shfl_xor_sync(unsigned(-1), amax, mask));
#endif
  }

  float scale = fmaxf(amax, 1e-4) / kFp8ScaleDivisor;

  if (use_ue8m0) {
    scale = exp2f(ceilf(log2f(scale)));
  }

  const int64_t dst_offset = block_idx * cache_block_stride +
                             block_offset * head_dim + head_dim_idx;
  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(k_val_ptr[i], scale);
  }
  if (threadIdx.x == 0) {
    const int64_t dst_scale_idx =
        block_idx * cache_block_stride +
        cache_block_size * head_dim +
        (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
  }
}

template <typename scalar_t, typename cache_t>
__global__ void indexer_k_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,                        // dimension of each head
    const int cache_block_size,                // cache block size
    const int cache_stride,  // stride for each token in kv_cache
    const int num_blocks
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                threadIdx.y * blockDim.x + threadIdx.x) *
                               VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  const int64_t max_slots = static_cast<int64_t>(num_blocks) * cache_block_size;
  //const int64_t max_slots = 1690 * cache_block_size;
  if (slot_idx < 0 || slot_idx >= max_slots || (head_dim_idx >= head_dim)) {
    return;
  }

  float2 k_val = (reinterpret_cast<const float2*>(
      k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);

  const int64_t dst_offset = block_idx * cache_block_size * cache_stride +
                             block_offset * cache_stride + head_dim_idx;
  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] = k_val_ptr[i];
  }
}

template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    char* __restrict__ dst_scale,  // [num_tokens, head_dim / quant_block_size *
                                   // 4]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,                 // batch size
    const int64_t token_stride,           // stride for each token in dst_k
    const int64_t head_dim,               // dimension of each head
    const int64_t block_stride,           // stride for each block in kv_cache
    const int64_t cache_token_stride,     // stride for each token in kv_cache
    const int64_t cache_block_size,  // num_tokens for each block in kv_cache
    const int num_blocks,            // number of blocks
    const int num_tokens,            // number of tokens
    const int quant_block_size       // quantization block size
) {
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  if (threadIdx.x == 0) {
    batch_idx[threadIdx.y] = -1;
  }
  __syncthreads();

  for (int iter = 0; iter < cuda_utils::ceil_div(batch_size, int(blockDim.x));
       iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }


  __syncthreads();

  // num_tokens may be an allocation upper bound when Python avoids a D2H sync.
  // Only tokens covered by the exact device-side cu_seq_lens are valid to
  // gather.
  const int batch = batch_idx[threadIdx.y];
  if (head_idx >= head_dim || token_idx >= num_tokens || batch < 0) {
    return;
  }
  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch];
  const int block_idx =
      block_table[batch * num_blocks + inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
  ;
  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
        reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
  }
}

template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_cache_kernel(
    const char* __restrict__ kv_cache,    // [num_blocks, block_size,
                                          // cache_stride]
    char* __restrict__ dst_k,             // [num_tokens, head_dim]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,                 // batch size
    const int64_t token_stride,           // stride for each token in dst_k
    const int64_t head_dim,               // dimension of each head
    const int64_t block_stride,           // stride for each block in kv_cache
    const int64_t cache_token_stride,     // stride for each token in kv_cache
    const int64_t cache_block_size,  // num_tokens for each block in kv_cache
    const int num_blocks,            // number of blocks
    const int num_tokens             // number of tokens
) {
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  if (threadIdx.x == 0) {
    batch_idx[threadIdx.y] = -1;
  }
  __syncthreads();

  for (int iter = 0; iter < cuda_utils::ceil_div(batch_size, int(blockDim.x));
       iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }

  // Block-wide barrier with a shared-memory fence. Correct on both 32- and
  // 64-lane physical warps; __syncwarp()'s default 32-bit mask does not cover
  // lanes 32-63 of a MetaX 64-lane warp, which left batch_idx reads racing and
  // caused wrong-token gathers for large blocks (BLOCK_Y_SIZE==32, i.e.
  // num_tokens>=512 as seen with >2k GLM inputs).
  __syncthreads();

  const int batch = batch_idx[threadIdx.y];
  if (head_idx >= head_dim || token_idx >= num_tokens || batch < 0) {
    return;
  }
  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch];
  const int block_idx =
      block_table[batch * num_blocks + inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE]; 

}
}  // namespace vllm

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, smem_bytes, stream>>>(                          \
          reinterpret_cast<KV_T*>(key.data_ptr()),                    \
          reinterpret_cast<KV_T*>(value.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),           \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),         \
          slot_mapping.data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x,                        \
          reinterpret_cast<const float*>(k_scale.data_ptr()),         \
          reinterpret_cast<const float*>(v_scale.data_ptr()),         \
          total_work, h_block_count, num_tokens);

void reshape_and_cache(
    torch::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int h_block_count = head_size / x;

  int64_t total_work = (int64_t)num_tokens * num_heads * h_block_count;

  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                         at::cuda::getCurrentCUDAStream().device_index());

  const int block_size_threads = 512;
  dim3 block(block_size_threads);
  // Grid: one block per (head, token-tile). SMEM-transpose path needs the tile
  // of tokens resident so both value read and write are 16B-coalesced.
  const int TILE_T = RC_TILE_T;
  int num_token_tiles = (num_tokens + TILE_T - 1) / TILE_T;
  dim3 grid(num_heads, num_token_tiles);
  // Padded token-major value tile: TILE_T * (head_size + 8) bf16 elements.
  size_t smem_bytes = (size_t)TILE_T * (head_size + 8) * key.element_size()
                    + (size_t)TILE_T * sizeof(int64_t)          /* s_slot */
                    + (size_t)(TILE_T/8) * (2*sizeof(int64_t)+sizeof(int)) /* meta */
                    + 64;                                       /* slack */
  (void)sm_count;
  (void)total_work;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // int8 is not part of the shared DISPATCH_BY_KV_CACHE_DTYPE macro (which only
  // routes auto/fp8_e4m3/fp8_e5m2 and is reused by other kernels). Intercept it
  // here and instantiate the bf16->int8_t path directly, so we neither touch
  // the shared macro nor force an int8 instantiation of unrelated kernels.
  if (kv_cache_dtype == "int8") {
    TORCH_CHECK(key.dtype() == at::ScalarType::BFloat16,
                "int8 kv cache requires bf16 key/value, got ", key.dtype());
    CALL_RESHAPE_AND_CACHE(__nv_bfloat16, int8_t,
                           vllm::Fp8KVCacheDataType::kInt8);
    return;
  }

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)             \
  vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>           \
      <<<grid, block, 0, stream>>>(                                       \
          reinterpret_cast<KV_T*>(key.data_ptr()),                        \
          reinterpret_cast<KV_T*>(value.data_ptr()),                      \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),               \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),             \
          slot_mapping.data_ptr<int64_t>(), block_stride, page_stride,    \
          head_stride, key_stride, value_stride, num_heads, head_size,    \
          block_size, reinterpret_cast<const float*>(k_scale.data_ptr()), \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,      // [num_tokens, num_heads, head_size]
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  
  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "nvfp4") {
#if defined(ENABLE_NVFP4_SM100) || defined(ENABLE_NVFP4_SM120)
    // NVFP4 dispatch is compiled separately for SM100+.
    extern void reshape_and_cache_nvfp4_dispatch(
        torch::Tensor & key, torch::Tensor & value, torch::Tensor & key_cache,
        torch::Tensor & value_cache, torch::Tensor & slot_mapping,
        torch::Tensor & k_scale, torch::Tensor & v_scale);
    reshape_and_cache_nvfp4_dispatch(key, value, key_cache, value_cache,
                                     slot_mapping, k_scale, v_scale);
    return;
#else
    TORCH_CHECK(false,
                "NVFP4 KV cache requires SM100+ (Blackwell). "
                "Please rebuild vllm with a Blackwell-compatible CUDA target.");
#endif
  }

  // Original FP8/auto path.
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

  DISPATCH_BY_KV_CACHE_DTYPE(key.dtype(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)              \
  vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>            \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)           \
  vllm::concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>         \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                     \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                     \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,   \
          reinterpret_cast<const float*>(scale.data_ptr()));

void concat_and_cache_mla(
    torch::Tensor& kv_c,          // [num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,          // [num_tokens, pe_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, (kv_lora_rank +
                                  // pe_dim)]
    torch::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  if (kv_cache_dtype == "fp8_ds_mla") {
    TORCH_CHECK(kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla");
    TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
    TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.itemsize(),
                "kv_cache.size(2) must be 656 bytes for fp8_ds_mla");
    TORCH_CHECK(kv_c.itemsize() == 2,
                "kv_c.itemsize() must be 2 for fp8_ds_mla");
    TORCH_CHECK(k_pe.itemsize() == 2,
                "k_pe.itemsize() must be 2 for fp8_ds_mla");
  } else {
    TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
  }

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_c));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "fp8_ds_mla") {
    dim3 grid(num_tokens);
    // For the NoPE part, each tile of 128 elements is handled by half of one
    // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32
    // threads). So in total, we use 3 warps (96 threads) per block.
    dim3 block(96);
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_DS_MLA);
  } else {
    dim3 grid(num_tokens);
    dim3 block(std::min(kv_lora_rank, 512));
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.dtype(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_MLA);
  }
}

void concat_and_cache_mla_grouped(
    torch::Tensor& kv_c,  // [num_layers, num_tokens, kv_lora_rank]
    torch::Tensor& k_pe,  // [num_layers, num_tokens, pe_dim]
    torch::Tensor& kv_cache_ptrs,  // [num_layers] int64, on device
    torch::Tensor& slot_mapping,   // [num_layers, num_tokens] int64
    int64_t block_size, int64_t block_stride, int64_t entry_stride) {
  int num_layers = kv_c.size(0);
  int num_tokens = kv_c.size(1);
  int kv_lora_rank = kv_c.size(2);
  int pe_dim = k_pe.size(2);

  TORCH_CHECK(
      kv_c.scalar_type() == at::ScalarType::BFloat16 &&
          k_pe.scalar_type() == at::ScalarType::BFloat16,
      "concat_and_cache_mla_grouped only supports a bf16 KV cache; got kv_c=",
      kv_c.scalar_type(), ", k_pe=", k_pe.scalar_type());
  TORCH_CHECK(
      kv_cache_ptrs.scalar_type() == at::ScalarType::Long,
      "kv_cache_ptrs must be int64");

  if (num_tokens == 0 || num_layers == 0) {
    return;
  }

  const int64_t kv_c_layer_stride = kv_c.stride(0);
  const int64_t kv_c_token_stride = kv_c.stride(1);
  const int64_t k_pe_layer_stride = k_pe.stride(0);
  const int64_t k_pe_token_stride = k_pe.stride(1);
  const int64_t slot_layer_stride = slot_mapping.stride(0);

  const at::cuda::CUDAGuard device_guard(kv_c.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(num_tokens, num_layers);
  dim3 block(std::min(kv_lora_rank, 512));
  vllm::concat_and_cache_mla_grouped_kernel<uint16_t>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(kv_c.data_ptr()),
          reinterpret_cast<const uint16_t*>(k_pe.data_ptr()),
          kv_cache_ptrs.const_data_ptr<int64_t>(),
          slot_mapping.const_data_ptr<int64_t>(), kv_c_layer_stride,
          kv_c_token_stride, k_pe_layer_stride, k_pe_token_stride,
          slot_layer_stride, block_stride, entry_stride, kv_lora_rank, pe_dim,
          block_size);
}

namespace vllm {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__global__ void convert_fp8_kernel(const Tin* __restrict__ src_cache,
                                   Tout* __restrict__ dst_cache,
                                   const float scale,
                                   const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] =
        fp8::scaled_convert<Tout, Tin, kv_dt>(src_cache[idx], scale);
  }
}

}  // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin, KV_DTYPE)                                \
  vllm::convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<Tin*>(src_cache.data_ptr()),                          \
      reinterpret_cast<Tout*>(dst_cache.data_ptr()), scale, block_stride);

// Only for testing.
void convert_fp8(torch::Tensor& dst_cache, torch::Tensor& src_cache,
                 const double scale, const std::string& kv_cache_dtype) {
  torch::Device src_device = src_cache.device();
  torch::Device dst_device = dst_cache.device();
  TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  TORCH_CHECK(src_device.index() == dst_device.index(),
              "src and dst must be on the same GPU");
  at::cuda::OptionalCUDAGuard device_guard(src_device);

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (kv_cache_dtype == "auto") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (src_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.dtype() == at::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else {
    TORCH_CHECK(false, "Unsupported data type: ", kv_cache_dtype);
  }
}

namespace vllm {

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int ENTRY_SIZE, int CTA_SIZE>
__global__ void gather_and_maybe_dequant_cache(
    const cache_t* __restrict__ src_cache,     // [NUM_BLOCKS, BLOCK_SIZE,
                                               // ENTRIES...]
    scalar_t* __restrict__ dst,                // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,   // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,   // [BATCH+1]
    const int32_t* __restrict__ token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNK]
    const int32_t num_tokens, const int32_t block_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const float* __restrict__ scale,
    const int32_t* __restrict__ seq_starts) {  // Optional: starting offsets per
                                               // batch
  constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
  using ltype = vllm::vec_n_t<cache_t, vec_size>;
  using stype = vllm::vec_n_t<scalar_t, vec_size>;
  // We are adding this for code readability which will be optimized out when
  // build in release.
  assert(CTA_SIZE == blockDim.x);

#pragma unroll
  for (int token_id = blockIdx.x; token_id < num_tokens;
       token_id += gridDim.x) {
    int64_t batch_id = token_to_seq[token_id];
    int64_t batch_start = cu_seq_lens[batch_id];
    int64_t batch_end = cu_seq_lens[batch_id + 1];
    int32_t batch_offset = token_id - batch_start;

    if (token_id >= batch_end) return;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[batch_id];
    }
    batch_offset += offset;
    int32_t block_table_id = batch_offset / block_size;
    int32_t slot_id = batch_offset % block_size;
    int32_t block_table_offset = batch_id * block_table_stride + block_table_id;
    int32_t block_id = block_table[block_table_offset];
    int64_t cache_offset =
        block_id * cache_block_stride + slot_id * cache_entry_stride;
    constexpr int32_t vec_iter_cnt = ENTRY_SIZE / vec_size;
    scalar_t* dst_ = dst + token_id * dst_entry_stride;
    cache_t* src_ = const_cast<cache_t*>(src_cache) + cache_offset;

#pragma unroll
    for (int idx = threadIdx.x; idx < vec_iter_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        reinterpret_cast<stype*>(dst_)[idx] =
            static_cast<stype>(reinterpret_cast<ltype*>(src_)[idx]);
      } else {
        ltype loaded_val = reinterpret_cast<ltype*>(src_)[idx];
        stype store_val;
#pragma unroll
        for (int j = 0; j < vec_size; ++j) {
          store_val.val[j] = fp8::scaled_convert<scalar_t, cache_t, kv_dt>(
              loaded_val.val[j], *scale);
        }
        reinterpret_cast<stype*>(dst_)[idx] = store_val;
      }
    }
    // process tail
    constexpr int32_t tail_cnt = ENTRY_SIZE % vec_size;
    dst_ = dst_ + ENTRY_SIZE - tail_cnt;
    src_ = src_ + ENTRY_SIZE - tail_cnt;
#pragma unroll
    for (int idx = threadIdx.x; idx < tail_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst_[idx] = static_cast<scalar_t>(src_[idx]);
      } else {
        dst_[idx] =
            fp8::scaled_convert<scalar_t, cache_t, kv_dt>(src_[idx], *scale);
      }
    }
  }
}

}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
// SCALAR_T is the data type of the destination tensor.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, ENTRY_SZ)              \
  vllm::gather_and_maybe_dequant_cache<SCALAR_T, CACHE_T, KV_DTYPE, ENTRY_SZ, \
                                       thread_block_size>                     \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<CACHE_T*>(src_cache.data_ptr()),                   \
          reinterpret_cast<SCALAR_T*>(dst.data_ptr()),                        \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(),   \
          token_to_seq.data_ptr<int32_t>(), num_tokens, block_size,           \
          block_table_stride, cache_block_stride, cache_entry_stride,         \
          dst_entry_stride, reinterpret_cast<const float*>(scale.data_ptr()), \
          seq_starts_ptr);

#define CALL_GATHER_CACHE_576(SCALAR_T, CACHE_T, KV_DTYPE) \
  CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, 576)

#define CALL_GATHER_CACHE_320(SCALAR_T, CACHE_T, KV_DTYPE) \
  CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, 320)

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - token_to_seq contains the back mapping from token_id to batch_id
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_and_maybe_dequant_cache(
    torch::Tensor const& src_cache,     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,           // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::Tensor const& scale,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(-1);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }
  TORCH_CHECK(
      head_dim == 320 || head_dim == 576,
      "gather_and_maybe_dequant_cache only support the head_dim to 320 or 576 "
      "for better performance")

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  constexpr int32_t thread_block_size = 64;
  dim3 grid(num_tokens);
  dim3 block(thread_block_size);

  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (head_dim == 576) {
    DISPATCH_BY_KV_CACHE_DTYPE(dst.dtype(), kv_cache_dtype,
                               CALL_GATHER_CACHE_576);
  } else {
    DISPATCH_BY_KV_CACHE_DTYPE(dst.dtype(), kv_cache_dtype,
                               CALL_GATHER_CACHE_320);
  }
}

namespace vllm {

// Gather and upconvert FP8 KV cache tokens to BF16 workspace
// Similar to cp_gather_cache but specifically for FP8->BF16 conversion
__global__ void cp_gather_and_upconvert_fp8_kv_cache(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    __nv_bfloat16* __restrict__ dst,          // [total_tokens, 576]
    const int32_t* __restrict__ block_table,  // [num_reqs, BLOCK_INDICES]
    const int32_t* __restrict__ workspace_starts,  // [num_reqs]
    const int32_t num_reqs, const int32_t block_size,
    const int32_t total_tokens, const int64_t block_table_stride,
    const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts) {  // Optional source offsets
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (flat_warp_id >= total_tokens) return;
  const int lane_id = threadIdx.x & 31;

  // Binary search to find which request owns this output token
  int lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (workspace_starts[mid] <= flat_warp_id)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int req_id = lo;

  // Compute physical token address via block table
  const int out_token_id = flat_warp_id;
  int token_offset = out_token_id - workspace_starts[req_id];
  if (seq_starts != nullptr) token_offset += seq_starts[req_id];
  const int cache_block_idx = token_offset / block_size;
  const int offset_in_block = token_offset % block_size;
  const int physical_block =
      block_table[req_id * block_table_stride + cache_block_idx];

  const uint8_t* token_ptr = src_cache + physical_block * cache_block_stride +
                             offset_in_block * cache_entry_stride;

  const int4* nope_src = reinterpret_cast<const int4*>(token_ptr);
  const int4 fp8_data = nope_src[lane_id];

  const float* scales_ptr = reinterpret_cast<const float*>(token_ptr + 512);
  const float scale = scales_ptr[lane_id >> 3];

  const uint2 fp8_lo = make_uint2(fp8_data.x, fp8_data.y);
  const uint2 fp8_hi = make_uint2(fp8_data.z, fp8_data.w);
  const bf16_8_t bf16_lo = fp8::scaled_vec_conversion<bf16_8_t, uint2>(fp8_lo, scale);
  const bf16_8_t bf16_hi = fp8::scaled_vec_conversion<bf16_8_t, uint2>(fp8_hi, scale);

  __nv_bfloat16* dst_ptr = dst + out_token_id * dst_entry_stride;
  int4* nope_dst = reinterpret_cast<int4*>(dst_ptr) + lane_id * 2;
  nope_dst[0] = *reinterpret_cast<const int4*>(&bf16_lo);
  nope_dst[1] = *reinterpret_cast<const int4*>(&bf16_hi);

  const int* rope_src = reinterpret_cast<const int*>(token_ptr + 528);
  int* rope_dst = reinterpret_cast<int*>(dst_ptr + 512);
  rope_dst[lane_id] = rope_src[lane_id];
}

template <typename scalar_t>
// Note(hc): The cp_gather_cache allows seq_starts to no longer be divisible by
// block_size.
__global__ void cp_gather_cache(
    const scalar_t* __restrict__ src_cache,   // [NUM_BLOCKS, BLOCK_SIZE,
                                              // ENTRY_SIZE]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, ENTRY_SIZE]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts  // Optional: starting offsets per
                                            // batch
) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch.
  // If seq_starts is provided, compute an offset based on it
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  if (seq_starts != nullptr) {
    offset += seq_starts[bid];
  }
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths.
  dst += seq_start * dst_entry_stride;

  auto copy_entry = [&](const scalar_t* __restrict__ _src,
                        scalar_t* __restrict__ _dst) {
    for (int i = threadIdx.x; i < entry_size; i += blockDim.x)
      _dst[i] = _src[i];
  };

  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + pid * dst_entry_stride;
    copy_entry(block_start_ptr + offset * cache_entry_stride, block_dst_ptr);
    offset += 1;
    // bump to next block
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}
}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
#define CALL_CP_GATHER_CACHE(CPY_DTYPE)                                 \
  vllm::cp_gather_cache<CPY_DTYPE><<<grid, block, 0, stream>>>(         \
      reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()),               \
      reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),                     \
      block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
      block_size, entry_size, block_table_stride, cache_block_stride,   \
      cache_entry_stride, dst_entry_stride, seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting slot index by
//  seq_starts[bid]
void cp_gather_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::Tensor const& dst,          // [TOT_TOKENS, ENTRIES...]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = src_cache.flatten(2, -1).size(2);

  TORCH_CHECK(block_table.dtype() == torch::kInt32,
              "block_table must be int32");
  TORCH_CHECK(cu_seq_lens.dtype() == torch::kInt32,
              "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(seq_starts.value().dtype() == torch::kInt32,
                "seq_starts must be int32");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
              "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
              "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
              "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(1024);

  TORCH_CHECK(src_cache.dtype() == dst.dtype(),
              "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().data_ptr<int32_t>() : nullptr;

  if (dtype_bits == 32) {
    CALL_CP_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_CP_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_CP_GATHER_CACHE(uint8_t);
  } else {
    TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}

void cp_gather_and_upconvert_fp8_kv_cache(
    torch::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::Tensor const& dst,          // [TOT_TOKENS, 576]
    torch::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size,
    std::optional<torch::Tensor> seq_starts = std::nullopt) {
  at::cuda::OptionalCUDAGuard device_guard(src_cache.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(1);

  TORCH_CHECK(
      block_table.scalar_type() == torch::headeronly::ScalarType::Int,
      "block_table must be int32");
  TORCH_CHECK(
      workspace_starts.scalar_type() == torch::headeronly::ScalarType::Int,
      "workspace_starts must be int32");
  if (seq_starts.has_value()) {
    TORCH_CHECK(
        seq_starts.value().scalar_type() == torch::headeronly::ScalarType::Int,
        "seq_starts must be int32");
  }

  TORCH_CHECK(src_cache.device() == dst.device(),
                  "src_cache and dst must be on the same device");
  TORCH_CHECK(src_cache.device() == block_table.device(),
                  "src_cache and block_table must be on the same device");
  TORCH_CHECK(src_cache.device() == workspace_starts.device(),
                  "src_cache and workspace_starts must be on the same device");
  if (seq_starts.has_value()) {
    TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                    "src_cache and seq_starts must be on the same device");
  }
  auto dtype = src_cache.scalar_type();
  TORCH_CHECK(
      dtype == torch::headeronly::ScalarType::Byte ||               // uint8
          dtype == torch::headeronly::ScalarType::Float8_e4m3fn ||  // fp8 e4m3
          dtype == torch::headeronly::ScalarType::Float8_e5m2,      // fp8 e5m2
      "src_cache must be uint8, float8_e4m3fn, or float8_e5m2, but got ",
      src_cache.scalar_type());
  TORCH_CHECK(dst.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "dst must be bfloat16");
  TORCH_CHECK(head_dim == 576, "head_dim must be 576 for MLA");

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  const uint8_t* src_ptr = nullptr;
  if (dtype == torch::headeronly::ScalarType::Byte) {
    src_ptr = src_cache.const_data_ptr<uint8_t>();
  } else {
    // float8_e4m3fn or float8_e5m2
    src_ptr = reinterpret_cast<const uint8_t*>(src_cache.data_ptr());
  }

  const int total_tokens = dst.size(0);
  constexpr int warps_per_block = 8;
  const int grid_size = (total_tokens + warps_per_block - 1) / warps_per_block;
  const int block_size_threads = warps_per_block * 32;  // 256 threads
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().const_data_ptr<int32_t>()
                             : nullptr;

  vllm::cp_gather_and_upconvert_fp8_kv_cache<<<grid_size, block_size_threads, 0,
                                               stream>>>(
      src_ptr, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
      block_table.const_data_ptr<int32_t>(),
      workspace_starts.const_data_ptr<int32_t>(),
      static_cast<int32_t>(batch_size), block_size, total_tokens,
      block_table_stride, cache_block_stride, cache_entry_stride,
      dst_entry_stride, seq_starts_ptr);
}


//indexer_k_cache op for metax glm5

// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_CACHE(KV_T, CACHE_T, KV_DTYPE)                      \
  vllm::indexer_k_cache_kernel<KV_T, CACHE_T><<<grid, block, 0, stream>>>( \
      reinterpret_cast<KV_T*>(k.data_ptr()),                               \
      reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                     \
      slot_mapping.data_ptr<int64_t>(), head_dim, cache_block_size,        \
      cache_stride, num_blocks);

void indexer_k_cache(
    torch::Tensor& k,            // [num_tokens, head_dim]
    torch::Tensor& kv_cache,     // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping  // [num_tokens]
) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  int cache_block_size = kv_cache.size(1);
  int cache_stride = kv_cache.size(2);
  int num_blocks = kv_cache.size(0);
  TORCH_CHECK(k.dtype() == kv_cache.dtype(),
              "indexer_k_cache op no quant k and kv_cache must have the same dtype");

  TORCH_CHECK(k.device() == kv_cache.device(),
              "k and kv_cache must be on the same device");
  TORCH_CHECK(k.device() == slot_mapping.device(),
              "k and slot_mapping must be on the same device");

  constexpr int vec_size = 4;
  dim3 grid(num_tokens, (head_dim + vec_size - 1) / vec_size);
  dim3 block(32, vec_size);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  DISPATCH_BY_KV_CACHE_DTYPE(k.dtype(), "auto", CALL_INDEXER_K_CACHE);
}

// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)         \
  vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>       \
      <<<grid, block, 0, stream>>>(                                     \
          reinterpret_cast<KV_T*>(k.data_ptr()),                        \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),              \
          slot_mapping.data_ptr<int64_t>(), head_dim, quant_block_size, \
          cache_block_size, cache_block_stride, use_ue8m0);

void indexer_k_quant_and_cache(
    torch::Tensor& k,             // [num_tokens, head_dim]
    torch::Tensor& kv_cache,      // [num_blocks, block_size, cache_stride]
    torch::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,     // quantization block size
    const std::string& scale_fmt) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  int cache_block_size = kv_cache.size(1);
  //int cache_stride = kv_cache.size(2);
  int64_t cache_block_stride = kv_cache.stride(0);
  bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(k.device() == kv_cache.device(),
              "k and kv_cache must be on the same device");
  TORCH_CHECK(k.device() == slot_mapping.device(),
              "k and slot_mapping must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 4;
  dim3 grid(num_tokens, (head_dim + quant_block_size * vec_size - 1) /
                            (quant_block_size * vec_size));
  dim3 block(32, vec_size);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  static const std::string kv_cache_dtype = "fp8_e4m3";
  DISPATCH_BY_KV_CACHE_DTYPE(k.dtype(), kv_cache_dtype,
                             CALL_INDEXER_K_QUANT_AND_CACHE);
}


// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_CACHE(BLOCK_Y_SIZE)                        \
  vllm::cp_gather_indexer_k_cache_kernel<BLOCK_Y_SIZE>                      \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,               \
              (head_dim_bytes + 8 * vec_size - 1) / (8 * vec_size)),       \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                               \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                     \
          reinterpret_cast<char*>(dst_k.data_ptr()),                        \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
          batch_size, dst_k_token_stride_bytes, head_dim_bytes,             \
          kv_block_stride_bytes, kv_token_stride_bytes, kv_cache.size(1),   \
          block_table.size(1), num_tokens);

void cp_gather_indexer_k_cache(
    const torch::Tensor& kv_cache,     // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,              // [num_tokens, head_dim]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = block_table.size(0);
  int num_tokens = dst_k.size(0);
  int head_dim = dst_k.size(1);
  // Kernel offsets are computed in BYTES (char* pointers, float4 copies), but
  // PyTorch strides/sizes are ELEMENT counts. Convert element strides -> bytes
  // so the gather is correct for any dtype (bf16/fp16/fp32), not just 1-byte.
  const int64_t kv_elem_size = kv_cache.element_size();
  const int64_t dst_elem_size = dst_k.element_size();
  const int64_t dst_k_token_stride_bytes = dst_k.stride(0) * dst_elem_size;
  const int64_t head_dim_bytes = static_cast<int64_t>(head_dim) * dst_elem_size;
  const int64_t kv_block_stride_bytes = kv_cache.stride(0) * kv_elem_size;
  const int64_t kv_token_stride_bytes = kv_cache.stride(1) * kv_elem_size;
  // int quant_block_size = head_dim * 4 / dst_scale.size(1);

  TORCH_CHECK(kv_cache.device() == dst_k.device(),
              "kv_cache and dst_k must be on the same device");
  // TORCH_CHECK(kv_cache.device() == dst_scale.device(),
  //             "kv_cache and dst_scale must be on the same device");
  TORCH_CHECK(kv_cache.device() == block_table.device(),
              "kv_cache and block_table must be on the same device");
  TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
              "kv_cache and cu_seq_lens must be on the same device");
  // TORCH_CHECK(head_dim % quant_block_size == 0,
  //             "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_CACHE(32);
  }
}


// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                  \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,               \
              (head_dim_bytes + 8 * vec_size - 1) / (8 * vec_size)),       \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                               \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                     \
          reinterpret_cast<char*>(dst_k.data_ptr()),                        \
          reinterpret_cast<char*>(dst_scale.data_ptr()),                    \
          block_table.data_ptr<int32_t>(), cu_seq_lens.data_ptr<int32_t>(), \
          batch_size, dst_k_token_stride_bytes, head_dim_bytes,             \
          kv_block_stride_bytes, kv_token_stride_bytes, kv_cache.size(1),   \
          block_table.size(1), num_tokens, quant_block_size);

void cp_gather_indexer_k_quant_cache(
    const torch::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::Tensor& dst_k,           // [num_tokens, head_dim]
    torch::Tensor& dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = block_table.size(0);
  int num_tokens = dst_k.size(0);
  int head_dim = dst_k.size(1);
  int quant_block_size = head_dim * 4 / dst_scale.size(1);
  // Convert element strides -> byte offsets (kernel uses char* + float4). For
  // the quant path dst_k/kv_cache are int8/uint8 so element_size==1 and this is
  // a no-op, but keep it symmetric with the non-quant path and dtype-safe.
  const int64_t kv_elem_size = kv_cache.element_size();
  const int64_t dst_elem_size = dst_k.element_size();
  const int64_t dst_k_token_stride_bytes = dst_k.stride(0) * dst_elem_size;
  const int64_t head_dim_bytes = static_cast<int64_t>(head_dim) * dst_elem_size;
  const int64_t kv_block_stride_bytes = kv_cache.stride(0) * kv_elem_size;
  const int64_t kv_token_stride_bytes = kv_cache.stride(1) * kv_elem_size;

  TORCH_CHECK(kv_cache.device() == dst_k.device(),
              "kv_cache and dst_k must be on the same device");
  TORCH_CHECK(kv_cache.device() == dst_scale.device(),
              "kv_cache and dst_scale must be on the same device");
  TORCH_CHECK(kv_cache.device() == block_table.device(),
              "kv_cache and block_table must be on the same device");
  TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
              "kv_cache and cu_seq_lens must be on the same device");
  TORCH_CHECK(head_dim % quant_block_size == 0,
              "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(kv_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }
}

// Concatenate ql_nope and q_pe into a contiguous q_out tensor for MLA/DSA.
// Replaces torch.cat((ql_nope, q_pe), dim=-1).
// Supports non-contiguous input tensors by using stride-aware kernel.
void concat_mla_q(torch::Tensor& ql_nope,  // [num_tokens, num_heads, nope_dim]
                  torch::Tensor& q_pe,     // [num_tokens, num_heads, rope_dim]
                  torch::Tensor& q_out     // [num_tokens, num_heads, nope_dim +
                                           // rope_dim]
) {
  const int num_tokens = ql_nope.size(0);
  const int num_heads = ql_nope.size(1);
  const int nope_dim = ql_nope.size(2);
  const int rope_dim = q_pe.size(2);

  TORCH_CHECK(nope_dim % 512 == 0, "nope_dim must be a multiple of 512, got ",
              nope_dim);
  TORCH_CHECK(rope_dim == 64, "rope_dim must be 64, got ", rope_dim);
  TORCH_CHECK(q_out.size(2) == nope_dim + rope_dim);

  // Innermost dimension must have stride 1 for vectorized memory access
  TORCH_CHECK(ql_nope.stride(2) == 1, "ql_nope must have stride 1 in dim 2");
  TORCH_CHECK(q_pe.stride(2) == 1, "q_pe must have stride 1 in dim 2");
  TORCH_CHECK(q_out.stride(2) == 1, "q_out must have stride 1 in dim 2");

  if (num_tokens == 0) return;

  // Get strides for proper memory addressing (supports non-contiguous tensors)
  const int64_t nope_stride_0 = ql_nope.stride(0);
  const int64_t nope_stride_1 = ql_nope.stride(1);
  const int64_t pe_stride_0 = q_pe.stride(0);
  const int64_t pe_stride_1 = q_pe.stride(1);
  const int64_t out_stride_0 = q_out.stride(0);
  const int64_t out_stride_1 = q_out.stride(1);

  constexpr int warps_per_block = 8;
  const int total_warps = num_tokens * num_heads;
  const int grid_size = (total_warps + warps_per_block - 1) / warps_per_block;
  const int block_size = warps_per_block * 32;

  const at::cuda::OptionalCUDAGuard device_guard(device_of(ql_nope));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto data_type = ql_nope.scalar_type();
  switch (data_type) {
      case torch::kFloat16:
          // Handle Float16
          vllm::ConcatMLAQKernel<half, 512><<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<half*>(q_out.mutable_data_ptr()),
              reinterpret_cast<const half*>(ql_nope.data_ptr()),
              reinterpret_cast<const half*>(q_pe.data_ptr()),
              num_tokens, num_heads,
              out_stride_0, out_stride_1,
              nope_stride_0, nope_stride_1,
              pe_stride_0, pe_stride_1);
          break;

      case torch::kBFloat16:
          // Handle BFloat16
          vllm::ConcatMLAQKernel<__nv_bfloat16, 512><<<grid_size, block_size, 0, stream>>>(
              reinterpret_cast<__nv_bfloat16*>(q_out.mutable_data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(ql_nope.data_ptr()),
              reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr()),
              num_tokens, num_heads,
              out_stride_0, out_stride_1,
              nope_stride_0, nope_stride_1,
              pe_stride_0, pe_stride_1);
          break;

      default:
          // Handle other data types
          throw std::invalid_argument(
              "Invalid dtype, only supports float16 and bfloat16");
          break;
  }
}
