/*
 * Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
 *
 * Native CUDA sparse attention for GLM H64/D512/TopK2112. General decode
 * and prefill use statically compiled MMA schedules; short decode uses the
 * prefix-16 specialization. No FlashInfer or TileLang runtime is required.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <list>
#include <mutex>
#include <vector>

#include "sparse_attention_generated.cuh"

namespace {

constexpr int kHeads = 64;
constexpr int kDim = 512;
constexpr int kThreads = 256;
constexpr int kMaxSparse = 16;
constexpr int64_t kTopK = 2112;
constexpr int64_t kSplitKV = 11;
constexpr int64_t kDecodeMaxQueryTokens = 128;
constexpr int kDecodeStage1Smem = 39296;
constexpr int kDecodeWideSmem = 72064;
constexpr int kPrefillStage1Smem = 39424;
constexpr int kPrefillWideSmem = 78720;
constexpr float kLog2E = 1.4426950408889634f;
constexpr size_t kEligibilityCacheLimit = 2;


struct EligibilityEntry {
  const void* tensor_impl;
  const void* data_ptr;
  uint32_t version;
  int64_t kv_rows;
  at::Tensor indices_ref;
  bool eligible;
};

struct DeviceSchedule {
  int device;
  bool wide;
};

std::mutex g_state_mutex;
std::list<EligibilityEntry> g_eligibility_cache;
std::vector<DeviceSchedule> g_device_schedules;

__global__ void sparse_decode_prefix16_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv,
    const int32_t* __restrict__ indices,
    __nv_bfloat16* __restrict__ output,
    int sparse_width,
    float sm_scale) {
  const int token = blockIdx.x / kHeads;
  const int head = blockIdx.x % kHeads;
  const int tid = threadIdx.x;

  __shared__ float reduction[kThreads];
  __shared__ float weights[kMaxSparse];
  __shared__ int32_t rows[kMaxSparse];

  if (tid < kMaxSparse) {
    rows[tid] = indices[token * sparse_width + tid];
  }
  __syncthreads();

  const __nv_bfloat16* q_ptr = q + (token * kHeads + head) * kDim;
  float local_max = -INFINITY;
  for (int item = 0; item < kMaxSparse; ++item) {
    float dot = 0.0f;
    const int32_t row = rows[item];
    if (row >= 0) {
      const __nv_bfloat16* kv_ptr = kv + static_cast<int64_t>(row) * kDim;
      #pragma unroll
      for (int d = tid; d < kDim; d += kThreads) {
        dot += __bfloat162float(q_ptr[d]) * __bfloat162float(kv_ptr[d]);
      }
    }
    reduction[tid] = dot;
    __syncthreads();
    #pragma unroll
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) reduction[tid] += reduction[tid + stride];
      __syncthreads();
    }
    if (tid == 0) {
      const float score = row >= 0 ? reduction[0] * sm_scale : -INFINITY;
      weights[item] = score;
      local_max = fmaxf(local_max, score);
    }
    __syncthreads();
  }

  if (tid == 0) {
    float denominator = 0.0f;
    #pragma unroll
    for (int item = 0; item < kMaxSparse; ++item) {
      const float value = expf(weights[item] - local_max);
      weights[item] = value;
      denominator += value;
    }
    const float reciprocal = 1.0f / denominator;
    #pragma unroll
    for (int item = 0; item < kMaxSparse; ++item) weights[item] *= reciprocal;
  }
  __syncthreads();

  __nv_bfloat16* out_ptr = output + (token * kHeads + head) * kDim;
  #pragma unroll
  for (int d = tid; d < kDim; d += kThreads) {
    float value = 0.0f;
    #pragma unroll
    for (int item = 0; item < kMaxSparse; ++item) {
      const int32_t row = rows[item];
      if (row >= 0) {
        value += weights[item] * __bfloat162float(kv[static_cast<int64_t>(row) * kDim + d]);
      }
    }
    out_ptr[d] = __float2bfloat16_rn(value);
  }
}

}  // namespace

namespace {

void check_inputs(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    int64_t d_v) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda() && indices.is_cuda(),
              "q, kv, and indices must be CUDA tensors");
  TORCH_CHECK(q.get_device() == kv.get_device() &&
                  q.get_device() == indices.get_device(),
              "q, kv, and indices must be on the same device");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous() && indices.is_contiguous(),
              "q, kv, and indices must be contiguous");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 &&
                  kv.scalar_type() == at::kBFloat16,
              "q and kv must be bfloat16");
  TORCH_CHECK(indices.scalar_type() == at::kInt, "indices must be int32");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == kHeads && q.size(2) == kDim,
              "q must have shape [S, 64, 512]");
  TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == kDim,
              "kv must have shape [N, 1, 512]");
  TORCH_CHECK(indices.dim() == 3 && indices.size(0) == q.size(0) &&
                  indices.size(1) == 1,
              "indices must have shape [S, 1, K]");
  TORCH_CHECK(q.size(0) > 0 && kv.size(0) > 0 && indices.size(2) > 0,
              "q, kv, and indices dimensions must be nonzero");
  TORCH_CHECK(d_v == kDim, "d_v must be 512, got ", d_v);
}

bool same_eligibility_key(
    const EligibilityEntry& entry,
    const at::Tensor& indices,
    int64_t kv_rows) {
  return entry.tensor_impl == indices.unsafeGetTensorImpl() &&
      entry.data_ptr == indices.const_data_ptr() &&
      entry.version ==
          indices.unsafeGetTensorImpl()->version_counter().current_version() &&
      entry.kv_rows == kv_rows;
}

bool can_use_prefix16(const at::Tensor& kv, const at::Tensor& indices) {
  if (indices.size(2) < kMaxSparse) {
    return false;
  }

  std::lock_guard<std::mutex> guard(g_state_mutex);
  auto it = std::find_if(
      g_eligibility_cache.begin(),
      g_eligibility_cache.end(),
      [&](const EligibilityEntry& entry) {
        return same_eligibility_key(entry, indices, kv.size(0));
      });
  if (it != g_eligibility_cache.end()) {
    const bool eligible = it->eligible;
    auto entry = std::move(*it);
    g_eligibility_cache.erase(it);
    g_eligibility_cache.push_front(std::move(entry));
    return eligible;
  }

  const auto flat = indices.select(1, 0);
  const auto prefix = flat.slice(1, 0, kMaxSparse);
  bool eligible = at::sum(prefix.ge(0), {1}, false, at::kInt)
                      .gt(0)
                      .all()
                      .item<bool>();
  eligible = eligible && !flat.lt(-1).any().item<bool>();
  eligible = eligible && prefix.max().item<int32_t>() < kv.size(0);
  if (indices.size(2) > kMaxSparse) {
    eligible = eligible &&
        !flat.slice(1, kMaxSparse, indices.size(2)).ge(0).any().item<bool>();
  }

  g_eligibility_cache.push_front(EligibilityEntry{
      indices.unsafeGetTensorImpl(),
      indices.const_data_ptr(),
      indices.unsafeGetTensorImpl()->version_counter().current_version(),
      kv.size(0),
      indices,
      eligible});
  while (g_eligibility_cache.size() > kEligibilityCacheLimit) {
    g_eligibility_cache.pop_back();
  }
  return eligible;
}

bool use_wide_schedule(int device) {
  std::lock_guard<std::mutex> guard(g_state_mutex);
  auto it = std::find_if(
      g_device_schedules.begin(),
      g_device_schedules.end(),
      [&](const DeviceSchedule& schedule) { return schedule.device == device; });
  if (it != g_device_schedules.end()) {
    return it->wide;
  }

  cudaDeviceProp properties{};
  C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  const bool wide = properties.sharedMemPerBlock >= kPrefillWideSmem;
  if (wide) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sparse_decode_partial_generated_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kDecodeWideSmem));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        sparse_prefill_generated_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kPrefillWideSmem));
  }
  g_device_schedules.push_back(DeviceSchedule{device, wide});
  return wide;
}

at::Tensor launch_prefix16(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    cudaStream_t stream) {
  auto output = at::empty_like(q);
  const int blocks = static_cast<int>(q.size(0)) * kHeads;
  sparse_decode_prefix16_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
      indices.data_ptr<int32_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(indices.size(2)),
      static_cast<float>(sm_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

__global__ void sparse_attention_lse_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv,
    const int32_t* __restrict__ indices,
    float* __restrict__ lse,
    int sparse_width,
    int kv_rows,
    float sm_scale) {
  const int token = blockIdx.x / kHeads;
  const int head = blockIdx.x % kHeads;
  const int tid = threadIdx.x;
  __shared__ float reduction[kThreads];
  __shared__ float running_max;
  __shared__ float running_sum;

  if (tid == 0) {
    running_max = -INFINITY;
    running_sum = 0.0f;
  }
  __syncthreads();

  const __nv_bfloat16* q_ptr = q + (token * kHeads + head) * kDim;
  for (int item = 0; item < sparse_width; ++item) {
    const int32_t row = indices[token * sparse_width + item];
    float dot = 0.0f;
    if (row >= 0 && row < kv_rows) {
      const __nv_bfloat16* kv_ptr = kv + static_cast<int64_t>(row) * kDim;
      for (int d = tid; d < kDim; d += kThreads) {
        dot += __bfloat162float(q_ptr[d]) * __bfloat162float(kv_ptr[d]);
      }
    }
    reduction[tid] = dot;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduction[tid] += reduction[tid + stride];
      }
      __syncthreads();
    }
    if (tid == 0 && row >= 0 && row < kv_rows) {
      const float score = reduction[0] * sm_scale;
      const float next_max = fmaxf(running_max, score);
      running_sum = running_sum * expf(running_max - next_max) +
          expf(score - next_max);
      running_max = next_max;
    }
    __syncthreads();
  }

  if (tid == 0) {
    lse[token * kHeads + head] =
        running_sum == 0.0f ? -INFINITY : logf(running_sum) + running_max;
  }
}

at::Tensor launch_lse(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    cudaStream_t stream) {
  auto lse = at::empty({q.size(0), kHeads}, q.options().dtype(at::kFloat));
  sparse_attention_lse_kernel<<<q.size(0) * kHeads, kThreads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr<at::BFloat16>()),
      indices.data_ptr<int32_t>(),
      lse.data_ptr<float>(),
      static_cast<int>(indices.size(2)),
      static_cast<int>(kv.size(0)),
      static_cast<float>(sm_scale));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return lse;
}

at::Tensor launch_general_decode(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    bool wide,
    cudaStream_t stream) {
  const int seq_len = static_cast<int>(q.size(0));
  const int seq_len_kv = static_cast<int>(kv.size(0));
  auto partial_o = at::empty(
      {q.size(0), kSplitKV, kHeads, kDim}, q.options());
  auto partial_lse = at::empty(
      {q.size(0), kSplitKV, kHeads}, q.options().dtype(at::kFloat));
  auto output = at::empty_like(q);
  const float scale_log2 = static_cast<float>(sm_scale) * kLog2E;

  const dim3 partial_grid(seq_len * 2, kSplitKV);
  if (wide) {
    sparse_decode_partial_generated_kernel<<<
        partial_grid, kThreads, kDecodeWideSmem, stream>>>(
        indices.data_ptr<int32_t>(),
        reinterpret_cast<const bfloat16_t*>(kv.data_ptr<at::BFloat16>()),
        partial_lse.data_ptr<float>(),
        reinterpret_cast<bfloat16_t*>(partial_o.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16_t*>(q.data_ptr<at::BFloat16>()),
        seq_len,
        seq_len_kv,
        scale_log2);
  } else {
    sparse_decode_partial_stage1_generated_kernel<<<
        partial_grid, kThreads, kDecodeStage1Smem, stream>>>(
        indices.data_ptr<int32_t>(),
        reinterpret_cast<const bfloat16_t*>(kv.data_ptr<at::BFloat16>()),
        partial_lse.data_ptr<float>(),
        reinterpret_cast<bfloat16_t*>(partial_o.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16_t*>(q.data_ptr<at::BFloat16>()),
        seq_len,
        seq_len_kv,
        scale_log2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  sparse_decode_combine_generated_kernel<<<
      seq_len * 4, kThreads, kSplitKV * 16 * sizeof(float), stream>>>(
      reinterpret_cast<bfloat16_t*>(output.data_ptr<at::BFloat16>()),
      partial_lse.data_ptr<float>(),
      reinterpret_cast<const bfloat16_t*>(partial_o.data_ptr<at::BFloat16>()),
      seq_len);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor launch_prefill(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    bool wide,
    cudaStream_t stream) {
  const int seq_len = static_cast<int>(q.size(0));
  const int seq_len_kv = static_cast<int>(kv.size(0));
  const float scale_log2 = static_cast<float>(sm_scale) * kLog2E;
  auto output = at::empty_like(q);

  if (wide) {
    sparse_prefill_generated_kernel<<<
        seq_len, 512, kPrefillWideSmem, stream>>>(
        indices.data_ptr<int32_t>(),
        reinterpret_cast<const bfloat16_t*>(kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16_t*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16_t*>(q.data_ptr<at::BFloat16>()),
        seq_len,
        seq_len_kv,
        scale_log2);
  } else {
    sparse_prefill_stage1_generated_kernel<<<
        seq_len * 2, kThreads, kPrefillStage1Smem, stream>>>(
        indices.data_ptr<int32_t>(),
        reinterpret_cast<const bfloat16_t*>(kv.data_ptr<at::BFloat16>()),
        reinterpret_cast<bfloat16_t*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bfloat16_t*>(q.data_ptr<at::BFloat16>()),
        seq_len,
        seq_len_kv,
        scale_log2);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

std::vector<at::Tensor> sparse_attention_fwd(
    const at::Tensor& q,
    const at::Tensor& kv,
    const at::Tensor& indices,
    double sm_scale,
    int64_t d_v,
    bool return_lse) {
  check_inputs(q, kv, indices, d_v);
  const c10::cuda::OptionalCUDAGuard device_guard(q.device());
  const cudaStream_t stream =
      at::cuda::getCurrentCUDAStream(q.get_device());

  at::Tensor output;
  if (q.size(0) <= kDecodeMaxQueryTokens && can_use_prefix16(kv, indices)) {
    output = launch_prefix16(q, kv, indices, sm_scale, stream);
  } else {
    TORCH_CHECK(indices.size(2) == kTopK,
                "general native sparse attention requires K=2112, got ",
                indices.size(2));
    const bool wide = use_wide_schedule(q.get_device());
    output = q.size(0) <= kDecodeMaxQueryTokens
        ? launch_general_decode(q, kv, indices, sm_scale, wide, stream)
        : launch_prefill(q, kv, indices, sm_scale, wide, stream);
  }

  if (return_lse) {
    return {output, launch_lse(q, kv, indices, sm_scale, stream)};
  }
  return {output};
}

void sparse_attention_clear_cache() {
  std::lock_guard<std::mutex> guard(g_state_mutex);
  g_eligibility_cache.clear();
}
