#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <ATen/ATen.h>
#include <ATen/cuda/Atomic.cuh>

#include "../cuda_compat.h"
#include "../dispatch_utils.h"
#include "core/math.hpp"

#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"
namespace {

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                         int32_t col) {
  return row * total_col + col;
}

}  // namespace

// TODO: Refactor common parts with moe_align_sum_kernels
template <typename scalar_t, typename token_cnts_t>
__global__ void moe_lora_align_sum_kernel(
    scalar_t* __restrict__ topk_ids, int32_t* token_lora_mapping,
    int64_t block_size, int num_experts, int max_loras, size_t numel,
    int max_num_tokens_padded, int max_num_m_blocks,
    int32_t* __restrict__ sorted_token_ids, int32_t* __restrict__ expert_ids,
    int topk_num, int32_t* total_tokens_post_pad, int32_t* adapter_enabled,
    int32_t* lora_ids) {
  const size_t tokens_per_thread = div_ceil(numel, blockDim.x);
  const size_t start_idx = threadIdx.x * tokens_per_thread;

  int lora_idx = blockIdx.x;
  int lora_id = lora_ids[lora_idx];
  if (lora_id == -1 || adapter_enabled[lora_id] == 0) {
    return;
  }
  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  token_cnts_t* tokens_cnts = (token_cnts_t*)(shared_mem + num_experts + 1);

  // Initialize sorted_token_ids with numel
  for (size_t it = threadIdx.x; it < max_num_tokens_padded; it += blockDim.x) {
    sorted_token_ids[lora_id * max_num_tokens_padded + it] = numel;
  }

  // Initialize expert_ids with -1
  for (size_t it = threadIdx.x; it < max_num_m_blocks; it += blockDim.x) {
    expert_ids[lora_id * max_num_m_blocks + it] = -1;
  }

  // Initialize total_tokens_post_pad with 0
  if (threadIdx.x == 0) {
    total_tokens_post_pad[lora_id] = 0;
  }

  // 初始化 tokens_cnts (这一步原本就是正确的，不需要改)
  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[index(num_experts, threadIdx.x + 1, i)] = 0;
  }

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int mask = token_lora_mapping[i / topk_num] == lora_id;
    int idx = index(num_experts, threadIdx.x + 1, topk_ids[i]);
    tokens_cnts[idx] += mask;
  }

  __syncthreads();

  // [FIX 1] 修改此处：支持 num_thread < num_experts
  // 原代码：if (threadIdx.x < num_experts) { ... }
  // 新代码：使用循环让有限的线程处理所有的专家列
  for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
    tokens_cnts[index(num_experts, 0, i)] = 0;
    for (int j = 1; j <= blockDim.x; ++j) {
      tokens_cnts[index(num_experts, j, i)] +=
          tokens_cnts[index(num_experts, j - 1, i)];
    }
  }

  __syncthreads();

  // We accumulate the token counts of all experts in thread 0.
  if (threadIdx.x == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] = cumsum[i - 1] +
                  div_ceil(tokens_cnts[index(num_experts, blockDim.x, i - 1)],
                           block_size) *
                      block_size;
    }
    total_tokens_post_pad[lora_id] = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  /**
   * For each expert, each thread processes the tokens of the corresponding
   * blocks and stores the corresponding expert_id for each block.
   */
  // [FIX 2] 修改此处：支持 num_thread < num_experts
  // 原代码：if (threadIdx.x < num_experts) { ... }
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    for (int i = cumsum[e]; i < cumsum[e + 1]; i += block_size) {
      expert_ids[index(max_num_m_blocks, lora_id, i / block_size)] = e;
    }
  }

  for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread; ++i) {
    int32_t expert_id = topk_ids[i];
    
    int32_t rank_post_pad =
        tokens_cnts[index(num_experts, threadIdx.x, expert_id)] +
        cumsum[expert_id];

    int mask = (int)token_lora_mapping[i / topk_num] == lora_id;
    atomicAdd(
        &sorted_token_ids[index(max_num_tokens_padded, lora_id, rank_post_pad)],
        (i - numel) * mask);
    tokens_cnts[index(num_experts, threadIdx.x, expert_id)] += mask;
  }
}

void moe_lora_align_block_size(
    torch::Tensor topk_ids, torch::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor adapter_enabled,
    torch::Tensor lora_ids, std::optional<torch::Tensor> maybe_expert_map) {
  DEBUG_TRACE_PARAMS(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, maybe_expert_map);
  DEBUG_DUMP_PARAMS(topk_ids, token_lora_mapping, num_experts, block_size, max_loras, max_num_tokens_padded, max_num_m_blocks, sorted_token_ids, expert_ids, num_tokens_post_pad, adapter_enabled, lora_ids, maybe_expert_map);
  const int topk_num = topk_ids.size(1);

  TORCH_CHECK(block_size > 0, "block_size should be greater than 0. ");

  int device_max_shared_mem;
  auto dev = topk_ids.get_device();
  cudaDeviceGetAttribute(&device_max_shared_mem,
                         cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const int32_t num_thread = min((int32_t)num_experts, 64);  // WARP_SIZE,
  
  TORCH_CHECK(num_thread <= 128,
              "num_thread must be less than 1024, "
              "and fallback is not implemented yet.");
  auto options_int =
      torch::TensorOptions().dtype(torch::kInt).device(topk_ids.device());
  bool has_expert_map = maybe_expert_map.has_value();
  torch::Tensor expert_map;
  if (has_expert_map) {
    expert_map = maybe_expert_map.value();
  } else {
    expert_map = torch::empty({0}, options_int);
  }

  const int32_t shared_mem = (num_thread + 1) * num_experts * sizeof(int32_t) +
                             (num_experts + 1) * sizeof(int32_t);
  //printf("moe_lora_align_block_size ====>%ld num_thread:%ld num_experts:%lld \n", shared_mem, num_thread, num_experts);
  if (shared_mem > device_max_shared_mem) {
    TORCH_CHECK(false,
                "Shared memory usage exceeds device limit, and global memory "
                "fallback is not implemented yet.");
  }

  VLLM_DISPATCH_INTEGRAL_TYPES(
      topk_ids.scalar_type(), "moe_lora_align_sum_kernel", [&] {
        dim3 blockDim(num_thread);
        auto kernel = moe_lora_align_sum_kernel<scalar_t, int32_t>;
        AT_CUDA_CHECK(VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(
            (void*)kernel, shared_mem));
        kernel<<<max_loras, blockDim, shared_mem, stream>>>(
            topk_ids.data_ptr<scalar_t>(),
            token_lora_mapping.data_ptr<int32_t>(), block_size, num_experts,
            max_loras, topk_ids.numel(), max_num_tokens_padded,
            max_num_m_blocks, sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(), topk_num,
            num_tokens_post_pad.data_ptr<int32_t>(),
            adapter_enabled.data_ptr<int32_t>(), lora_ids.data_ptr<int32_t>());
      });
}