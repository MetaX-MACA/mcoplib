// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
/*
 * Modified by Neural Magic
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Adapted from https://github.com/IST-DASLab/marlin
 */
/*
 * Adapted from  https://github.com/vllm-project/vllm/tree/main/csrc/quantization/gptq_marlin
 */
#include "gptq_marlin.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

//size_m_tensor is the real size_m need calculated
//while size_m is tensor a.shape[0]
torch::Tensor gptq_marlin_gemm_impl(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, const int32_t* size_m_ptr,
                               int64_t size_m, int64_t size_n,
                               int64_t size_k, int sms, bool is_k_full,
                               at::ScalarType dtype=torch::kBFloat16,
                               bool use_atomic_cache = true) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  // Verify num_bits
  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  int pack_factor = 32 / num_bits;

  // Verify A
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  // In order to fit rules of cuda graph, we need to confirm that this gemm will not splited into more than one kernel
  // so size_m must be less than MAX_M_BLOCKS * SLICE_M or size_m can be divisible by MAX_M_BLOCKS * SLICE_M
  if (size_m_ptr != nullptr) {
    constexpr int FULL_M_BLOCK = hgemm_marlin_gptq::MAX_BLOCKS_M * hgemm_marlin_gptq::SLICE_M;
    TORCH_CHECK(size_m < FULL_M_BLOCK || size_m % FULL_M_BLOCK == 0, "size_m = ", size_m,
                " should less than ", FULL_M_BLOCK, " or divisible by ", FULL_M_BLOCK);
  }
#if 0
  // Verify B
  TORCH_CHECK(size_k % gptq_marlin::tile_size == 0, "size_k = ", size_k,
              " is not divisible by tile_size = ", gptq_marlin::tile_size);
  TORCH_CHECK((size_k / gptq_marlin::tile_size) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k, ", tile_size = ", gptq_marlin::tile_size);
  TORCH_CHECK(b_q_weight.size(1) % gptq_marlin::tile_size == 0,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not divisible by tile_size = ", gptq_marlin::tile_size);
  int actual_size_n =
      (b_q_weight.size(1) / gptq_marlin::tile_size) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);
#endif

  // Verify device and strides
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
  TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");

  TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
  TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");

  // Alloc buffers
  auto options = torch::TensorOptions().dtype(dtype).device(a.device());
  torch::Tensor c = torch::empty({size_m, size_n}, options);

  // thread_k: `k` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_k = -1;
  // thread_n: `n` size of a thread_tile in `weights` (can usually be left as
  // auto -1)
  int thread_n = -1;
  // sms: number of SMs to use for the kernel (can usually be left as auto -1)
  //int sms = -1;

  // Verify g_idx and perm
  TORCH_CHECK((g_idx.size(0) == 0 && perm.size(0) == 0) ||
                  (g_idx.size(0) == size_k && perm.size(0) == size_k),
              "Unexpected g_idx.size(0) = ", g_idx.size(0),
              " and perm.size(0) = ", perm.size(0),
              ", where size_k = ", size_k);

  // Detect groupsize and act_order
  int num_groups = -1;
  int group_size = -1;
  bool has_act_order = g_idx.size(0) != 0;

  int b_rank = b_scales.sizes().size();
  TORCH_CHECK(b_rank == 2, "b_scales rank = ", b_rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ", b_scales.size(1),
              " is not size_n = ", size_n);
  num_groups = b_scales.size(0);

  if (has_act_order) {
    printf("[WARNING]: ACT ORDER is not supported\n");
    return c;
  }
  group_size = size_k / num_groups;


  // TODO: Here we directly return fp32 data and MarlinLinear will transpose fp32 to bf16
  auto options_f = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
  int64_t size_atomic_cache = 1;
  {
    //Need to calculate maximum atomic add operation originally
    //so we can save each value that use atomicAdd originally to a memory cache
    //After gemm finished, a sequential add order is promised whenever it is called
    using namespace hgemm_marlin_gptq;
    //Calculate atomic_cache
    int64_t PEUS = 13 * 4 * 8;

    int BLOCKS_N = 8;
    //It is better let TILE_K = quant_group
    //But if quant_group is too large, a quant_group can be divided into two parts
    int BLOCKS_K = group_size / SLICE_K;
    if (group_size > 128) BLOCKS_K = 128 / SLICE_K;
    if (size_m <= SLICE_M*2) {
        BLOCKS_N = 16;
    }

    int total_tiles = size_n / TILE_N * size_k / TILE_K;
    int blocks = PEUS;
    int iters = div_ceil(total_tiles, PEUS);
    // std::cout << "group_size = " << group_size << ", TILE_N = " << TILE_N << ", TILE_K = " << TILE_K << ", total_tiles = " << total_tiles << ", rough iters = " << iters << std::endl;
    if (total_tiles < PEUS) {
        if (TILE_K < group_size) {
            iters = group_size / TILE_K;
        } else {
            iters = 1;
        }
    } else {
        if (TILE_K < group_size) {
            iters = div_ceil(iters, group_size / TILE_K) * group_size / TILE_K;
        }
    }

    size_atomic_cache = div_ceil(size_k, iters*TILE_K)+1;
    if (iters*TILE_K > size_k && iters*TILE_K % size_k != 0) iters = 2;
    // std::cout << "size_k = " << size_k << ", iters = " << iters << ", TILE_K = " << TILE_K
    //   << ", size_atomic_cache = " << size_atomic_cache << std::endl;
    TORCH_CHECK(size_atomic_cache > 0, "Get invalid size_atomic_cache = ", size_atomic_cache);
  }
  //c_temp needs to use torch::zeros to promise all values to be zero
  //as we are not sure if each column saves the same size of atomic caches
  //But if we use torch::zeros here, an elementwise kernel will be called here, which blocks although
  //only a few millin seconds but still significant.
  //Fortunately there is a way to fill unused cache position in gemm kernel, we can save time blocked here.
  torch::Tensor c_temp = torch::empty({size_m, size_atomic_cache, size_n}, options_f);

#if 0
  // Verify workspace size
  TORCH_CHECK(
      size_n % gptq_marlin::min_thread_n == 0, "size_n = ", size_n,
      ", is not divisible by min_thread_n = ", gptq_marlin::min_thread_n);
  int min_workspace_size =
      (size_n / gptq_marlin::min_thread_n) * gptq_marlin::max_par;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);
#endif

#define half_half_body  size_m, size_n, size_k, group_size, size_atomic_cache, \
            (const half*)(a.data_ptr<at::Half>()), size_k, \
            (const uint32_t *)(b_q_weight.data_ptr()), size_n, \
            (half*)(c.data_ptr<at::Half>()), (float*)c_temp.data_ptr(), size_m_ptr, \
            size_n, nullptr, (half*)(b_scales.data_ptr<at::Half>()), (uint32_t*)(g_idx.data_ptr()), true, \
            at::cuda::getCurrentCUDAStream(dev)

#define half_bfloat_body  size_m, size_n, size_k, group_size, size_atomic_cache, \
            (const half*)(a.data_ptr<at::Half>()), size_k, \
            (const uint32_t *)(b_q_weight.data_ptr()), size_n, \
            (nv_bfloat16*)(c.data_ptr<at::BFloat16>()), (float*)c_temp.data_ptr(), size_m_ptr, \
            size_n, nullptr, (half*)(b_scales.data_ptr<at::Half>()), (uint32_t*)(g_idx.data_ptr()), true, \
            at::cuda::getCurrentCUDAStream(dev)

#define bfloat_half_body size_m, size_n, size_k, group_size,size_atomic_cache, \
          (const nv_bfloat16*)(a.data_ptr<at::BFloat16>()), size_k, \
          (const uint32_t *)(b_q_weight.data_ptr()), size_n, \
          (half*)(c.data_ptr<at::Half>()), (float*)(c_temp.data_ptr()), size_m_ptr, \
          size_n, nullptr, (nv_bfloat16*)(b_scales.data_ptr<at::BFloat16>()), (uint32_t*)(g_idx.data_ptr()), true, \
          at::cuda::getCurrentCUDAStream(dev)

#define bfloat_bfloat_body size_m, size_n, size_k, group_size,size_atomic_cache, \
          (const nv_bfloat16*)(a.data_ptr<at::BFloat16>()), size_k, \
          (const uint32_t *)(b_q_weight.data_ptr()), size_n, \
          (nv_bfloat16*)(c.data_ptr<at::BFloat16>()), (float*)(c_temp.data_ptr()), size_m_ptr, \
          size_n, nullptr, (nv_bfloat16*)(b_scales.data_ptr<at::BFloat16>()), (uint32_t*)(g_idx.data_ptr()), true, \
          at::cuda::getCurrentCUDAStream(dev)

  int dev = a.get_device();
  if((a.scalar_type() != at::ScalarType::Half) && (a.scalar_type() != at::ScalarType::BFloat16))
    TORCH_CHECK(false, "gpt_marlin_gemm only supports bfloat16 and float16, a.scalar_type unsupported!");
  if((dtype != at::ScalarType::Half) && (dtype != at::ScalarType::BFloat16))
    TORCH_CHECK(false, "gpt_marlin_gemm only supports bfloat16 and float16, dtype unsupported!");
  if (a.scalar_type() == at::ScalarType::Half) {
    if (num_bits == 4) {
      if(dtype == at::ScalarType::Half)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<half, sglang::kU4B8.id(), half, uint32_t, true>(half_half_body);
        else
          mcOpLib::launch_gemm<half, sglang::kU4B8.id(), half, uint32_t, false>(half_half_body);
      else if (dtype == at::ScalarType::BFloat16)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<half, sglang::kU4B8.id(), nv_bfloat16, uint32_t, true>(half_bfloat_body);
        else
          mcOpLib::launch_gemm<half, sglang::kU4B8.id(), nv_bfloat16, uint32_t, false>(half_bfloat_body);
    } 
    else {
      if(dtype == at::ScalarType::Half)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<half, sglang::kU8B128.id(), half, uint32_t, true>(half_half_body);
        else
          mcOpLib::launch_gemm<half, sglang::kU8B128.id(), half, uint32_t, false>(half_half_body);
      else if (dtype == at::ScalarType::BFloat16)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<half, sglang::kU8B128.id(), nv_bfloat16, uint32_t, true>(half_bfloat_body);
        else
          mcOpLib::launch_gemm<half, sglang::kU8B128.id(), nv_bfloat16, uint32_t, false>(half_bfloat_body);
    }
  }
  else if (a.scalar_type() == at::ScalarType::BFloat16) {
    if (num_bits == 4) {
      if(dtype == at::ScalarType::Half)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU4B8.id(), half, uint32_t, true>(bfloat_half_body);
        else
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU4B8.id(), half, uint32_t, false>(bfloat_half_body);
      else if (dtype == at::ScalarType::BFloat16)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU4B8.id(), nv_bfloat16, uint32_t, true>(bfloat_bfloat_body);
        else
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU4B8.id(), nv_bfloat16, uint32_t, false>(bfloat_bfloat_body);
    } else {
      if(dtype == at::ScalarType::Half)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU8B128.id(), half, uint32_t, true>(bfloat_half_body);
        else
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU8B128.id(), half, uint32_t, false>(bfloat_half_body);
      else if (dtype == at::ScalarType::BFloat16)
        if(use_atomic_cache)
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU8B128.id(), nv_bfloat16, uint32_t, true>(bfloat_bfloat_body);
        else
          mcOpLib::launch_gemm<nv_bfloat16, sglang::kU8B128.id(), nv_bfloat16, uint32_t, false>(bfloat_bfloat_body);
    }
  } 
  //return c_temp;
  return c;
}

torch::Tensor gptq_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits, torch::Tensor& size_m_tensor,
                               int64_t size_m, int64_t size_n,
                               int64_t size_k, int sms, bool is_k_full, 
                               at::ScalarType dtype=torch::kBFloat16,
                               bool use_atomic_cache = true) {
  DEBUG_TRACE_PARAMS(a, b_q_weight, b_scales, g_idx, perm, workspace, num_bits, size_m_tensor, size_m, size_n, size_k, sms, is_k_full, dtype, use_atomic_cache);
	DEBUG_DUMP_PARAMS(a, b_q_weight, b_scales, g_idx, perm, workspace, num_bits, size_m_tensor, size_m, size_n, size_k, sms, is_k_full, dtype, use_atomic_cache);

  return gptq_marlin_gemm_impl(a, b_q_weight, b_scales, g_idx, perm, workspace, num_bits,
      (const int32_t*)size_m_tensor.data_ptr(), size_m, size_n, size_k, sms, is_k_full, dtype, use_atomic_cache);
}

torch::Tensor gptq_marlin_gemm_legacy(torch::Tensor& a, torch::Tensor& b_q_weight,
                               torch::Tensor& b_scales, torch::Tensor& g_idx,
                               torch::Tensor& perm, torch::Tensor& workspace,
                               int64_t num_bits,
                               int64_t size_m, int64_t size_n,
                               int64_t size_k, bool is_k_full,
                               at::ScalarType dtype=torch::kBFloat16,
                               bool use_atomic_cache = true) {
  DEBUG_TRACE_PARAMS(a, b_q_weight, b_scales, g_idx, perm, workspace, num_bits, size_m, size_n, size_k, is_k_full, dtype, use_atomic_cache);
	DEBUG_DUMP_PARAMS(a, b_q_weight, b_scales, g_idx, perm, workspace, num_bits, size_m, size_n, size_k, is_k_full, dtype, use_atomic_cache);
  return gptq_marlin_gemm_impl(a, b_q_weight, b_scales, g_idx, perm, workspace, num_bits,
      nullptr, size_m, size_n, size_k, -1, is_k_full, dtype, use_atomic_cache);
}
