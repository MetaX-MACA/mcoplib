// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <cudaTypedefs.h>

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "scaled_mm_c2x.cu"
#include "cutlass_extensions/common.hpp"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

int32_t get_sm_version_num() {
  int32_t major_capability, minor_capability;
  cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, 0);
  cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, 0);
  int32_t version_num = major_capability * 10 + minor_capability;
  return version_num;
}


void cutlass_moe_mm_sm75(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
                         torch::Tensor const& moe_weight,
                         torch::Tensor const& token_ids, torch::Tensor const& expert_ids, 
                         torch::Tensor const& num_tokens_post_padded, int64_t num_valid_tokens, 
                         int64_t topk, bool mul_routed_weight);
                         
// void cutlass_scaled_mm_sm75(torch::Tensor& c, torch::Tensor const& a,
//                             torch::Tensor const& b,
//                             torch::Tensor const& a_scales,
//                             torch::Tensor const& b_scales,
//                             std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm80(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_sm89(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_sm90(torch::Tensor& c, torch::Tensor const& a,
                            torch::Tensor const& b,
                            torch::Tensor const& a_scales,
                            torch::Tensor const& b_scales,
                            std::optional<torch::Tensor> const& bias);
#endif
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
void cutlass_moe_mm_sm90(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);

#endif

#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100
void cutlass_moe_mm_sm100(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch);
#endif

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
void cutlass_scaled_mm_sm120(torch::Tensor& c, torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             std::optional<torch::Tensor> const& bias);
#endif

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
void cutlass_scaled_mm_sm100(torch::Tensor& c, torch::Tensor const& a,
                             torch::Tensor const& b,
                             torch::Tensor const& a_scales,
                             torch::Tensor const& b_scales,
                             std::optional<torch::Tensor> const& bias);
#endif

#if defined(ENABLE_SCALED_MM_SM90) && ENABLE_SCALED_MM_SM90 || \
    defined(ENABLE_SCALED_MM_SM100) && ENABLE_SCALED_MM_SM100
void get_cutlass_moe_mm_data_caller(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets);

void get_cutlass_pplx_moe_mm_data_caller(torch::Tensor& expert_offsets,
                                         torch::Tensor& problem_sizes1,
                                         torch::Tensor& problem_sizes2,
                                         const torch::Tensor& expert_num_tokens,
                                         const int64_t num_local_experts,
                                         const int64_t padded_m,
                                         const int64_t n, const int64_t k);
#endif

void cutlass_scaled_mm_azp_sm75(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm80(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm89(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_azp_sm90(torch::Tensor& c, torch::Tensor const& a,
                                torch::Tensor const& b,
                                torch::Tensor const& a_scales,
                                torch::Tensor const& b_scales,
                                torch::Tensor const& azp_adj,
                                std::optional<torch::Tensor> const& azp,
                                std::optional<torch::Tensor> const& bias);
#endif

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  // CUTLASS FP8 kernels need at least
  //   CUDA 12.0 on SM90 systems (Hopper)
  //   CUDA 12.4 on SM89 systems (Lovelace)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  } else if (cuda_device_capability >= 89) {
    return CUDA_VERSION >= 12040;
  }
#endif

  return false;
}

bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability) {
  // CUTLASS block-quantized FP8 kernels need at least CUDA 12.0
  // and at least SM90 (Hopper)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 100) {
    return CUDA_VERSION >= 12080;
  } else if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  }
#endif

  return false;
}

bool cutlass_group_gemm_supported(int64_t cuda_device_capability) {
  // CUTLASS grouped FP8 kernels need at least CUDA 12.3 and SM90 (Hopper)
  // or CUDA 12.8 and SM100 (Blackwell)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 100) {
    return CUDA_VERSION >= 12080;
  }
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12030;
  }
#endif

  return false;
}

void cutlass_scaled_mm(torch::Tensor& c, torch::Tensor const& a,
                       torch::Tensor const& b, torch::Tensor const& a_scales,
                       torch::Tensor const& b_scales,
                       std::optional<torch::Tensor> const& bias) {
  DEBUG_TRACE_PARAMS(c, a, b, a_scales, b_scales, bias);
  DEBUG_DUMP_PARAMS(c, a, b, a_scales, b_scales, bias);
#ifdef USE_MACA
  cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
#else               
  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment

  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                bias->dim() == 1);
  }

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));
  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
  if (version_num >= 120) {
    cutlass_scaled_mm_sm120(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) {
    cutlass_scaled_mm_sm100(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90 && version_num < 100) {
    // Hopper
    cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 75) {
    // Turing
    cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
#endif // USE_MACA
}


void cutlass_moe_bf16_mm(
  torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
  torch::Tensor const& moe_weight,
  torch::Tensor const& token_ids, torch::Tensor const& expert_ids, 
  torch::Tensor const& num_tokens_post_padded, int64_t num_valid_tokens, 
  int64_t topk, bool mul_routed_weight) {

#ifdef USE_MACA
  cutlass_moe_mm_sm75(out, a, b, moe_weight, token_ids, expert_ids, 
                      num_tokens_post_padded, num_valid_tokens, 
                      topk, mul_routed_weight);
  
  return;
#endif
TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_moe_bf16_mm for current ops");
}

void cutlass_moe_mm(
    torch::Tensor& out_tensors, torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors, torch::Tensor const& a_scales,
    torch::Tensor const& b_scales, torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes, torch::Tensor const& a_strides,
    torch::Tensor const& b_strides, torch::Tensor const& c_strides,
    bool per_act_token, bool per_out_ch) {
  int32_t version_num = get_sm_version_num();
#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100
  if (version_num >= 100) {
    cutlass_moe_mm_sm100(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                         expert_offsets, problem_sizes, a_strides, b_strides,
                         c_strides, per_act_token, per_out_ch);
    return;
  }
#endif
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
  if (version_num >= 90) {
    cutlass_moe_mm_sm90(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                        expert_offsets, problem_sizes, a_strides, b_strides,
                        c_strides, per_act_token, per_out_ch);
    return;
  }
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for CUDA device capability: ", version_num,
      ". Required capability: 90 or 100");
}

void get_cutlass_moe_mm_data(
    const torch::Tensor& topk_ids, torch::Tensor& expert_offsets,
    torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2,
    torch::Tensor& input_permutation, torch::Tensor& output_permutation,
    const int64_t num_experts, const int64_t n, const int64_t k,
    const std::optional<torch::Tensor>& blockscale_offsets) {
  // This function currently gets compiled only if we have a valid cutlass moe
  // mm to run it for.
  int32_t version_num = get_sm_version_num();
#if (defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90) || \
    (defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100)
  get_cutlass_moe_mm_data_caller(topk_ids, expert_offsets, problem_sizes1,
                                 problem_sizes2, input_permutation,
                                 output_permutation, num_experts, n, k,
                                 blockscale_offsets);
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_data: no cutlass_scaled_mm kernel for "
      "CUDA device capability: ",
      version_num, ". Required capability: 90 or 100");
}

void get_cutlass_pplx_moe_mm_data(torch::Tensor& expert_offsets,
                                  torch::Tensor& problem_sizes1,
                                  torch::Tensor& problem_sizes2,
                                  const torch::Tensor& expert_num_tokens,
                                  const int64_t num_local_experts,
                                  const int64_t padded_m, const int64_t n,
                                  const int64_t k) {
  // This function currently gets compiled only if we have a valid cutlass moe
  // mm to run it for.
  int32_t version_num = get_sm_version_num();
#if (defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90) || \
    (defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100)
  get_cutlass_pplx_moe_mm_data_caller(expert_offsets, problem_sizes1,
                                      problem_sizes2, expert_num_tokens,
                                      num_local_experts, padded_m, n, k);
  return;
#endif
  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_pplx_moe_mm_data: no cutlass_scaled_mm kernel "
      "for CUDA device capability: ",
      version_num, ". Required capability: 90 or 100");
}

void cutlass_scaled_mm_azp(torch::Tensor& c, torch::Tensor const& a,
                           torch::Tensor const& b,
                           torch::Tensor const& a_scales,
                           torch::Tensor const& b_scales,
                           torch::Tensor const& azp_adj,
                           std::optional<torch::Tensor> const& azp,
                           std::optional<torch::Tensor> const& bias) {
  DEBUG_TRACE_PARAMS(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  DEBUG_DUMP_PARAMS(c, a, b, a_scales, b_scales, azp_adj, azp, bias);

  // Checks for conformality
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
              b.size(1) == c.size(1));
  TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  TORCH_CHECK(c.stride(0) % 16 == 0 &&
              b.stride(1) % 16 == 0);  // 16 Byte Alignment
  TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  // bias, azp, azp_adj are all 1d
  // bias and azp_adj have n elements, azp has m elements
  if (bias) {
    TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp & bias types
  TORCH_CHECK(azp_adj.dtype() == torch::kInt32);
  TORCH_CHECK(!azp || azp->dtype() == torch::kInt32);
  TORCH_CHECK(!bias || bias->dtype() == c.dtype(),
              "currently bias dtype must match output dtype ", c.dtype());

  at::cuda::OptionalCUDAGuard const device_guard(device_of(a));

#ifdef USE_MACA

  if (!bias) {
    // mctlass not support None bias

    int32_t n = b.size(1);
    int32_t batchsize = 1;
    if (a.dim() == 3 && b.dim() == 3) {
      // a.size = [batch_size, M, K], b.size = [batch_size, K, N]
      n = b.size(2);
      batchsize = a.size(0);
    }
    auto options = torch::TensorOptions()
                     .dtype(c.dtype())
                     .device(a.device());
    torch::Tensor zero_bias = torch::zeros({batchsize,  n}, options);
    cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, zero_bias);
  } else {
    cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  }

#else

  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90) {
    cutlass_scaled_mm_azp_sm90(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_azp_sm89(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_azp_sm80(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  // Turing
  TORCH_CHECK(version_num >= 75);
  cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  return;
#endif

  TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm_azp for a compute capability less than "
      "CUDA device capability: ",
      version_num);

#endif // USE_MACA
}
