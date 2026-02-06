// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cmath>
#include <cub/cub.cuh>
#include "../kernel/dispatch_utils.h"
#include "../kernel/utils.h"
#include "../kernel/utils.cuh"
#include "../kernel/rms_norm_vllm.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

namespace vllm {

template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__device__ void rms_norm_dynamic_per_token_quant_vec(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon,
    float const min_scaling_factor, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute rms
  vllm::vectorized::compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, var_epsilon, residual);

  // Compute scale
  vllm::vectorized::compute_dynamic_per_token_scales<scalar_t, scalar_out_t,
                                                     has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub,
      hidden_size, residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, true,
                                     has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert token_scale for exact match with FBGemm
    vllm::vectorized::norm_and_quant<scalar_t, scalar_out_t, false,
                                     has_residual>(
        out, input, weight, rms, token_scale, hidden_size, residual);
  }
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, float const var_epsilon,
    float const min_scaling_factor, int32_t const hidden_size,
    scalar_t* __restrict__ residual = nullptr) {
  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  if (can_vectorize) {
    return rms_norm_dynamic_per_token_quant_vec<scalar_t, scalar_out_t,
                                                has_residual>(
        out, scales, input, weight, scale_ub, var_epsilon, min_scaling_factor,
        hidden_size, residual);
  }

  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute RMS
  vllm::compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                            var_epsilon, residual);
  // Compute Scale
  vllm::compute_dynamic_per_token_scales<scalar_t, scalar_out_t, has_residual>(
      &token_scale, scales, input, weight, rms, scale_ub,
      hidden_size, residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    vllm::norm_and_quant<scalar_t, scalar_out_t, true, has_residual>(
        out, input, weight, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    vllm::norm_and_quant<scalar_t, scalar_out_t, false, has_residual>(
        out, input, weight, rms, token_scale, hidden_size, residual);
  }
}

template <typename scalar_t, typename scalar_out_t, int32_t VPT, int32_t num_vec_elems, int32_t block_dim, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_opt_kernel_(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const* scale_ub, 
    float const var_epsilon,
    float const min_scaling_factor, 
    int32_t const hidden_size, 
    scalar_t* __restrict__ residual = nullptr) {

  int32_t block_idx = blockIdx.x;
  int32_t token_offset = block_idx * hidden_size;
  float ss = 0.0f;
  float block_absmax_val_maybe = 0.0f;
  
  vec4_t<scalar_t> const* vec_input = reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
  vec4_t<scalar_t> const* vec_weight = reinterpret_cast<vec4_t<scalar_t> const*>(weight);
  vec4_t<scalar_t>* vec_residual = nullptr;
  if constexpr (has_residual) {
    vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
  }
  constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

  float in_reg[16];
  int idx = 0;

  for (int32_t i = threadIdx.x; i < num_vec_elems; i+= block_dim) {
    vec4_t<scalar_t> in = vec_input[i];
    vec4_t<scalar_t> const w = vec_weight[i];

    in_reg[idx + 0] = static_cast<float>(in.x);
    in_reg[idx + 1] = static_cast<float>(in.y);
    in_reg[idx + 2] = static_cast<float>(in.z);
    in_reg[idx + 3] = static_cast<float>(in.w);

    if constexpr (has_residual) {
      vec4_t<scalar_t> res = vec_residual[i];
      in_reg[idx + 0] += static_cast<float>(res.x);
      in_reg[idx + 1] += static_cast<float>(res.y);
      in_reg[idx + 2] += static_cast<float>(res.z);
      in_reg[idx + 3] += static_cast<float>(res.w);
      in.x = static_cast<scalar_t>(in_reg[idx + 0]);
      in.y = static_cast<scalar_t>(in_reg[idx + 1]);
      in.z = static_cast<scalar_t>(in_reg[idx + 2]);
      in.w = static_cast<scalar_t>(in_reg[idx + 3]);
      vec_residual[i] = in;
    }

    ss += square(in_reg[idx + 0]);
    ss += square(in_reg[idx + 1]);
    ss += square(in_reg[idx + 2]);
    ss += square(in_reg[idx + 3]);

    in_reg[idx + 0] = in_reg[idx + 0] * static_cast<float>(w.x);
    in_reg[idx + 1] = in_reg[idx + 1] * static_cast<float>(w.y);
    in_reg[idx + 2] = in_reg[idx + 2] * static_cast<float>(w.z);
    in_reg[idx + 3] = in_reg[idx + 3] * static_cast<float>(w.w);

    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(in_reg[idx + 0]));
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(in_reg[idx + 1]));
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(in_reg[idx + 2]));
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(in_reg[idx + 3]));
    idx += VPT;
  }
  reduce_max_sum<float, block_dim>(block_absmax_val_maybe, ss);

  __shared__ float s_rms;
  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + var_epsilon);
    float* out_scale_ptr = reinterpret_cast<float*>(&out[token_offset]);
    float scale = block_absmax_val_maybe * s_rms;
    scale = max(scale / qmax, min_scaling_factor);
    s_token_scale = 1.0f / scale;
    scales[blockIdx.x] = scale;
  }
  __syncthreads();

  q8x4_t<scalar_out_t>* vec_output = reinterpret_cast<q8x4_t<scalar_out_t>*>(&out[token_offset]);
  idx = 0;
  for (int32_t i = threadIdx.x; i < num_vec_elems; i += block_dim){
    vec4_t<scalar_t> const w = vec_weight[i];
    q8x4_t<scalar_out_t> out;
    out.x = ScaledQuant<scalar_out_t, true>::quant_fn(in_reg[idx + 0] * s_rms, s_token_scale);
    out.y = ScaledQuant<scalar_out_t, true>::quant_fn(in_reg[idx + 1] * s_rms, s_token_scale);
    out.z = ScaledQuant<scalar_out_t, true>::quant_fn(in_reg[idx + 2] * s_rms, s_token_scale);
    out.w = ScaledQuant<scalar_out_t, true>::quant_fn(in_reg[idx + 3] * s_rms, s_token_scale);
    vec_output[i] = out;
    idx += VPT;
  }
}

}

// Residual add + RMS norm + dynamic per token
template <typename scalar_in_t>
void  rms_norm_dynamic_per_token_quant_dispatch_(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    c10::optional<at::Tensor> const& scale_ub,
    c10::optional<at::Tensor>& residual) {
  int32_t hidden_size = input.size(-1);
  int32_t num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  constexpr int VPT = 4;
  constexpr int block_size = 512;
  dim3 block(std::min(hidden_size, block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  const float min_scaling_factor =
      out.dtype() == torch::kInt8
          ? std::numeric_limits<float>::epsilon()
          : 1.0f / (std::numeric_limits<c10::Float8_e4m3fn>::max() * 512.f);

  if (residual.has_value()) {
    if (hidden_size == 7168) {
      MOE_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_opt_kernel_", [&] {
          vllm::rms_norm_dynamic_per_token_quant_opt_kernel_<scalar_in_t, scalar_t, VPT, 7168 / VPT, block_size, true>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, min_scaling_factor, hidden_size,
                  residual->data_ptr<scalar_in_t>());
        });
    } else {
      MOE_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,
                                                        true>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, min_scaling_factor, hidden_size,
                  residual->data_ptr<scalar_in_t>());
        });
    }
  } else {
    if (hidden_size == 7168) {
      MOE_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_opt_kernel_", [&] {
          vllm::rms_norm_dynamic_per_token_quant_opt_kernel_<scalar_in_t, scalar_t, VPT, 7168 / VPT, block_size, false>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, min_scaling_factor, hidden_size, nullptr);
        });
    } else {
      MOE_DISPATCH_QUANT_TYPES(
        out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
          vllm::rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,
                                                        false>
              <<<grid, block, 0, stream>>>(
                  out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                  input.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->data_ptr<float>() : nullptr,
                  var_epsilon, min_scaling_factor, hidden_size, nullptr);
        });
    }
  }
}

void rms_norm_dynamic_per_token_quant_custom(
    at::Tensor& out,           // [..., hidden_size]
    at::Tensor const& input,   // [..., hidden_size]
    at::Tensor const& weight,  // [hidden_size]
    at::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    c10::optional<at::Tensor> scale_ub, c10::optional<at::Tensor> residual) {
	DEBUG_TRACE_PARAMS(out, input, weight, scales, var_epsilon, scale_ub, residual);
	DEBUG_DUMP_PARAMS(out, input, weight, scales, var_epsilon, scale_ub, residual);
  TORCH_CHECK(out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());

  TORCH_CHECK(scales.dtype() == torch::kFloat32);

  MOE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), " rms_norm_dynamic_per_token_quant_dispatch_", [&] {
         rms_norm_dynamic_per_token_quant_dispatch_<scalar_t>(
            out, input, weight, scales, var_epsilon, scale_ub, residual);
      });
}