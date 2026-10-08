// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
// 2026 - Added Gemma RMSNorm variant with optional residual and pack output support
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cmath>
#include <cub/cub.cuh>
#include "../kernel/dispatch_utils.h"
#include "../kernel/utils.h"
#include "../kernel/rms_norm_vllm.cuh"
#include "../kernel/utils.cuh"

namespace vllm {

// Optional residual helper: returns nullptr if tensor is not defined
inline bool has_residual(const torch::Tensor& residual) {
    return residual.defined() && residual.numel() > 0;
}

template <int N, typename scalar_t>
struct Selector_vec;

template <typename scalar_t>
struct Selector_vec<4, scalar_t> {
  using type = vec4_t<scalar_t>;
};

template <typename scalar_t>
struct Selector_vec<8, scalar_t> {
  using type = vec8_t<scalar_t>;
};

template <int N, typename scalar_t>
struct Selector_q8;

template <typename scalar_t>
struct Selector_q8<4, scalar_t> {
  using type = q8x4_t<scalar_t>;
};

template <typename scalar_t>
struct Selector_q8<8, scalar_t> {
  using type = q8x8_t<scalar_t>;
};

__forceinline__ __device__ float square(float input)
{
    return input * input;
}


//=============================================================================
// Gemma RMSNorm Variant Kernels (in namespace vllm)
//=============================================================================

/**
 * Gemma RMSNorm kernel with weight max computation for accurate scale
 * This version properly computes max(|x*(1+weight)|) for accurate dynamic scale
 * Gemma RMSNorm: y = x * (1.0 + weight) / sqrt(mean(x^2) + eps)
 */
template<typename scalar_t, typename scalar_out_t, bool HAS_RESIDUAL, bool BNEED_PACK>
__global__ void gemma_rms_norm_dynamic_per_token_quant_padding_output_kernel(
    scalar_out_t* __restrict__ out,
    scalar_t* __restrict__ out_rms,
    scalar_out_t* __restrict__ output_quant_int8,
    float* __restrict__ out_scales,
    scalar_t* const __restrict__ input,
    scalar_t* __restrict__ residual,
    scalar_t const* __restrict__ weight,
    float const var_epsilon,
    float const min_scaling_factor,
    int32_t const hidden_size,
    int32_t const pad_size,
    int32_t const num_tokens){

    // Grid-stride loop to handle multiple tokens per block
    for (int32_t block_idx = blockIdx.x; block_idx < num_tokens; block_idx += gridDim.x) {
        int64_t token_offset = block_idx * static_cast<int64_t>(hidden_size);
        float ss = 0.0f;                           // Sum of squares for RMS computation
        float block_absmax_xw = 0.0f;              // Max |x*(1+weight)| for accurate scale
        constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

        // Scale pointer in output padding area (only used when BNEED_PACK is true)
        float* out_scale_ptr = nullptr;
        if (BNEED_PACK) {
            out_scale_ptr = reinterpret_cast<float*>(&out[hidden_size + block_idx * pad_size * 2]);
        }

        // Phase 1: Add (if residual) + compute variance + track max(|x*(1+weight)|)
        for(int32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float x = static_cast<float>(input[token_offset + i]);

            if (HAS_RESIDUAL) {
                x += static_cast<float>(residual[token_offset + i]);
                residual[token_offset + i] = static_cast<scalar_t>(x);
            }

            ss += x * x;
            scalar_t const w = weight[i];
            // Gemma公式: 使用 (1.0 + weight) 而不是 weight
            block_absmax_xw = fmaxf(block_absmax_xw, fabs(x * (1.0f + static_cast<float>(w))));
        }

        using BlockReduce = cub::BlockReduce<float, 512>;
        __shared__ typename BlockReduce::TempStorage reduceStorageSum;
        ss = BlockReduce(reduceStorageSum).Reduce(ss, cub::Sum{}, blockDim.x);
        __shared__ typename BlockReduce::TempStorage reduceStorageMax;
        block_absmax_xw = BlockReduce(reduceStorageMax).Reduce(block_absmax_xw, cub::Max{}, blockDim.x);

        __shared__ float s_rms;
        __shared__ float s_inv_scale;

        if (threadIdx.x == 0) {
            s_rms = rsqrtf(ss / hidden_size + var_epsilon);
            float scale = block_absmax_xw * s_rms;
            scale = max(scale / qmax, min_scaling_factor);
            s_inv_scale = 1.0f / scale;
            out_scales[block_idx] = scale;
            if (BNEED_PACK) {
                *out_scale_ptr = scale;
            }
        }
        __syncthreads();

        // Phase 2: Gemma RMSNorm + Quantization + Output writes
        for(int32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float x;
            if (HAS_RESIDUAL) {
                x = static_cast<float>(residual[token_offset + i]);
            } else {
                x = static_cast<float>(input[token_offset + i]);
            }
            scalar_t const w = weight[i];
            // Gemma公式: y = x * (1.0 + weight) * rms
            float gemma_norm = x * (1.0f + static_cast<float>(w)) * s_rms;

            // Write BF16 normalized output to out_rms
            out_rms[token_offset + i] = static_cast<scalar_t>(gemma_norm);

            // Quantize and write to output_quant_int8
            scalar_out_t quant_val = ScaledQuant<scalar_out_t, true>::quant_fn(gemma_norm, s_inv_scale);
            output_quant_int8[token_offset + i] = quant_val;

            // Write to packed/unpacked output
            int64_t out_offset = BNEED_PACK ? (block_idx * pad_size * 2 + i) : (block_idx * hidden_size + i);
            out[out_offset] = quant_val;
        }
    }
}

/**
 * Vectorized Gemma RMSNorm kernel with 4-element vectorization
 * Optimized for fixed hidden sizes: 7168, 5120, 6144, 4096
 * Gemma RMSNorm (官方实现): y = x * (1.0 + weight) / sqrt(mean(x^2) + eps)
 */
template<typename scalar_t, typename scalar_out_t, int32_t VPT, int32_t NUM_VEC_ELEMS, int32_t BLOCK_DIM, bool HAS_RESIDUAL, bool BNEED_PACK>
__global__ void gemma_rms_norm_quant_padding_output_opt_kernel(
    scalar_out_t* __restrict__ out,
    scalar_t* __restrict__ out_rms,
    scalar_out_t* __restrict__ output_quant_int8,
    float* __restrict__ out_scales,
    scalar_t* const __restrict__ input,
    scalar_t* __restrict__ residual,
    scalar_t const* __restrict__ weight,
    float const var_epsilon,
    float const min_scaling_factor,
    int32_t const hidden_size,
    int32_t const pad_size,
    int32_t const num_tokens){

    float ss = 0.0f;
    float block_absmax_xw = 0.0f;
    int64_t token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

    vec4_t<scalar_t> const* vec_weight = reinterpret_cast<vec4_t<scalar_t> const*>(weight);
    vec4_t<scalar_t> const* vec_input = reinterpret_cast<vec4_t<scalar_t> const*>(&input[token_offset]);
    vec4_t<scalar_t>* vec_residual = nullptr;
    if (HAS_RESIDUAL) {
        vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[token_offset]);
    }
    constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

    float red_reg[16];
    int idx = 0;

    // Phase 1: Vectorized add + variance + max(|x*(1+weight)|)
    for(int32_t i = threadIdx.x; i < NUM_VEC_ELEMS; i += BLOCK_DIM) {
        vec4_t<scalar_t> in = vec_input[i];
        vec4_t<scalar_t> r;
        vec4_t<scalar_t> const w = vec_weight[i];

        if (HAS_RESIDUAL) {
            r = vec_residual[i];
        }

        red_reg[idx + 0] = static_cast<float>(in.x);
        red_reg[idx + 1] = static_cast<float>(in.y);
        red_reg[idx + 2] = static_cast<float>(in.z);
        red_reg[idx + 3] = static_cast<float>(in.w);

        if (HAS_RESIDUAL) {
            red_reg[idx + 0] += static_cast<float>(r.x);
            red_reg[idx + 1] += static_cast<float>(r.y);
            red_reg[idx + 2] += static_cast<float>(r.z);
            red_reg[idx + 3] += static_cast<float>(r.w);
            r.x = static_cast<scalar_t>(red_reg[idx + 0]);
            r.y = static_cast<scalar_t>(red_reg[idx + 1]);
            r.z = static_cast<scalar_t>(red_reg[idx + 2]);
            r.w = static_cast<scalar_t>(red_reg[idx + 3]);
            vec_residual[i] = r;
        }

        ss += square(red_reg[idx + 0]);
        ss += square(red_reg[idx + 1]);
        ss += square(red_reg[idx + 2]);
        ss += square(red_reg[idx + 3]);

        // Gemma公式: 使用 (1.0 + weight) 而不是 weight
        red_reg[idx + 0] *= (1.0f + static_cast<float>(w.x));
        red_reg[idx + 1] *= (1.0f + static_cast<float>(w.y));
        red_reg[idx + 2] *= (1.0f + static_cast<float>(w.z));
        red_reg[idx + 3] *= (1.0f + static_cast<float>(w.w));

        block_absmax_xw = fmaxf(block_absmax_xw, fabs(red_reg[idx + 0]));
        block_absmax_xw = fmaxf(block_absmax_xw, fabs(red_reg[idx + 1]));
        block_absmax_xw = fmaxf(block_absmax_xw, fabs(red_reg[idx + 2]));
        block_absmax_xw = fmaxf(block_absmax_xw, fabs(red_reg[idx + 3]));
        idx += VPT;
    }

    reduce_max_sum<float, BLOCK_DIM>(block_absmax_xw, ss);

    __shared__ float s_rms;
    __shared__ float s_inv_scale;

    if (threadIdx.x == 0) {
        s_rms = rsqrtf(ss / hidden_size + var_epsilon);
        float scale = block_absmax_xw * s_rms;
        scale = max(scale / qmax, min_scaling_factor);
        s_inv_scale = 1.0f / scale;
        out_scales[blockIdx.x] = scale;
        if (BNEED_PACK) {
            float* out_scale_ptr = reinterpret_cast<float*>(&out[hidden_size + blockIdx.x * pad_size * 2]);
            *out_scale_ptr = scale;
        }
    }
    __syncthreads();

    q8x4_t<scalar_out_t>* vec_output = nullptr;
    q8x4_t<scalar_out_t>* vec_output_unpacked = reinterpret_cast<q8x4_t<scalar_out_t>*>(&output_quant_int8[blockIdx.x * hidden_size]);
    vec4_t<scalar_t>* vec_out_bf16 = reinterpret_cast<vec4_t<scalar_t>*>(&out_rms[blockIdx.x * hidden_size]);

    if (BNEED_PACK) {
        vec_output = reinterpret_cast<q8x4_t<scalar_out_t>*>(&out[blockIdx.x * pad_size * 2]);
    } else {
        vec_output = reinterpret_cast<q8x4_t<scalar_out_t>*>(&out[blockIdx.x * hidden_size]);
    }

    idx = 0;
    for(int32_t i = threadIdx.x; i < NUM_VEC_ELEMS; i += BLOCK_DIM) {
        vec4_t<scalar_t> bf16_out;
        q8x4_t<scalar_out_t> quant_out;

        // Apply RMS normalization (x * (1+weight) already computed in Phase 1)
        red_reg[idx + 0] *= s_rms;
        red_reg[idx + 1] *= s_rms;
        red_reg[idx + 2] *= s_rms;
        red_reg[idx + 3] *= s_rms;

        bf16_out.x = static_cast<scalar_t>(red_reg[idx + 0]);
        bf16_out.y = static_cast<scalar_t>(red_reg[idx + 1]);
        bf16_out.z = static_cast<scalar_t>(red_reg[idx + 2]);
        bf16_out.w = static_cast<scalar_t>(red_reg[idx + 3]);
        vec_out_bf16[i] = bf16_out;

        quant_out.x = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 0], s_inv_scale);
        quant_out.y = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 1], s_inv_scale);
        quant_out.z = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 2], s_inv_scale);
        quant_out.w = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 3], s_inv_scale);

        vec_output[i] = quant_out;
        vec_output_unpacked[i] = quant_out;
        idx += VPT;
    }
}

} // namespace vllm

//=============================================================================
// Gemma RMSNorm Dispatch Functions and Public API
//=============================================================================

// Helper function for dispatching Gemma RMSNorm kernels
template<typename scalar_in_t, bool HAS_RESIDUAL, bool BNEED_PACK>
void gemma_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch(
    torch::Tensor& output,
    torch::Tensor& output_rms,
    torch::Tensor& output_quant_int8,
    torch::Tensor& out_scales,
    torch::Tensor const& input,
    torch::Tensor& residual,
    torch::Tensor const& weight,
    int64_t const pad_size,
    double const var_epsilon){

    int32_t hidden_size = input.size(-1);
    int32_t num_tokens = input.numel() / hidden_size;

    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    dim3 grid(std::min(sm_count, num_tokens));
    constexpr int block_d = 512;
    dim3 block(std::min(hidden_size, block_d));

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int VPT = 4;

    const float min_scaling_factor =
        output.dtype() == torch::kInt8
            ? std::numeric_limits<float>::epsilon()
            : 1.0f / (std::numeric_limits<c10::Float8_e4m3fn>::max() * 448.f);

    // Get residual pointer (nullptr if not defined)
    scalar_in_t* residual_ptr = nullptr;
    if (HAS_RESIDUAL && residual.defined() && residual.numel() > 0) {
        residual_ptr = residual.data_ptr<scalar_in_t>();
    }

    // Dispatch based on hidden_size for optimized kernels
    if (hidden_size == 7168) {
        MOE_DISPATCH_QUANT_TYPES(output.scalar_type(), "gemma_rms_norm_quant_padding_output_opt_kernel", [&] {
            vllm::gemma_rms_norm_quant_padding_output_opt_kernel<scalar_in_t, scalar_t, VPT, 7168 / VPT, block_d, HAS_RESIDUAL, BNEED_PACK>
                <<<num_tokens, block_d, 0, stream>>>(
                    output.data_ptr<scalar_t>(),
                    output_rms.data_ptr<scalar_in_t>(),
                    output_quant_int8.data_ptr<scalar_t>(),
                    out_scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(),
                    residual_ptr,
                    weight.data_ptr<scalar_in_t>(),
                    var_epsilon, min_scaling_factor, hidden_size, pad_size, num_tokens
                );
        });
    } else if (hidden_size == 5120) {
        MOE_DISPATCH_QUANT_TYPES(output.scalar_type(), "gemma_rms_norm_quant_padding_output_opt_kernel", [&] {
            vllm::gemma_rms_norm_quant_padding_output_opt_kernel<scalar_in_t, scalar_t, VPT, 5120 / VPT, block_d, HAS_RESIDUAL, BNEED_PACK>
                <<<num_tokens, block_d, 0, stream>>>(
                    output.data_ptr<scalar_t>(),
                    output_rms.data_ptr<scalar_in_t>(),
                    output_quant_int8.data_ptr<scalar_t>(),
                    out_scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(),
                    residual_ptr,
                    weight.data_ptr<scalar_in_t>(),
                    var_epsilon, min_scaling_factor, hidden_size, pad_size, num_tokens
                );
        });
    } else if (hidden_size == 6144) {
        MOE_DISPATCH_QUANT_TYPES(output.scalar_type(), "gemma_rms_norm_quant_padding_output_opt_kernel", [&] {
            vllm::gemma_rms_norm_quant_padding_output_opt_kernel<scalar_in_t, scalar_t, VPT, 6144 / VPT, block_d, HAS_RESIDUAL, BNEED_PACK>
                <<<num_tokens, block_d, 0, stream>>>(
                    output.data_ptr<scalar_t>(),
                    output_rms.data_ptr<scalar_in_t>(),
                    output_quant_int8.data_ptr<scalar_t>(),
                    out_scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(),
                    residual_ptr,
                    weight.data_ptr<scalar_in_t>(),
                    var_epsilon, min_scaling_factor, hidden_size, pad_size, num_tokens
                );
        });
    } else if (hidden_size == 4096) {
        MOE_DISPATCH_QUANT_TYPES(output.scalar_type(), "gemma_rms_norm_quant_padding_output_opt_kernel", [&] {
            vllm::gemma_rms_norm_quant_padding_output_opt_kernel<scalar_in_t, scalar_t, VPT, 4096 / VPT, block_d, HAS_RESIDUAL, BNEED_PACK>
                <<<num_tokens, block_d, 0, stream>>>(
                    output.data_ptr<scalar_t>(),
                    output_rms.data_ptr<scalar_in_t>(),
                    output_quant_int8.data_ptr<scalar_t>(),
                    out_scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(),
                    residual_ptr,
                    weight.data_ptr<scalar_in_t>(),
                    var_epsilon, min_scaling_factor, hidden_size, pad_size, num_tokens
                );
        });
    } else {
        // Fallback to generic kernel for other hidden sizes
        MOE_DISPATCH_QUANT_TYPES(output.scalar_type(), "gemma_rms_norm_dynamic_per_token_quant_padding_output_kernel", [&] {
            vllm::gemma_rms_norm_dynamic_per_token_quant_padding_output_kernel<scalar_in_t, scalar_t, HAS_RESIDUAL, BNEED_PACK>
                <<<grid, block, 0, stream>>>(
                    output.data_ptr<scalar_t>(),
                    output_rms.data_ptr<scalar_in_t>(),
                    output_quant_int8.data_ptr<scalar_t>(),
                    out_scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(),
                    residual_ptr,
                    weight.data_ptr<scalar_in_t>(),
                    var_epsilon, min_scaling_factor, hidden_size, pad_size, num_tokens
                );
        });
    }
}

// Public API for Gemma RMSNorm
void add_gemma_rms_norm_dynamic_per_token_quant_padding_output(
    at::Tensor& output,
    at::Tensor& output_rms,
    at::Tensor& output_quant_int8,
    at::Tensor& out_scales,
    at::Tensor const& input,
    at::Tensor& residual,
    at::Tensor const& weight,
    const int pad_size,
    const float epsilon,
    const bool bneed_pack){

    // Input validation
    TORCH_CHECK(input.dtype() == at::ScalarType::BFloat16, "Input must be BFloat16");
    TORCH_CHECK(output.is_contiguous() && input.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(out_scales.is_contiguous(), "out_scales must be contiguous");

    // Check if residual is provided
    bool has_residual_val = residual.defined() && residual.numel() > 0;

    // Determine output size based on bneed_pack
    int32_t hidden_size = input.size(-1);
    int32_t num_tokens = input.numel() / hidden_size;

    if (bneed_pack) {
        TORCH_CHECK(output.size(0) >= num_tokens &&
                    output.size(1) >= pad_size * 2,
                    "Output size must be at least [num_tokens, pad_size*2] for packed output");
    } else {
        TORCH_CHECK(output.size(0) >= num_tokens &&
                    output.size(1) >= hidden_size,
                    "Output size must be at least [num_tokens, hidden_size] for unpacked output");
    }

    // Dispatch based on residual presence and pack mode
    MOE_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "gemma_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch", [&] {
            if (has_residual_val) {
                if (bneed_pack) {
                    gemma_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch<scalar_t, true, true>(
                        output, output_rms, output_quant_int8, out_scales, input, residual, weight, pad_size, epsilon);
                } else {
                    gemma_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch<scalar_t, true, false>(
                        output, output_rms, output_quant_int8, out_scales, input, residual, weight, pad_size, epsilon);
                }
            } else {
                if (bneed_pack) {
                    gemma_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch<scalar_t, false, true>(
                        output, output_rms, output_quant_int8, out_scales, input, residual, weight, pad_size, epsilon);
                } else {
                    gemma_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch<scalar_t, false, false>(
                        output, output_rms, output_quant_int8, out_scales, input, residual, weight, pad_size, epsilon);
                }
            }
        });
}