// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <cmath>
#include <cub/cub.cuh>
#include "../kernel/dispatch_utils.h"
#include "../kernel/utils.h"
#include "../kernel/rms_norm_vllm.cuh"
#include "../kernel/utils.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

namespace vllm {

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

template<typename scalar_t, typename scalar_out_t, int VPT>
__global__ void add_rms_norm_dynamic_per_token_quant_padding_output_kernel_vectorized(
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

    int32_t block_idx = blockIdx.x;
    int32_t token_offset = block_idx * hidden_size;
    float ss = 0.0f;
    float block_absmax_val_maybe = 0.0f;
    constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};
    float* out_scale_ptr = reinterpret_cast<float*>(&out[hidden_size + block_idx * pad_size * 2]);

    vec8_t<scalar_t> const* vec_input = reinterpret_cast<vec8_t<scalar_t> const*>(&input[token_offset]);
    vec8_t<scalar_t> const* vec_weight = reinterpret_cast<vec8_t<scalar_t> const*>(weight);
    vec8_t<scalar_t>* vec_residual = reinterpret_cast<vec8_t<scalar_t>*>(&residual[token_offset]);
    int32_t const num_vec_elems = hidden_size >> 3;
    #pragma unroll VPT
    for(int32_t i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
        vec8_t<scalar_t> in = vec_input[i];
        vec8_t<scalar_t> const w = vec_weight[i];
        vec8_t<scalar_t> r = vec_residual[i];

        r.a = in.a + r.a;
        r.b = in.b + r.b;
        r.c = in.c + r.c;
        r.d = in.d + r.d;
        r.x = in.x + r.x;
        r.y = in.y + r.y;
        r.z = in.z + r.z;
        r.w = in.w + r.w;

        vec_residual[i] = r;

        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.a * w.a));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.b * w.b));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.c * w.c));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.d * w.d));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.x * w.x));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.y * w.y));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.z * w.z));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(r.w * w.w));

        ss += square(static_cast<float>(r.a));
        ss += square(static_cast<float>(r.b));
        ss += square(static_cast<float>(r.c));
        ss += square(static_cast<float>(r.d));
        ss += square(static_cast<float>(r.x));
        ss += square(static_cast<float>(r.y));
        ss += square(static_cast<float>(r.z));
        ss += square(static_cast<float>(r.w));
    }
    using BlockReduce = cub::BlockReduce<float, 512>;
    __shared__ typename BlockReduce::TempStorage reduceStorageSum;
    ss = BlockReduce(reduceStorageSum).Reduce(ss, cub::Sum{}, blockDim.x);
    __shared__ typename BlockReduce::TempStorage reduceStorageMax;
    block_absmax_val_maybe = BlockReduce(reduceStorageMax).Reduce(block_absmax_val_maybe, cub::Max{}, blockDim.x);

    __shared__ float s_rms;
    __shared__ float s_token_scale;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(ss / hidden_size + var_epsilon);
        float scale = 0.0f;
        scale = block_absmax_val_maybe * s_rms;
        scale = max(scale / qmax, min_scaling_factor);
        s_token_scale = scale;
        *out_scale_ptr = scale;
        out_scales[blockIdx.x] = scale;
    }
    __syncthreads();
    q8x8_t<scalar_out_t>* vec_output = reinterpret_cast<q8x8_t<scalar_out_t>*>(&out[block_idx * pad_size * 2]);
    q8x8_t<scalar_out_t>* vec_output_unpacked = reinterpret_cast<q8x8_t<scalar_out_t>*>(&output_quant_int8[token_offset]);
    vec8_t<scalar_t>* vec_out_bf16 = reinterpret_cast<vec8_t<scalar_t>*>(&out_rms[token_offset]);
    #pragma unroll VPT
    for(int32_t i = threadIdx.x; i < num_vec_elems; i += blockDim.x) {
        vec8_t<float> x;
        vec8_t<scalar_t> in = vec_input[i];
        vec8_t<scalar_t> const w = vec_weight[i];
        vec8_t<scalar_t> r = vec_residual[i];

        x.a = static_cast<float>(r.a);
        x.b = static_cast<float>(r.b);
        x.c = static_cast<float>(r.c);
        x.d = static_cast<float>(r.d);
        x.x = static_cast<float>(r.x);
        x.y = static_cast<float>(r.y);
        x.z = static_cast<float>(r.z);
        x.w = static_cast<float>(r.w);

        q8x8_t<scalar_out_t> out;
        vec8_t<scalar_t> bf16_out;
        bf16_out.a = static_cast<scalar_t>(x.a * s_rms) * w.a;
        bf16_out.b = static_cast<scalar_t>(x.b * s_rms) * w.b;
        bf16_out.c = static_cast<scalar_t>(x.c * s_rms) * w.c;
        bf16_out.d = static_cast<scalar_t>(x.d * s_rms) * w.d;
        bf16_out.x = static_cast<scalar_t>(x.x * s_rms) * w.x;
        bf16_out.y = static_cast<scalar_t>(x.y * s_rms) * w.y;
        bf16_out.z = static_cast<scalar_t>(x.z * s_rms) * w.z;
        bf16_out.w = static_cast<scalar_t>(x.w * s_rms) * w.w;
        vec_out_bf16[i] = bf16_out;
        
        float token_scale_ = 1.0f / s_token_scale;
        out.a = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.a, token_scale_);
        out.b = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.b, token_scale_);
        out.c = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.c, token_scale_);
        out.d = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.d, token_scale_);
        out.x = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.x, token_scale_);
        out.y = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.y, token_scale_);
        out.z = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.z, token_scale_);
        out.w = ScaledQuant<scalar_out_t, true>::quant_fn(bf16_out.w, token_scale_);
        vec_output[i] = out;
        vec_output_unpacked[i] = out;
    }
}

template<typename scalar_t>
__device__ float multiple_cast_f(scalar_t input, float rms){
  return static_cast<float>(input) * rms;
}

template<typename scalar_t>
__device__ scalar_t multiple_cast(scalar_t input, float rms){
  return static_cast<scalar_t>(static_cast<float>(input) * rms);
}

template<typename scalar_t, typename scalar_out_t, int32_t VPT, int32_t num_vec_elems, int32_t block_dim>
__global__ void add_rms_norm_quant_padding_output_opt_kernel(
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
    float block_absmax_val_maybe = 0.0f;

    vec4_t<scalar_t> const* vec_weight = reinterpret_cast<vec4_t<scalar_t> const*>(weight);
    vec4_t<scalar_t> const* vec_input = reinterpret_cast<vec4_t<scalar_t> const*>(&input[blockIdx.x * hidden_size]);
    vec4_t<scalar_t>* vec_residual = reinterpret_cast<vec4_t<scalar_t>*>(&residual[blockIdx.x * hidden_size]);
    constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};

    float red_reg[16];
    int idx = 0;

    for(int32_t i = threadIdx.x; i < num_vec_elems; i += block_dim) {
        vec4_t<scalar_t> x1 = vec_input[i];
        vec4_t<scalar_t> x2 = vec_residual[i];
        vec4_t<scalar_t> const w = vec_weight[i];

        red_reg[idx + 0] = static_cast<float>(x2.x) + static_cast<float>(x1.x);
        red_reg[idx + 1] = static_cast<float>(x2.y) + static_cast<float>(x1.y);
        red_reg[idx + 2] = static_cast<float>(x2.z) + static_cast<float>(x1.z);
        red_reg[idx + 3] = static_cast<float>(x2.w) + static_cast<float>(x1.w);

        ss += square(red_reg[idx + 0]);
        ss += square(red_reg[idx + 1]);
        ss += square(red_reg[idx + 2]);
        ss += square(red_reg[idx + 3]);

        x2.x = static_cast<scalar_t>(red_reg[idx + 0]);
        x2.y = static_cast<scalar_t>(red_reg[idx + 1]);
        x2.z = static_cast<scalar_t>(red_reg[idx + 2]);
        x2.w = static_cast<scalar_t>(red_reg[idx + 3]);
        vec_residual[i] = x2;

        red_reg[idx + 0] *= static_cast<float>(w.x);
        red_reg[idx + 1] *= static_cast<float>(w.y);
        red_reg[idx + 2] *= static_cast<float>(w.z);
        red_reg[idx + 3] *= static_cast<float>(w.w);

        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(red_reg[idx + 0]));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(red_reg[idx + 1]));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(red_reg[idx + 2]));
        block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(red_reg[idx + 3]));
        idx += VPT;
    }
    reduce_max_sum<float, block_dim>(block_absmax_val_maybe, ss);
    

    __shared__ float s_rms;
    __shared__ float s_token_scale;
    if (threadIdx.x == 0){
        s_rms = rsqrtf(ss / hidden_size + var_epsilon);
        float* out_scale_ptr = reinterpret_cast<float*>(&out[hidden_size + blockIdx.x * pad_size * 2]);
        float scale = block_absmax_val_maybe * s_rms;
        scale = max(scale / qmax, min_scaling_factor);
        s_token_scale = 1.0f / scale; 
        *out_scale_ptr = scale;
        out_scales[blockIdx.x] = scale;
    }
    __syncthreads();

    q8x4_t<scalar_out_t>* vec_output = reinterpret_cast<q8x4_t<scalar_out_t>*>(&out[blockIdx.x * pad_size * 2]);
    q8x4_t<scalar_out_t>* vec_output_unpacked = reinterpret_cast<q8x4_t<scalar_out_t>*>(&output_quant_int8[blockIdx.x * hidden_size]);
    vec4_t<scalar_t>* vec_out_bf16 = reinterpret_cast<vec4_t<scalar_t>*>(&out_rms[blockIdx.x * hidden_size]);
    idx = 0;
    for(int32_t i = threadIdx.x; i < num_vec_elems; i += block_dim) {
        vec4_t<scalar_t> x;
        q8x4_t<scalar_out_t> out_;
        vec4_t<scalar_t> const w = vec_weight[i];

        red_reg[idx + 0] *= s_rms;
        red_reg[idx + 1] *= s_rms;
        red_reg[idx + 2] *= s_rms;
        red_reg[idx + 3] *= s_rms;

        x.x = static_cast<scalar_t>(red_reg[idx + 0]);
        x.y = static_cast<scalar_t>(red_reg[idx + 1]);
        x.z = static_cast<scalar_t>(red_reg[idx + 2]);
        x.w = static_cast<scalar_t>(red_reg[idx + 3]);
        vec_out_bf16[i] = x;

        out_.x = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 0], s_token_scale);
        out_.y = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 1], s_token_scale);
        out_.z = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 2], s_token_scale);
        out_.w = ScaledQuant<scalar_out_t, true>::quant_fn(red_reg[idx + 3], s_token_scale);
        vec_output[i] = out_;
        vec_output_unpacked[i] = out_;
        idx += VPT;
    }
}



template<typename scalar_t, typename scalar_out_t>
__global__ void add_rms_norm_dynamic_per_token_quant_padding_output_kernel(
    scalar_out_t* __restrict__ out,
    scalar_t* const __restrict__ input,
    scalar_t* __restrict__ residual,
    scalar_t const* __restrict__ weight,
    float const var_epsilon,
    float const min_scaling_factor,
    int32_t const hidden_size,
    int32_t const pad_size,
    int32_t const num_tokens){
    for (int32_t block_idx = blockIdx.x; block_idx < num_tokens; block_idx += gridDim.x) {
        int64_t token_offset = block_idx * static_cast<int64_t>(hidden_size);
        float ss = 0.0f;
        float block_absmax_val_maybe = 0.0f;
        constexpr scalar_out_t qmax{std::numeric_limits<scalar_out_t>::max()};
        float* out_scale_ptr = reinterpret_cast<float*>(&out[hidden_size + block_idx * pad_size * 2]);

        for(int32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float x = static_cast<float>(input[token_offset + i]) + static_cast<float>(residual[token_offset + i]);
            scalar_t const w = weight[i];
            block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabs(static_cast<scalar_t>(x) * w));
            ss += x * x;
            residual[token_offset + i] = static_cast<scalar_t>(x);
        }

        using BlockReduce = cub::BlockReduce<float, 512>;
        __shared__ typename BlockReduce::TempStorage reduceStorageSum;
        ss = BlockReduce(reduceStorageSum).Reduce(ss, cub::Sum{}, blockDim.x);
        __shared__ typename BlockReduce::TempStorage reduceStorageMax;
        block_absmax_val_maybe = BlockReduce(reduceStorageMax).Reduce(block_absmax_val_maybe, cub::Max{}, blockDim.x);

        __shared__ float s_rms;
        __shared__ float s_token_scale;

        if (threadIdx.x == 0) {
        s_rms = rsqrtf(ss / hidden_size + var_epsilon);
        float scale = 0.0f;
        scale = block_absmax_val_maybe * s_rms;
        
        scale = max(scale / qmax, min_scaling_factor);
        s_token_scale = scale;
        *out_scale_ptr = scale;
        }
        __syncthreads();

        for(int32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
            float x = static_cast<float>(residual[token_offset + i]);
            scalar_t const w = weight[i];
            float token_scale_ = 1.0f / s_token_scale;
            out[block_idx * pad_size * 2 + i] = ScaledQuant<scalar_out_t, true>::quant_fn(static_cast<scalar_t>(x * s_rms) * w, token_scale_);
        }
    }
}

}

template<typename scalar_in_t>
void add_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch(torch::Tensor& output,
                                                                        torch::Tensor& out_rms,
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

    if (hidden_size == 7168) {
        MOE_DISPATCH_QUANT_TYPES(output.scalar_type(), "add_rms_norm_quant_padding_output_opt_kernel", [&] {
            vllm::add_rms_norm_quant_padding_output_opt_kernel<scalar_in_t, scalar_t, VPT, 7168 / VPT, block_d>
                <<<num_tokens, block_d, 0, stream>>>(
                    output.data_ptr<scalar_t>(), out_rms.data_ptr<scalar_in_t>(), output_quant_int8.data_ptr<scalar_t>(), 
                    out_scales.data_ptr<float>(), input.data_ptr<scalar_in_t>(), 
                    residual.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                    var_epsilon, min_scaling_factor, hidden_size, pad_size, num_tokens
                );
        });
    } else {
        MOE_DISPATCH_QUANT_TYPES(output.scalar_type(), "add_rms_norm_dynamic_per_token_quant_padding_output_kernel", [&] {
            vllm::add_rms_norm_dynamic_per_token_quant_padding_output_kernel<scalar_in_t, scalar_t>
                <<<grid, block, 0, stream>>>(
                    output.data_ptr<scalar_t>(), input.data_ptr<scalar_in_t>(), 
                    residual.data_ptr<scalar_in_t>(), weight.data_ptr<scalar_in_t>(),
                    var_epsilon, min_scaling_factor, hidden_size, pad_size, num_tokens
                );
        });
    }
    
}

void add_rms_norm_dynamic_per_token_quant_padding_output(at::Tensor& output,
                                                         at::Tensor& output_rms,
                                                         at::Tensor& output_quant_int8,
                                                         at::Tensor& out_scales,
                                                         at::Tensor const& input,
                                                         at::Tensor& residual,
                                                         at::Tensor const& weight, 
                                                         const int pad_size, 
                                                         const float epsilon){
	DEBUG_TRACE_PARAMS(output, output_rms, output_quant_int8, out_scales, input, residual,weight, pad_size, epsilon);
	DEBUG_DUMP_PARAMS(output, output_rms, output_quant_int8, out_scales, input, residual,weight, pad_size, epsilon);
    TORCH_CHECK(output.dtype() == torch::kInt8);
    TORCH_CHECK(input.dtype() == at::ScalarType::BFloat16);
    TORCH_CHECK(output.is_contiguous() && input.is_contiguous());
    MOE_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "add_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch", [&] {
            add_rms_norm_dynamic_per_token_quant_padding_output_with_dispatch<scalar_t>(output, output_rms, output_quant_int8, out_scales, input, residual, weight, 
                                                                            pad_size, epsilon);
        });
}