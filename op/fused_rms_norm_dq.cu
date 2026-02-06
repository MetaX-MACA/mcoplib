// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "../kernel/dispatch_utils.h"
#include "../kernel/utils.cuh"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

template<typename T>
static __device__ __forceinline__ float convert_to_float(T value) {
    return float(value);
}

template<>
static __device__ __forceinline__ float convert_to_float<maca_bfloat16>(maca_bfloat16 value) {
    return __bfloat162float(value);
}

template<>
static __device__ __forceinline__ float convert_to_float<half>(half value) {
    return __half2float(value);
}


template<typename T>
static __device__ __forceinline__ T float_to_dstT(float value) {
  return static_cast<T>(value);
}

template<>
static __device__ __forceinline__ maca_bfloat16 float_to_dstT(float value) {
  return __float2bfloat16(value);
}

template<>
static __device__ __forceinline__ half float_to_dstT(float value) {
  return __float2half(value);
}

template <typename fp8_type>
static __device__ __forceinline__ fp8_type float_to_fp8(float const x) {
  float const r =
      fmax(-quant_type_max_v<fp8_type>, fmin(x, quant_type_max_v<fp8_type>));
  return static_cast<fp8_type>(r);
}

template <typename quant_type_t, bool is_scale_inverted, typename enable = void>
struct ScaledQuant;

template <typename quant_type_t, bool is_scale_inverted>
struct ScaledQuant<
    quant_type_t, is_scale_inverted,
    typename std::enable_if_t<std::is_same_v<quant_type_t, int8_t>>> {
  static __device__ __forceinline__ quant_type_t quant_fn(float const x,
                                                          float const scale) {
    if constexpr (is_scale_inverted) {
      return float_to_int8_rn(x * scale);
    } else {
      return float_to_int8_rn(x / scale);
    }
  }
};

// has_residual must be true, if residual is not a nullptr
template <typename scalar_t, bool has_residual = false>
__device__ void compute_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon,
                            scalar_t * after_res, scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  // sum of squares
  float ss = 0.0f;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    if constexpr (has_residual) {
      x += static_cast<float>(residual[token_offset + i]);
    }
    if(after_res) {
      after_res[token_offset + i] = float_to_dstT<scalar_t>(x);
    }
    ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

template <typename scalar_t>
__device__ void compute_head_rms(float* rms, scalar_t const* __restrict__ input,
                            int32_t const hidden_size, float const epsilon) {
  // sum of squares
  float ss = 0.0f;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[i]);
    ss += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, blockDim.x);

  __shared__ float s_rms;
  if (threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + epsilon);
  }
  __syncthreads();

  *rms = s_rms;
}

template <typename scalar_t, typename scalar_out_t, typename smooth_scalar_t, bool has_residual = false>
__device__ void compute_dynamic_per_token_scales(
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, smooth_scalar_t const* __restrict__ weight,
    float const rms, smooth_scalar_t const* __restrict__ smooth_scale,
    int32_t const hidden_size,
    scalar_t const* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;
  constexpr scalar_out_t qmax{quant_type_max_v<scalar_out_t>};

  float block_absmax_val_maybe = 0.0f;
  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    x = x / rms * convert_to_float<smooth_scalar_t>(weight[i])*convert_to_float<smooth_scalar_t>(smooth_scale[i]);
    block_absmax_val_maybe = fmaxf(block_absmax_val_maybe, fabsf(x));
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  block_absmax_val_maybe =
      BlockReduce(reduceStore)
          .Reduce(block_absmax_val_maybe, cub::Max{}, blockDim.x);

  __shared__ float s_token_scale;
  if (threadIdx.x == 0) {
    float scale = 0.0f;
    scale = block_absmax_val_maybe;
    // token scale computation
    scale = scale / 127.0;
    s_token_scale = scale;                 // Shared memory store
    all_token_scales[blockIdx.x] = scale;  // Global output store
  }
  __syncthreads();

  *token_scale = s_token_scale;
}

template <typename scalar_t, typename scalar_out_t, typename scalar_smooth_t, bool is_scale_inverted,
          bool has_residual = false>
__device__ void norm_and_quant(scalar_out_t* __restrict__ output,
                               scalar_t* __restrict__ after_norm,
                               scalar_t const* __restrict__ input,
                               scalar_smooth_t const* __restrict__ weight,
                               scalar_smooth_t const* __restrict__ smooth_scale,
                               float const rms, float const scale,
                               int32_t const hidden_size,
                               scalar_t* __restrict__ residual = nullptr) {
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);
  ;

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    // Norm
    x =x / rms * convert_to_float<scalar_smooth_t>(weight[i]); 
    after_norm[token_offset + i] = static_cast<scalar_t>(x);
    x = x * convert_to_float<scalar_smooth_t>(smooth_scale[i]);
    // Quant
    output[token_offset + i] =
        ScaledQuant<scalar_out_t, is_scale_inverted>::quant_fn(x, scale);
  }
}

template <typename scalar_t, typename scalar_weight_t>
__device__ void norm_without_smooth(scalar_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_weight_t const* __restrict__ weight,
                               float const rms, int32_t const hidden_size,
                               bool rms_div, scalar_t* __restrict__ residual = nullptr) {
                                
  int64_t const token_offset = blockIdx.x * static_cast<int64_t>(hidden_size);

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[token_offset + i]);
    // Norm
    if(rms_div == false) {
       x =x * rms * convert_to_float<scalar_weight_t>(weight[i]); 
    }else{
       x =x / rms * convert_to_float<scalar_weight_t>(weight[i]); 
    }
    
    // Quant
    output[token_offset + i] = static_cast<scalar_t>(x);
  }
}

template <typename scalar_t, typename scalar_smooth_t>
__device__ void do_head_norm(scalar_t* __restrict__ output,
                               scalar_t const* __restrict__ input,
                               scalar_smooth_t const* __restrict__ weight,
                               float const rms, int32_t const hidden_size) {

  for (auto i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    float x = static_cast<float>(input[i]);
    // Norm
    x = x / rms * convert_to_float<scalar_smooth_t>(weight[i]);
    // Quant
    output[i] = float_to_dstT<scalar_t>(x);
  }
}

template <typename scalar_t, typename scalar_out_t, typename scalar_smooth_t, bool has_residual = false>
__device__ void rms_norm_dynamic_per_token_quant_vec(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_smooth_t const* __restrict__ weight,  // [hidden_size]
    scalar_smooth_t const* smooth_scale, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ after_res, scalar_t* __restrict__ after_norm,
    scalar_t* __restrict__ residual = nullptr) {
  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute rms
  compute_rms<scalar_t, has_residual>(
      &rms, input, hidden_size, var_epsilon, after_res, residual);

  // Compute scale
  compute_dynamic_per_token_scales<scalar_t, scalar_out_t,scalar_smooth_t, has_residual>(
      &token_scale, scales, after_res, weight, rms, smooth_scale, hidden_size,
      residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    norm_and_quant<scalar_t, scalar_out_t, scalar_smooth_t, true,
                                     has_residual>(
        out, after_norm, after_res, weight, smooth_scale, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert token_scale for exact match with FBGemm
    norm_and_quant<scalar_t, scalar_out_t,scalar_smooth_t, false,
                                     has_residual>(
        out, after_norm,after_res, weight, smooth_scale,  rms, token_scale, hidden_size, residual);
  }
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_out_t, typename scalar_smooth_t, bool has_residual = false>
__global__ void rms_norm_dynamic_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,       // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_smooth_t const* __restrict__ weight,  // [hidden_size]
    scalar_smooth_t const* smooth_scale, float const var_epsilon, int32_t const hidden_size,
    scalar_t* __restrict__ after_res,
    scalar_t* __restrict__ after_norm,
    scalar_t* __restrict__ residual = nullptr
    ) {
  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  bool const can_vectorize = hidden_size % 4 == 0;

  if (can_vectorize) {
    return rms_norm_dynamic_per_token_quant_vec<scalar_t, scalar_out_t,scalar_smooth_t,
                                                has_residual>(
        out, scales, input, weight, smooth_scale, var_epsilon, hidden_size, after_res, after_norm,
        residual);
  }

  float rms = 0.0f;
  float token_scale = 0.0f;

  // Compute RMS
  compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                            var_epsilon, after_res, residual);
  // Compute Scale
  compute_dynamic_per_token_scales<scalar_t, scalar_out_t, scalar_smooth_t, has_residual>(
      &token_scale, scales, after_res, weight, rms, smooth_scale, hidden_size,
      residual);

  // RMS Norm + Quant
  if constexpr (std::is_same_v<scalar_out_t, int8_t>) {
    norm_and_quant<scalar_t, scalar_out_t,scalar_smooth_t, true, has_residual>(
        out,after_norm, after_res, weight, smooth_scale, rms, 1.0f / token_scale, hidden_size, residual);
  } else {
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    norm_and_quant<scalar_t, scalar_out_t,scalar_smooth_t, false, has_residual>(
        out,after_norm, after_res, weight,smooth_scale, rms, token_scale, hidden_size, residual);
  }
}

template<typename scalar_t, typename scalar_weight_t, typename VT, int N, int NUM_REG, typename VWT, int NUM_THREADS, bool has_residual = false>
__global__ void rms_norm_kernel_align(
  scalar_t* __restrict__ out,
  scalar_t const* __restrict__ input,
  scalar_weight_t const* __restrict__ weight,
  float const var_epsilon,
  int32_t const hidden_size, bool rms_div,
  scalar_t* __restrict__ after_res = nullptr,
  scalar_t* __restrict__ residual = nullptr
  
)
{
  int64_t stride = static_cast<int64_t>(hidden_size);
  float rms = 0;
  int64_t offset = blockIdx.x * stride;
  scalar_t const* ptr_input = input + offset;
  scalar_t* ptr_output = out + offset;
  scalar_t* ptr_residual = nullptr;
  scalar_t* ptr_after_res = nullptr;
  if constexpr(has_residual) {
    ptr_residual = residual + offset;
    ptr_after_res = after_res + offset;
  }
  float reg_input[NUM_REG][N];
  // sum of squares
  float ss = 0.0f;
  int tid = threadIdx.x * N;
  int block_stride = NUM_THREADS * N;
  int k = 0;
  for(int i = tid; i < hidden_size; i += block_stride) {
    VT local = *(VT*)(ptr_input + i);
    scalar_t* ptr_local = (scalar_t*)&local;
    VT reg_residual;
    if constexpr(has_residual) {
      reg_residual = *(VT*)(ptr_residual + i);
    }
    scalar_t* ptr_reg_residual = (scalar_t*)&reg_residual;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
      float x = static_cast<float>(ptr_local[j]);
      if constexpr(has_residual) {
        x += static_cast<float>(ptr_reg_residual[j]);
        ptr_reg_residual[j] = float_to_dstT<scalar_t>(x);
      }
      ss += x * x;
      reg_input[k][j] = x;
    }
    if constexpr(has_residual) {
      *(VT*)(ptr_after_res + i) = reg_residual;
    }
    k++;
  }
  using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, NUM_THREADS);
  __shared__ float s_rms;
  if(threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + var_epsilon);
  }
  __syncthreads();
  rms = s_rms;
  float div_rms = 0.0f;
  if(rms_div){
    div_rms = 1.0 / rms;
  }else{
    div_rms = rms;
  }
  scalar_weight_t const* ptr_weight = weight;
  k = 0;
  for(int i = tid; i < hidden_size; i += block_stride) {
    VWT local_weight = *(VWT*)(ptr_weight + i);
    scalar_weight_t* ptr_local_weight = (scalar_weight_t*)&local_weight;
    VT reg_dst;
    scalar_t *ptr_reg_dst = (scalar_t*)&reg_dst;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
      ptr_reg_dst[j] = float_to_dstT<scalar_t>(reg_input[k][j] * div_rms * convert_to_float<scalar_weight_t>(ptr_local_weight[j]));
    }
    k++;
    *(VT*)(ptr_output + i) = reg_dst;
  }
}

// RMS norm + quant kernel
template <typename scalar_t, typename scalar_weight_t, bool has_residual = false>
__global__ void rms_norm_per_token_kernel(
    scalar_t* __restrict__ out,       // [..., hidden_size]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_weight_t const* __restrict__ weight,  // [hidden_size]
    float const var_epsilon, int32_t const hidden_size, bool rms_div,
    scalar_t* __restrict__ after_res = nullptr,
    scalar_t* __restrict__ residual = nullptr
    
    ) {
  // For vectorization, token_input and token_output pointers need to be
  // aligned at 8-byte and 4-byte addresses respectively.
  float rms = 0.0f;
  float token_scale = 0.0f;
  if constexpr(has_residual) {
    // Compute RMS
    compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                              var_epsilon, after_res, residual);
    // RMS Norm + Quant
    
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    norm_without_smooth<scalar_t, scalar_weight_t>(out, after_res, weight, rms, hidden_size, rms_div, residual);
  } else {
    // Compute RMS
    compute_rms<scalar_t, has_residual>(&rms, input, hidden_size,
                                              var_epsilon, after_res, residual);
    // RMS Norm + Quant
    
    // FP8 - Do not invert s_token_scale for exact match with FBGemm
    norm_without_smooth<scalar_t, scalar_weight_t>(out, input, weight, rms, hidden_size, rms_div, residual);
  }
}

// template<typename scalar_t, typename scalar_weight_t, typename VT, int N, int NUM_REG, typename VWT, int NUM_THREADS>
// __global__ void head_rms_norm_kernel_warp_algin(
//   scalar_t* __restrict__ out,
//   scalar_t const* __restrict__ input,
//   scalar_weight_t const* __restrict__ weight,
//   float const var_epsilon,
//   int32_t const hidden_size,
//   int32_t const head_offset,
//   int32_t const head_num,
//   int32_t const num_tokens
// )
// {
//   constexpr int NUM_WARPS = NUM_THREADS >> 6;
//   int64_t stride = static_cast<int64_t>(hidden_size);
//   int lane_id = threadIdx.x & 63;
//   int warp_id = threadIdx.x >> 6;
//   int token_id = blockIdx.y * NUM_WARPS + warp_id;
//   if(token_id >= num_tokens) return;
//   int offset = (token_id * head_num + head_offset + blockIdx.x) * stride;
//   scalar_t const* ptr_input = input + offset;
//   scalar_t* ptr_output = out + offset;

//   float reg_input[NUM_REG][N];
//   float ss = 0.0f;
//   float rms = 0;
//   int tid = lane_id * N;
//   int block_stride = 64 * N;
//   int k = 0;
//   for(int i = tid; i < hidden_size; i += block_stride) {
//     VT local = *(VT*)(ptr_input + i);
//     scalar_t* ptr_local = (scalar_t*)&local;
//     #pragma unroll N
//     for(int j = 0; j < N; j++) {
//       float x = static_cast<float>(ptr_local[j]);
//       ss += x * x;
//       reg_input[k][j] = x;
//     }
//     k++;
//   }
  
//   for (int mask = 32; mask > 0; mask = mask >> 1) {
//     ss += __shfl_xor_sync(0xffffffffffffffff, ss, mask, 64);
//   }

//   __shared__ float s_rms[NUM_WARPS];
//   if(lane_id == 0) {
//     s_rms[warp_id] = rsqrtf(ss / hidden_size + var_epsilon);
//   }
//   __syncthreads();
//   rms = s_rms[warp_id];
//   float div_rms = 1.0 / rms;

//   scalar_weight_t const* ptr_weight = weight + (head_offset + blockIdx.x) * hidden_size;

//   k = 0;
//   for(int i = tid; i < hidden_size; i += block_stride) {
//     VT reg_dst;
//     scalar_t *ptr_reg_dst = (scalar_t*)&reg_dst;
//     VWT local_weight = *(VWT*)(ptr_weight + i);
//     scalar_weight_t* ptr_local_weight = (scalar_weight_t*)&local_weight;

//     #pragma unroll N
//     for(int j = 0; j < N; j++) {
//       ptr_reg_dst[j] = float_to_dstT<scalar_t>(reg_input[k][j] * div_rms * convert_to_float<scalar_weight_t>(ptr_local_weight[j]));
//     }
//     k++;
    
//     *(VT*)(ptr_output + i) = reg_dst;
//   }
// }

template<typename scalar_t, typename scalar_weight_t, typename VT, int N, int NUM_REG, typename VWT, int NUM_THREADS>
__global__ void head_rms_norm_kernel_align(
  scalar_t* __restrict__ out,
  scalar_t const* __restrict__ input,
  scalar_weight_t const* __restrict__ weight,
  float const var_epsilon,
  int32_t const hidden_size,
  int32_t const head_offset,
  int32_t const head_num,
  bool rms_div = true
)
{
  int64_t stride = static_cast<int64_t>(hidden_size);
  float rms = 0;
  int64_t offset = (blockIdx.y * head_num + head_offset + blockIdx.x) * stride;
  scalar_t const* ptr_input = input + offset;
  scalar_t* ptr_output = out + offset;
  float reg_input[NUM_REG][N];
  // sum of squares
  float ss = 0.0f;
  int tid = threadIdx.x * N;
  int block_stride = NUM_THREADS * N;
  int k = 0;
  for(int i = tid; i < hidden_size; i += block_stride) {
    VT local = *(VT*)(ptr_input + i);
    scalar_t* ptr_local = (scalar_t*)&local;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
      float x = static_cast<float>(ptr_local[j]);
      ss += x * x;
      reg_input[k][j] = x;
    }
    k++;
  }
  using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  ss = BlockReduce(reduceStore).Reduce(ss, cub::Sum{}, NUM_THREADS);
  __shared__ float s_rms;
  if(threadIdx.x == 0) {
    s_rms = rsqrtf(ss / hidden_size + var_epsilon);
  }
  __syncthreads();
  rms = s_rms;
  float div_rms = 0.0f;
  if(rms_div){
    div_rms = 1.0 / rms;
  }else{
    div_rms = rms;
  }
  
  scalar_weight_t const* ptr_weight = weight + (head_offset + blockIdx.x) * hidden_size;
  k = 0;
  for(int i = tid; i < hidden_size; i += block_stride) {
    VWT local_weight = *(VWT*)(ptr_weight + i);
    scalar_weight_t* ptr_local_weight = (scalar_weight_t*)&local_weight;
    VT reg_dst;
    scalar_t *ptr_reg_dst = (scalar_t*)&reg_dst;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
      ptr_reg_dst[j] = float_to_dstT<scalar_t>(reg_input[k][j] * div_rms * convert_to_float<scalar_weight_t>(ptr_local_weight[j]));
    }
    k++;
    *(VT*)(ptr_output + i) = reg_dst;
  }
}

template<typename scalar_t, typename scalar_weight_t>
__global__ void head_rms_norm_kernel(
  scalar_t* __restrict__ out,
  scalar_t const* __restrict__ input,
  scalar_weight_t const* __restrict__ weight,
  float const var_epsilon,
  int32_t const hidden_size,
  int32_t const head_offset,
  int32_t const head_num
)
{
  float rms = 0.0f;
  int64_t stride = static_cast<int64_t>(hidden_size);
  int64_t offset = blockIdx.y * head_num * stride + (head_offset + blockIdx.x) * stride;
  // Compute RMS
  compute_head_rms<scalar_t>(&rms, input + offset, hidden_size,
                                            var_epsilon);
  do_head_norm<scalar_t, scalar_weight_t>(out + offset, input + offset, weight + (head_offset + blockIdx.x) * hidden_size, rms, hidden_size);
}

// Residual add + RMS norm + dynamic per token
template <typename scalar_in_t>
void rms_norm_dynamic_per_token_quant_dispatch(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor const& smooth_scale,
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    torch::Tensor& after_res,
    torch::Tensor& after_norm,
    std::optional<at::Tensor>& residual) {
  int32_t hidden_size = input.size(-1);
  int32_t num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  if(weight.dtype() == torch::kBFloat16) {
    if (residual.has_value()) {
      MOE_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
            rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,bfloat16,true>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                    reinterpret_cast<bfloat16*>(smooth_scale.data_ptr<at::BFloat16>()),
                    var_epsilon, hidden_size, after_res.data_ptr<scalar_in_t>(), after_norm.data_ptr<scalar_in_t>(), residual->data_ptr<scalar_in_t>());
          });
    } else {
      MOE_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
            rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,bfloat16, false>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                    reinterpret_cast<bfloat16*>(smooth_scale.data_ptr<at::BFloat16>()),
                    var_epsilon, hidden_size, after_res.data_ptr<scalar_in_t>(), after_norm.data_ptr<scalar_in_t>(), nullptr);
          });
    }
  } else if(weight.dtype() == torch::kFloat32) {
    if (residual.has_value()) {
      MOE_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
            rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,float,true>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), reinterpret_cast<float*>(weight.data_ptr<float>()),
                    reinterpret_cast<float*>(smooth_scale.data_ptr<float>()),
                    var_epsilon, hidden_size, after_res.data_ptr<scalar_in_t>(), after_norm.data_ptr<scalar_in_t>(), residual->data_ptr<scalar_in_t>());
          });

    } else {
      MOE_DISPATCH_QUANT_TYPES(
          out.scalar_type(), "rms_norm_dynamic_per_token_quant_kernel", [&] {
            rms_norm_dynamic_per_token_quant_kernel<scalar_in_t, scalar_t,float, false>
                <<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(), scales.data_ptr<float>(),
                    input.data_ptr<scalar_in_t>(), reinterpret_cast<float*>(weight.data_ptr<float>()),
                    reinterpret_cast<float*>(smooth_scale.data_ptr<float>()),
                    var_epsilon, hidden_size, after_res.data_ptr<scalar_in_t>(), after_norm.data_ptr<scalar_in_t>(),  nullptr);
          });
    }
  }
}

void rms_norm_dynamic_per_token_quant(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor const& smooth_scale,
    torch::Tensor& scales,        // [num_tokens]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    torch::Tensor& after_res,
    torch::Tensor& after_norm,
    std::optional<at::Tensor> residual) {
	DEBUG_TRACE_PARAMS(out, input, weight, smooth_scale, scales, var_epsilon, after_res, after_norm, residual);
	DEBUG_DUMP_PARAMS(out, input, weight, smooth_scale, scales, var_epsilon, after_res, after_norm, residual);
  TORCH_CHECK(out.dtype() == torch::kInt8);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  TORCH_CHECK(scales.dtype() == torch::kFloat32);

  MOE_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "rms_norm_dynamic_per_token_quant_dispatch", [&] {
      rms_norm_dynamic_per_token_quant_dispatch<scalar_t>(
          out, input, weight, smooth_scale, scales, var_epsilon, after_res, after_norm, residual);
    });
}

template <typename scalar_in_t>
void head_rms_norm_dispatch(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    int head_offset,
    int head_norm
    )
{
  int32_t hidden_size = input.size(-1);
  int32_t num_tokens = input.size(0);
  int32_t head_dim = input.size(1);

  dim3 grid(head_norm, num_tokens, 1);
  dim3 block(std::min(hidden_size, 1024));
  int blocksize = 1024;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int N = 16 / sizeof(scalar_in_t);
  if(weight.dtype() == torch::kBFloat16) {
    if((hidden_size & (N - 1)) == 0) {
      if(hidden_size <= 64 * N) {
        head_rms_norm_kernel_align<scalar_in_t, bfloat16,float4, N, 1, float4, 64><<<grid, 64, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                  var_epsilon, hidden_size, head_offset, head_dim);
        return;
      } else if(hidden_size <= 128 * N) {
        head_rms_norm_kernel_align<scalar_in_t, bfloat16,float4, N, 1, float4, 128><<<grid, 128, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                  var_epsilon, hidden_size, head_offset, head_dim);
        return;
      } else if(hidden_size <= 256 * N) {
        head_rms_norm_kernel_align<scalar_in_t, bfloat16,float4, N, 1, float4, 256><<<grid, 256, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                  var_epsilon, hidden_size, head_offset, head_dim);
        return;
      } else if(hidden_size <= 512 * N) {
        head_rms_norm_kernel_align<scalar_in_t, bfloat16,float4, N, 1, float4, 512><<<grid, 512, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                  var_epsilon, hidden_size, head_offset, head_dim);
        return;
      } else if(hidden_size <= blocksize*N) {
        head_rms_norm_kernel_align<scalar_in_t, bfloat16,float4, N, 1, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                  var_epsilon, hidden_size, head_offset, head_dim);
        return;
      } else if(hidden_size <= 2*blocksize*N) {
        head_rms_norm_kernel_align<scalar_in_t, bfloat16,float4, N, 2, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                  var_epsilon, hidden_size, head_offset, head_dim);
        return;
      } else if(hidden_size <= 3*blocksize*N) {
        head_rms_norm_kernel_align<scalar_in_t, bfloat16,float4, N, 3, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                  var_epsilon, hidden_size, head_offset, head_dim);
        return;
      }
    }
      head_rms_norm_kernel<scalar_in_t, bfloat16>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), reinterpret_cast<bfloat16*>(weight.data_ptr<at::BFloat16>()),
                var_epsilon, hidden_size, head_offset, head_dim);
  } 
  else if(weight.dtype() == torch::kFloat32) 
  {
    if constexpr (N == 4) {
        if((hidden_size & (N - 1)) == 0) {
          if(hidden_size <= 64*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 1, float4, 64><<<grid, 64, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                    var_epsilon, hidden_size, head_offset, head_dim);
            return;
          } else if(hidden_size <= 128*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 1, float4, 128><<<grid, 128, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                    var_epsilon, hidden_size, head_offset, head_dim);
            return;
          } else if(hidden_size <= 256*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 1, float4, 256><<<grid, 256, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                    var_epsilon, hidden_size, head_offset, head_dim);
            return;
          } else if(hidden_size <= 512*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 1, float4, 512><<<grid, 512, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                    var_epsilon, hidden_size, head_offset, head_dim);
            return;
          } else if(hidden_size <= blocksize*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 1, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                    var_epsilon, hidden_size, head_offset, head_dim);
            return;
          } else if(hidden_size <= blocksize*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 1, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
          } else if(hidden_size <= 2*blocksize*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 2, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
          } else if(hidden_size <= 3*blocksize*N) {
            head_rms_norm_kernel_align<scalar_in_t, float,float4, N, 3, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
          }
      } 
    } else if constexpr(N == 8) {
      constexpr int N2 = N / 2;
      if((hidden_size & (N2 - 1)) == 0) {
        if(hidden_size <= 64*N2) {
            head_rms_norm_kernel_align<scalar_in_t, float, float2, N2, 1, float4, 64><<<grid, 64, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
        } else if(hidden_size <= 128*N2) {
            head_rms_norm_kernel_align<scalar_in_t, float, float2, N2, 1, float4, 128><<<grid, 128, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
        } else if(hidden_size <= 256*N2) {
            head_rms_norm_kernel_align<scalar_in_t, float, float2, N2, 1, float4, 256><<<grid, 256, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
        } else if(hidden_size <= 512*N2) {
            head_rms_norm_kernel_align<scalar_in_t, float, float2, N2, 1, float4, 512><<<grid, 512, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
        } else if(hidden_size <= blocksize*N2) {
            head_rms_norm_kernel_align<scalar_in_t, float, float2, N2, 1, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                      var_epsilon, hidden_size, head_offset, head_dim);
            return;
        } else if(hidden_size <= 2*blocksize*N2) {
          head_rms_norm_kernel_align<scalar_in_t, float, float2, N2, 2, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                    var_epsilon, hidden_size, head_offset, head_dim);
          return;
        } else if(hidden_size <= 3*blocksize*N2) {
          head_rms_norm_kernel_align<scalar_in_t, float, float2, N2, 3, float4, 1024><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
                    var_epsilon, hidden_size, head_offset, head_dim);
          return;
        }
      }
    }
    head_rms_norm_kernel<scalar_in_t, float>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_in_t>(), input.data_ptr<scalar_in_t>(), weight.data_ptr<float>(),
              var_epsilon, hidden_size, head_offset, head_dim);
  }
}

void head_rms_norm(torch::Tensor& out, torch::Tensor const& hidden_states, torch::Tensor const &weight, double const var_epsilon, int head_offset, int head_norm)
{
    DEBUG_TRACE_PARAMS(out, hidden_states, weight, var_epsilon, head_offset, head_norm);
    DEBUG_TRACE_PARAMS(out, hidden_states, weight, var_epsilon, head_offset, head_norm);
    TORCH_CHECK(out.is_contiguous() && weight.is_contiguous() && hidden_states.is_contiguous());
    MOE_DISPATCH_FLOATING_TYPES(hidden_states.scalar_type(), "head_rms_norm_dispatch", [&]{
      head_rms_norm_dispatch<scalar_t>(out, hidden_states, weight, var_epsilon, head_offset, head_norm);
    });
}

template <typename scalar_t>
void rms_norm_per_token_dispatch(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor>& after_res,
    std::optional<at::Tensor>& residual, bool rms_div = true) {
  int32_t hidden_size = input.size(-1);
  int32_t num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  int blocksize = 1024;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int N = 16 / sizeof(scalar_t);
  if(weight.dtype() == torch::kFloat32) {
      if (residual.has_value()) {
        if constexpr (N == 4) {
          if((hidden_size & (N - 1)) == 0) {
          if(hidden_size <= blocksize*N) {
            rms_norm_kernel_align<scalar_t, float,float4, N, 1, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          } else if(hidden_size <= 2*blocksize*N) {
            rms_norm_kernel_align<scalar_t, float,float4, N, 2, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          } else if(hidden_size <= 3*blocksize*N) {
            rms_norm_kernel_align<scalar_t, float,float4, N, 3, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          }
        } 
      } else if constexpr(N == 8) {
        constexpr int N2 = N / 2;
        if((hidden_size & (N2 - 1)) == 0) {
          if(hidden_size <= blocksize*N2) {
              rms_norm_kernel_align<scalar_t, float, float2, N2, 1, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
              return;
          } else if(hidden_size <= 2*blocksize*N2) {
            rms_norm_kernel_align<scalar_t, float, float2, N2, 2, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          } else if(hidden_size <= 3*blocksize*N2) {
            rms_norm_kernel_align<scalar_t, float, float2, N2, 3, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          }
        }
      }

      rms_norm_per_token_kernel<scalar_t, float, true>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());

    } else {
      if constexpr (N == 4) {
          if((hidden_size & (N - 1)) == 0) {
          if(hidden_size <= blocksize*N) {
            rms_norm_kernel_align<scalar_t, float,float4, N, 1, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
            return;
          } else if(hidden_size <= 2*blocksize*N) {
            rms_norm_kernel_align<scalar_t, float,float4, N, 2, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
            return;
          } else if(hidden_size <= 3*blocksize*N) {
            rms_norm_kernel_align<scalar_t, float,float4, N, 3, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
            return;
          }
        } 
      } else if constexpr(N == 8) {
        constexpr int N2 = N / 2;
        if((hidden_size & (N2 - 1)) == 0) {
          if(hidden_size <= blocksize*N2) {
              rms_norm_kernel_align<scalar_t, float, float2, N2, 1, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
              return;
          } else if(hidden_size <= 2*blocksize*N2) {
            rms_norm_kernel_align<scalar_t, float, float2, N2, 2, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
            return;
          } else if(hidden_size <= 3*blocksize*N2) {
            rms_norm_kernel_align<scalar_t, float, float2, N2, 3, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
            return;
          }
        }
      }
      rms_norm_per_token_kernel<scalar_t, float, false>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), reinterpret_cast<const float*>(weight.data_ptr<float>()),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
    }
  } else {
    if (residual.has_value()) {
      if((hidden_size & (N - 1)) == 0) {
          if(hidden_size <= blocksize*N) {
            rms_norm_kernel_align<scalar_t, scalar_t,float4, N, 1, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          } else if(hidden_size <= 2*blocksize*N) {
            rms_norm_kernel_align<scalar_t, scalar_t,float4, N, 2, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          } else if(hidden_size <= 3*blocksize*N) {
            rms_norm_kernel_align<scalar_t, scalar_t,float4, N, 3, float4, 1024, true><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());
            return;
          }
      }   
      rms_norm_per_token_kernel<scalar_t, scalar_t, true>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size, rms_div, after_res->data_ptr<scalar_t>(), residual->data_ptr<scalar_t>());

    } else {
      if((hidden_size & (N - 1)) == 0) {
          if(hidden_size <= blocksize*N) {
            rms_norm_kernel_align<scalar_t, scalar_t,float4, N, 1, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
            return;
          } else if(hidden_size <= 2*blocksize*N) {
            rms_norm_kernel_align<scalar_t, scalar_t,float4, N, 2, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
            return;
          } else if(hidden_size <= 3*blocksize*N) {
            rms_norm_kernel_align<scalar_t, scalar_t,float4, N, 3, float4, 1024, false><<<grid, blocksize, 0, stream>>>(out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size,rms_div,  nullptr, nullptr);
            return;
          }
      }
      rms_norm_per_token_kernel<scalar_t, scalar_t, false>
          <<<grid, block, 0, stream>>>(
              out.data_ptr<scalar_t>(),
              input.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
              var_epsilon, hidden_size, rms_div, nullptr, nullptr);
    }
  }
}

void rms_norm(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    double const var_epsilon,     // Variance epsilon used in norm calculation
    std::optional<at::Tensor> after_res,
    std::optional<at::Tensor> residual,
    bool rms_div
)
{
  DEBUG_TRACE_PARAMS(out, input, weight, var_epsilon, after_res, residual, rms_div);
	DEBUG_DUMP_PARAMS(out, input, weight, var_epsilon, after_res, residual, rms_div);
  TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  TORCH_CHECK(weight.is_contiguous());
  MOE_DISPATCH_FLOATING_TYPES(
    input.scalar_type(), "rms_norm_per_token_dispatch", [&] {
      rms_norm_per_token_dispatch<scalar_t>(
          out, input, weight, var_epsilon, after_res, residual, rms_div);
    });
}