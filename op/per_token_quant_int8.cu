// 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#include "../kernel/dispatch_utils.h"
#include <cub/cub.cuh>
#include <cub/util_type.cuh>

#include "../include/int8_quant_kernel.h"
typedef __NATIVE_VECTOR__(4, float) v4f32;
typedef __NATIVE_VECTOR__(4, _Float16) v4f16;

static __forceinline__ __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static constexpr auto i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static constexpr auto i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());

  // To match the rounding mode of CUDA, we use nearbyint.
  // It uses the current rounding mode, which is always FE_TONEAREST on HIP.
  // If that changes in the future, we may need to set the rounding mode
  // explicitly, either at runtime or compile time.
  float dst = std::nearbyint(x);

  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  //uint32_t dst;
  //asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  //return reinterpret_cast<const int8_t&>(dst);
  int32_t dst;
  dst = __float2int_rn(x);
  dst = min(dst, 127);
  dst = max(dst, -127);
  return reinterpret_cast<const int8_t&>(dst);
#endif
}


template <typename scalar_t, typename scale_type, typename VT, typename VT1, int NUM_THREADS, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_sreg_opt(
    scalar_t const* __restrict__ input, scalar_t* __restrict__ out,
    const int hidden_size, int num_tokens, int out_stride, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  float reg_src0[N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int reg_length = NUM_THREADS * N;
  int length = min(hidden_size, reg_length);
  int index = tid * N;
  if(index < length) {
    VT reg_src;
    reg_src = *(VT*)(ptr_input + index);
    scalar_t* ptr_reg_src = (scalar_t*)&reg_src;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      reg_src0[i] = (float)ptr_reg_src[i];
    }
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      float val = abs(reg_src0[i]);
      absmax_val = max(absmax_val, val);
    }
  }

  // using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  // __shared__ typename BlockReduce::TempStorage reduceStorage;
  // float const block_absmax_val_maybe =
  //     BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);
  // __shared__ float block_absmax_val;
  constexpr int sm_size = NUM_THREADS >> 4;
  constexpr int sm_size2 = sm_size / 2;

  __shared__ float sm_max[sm_size];
  float block_absmax_val;
  if constexpr(sm_size == 64) {
    for(int i = 8; i > 0; i >>= 1) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i), absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    __shared__ float sm_max2[sm_size>>4];
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 8; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      int local_group_id = threadIdx.x >> 4;
      int local_lane_id = threadIdx.x & 15;
      if(local_lane_id == 0) {
        sm_max2[local_group_id] = data;
      }
    }
    __syncthreads();
    block_absmax_val = max(sm_max2[0], sm_max2[1]);
    block_absmax_val = max(block_absmax_val, sm_max2[2]);
    block_absmax_val = max(block_absmax_val, sm_max2[3]);
  } else if constexpr (sm_size == 32) {
    for(int i = 8; i > 0; i >>= 1) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i), absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    __shared__ float sm_max2[sm_size>>4];
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 8; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      int local_group_id = threadIdx.x >> 4;
      int local_lane_id = threadIdx.x & 15;
      if(local_lane_id == 0) {
        sm_max2[local_group_id] = data;
      }
    }
    __syncthreads();
    block_absmax_val = max(sm_max2[0], sm_max2[1]);
  } else if constexpr(sm_size == 16) {
    for(int i = 8; i > 0; i >>=1 ) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i),absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 8; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      if(threadIdx.x == 0) {
        sm_max[0] = data;
      }
    }
    __syncthreads();
    block_absmax_val = sm_max[0];
  } else if constexpr(sm_size == 8) {
    for(int i = 8; i > 0; i >>=1 ) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i) , absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 4; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      if(threadIdx.x == 0) {
        sm_max[0] = data;
      }
    }
    __syncthreads();
    block_absmax_val = sm_max[0];
  } else if constexpr(sm_size == 4) {
    for(int i = 8; i > 0; i >>=1 ) {
      absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i), absmax_val);
    }
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if(lane_id == 0) {
      sm_max[group_id] = absmax_val;
    }
    __syncthreads();
    if(threadIdx.x < sm_size) {
      float data = sm_max[threadIdx.x];
      for(int i = 2; i >= 1; i >>= 1) {
        data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
      }
      if(threadIdx.x == 0) {
        sm_max[0] = data;
      }
    }
    __syncthreads();
    block_absmax_val = sm_max[0];
  }
  int8_t* ptr_output = (int8_t*)(out + token_idx * out_stride);
  scale_type* scale = (scale_type*)(ptr_output + hidden_size);
  if (tid == 0) {
    // block_absmax_val = block_absmax_val_maybe;
    scale[0] = static_cast<scale_type>(block_absmax_val * 0.0078740157);
  }
  __syncthreads();
  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
  
  if(index < length) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(reg_src0[i] * tmp_scale);
    }
    *(VT1*)(ptr_output + index) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_reg_opt(
    scalar_t const* __restrict__ input, scalar_t* __restrict__ out,
    const int hidden_size, int blockDim_x, int num_tokens, int out_stride, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  float reg_src0[N];
  float reg_src1[N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  int reg_length = 2 * blockDim_x * N;
  int length = min(hidden_size, reg_length);
  int index = 2 * tid * N;
  if(index < length) {
    VT reg_src = *(VT*)(ptr_input + index);
    scalar_t* ptr_reg_src = (scalar_t*)&reg_src;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      reg_src0[i] = (float)ptr_reg_src[i];
    }
    reg_src = *(VT*)(ptr_input + index + N);
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      reg_src1[i] = (float)ptr_reg_src[i];
    }
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      float val = abs(reg_src0[i]);
      absmax_val =  max(val, absmax_val);
      val = abs(reg_src1[i]);
      absmax_val = max(val, absmax_val);
    }
  }

  // using BlockReduce = cub::BlockReduce<float, 512>;
  // __shared__ typename BlockReduce::TempStorage reduceStorage;
  // float const block_absmax_val_maybe =
  //     BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
  // __shared__ float block_absmax_val;
  __shared__ float sm_max[32];
  __shared__ float sm_max2[2];
  for(int i = 8; i >= 1; i >>= 1){
    absmax_val = max(__shfl_down_sync_16(0xffffffffffffffff, absmax_val, i) , absmax_val);
  }
  int lane_id = threadIdx.x & 15;
  int group_id = threadIdx.x >> 4;
  if(lane_id == 0) {
    sm_max[group_id] = absmax_val;
  }
  __syncthreads();
  if(threadIdx.x < 32) {
    float data = sm_max[threadIdx.x];
    for(int i = 8; i >= 1; i>>=1) {
      data = max(__shfl_down_sync_16(0xffffffffffffffff, data, i), data);
    }
    int local_group_id = threadIdx.x >> 4;
    int local_lane_id = threadIdx.x & 15;
    if(local_lane_id == 0) {
      sm_max2[local_group_id] = data;
    }
  }
  __syncthreads();
  float block_absmax_val = max(sm_max2[0], sm_max2[1]);

  int8_t* ptr_output = (int8_t*)(out + token_idx * out_stride);
  scale_type* scale = (scale_type*)(ptr_output + hidden_size);
  if (tid == 0) {
    // block_absmax_val = (block_absmax_val_maybe);
    scale[0] = static_cast<scale_type>(block_absmax_val * 0.0078740157);
  }
  // __syncthreads();
  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
  if(index < length) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(
             reg_src0[i] * tmp_scale);
    }
    ptr_reg = ptr_reg + N;
    #pragma unroll N
    for(int i = 0; i < N; i++) {
      ptr_reg[i] = float_to_int8_rn(
            reg_src1[i] * tmp_scale);
    }
    *(VT1*)(ptr_output + index) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel_sm_opt(
    scalar_t const* __restrict__ input, scalar_t* __restrict__ out,
    const int hidden_size, int blockDim_x, int num_tokens, int out_stride, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int stride = blockDim_x * N;
  __shared__ float sm_buffer[8064];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  for(int i = tid * N; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    float* ptr_sm_buffer = sm_buffer + i;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        ptr_sm_buffer[j] = val;
        val = abs(val);
        absmax_val = max(val, absmax_val);
    }
  }
  using BlockReduce = cub::BlockReduce<float, 512>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  int8_t* ptr_output = (int8_t*)(out + token_idx * out_stride);
  scale_type* scale = (scale_type*)(ptr_output + hidden_size);
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[0] = block_absmax_val * 0.0078740157;
  }
  
  __syncthreads();

  float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
  
  for(int i = tid * N; i < hidden_size; i += stride) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    float* ptr_sm_buffer = sm_buffer + i;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        ptr_reg[j] = float_to_int8_rn(
            ptr_sm_buffer[j] * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, int NUM_REG, int NUM_THREADS, bool WITHMASK>
__global__ __launch_bounds__(1024) void dynamic_scaled_int8_quant_kernel_lh_opt(
    scalar_t const* __restrict__ input, scalar_t* __restrict__ out,
    const int hidden_size, int num_tokens, int out_stride, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int stride = NUM_THREADS * N;
  float reg_src[NUM_REG][N];
  scalar_t const* ptr_input = input + token_idx * hidden_size;
  for(int i = tid * N, k = 0; i < hidden_size; i += stride, k++) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        reg_src[k][j] = val;
        val = abs(val);
        absmax_val = max(val, absmax_val);
    }
  }
  using BlockReduce = cub::BlockReduce<float, NUM_THREADS>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, NUM_THREADS);
  __shared__ float block_absmax_val;
  int8_t* ptr_output = (int8_t*)(out + token_idx * out_stride);
  scale_type* scale = (scale_type*)(ptr_output + hidden_size);
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[0] = block_absmax_val * 0.0078740157;
  }
  
  __syncthreads();

  float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);

  for(int i = tid * N, k = 0; i < hidden_size; i += stride, k++) {
    VT1 vdst;
    int8_t* ptr_reg = (int8_t*)&vdst;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        ptr_reg[j] = float_to_int8_rn(
            reg_src[k][j] * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, typename VT, typename VT1, bool WITHMASK>
__launch_bounds__(1024) __global__ void dynamic_scaled_int8_quant_kernel_opt(
    scalar_t const* __restrict__ input, scalar_t* __restrict__ out,
    const int hidden_size, const int blockDim_x, const int num_tokens, int out_stride, int* mask_buffer=NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  constexpr int N = sizeof(VT) / sizeof(scalar_t);
  int const tid = threadIdx.x * N;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  int stride = blockDim_x * N;
  const scalar_t * ptr_input = input + token_idx * hidden_size;

  for (int i = tid ; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    #pragma unroll N
    for(int j = 0; j < N; j++) {
        float val = static_cast<float>(ptr_src[j]);
        val = abs(val);
        absmax_val = max(val, absmax_val);
    }
  }

    using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  int8_t* ptr_output = (int8_t*)(out + token_idx * out_stride);
  scale_type* scale = (scale_type*)(ptr_output + hidden_size);

  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[0] = block_absmax_val * 0.0078740157;
  }
  __syncthreads();

  float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
  for (int i = tid; i < hidden_size; i += stride) {
    VT vsrc = *(VT*)(ptr_input + i);
    VT1 vdst;
    scalar_t *ptr_src = (scalar_t*)&vsrc;
    int8_t* ptr_dst = (int8_t*)&vdst;
    #pragma unroll N
    for(int j = 0; j < N; ++j) {
        ptr_dst[j] = float_to_int8_rn(
        static_cast<float>(ptr_src[j]) * tmp_scale);
    }
    *(VT1*)(ptr_output + i) = vdst;
  }
}

template <typename scalar_t, typename scale_type, bool WITHMASK>
__global__ void dynamic_scaled_int8_quant_kernel(
    scalar_t const* __restrict__ input, scalar_t* __restrict__ out,
    const int hidden_size, const int num_tokens, const int out_stride, int *mask_buffer = NULL) {
  if constexpr(WITHMASK) {
    __shared__ int sm_max_token;
    if(threadIdx.x == 0) sm_max_token = mask_buffer[blockIdx.y]; 
    __syncthreads();
    if(blockIdx.x >= sm_max_token) return;
  }
  int const tid = threadIdx.x;
  int64_t const token_idx = blockIdx.y * num_tokens + blockIdx.x;
  float absmax_val = 0.0f;
  float const zero = 0.0f;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    float val = static_cast<float>(input[token_idx * hidden_size + i]);
    val = val > zero ? val : -val;
    absmax_val = val > absmax_val ? val : absmax_val;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStorage;
  float const block_absmax_val_maybe =
      BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim.x);
  __shared__ float block_absmax_val;
  int8_t* ptr_output = (int8_t*)(out + token_idx * out_stride);
  scale_type* scale = (scale_type*)(ptr_output + hidden_size);
  if (tid == 0) {
    block_absmax_val = block_absmax_val_maybe;
    scale[0] = block_absmax_val * 0.0078740157;
  }
  __syncthreads();

  float const tmp_scale = 127.0f *__builtin_mxc_rcpf(block_absmax_val);
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    ptr_output[i] = float_to_int8_rn(
        static_cast<float>(input[token_idx * hidden_size + i]) * tmp_scale);
  }
}

void per_token_quant_int8_pack(
    at::Tensor& out,          // [..., hidden_size]
    at::Tensor const& input  // [..., hidden_size]
    )
{
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(input.element_size() == 2, "input type only support bf16/fp16");
  if(input.numel() == 0) return;
  int const hidden_size = input.size(-1);
  int const num_tokens = input.numel() / hidden_size;
  int out_stride = (hidden_size / 2 + 257) / 256 * 256;
  dim3 const grid(num_tokens,1,1);
  dim3 const block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  MOE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "dynamic_scaled_int8_quant_kernel", [&] {
          int n = 16 / sizeof(scalar_t);
          if(hidden_size <= 4096) {
            if (num_tokens <= 32) {
              n = 8 / sizeof(scalar_t);
              if (((hidden_size & (n - 1)) == 0) && n == 4) {
                if(hidden_size > 512 * n) {
                  dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float2, float, 1024, false><<<grid, 1024, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
                } else if(hidden_size > 256 * n) {
                  dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float2, float, 512, false><<<grid, 512, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
                } else if(hidden_size > 128 * n) {
                  dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float2, float, 256, false><<<grid, 256, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
                } else {
                  dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float2, float, 128, false><<<grid, 128, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
                }
              }
            } else if (((hidden_size & (n - 1)) == 0) && n == 8) {
              if(hidden_size > 256 * n) {
                dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 512, false><<<grid, 512, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(), hidden_size, num_tokens, out_stride);
              } else if(hidden_size > 128 * n) {
                dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 256, false><<<grid, 256, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
              } else if(hidden_size > 64 * n) {
                dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 128, false><<<grid, 128, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
              } else {
                dynamic_scaled_int8_quant_kernel_sreg_opt<scalar_t, float, float4, float2, 64, false><<<grid, 64, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
              }
            }
          } else if(hidden_size > 4096 &&hidden_size <= 8192 && ((hidden_size & (2*n - 1)) == 0) && n == 8) {
            int blocksize = 512;
            dynamic_scaled_int8_quant_kernel_reg_opt<scalar_t, float, float4, float4, false><<<grid, blocksize, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size,blocksize, num_tokens, out_stride);
          } else if(hidden_size <= 8064 && (hidden_size & (n - 1)) == 0 && n == 8) {
            int blocksize = 512;
            dynamic_scaled_int8_quant_kernel_sm_opt<scalar_t, float, float4, float2,false><<<grid, blocksize, 0, stream>>>(
              input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size,blocksize,num_tokens, out_stride);
          } else if(hidden_size >= 16384 && hidden_size <= 18432 && (hidden_size & (n - 1)) == 0 && n == 8) {
            dynamic_scaled_int8_quant_kernel_lh_opt<scalar_t, float, float4, float2, 3,1024, false><<<grid, 1024, 0, stream>>>(input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size, num_tokens, out_stride);
          } else if (hidden_size > 8064 && ((hidden_size & (n - 1)) == 0 && n == 8)) {
            int blocksize = 1024;
            dynamic_scaled_int8_quant_kernel_opt<scalar_t, float,float4,float2, false>
                    <<<grid, blocksize, 0, stream>>>(
                        input.data_ptr<scalar_t>(),out.data_ptr<scalar_t>(),hidden_size,blocksize, num_tokens, out_stride);
          } else {
              dynamic_scaled_int8_quant_kernel<scalar_t, float, false>
                  <<<grid, block, 0, stream>>>(
                      input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
                      hidden_size, num_tokens, out_stride);
          }
      });
}
