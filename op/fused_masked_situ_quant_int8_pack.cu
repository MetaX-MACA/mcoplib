#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>

#include "../kernel/dispatch_utils.h"
#include <cub/cub.cuh>
#include <cub/util_type.cuh>
#include "../include/int8_quant_kernel.h"

typedef __NATIVE_VECTOR__(4, float) v4f32;
typedef __NATIVE_VECTOR__(4, _Float16) v4f16;

template <bool kHasLinearBeta>
__device__ float situ_activate(float g, float u, float beta, float inv_beta, float linear_beta, float inv_linear_beta) {
  const float gate_out = beta * tanhf(g * inv_beta)  * __builtin_mxc_rcpf(1.0f + __builtin_expf(-g));
  float up_out;
  if constexpr (kHasLinearBeta) {
    up_out = linear_beta * tanhf(u * inv_linear_beta);
  } else {
    up_out = u;
  }
  return gate_out * up_out;
}

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

template<typename T, typename T1, typename VT, typename VT1, int NUM_VT, int type, bool kHasLinearBeta, int NUM_THREADS> 
__global__ void situ_mul_mask_quant_pack(T* input, T* output,T1* mask, int mask_size, int64_t grid_size, int64_t num_tokens, 
    int64_t hidden_size, int64_t out_stirde, float beta, float inv_beta, float linear_beta, float inv_linear_beta)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;
    __shared__ T1 sm_mask[1024];
    __shared__ T1 sm_stride[1024];
    if(tid < mask_size) {
        sm_mask[tid] = mask[tid];
    }

    int64_t hidden_size2 = hidden_size << 1;
    __syncthreads();
    if(tid < mask_size) {
        T1 tmp = 0;
        for(int i = 0; i < tid; i++) {
        tmp += sm_mask[i];
        }
        sm_stride[tid] = tmp;
    }
    int stride = NUM_THREADS * N;
    __syncthreads();
    int64_t total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];

    for(int64_t idx = blockIdx.x; idx < total_tokens; idx += grid_size) {
        float reg_i[NUM_VT][N];
        int64_t token_id = mask_size - 1;
        while(idx < sm_stride[token_id]) {
            token_id--;
        }
        int64_t const token_idx = token_id * num_tokens + idx - sm_stride[token_id];
        const T* ptr_input0 = input + token_idx * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        float absmax_val = 0.0f;
        for(int i = tid*N, j = 0; i < hidden_size; i += stride, j++) {
            VT vsrc0, vsrc1;
            vsrc0 = *(VT*)(ptr_input0 + i);
            vsrc1 = *(VT*)(ptr_input1 + i);
            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;
            
            #pragma unroll N
            for(int k = 0; k < N; k++) {
                float val0 = static_cast<float>(ptr_local0[k]);
                float val1 = static_cast<float>(ptr_local1[k]);

                reg_i[j][k] = situ_activate<kHasLinearBeta>(val0, val1, beta, inv_beta, linear_beta, inv_linear_beta);
                absmax_val = max(absmax_val, abs(reg_i[j][k]));
            }
        }

        // using BlockReduce = cub::BlockReduce<float, 512>;
        // __shared__ typename BlockReduce::TempStorage reduceStorage;
        // float const block_absmax_val_maybe =
        //     BlockReduce(reduceStorage).Reduce(absmax_val, cub::Max{}, blockDim_x);
        // __shared__ float block_absmax_val;
        constexpr int sm_size = NUM_THREADS >> 4;
        constexpr int sm_size2 = sm_size / 2;

        __shared__ float sm_max[sm_size];
        float block_absmax_val;
        if constexpr (sm_size == 32) {
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

        int8_t* ptr_output = (int8_t*)(output + token_idx * out_stirde);
        float* ptr_scale = (float*)(ptr_output + hidden_size);
        if constexpr(type == 0) {
            if (tid == 0) {
              // block_absmax_val = block_absmax_val_maybe;
              ptr_scale[0] = block_absmax_val * 0.002232142857;
            }
            // __syncthreads();
            float const tmp_scale = 448.0f * __builtin_mxc_rcpf(block_absmax_val);
            for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
              VT1 vdst;
              uint32_t* ptr_reg_dst = (uint32_t*)&vdst;
              #pragma unroll
              for(int j = 0; j < N; j += 4) {
                v4f16 reg_tmp;
                #pragma unroll 4
                for(int t = 0; t < 4; t++) {
                  reg_tmp[t] = _Float16(reg_i[k][j + t] * tmp_scale);
                }
                *ptr_reg_dst++ = __builtin_mxc_cvt_pk4_f16tof8(reg_tmp);
              }
              *(VT1*)(ptr_output + i) = vdst;
          }
        } else {
          if (tid == 0) {
              // block_absmax_val = block_absmax_val_maybe;
              ptr_scale[0] = block_absmax_val * 0.0078740157;
          }
          // __syncthreads();
          float const tmp_scale = 127.0f * __builtin_mxc_rcpf(block_absmax_val);
          for (int i = tid*N, k = 0; i < hidden_size; i += stride, k++) {
              VT1 vdst;
              int8_t* ptr_dst = (int8_t*)&vdst;
              #pragma unroll N
              for(int j = 0; j < N; ++j) {
                  ptr_dst[j] = float_to_int8_rn(reg_i[k][j] * tmp_scale);
              }
              *(VT1*)(ptr_output + i) = vdst;
          } 
        }
    }
}

template<typename T, typename T1, int type>
void launch_situ_mul_mask_quant_pack(T* input, T* output, T1* mask, bool has_linear_beta, float beta,
     float inv_beta, float linear_beta, float inv_linear_beta, int64_t num_tokens, int64_t hidden_size, int64_t out_stride, int64_t mask_size,cudaStream_t stream) {
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int gridsize = sm_count*4;
    int64_t inner_hidden_size = hidden_size / 2;
    int blocksize = 512;
    int N = sizeof(float4) / sizeof(T);
   if(N == 8&&(inner_hidden_size & (N - 1)) == 0 && (out_stride & (N -1)) == 0) {
        int base = blocksize * N;
        if(inner_hidden_size <= 64 * N) {
            constexpr int NUM_THREADS = 64;
            gridsize = gridsize * 8;
            if(has_linear_beta) {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, true, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            } else {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, false, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            }
        } else if(inner_hidden_size <= 128 * N) {
            constexpr int NUM_THREADS = 128;
            gridsize = gridsize * 4;
            if(has_linear_beta) {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, true, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride,  beta, inv_beta, linear_beta, inv_linear_beta);
            } else {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, false, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            }
        } else if(inner_hidden_size <= 256 * N) {
            constexpr int NUM_THREADS = 256;
            gridsize = gridsize * 2;
            if(has_linear_beta) {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, true, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            } else {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, false, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            }
        } else if(inner_hidden_size <= base) {
            constexpr int NUM_THREADS = 512;
            gridsize = gridsize * 2;
            if(has_linear_beta) {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, true, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            } else {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 1, type, false, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            }
        } else if(inner_hidden_size <= base*2) {
            constexpr int NUM_THREADS = 512;
            gridsize = gridsize * 2;
            if(has_linear_beta) {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 2, type, true, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            } else {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 2, type, false, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            }
        } else if(inner_hidden_size <= base * 3) {
            constexpr int NUM_THREADS = 512;
            gridsize = gridsize * 2;
            if(has_linear_beta) {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 3, type, true, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            } else {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 3, type, false, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            }
        } else if(inner_hidden_size <= base * 4) {
            constexpr int NUM_THREADS = 512;
            gridsize = gridsize * 2;
            if(has_linear_beta) {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 4, type, true, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            } else {
                situ_mul_mask_quant_pack<T, T1, float4, float2, 4, type, false, NUM_THREADS><<<gridsize, NUM_THREADS,0,stream>>>(input, 
                    output, mask, mask_size, gridsize, num_tokens, inner_hidden_size, out_stride, beta, inv_beta, linear_beta, inv_linear_beta);
            }
        } else {
            TORCH_CHECK(false, "situ_mul_mask_quant_pack: inner_hidden_size ", inner_hidden_size,
                        " exceeds max supported (base*4=", base * 4, ", mask_size=", mask_size,
                        ", sizeof(T1)=", sizeof(T1), ", type=", type, ")");
        }
    } else {
        TORCH_CHECK(false, "launch_situ_mul_mask_quant_pack: unsupported configuration: inner_hidden_size=",
                    inner_hidden_size, ", N=", N, ", mask_size=", mask_size,
                    ", out_stride=", out_stride, ", sizeof(T1)=", sizeof(T1), ", type=", type,
                    " (requires N==8 and inner_hidden_size%N==0 and out_stride%N==0)");
    }
}


void fused_situ_mul_dq_mask_quant_pack(
    torch::Tensor& out,
    torch::Tensor const& input,
    torch::Tensor const& mask,
    float beta,
    float linear_beta,
    bool has_linear_beta
)
{
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(out.is_contiguous());
    TORCH_CHECK(mask.is_contiguous());
    int64_t const hidden_size = input.size(-1);
    int64_t const num_tokens = input.numel() / hidden_size;
    int64_t const mask_size = mask.numel();
    int64_t const num_tokens_batch = num_tokens / mask_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int64_t out_stride = ((hidden_size/4 + 2) + 255)/ 256 * 256;

    TORCH_CHECK(mask_size <= 1024 && mask_size >= 0, "mask_size must less than or equal 1024 and must large or equal zero");
    TORCH_CHECK(input.element_size() == 2, "only support fp16 or bf16");
    TORCH_CHECK((hidden_size &1) == 0, "hiddensize must can be divided by 2");
    TORCH_CHECK(((hidden_size /2) & 7) == 0, "half hiddensize must can be diveded by 8");
    TORCH_CHECK(mask.element_size() == 8 || mask.element_size() == 4, "mask tensor only support int/int64_t type");
    TORCH_CHECK(hidden_size <= 32768, "hidden_size only support less than or equal 32768");
    float inv_beta = beta == 0? 0 : 1.0f / beta;
    float inv_linear_beta = linear_beta == 0 ? 0 : 1.0f / linear_beta;
    
    switch(mask.element_size()) {
    case 8:
        MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "launch_situ_mul_mask_quant_pack", [&] {
        launch_situ_mul_mask_quant_pack<scalar_t, int64_t, 1>(input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), mask.data_ptr<int64_t>(), has_linear_beta, beta,inv_beta,linear_beta, inv_linear_beta, 
            num_tokens_batch, hidden_size, out_stride, mask_size, stream);
        });
    break;
    case 4:
        MOE_DISPATCH_FLOATING_TYPES(input.scalar_type(), "launch_situ_mul_mask_quant_pack", [&] {
        launch_situ_mul_mask_quant_pack<scalar_t, int32_t, 1>(input.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), mask.data_ptr<int32_t>(), has_linear_beta, beta,inv_beta,linear_beta, inv_linear_beta,
            num_tokens_batch, hidden_size, out_stride, mask_size, stream);
        });
        break;
    default:
        TORCH_CHECK(false, "Unsupported mask element size: expected 4 (int32) or 8 (int64), got ", mask.element_size());
    }
}
