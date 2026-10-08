#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>
#include "../kernel/dispatch_utils.h"
#include <maca_fp8.h>
#include "../include/fp8_quant_kernel.h"

typedef __NATIVE_VECTOR__(4, _Float16) v4f16;


template<typename T, typename T1, typename VT, typename VT1, bool kApplySwigluLimit>
__global__ void silu_and_mul_mask_quant_nopack(T* input, __maca_fp8_e4m3* output, float* output_scale, T1* mask, int mask_size, int64_t num_tokens,
    int64_t hidden_size, float swiglu_limit)
{
    constexpr int N = sizeof(VT) / sizeof(T);
    int const tid = threadIdx.x;

    __shared__ T1 sm_mask[1024];
    __shared__ T1 sm_stride[1024];
    for(int i = tid; i < mask_size; i += blockDim.x) {
        sm_mask[i] = mask[i];
    }
    __syncthreads();

    for(int i = tid; i < mask_size; i += blockDim.x) {
        if (i < mask_size) {
            T1 tmp = 0;
            for (int j = 0; j < i; ++j) {
                tmp += sm_mask[j];
            }
            sm_stride[i] = tmp;
        }
    }

    __syncthreads();

    int64_t hidden_size2 = hidden_size << 1;
    int lane_id = threadIdx.x & 15;
    int num_lane_id = threadIdx.x >> 4;
    int64_t total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];
    int out_scale_stride = hidden_size >> 7;
    __shared__ float sm_max[32];
    for (int64_t idx = blockIdx.y; idx < total_tokens; idx += gridDim.y) {
        int64_t token_id = mask_size - 1;
        while (token_id > 0 && idx < sm_stride[token_id]) {
            token_id--;
        }
        int64_t token_idx = token_id * num_tokens + idx - sm_stride[token_id];

        const T* ptr_input0 = input + token_idx * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        float* ptr_out_scale = output_scale +  token_idx * out_scale_stride;
        float absmax_val = 0.f;

        int offset_x = (blockIdx.x * blockDim.x + threadIdx.x) * N;
        float reg_i[N];
        if(offset_x < hidden_size) {
            VT vsrc0 = *(VT*)(ptr_input0 + offset_x);
            VT vsrc1 = *(VT*)(ptr_input1 + offset_x);

            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                float val0 = static_cast<float>(ptr_local0[k]);
                float val1 = static_cast<float>(ptr_local1[k]);
                
                if constexpr(kApplySwigluLimit) {
                    val0 = min(val0, swiglu_limit);
                    val1 = max(val1, -swiglu_limit);
                    val1 = min(val1, swiglu_limit);
                }

                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid;

                reg_i[k] = gate_up;
                absmax_val = max(absmax_val, fabsf(gate_up));
            }
        }

        for (int offset = 8; offset >= 1; offset >>= 1) {
            absmax_val = max(absmax_val, __shfl_down_sync_16(0xffffffff, absmax_val, offset));
        }
        float block_absmax = absmax_val;
        block_absmax = max(block_absmax, 1e-10f);
        
        if(offset_x < hidden_size) {
            if (lane_id == 0) {
                ptr_out_scale[offset_x / 128] = block_absmax * 0.002232142857f;
                sm_max[num_lane_id] = block_absmax;
            }
        }
        __syncthreads();

        if(offset_x < hidden_size) {
            VT1 vdst;
            uint32_t* ptr_reg_dst = (uint32_t*)&vdst;
            float reg_max = sm_max[num_lane_id];
            float scale = 448.0f * __builtin_mxc_rcpf(reg_max);
            #pragma unroll 4
            for (int j = 0; j < N; j += 4) {
                v4f16 reg_tmp;

                #pragma unroll 4
                for (int t = 0; t < 4; ++t) {
                    float reg = reg_i[j + t] * scale;
                    reg = min(reg, 448.0);
                    reg = max(reg, -448.0);
                    reg_tmp[t] = (_Float16)(reg);
                }

                *ptr_reg_dst++ = __builtin_mxc_cvt_pk4_f16tof8(reg_tmp);
            }

            *(VT1*)(output + token_idx * hidden_size + offset_x) = vdst;
        }
    }
}

template<typename T, typename T1, typename VT, typename VT1, bool kApplySwigluLimit>
__global__ void silu_and_mul_mask_quant_nopack_transpose(T* input, __maca_fp8_e4m3* output, float* output_scale, T1* mask, int mask_size, int64_t batch,
    int64_t num_tokens, int64_t hidden_size, int64_t group_num, float swiglu_limit)
{
    constexpr int N = sizeof(VT) / sizeof(T);

    int const tid = threadIdx.x;

    __shared__ T1 sm_mask[1024];
    __shared__ T1 sm_stride[1024];
    for(int i = tid; i < mask_size; i += blockDim.x) {
        if (i < mask_size) {
            sm_mask[i] = mask[i];
        }
    }
    __syncthreads();

    for(int i = tid; i < mask_size; i += blockDim.x) {
        if (i < mask_size) {
            T1 tmp = 0;
            for (int j = 0; j < i; ++j) {
                tmp += sm_mask[j];
            }
            sm_stride[i] = tmp;
        }
    }

    __syncthreads();

    int64_t hidden_size2 = hidden_size << 1;
    int lane_id = threadIdx.x & 15;
    int num_lane_id = threadIdx.x >> 4;
    int64_t total_tokens = sm_stride[mask_size - 1] + sm_mask[mask_size - 1];
    
    __shared__ float sm_max[32];
    for (int64_t idx = blockIdx.y; idx < total_tokens; idx += gridDim.y) {
        int64_t token_id = mask_size - 1;
        while (token_id > 0 && idx < sm_stride[token_id]) {
            token_id--;
        }
        int64_t num = idx - sm_stride[token_id];
        int64_t token_idx = token_id * num_tokens + num;

        const T* ptr_input0 = input + token_idx * hidden_size2;
        const T* ptr_input1 = ptr_input0 + hidden_size;
        float absmax_val = 0.f;

        int offset_x = (blockIdx.x * blockDim.x + threadIdx.x) * N;
        int group_id = offset_x / 128;
        // int batch_id = token_idx / num_tokens;
        // int num_id = token_idx % num_tokens;
        // int scale_offset = batch_id * group_num * num_tokens + group_id * num_tokens + num_id;
        int scale_offset = token_id * group_num * num_tokens + group_id * num_tokens + num;
        float reg_i[N];
        if(offset_x < hidden_size) {
            VT vsrc0 = *(VT*)(ptr_input0 + offset_x);
            VT vsrc1 = *(VT*)(ptr_input1 + offset_x);

            T* ptr_local0 = (T*)&vsrc0;
            T* ptr_local1 = (T*)&vsrc1;

            #pragma unroll
            for (int k = 0; k < N; ++k) {
                float val0 = static_cast<float>(ptr_local0[k]);
                float val1 = static_cast<float>(ptr_local1[k]);
                
                if constexpr(kApplySwigluLimit) {
                    val0 = min(val0, swiglu_limit);
                    val1 = max(val1, -swiglu_limit);
                    val1 = min(val1, swiglu_limit);
                }

                float sigmoid = val0 * __builtin_mxc_rcpf(1.0f + __builtin_expf(-val0));
                float gate_up = val1 * sigmoid;

                reg_i[k] = gate_up;
                absmax_val = max(absmax_val, fabsf(gate_up));
            }
        }

        for (int offset = 8; offset >= 1; offset >>= 1) {
            absmax_val = max(absmax_val, __shfl_down_sync_16(0xffffffff, absmax_val, offset));
        }
        float block_absmax = absmax_val;
        block_absmax = max(block_absmax, 1e-10f);
        
        if(offset_x < hidden_size) {
            if (lane_id == 0) {
                output_scale[scale_offset] = block_absmax * 0.002232142857f;
                sm_max[num_lane_id] = block_absmax;
            }
        }
        __syncthreads();

        if(offset_x < hidden_size) {
            VT1 vdst;
            uint32_t* ptr_reg_dst = (uint32_t*)&vdst;
            float reg_max = sm_max[num_lane_id];
            float scale = 448.0f * __builtin_mxc_rcpf(reg_max);
            #pragma unroll 4
            for (int j = 0; j < N; j += 4) {
                v4f16 reg_tmp;

                #pragma unroll 4
                for (int t = 0; t < 4; ++t) {
                    float reg = reg_i[j + t] * scale;
                    reg = min(reg, 448.0);
                    reg = max(reg, -448.0);
                    reg_tmp[t] = (_Float16)(reg);
                }

                *ptr_reg_dst++ = __builtin_mxc_cvt_pk4_f16tof8(reg_tmp);
            }

            *(VT1*)(output + token_idx * hidden_size + offset_x) = vdst;
        }
    }
}

template<typename T, typename T1>
void launch_silu_mul_quant_nopack(T* input, __maca_fp8_e4m3* output, float* output_scale, T1* mask, int64_t num_tokens, int64_t hidden_size, 
        int64_t mask_size, cudaStream_t stream, std::optional<float> swiglu_limit)
{
    int dev = 0;
    cudaGetDevice(&dev);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

    int gridsize  = sm_count * 4;
    int blocksize = 512;
    int64_t inner_hidden_size = hidden_size / 2;
    constexpr int N = 16 / sizeof(T);
    if((inner_hidden_size & (N - 1)) == 0) {
        int block_x = (inner_hidden_size + blocksize * N - 1) / (blocksize * N);
        int block_y = (gridsize + block_x - 1) / block_x;
        dim3 GridSize(block_x, block_y , 1);
        if(swiglu_limit.has_value()) {
            silu_and_mul_mask_quant_nopack<T, T1, float4, float4, 1><<<GridSize, blocksize, 0, stream>>>(input, output, output_scale, mask, mask_size,
                num_tokens, inner_hidden_size, *swiglu_limit);
        } else {
            silu_and_mul_mask_quant_nopack<T, T1, float4, float4, 0><<<GridSize, blocksize, 0, stream>>>(input, output, output_scale, mask, mask_size,
                num_tokens, inner_hidden_size, 0);
        }
        
    } else {
        printf("silu_and_mul_mask not support this hidden_size\n");
    }
}

template<typename T, typename T1>
void launch_silu_mul_quant_nopack_transpose(T* input, __maca_fp8_e4m3* output, float* output_scale, T1* mask, int64_t batch, int64_t num_tokens, int64_t hidden_size, 
        int64_t mask_size, int quant_group, cudaStream_t stream, std::optional<float> swiglu_limit)
{
    int dev = 0;
    cudaGetDevice(&dev);

    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    int gridsize  = sm_count * 4;
    int blocksize = 512;
    int64_t inner_hidden_size = hidden_size / 2;
    int64_t group_num = inner_hidden_size / quant_group;
    constexpr int N = 16 / sizeof(T);
    if((inner_hidden_size & (N - 1)) == 0) {
        int block_x = (inner_hidden_size + blocksize * N - 1) / (blocksize * N);
        int block_y = (gridsize + block_x - 1) / block_x;
        dim3 GridSize(block_x, block_y , 1);
        if(swiglu_limit.has_value()) {
            silu_and_mul_mask_quant_nopack_transpose<T, T1, float4, float4, 1><<<GridSize, blocksize, 0, stream>>>(input, output, output_scale, mask, mask_size, batch,
                num_tokens, inner_hidden_size, group_num, *swiglu_limit);
        } else {
            silu_and_mul_mask_quant_nopack_transpose<T, T1, float4, float4, 0><<<GridSize, blocksize, 0, stream>>>(input, output, output_scale, mask, mask_size, batch,
                num_tokens, inner_hidden_size, group_num, 0);
        }
    } else {
        printf("silu_and_mul_mask_quant_nopack_transpose not support this hidden_size\n");
    }
}

void fused_silu_mul_dq_mask_quant_fp8_nopack(
    torch::Tensor& output,
    torch::Tensor& output_scale,
    torch::Tensor const& input,
    torch::Tensor const& mask,
    int quant_group,
    std::optional<float> swiglu_limit,
    std::optional<int> isTranspose
)
{
    TORCH_CHECK(input.dim() == 3 , "only support 3 dims");
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(output.is_contiguous());
    TORCH_CHECK(mask.is_contiguous());
    TORCH_CHECK(quant_group == 128, "Only support quant_group 128");
    TORCH_CHECK(input.size(0) == mask.size(0), "input batch dim should be the same with mask batch");
    int64_t const hidden_size = input.size(-1);
    int64_t const num_tokens = input.numel() / hidden_size;
    int64_t const mask_size = mask.numel();
    int64_t const num_tokens_batch = num_tokens / mask_size;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    TORCH_CHECK(
        mask_size <= 1024,
        "fused_silu_mul_dq_mask_quant_fp8_nopack only supports mask_size <= 1024, but got ",
        mask_size);

    TORCH_CHECK(input.element_size() == 2, "only support fp16 or bf16");
    TORCH_CHECK((hidden_size &1) == 0, "hiddensize must can be divided by 2");
    TORCH_CHECK(((hidden_size /2) & 7) == 0, "half hiddensize must can be diveded by 8");
    auto out_buffer = reinterpret_cast<__maca_fp8_e4m3 *>(output.data_ptr<at::Float8_e4m3fn>());
    
    if(isTranspose.has_value()) {
        switch (mask.element_size()) {
            case 8:
                MOE_DISPATCH_FLOATING_TYPES(
                    input.scalar_type(),
                    "launch_silu_mul_quant_nopack_transpose",
                    [&] {

                    launch_silu_mul_quant_nopack_transpose<scalar_t, int64_t>(input.data_ptr<scalar_t>(), out_buffer, output_scale.data_ptr<float>(),
                        mask.data_ptr<int64_t>(), input.size(0), input.size(1), hidden_size, mask_size, quant_group,  stream, swiglu_limit);
                });
                break;

            case 4:
                MOE_DISPATCH_FLOATING_TYPES(
                    input.scalar_type(),
                    "launch_silu_mul_quant_nopack_transpose",
                    [&] {

                    launch_silu_mul_quant_nopack_transpose<scalar_t, int32_t>(input.data_ptr<scalar_t>(), out_buffer, output_scale.data_ptr<float>(),
                        mask.data_ptr<int32_t>(), input.size(0), input.size(1), hidden_size, mask_size, quant_group, stream, swiglu_limit);
                });

                break;

            default:

                TORCH_CHECK(
                    false,
                    "Unsupported mask dtype, only int32/int64 supported.");
        }
        return;
    }
    TORCH_CHECK(output_scale.is_contiguous());
    switch (mask.element_size()) {
        case 8:
            MOE_DISPATCH_FLOATING_TYPES(
                input.scalar_type(),
                "launch_silu_mul_quant_nopack",
                [&] {

                launch_silu_mul_quant_nopack<scalar_t, int64_t>(input.data_ptr<scalar_t>(), out_buffer, output_scale.data_ptr<float>(),
                    mask.data_ptr<int64_t>(), num_tokens_batch, hidden_size, mask_size, stream, swiglu_limit);
            });
            break;

        case 4:
            MOE_DISPATCH_FLOATING_TYPES(
                input.scalar_type(),
                "launch_silu_mul_quant_nopack",
                [&] {

                launch_silu_mul_quant_nopack<scalar_t, int32_t>(input.data_ptr<scalar_t>(), out_buffer, output_scale.data_ptr<float>(),
                    mask.data_ptr<int32_t>(), num_tokens_batch, hidden_size, mask_size, stream, swiglu_limit);
            });

            break;

        default:

            TORCH_CHECK(
                false,
                "Unsupported mask dtype, only int32/int64 supported.");
    }
}
