// Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
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

template<typename scalar_t, typename VT, typename VT1, typename VDT, int N>
__global__ void store_kv_cache_kernel_opt(
    const scalar_t* __restrict__ packed_qkv,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    int8_t* __restrict__ k_cache,
    int8_t* __restrict__ v_cache,
    const int32_t* q_lens,
    const int32_t* accum_q_lens,
    const int32_t* cache_lens,
    const int32_t* cache_slot_ids,
    int batch_size,
    int q_head_num,
    int kv_head_num,
    int head_dim,
    int qkv_stride0,
    int qkv_stride1,
    int qkv_stride2,
    int k_cache_stride0,
    int k_cache_stride1,
    int k_cache_stride2,
    int v_cache_stride0,
    int v_cache_stride1,
    int v_cache_stride2
)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    int NUM_THREADS = blockDim.x;
    const int32_t q_len = q_lens[batch_idx];
    const int32_t q_offset = accum_q_lens[batch_idx];
    const int32_t cur_cache_len = cache_lens[batch_idx];
    const int32_t cur_slot_id = cache_slot_ids[batch_idx];
    const int k_head_start = q_head_num;
    const int v_head_start = q_head_num + kv_head_num;
    int stride = gridDim.y;
    int length = kv_head_num * head_dim;
    int tid = threadIdx.x * N;
    int block_stride = NUM_THREADS * N;
    const int64_t k_tgt_idx = cur_slot_id * k_cache_stride0 + cur_cache_len * k_cache_stride2;
    const int64_t v_tgt_idx = cur_slot_id * v_cache_stride0 + cur_cache_len * v_cache_stride2;
    k_cache = k_cache + k_tgt_idx;
    v_cache = v_cache + v_tgt_idx;
#if 1
    for(int i = tid; i < length; i += block_stride) {
        int head_idx = i / head_dim;
        int dim_idx = i % head_dim;
        int local_src_idx = head_idx * qkv_stride1 + dim_idx;
        int local_k_tgt_idx = head_idx * k_cache_stride1 + dim_idx;
        int local_v_tgt_idx = head_idx * v_cache_stride1 + dim_idx;
        int8_t* ptr_k_cache = k_cache + local_k_tgt_idx;
        int8_t* ptr_v_cache = v_cache + local_v_tgt_idx;
        const scalar_t * ptr_packed_qkv = packed_qkv + local_src_idx;
        VT1 reg_k_scale = *(VT1*)(k_scale + i);
        VT1 reg_v_scale = *(VT1*)(v_scale + i);
        float* ptr_reg_k_scale = (float*)&reg_k_scale;
        float* ptr_reg_v_scale = (float*)&reg_v_scale;

        for(int token_idx = blockIdx.y; token_idx < q_len; token_idx += stride) {
            const int64_t src_token = q_offset + token_idx;
            const int64_t src_idx = src_token * qkv_stride0 + k_head_start * qkv_stride1;
            const int64_t tgt_cache_pos = token_idx;
            const int64_t k_tgt_idx = tgt_cache_pos * k_cache_stride2;
            const int64_t v_tgt_idx = tgt_cache_pos * v_cache_stride2;    
            VT reg_k = *(VT*)(ptr_packed_qkv + src_idx);
            VT reg_v = *(VT*)(ptr_packed_qkv + src_idx + kv_head_num * qkv_stride1);
            scalar_t* ptr_reg_k = (scalar_t*)&reg_k;
            scalar_t* ptr_reg_v = (scalar_t*)&reg_v;
            VDT reg_k_dst , reg_v_dst;
            int8_t* ptr_reg_k_dst = (int8_t*)&reg_k_dst;
            int8_t* ptr_reg_v_dst = (int8_t*)&reg_v_dst;
            #pragma unroll N
            for(int j = 0; j < N; j++) {
                ptr_reg_k_dst[j] = float_to_int8_rn(convert_to_float<scalar_t>(ptr_reg_k[j]) * ptr_reg_k_scale[j]);
                ptr_reg_v_dst[j] = float_to_int8_rn(convert_to_float<scalar_t>(ptr_reg_v[j]) * ptr_reg_v_scale[j]);
            }
            *(VDT*)(ptr_k_cache + k_tgt_idx) = reg_k_dst;
            *(VDT*)(ptr_v_cache + v_tgt_idx) = reg_v_dst;
        }
    }
#else
    for(int token_idx = blockIdx.y; token_idx < q_len; token_idx += stride) {
        const int64_t src_token = q_offset + token_idx;
        const int64_t src_idx = src_token * qkv_stride0 + k_head_start * qkv_stride1;
        const int64_t tgt_cache_pos = token_idx;
        const int64_t k_tgt_idx = tgt_cache_pos * k_cache_stride2;
        const int64_t v_tgt_idx = tgt_cache_pos * v_cache_stride2;
        const scalar_t * ptr_packed_qkv = packed_qkv + src_idx;
        int8_t* ptr_k_cache = k_cache + k_tgt_idx;
        int8_t* ptr_v_cache = v_cache + v_tgt_idx;
        for(int i = tid; i < length; i += block_stride) {
            int head_idx = i / head_dim;
            int dim_idx = i % head_dim;
            int local_src_idx = head_idx * qkv_stride1 + dim_idx;
            int local_k_tgt_idx = head_idx * k_cache_stride1 + dim_idx;
            int local_v_tgt_idx = head_idx * v_cache_stride1 + dim_idx;
            VT reg_k = *(VT*)(ptr_packed_qkv + local_src_idx);
            VT reg_v = *(VT*)(ptr_packed_qkv + local_src_idx + kv_head_num * qkv_stride1);
            scalar_t* ptr_reg_k = (scalar_t*)&reg_k;
            scalar_t* ptr_reg_v = (scalar_t*)&reg_v;
            VDT reg_k_dst , reg_v_dst;
            int8_t* ptr_reg_k_dst = (int8_t*)&reg_k_dst;
            int8_t* ptr_reg_v_dst = (int8_t*)&reg_v_dst;
            VT1 reg_k_scale = *(VT1*)(k_scale + i);
            VT1 reg_v_scale = *(VT1*)(v_scale + i);
            float* ptr_reg_k_scale = (float*)&reg_k_scale;
            float* ptr_reg_v_scale = (float*)&reg_v_scale;
            #pragma unroll N
            for(int j = 0; j < N; j++) {
                ptr_reg_k_dst[j] = float_to_int8_rn(convert_to_float<scalar_t>(ptr_reg_k[j]) * ptr_reg_k_scale[j]);
                ptr_reg_v_dst[j] = float_to_int8_rn(convert_to_float<scalar_t>(ptr_reg_v[j]) * ptr_reg_v_scale[j]);
            }
            *(VDT*)(ptr_k_cache + local_k_tgt_idx) = reg_k_dst;
            *(VDT*)(ptr_v_cache + local_v_tgt_idx) = reg_v_dst;
        }
    }
#endif
}

template <typename scalar_t>
__global__ void store_kv_cache_kernel(
    const scalar_t* __restrict__ packed_qkv,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    int8_t* __restrict__ k_cache,
    int8_t* __restrict__ v_cache,
    const int32_t* q_lens,
    const int32_t* accum_q_lens,
    const int32_t* cache_lens,
    const int32_t* cache_slot_ids,
    int batch_size,
    int q_head_num,
    int kv_head_num,
    int head_dim,
    int qkv_stride0,
    int qkv_stride1,
    int qkv_stride2,
    int k_cache_stride0,
    int k_cache_stride1,
    int k_cache_stride2,
    int v_cache_stride0,
    int v_cache_stride1,
    int v_cache_stride2
) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const int32_t q_len = q_lens[batch_idx];
    const int32_t q_offset = accum_q_lens[batch_idx];
    const int32_t cur_cache_len = cache_lens[batch_idx];
    const int32_t cur_slot_id = cache_slot_ids[batch_idx];

    const int k_head_start = q_head_num;
    const int v_head_start = q_head_num + kv_head_num;

    // 三维索引: [head_idx, token_idx, dim_idx]
    const int head_idx = blockIdx.y;

    if (head_idx >= kv_head_num) 
        return;
    int stride = gridDim.z;
    int thread_stride = blockDim.x;
    for(int dim_idx = threadIdx.x; dim_idx < head_dim; dim_idx += thread_stride) {
        float value_k = k_scale[head_idx * head_dim + dim_idx];
        float value_v = v_scale[head_idx * head_dim + dim_idx];
        for(int token_idx = blockIdx.z; token_idx < q_len; token_idx += stride) {
            // 源数据索引计算
            const int64_t src_token = q_offset + token_idx;
            const int64_t src_idx = src_token * qkv_stride0 + 
                                (k_head_start + head_idx) * qkv_stride1 + 
                                dim_idx;

            // 目标缓存索引计算
            const int64_t tgt_cache_pos = cur_cache_len + token_idx;
            const int64_t k_tgt_idx = cur_slot_id * k_cache_stride0 + 
                                    (head_idx) * k_cache_stride1 + 
                                    tgt_cache_pos * k_cache_stride2 + 
                                    dim_idx;

            const int64_t v_tgt_idx = cur_slot_id * v_cache_stride0 + 
                                    (head_idx) * v_cache_stride1 + 
                                    tgt_cache_pos * v_cache_stride2 + 
                                    dim_idx;

            // 量化处理
            scalar_t k_val = packed_qkv[src_idx];
            scalar_t v_val = packed_qkv[src_idx + kv_head_num * qkv_stride1];

            k_cache[k_tgt_idx] = float_to_int8_rn(convert_to_float<scalar_t>(k_val) * value_k);
            v_cache[v_tgt_idx] = float_to_int8_rn(convert_to_float<scalar_t>(v_val) * value_v);
        }
    }
}

void store_kv_cache_cuda_interface(
    torch::Tensor packed_qkv,
    torch::Tensor q_lens,
    torch::Tensor accum_q_lens,
    torch::Tensor cache_lens,
    torch::Tensor cache_slot_ids,
    torch::Tensor &k_cache,
    torch::Tensor &v_cache,
    torch::Tensor k_scale,
    torch::Tensor v_scale,
    int batch_size,
    int q_head_num,
    int kv_head_num
) {
	DEBUG_TRACE_PARAMS(packed_qkv, q_lens, accum_q_lens, cache_lens, cache_slot_ids, &k_cache, &v_cache, k_scale, v_scale, batch_size, q_head_num, kv_head_num);
	DEBUG_DUMP_PARAMS(packed_qkv, q_lens, accum_q_lens, cache_lens, cache_slot_ids, &k_cache, &v_cache, k_scale, v_scale, batch_size, q_head_num, kv_head_num);
    CHECK_DEVICE(packed_qkv);
    CHECK_DEVICE(q_lens);
    CHECK_DEVICE(accum_q_lens);
    CHECK_DEVICE(cache_lens);
    CHECK_DEVICE(cache_slot_ids);
    CHECK_DEVICE(k_cache);
    CHECK_DEVICE(v_cache);
    CHECK_DEVICE(k_scale);
    CHECK_DEVICE(v_scale);

    int head_dim = packed_qkv.size(2);
    int total_kv_heads = k_cache.size(1);

    auto qkv_strides = packed_qkv.strides();
    auto k_cache_strides = k_cache.strides();
    auto v_cache_strides = v_cache.strides();
    int dev = 0;
    cudaGetDevice(&dev);
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);

    dim3 blocks(batch_size, kv_head_num, (sm_count + batch_size - 1) / batch_size); // 第三维动态调整
    const int threads = std::min(head_dim, 512);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if(packed_qkv.dtype() == at::ScalarType::Half) {
        // if((head_dim & 3) == 0 ) {
        //     dim3 gridSize(batch_size, (sm_count + batch_size - 1) / batch_size, 1);
        //     int blocksize = std::min(head_dim * kv_head_num / 4, 512);
        //     store_kv_cache_kernel_opt<half, float2, float4 , float, 4><<<gridSize, blocksize, 0, stream>>>(
        //         reinterpret_cast<const half*>(packed_qkv.data_ptr<at::Half>()),
        //         k_scale.data_ptr<float>(),
        //         v_scale.data_ptr<float>(),
        //         k_cache.data_ptr<int8_t>(),
        //         v_cache.data_ptr<int8_t>(),
        //         q_lens.data_ptr<int32_t>(),
        //         accum_q_lens.data_ptr<int32_t>(),
        //         cache_lens.data_ptr<int32_t>(),
        //         cache_slot_ids.data_ptr<int32_t>(),
        //         batch_size,
        //         q_head_num,
        //         kv_head_num,
        //         head_dim,
        //         qkv_strides[0], qkv_strides[1], qkv_strides[2],
        //         k_cache_strides[0], k_cache_strides[1], k_cache_strides[2],
        //         v_cache_strides[0], v_cache_strides[1], v_cache_strides[2]
        //     );
        // } else 
        {
            store_kv_cache_kernel<half><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const half*>(packed_qkv.data_ptr<at::Half>()),
                k_scale.data_ptr<float>(),
                v_scale.data_ptr<float>(),
                k_cache.data_ptr<int8_t>(),
                v_cache.data_ptr<int8_t>(),
                q_lens.data_ptr<int32_t>(),
                accum_q_lens.data_ptr<int32_t>(),
                cache_lens.data_ptr<int32_t>(),
                cache_slot_ids.data_ptr<int32_t>(),
                batch_size,
                q_head_num,
                kv_head_num,
                head_dim,
                qkv_strides[0], qkv_strides[1], qkv_strides[2],
                k_cache_strides[0], k_cache_strides[1], k_cache_strides[2],
                v_cache_strides[0], v_cache_strides[1], v_cache_strides[2]
            );
        }
    } else if(packed_qkv.dtype() == at::ScalarType::BFloat16) {
        // if((head_dim & 3) == 0 ) {
        //     dim3 gridSize(batch_size, (sm_count + batch_size - 1) / batch_size, 1);
        //     int blocksize = std::min(head_dim * kv_head_num / 4, 512);
        //     store_kv_cache_kernel_opt<half, float2, float4 , float, 4><<<gridSize, blocksize, 0, stream>>>(
        //         reinterpret_cast<const half*>(packed_qkv.data_ptr<at::Half>()),
        //         k_scale.data_ptr<float>(),
        //         v_scale.data_ptr<float>(),
        //         k_cache.data_ptr<int8_t>(),
        //         v_cache.data_ptr<int8_t>(),
        //         q_lens.data_ptr<int32_t>(),
        //         accum_q_lens.data_ptr<int32_t>(),
        //         cache_lens.data_ptr<int32_t>(),
        //         cache_slot_ids.data_ptr<int32_t>(),
        //         batch_size,
        //         q_head_num,
        //         kv_head_num,
        //         head_dim,
        //         qkv_strides[0], qkv_strides[1], qkv_strides[2],
        //         k_cache_strides[0], k_cache_strides[1], k_cache_strides[2],
        //         v_cache_strides[0], v_cache_strides[1], v_cache_strides[2]
        //     );
        // } else 
        {
            store_kv_cache_kernel<maca_bfloat16><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const maca_bfloat16*>(packed_qkv.data_ptr<at::BFloat16>()),
                k_scale.data_ptr<float>(),
                v_scale.data_ptr<float>(),
                k_cache.data_ptr<int8_t>(),
                v_cache.data_ptr<int8_t>(),
                q_lens.data_ptr<int32_t>(),
                accum_q_lens.data_ptr<int32_t>(),
                cache_lens.data_ptr<int32_t>(),
                cache_slot_ids.data_ptr<int32_t>(),
                batch_size,
                q_head_num,
                kv_head_num,
                head_dim,
                qkv_strides[0], qkv_strides[1], qkv_strides[2],
                k_cache_strides[0], k_cache_strides[1], k_cache_strides[2],
                v_cache_strides[0], v_cache_strides[1], v_cache_strides[2]
            );
        }
    }else {
        TORCH_CHECK(false, "Only float16, bfloat16 are supported");
    }
}