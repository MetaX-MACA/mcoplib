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


// ============================================================================
// Optimized kernel v8 (C600-U): 128-bit vectorized quant-copy.
//   Each thread owns N=8 consecutive dims of one (token, kv-head) => a 16B
//   uint4 bf16 load + an 8B packed-int8 (int64) store  (>=32B/thread rule).
//   Scales are token-invariant -> preloaded into registers ONCE.
//   block = kv_head_num*head_dim/N (=128 for kv=8), grid.z distributes tokens
//   across all APs.  No shared memory: a pure elementwise copy has zero reuse,
//   so SMEM staging would only add __syncthreads overhead (see dead v7).
// ============================================================================
template <typename scalar_t, int N>
__global__ void store_kv_cache_kernel_v8(
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
    if (q_len <= 0) return;
    const int32_t q_offset = accum_q_lens[batch_idx];
    const int32_t cur_cache_len = cache_lens[batch_idx];
    const int32_t cur_slot_id = cache_slot_ids[batch_idx];
    const int k_head_start = q_head_num;

    const int base = threadIdx.x * N;           // element offset in flat [kv_head*head_dim]
    const int head_idx = base / head_dim;
    const int dim0 = base - head_idx * head_dim;

    // Preload N scales into registers (token-invariant).
    float sc_k[N], sc_v[N];
    #pragma unroll
    for (int j = 0; j < N; j++) {
        sc_k[j] = k_scale[head_idx * head_dim + dim0 + j];
        sc_v[j] = v_scale[head_idx * head_dim + dim0 + j];
    }

    // Per-token source element offsets (K then V), qkv_stride1=head_dim, qkv_stride2=1.
    const int64_t k_src_tok = (int64_t)(k_head_start + head_idx) * qkv_stride1 + dim0;
    const int64_t v_src_tok = k_src_tok + (int64_t)kv_head_num * qkv_stride1;

    const int64_t k_dst_hd = (int64_t)cur_slot_id * k_cache_stride0
                           + (int64_t)head_idx * k_cache_stride1 + dim0;
    const int64_t v_dst_hd = (int64_t)cur_slot_id * v_cache_stride0
                           + (int64_t)head_idx * v_cache_stride1 + dim0;

    for (int t = blockIdx.z; t < q_len; t += gridDim.z) {
        const int64_t src_token = q_offset + t;
        const scalar_t* kp = packed_qkv + src_token * qkv_stride0 + k_src_tok;
        const scalar_t* vp = packed_qkv + src_token * qkv_stride0 + v_src_tok;

        // 128-bit vectorized loads (N=8 bf16 = 16B = uint4).
        uint4 rk = *reinterpret_cast<const uint4*>(kp);
        uint4 rv = *reinterpret_cast<const uint4*>(vp);
        const scalar_t* pk = reinterpret_cast<const scalar_t*>(&rk);
        const scalar_t* pv = reinterpret_cast<const scalar_t*>(&rv);

        int8_t ok[N], ov[N];
        #pragma unroll
        for (int j = 0; j < N; j++) {
            ok[j] = float_to_int8_rn(convert_to_float<scalar_t>(pk[j]) * sc_k[j]);
            ov[j] = float_to_int8_rn(convert_to_float<scalar_t>(pv[j]) * sc_v[j]);
        }

        const int64_t cache_pos = cur_cache_len + t;
        int8_t* kd = k_cache + k_dst_hd + cache_pos * k_cache_stride2;
        int8_t* vd = v_cache + v_dst_hd + cache_pos * v_cache_stride2;
        // Packed 8B stores (N=8 int8 = int64).
        *reinterpret_cast<int64_t*>(kd) = *reinterpret_cast<const int64_t*>(ok);
        *reinterpret_cast<int64_t*>(vd) = *reinterpret_cast<const int64_t*>(ov);
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
    DEBUG_TRACE_PARAMS(packed_qkv, q_lens, accum_q_lens, cache_lens, cache_slot_ids,
                       k_cache, v_cache, k_scale, v_scale,
                       batch_size, q_head_num, kv_head_num);
    DEBUG_DUMP_PARAMS(packed_qkv, q_lens, accum_q_lens, cache_lens, cache_slot_ids,
                      k_cache, v_cache, k_scale, v_scale,
                      batch_size, q_head_num, kv_head_num);

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

    auto qkv_strides = packed_qkv.strides();
    auto k_cache_strides = k_cache.strides();
    auto v_cache_strides = v_cache.strides();
    int dev = 0;
    cudaGetDevice(&dev);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // 3D grid: distribute tokens across SMs for parallelism
    // block.x = batch_idx, block.y = head_idx, block.z = token-group
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    // v8: fold (head, dim) into one block; N=8 => 16B load + 8B store/thread.
    const int VEC_N = 8;
    // v8: token-parallel grid.z. Cap by the token-row count so prefill
    // (large) gets full occupancy (32 blocks/AP) while decode (tiny) avoids
    // launching thousands of idle blocks.
    int tok_hint = (int)packed_qkv.size(0);
    int gridz = sm_count * 32;
    if (tok_hint < gridz) gridz = tok_hint < 1 ? 1 : tok_hint;
    dim3 blocks(batch_size, 1, gridz);
    const int threads = (kv_head_num * head_dim) / VEC_N;

    if(packed_qkv.dtype() == at::ScalarType::Half) {
        store_kv_cache_kernel_v8<half, 8><<<blocks, threads, 0, stream>>>(
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
    } else if(packed_qkv.dtype() == at::ScalarType::BFloat16) {
        store_kv_cache_kernel_v8<maca_bfloat16, 8><<<blocks, threads, 0, stream>>>(
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
    } else {
        TORCH_CHECK(false, "Only float16, bfloat16 are supported");
    }
}