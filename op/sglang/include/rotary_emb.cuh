#include <cuda_runtime.h>

namespace fused_mla {

template<typename scalar_t, int QK_ROPE_HEAD_DIM=64>
__device__ __forceinline__ void rotary_emb(
    int tid,
    int pid,
    const scalar_t* pe_ptr,
    const float* cos_sin_cache,
    const int64_t* positions,
    scalar_t* out_ptr
) {
    int64_t pos = positions[pid];
    const float* cos_sin_cache_ptr = cos_sin_cache + pos * QK_ROPE_HEAD_DIM;

    // 1. Each thread loads its own bf16
    __nv_bfloat16 my_pe_bf16 = pe_ptr[tid];

    // 2. Each thread loads its own cos or sin
    float my_cos_or_sin = cos_sin_cache_ptr[tid];

    // 3. Exchange bf16 with its partner
    int peer_tid = (tid & ~1) | ((tid + 1) & 1);  // Cross pair
    __nv_bfloat16 peer_pe_bf16 = __shfl_sync(0xffffffffffffffff, my_pe_bf16, peer_tid);

    // 4. Exchange cos and sin
    int pe_idx = tid >> 1;
    int cos_tid = pe_idx;       // cos values are in the first 32 elements
    int sin_tid = pe_idx + 32;  // sin values are in the last 32 elements

    float cos_val = __shfl_sync(0xffffffffffffffff, my_cos_or_sin, cos_tid);
    float sin_val = __shfl_sync(0xffffffffffffffff, my_cos_or_sin, sin_tid);

    // 5. Compute
    float my_pe_f32 = __bfloat162float(my_pe_bf16);
    float peer_pe_f32 = __bfloat162float(peer_pe_bf16);

    scalar_t* out_ptr_local = out_ptr + pe_idx * 2;

    if ((tid & 1) == 0) {
        // Even threads are responsible for out_ptr[0]
        out_ptr_local[0] = __float2bfloat16_rn(my_pe_f32 * cos_val - peer_pe_f32 * sin_val);
    } else {
        // Odd threads are responsible for out_ptr[1]
        out_ptr_local[1] = __float2bfloat16_rn(my_pe_f32 * cos_val + peer_pe_f32 * sin_val);
    }
}

template<typename scalar_t, int QK_ROPE_HEAD_DIM=64>
__device__ __forceinline__ void rotary_emb_float4(
    int lane_in_seg,
    int pid,
    const scalar_t* pe_ptr,
    const float* cos_sin_cache,
    const int64_t* positions,
    scalar_t* out_ptr
) {
    
    constexpr int kPairsPerThread = QK_ROPE_HEAD_DIM * sizeof(scalar_t) / 16 / 2;  // 4 rope pair/lane
    const int pair0 = lane_in_seg * kPairsPerThread;
    const int64_t pos = positions[pid];
    const float* cos_sin_cache_ptr = cos_sin_cache + pos * QK_ROPE_HEAD_DIM;
    float4 raw = __ldg((const float4*)pe_ptr + lane_in_seg);
    __nv_bfloat16* v = (__nv_bfloat16*)&raw;
    #pragma unroll
    for (int p = 0; p < kPairsPerThread; ++p) {
        const int k = pair0 + p;
        const float cos_val = __ldg(cos_sin_cache_ptr + k);
        const float sin_val = __ldg(cos_sin_cache_ptr + QK_ROPE_HEAD_DIM / 2 + k);
        const float x_f32 = __bfloat162float(v[2 * p]);
        const float y_f32 = __bfloat162float(v[2 * p + 1]);
        v[2 * p]     = __float2bfloat16_rn(x_f32 * cos_val - y_f32 * sin_val);
        v[2 * p + 1] = __float2bfloat16_rn(x_f32 * sin_val + y_f32 * cos_val);
    }
    ((float4*)out_ptr)[lane_in_seg] = raw;
}

}
