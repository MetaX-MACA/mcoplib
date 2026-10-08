#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cub/cub.cuh>
#include "../kernel/utils.h"
#include "mcoplib_ops_params_info.hpp"
#include "mcoplib_ops_params_dump.hpp"

__device__ __forceinline__ float2 bf162_to_float2(__maca_bfloat162 v) {
    return {__bfloat162float(v.x), __bfloat162float(v.y)};
}
__device__ __forceinline__ __maca_bfloat162 f2_to_bf162(float a, float b) {
    return __floats2bfloat162_rn(a, b);
}
static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ int find_batch_bin(
        const int* __restrict__ accum_q_lens, int tid, int batch_size) {
    int lo = 0, hi = batch_size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (accum_q_lens[mid + 1] > tid) hi = mid;
        else                              lo = mid + 1;
    }
    return lo;
}

#define ROPE_ONE_HEAD(out_head, i, c4, s4, c4b, s4b)                                        \
    do {                                                                                     \
        uint4 X = *reinterpret_cast<const uint4*>((out_head) + (i));                        \
        uint4 Y = *reinterpret_cast<const uint4*>((out_head) + HALF + (i));                 \
        __maca_bfloat162* x2 = reinterpret_cast<__maca_bfloat162*>(&X);                     \
        __maca_bfloat162* y2 = reinterpret_cast<__maca_bfloat162*>(&Y);                     \
        float2 xf0 = bf162_to_float2(x2[0]), yf0 = bf162_to_float2(y2[0]);                  \
        float2 xf1 = bf162_to_float2(x2[1]), yf1 = bf162_to_float2(y2[1]);                  \
        float2 xf2 = bf162_to_float2(x2[2]), yf2 = bf162_to_float2(y2[2]);                  \
        float2 xf3 = bf162_to_float2(x2[3]), yf3 = bf162_to_float2(y2[3]);                  \
        uint4 XO, YO;                                                                        \
        __maca_bfloat162* xo = reinterpret_cast<__maca_bfloat162*>(&XO);                    \
        __maca_bfloat162* yo = reinterpret_cast<__maca_bfloat162*>(&YO);                    \
        xo[0] = f2_to_bf162(xf0.x*(c4).x - yf0.x*(s4).x, xf0.y*(c4).y - yf0.y*(s4).y);      \
        yo[0] = f2_to_bf162(yf0.x*(c4).x + xf0.x*(s4).x, yf0.y*(c4).y + xf0.y*(s4).y);      \
        xo[1] = f2_to_bf162(xf1.x*(c4).z - yf1.x*(s4).z, xf1.y*(c4).w - yf1.y*(s4).w);      \
        yo[1] = f2_to_bf162(yf1.x*(c4).z + xf1.x*(s4).z, yf1.y*(c4).w + xf1.y*(s4).w);      \
        xo[2] = f2_to_bf162(xf2.x*(c4b).x - yf2.x*(s4b).x, xf2.y*(c4b).y - yf2.y*(s4b).y);  \
        yo[2] = f2_to_bf162(yf2.x*(c4b).x + xf2.x*(s4b).x, yf2.y*(c4b).y + xf2.y*(s4b).y);  \
        xo[3] = f2_to_bf162(xf3.x*(c4b).z - yf3.x*(s4b).z, xf3.y*(c4b).w - yf3.y*(s4b).w);  \
        yo[3] = f2_to_bf162(yf3.x*(c4b).z + xf3.x*(s4b).z, yf3.y*(c4b).w + xf3.y*(s4b).w);  \
        *reinterpret_cast<uint4*>((out_head) + (i))        = XO;                             \
        *reinterpret_cast<uint4*>((out_head) + HALF + (i)) = YO;                             \
    } while (0)


template <int ROPE_DIM, int BLOCK_THREADS, int TOKENS_PER_BLK, bool BATCH1>
__launch_bounds__(BLOCK_THREADS, 32)
__global__ void rope_prefill_kernel(
        __maca_bfloat16* __restrict__ qkv,
        const float* __restrict__ cos_base,
        const float* __restrict__ sin_base,
        const int*   __restrict__ accum_q_lens,
        const int*   __restrict__ cache_lens,
        int batch_size,
        int num_tokens,
        int num_heads,
        int total_head_num,
        int head_dim,
        int rope_offset)
{
    constexpr int HALF          = ROPE_DIM / 2;
    constexpr int THR_PER_HEAD  = HALF / 8;
    constexpr int HEADS_PER_BLK = BLOCK_THREADS / THR_PER_HEAD;

    const int tid  = threadIdx.x;
    const int lane = tid % THR_PER_HEAD;
    const int hoff = tid / THR_PER_HEAD;
    const int i    = lane * 8;
    const size_t token_stride = (size_t)total_head_num * head_dim;
    const int base_token = blockIdx.x * TOKENS_PER_BLK;

    #pragma unroll
    for (int t = 0; t < TOKENS_PER_BLK; ++t) {
        const int token_id = base_token + t;
        if (token_id >= num_tokens) return;

        int cache_idx;
        if constexpr (BATCH1) {
            cache_idx = cache_lens[0] + token_id;
        } else {
            __shared__ int s_cache_idx;
            if (tid == 0) {
                int b = find_batch_bin(accum_q_lens, token_id, batch_size);
                s_cache_idx = cache_lens[b] + (token_id - accum_q_lens[b]);
            }
            __syncthreads();
            cache_idx = s_cache_idx;
        }

        const float4 c4  = *reinterpret_cast<const float4*>(cos_base + (size_t)cache_idx*ROPE_DIM + i);
        const float4 s4  = *reinterpret_cast<const float4*>(sin_base + (size_t)cache_idx*ROPE_DIM + i);
        const float4 c4b = *reinterpret_cast<const float4*>(cos_base + (size_t)cache_idx*ROPE_DIM + i + 4);
        const float4 s4b = *reinterpret_cast<const float4*>(sin_base + (size_t)cache_idx*ROPE_DIM + i + 4);

        __maca_bfloat16* token_ptr = qkv + (size_t)token_id * token_stride + rope_offset;

        #pragma unroll 1
        for (int h = hoff; h < num_heads; h += HEADS_PER_BLK) {
            __maca_bfloat16* out_head = token_ptr + h * head_dim;
            ROPE_ONE_HEAD(out_head, i, c4, s4, c4b, s4b);
        }

        if constexpr (!BATCH1) __syncthreads();
    }
}

template <int ROPE_DIM, int BLOCK_THREADS, bool BATCH1>
__launch_bounds__(BLOCK_THREADS, 32)
__global__ void rope_decode_kernel(
        __maca_bfloat16* __restrict__ qkv,
        const float* __restrict__ cos_base,
        const float* __restrict__ sin_base,
        const int*   __restrict__ accum_q_lens,
        const int*   __restrict__ cache_lens,
        int batch_size,
        int num_heads,
        int total_head_num,
        int head_dim,
        int rope_offset)
{
    constexpr int HALF          = ROPE_DIM / 2;
    constexpr int THR_PER_HEAD  = HALF / 8;
    constexpr int HEADS_PER_BLK = BLOCK_THREADS / THR_PER_HEAD;

    const int token_id = blockIdx.x;
    const int h_base   = blockIdx.y * HEADS_PER_BLK;

    const int tid  = threadIdx.x;
    const int lane = tid % THR_PER_HEAD;
    const int hoff = tid / THR_PER_HEAD;
    const int i    = lane * 8;
    const int h    = h_base + hoff;

    if (h >= num_heads) return;

    int cache_idx;
    if constexpr (BATCH1) {
        cache_idx = cache_lens[0] + token_id;
    } else {
        int b = find_batch_bin(accum_q_lens, token_id, batch_size);
        cache_idx = cache_lens[b] + (token_id - accum_q_lens[b]);
    }

    const float4 c4  = *reinterpret_cast<const float4*>(cos_base + (size_t)cache_idx*ROPE_DIM + i);
    const float4 s4  = *reinterpret_cast<const float4*>(sin_base + (size_t)cache_idx*ROPE_DIM + i);
    const float4 c4b = *reinterpret_cast<const float4*>(cos_base + (size_t)cache_idx*ROPE_DIM + i + 4);
    const float4 s4b = *reinterpret_cast<const float4*>(sin_base + (size_t)cache_idx*ROPE_DIM + i + 4);

    const size_t token_stride = (size_t)total_head_num * head_dim;
    __maca_bfloat16* out_head = qkv + (size_t)token_id * token_stride
                              + h * head_dim + rope_offset;

    ROPE_ONE_HEAD(out_head, i, c4, s4, c4b, s4b);
}

__global__ void rope_neox_fallback_kernel(
        __maca_bfloat16*  __restrict__ qkv,
        const float* __restrict__ cos_base,
        const float* __restrict__ sin_base,
        const int*   __restrict__ accum_q_lens,
        const int*   __restrict__ cache_lens,
        int batch_size, int num_tokens, int num_heads,
        int total_head_num, int head_dim, int rope_offset, int rope_dim)
{
    const int token_id = blockIdx.x;
    const int head_id  = blockIdx.y;
    const int tid = threadIdx.x;
    if (token_id >= num_tokens || head_id >= num_heads) return;

    const int half = rope_dim >> 1;
    __shared__ int s_cache_idx;
    if (tid == 0) {
        int b = batch_size - 1;
        #pragma unroll 1
        for (int i = 0; i < batch_size; ++i) {
            if (token_id < accum_q_lens[i + 1]) { b = i; break; }
        }
        s_cache_idx = cache_lens[b] + (token_id - accum_q_lens[b]);
    }
    __syncthreads();
    const int cache_idx = s_cache_idx;

    const size_t token_stride = (size_t)total_head_num * head_dim;
    __maca_bfloat16* out_head = qkv + token_id * token_stride + head_id * head_dim + rope_offset;
    const float* cos_ptr = cos_base + (size_t)cache_idx * rope_dim;
    const float* sin_ptr = sin_base + (size_t)cache_idx * rope_dim;

    for (int i = tid; i < half; i += blockDim.x) {
        float xf = __bfloat162float(out_head[i]);
        float yf = __bfloat162float(out_head[half + i]);
        float c  = cos_ptr[i], s = sin_ptr[i];
        out_head[i]         = __float2bfloat16_rn(xf * c - yf * s);
        out_head[half + i]  = __float2bfloat16_rn(yf * c + xf * s);
    }
}

static void launch_rope_fallback(
        __maca_bfloat16* qkv,
        const float* cos, const float* sin,
        const int* accum_q_lens, const int* cache_lens,
        int num_tokens, int batch_size, int num_heads,
        int total_head_num, int head_dim, int rope_offset, int rope_dim,
        cudaStream_t stream)
{
    if (num_tokens == 0 || num_heads == 0 || rope_dim == 0) return;
    const int half = rope_dim / 2;
    int block = 32;
    while (block < half && block < 256) block *= 2;
    dim3 grid(num_tokens, num_heads);
    dim3 blk(block);
    rope_neox_fallback_kernel<<<grid, blk, 0, stream>>>(
        qkv, cos, sin, accum_q_lens, cache_lens,
        batch_size, num_tokens, num_heads,
        total_head_num, head_dim, rope_offset, rope_dim);
}

static inline void launch_rope_fast(
        __maca_bfloat16* qkv_ptr,
        const float* cos_ptr, const float* sin_ptr,
        const int* accum_ptr, const int* clen_ptr,
        int num_tokens, int batch_size, int num_heads,
        int total_head_num, int head_dim, int rope_offset, int ROPE_DIM,
        cudaStream_t stream)
{
    constexpr int BLOCK = 64;
    const bool batch1 = (batch_size == 1);
    const bool use_prefill = (num_tokens >= 256);

    if (!use_prefill) {
        int heads_per_blk;
        switch (ROPE_DIM) {
            case 64:  heads_per_blk = BLOCK / (64/2/8);  break;
            case 128: heads_per_blk = BLOCK / (128/2/8); break;
            case 256: heads_per_blk = BLOCK / (256/2/8); break;
            default:  heads_per_blk = 1;
        }
        dim3 grid(num_tokens, cdiv(num_heads, heads_per_blk));
        dim3 block(BLOCK);

#define LAUNCH_DECODE(DIM, B1)                                                    \
        rope_decode_kernel<DIM, BLOCK, B1><<<grid, block, 0, stream>>>(           \
            qkv_ptr, cos_ptr, sin_ptr, accum_ptr, clen_ptr,                       \
            batch_size, num_heads, total_head_num, head_dim, rope_offset)

        switch (ROPE_DIM) {
            case 64:  if (batch1) { LAUNCH_DECODE(64,  true); } else { LAUNCH_DECODE(64,  false); } return;
            case 128: if (batch1) { LAUNCH_DECODE(128, true); } else { LAUNCH_DECODE(128, false); } return;
            case 256: if (batch1) { LAUNCH_DECODE(256, true); } else { LAUNCH_DECODE(256, false); } return;
        }
#undef LAUNCH_DECODE
        return;
    }

    int tpb;
    if      (num_tokens >= 8192) tpb = 4;
    else if (num_tokens >= 2048) tpb = 2;
    else                          tpb = 1;

    dim3 block(BLOCK);

#define LAUNCH_PREFILL(DIM, TPB, B1)                                              \
    do {                                                                          \
        dim3 grid(cdiv(num_tokens, TPB));                                         \
        rope_prefill_kernel<DIM, BLOCK, TPB, B1><<<grid, block, 0, stream>>>(     \
            qkv_ptr, cos_ptr, sin_ptr, accum_ptr, clen_ptr,                       \
            batch_size, num_tokens, num_heads, total_head_num, head_dim, rope_offset); \
    } while(0)

#define DISPATCH_TPB(DIM, B1)                            \
    do {                                                 \
        if      (tpb == 4) LAUNCH_PREFILL(DIM, 4, B1);   \
        else if (tpb == 2) LAUNCH_PREFILL(DIM, 2, B1);   \
        else               LAUNCH_PREFILL(DIM, 1, B1);   \
    } while(0)

    switch (ROPE_DIM) {
        case 64:  if (batch1) DISPATCH_TPB(64,  true);  else DISPATCH_TPB(64,  false); return;
        case 128: if (batch1) DISPATCH_TPB(128, true);  else DISPATCH_TPB(128, false); return;
        case 256: if (batch1) DISPATCH_TPB(256, true);  else DISPATCH_TPB(256, false); return;
    }
#undef DISPATCH_TPB
#undef LAUNCH_PREFILL
}

void rotary_embedding(
        at::Tensor packed_qkv,
        at::Tensor q_len,
        at::Tensor accum_q_lens,
        at::Tensor cache_lens,
        at::Tensor cos,
        at::Tensor sin,
        const int q_head_num,
        const int kv_head_num,
        const int rope_offset = 0)
{
    DEBUG_TRACE_PARAMS(packed_qkv, q_len, accum_q_lens, cache_lens, cos, sin, q_head_num, kv_head_num, rope_offset);
    DEBUG_DUMP_PARAMS(packed_qkv, q_len, accum_q_lens, cache_lens, cos, sin, q_head_num, kv_head_num, rope_offset);
    TORCH_CHECK(packed_qkv.dtype() == at::ScalarType::BFloat16, "bf16 only");
    TORCH_CHECK(cos.dtype() == at::kFloat && sin.dtype() == at::kFloat, "cos/sin fp32");
    TORCH_CHECK(accum_q_lens.dtype() == at::kInt && cache_lens.dtype() == at::kInt, "int32");

    const int head_dim       = packed_qkv.size(-1);
    const int total_head_num = packed_qkv.size(-2);
    const int num_tokens     = packed_qkv.numel() / (head_dim * total_head_num);
    const int batch_size     = cache_lens.numel();
    const int num_heads      = q_head_num + kv_head_num;
    const int ROPE_DIM       = cos.size(-1);

    TORCH_CHECK(rope_offset + ROPE_DIM <= head_dim, "rope range OOB");
    TORCH_CHECK((ROPE_DIM % 2) == 0, "ROPE_DIM must be even");
    TORCH_CHECK(accum_q_lens.numel() == batch_size + 1, "accum size mismatch");

    if (num_tokens == 0) return;

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto* qkv_ptr = reinterpret_cast<__maca_bfloat16*>(packed_qkv.data_ptr<at::BFloat16>());
    const float* cos_ptr = cos.data_ptr<float>();
    const float* sin_ptr = sin.data_ptr<float>();
    const int* accum_ptr = accum_q_lens.data_ptr<int>();
    const int* clen_ptr  = cache_lens.data_ptr<int>();

    const bool fast_ok =
        (ROPE_DIM == 64 || ROPE_DIM == 128 || ROPE_DIM == 256) &&
        ((rope_offset & 7) == 0) &&
        (reinterpret_cast<uintptr_t>(qkv_ptr) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(cos_ptr) % 16 == 0) &&
        (reinterpret_cast<uintptr_t>(sin_ptr) % 16 == 0);

    if (fast_ok) {
        launch_rope_fast(qkv_ptr, cos_ptr, sin_ptr, accum_ptr, clen_ptr,
                         num_tokens, batch_size, num_heads,
                         total_head_num, head_dim, rope_offset, ROPE_DIM, stream);
        return;
    }

    launch_rope_fallback(qkv_ptr, cos_ptr, sin_ptr, accum_ptr, clen_ptr,
                         num_tokens, batch_size, num_heads,
                         total_head_num, head_dim, rope_offset, ROPE_DIM, stream);
}
