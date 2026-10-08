#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <vector>

// ============================================================================
// Metax C500 极致优化内核 (Zero-Shared-Memory, Multi-Head Block, SREG Caching)
// ============================================================================
template <typename scalar_t, int VEC_SIZE, int THREADS_PER_HEAD>
__global__ void fused_gemma_rmsnorm_rope_neox_opt_kernel_no_pack(
    scalar_t const* __restrict__ qkv,
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ k,
    scalar_t* __restrict__ v,
    scalar_t const* __restrict__ q_weight,
    scalar_t const* __restrict__ k_weight,
    scalar_t const* __restrict__ cos_sin_cache,
    int64_t const* __restrict__ positions,
    float const eps,
    int64_t const qkv_stride_batch,
    int64_t const q_stride_batch,
    int64_t const k_stride_batch,
    int64_t const v_stride_batch,
    int32_t const q_size,
    int32_t const kv_size,
    int32_t const num_heads,
    int32_t const num_kv_heads,
    int32_t const head_dim,
    int32_t const total_tasks) 
{
    // 每个 Block 固定处理 512 个线程
    // 计算当前线程负责的全局 Head 任务 ID
    int heads_per_block = blockDim.x / THREADS_PER_HEAD;
    int local_head_idx = threadIdx.x / THREADS_PER_HEAD;
    int lane_idx = threadIdx.x % THREADS_PER_HEAD; 
    
    int global_task_idx = blockIdx.x * heads_per_block + local_head_idx;

    // 越界保护
    if (global_task_idx >= total_tasks) return;

    // 解码当前任务 (属于哪个 Token 的哪个 Head)
    int total_active_heads = num_heads + 2 * num_kv_heads;
    int token_idx = global_task_idx / total_active_heads;
    int head_idx = global_task_idx % total_active_heads;


    int vec_offset = lane_idx * VEC_SIZE;

    if (head_idx >= num_heads + num_kv_heads) {
        int v_head_idx = head_idx - num_heads - num_kv_heads;
        scalar_t const* v_ptr = qkv + token_idx * qkv_stride_batch + q_size + kv_size + v_head_idx * head_dim;
        scalar_t* out_ptr = v + token_idx * v_stride_batch + v_head_idx * head_dim;
        *(float4*)(out_ptr + vec_offset) = *(float4 const*)(v_ptr + vec_offset);
        return;
    }

    int64_t position = positions[token_idx];


    scalar_t const* head_ptr;
    scalar_t const* weight;
    scalar_t* out_ptr;
    if (head_idx < num_heads) {
        //  Q head
        head_ptr = qkv + token_idx * qkv_stride_batch + head_idx * head_dim;
        weight = q_weight;
        out_ptr = q + token_idx * q_stride_batch + head_idx * head_dim;
    } else {
        //  K head，物理偏移需要跨过整个 Q_size
        int k_head_idx = head_idx - num_heads;
        head_ptr = qkv + token_idx * qkv_stride_batch + q_size + k_head_idx * head_dim;
        weight = k_weight;
        out_ptr = k + token_idx * k_stride_batch + k_head_idx * head_dim;
    }


    float reg_input[VEC_SIZE];
    float reg_weight[VEC_SIZE];
    float ss = 0.0f;

    scalar_t local_x[VEC_SIZE];
    scalar_t local_w[VEC_SIZE];
    
    // ldg.b128 强制 16 Byte 合并读取
    *(float4*)local_x = *(float4 const*)(head_ptr + vec_offset);
    *(float4*)local_w = *(float4 const*)(weight + vec_offset); 

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        float x = static_cast<float>(local_x[i]);
        float w = static_cast<float>(local_w[i]);
        reg_input[i] = x;
        reg_weight[i] = w;
        ss += x * x;
    }

    #pragma unroll
    for (int mask = THREADS_PER_HEAD / 2; mask > 0; mask >>= 1) {
        ss += __shfl_xor_sync(0xffffffff, ss, mask);
    }
    

    float rms = rsqrtf(ss / static_cast<float>(head_dim) + eps);

    float normed[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        // Gemma 公式: (1.0 + weight)
        normed[i] = reg_input[i] * rms * (1.0f + reg_weight[i]);
    }


    int half_d = head_dim / 2;
    int logical_idx = (vec_offset < half_d) ? vec_offset : (vec_offset - half_d);
    
    int cos_base = position * head_dim + logical_idx;
    int sin_base = cos_base + half_d;

    scalar_t local_cos[VEC_SIZE];
    scalar_t local_sin[VEC_SIZE];
    *(float4*)local_cos = *(float4 const*)(cos_sin_cache + cos_base);
    *(float4*)local_sin = *(float4 const*)(cos_sin_cache + sin_base);


    float partner_normed[VEC_SIZE];
    int swap_mask = THREADS_PER_HEAD / 2; 

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        partner_normed[i] = __shfl_xor_sync(0xffffffff, normed[i], swap_mask);
    }

    scalar_t out[VEC_SIZE];
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; i++) {
        float self_val = normed[i];
        float partner_val = partner_normed[i];

        float cos_v = static_cast<float>(local_cos[i]);
        float sin_v = static_cast<float>(local_sin[i]);

        float rotated_val;
        // Neox Style: 前一半实部，后一半虚部
        if (lane_idx < swap_mask) {
            rotated_val = self_val * cos_v - partner_val * sin_v;
        } else {
            rotated_val = partner_val * sin_v + self_val * cos_v;
        }
        out[i] = static_cast<scalar_t>(rotated_val);
    }

    *(float4*)(out_ptr + vec_offset) = *(float4*)out;
}


void gemma_fused_rmsnorm_rope_no_pack_out(
    at::Tensor const& qkv,
    at::Tensor& q,
    at::Tensor& k,
    at::Tensor& v,
    at::Tensor const& q_weight,
    at::Tensor const& k_weight,
    at::Tensor const& positions,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_dim,
    double eps,
    at::Tensor const& cos_sin_cache) 
{
    TORCH_CHECK(head_dim % 8 == 0, "head_dim must be a multiple of 8");
    TORCH_CHECK(head_dim <= 256, "head_dim > 256 requires wider warp reduction handling");

    const c10::cuda::OptionalCUDAGuard device_guard(device_of(qkv));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int32_t num_tokens = qkv.size(0);
    int32_t num_heads = q_size / head_dim;
    int32_t num_kv_heads = kv_size / head_dim;
    int64_t qkv_stride_batch = qkv.stride(0);
    int64_t q_stride_batch = q.stride(0);
    int64_t k_stride_batch = k.stride(0);
    int64_t v_stride_batch = v.stride(0);

    // 总任务数：每个token 下的 Q、K和V的 head 数量之和
    int total_tasks = num_tokens * (num_heads + 2 * num_kv_heads);
    

    constexpr int BLOCK_THREADS = 512;
    int threads_per_head = head_dim / 8; // VEC_SIZE = 8
    int heads_per_block = BLOCK_THREADS / threads_per_head;
    
    dim3 block(BLOCK_THREADS);

    dim3 grid((total_tasks + heads_per_block - 1) / heads_per_block);

    auto dispatch_kernel = [&](auto scalar_type_tag) {
        using scalar_t = decltype(scalar_type_tag);
        if (head_dim == 128) {  // THREADS_PER_HEAD = 16
            fused_gemma_rmsnorm_rope_neox_opt_kernel_no_pack<scalar_t, 8, 16><<<grid, block, 0, stream>>>(
                reinterpret_cast<scalar_t const*>(qkv.data_ptr()), reinterpret_cast<scalar_t*>(q.data_ptr()),
                reinterpret_cast<scalar_t*>(k.data_ptr()), reinterpret_cast<scalar_t*>(v.data_ptr()),
                reinterpret_cast<scalar_t const*>(q_weight.data_ptr()), reinterpret_cast<scalar_t const*>(k_weight.data_ptr()),
                reinterpret_cast<scalar_t const*>(cos_sin_cache.data_ptr()), reinterpret_cast<int64_t const*>(positions.data_ptr()),
                static_cast<float>(eps), qkv_stride_batch, q_stride_batch, k_stride_batch, v_stride_batch,
                q_size, kv_size, num_heads, num_kv_heads, head_dim, total_tasks);
        } 
        else if (head_dim == 64) { // THREADS_PER_HEAD = 8
            fused_gemma_rmsnorm_rope_neox_opt_kernel_no_pack<scalar_t, 8, 8><<<grid, block, 0, stream>>>(
                reinterpret_cast<scalar_t const*>(qkv.data_ptr()), reinterpret_cast<scalar_t*>(q.data_ptr()),
                reinterpret_cast<scalar_t*>(k.data_ptr()), reinterpret_cast<scalar_t*>(v.data_ptr()),
                reinterpret_cast<scalar_t const*>(q_weight.data_ptr()), reinterpret_cast<scalar_t const*>(k_weight.data_ptr()),
                reinterpret_cast<scalar_t const*>(cos_sin_cache.data_ptr()), reinterpret_cast<int64_t const*>(positions.data_ptr()),
                static_cast<float>(eps), qkv_stride_batch, q_stride_batch, k_stride_batch, v_stride_batch,
                q_size, kv_size, num_heads, num_kv_heads, head_dim, total_tasks);
        } 
        else if (head_dim == 256) { // THREADS_PER_HEAD = 32
            fused_gemma_rmsnorm_rope_neox_opt_kernel_no_pack<scalar_t, 8, 32><<<grid, block, 0, stream>>>(
                reinterpret_cast<scalar_t const*>(qkv.data_ptr()), reinterpret_cast<scalar_t*>(q.data_ptr()),
                reinterpret_cast<scalar_t*>(k.data_ptr()), reinterpret_cast<scalar_t*>(v.data_ptr()),
                reinterpret_cast<scalar_t const*>(q_weight.data_ptr()), reinterpret_cast<scalar_t const*>(k_weight.data_ptr()),
                reinterpret_cast<scalar_t const*>(cos_sin_cache.data_ptr()), reinterpret_cast<int64_t const*>(positions.data_ptr()),
                static_cast<float>(eps), qkv_stride_batch, q_stride_batch, k_stride_batch, v_stride_batch,
                q_size, kv_size, num_heads, num_kv_heads, head_dim, total_tasks);
        } else {
            TORCH_CHECK(false, "head_dim must be 64, 128, or 256 for this highly optimized kernel.");
        }
    };

    if (qkv.dtype() == at::ScalarType::BFloat16) {
        dispatch_kernel(__nv_bfloat16());
    } else if (qkv.dtype() == at::ScalarType::Half) {
        dispatch_kernel(__half());
    } else {
        TORCH_CHECK(false, "Only float16 and bfloat16 are supported");
    }
}


std::vector<at::Tensor> gemma_fused_rmsnorm_rope_no_pack(
    at::Tensor const& qkv,
    at::Tensor const& q_weight,
    at::Tensor const& k_weight,
    at::Tensor const& positions,
    int64_t q_size,
    int64_t kv_size,
    int64_t head_dim,
    double eps,
    at::Tensor const& cos_sin_cache)
{
    at::Tensor q = at::empty({qkv.size(0), q_size}, qkv.options());
    at::Tensor k = at::empty({qkv.size(0), kv_size}, qkv.options());
    at::Tensor v = at::empty({qkv.size(0), kv_size}, qkv.options());

    gemma_fused_rmsnorm_rope_no_pack_out(
        qkv,
        q,
        k,
        v,
        q_weight,
        k_weight,
        positions,
        q_size,
        kv_size,
        head_dim,
        eps,
        cos_sin_cache);

    return {q, k, v};
}