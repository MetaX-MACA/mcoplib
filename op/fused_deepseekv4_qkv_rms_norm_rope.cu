#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef __shfl_down_sync_16
#define __shfl_down_sync_16(mask, val, offset) __shfl_down_sync(mask, val, offset, 16)
#endif

template <typename scalar_t, int VEC_SIZE = 8, int NUM_THREADS = 64>
__global__ void fused_rms_norm_rope_kernel_metax_opt(
    scalar_t* __restrict__ q,
    scalar_t* __restrict__ kv,
    scalar_t const* __restrict__ weight_q,  
    scalar_t const* __restrict__ weight_kv, 
    float2 const* __restrict__ freqs_cis, 
    int64_t const* __restrict__ positions,
    float const eps,
    int64_t q_stride_batch,
    int64_t kv_stride_batch,
    int64_t q_stride_head,
    int32_t const num_heads,
    int32_t const head_dim,
    int32_t const kv_dim,
    int32_t const qk_rope_head_dim,
    int32_t const batch_size,
    bool q_is_3d) 
{
    int block_id = blockIdx.x;
    int num_q_blocks = batch_size * num_heads;
    
    scalar_t* ptr_input;
    scalar_t const* cur_weight_ptr; // 当前Block对应的weight指针
    uint32_t d;
    int64_t position;

    // 路由分发: 判断当前是 Q 还是 KV
    if (block_id < num_q_blocks) {
        int batch_idx = block_id / num_heads;
        int head_idx = block_id % num_heads;
        position = positions[batch_idx];
        d = head_dim;
        int64_t offset = q_is_3d ? (head_idx * q_stride_head) : (head_idx * head_dim);
        ptr_input = q + batch_idx * q_stride_batch + offset;
        cur_weight_ptr = weight_q;
    } else {
        int batch_idx = block_id - num_q_blocks;
        if (batch_idx >= batch_size) return;
        position = positions[batch_idx];
        d = kv_dim;
        ptr_input = kv + batch_idx * kv_stride_batch;
        cur_weight_ptr = weight_kv;
    }

    // =====================================================
    // SREG 缓存与平方和计算
    // =====================================================
    float ss = 0.0f;
    uint32_t tid = threadIdx.x * VEC_SIZE;
    uint32_t block_stride = NUM_THREADS * VEC_SIZE;
    uint32_t k = 0;

    constexpr int MAX_REG = 64; 
    float reg_input[MAX_REG][VEC_SIZE];

    for (uint32_t i = tid; i < d; i += block_stride) {
        scalar_t local[VEC_SIZE];
        // 128-bit 强制内存合并加载
        *(float4*)local = *(float4*)(ptr_input + i);

        #pragma unroll VEC_SIZE
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            float x = static_cast<float>(local[j]);
            ss += x * x;
            reg_input[k][j] = x; 
        }
        k++;
    }

    constexpr int sm_size = NUM_THREADS >> 4; 
    __shared__ float sm_sum[sm_size];

    #pragma unroll
    for (int i = 8; i > 0; i >>= 1) {
        ss += __shfl_down_sync_16(0xffffffffffffffff, ss, i);
    }
    
    int lane_id = threadIdx.x & 15;
    int group_id = threadIdx.x >> 4;
    if (lane_id == 0) sm_sum[group_id] = ss;
    __syncthreads();
    
    // 组间归约
    if (threadIdx.x < sm_size) {
        float data = sm_sum[threadIdx.x];
        #pragma unroll
        for (int i = 2; i >= 1; i >>= 1) {
            data += __shfl_down_sync_16(0xffffffffffffffff, data, i);
        }
        if (threadIdx.x == 0) sm_sum[0] = data;
    }
    __syncthreads();
    ss = sm_sum[0];

    __shared__ float s_rms;
    if (threadIdx.x == 0) {
        s_rms = rsqrtf(ss / static_cast<float>(d) + eps);
    }
    __syncthreads();
    float rms = s_rms;

    // =====================================================
    // 共享内存加载 RoPE 旋转矩阵
    // =====================================================
    int rope_start = d - qk_rope_head_dim;
    int rope_half_dim = qk_rope_head_dim / 2;
    int64_t freqs_offset = position * rope_half_dim;

    __shared__ float smem_cos[128];
    __shared__ float smem_sin[128];
    
    for (int i = threadIdx.x; i < rope_half_dim && i < 128; i += NUM_THREADS) {
        float2 cplx = freqs_cis[freqs_offset + i];
        smem_cos[i] = cplx.x;
        smem_sin[i] = cplx.y;
    }
    __syncthreads();

    // =====================================================
    //  128-bit 写回
    // =====================================================
    k = 0;
    for (uint32_t i = tid; i < d; i += block_stride) {
        scalar_t reg_dst[VEC_SIZE];
        scalar_t weight_local[VEC_SIZE];

        // 128-bit 合并读取 weight (如果有)
        if (cur_weight_ptr != nullptr) {
            *(float4*)weight_local = *(float4*)(cur_weight_ptr + i);
        }

        #pragma unroll VEC_SIZE
        for (uint32_t j = 0; j < VEC_SIZE; j++) {
            float w = cur_weight_ptr != nullptr ? static_cast<float>(weight_local[j]) : 1.0f;
            float val = reg_input[k][j] * rms * w;
            
            int elem_idx = i + j;
            if (elem_idx >= rope_start) {
                int rope_idx = elem_idx - rope_start;
                int cos_sin_idx = rope_idx / 2; 
                bool is_real = (rope_idx % 2 == 0); // 偶数索引是实部，奇数索引是虚部

                float cos_val = smem_cos[cos_sin_idx];
                float sin_val = smem_sin[cos_sin_idx];

                // VEC_SIZE=8 且 rope_start 是 8 的倍数，实部(0,2,4,6)和虚部(1,3,5,7)必处于同一个寄存器数组中
                int partner_j = is_real ? (j + 1) : (j - 1);
                
                float partner_w = cur_weight_ptr != nullptr ? static_cast<float>(weight_local[partner_j]) : 1.0f;
                float partner_val = reg_input[k][partner_j] * rms * partner_w;

                if (is_real) {
                    val = val * cos_val - partner_val * sin_val;
                } else {
                    val = partner_val * sin_val + val * cos_val;
                }
            }
            reg_dst[j] = static_cast<scalar_t>(val);
        }
        
        // 128-bit 合并写回
        *(float4*)(ptr_input + i) = *(float4*)reg_dst;
        k++;
    }
}

// =====================================================
// Dispatch function
// =====================================================
void fused_rms_norm_rope(
    at::Tensor& q,
    at::Tensor& kv,
    at::Tensor const& positions,
    at::Tensor const& freqs_cis,
    int64_t qk_rope_head_dim,
    double eps,
    c10::optional<at::Tensor> weight_q = c10::nullopt,
    c10::optional<at::Tensor> weight_kv = c10::nullopt) 
{

  TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
  TORCH_CHECK(kv.is_cuda(), "kv must be a CUDA tensor");
  TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
  TORCH_CHECK(freqs_cis.is_cuda(), "freqs_cis must be a CUDA tensor");
  TORCH_CHECK(positions.is_contiguous(), "positions must be contiguous");
  
  float2 const* freqs_ptr = reinterpret_cast<float2 const*>(freqs_cis.data_ptr());

  int32_t batch_size = q.size(0);
  int32_t num_heads, head_dim, kv_dim;
  int64_t q_stride_batch, q_stride_head;
  bool q_is_3d = (q.dim() == 3);

  if (q_is_3d) {
    num_heads = q.size(1);
    head_dim = q.size(2);
    q_stride_batch = q.stride(0);
    q_stride_head = q.stride(1);
  } else {
    head_dim = 64; 
    int32_t q_total_dim = q.size(1);
    num_heads = q_total_dim / head_dim;
    q_stride_batch = q.stride(0);
    q_stride_head = head_dim;  
  }
  
  kv_dim = kv.size(1); 
  int64_t kv_stride_batch = kv.stride(0);

  TORCH_CHECK(head_dim % 8 == 0, "head_dim must be a multiple of 8 for 128-bit vectorized processing");
  TORCH_CHECK(kv_dim % 8 == 0, "kv_dim must be a multiple of 8 for 128-bit vectorized processing");
  TORCH_CHECK(qk_rope_head_dim % 8 == 0, "qk_rope_head_dim must be a multiple of 8 to ensure safe interleaved offset");


  if (weight_q.has_value()) {
      TORCH_CHECK(weight_q->is_cuda(), "weight_q must be a CUDA tensor");
      TORCH_CHECK(weight_q->dtype() == q.dtype(), "weight_q dtype must match q dtype");
      TORCH_CHECK(weight_q->dim() == 1, "weight_q must be 1D");
      TORCH_CHECK(weight_q->size(0) == head_dim, "weight_q size must match head_dim");
      TORCH_CHECK(weight_q->is_contiguous(), "weight_q must be contiguous");
  }
  if (weight_kv.has_value()) {
      TORCH_CHECK(weight_kv->is_cuda(), "weight_kv must be a CUDA tensor");
      TORCH_CHECK(weight_kv->dtype() == kv.dtype(), "weight_kv dtype must match kv dtype");
      TORCH_CHECK(weight_kv->dim() == 1, "weight_kv must be 1D");
      TORCH_CHECK(weight_kv->size(0) == kv_dim, "weight_kv size must match kv_dim");
      TORCH_CHECK(weight_kv->is_contiguous(), "weight_kv must be contiguous");
  }

  const c10::cuda::OptionalCUDAGuard device_guard(device_of(q));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  int num_q_blocks = batch_size * num_heads;
  int num_kv_blocks = batch_size;
  int total_blocks = num_q_blocks + num_kv_blocks;

  dim3 grid_size_opt(total_blocks);
  dim3 block_size_opt(64); 

  // 获取底层指针
  void* w_q_ptr = weight_q.has_value() ? weight_q->data_ptr() : nullptr;
  void* w_kv_ptr = weight_kv.has_value() ? weight_kv->data_ptr() : nullptr;

  if (q.dtype() == at::ScalarType::Half) {
    fused_rms_norm_rope_kernel_metax_opt<__half, 8, 64><<<grid_size_opt, block_size_opt, 0, stream>>>(
        reinterpret_cast<__half*>(q.data_ptr()),
        reinterpret_cast<__half*>(kv.data_ptr()),
        reinterpret_cast<__half const*>(w_q_ptr),
        reinterpret_cast<__half const*>(w_kv_ptr),
        freqs_ptr,
        reinterpret_cast<int64_t const*>(positions.data_ptr()),
        static_cast<float>(eps),
        q_stride_batch, kv_stride_batch, q_stride_head,
        num_heads, head_dim, kv_dim,
        static_cast<int32_t>(qk_rope_head_dim), batch_size, q_is_3d
    );
  } else if (q.dtype() == at::ScalarType::BFloat16) {
    fused_rms_norm_rope_kernel_metax_opt<__nv_bfloat16, 8, 64><<<grid_size_opt, block_size_opt, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(kv.data_ptr()),
        reinterpret_cast<__nv_bfloat16 const*>(w_q_ptr),
        reinterpret_cast<__nv_bfloat16 const*>(w_kv_ptr),
        freqs_ptr,
        reinterpret_cast<int64_t const*>(positions.data_ptr()),
        static_cast<float>(eps),
        q_stride_batch, kv_stride_batch, q_stride_head,
        num_heads, head_dim, kv_dim,
        static_cast<int32_t>(qk_rope_head_dim), batch_size, q_is_3d
    );
  } else {
    TORCH_CHECK(false, "Only float16 and bfloat16 are supported for fused_rms_norm_rope");
  }
}