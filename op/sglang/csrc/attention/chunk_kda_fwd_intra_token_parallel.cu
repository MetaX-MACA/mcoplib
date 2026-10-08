#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {
constexpr int K = 128;
constexpr int BC = 16;
constexpr int PADK = 128;

__device__ __forceinline__ float lane8_sum(float x, int lane) {
  (void)lane;
  x += __shfl_xor_sync(0xffffffffu, x, 4);
  x += __shfl_xor_sync(0xffffffffu, x, 2);
  x += __shfl_xor_sync(0xffffffffu, x, 1);
  return x;
}

__global__ void chunk_kda_fwd_kernel_intra_token_parallel_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const float* __restrict__ g,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ aqk,
    float* __restrict__ akk,
    int tokens, int heads, float scale) {
  const int head = blockIdx.y;
  const int off = blockIdx.x * BC;
  const int n = min(BC, tokens - off);
  if (head >= heads || n <= 0) return;

  __shared__ float qg[BC][PADK];
  __shared__ float kg[BC][PADK];
  __shared__ float kbg[BC][PADK];

  for (int x = threadIdx.x; x < BC * K; x += blockDim.x) {
    const int row = x / K;
    const int col = x - row * K;
    float qv = 0.f, kv = 0.f, gv = 0.f, bv = 0.f, g0 = 0.f;
    if (row < n) {
      const int64_t idx = (int64_t(off + row) * heads + head) * K + col;
      const int64_t idx0 = (int64_t(off) * heads + head) * K + col;
      qv = __bfloat162float(q[idx]);
      kv = __bfloat162float(k[idx]);
      gv = g[idx];
      g0 = g[idx0];
      bv = beta[(int64_t(off + row) * heads + head)];
    }
    const float dp = row < n ? exp2f(gv - g0) : 0.f;
    const float dm = row < n ? 1.f / dp : 0.f;
    qg[row][col] = qv * dp;
    kg[row][col] = kv * dm;
    kbg[row][col] = kv * bv * dp;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int NWARP = 8;
  for (int i = warp; i < n; i += NWARP) {
    const int group = lane >> 3;
    const int col0 = lane & 7;
    float lhs_q[16], lhs_k[16];
#pragma unroll
    for (int t = 0; t < 16; ++t) {
      lhs_q[t] = qg[i][col0 + t * 8];
      lhs_k[t] = kbg[i][col0 + t * 8];
    }
#pragma unroll
    for (int jj = 0; jj < 4; ++jj) {
      const int j = jj * 4 + group;
      if (j <= i) {
        float sq = 0.f, sk = 0.f;
#pragma unroll
        for (int t = 0; t < 16; ++t) {
          const float rhs = kg[j][col0 + t * 8];
          sq = fmaf(lhs_q[t], rhs, sq);
          sk = fmaf(lhs_k[t], rhs, sk);
        }
        sq = lane8_sum(sq, lane);
        sk = lane8_sum(sk, lane);
        if ((lane & 7) == 0) {
          const int64_t aq_idx = (int64_t(off + i) * heads + head) * 64
                               + ((off & 63) + j);
          const int64_t ak_idx = (int64_t(off + i) * heads + head) * BC + j;
          aqk[aq_idx] = __float2bfloat16(sq * scale);
          akk[ak_idx] = (j < i) ? sk : 0.f;
        }
      }
    }
  }
}
}  // namespace

void chunk_kda_fwd_intra_token_parallel(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& g,
    const at::Tensor& beta, const at::Tensor& aqk, const at::Tensor& akk,
    double scale) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && g.is_cuda() && beta.is_cuda(), "inputs must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 && k.scalar_type() == torch::kBFloat16, "q/k must be BF16");
  TORCH_CHECK(g.scalar_type() == torch::kFloat32 && beta.scalar_type() == torch::kFloat32, "g/beta must be FP32");
  TORCH_CHECK(aqk.scalar_type() == torch::kBFloat16 && akk.scalar_type() == torch::kFloat32, "invalid output dtype");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1 && q.size(3) == K, "specialized path requires [1,T,H,128]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && g.is_contiguous() && beta.is_contiguous(), "inputs must be contiguous");
  const int tokens = q.size(1), heads = q.size(2);
  dim3 grid((tokens + BC - 1) / BC, heads);
  chunk_kda_fwd_kernel_intra_token_parallel_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(k.data_ptr()), g.data_ptr<float>(),
      beta.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(aqk.data_ptr()),
      akk.data_ptr<float>(), tokens, heads, static_cast<float>(scale));
}

#ifdef KDA_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("chunk_kda_fwd_intra_token_parallel", &chunk_kda_fwd_intra_token_parallel);
}
#endif
