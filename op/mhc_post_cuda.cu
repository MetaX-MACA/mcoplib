#include "mhc_post_cuda.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <utility>

namespace {

// Portable default: HC=4 with complete, 8-byte-aligned packs can use the
// vector kernel on any device. AP-specific tuning overrides only known shapes.
static std::pair<int, int> select_mhc_post_config(
    int ap_count, int64_t n, int64_t hc, int64_t h, uintptr_t alignment_bits) {
  int vector_width = 1, threads = 256;
  if (hc == 4 && h % 4 == 0 && alignment_bits % 8 == 0) {
    vector_width = 4;
    threads = 128;
  }
  if ((ap_count == 28 || ap_count == 32) && hc == 4 &&
      (h == 4096 || h == 7168) && alignment_bits % 16 == 0) {
    // Preserve the measured small-token and large-token policies on MetaX.
    vector_width = 1;
    threads = 256;
    const bool enough_tokens = h == 4096 ? n > 4 : n > 2;
    if (enough_tokens) {
      if (n < 64 || (h == 4096 && n < 128)) {
        vector_width = 4;
        threads = 64;
      } else {
        vector_width = 8;
        threads = h == 4096 ? 256 : 128;
      }
    }
  }
  return {vector_width, threads};
}

// A whole pack belongs to one thread. Alignment and row divisibility are
// checked on the host; the last block masks complete packs, never partial ones.
template<int V> struct alignas(V * 2) Bf16Pack { at::BFloat16 value[V]; };

// The suffix "4" means HC=4, not the vector width (V can be 4 or 8).
template<int V, int THREADS>
__global__ void mhc_post_vector4(const float* __restrict__ a,
    const at::BFloat16* __restrict__ b, const float* __restrict__ c,
    const at::BFloat16* __restrict__ d, at::BFloat16* __restrict__ out,
    int64_t hidden) {
  const int64_t n = blockIdx.x;
  const int64_t h = (int64_t(blockIdx.y) * THREADS + threadIdx.x) * V;
  if (h >= hidden) return;
  const Bf16Pack<V> dp = *reinterpret_cast<const Bf16Pack<V>*>(d + n * hidden + h);
  float acc[4][V];
  #pragma unroll
  for (int o = 0; o < 4; ++o) {
    const float cv = c[n * 4 + o];
    #pragma unroll
    for (int v = 0; v < V; ++v) acc[o][v] = cv * float(dp.value[v]);
  }
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    const Bf16Pack<V> bp = *reinterpret_cast<const Bf16Pack<V>*>(b + (n * 4 + i) * hidden + h);
    #pragma unroll
    for (int o = 0; o < 4; ++o) {
      const float av = a[n * 16 + i * 4 + o];
      #pragma unroll
      for (int v = 0; v < V; ++v)
        acc[o][v] = fmaf(av, float(bp.value[v]), acc[o][v]);
    }
  }
  #pragma unroll
  for (int o = 0; o < 4; ++o) {
    Bf16Pack<V> yp;
    #pragma unroll
    for (int v = 0; v < V; ++v) yp.value[v] = at::BFloat16(acc[o][v]);
    *reinterpret_cast<Bf16Pack<V>*>(out + (n * 4 + o) * hidden + h) = yp;
  }
}

// Each thread owns one hidden coordinate and reuses residual across outputs.
template<int HC>
__global__ void mhc_post_fixed(const float* __restrict__ a,
    const at::BFloat16* __restrict__ b, const float* __restrict__ c,
    const at::BFloat16* __restrict__ d, at::BFloat16* __restrict__ out,
    int64_t hidden) {
  const int64_t n = blockIdx.x;
  const int64_t h = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
  if (h >= hidden) return;
  float acc[HC];
  const float dv = float(d[n * hidden + h]);
  #pragma unroll
  for (int o = 0; o < HC; ++o) acc[o] = c[n * HC + o] * dv;
  #pragma unroll
  for (int i = 0; i < HC; ++i) {
    const float bv = float(b[(n * HC + i) * hidden + h]);
    #pragma unroll
    for (int o = 0; o < HC; ++o)
      acc[o] = fmaf(a[(n * HC + i) * HC + o], bv, acc[o]);
  }
  #pragma unroll
  for (int o = 0; o < HC; ++o)
    out[(n * HC + o) * hidden + h] = at::BFloat16(acc[o]);
}

// Generic fallback avoids an unbounded register array for arbitrary HC.
__global__ void mhc_post_generic(const float* __restrict__ a,
    const at::BFloat16* __restrict__ b, const float* __restrict__ c,
    const at::BFloat16* __restrict__ d, at::BFloat16* __restrict__ out,
    int64_t hidden, int64_t hc) {
  const int64_t n = blockIdx.x;
  const int64_t h = int64_t(blockIdx.y) * blockDim.x + threadIdx.x;
  if (h >= hidden) return;
  const float dv = float(d[n * hidden + h]);
  for (int64_t o = 0; o < hc; ++o) {
    float acc = c[n * hc + o] * dv;
    for (int64_t i = 0; i < hc; ++i)
      acc = fmaf(a[(n * hc + i) * hc + o],
                 float(b[(n * hc + i) * hidden + h]), acc);
    out[(n * hc + o) * hidden + h] = at::BFloat16(acc);
  }
}

torch::Tensor mhc_post_dispatch(torch::Tensor x, torch::Tensor residual,
    torch::Tensor post, torch::Tensor comb, torch::Tensor out,
    int vector_width, int threads) {
  TORCH_CHECK(residual.dim() == 3, "residual must be [N, HC, H]");
  const auto n = residual.size(0), hc = residual.size(1), h = residual.size(2);
  TORCH_CHECK(hc > 0 && h > 0, "HC and H must be positive");
  for (const auto& t : {x, residual, post, comb, out}) {
    TORCH_CHECK(t.is_cuda() && t.device() == residual.device(), "all tensors must be on the same CUDA/MACA device");
    TORCH_CHECK(t.is_contiguous(), "all tensors must be contiguous");
  }
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && residual.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "x, residual, out must be BF16");
  TORCH_CHECK(post.scalar_type() == at::kFloat && comb.scalar_type() == at::kFloat, "mix tensors must be FP32");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == n && x.size(1) == h, "x must be [N, H]");
  TORCH_CHECK((post.dim() == 2 || (post.dim() == 3 && post.size(2) == 1)) && post.size(0) == n && post.size(1) == hc, "post must be [N, HC] or [N, HC, 1]");
  TORCH_CHECK(comb.dim() == 3 && comb.size(0) == n && comb.size(1) == hc && comb.size(2) == hc, "comb must be [N, HC, HC]");
  TORCH_CHECK(out.sizes() == residual.sizes(), "out shape mismatch");
  for (const auto& t : {x, residual, post, comb})
    TORCH_CHECK(!out.is_alias_of(t), "out must not alias an input");
  c10::cuda::CUDAGuard guard(residual.device());
  // -1: production policy; 1: preserved scalar baseline; 4/8: tuning API.
  TORCH_CHECK(vector_width == -1 || vector_width == 1 || vector_width == 4 || vector_width == 8, "invalid vector width");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256, "invalid thread count");
  if (n == 0) return out;
  const auto a = comb.data_ptr<float>();
  const auto b = residual.data_ptr<at::BFloat16>();
  const auto c = post.data_ptr<float>();
  const auto d = x.data_ptr<at::BFloat16>();
  const auto y = out.data_ptr<at::BFloat16>();
  const uintptr_t alignment_bits = reinterpret_cast<uintptr_t>(b) |
      reinterpret_cast<uintptr_t>(d) | reinterpret_cast<uintptr_t>(y);
  if (vector_width == -1) {
    const auto* props = at::cuda::getDeviceProperties(residual.get_device());
    // PyTorch caches cudaGetDeviceProperties for the tensor's actual device.
    // On MetaX, multiProcessorCount is the AP count: C600U=28, C600UL=32.
    const auto config = select_mhc_post_config(
        props->multiProcessorCount, n, hc, h, alignment_bits);
    vector_width = config.first;
    threads = config.second;
  }
  if (vector_width > 1) {
    TORCH_CHECK(hc == 4 && h % vector_width == 0 && alignment_bits % (2 * vector_width) == 0,
                "vector path requires HC=4 and aligned, divisible rows");
    const int tile = threads * vector_width;
    TORCH_CHECK(n <= 2147483647 && (h + tile - 1) / tile <= 65535, "shape exceeds CUDA grid limits");
    dim3 grid(n, (h + tile - 1) / tile);
    auto stream = at::cuda::getCurrentCUDAStream();
    #define VECTOR(V,T) mhc_post_vector4<V,T><<<grid,T,0,stream>>>(a,b,c,d,y,h)
    #define BY_THREADS(V) if (threads == 64) { VECTOR(V,64); } else if (threads == 128) { VECTOR(V,128); } else { VECTOR(V,256); }
    if (vector_width == 4) { BY_THREADS(4); } else { BY_THREADS(8); }
    #undef BY_THREADS
    #undef VECTOR
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
  }
  TORCH_CHECK(n <= 2147483647 && (h + threads - 1) / threads <= 65535, "shape exceeds CUDA grid limits");
  dim3 grid(n, (h + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();
  #define LAUNCH(HC) mhc_post_fixed<HC><<<grid, threads, 0, stream>>>(a,b,c,d,y,h)
  switch (hc) {
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 4: LAUNCH(4); break;
    case 8: LAUNCH(8); break;
    default: mhc_post_generic<<<grid, threads, 0, stream>>>(a,b,c,d,y,h,hc);
  }
  #undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

torch::Tensor mhc_post_cuda(torch::Tensor x, torch::Tensor residual,
    torch::Tensor post_layer_mix, torch::Tensor comb_res_mix) {
  return mhc_post_dispatch(x, residual, post_layer_mix, comb_res_mix,
                           torch::empty_like(residual), -1, 256);
}
