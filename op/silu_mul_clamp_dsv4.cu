#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <cmath>
#include "../kernel/dispatch_utils.h"
#include <maca_fp8.h>
#include "../include/silu_mul_clamp_dsv4.h"
namespace {
__device__ __forceinline__ float silu_mul_clamp_one(float g_raw, float u_raw,
                                                    float limit) {
  const float g = fminf(g_raw, limit);                 // gate: 只截上界（与原版一致）
  const float silu = g * __builtin_mxc_rcpf(1.0f + __builtin_expf(-g));
  const float u = fmaxf(fminf(u_raw, limit), -limit);  // up: 双向截断
  return u * silu;
}

}  // namespace

template <typename T>
__global__ __launch_bounds__(1024) void silu_mul_clamp_kernel(
    const T* __restrict__ input,  // [2M, d]: row 2i = gate, row 2i+1 = up
    T* __restrict__ output,       // [M, d]
    const int d,                  // gate/up 单行元素数
    const float limit) {
    constexpr int kElemsPerVec = 16 / sizeof(T);  // = 8
    const int64_t row = blockIdx.x;  // 每个 block 负责一行输出（同原版 bid）
    const int tid = threadIdx.x;

    const T* __restrict__ gate = input + row * 2 * int64_t(d);
    T* __restrict__ out = output + row * int64_t(d);

    const int n_vec = d / kElemsPerVec;
    for (int v = tid; v < n_vec; v += blockDim.x) {
        // ---- 向量化加载: 每线程一次读 gate 16B + up 16B ----
        const uint4 * ptr_gate_v = reinterpret_cast<const uint4*>(gate + int64_t(v) * kElemsPerVec * 2);
        const uint4 g4 = *ptr_gate_v;
        const uint4 u4 = *(ptr_gate_v + 1);
            
        const T* gp = reinterpret_cast<const T*>(&g4);
        const T* upp = reinterpret_cast<const T*>(&u4);
        uint4 o4;
        T* op = reinterpret_cast<T*>(&o4);
    #pragma unroll
        for (int e = 0; e < kElemsPerVec; ++e) {
        op[e] = static_cast<T>(silu_mul_clamp_one(static_cast<float>(gp[e]),
                                                    static_cast<float>(upp[e]),
                                                    limit));
        }
        // ---- 向量化存储: 一次写 16B ----
        *reinterpret_cast<uint4*>(out + int64_t(v) * kElemsPerVec) = o4;
    } 
}

void silu_and_mul_clamp(const torch::Tensor& input,
                        torch::Tensor& output, float swiglu_limit) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(output.is_contiguous());
  TORCH_CHECK(input.scalar_type() == output.scalar_type());
  TORCH_CHECK(input.element_size() == 2, "input type only support bf16/fp16");

  const int64_t two_d = input.size(-1);
  TORCH_CHECK(two_d % 2 == 0,
                  "input last dim must be even (gate||up), got ", two_d);
  const int64_t d = two_d / 2;
  TORCH_CHECK(d % (16 / input.element_size()) == 0, "output dim should be divided by packed_size");

  const int64_t rows = input.numel() / two_d;
  TORCH_CHECK(d <= INT32_MAX, "d too large: ", d);
  TORCH_CHECK(rows <= INT32_MAX, "too many rows for grid.x: ", rows);
  TORCH_CHECK(output.numel() == rows * d, "output numel mismatch: got ",
                  output.numel(), ", expect ", rows * d);
  if (rows == 0) {
    return;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  constexpr int kThreads = 512;
  const dim3 grid(static_cast<unsigned int>(rows));
  const dim3 block(kThreads);

  MOE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_clamp", ([&] {
        silu_mul_clamp_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                static_cast<const scalar_t*>(input.data_ptr()),
                static_cast<scalar_t*>(output.data_ptr()),
                static_cast<int>(d), swiglu_limit);
      }));
}