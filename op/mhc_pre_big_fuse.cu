#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <type_traits>

#include "../include/mhc_pre_big_fuse.h"
#include "mhc_pre_big_fuse_kernel.cuh"

namespace {

void check_tensor(const torch::Tensor& tensor, at::ScalarType dtype,
                  const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
}

}  // namespace

void mhc_pre_big_fuse_out(
    torch::Tensor gemm_out_mul, torch::Tensor gemm_out_sqrsum,
    torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor residual,
    torch::Tensor post_mix, torch::Tensor comb_mix, torch::Tensor layer_input,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, int64_t n_splits) {
  check_tensor(gemm_out_mul, at::kFloat, "gemm_out_mul");
  check_tensor(gemm_out_sqrsum, at::kFloat, "gemm_out_sqrsum");
  check_tensor(hc_scale, at::kFloat, "hc_scale");
  check_tensor(hc_base, at::kFloat, "hc_base");
  check_tensor(residual, at::kBFloat16, "residual");
  check_tensor(post_mix, at::kFloat, "post_mix");
  check_tensor(comb_mix, at::kFloat, "comb_mix");
  check_tensor(layer_input, at::kBFloat16, "layer_input");

  const int64_t num_tokens = residual.size(0);
  TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
  TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4,
              "residual must have shape [N, 4, H]");
  const int64_t hidden_size = residual.size(2);
  TORCH_CHECK(hidden_size == 4096 || hidden_size == 7168,
              "mHC pre big fuse supports hidden 4096 or 7168; got ",
              hidden_size);
  TORCH_CHECK(n_splits == 16 || n_splits == 64,
              "mHC pre optimized kernel requires n_splits=16 or 64");
  TORCH_CHECK(gemm_out_mul.sizes() ==
                  torch::IntArrayRef({n_splits, num_tokens, 24}),
              "gemm_out_mul must have shape [n_splits, N, 24]");
  TORCH_CHECK(gemm_out_sqrsum.sizes() ==
                  torch::IntArrayRef({n_splits, num_tokens}),
              "gemm_out_sqrsum must have shape [n_splits, N]");
  TORCH_CHECK(hc_scale.sizes() == torch::IntArrayRef({3}),
              "hc_scale must have shape [3]");
  TORCH_CHECK(hc_base.sizes() == torch::IntArrayRef({24}),
              "hc_base must have shape [24]");
  TORCH_CHECK(post_mix.sizes() == torch::IntArrayRef({num_tokens, 4}),
              "post_mix must have shape [N, 4]");
  TORCH_CHECK(comb_mix.sizes() == torch::IntArrayRef({num_tokens, 16}),
              "comb_mix must have shape [N, 16]");
  TORCH_CHECK(layer_input.sizes() ==
                  torch::IntArrayRef({num_tokens, hidden_size}),
              "layer_input must have shape [N, H]");
  TORCH_CHECK(rms_eps == 1e-6 && hc_pre_eps == 1e-6 &&
                  hc_sinkhorn_eps == 1e-6 && hc_post_mult_value == 2.0 &&
                  sinkhorn_repeat == 20,
              "mHC pre optimized kernel requires eps=1e-6, post_mult=2, "
              "sinkhorn_repeat=20");

  auto launch_with_hidden = [&](auto hidden, auto split,
                                auto fast_sinkhorn_divide,
                                auto static_tokens) {
    // Residual-warp count is hidden-size dependent: 7168 has 7 hidden-block
    // iterations per CTA, where two streaming warps beat one; 4096 has only 4,
    // so the smaller 128-thread CTA (which allows more resident blocks per AP)
    // wins instead.
    constexpr int residual_warps = decltype(hidden)::value == 4096 ? 1 : 2;
    constexpr int threads = 64 * (1 + residual_warps);
    mhc_pre_big_fuse_kernel_kernel<
        decltype(split)::value, 1024, residual_warps,
        decltype(fast_sinkhorn_divide)::value,
        decltype(static_tokens)::value,
        decltype(hidden)::value><<<
        static_cast<int>(num_tokens), threads, 4320,
        at::cuda::getCurrentCUDAStream()>>>(
        comb_mix.data_ptr<float>(), gemm_out_mul.data_ptr<float>(),
        gemm_out_sqrsum.data_ptr<float>(),
        reinterpret_cast<bfloat16_t*>(layer_input.data_ptr<at::BFloat16>()),
        hc_base.data_ptr<float>(), hc_scale.data_ptr<float>(),
        post_mix.data_ptr<float>(),
        reinterpret_cast<const bfloat16_t*>(
            residual.data_ptr<at::BFloat16>()),
        static_cast<int>(num_tokens));
  };
  auto launch = [&](auto split, auto fast_sinkhorn_divide,
                    auto static_tokens) {
    if (hidden_size == 4096) {
      launch_with_hidden(std::integral_constant<int, 4096>{}, split,
                         fast_sinkhorn_divide, static_tokens);
    } else {
      launch_with_hidden(std::integral_constant<int, 7168>{}, split,
                         fast_sinkhorn_divide, static_tokens);
    }
  };
  if (n_splits == 16) {
    launch(std::integral_constant<int, 16>{}, std::false_type{},
           std::integral_constant<int, 0>{});
  } else if (num_tokens == 6) {
    launch(std::integral_constant<int, 64>{}, std::true_type{},
           std::integral_constant<int, 6>{});
  } else if (num_tokens == 12) {
    launch(std::integral_constant<int, 64>{}, std::true_type{},
           std::integral_constant<int, 12>{});
  } else if (num_tokens < 18) {
    launch(std::integral_constant<int, 64>{}, std::true_type{},
           std::integral_constant<int, 0>{});
  } else if (num_tokens == 18) {
    launch(std::integral_constant<int, 64>{}, std::true_type{},
           std::integral_constant<int, 18>{});
  } else {
    launch(std::integral_constant<int, 64>{}, std::false_type{},
           std::integral_constant<int, 0>{});
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "mHC pre optimized kernel launch failed");
}
