#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cstdint>
#include <cstring>
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

void check_optional_tensor(const c10::optional<torch::Tensor>& tensor,
                           at::ScalarType dtype, const char* name) {
  check_tensor(*tensor, dtype, name);
}

bool ranges_overlap(const torch::Tensor& lhs, const torch::Tensor& rhs) {
  if (lhs.device() != rhs.device() || lhs.numel() == 0 || rhs.numel() == 0) {
    return false;
  }
  const uintptr_t lhs_begin = reinterpret_cast<uintptr_t>(lhs.data_ptr());
  const uintptr_t rhs_begin = reinterpret_cast<uintptr_t>(rhs.data_ptr());
  const uintptr_t lhs_bytes =
      static_cast<uintptr_t>(lhs.numel()) * lhs.element_size();
  const uintptr_t rhs_bytes =
      static_cast<uintptr_t>(rhs.numel()) * rhs.element_size();
  return lhs_begin < rhs_begin + rhs_bytes && rhs_begin < lhs_begin + lhs_bytes;
}

// Hidden == 5120 specialization.  Covers the delayed mHC pre contract used by
// vLLM-metax `mhc_pre_delayed_tilelang` / `MHC_PRE_NORM_KERNEL`:
//   * runtime n_splits in {1, 4, 16} (the producing GEMM picks the split);
//   * optional carried pre-mix (Path B) and next pre-mix output (both paths);
//   * fused RMSNorm with the delayed BF16-before-RMS square sum;
//   * optional aux output = mean over the four residual streams.
void mhc_pre_big_fuse_extended(
    const torch::Tensor& gemm_out_mul, const torch::Tensor& gemm_out_sqrsum,
    const torch::Tensor& hc_scale, const torch::Tensor& hc_base,
    const torch::Tensor& residual, torch::Tensor& post_mix,
    torch::Tensor& comb_mix, torch::Tensor& layer_input,
    const c10::optional<torch::Tensor>& norm_weight,
    const c10::optional<torch::Tensor>& pre_mix_in,
    const c10::optional<torch::Tensor>& pre_mix_out,
    const c10::optional<torch::Tensor>& aux_out,
    const c10::optional<torch::Tensor>& residual_out,
    const c10::optional<torch::Tensor>& quant_scales,
    const c10::optional<torch::Tensor>& output_rms,
    const c10::optional<torch::Tensor>& output_quant_int8, double rms_eps,
    double hc_pre_eps, double hc_sinkhorn_eps, double hc_post_mult_value,
    int64_t sinkhorn_repeat, int64_t n_splits, double norm_eps,
    int64_t rms_numel) {
  const int64_t num_tokens = residual.size(0);
  const bool broadcast_residual = residual_out.has_value();
  const int64_t hidden_size =
      broadcast_residual ? residual.size(1) : residual.size(2);
  constexpr int64_t kPackedRowBytes = ((5120 + 4 + 511) / 512) * 512;
  constexpr int64_t kPackedRowElements = kPackedRowBytes / 2;
  const int quant_mode =
      layer_input.scalar_type() == at::kChar
          ? 1
          : (layer_input.size(1) == kPackedRowElements ? 2 : 0);
  if (num_tokens == 0) {
    return;
  }

  TORCH_CHECK(n_splits == 1 || n_splits == 4 || n_splits == 16,
              "mHC pre hidden=5120 requires n_splits in {1, 4, 16}; got ",
              n_splits);
  TORCH_CHECK(sinkhorn_repeat == 20,
              "mHC pre hidden=5120 requires sinkhorn_repeat=20; got ",
              sinkhorn_repeat);
  TORCH_CHECK(gemm_out_mul.dim() == 3 && gemm_out_mul.size(0) == n_splits &&
                  gemm_out_mul.size(1) == num_tokens &&
                  gemm_out_mul.size(2) >= 24,
              "gemm_out_mul must have shape [n_splits, N, L] with L >= 24");
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
  TORCH_CHECK(
      layer_input.sizes() == torch::IntArrayRef(
                                 {num_tokens, quant_mode == 2
                                                  ? kPackedRowElements
                                                  : hidden_size}),
      "layer_input has an invalid shape for its output mode");
  const int64_t effective_rms_numel =
      rms_numel <= 0
          ? (broadcast_residual ? hidden_size : 4 * hidden_size)
          : rms_numel;
  TORCH_CHECK(effective_rms_numel == 5120 || effective_rms_numel == 20480,
              "rms_numel must be 5120 or 20480; got ", effective_rms_numel);

  const bool fuse_norm = norm_weight.has_value();
  const bool use_pre_mix_in = pre_mix_in.has_value();
  const bool save_pre_mix = pre_mix_out.has_value();
  const bool write_aux = aux_out.has_value();
  if (quant_mode != 0) {
    TORCH_CHECK(fuse_norm,
                "INT8 and packed INT8 mHC pre output require norm_weight");
    TORCH_CHECK(!broadcast_residual,
                "broadcast mHC pre does not support quantized output");
    TORCH_CHECK(quant_scales.has_value(),
                "quant_scales is required for quantized mHC pre output");
    check_optional_tensor(quant_scales, at::kFloat, "quant_scales");
    TORCH_CHECK(quant_scales->sizes() == torch::IntArrayRef({num_tokens}),
                "quant_scales must have shape [N]");
    TORCH_CHECK(quant_scales->device() == residual.device(),
                "quant_scales must be on the same device as residual");
  } else {
    TORCH_CHECK(!quant_scales.has_value(),
                "quant_scales requires INT8 or packed INT8 layer_input");
  }
  if (quant_mode != 0) {
    TORCH_CHECK(output_rms.has_value(),
                "quantized mHC pre output requires output_rms");
    check_optional_tensor(output_rms, at::kBFloat16, "output_rms");
    TORCH_CHECK(output_rms->sizes() ==
                    torch::IntArrayRef({num_tokens, hidden_size}),
                "output_rms must have shape [N, H]");
    TORCH_CHECK(output_rms->device() == residual.device(),
                "output_rms must be on the same device as residual");
  } else {
    TORCH_CHECK(!output_rms.has_value(),
                "output_rms is only valid for quantized mHC pre output");
  }
  if (quant_mode == 2) {
    TORCH_CHECK(output_quant_int8.has_value(),
                "packed INT8 mHC pre output requires output_quant_int8");
    check_optional_tensor(output_quant_int8, at::kChar,
                          "output_quant_int8");
    TORCH_CHECK(output_quant_int8->sizes() ==
                    torch::IntArrayRef({num_tokens, hidden_size}),
                "output_quant_int8 must have shape [N, H]");
    TORCH_CHECK(output_quant_int8->device() == residual.device(),
                "output_quant_int8 must be on the same device as residual");
  } else {
    TORCH_CHECK(!output_quant_int8.has_value(),
                "output_quant_int8 is only valid for packed INT8 mHC pre "
                "output");
  }

  if (broadcast_residual) {
    check_optional_tensor(residual_out, at::kBFloat16, "residual_out");
    TORCH_CHECK(residual_out->sizes() ==
                    torch::IntArrayRef({num_tokens, 4, hidden_size}),
                "residual_out must have shape [N, 4, H]");
    TORCH_CHECK(residual_out->device() == residual.device(),
                "residual_out must be on the same device as residual");
    TORCH_CHECK(fuse_norm, "broadcast mHC pre requires norm_weight");
    TORCH_CHECK(!use_pre_mix_in && !save_pre_mix && !write_aux,
                "broadcast mHC pre does not support pre_mix_in, pre_mix_out, "
                "or aux_out");
  }

  if (fuse_norm) {
    check_optional_tensor(norm_weight, at::kBFloat16, "norm_weight");
    TORCH_CHECK(norm_weight->sizes() == torch::IntArrayRef({hidden_size}),
                "norm_weight must have shape [H]");
    TORCH_CHECK(norm_weight->device() == residual.device(),
                "norm_weight must be on the same device as residual");
  }
  if (use_pre_mix_in) {
    check_optional_tensor(pre_mix_in, at::kFloat, "pre_mix_in");
    TORCH_CHECK(pre_mix_in->sizes() == torch::IntArrayRef({num_tokens, 4}),
                "pre_mix_in must have shape [N, 4]");
    TORCH_CHECK(pre_mix_in->device() == residual.device(),
                "pre_mix_in must be on the same device as residual");
  }
  if (save_pre_mix) {
    check_optional_tensor(pre_mix_out, at::kFloat, "pre_mix_out");
    TORCH_CHECK(pre_mix_out->sizes() == torch::IntArrayRef({num_tokens, 4}),
                "pre_mix_out must have shape [N, 4]");
    TORCH_CHECK(pre_mix_out->device() == residual.device(),
                "pre_mix_out must be on the same device as residual");
  }
  if (write_aux) {
    check_optional_tensor(aux_out, at::kBFloat16, "aux_out");
    TORCH_CHECK(aux_out->sizes() ==
                    torch::IntArrayRef({num_tokens, hidden_size}),
                "aux_out must have shape [N, H]");
    TORCH_CHECK(aux_out->device() == residual.device(),
                "aux_out must be on the same device as residual");
  }

  const torch::Tensor* outputs[] = {&post_mix, &comb_mix, &layer_input};
  const torch::Tensor* inputs[] = {&gemm_out_mul, &gemm_out_sqrsum, &hc_scale,
                                   &hc_base, &residual};
  for (const auto* out : outputs) {
    for (const auto* in : inputs) {
      TORCH_CHECK(!ranges_overlap(*out, *in),
                  "mHC pre hidden=5120 outputs must not alias inputs");
    }
  }
  TORCH_CHECK(!ranges_overlap(post_mix, comb_mix) &&
                  !ranges_overlap(post_mix, layer_input) &&
                  !ranges_overlap(comb_mix, layer_input),
              "mHC pre hidden=5120 outputs must not alias each other");
  if (quant_mode != 0) {
    auto check_quant_extra_output = [&](const torch::Tensor& extra) {
      for (const auto* in : inputs) {
        TORCH_CHECK(!ranges_overlap(extra, *in),
                    "quantized auxiliary outputs must not alias mHC pre "
                    "inputs");
      }
      for (const auto* out : outputs) {
        TORCH_CHECK(!ranges_overlap(extra, *out),
                    "quantized auxiliary outputs must not alias other mHC "
                    "pre outputs");
      }
      TORCH_CHECK(!ranges_overlap(extra, *quant_scales) &&
                      !ranges_overlap(extra, *norm_weight),
                  "quantized auxiliary outputs must not alias quant_scales "
                  "or norm_weight");
      if (use_pre_mix_in) {
        TORCH_CHECK(!ranges_overlap(extra, *pre_mix_in),
                    "quantized auxiliary outputs must not alias pre_mix_in");
      }
      if (save_pre_mix) {
        TORCH_CHECK(!ranges_overlap(extra, *pre_mix_out),
                    "quantized auxiliary outputs must not alias pre_mix_out");
      }
      if (write_aux) {
        TORCH_CHECK(!ranges_overlap(extra, *aux_out),
                    "quantized auxiliary outputs must not alias aux_out");
      }
    };
    check_quant_extra_output(*output_rms);
    if (quant_mode == 2) {
      check_quant_extra_output(*output_quant_int8);
      TORCH_CHECK(!ranges_overlap(*output_rms, *output_quant_int8),
                  "output_rms and output_quant_int8 must not alias");
    }
  }
  if (quant_mode != 0) {
    for (const auto* in : inputs) {
      TORCH_CHECK(!ranges_overlap(*quant_scales, *in),
                  "quant_scales must not alias mHC pre inputs");
    }
    for (const auto* out : outputs) {
      TORCH_CHECK(!ranges_overlap(*quant_scales, *out),
                  "quant_scales must not alias mHC pre outputs");
    }
    if (fuse_norm) {
      TORCH_CHECK(!ranges_overlap(*quant_scales, *norm_weight) &&
                      !ranges_overlap(layer_input, *norm_weight),
                  "quantized mHC pre outputs must not alias norm_weight");
    }
    if (use_pre_mix_in) {
      TORCH_CHECK(!ranges_overlap(*quant_scales, *pre_mix_in) &&
                      !ranges_overlap(layer_input, *pre_mix_in),
                  "quantized mHC pre outputs must not alias pre_mix_in");
    }
    if (save_pre_mix) {
      TORCH_CHECK(!ranges_overlap(*quant_scales, *pre_mix_out) &&
                      !ranges_overlap(layer_input, *pre_mix_out),
                  "quantized mHC pre outputs must not alias pre_mix_out");
      if (use_pre_mix_in) {
        TORCH_CHECK(!ranges_overlap(*pre_mix_in, *pre_mix_out),
                    "pre_mix_in and pre_mix_out must not alias");
      }
    }
    if (write_aux) {
      TORCH_CHECK(!ranges_overlap(*quant_scales, *aux_out) &&
                      !ranges_overlap(layer_input, *aux_out),
                  "quantized mHC pre outputs must not alias aux_out");
    }
  }
  if (broadcast_residual) {
    const torch::Tensor* broadcast_outputs[] = {&post_mix, &comb_mix,
                                                 &layer_input, &*residual_out};
    for (const auto* out : broadcast_outputs) {
      for (const auto* in : inputs) {
        TORCH_CHECK(!ranges_overlap(*out, *in),
                    "broadcast mHC pre outputs must not alias inputs");
      }
      TORCH_CHECK(!ranges_overlap(*out, *norm_weight),
                  "broadcast mHC pre outputs must not alias norm_weight");
    }
    for (int i = 0; i < 4; ++i) {
      for (int j = i + 1; j < 4; ++j) {
        TORCH_CHECK(!ranges_overlap(*broadcast_outputs[i], *broadcast_outputs[j]),
                    "broadcast mHC pre outputs must not alias each other");
      }
    }
  }

  const float* pre_mix_in_ptr =
      use_pre_mix_in ? pre_mix_in->data_ptr<float>() : nullptr;
  float* pre_mix_out_ptr =
      save_pre_mix ? pre_mix_out->data_ptr<float>() : nullptr;
  const bfloat16_t* norm_weight_ptr =
      fuse_norm ? reinterpret_cast<const bfloat16_t*>(
                      norm_weight->data_ptr<at::BFloat16>())
                : nullptr;
  bfloat16_t* aux_out_ptr =
      write_aux ? reinterpret_cast<bfloat16_t*>(
                      aux_out->data_ptr<at::BFloat16>())
                : nullptr;
  const bfloat16_t* residual_ptr = reinterpret_cast<const bfloat16_t*>(
      residual.data_ptr<at::BFloat16>());
  bfloat16_t* layer_input_ptr = quant_mode == 0
      ? reinterpret_cast<bfloat16_t*>(layer_input.data_ptr<at::BFloat16>())
      : nullptr;
  int8_t* quant_output_ptr = quant_mode != 0
      ? reinterpret_cast<int8_t*>(layer_input.data_ptr())
      : nullptr;
  float* quant_scales_ptr = quant_mode != 0
      ? quant_scales->data_ptr<float>()
      : nullptr;
  bfloat16_t* output_rms_ptr = quant_mode != 0
      ? reinterpret_cast<bfloat16_t*>(output_rms->data_ptr<at::BFloat16>())
      : nullptr;
  int8_t* output_quant_int8_ptr = quant_mode == 2
      ? output_quant_int8->data_ptr<int8_t>()
      : nullptr;
  bfloat16_t* residual_out_ptr =
      broadcast_residual
          ? reinterpret_cast<bfloat16_t*>(
                residual_out->data_ptr<at::BFloat16>())
          : nullptr;

  const int grid = static_cast<int>(num_tokens);
  const int threads = 128;
  const int last_dim = static_cast<int>(gemm_out_mul.size(2));
  // Match the compile-time layout in the selected device image. Only C500
  // images compact the header; C600/C600-U retain the 1024-byte baseline.
  const bool compact_norm_header =
      fuse_norm &&
      std::strstr(at::cuda::getDeviceProperties(residual.get_device())->name,
                  "C500") != nullptr;
  const int collapsed_offset = compact_norm_header ? 128 : 1024;
  const bool cache_norm_tail = quant_mode == 0 && compact_norm_header && save_pre_mix &&
                               !write_aux && n_splits == 16;
  constexpr int kCandidateCachedTailBlocks = 2;
  constexpr int kStream0CachedTailBlocks = 3;
  // The delayed residual warp does not consume the current control values.
  // Keep overlapping optional storage on the P0 path before decoupling warps.
  bool decouple_control = cache_norm_tail;
  if (decouple_control) {
    decouple_control = !ranges_overlap(*pre_mix_out, residual) &&
                       !ranges_overlap(*pre_mix_out, layer_input) &&
                       !ranges_overlap(*pre_mix_out, *norm_weight) &&
                       !ranges_overlap(post_mix, *norm_weight) &&
                       !ranges_overlap(comb_mix, *norm_weight);
    if (use_pre_mix_in) {
      decouple_control = decouple_control &&
                         !ranges_overlap(*pre_mix_out, *pre_mix_in) &&
                         !ranges_overlap(post_mix, *pre_mix_in) &&
                         !ranges_overlap(comb_mix, *pre_mix_in);
    }
  }
  // N3 is validated for stream0 at N=2048; keep all other sizes on P1-B.
  const bool cache_third_tail =
      decouple_control && !use_pre_mix_in && num_tokens == 2048;
  // E1 lets the control wave emit shared norm blocks while the residual wave
  // emits cache2; C4 extends the same dual-wave epilogue to the stream0 N2048
  // instance, whose third block stays in registers. Reordered stores require
  // layer/weight non-aliasing on either path.
  const bool parallel_norm_epilogue =
      decouple_control && num_tokens == 2048 &&
      !ranges_overlap(layer_input, *norm_weight);
  const int cached_tail_blocks =
      cache_norm_tail
          ? (cache_third_tail ? kStream0CachedTailBlocks
                              : kCandidateCachedTailBlocks)
          : 0;
  const int collapsed_hidden = hidden_size - cached_tail_blocks * 1024;
  const int smem_bytes =
      fuse_norm ? static_cast<int>(collapsed_offset + collapsed_hidden * 2 +
                                   (parallel_norm_epilogue ? sizeof(float) : 0))
                : 1024;
  auto stream = at::cuda::getCurrentCUDAStream();

  auto launch = [&](auto fuse_norm_tag, auto save_pre_mix_tag,
                    auto cached_tail_blocks_tag, auto write_aux_tag,
                    auto decouple_control_tag, auto pre_mix_mode_tag,
                    auto prefetch_carried_tag,
                    auto parallel_norm_epilogue_tag) {
    constexpr bool kFuseNorm = decltype(fuse_norm_tag)::value;
    constexpr bool kSavePreMix = decltype(save_pre_mix_tag)::value;
    constexpr int kCachedTailBlocks = decltype(cached_tail_blocks_tag)::value;
    constexpr bool kWriteAux = decltype(write_aux_tag)::value;
    constexpr bool kDecoupleControl = decltype(decouple_control_tag)::value;
    constexpr int kPreMixMode = decltype(pre_mix_mode_tag)::value;
    constexpr bool kPrefetchCarriedChunk = decltype(prefetch_carried_tag)::value;
    constexpr bool kParallelNormEpilogue =
        decltype(parallel_norm_epilogue_tag)::value;
    mhc_pre_big_fuse_ext_kernel<1024, 1, 5120, kFuseNorm, kSavePreMix,
                              kCachedTailBlocks, kWriteAux, kDecoupleControl,
                              kPreMixMode, kPrefetchCarriedChunk,
                              kParallelNormEpilogue>
        <<<grid, threads, smem_bytes, stream>>>(
            comb_mix.data_ptr<float>(), gemm_out_mul.data_ptr<float>(),
            gemm_out_sqrsum.data_ptr<float>(), layer_input_ptr,
            hc_base.data_ptr<float>(), hc_scale.data_ptr<float>(),
            post_mix.data_ptr<float>(), residual_ptr, pre_mix_out_ptr,
            pre_mix_in_ptr, norm_weight_ptr, aux_out_ptr, grid,
            static_cast<int>(n_splits), last_dim,
            static_cast<float>(effective_rms_numel),
            static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
            static_cast<float>(hc_sinkhorn_eps),
            static_cast<float>(hc_post_mult_value),
            static_cast<float>(norm_eps), nullptr, nullptr, nullptr, nullptr,
            nullptr);
  };

  auto launch_broadcast = [&] {
    mhc_pre_big_fuse_ext_kernel<1024, 1, 5120, true, false, 0, false,
                                false, -1, false, false, true>
        <<<grid, threads, smem_bytes, stream>>>(
            comb_mix.data_ptr<float>(), gemm_out_mul.data_ptr<float>(),
            gemm_out_sqrsum.data_ptr<float>(), layer_input_ptr,
            hc_base.data_ptr<float>(), hc_scale.data_ptr<float>(),
            post_mix.data_ptr<float>(), residual_ptr, nullptr, nullptr,
            norm_weight_ptr, nullptr, grid,
            static_cast<int>(n_splits), last_dim,
            static_cast<float>(effective_rms_numel),
            static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
            static_cast<float>(hc_sinkhorn_eps),
            static_cast<float>(hc_post_mult_value),
            static_cast<float>(norm_eps), residual_out_ptr, nullptr, nullptr,
            nullptr, nullptr);
  };

  auto launch_quant = [&](auto save_pre_mix_tag, auto quant_mode_tag) {
    mhc_pre_big_fuse_ext_kernel<
        1024, 1, 5120, true, decltype(save_pre_mix_tag)::value,
        0, true, false, -1, false, false, false,
        decltype(quant_mode_tag)::value>
        <<<grid, threads, smem_bytes, stream>>>(
            comb_mix.data_ptr<float>(), gemm_out_mul.data_ptr<float>(),
            gemm_out_sqrsum.data_ptr<float>(), nullptr,
            hc_base.data_ptr<float>(), hc_scale.data_ptr<float>(),
            post_mix.data_ptr<float>(), residual_ptr, pre_mix_out_ptr,
            pre_mix_in_ptr, norm_weight_ptr, aux_out_ptr, grid,
            static_cast<int>(n_splits), last_dim,
            static_cast<float>(effective_rms_numel),
            static_cast<float>(rms_eps), static_cast<float>(hc_pre_eps),
            static_cast<float>(hc_sinkhorn_eps),
            static_cast<float>(hc_post_mult_value),
            static_cast<float>(norm_eps), nullptr, quant_output_ptr,
            quant_scales_ptr, output_rms_ptr, output_quant_int8_ptr);
  };

  if (quant_mode == 1) {
    if (save_pre_mix) {
      launch_quant(std::true_type{}, std::integral_constant<int, 1>{});
    } else {
      launch_quant(std::false_type{}, std::integral_constant<int, 1>{});
    }
  } else if (quant_mode == 2) {
    if (save_pre_mix) {
      launch_quant(std::true_type{}, std::integral_constant<int, 2>{});
    } else {
      launch_quant(std::false_type{}, std::integral_constant<int, 2>{});
    }
  } else if (broadcast_residual) {
    launch_broadcast();
  } else if (fuse_norm) {
    if (save_pre_mix) {
      if (cache_norm_tail) {
        // This C500 delayed-norm dispatch already requires aux_out == nullptr.
        if (decouple_control) {
          if (use_pre_mix_in) {
            if (num_tokens == 2048) {
              if (parallel_norm_epilogue) {
                launch(std::true_type{}, std::true_type{},
                       std::integral_constant<int,
                                              kCandidateCachedTailBlocks>{},
                       std::false_type{}, std::true_type{},
                       std::integral_constant<int, 1>{}, std::true_type{},
                       std::true_type{});
              } else {
                launch(std::true_type{}, std::true_type{},
                       std::integral_constant<int,
                                              kCandidateCachedTailBlocks>{},
                       std::false_type{}, std::true_type{},
                       std::integral_constant<int, 1>{}, std::true_type{},
                       std::false_type{});
              }
            } else {
              launch(std::true_type{}, std::true_type{},
                     std::integral_constant<int, kCandidateCachedTailBlocks>{},
                     std::false_type{}, std::true_type{},
                     std::integral_constant<int, 1>{}, std::false_type{},
                     std::false_type{});
            }
          } else if (cache_third_tail) {
            if (parallel_norm_epilogue) {
              launch(std::true_type{}, std::true_type{},
                     std::integral_constant<int, kStream0CachedTailBlocks>{},
                     std::false_type{}, std::true_type{},
                     std::integral_constant<int, -1>{}, std::false_type{},
                     std::true_type{});
            } else {
              launch(std::true_type{}, std::true_type{},
                     std::integral_constant<int, kStream0CachedTailBlocks>{},
                     std::false_type{}, std::true_type{},
                     std::integral_constant<int, -1>{}, std::false_type{},
                     std::false_type{});
            }
          } else {
            launch(std::true_type{}, std::true_type{},
                   std::integral_constant<int, kCandidateCachedTailBlocks>{},
                   std::false_type{}, std::true_type{},
                   std::integral_constant<int, -1>{}, std::false_type{},
                   std::false_type{});
          }
        } else {
          launch(std::true_type{}, std::true_type{},
                 std::integral_constant<int, kCandidateCachedTailBlocks>{},
                 std::false_type{}, std::false_type{},
                 std::integral_constant<int, -1>{}, std::false_type{},
                 std::false_type{});
        }
      } else {
        launch(std::true_type{}, std::true_type{},
               std::integral_constant<int, 0>{}, std::true_type{},
               std::false_type{}, std::integral_constant<int, -1>{},
               std::false_type{}, std::false_type{});
      }
    } else {
      launch(std::true_type{}, std::false_type{},
             std::integral_constant<int, 0>{}, std::true_type{},
             std::false_type{}, std::integral_constant<int, -1>{},
             std::false_type{}, std::false_type{});
    }
  } else {
    if (save_pre_mix) {
      launch(std::false_type{}, std::true_type{},
             std::integral_constant<int, 0>{}, std::true_type{},
             std::false_type{}, std::integral_constant<int, -1>{},
             std::false_type{}, std::false_type{});
    } else {
      launch(std::false_type{}, std::false_type{},
             std::integral_constant<int, 0>{}, std::true_type{},
             std::false_type{}, std::integral_constant<int, -1>{},
             std::false_type{}, std::false_type{});
    }
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess,
              "mHC pre hidden=5120 kernel launch failed");
}

}  // namespace

void mhc_pre_big_fuse_out(
    torch::Tensor gemm_out_mul, torch::Tensor gemm_out_sqrsum,
    torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor residual,
    torch::Tensor post_mix, torch::Tensor comb_mix, torch::Tensor layer_input,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, int64_t n_splits,
    c10::optional<torch::Tensor> norm_weight,
    c10::optional<torch::Tensor> pre_mix_in,
    c10::optional<torch::Tensor> pre_mix_out,
    c10::optional<torch::Tensor> aux_out, double norm_eps,
    int64_t rms_numel, c10::optional<torch::Tensor> residual_out,
    c10::optional<torch::Tensor> quant_scales,
    c10::optional<torch::Tensor> output_rms,
    c10::optional<torch::Tensor> output_quant_int8) {
  check_tensor(gemm_out_mul, at::kFloat, "gemm_out_mul");
  check_tensor(gemm_out_sqrsum, at::kFloat, "gemm_out_sqrsum");
  check_tensor(hc_scale, at::kFloat, "hc_scale");
  check_tensor(hc_base, at::kFloat, "hc_base");
  check_tensor(residual, at::kBFloat16, "residual");
  check_tensor(post_mix, at::kFloat, "post_mix");
  check_tensor(comb_mix, at::kFloat, "comb_mix");
  TORCH_CHECK(layer_input.scalar_type() == at::kBFloat16 ||
                  layer_input.scalar_type() == at::kChar,
              "layer_input must be BF16 or INT8");
  TORCH_CHECK(layer_input.is_cuda() && layer_input.is_contiguous(),
              "layer_input must be a contiguous CUDA tensor");
  TORCH_CHECK(layer_input.dim() == 2,
              "layer_input must have shape [N, H] or [N, packed H]");

  const bool broadcast_residual = residual_out.has_value();
  if (broadcast_residual) {
    TORCH_CHECK(residual.dim() == 2,
                "broadcast residual must have shape [N, H]");
  } else {
    TORCH_CHECK(residual.dim() == 3 && residual.size(1) == 4,
                "residual must have shape [N, 4, H]");
  }
  const int64_t hidden_size =
      broadcast_residual ? residual.size(1) : residual.size(2);

  if (hidden_size == 5120) {
    mhc_pre_big_fuse_extended(
        gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, post_mix,
        comb_mix, layer_input, norm_weight, pre_mix_in, pre_mix_out, aux_out,
        residual_out, quant_scales, output_rms, output_quant_int8, rms_eps,
        hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
        n_splits, norm_eps, rms_numel);
    return;
  }

  // ---- legacy path: hidden 4096/7168, no delayed pre-mix, no RMSNorm ----
  TORCH_CHECK(!norm_weight.has_value() && !pre_mix_in.has_value() &&
                  !pre_mix_out.has_value() && !aux_out.has_value() &&
                  !residual_out.has_value() && !quant_scales.has_value() &&
                  !output_rms.has_value() &&
                  !output_quant_int8.has_value() &&
                  layer_input.scalar_type() == at::kBFloat16,
              "extended mHC pre options require hidden_size=5120");
  const int64_t num_tokens = residual.size(0);
  TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
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
