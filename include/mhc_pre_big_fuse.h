#pragma once

#include <torch/extension.h>

// Legacy entry: hidden in {4096, 7168}, no carried pre-mix, no fused RMSNorm.
//
// Extended entry (same function): hidden == 5120 with hc_mult == 4 and
// n_splits in {1, 4, 16} additionally supports the delayed mHC pre contract
// used by vLLM-metax `mhc_pre_delayed_tilelang` / `MHC_PRE_NORM_KERNEL`:
// carried pre-mix in/out, fused RMSNorm with the delayed BF16-before-RMS
// square sum, and an optional plain-stream-mean aux output.
//
// The extra arguments are appended after the legacy parameter list and are
// optional, so existing positional call sites keep working unchanged.  Their
// presence selects the semantics:
//   norm_weight -> fuse RMSNorm into the layer_input write path
//   pre_mix_in  -> active pre-mix comes from the caller (Path B)
//   pre_mix_out -> current pre logits are saved as the next pre-mix and,
//                  without pre_mix_in, the collapse uses the one-hot
//                  [1, 0, 0, 0] sentinel (Path A)
//   aux_out     -> write the mean over the four residual streams
//   residual_out -> select the first-layer broadcast contract: residual is
//                   [N, H], residual_out is filled with four identical streams,
//                   and rms_numel defaults to H instead of 4 * H
//   quant_scales -> enables per-token INT8 output when layer_input is INT8
//                   [N, H], or packed INT8 when layer_input is BF16
//                   [N, 2816]. Both modes require norm_weight. The packed
//                   buffer stores H INT8 bytes, one FP32 scale, then padding.
//                   BF16 [N, H] without quant_scales keeps the original path.
//   output_rms -> required for both INT8 modes; additionally expose the BF16
//                 RMSNorm result with shape [N, H].
//   output_quant_int8 -> required only for packed INT8 output; additionally
//                        expose the unpacked INT8 values with shape [N, H].
void mhc_pre_big_fuse_out(
    torch::Tensor gemm_out_mul, torch::Tensor gemm_out_sqrsum,
    torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor residual,
    torch::Tensor post_mix, torch::Tensor comb_mix, torch::Tensor layer_input,
    double rms_eps, double hc_pre_eps, double hc_sinkhorn_eps,
    double hc_post_mult_value, int64_t sinkhorn_repeat, int64_t n_splits,
    c10::optional<torch::Tensor> norm_weight = c10::nullopt,
    c10::optional<torch::Tensor> pre_mix_in = c10::nullopt,
    c10::optional<torch::Tensor> pre_mix_out = c10::nullopt,
    c10::optional<torch::Tensor> aux_out = c10::nullopt,
    double norm_eps = 1e-6, int64_t rms_numel = -1,
    c10::optional<torch::Tensor> residual_out = c10::nullopt,
    c10::optional<torch::Tensor> quant_scales = c10::nullopt,
    c10::optional<torch::Tensor> output_rms = c10::nullopt,
    c10::optional<torch::Tensor> output_quant_int8 = c10::nullopt);