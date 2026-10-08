import math

import torch

from mcoplib.tilelang_mhc_post import mhc_post
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


def _torch_reference(
    layer_output,
    residual,
    post_layer_mix,
    comb_res_mix,
):
    """FP32 accumulation reference followed by the kernel's BF16 cast."""
    output = torch.bmm(comb_res_mix.transpose(1, 2), residual.float())
    output.add_(post_layer_mix * layer_output.float().unsqueeze(1))
    return output.to(residual.dtype)


class Mhc_post_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.hc_mult = config.get("hc_mult", 4)
        self.hidden_size = config.get("hidden_size", 4096)

        if self.dtype != torch.bfloat16:
            raise ValueError("mhc_post benchmark only supports bfloat16 activations")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16/fp32_mix")
        state.add_summary(
            "Shape",
            f"({self.num_tokens} {self.hc_mult} {self.hidden_size})",
        )
        state.add_summary("Kernel", "split_hidden")

        residual_elems = self.num_tokens * self.hc_mult * self.hidden_size
        layer_output_elems = self.num_tokens * self.hidden_size
        post_mix_elems = self.num_tokens * self.hc_mult
        comb_mix_elems = self.num_tokens * self.hc_mult * self.hc_mult
        output_elems = residual_elems
        hidden_tiles = self.hidden_size // math.gcd(self.hidden_size, 512)

        total_elems = (
            residual_elems
            + layer_output_elems
            + post_mix_elems
            + comb_mix_elems
            + output_elems
        )
        read_bytes = (
            residual_elems * torch.bfloat16.itemsize
            + layer_output_elems * torch.bfloat16.itemsize
            + post_mix_elems * hidden_tiles * torch.float32.itemsize
            + comb_mix_elems * hidden_tiles * torch.float32.itemsize
        )
        write_bytes = output_elems * torch.bfloat16.itemsize

        state.add_element_count(total_elems)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _prepare(self, dev_id, num_tokens=None, seed=42):
        num_tokens = self.num_tokens if num_tokens is None else num_tokens
        device = f"cuda:{dev_id}"
        torch.manual_seed(seed)

        comb_res_mix = torch.rand(
            (num_tokens, self.hc_mult, self.hc_mult),
            dtype=torch.float32,
            device=device,
        )
        comb_res_mix /= comb_res_mix.sum(dim=1, keepdim=True)
        residual = torch.randn(
            (num_tokens, self.hc_mult, self.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        post_layer_mix = 2.0 * torch.rand(
            (num_tokens, self.hc_mult, 1),
            dtype=torch.float32,
            device=device,
        )
        layer_output = torch.randn(
            (num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device,
        )
        return layer_output, residual, post_layer_mix, comb_res_mix

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            args = self._prepare(dev_id)
        return self.make_launcher(dev_id, mhc_post, *args)

    def run_verification(self, dev_id):
        verification_tokens = min(self.num_tokens, 128)
        args = self._prepare(dev_id, num_tokens=verification_tokens, seed=7)
        layer_output, residual, post_layer_mix, comb_res_mix = args

        output = mhc_post(*args)
        expected = _torch_reference(
            layer_output,
            residual,
            post_layer_mix,
            comb_res_mix,
        )
        torch.cuda.synchronize()

        close = torch.allclose(output, expected, rtol=2e-2, atol=2e-2)
        cosine_passed, diff_val = self.check_diff(
            output,
            expected,
            threshold=0.9999,
        )
        return bool(close and cosine_passed), diff_val
