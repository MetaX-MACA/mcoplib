# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5


class Kimi_k3_attn_res_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.num_tokens = config.get("batch_size", config.get("num_tokens", 3))
        self.num_blocks = config.get("num_blocks", 8)
        self.has_delta = config.get("has_delta", False)
        self.seed = config.get("seed", 42)

        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {self.num_tokens}")

        if self.num_blocks < 1 or self.num_blocks > MAX_BLOCKS:
            raise ValueError(f"num_blocks must be 1..{MAX_BLOCKS}, got {self.num_blocks}")

        if not hasattr(torch.ops._C, "kimi_k3_attn_res"):
            raise RuntimeError("torch.ops._C.kimi_k3_attn_res is not registered")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "torch.bfloat16")
        state.add_summary("Shape", f"tokens={self.num_tokens} blocks={self.num_blocks} hidden={HIDDEN_SIZE}")

        read_rows = self.num_blocks + 1 + (1 if self.has_delta else 0)
        write_rows = 1 + (1 if self.has_delta else 0)

        state.add_element_count(self.num_tokens * HIDDEN_SIZE)

        state.add_global_memory_reads(self.num_tokens * read_rows * HIDDEN_SIZE * 2)
        state.add_global_memory_writes(self.num_tokens * write_rows * HIDDEN_SIZE * 2)

    def _make_inputs(self, dev):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        prefix = torch.randn((self.num_tokens, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16).contiguous()

        if self.has_delta:
            delta = torch.randn((self.num_tokens, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16).contiguous()
        else:
            delta = torch.empty((0, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16)

        blocks = torch.randn((self.num_tokens, MAX_BLOCKS, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16).contiguous()

        norm_weight = (1.0 + 0.1 * torch.randn(HIDDEN_SIZE, device=dev, dtype=torch.bfloat16)).contiguous()

        qk_weight = (torch.randn(HIDDEN_SIZE, device=dev, dtype=torch.bfloat16) / HIDDEN_SIZE ** 0.5).contiguous()

        output_norm_weight = (1.0 + 0.1 * torch.randn(HIDDEN_SIZE, device=dev, dtype=torch.bfloat16)).contiguous()

        output = torch.empty((self.num_tokens, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16).contiguous()

        return prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.kimi_k3_attn_res, prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output, self.num_blocks, EPS, EPS)

    def _build_reference(self, prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight):
        if delta.numel() != 0:
            prefix = prefix + delta

        values = torch.cat([blocks[:, :self.num_blocks], prefix.unsqueeze(1)], dim=1)

        keys = F.rms_norm(values, (HIDDEN_SIZE,), norm_weight, EPS)

        probs = (keys @ qk_weight).softmax(dim=-1)

        output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)

        if output_norm_weight.numel() != 0:
            output = F.rms_norm(output, (HIDDEN_SIZE,), output_norm_weight, EPS)

        return output

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output = self._make_inputs(dev)

        if prefix.shape != (self.num_tokens, HIDDEN_SIZE):
            return False, float("inf")

        if prefix.stride(0) != HIDDEN_SIZE or prefix.stride(1) != 1:
            return False, float("inf")

        if delta.numel() != 0:
            if delta.shape != (self.num_tokens, HIDDEN_SIZE):
                return False, float("inf")
            if delta.stride(0) != HIDDEN_SIZE or delta.stride(1) != 1:
                return False, float("inf")
        else:
            if delta.shape != (0, HIDDEN_SIZE):
                return False, float("inf")

        if blocks.shape != (self.num_tokens, MAX_BLOCKS, HIDDEN_SIZE):
            return False, float("inf")

        if not blocks.is_contiguous():
            return False, float("inf")

        if blocks.stride(2) != 1:
            return False, float("inf")

        if blocks.stride(1) != HIDDEN_SIZE:
            return False, float("inf")

        if blocks.stride(0) != MAX_BLOCKS * HIDDEN_SIZE:
            return False, float("inf")

        if norm_weight.shape != (HIDDEN_SIZE,) or norm_weight.stride(0) != 1:
            return False, float("inf")

        if qk_weight.shape != (HIDDEN_SIZE,) or qk_weight.stride(0) != 1:
            return False, float("inf")

        if output_norm_weight.shape != (HIDDEN_SIZE,) or output_norm_weight.stride(0) != 1:
            return False, float("inf")

        if output.shape != (self.num_tokens, HIDDEN_SIZE):
            return False, float("inf")

        if output.stride(0) != HIDDEN_SIZE or output.stride(1) != 1:
            return False, float("inf")

        expected = self._build_reference(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight)

        torch.ops._C.kimi_k3_attn_res(prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output, self.num_blocks, EPS, EPS)

        torch.cuda.synchronize(dev)

        if not torch.isfinite(output).all():
            return False, float("inf")

        if output.shape != expected.shape:
            return False, float("inf")

        diff = (output.float() - expected.float()).abs()

        max_abs_diff = diff.max().item() if diff.numel() > 0 else 0.0

        cosine = torch.nn.functional.cosine_similarity(output.float().flatten().unsqueeze(0), expected.float().flatten().unsqueeze(0), dim=1).item()

        passed = torch.allclose(output, expected, atol=8e-2, rtol=3e-2) and cosine >= 0.999

        return passed, max_abs_diff