
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
BLOCK_WRITE_IDX = 0


class Kimi_k3_attn_res_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.num_tokens = config.get("batch_size", config.get("num_tokens", 3))
        self.num_blocks = config.get("num_blocks", 8)
        self.has_delta = config.get("has_delta", False)
        self.block_write_idx = config.get("block_write_idx", BLOCK_WRITE_IDX)
        self.seed = config.get("seed", 42)

        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {self.num_tokens}")

        if self.num_blocks < 1 or self.num_blocks > MAX_BLOCKS:
            raise ValueError(f"num_blocks must be 1..{MAX_BLOCKS}, got {self.num_blocks}")

        if self.block_write_idx < 0 or self.block_write_idx >= MAX_BLOCKS:
            raise ValueError(f"block_write_idx must be 0..{MAX_BLOCKS - 1}, got {self.block_write_idx}")

        if not hasattr(torch.ops._C, "kimi_k3_attn_res"):
            raise RuntimeError("torch.ops._C.kimi_k3_attn_res is not registered")

        # The installed _C.abi3.so may predate the block_write_idx API
        # (MC3-15561): probe the registered schema once and call whichever
        # signature it exposes (see _op_args).
        schemas = getattr(torch.ops._C.kimi_k3_attn_res, "_schemas", {})
        schema = str(next(iter(schemas.values()))) if schemas else ""
        self.op_takes_block_write_idx = "block_write_idx" in schema

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "torch.bfloat16")
        state.add_summary("Shape", f"tokens={self.num_tokens} blocks={self.num_blocks} hidden={HIDDEN_SIZE}")
        state.add_summary("has_delta", str(self.has_delta))
        state.add_summary("block_write_idx", str(self.block_write_idx))

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

        delta = torch.randn((self.num_tokens, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16).contiguous() if self.has_delta else None

        blocks = torch.randn((self.num_tokens, MAX_BLOCKS, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16).contiguous()

        norm_weight = (1.0 + 0.1 * torch.randn(HIDDEN_SIZE, device=dev, dtype=torch.bfloat16)).contiguous()
        qk_weight = (torch.randn(HIDDEN_SIZE, device=dev, dtype=torch.bfloat16) / HIDDEN_SIZE ** 0.5).contiguous()
        output_norm_weight = (1.0 + 0.1 * torch.randn(HIDDEN_SIZE, device=dev, dtype=torch.bfloat16)).contiguous()

        output = torch.empty((self.num_tokens, HIDDEN_SIZE), device=dev, dtype=torch.bfloat16).contiguous()

        return prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output

    def _op_args(self, dev, prefix, delta, blocks, norm_weight, qk_weight,
                 output_norm_weight, output):
        """Argument tuple matching the registered kimi_k3_attn_res schema.
        Newer builds take (..., num_blocks, block_write_idx, eps,
        output_norm_eps) with an optional delta; the pre-MC3-15561 build
        installed here takes (..., num_blocks, eps, output_norm_eps) with a
        non-optional delta, where "no delta" is signaled by an empty
        (0, hidden) tensor (numel()==0, stride(0)==hidden keeps the
        kernel's packed-rows TORCH_CHECK happy)."""
        if self.op_takes_block_write_idx:
            return (prefix, delta, blocks, norm_weight, qk_weight,
                    output_norm_weight, output, self.num_blocks,
                    self.block_write_idx, EPS, EPS)
        return (prefix,
                delta if self.has_delta
                else torch.empty((0, HIDDEN_SIZE), device=dev,
                                 dtype=torch.bfloat16),
                blocks, norm_weight, qk_weight, output_norm_weight, output,
                self.num_blocks, EPS, EPS)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.kimi_k3_attn_res,
                                  *self._op_args(dev, prefix, delta, blocks,
                                                 norm_weight, qk_weight,
                                                 output_norm_weight, output))

    def _build_reference(self, prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight):
        current_prefix = prefix if delta is None else prefix + delta
        values = torch.cat([blocks[:, :self.num_blocks], current_prefix.unsqueeze(1)], dim=1)
        keys = F.rms_norm(values, (HIDDEN_SIZE,), norm_weight, EPS)
        probs = (keys @ qk_weight).softmax(dim=-1)
        expected = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
        expected = F.rms_norm(expected, (HIDDEN_SIZE,), output_norm_weight, EPS)
        return expected

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight, output = self._make_inputs(dev)

        if prefix.shape != (self.num_tokens, HIDDEN_SIZE) or not prefix.is_contiguous():
            return False, float("inf")

        if delta is not None and (delta.shape != (self.num_tokens, HIDDEN_SIZE) or not delta.is_contiguous()):
            return False, float("inf")

        if blocks.shape != (self.num_tokens, MAX_BLOCKS, HIDDEN_SIZE) or not blocks.is_contiguous():
            return False, float("inf")

        if blocks.stride(2) != 1 or blocks.stride(1) != HIDDEN_SIZE or blocks.stride(0) != MAX_BLOCKS * HIDDEN_SIZE:
            return False, float("inf")

        if norm_weight.shape != (HIDDEN_SIZE,) or norm_weight.stride(0) != 1:
            return False, float("inf")

        if qk_weight.shape != (HIDDEN_SIZE,) or qk_weight.stride(0) != 1:
            return False, float("inf")

        if output_norm_weight.shape != (HIDDEN_SIZE,) or output_norm_weight.stride(0) != 1:
            return False, float("inf")

        if output.shape != (self.num_tokens, HIDDEN_SIZE) or not output.is_contiguous():
            return False, float("inf")

        # 保存原始 blocks，防止算子写入影响参考计算。
        expected = self._build_reference(prefix.clone(), delta, blocks.clone(), norm_weight, qk_weight, output_norm_weight)

        torch.ops._C.kimi_k3_attn_res(*self._op_args(dev, prefix, delta, blocks,
                                                    norm_weight, qk_weight,
                                                    output_norm_weight,
                                                    output))

        torch.cuda.synchronize(dev)

        if not torch.isfinite(output).all():
            return False, float("inf")

        if output.shape != expected.shape:
            return False, float("inf")

        diff = (output.float() - expected.float()).abs()
        max_abs_diff = diff.max().item() if diff.numel() > 0 else 0.0

        cosine = F.cosine_similarity(output.float().flatten().unsqueeze(0), expected.float().flatten().unsqueeze(0), dim=1).item()
        passed = torch.allclose(output, expected, atol=8e-2, rtol=3e-2) and cosine >= 0.999

        return passed, max_abs_diff
