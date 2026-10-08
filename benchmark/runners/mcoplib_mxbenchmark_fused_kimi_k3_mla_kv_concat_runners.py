# SPDX-License-Identifier: Apache-2.0

import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError as e:
    raise RuntimeError(f"Failed to import mcoplib._C: {e}") from e


OP_NAME = "fused_kimi_k3_mla_kv_concat"

K_NOPE_DIM = 128
K_PE_DIM = 64
K_OUT_DIM = 192


class Fused_kimi_k3_mla_kv_concat_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_heads = config.get("num_heads", 64)
        self.dtype = self._parse_dtype(config.get("dtype", "bfloat16"))
        self.seed = config.get("seed", 5001)

        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"dtype must be float16 or bfloat16, got {self.dtype}")

        if self.num_tokens < 0:
            raise ValueError(f"num_tokens must be >= 0, got {self.num_tokens}")

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")

        if not hasattr(torch.ops._C, OP_NAME):
            raise RuntimeError(f"torch.ops._C.{OP_NAME} is not registered")

    @staticmethod
    def _parse_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            return dtype

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return dtype_map[dtype]

    def _make_inputs(self, dev):
        torch.manual_seed(self.seed)

        k_nope = torch.randn((self.num_tokens, self.num_heads, K_NOPE_DIM), dtype=self.dtype, device=dev).contiguous()

        k_pe = torch.randn((self.num_tokens, K_PE_DIM), dtype=self.dtype, device=dev).contiguous()

        k_out = torch.empty((self.num_tokens, self.num_heads, K_OUT_DIM), dtype=self.dtype, device=dev).contiguous()

        return k_nope, k_pe, k_out

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"({self.num_tokens} {self.num_heads} {K_OUT_DIM})")

        k_nope_elements = self.num_tokens * self.num_heads * K_NOPE_DIM
        k_pe_elements = self.num_tokens * K_PE_DIM
        k_out_elements = self.num_tokens * self.num_heads * K_OUT_DIM

        state.add_element_count(k_out_elements)

        element_size = 2

        k_nope_bytes = k_nope_elements * element_size
        k_pe_bytes = k_pe_elements * element_size
        k_out_bytes = k_out_elements * element_size

        state.add_global_memory_reads(k_nope_bytes + k_pe_bytes)
        state.add_global_memory_writes(k_out_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            k_nope, k_pe, k_out = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_kimi_k3_mla_kv_concat, k_nope, k_pe, k_out)

    def _build_reference(self, k_nope, k_pe):
        k_pe_expand = k_pe.unsqueeze(1).expand(k_nope.shape[0], k_nope.shape[1], K_PE_DIM)

        return torch.cat([k_nope, k_pe_expand], dim=-1)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        k_nope, k_pe, k_out = self._make_inputs(dev)

        k_nope_before = k_nope.clone()
        k_pe_before = k_pe.clone()

        torch.ops._C.fused_kimi_k3_mla_kv_concat(k_nope, k_pe, k_out)

        torch.cuda.synchronize(dev)

        if k_out.shape != (self.num_tokens, self.num_heads, K_OUT_DIM):
            return False, float("inf")

        if k_out.dtype != self.dtype:
            return False, float("inf")

        if k_out.device.type != "cuda":
            return False, float("inf")

        if k_out.stride(2) != 1:
            return False, float("inf")

        if k_out.stride(1) != K_OUT_DIM:
            return False, float("inf")

        if k_out.stride(0) != self.num_heads * K_OUT_DIM:
            return False, float("inf")

        if not k_out.is_contiguous():
            return False, float("inf")

        if not torch.equal(k_nope, k_nope_before):
            return False, float("inf")

        if not torch.equal(k_pe, k_pe_before):
            return False, float("inf")

        if self.num_tokens == 0:
            return True, 0.0

        expected = self._build_reference(k_nope, k_pe)

        diff = (k_out.float() - expected.float()).abs()
        max_abs_diff = diff.max().item() if diff.numel() > 0 else 0.0

        if not torch.equal(k_out, expected):
            return False, max_abs_diff

        actual_pe = k_out[:, :, K_NOPE_DIM:]
        reference_pe = k_pe.unsqueeze(1).expand(self.num_tokens, self.num_heads, K_PE_DIM)

        if not torch.equal(actual_pe, reference_pe):
            return False, max_abs_diff

        actual_nope = k_out[:, :, :K_NOPE_DIM]

        if not torch.equal(actual_nope, k_nope):
            return False, max_abs_diff

        return True, max_abs_diff