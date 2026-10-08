# SPDX-License-Identifier: Apache-2.0

import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError as e:
    raise RuntimeError(f"Failed to import mcoplib._C: {e}") from e


OP_NAME = "fused_kimi_k3_mla_kv_concat_quant_fp8"
FP8_DTYPE = torch.float8_e4m3fn

K_NOPE = 128
K_PE = 64
K_TOTAL = 192
V_DIM = 128


class Fused_kimi_k3_mla_kv_concat_quant_fp8_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_heads = config.get("num_heads", 64)
        self.dtype = self._parse_dtype(config.get("dtype", "float16"))
        self.k_pe_dtype = self._parse_dtype(config.get("k_pe_dtype", "float8_e4m3fn"))
        self.seed = config.get("seed", 8888)

        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"dtype must be float16 or bfloat16, got {self.dtype}")

        if self.k_pe_dtype not in (self.dtype, FP8_DTYPE):
            raise ValueError(f"k_pe_dtype must equal dtype or float8_e4m3fn, got {self.k_pe_dtype}")

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
            "float8_e4m3fn": torch.float8_e4m3fn,
            "fp8": torch.float8_e4m3fn,
            "fp8_e4m3fn": torch.float8_e4m3fn,
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return dtype_map[dtype]

    def _make_inputs(self, dev):
        torch.manual_seed(self.seed)

        k_nope = torch.randn((self.num_tokens, self.num_heads, K_NOPE), dtype=self.dtype, device=dev).contiguous()

        k_pe = torch.randn((self.num_tokens, K_PE), dtype=self.dtype, device=dev).contiguous()

        if self.k_pe_dtype == FP8_DTYPE:
            k_pe = k_pe.to(FP8_DTYPE)

        v = torch.randn((self.num_tokens, self.num_heads, V_DIM), dtype=self.dtype, device=dev).contiguous()

        k_fp8 = torch.empty((self.num_tokens, self.num_heads, K_TOTAL), dtype=FP8_DTYPE, device=dev).contiguous()

        v_fp8 = torch.empty((self.num_tokens, self.num_heads, V_DIM), dtype=FP8_DTYPE, device=dev).contiguous()

        return k_nope, k_pe, v, k_fp8, v_fp8

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", f"{self.dtype}+k_pe:{self.k_pe_dtype}")
        state.add_summary("Shape", f"({self.num_tokens} {self.num_heads} {K_TOTAL})")

        input_element_size = 2
        k_pe_element_size = 1 if self.k_pe_dtype == FP8_DTYPE else 2
        output_element_size = 1

        k_nope_elements = self.num_tokens * self.num_heads * K_NOPE
        k_pe_elements = self.num_tokens * K_PE
        v_elements = self.num_tokens * self.num_heads * V_DIM
        k_out_elements = self.num_tokens * self.num_heads * K_TOTAL
        v_out_elements = self.num_tokens * self.num_heads * V_DIM

        total_elements = k_nope_elements + k_pe_elements + v_elements + k_out_elements + v_out_elements

        state.add_element_count(total_elements)

        k_nope_bytes = k_nope_elements * input_element_size
        k_pe_bytes = k_pe_elements * k_pe_element_size
        v_bytes = v_elements * input_element_size

        k_out_bytes = k_out_elements * output_element_size
        v_out_bytes = v_out_elements * output_element_size

        state.add_global_memory_reads(k_nope_bytes + k_pe_bytes + v_bytes)
        state.add_global_memory_writes(k_out_bytes + v_out_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            k_nope, k_pe, v, k_fp8, v_fp8 = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8, k_nope, k_pe, v, k_fp8, v_fp8)

    def _build_reference(self, k_nope, k_pe, v):
        if k_pe.dtype == FP8_DTYPE:
            k_pe_ref = k_pe.to(self.dtype)
        else:
            k_pe_ref = k_pe

        k_pe_ref = k_pe_ref.unsqueeze(1).expand(k_nope.shape[0], k_nope.shape[1], K_PE)

        k_expected = torch.cat((k_nope, k_pe_ref), dim=-1)

        v_expected = v

        return k_expected, v_expected

    def _check_close(self, actual_fp8, expected):
        actual = actual_fp8.to(self.dtype)
        expected_fp8 = expected.to(FP8_DTYPE)
        expected_roundtrip = expected_fp8.to(self.dtype)

        if self.dtype == torch.float16:
            atol = 2e-1
            rtol = 2e-1
        else:
            atol = 3e-1
            rtol = 3e-1

        if actual.numel() == 0:
            return True, 0.0

        actual_float = actual.float()
        expected_float = expected_roundtrip.float()

        diff = (actual_float - expected_float).abs()
        max_diff = diff.max().item()

        passed = torch.allclose(actual, expected_roundtrip, atol=atol, rtol=rtol)

        return passed, max_diff

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        k_nope, k_pe, v, k_fp8, v_fp8 = self._make_inputs(dev)

        torch.ops._C.fused_kimi_k3_mla_kv_concat_quant_fp8(k_nope, k_pe, v, k_fp8, v_fp8)

        torch.cuda.synchronize(dev)

        if k_fp8.dtype != FP8_DTYPE:
            return False, float("inf")

        if v_fp8.dtype != FP8_DTYPE:
            return False, float("inf")

        if k_fp8.shape != (self.num_tokens, self.num_heads, K_TOTAL):
            return False, float("inf")

        if v_fp8.shape != (self.num_tokens, self.num_heads, V_DIM):
            return False, float("inf")

        if not k_fp8.is_contiguous():
            return False, float("inf")

        if not v_fp8.is_contiguous():
            return False, float("inf")

        if self.num_tokens > 0:
            if not torch.isfinite(k_fp8.to(self.dtype)).all():
                return False, float("inf")

            if not torch.isfinite(v_fp8.to(self.dtype)).all():
                return False, float("inf")

        k_expected, v_expected = self._build_reference(k_nope, k_pe, v)

        passed_k, diff_k = self._check_close(k_fp8, k_expected)
        passed_v, diff_v = self._check_close(v_fp8, v_expected)

        if not passed_k or not passed_v:
            return False, max(diff_k, diff_v)

        if self.num_tokens > 0:
            actual_pe = k_fp8[:, :, K_NOPE:]

            if self.num_heads > 1:
                reference_pe = actual_pe[:, :1, :].expand(self.num_tokens, self.num_heads, K_PE)

                if not torch.equal(actual_pe, reference_pe):
                    return False, float("inf")

            actual_k_nope = k_fp8[:, :, :K_NOPE]
            expected_k_nope = k_nope

            passed_k_nope, diff_k_nope = self._check_close(actual_k_nope, expected_k_nope)

            if not passed_k_nope:
                return False, diff_k_nope

            expected_v = v

            passed_v_roundtrip, diff_v_roundtrip = self._check_close(v_fp8, expected_v)

            if not passed_v_roundtrip:
                return False, diff_v_roundtrip

        return True, max(diff_k, diff_v)