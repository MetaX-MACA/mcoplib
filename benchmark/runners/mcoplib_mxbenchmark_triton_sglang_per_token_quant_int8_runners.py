from typing import Dict, Optional, Tuple

import torch

from mcoplib.triton_sglang_per_token_quant_int8 import (
    launch_per_token_quant_int8,
)
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Triton_sglang_per_token_quant_int8_runner(OpBenchmarkBase):
    """Benchmark the optimized, preallocated SGLang INT8 quantizer."""

    def __init__(self, name: str, config: Dict):
        super().__init__(name, config)
        shape = config.get("case", config)
        self.M = int(shape["M"])
        self.K = int(shape["K"])
        self.cal_sum = bool(config.get("cal_sum", False))
        self.seed = int(config.get("seed", 20260911))
        self.scale_dtype = getattr(
            torch, config.get("scale_dtype", "float32")
        )
        self._verification_result: Optional[Tuple[bool, float]] = None

        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float16 or bfloat16")
        if self.scale_dtype != torch.float32:
            raise ValueError("scale_dtype must be float32")
        if self.M <= 0 or self.K <= 0:
            raise ValueError("M and K must be positive")

    def _make_case(self, dev_id: int):
        device = torch.device(f"cuda:{dev_id}")
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        x = torch.randn(
            (self.M, self.K), device=device, dtype=self.dtype
        )
        x_q = torch.empty_like(x, dtype=torch.int8)
        scales = torch.empty(
            (self.M, 1), device=device, dtype=self.scale_dtype
        )
        x_sum = (
            torch.empty((self.M,), device=device, dtype=self.dtype)
            if self.cal_sum
            else None
        )
        return x, x_q, scales, x_sum

    def define_metrics(self, state):
        input_bytes = self.M * self.K * self.dtype.itemsize
        output_bytes = self.M * self.K * torch.int8.itemsize
        scale_bytes = self.M * self.scale_dtype.itemsize
        sum_bytes = self.M * self.dtype.itemsize if self.cal_sum else 0

        state.add_summary("Op", self.name)
        state.add_summary(
            "dtype", self.config.get("dtype", str(self.dtype))
        )
        state.add_summary(
            "Shape", f"M={self.M} K={self.K} cal_sum={self.cal_sum}"
        )
        state.add_element_count(self.M * self.K)
        state.add_global_memory_reads(input_bytes)
        state.add_global_memory_writes(
            output_bytes + scale_bytes + sum_bytes
        )

    def prepare_and_get_launcher(self, dev_id: int, tc_s):
        with torch.cuda.stream(tc_s):
            x, x_q, scales, x_sum = self._make_case(dev_id)
        return self.make_launcher(
            dev_id,
            launch_per_token_quant_int8,
            x,
            x_q,
            scales,
            x_sum,
        )

    @torch.inference_mode()
    def run_verification(self, dev_id: int):
        if self._verification_result is not None:
            return self._verification_result

        x, x_q, scales, x_sum = self._make_case(dev_id)
        launch_per_token_quant_int8(x, x_q, scales, x_sum)

        x_f32 = x.float()
        absmax = torch.clamp_min(x_f32.abs().amax(dim=1), 1e-10)
        expected_scales = (absmax / 127.0).unsqueeze(1)
        scaled = x_f32 * (127.0 / absmax).unsqueeze(1)
        expected_q = torch.where(
            scaled >= 0.0,
            torch.floor(scaled + 0.5),
            torch.ceil(scaled - 0.5),
        ).clamp(-128, 127).to(torch.int8)

        q_max_diff = int(
            (x_q.to(torch.int16) - expected_q.to(torch.int16))
            .abs()
            .max()
            .item()
        )
        scale_ok = torch.allclose(
            scales.float(), expected_scales, rtol=1e-6, atol=1e-7
        )
        comparisons = [
            self.check_diff(x_q.float(), expected_q.float()),
            self.check_diff(scales.float(), expected_scales),
        ]

        sum_ok = True
        if self.cal_sum:
            expected_sum = x_f32.sum(dim=1).to(self.dtype)
            sum_ok = torch.allclose(
                x_sum, expected_sum, rtol=1e-2, atol=1e-2
            )
            comparisons.append(
                self.check_diff(x_sum.float(), expected_sum.float())
            )

        finite = bool(torch.isfinite(scales).all().item())
        if self.cal_sum:
            finite = finite and bool(torch.isfinite(x_sum).all().item())
        passed = (
            finite
            and q_max_diff <= 1
            and scale_ok
            and sum_ok
            and all(result[0] for result in comparisons)
        )
        max_cosine_distance = max(result[1] for result in comparisons)
        self._verification_result = (passed, max_cosine_distance)
        return self._verification_result
