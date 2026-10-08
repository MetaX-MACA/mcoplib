# SPDX-License-Identifier: Apache-2.0

import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


OP_NAME = "dsv3_fused_a_gemm"


class Dsv3_fused_a_gemm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.num_tokens = config.get("num_tokens", 16)
        self.hd_out = config.get("hd_out", 1536)
        self.hd_in = config.get("hd_in", 128)
        self.enable_pdl = config.get("enable_pdl", True)
        self.seed = config.get("seed", 42)

        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {self.num_tokens}")

        if self.hd_in != 128:
            raise ValueError(f"hd_in must be 128, got {self.hd_in}")

        if self.hd_out != 1536:
            raise ValueError(f"hd_out must be 1536, got {self.hd_out}")

        if not hasattr(torch.ops._C, OP_NAME):
            raise RuntimeError(f"torch.ops._C.{OP_NAME} is not registered")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "torch.bfloat16")
        state.add_summary("Shape", f"({self.num_tokens} {self.hd_out} {self.hd_in})")
        state.add_summary("enable_pdl", str(self.enable_pdl))

        x_elements = self.num_tokens * self.hd_in
        weight_elements = self.hd_out * self.hd_in
        output_elements = self.num_tokens * self.hd_out

        state.add_element_count(output_elements)

        state.add_global_memory_reads((x_elements + weight_elements) * 2)
        state.add_global_memory_writes(output_elements * 2)

    def _make_inputs(self, dev):
        torch.manual_seed(self.seed)

        x = torch.ones((self.num_tokens, self.hd_in), dtype=torch.bfloat16, device=dev).contiguous()

        weight = torch.ones((self.hd_out, self.hd_in), dtype=torch.bfloat16, device=dev).contiguous()

        output = torch.full((self.num_tokens, self.hd_out), float("nan"), dtype=torch.bfloat16, device=dev).contiguous()

        return x, weight, output

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            x, weight, output = self._make_inputs(dev)
            weight_t = weight.t()

        return self.make_launcher(dev_id, torch.ops._C.dsv3_fused_a_gemm, output, x, weight_t, self.enable_pdl)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        x, weight, output = self._make_inputs(dev)

        weight_t = weight.t()

        if x.shape != (self.num_tokens, self.hd_in):
            return False, float("inf")

        if x.dtype != torch.bfloat16:
            return False, float("inf")

        if x.stride() != (self.hd_in, 1):
            return False, float("inf")

        if weight.shape != (self.hd_out, self.hd_in):
            return False, float("inf")

        if weight.dtype != torch.bfloat16:
            return False, float("inf")

        if weight_t.shape != (self.hd_in, self.hd_out):
            return False, float("inf")

        if weight_t.stride() != (1, self.hd_in):
            return False, float("inf")

        if output.shape != (self.num_tokens, self.hd_out):
            return False, float("inf")

        if output.dtype != torch.bfloat16:
            return False, float("inf")

        if output.stride() != (self.hd_out, 1):
            return False, float("inf")

        torch.ops._C.dsv3_fused_a_gemm(output, x, weight_t, enable_pdl=self.enable_pdl)

        torch.cuda.synchronize(dev)

        if output.shape != (self.num_tokens, self.hd_out):
            return False, float("inf")

        if output.dtype != torch.bfloat16:
            return False, float("inf")

        return True, 0.0