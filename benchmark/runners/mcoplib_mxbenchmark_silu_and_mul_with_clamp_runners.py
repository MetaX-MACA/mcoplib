import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Silu_and_mul_with_clamp_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16384)
        self.hidden_size = config.get("hidden_size", 4096)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"{self.batch_size} {self.hidden_size}")

        input_elements = self.batch_size * self.hidden_size * 2
        output_elements = self.batch_size * self.hidden_size

        state.add_element_count(output_elements)

        element_size = 2 if self.dtype == torch.float16 else 4

        state.add_global_memory_reads(input_elements * element_size)
        state.add_global_memory_writes(output_elements * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"

            input_shape = (self.batch_size, self.hidden_size * 2)
            output_shape = (self.batch_size, self.hidden_size)

            x = torch.randn(input_shape, dtype=self.dtype, device=dev)
            out = torch.empty(output_shape, dtype=self.dtype, device=dev)

            limit = 1.0
            alpha = 1.0
            beta = 0.0

            return self.make_launcher(
                dev_id,
                torch.ops._C.silu_and_mul_with_clamp,
                out,
                x,
                limit,
                alpha,
                beta,
            )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        input_shape = (self.batch_size, self.hidden_size * 2)
        output_shape = (self.batch_size, self.hidden_size)

        x = torch.randn(input_shape, dtype=self.dtype, device=dev)
        out = torch.empty(output_shape, dtype=self.dtype, device=dev)

        limit = 1.0
        alpha = 1.0
        beta = 0.0

        torch.ops._C.silu_and_mul_with_clamp(
            out,
            x,
            limit,
            alpha,
            beta,
        )

        gate, up = x.chunk(2, dim=-1)

        gate = torch.clamp(gate.float(), max=limit)
        up = torch.clamp(up.float(), min=-limit, max=limit)

        ref = gate * torch.sigmoid(alpha * gate)
        ref = ref * (up + beta)

        ref = ref.to(self.dtype)

        return self.check_diff(out, ref)