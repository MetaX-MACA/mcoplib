import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


class Moe_sum_reduce_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.token_num = config.get("token_num", 4096)
        self.topk_num = config.get("topk_num", 8)
        self.hidden_dim = config.get("hidden_dim", 4096)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.token_num} {self.topk_num} {self.hidden_dim})")
        total_in_elements = self.token_num * self.topk_num * self.hidden_dim
        total_out_elements = self.token_num * self.hidden_dim
        state.add_element_count(total_in_elements + total_out_elements)
        element_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        read_bytes = total_in_elements * element_size
        write_bytes = total_out_elements * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def _prepare(self, dev_id, seed=42):
        dev = f'cuda:{dev_id}'
        gen = torch.Generator(device=dev).manual_seed(seed)
        shape_in = (self.token_num, self.topk_num, self.hidden_dim)
        shape_out = (self.token_num, self.hidden_dim)
        input_tensor = torch.randn(shape_in, dtype=self.dtype, device=dev, generator=gen)
        output_tensor = torch.empty(shape_out, dtype=self.dtype, device=dev)
        return input_tensor, output_tensor

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            input_tensor, output_tensor = self._prepare(dev_id)
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.moe_sum_reduce,
            input_tensor, output_tensor, self.routed_scaling_factor
        )

    def run_verification(self, dev_id):
        input_tensor, output_tensor = self._prepare(dev_id, seed=7)
        ref_input = input_tensor.clone()
        torch.ops.sgl_kernel.moe_sum_reduce(
            input_tensor, output_tensor, self.routed_scaling_factor
        )
        torch.cuda.synchronize()
        expected_out = torch.sum(ref_input.float(), dim=1) * self.routed_scaling_factor
        return self.check_diff(output_tensor, expected_out.to(self.dtype))
