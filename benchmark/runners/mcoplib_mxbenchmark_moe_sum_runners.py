import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
try:
    import mcoplib._moe_C
except ImportError:
    pass
class Moe_sum_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 4096)
        self.top_k = config.get("top_k", 2)
        self.hidden_size = config.get("hidden_size", 4096)
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.top_k} {self.hidden_size})")
        total_out_elements = self.batch_size * self.hidden_size
        state.add_element_count(total_out_elements)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        reads = (self.batch_size * self.top_k * self.hidden_size) * element_size
        writes = (self.batch_size * self.hidden_size) * element_size
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)
    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_tensor = torch.randn(self.batch_size, self.top_k, self.hidden_size, dtype=self.dtype, device=dev)
            output_tensor = torch.empty(self.batch_size, self.hidden_size, dtype=self.dtype, device=dev)
        return self.make_launcher(dev_id, torch.ops._moe_C.moe_sum, input_tensor, output_tensor)
    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        N, K, H = 16, 2, 128
        input_tensor = torch.randn(N, K, H, dtype=self.dtype, device=dev)
        output_op = torch.empty(N, H, dtype=self.dtype, device=dev)
        torch.ops._moe_C.moe_sum(input_tensor, output_op)
        output_ref = input_tensor.sum(dim=1)
        return self.check_diff(output_op, output_ref)