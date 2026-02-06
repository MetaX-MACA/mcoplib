import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Silu_and_mul_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 4096)
        self.hidden_size = config.get("hidden_size", 11008)
        self.input_size = self.hidden_size * 2

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.input_size}) -> ({self.batch_size} {self.hidden_size})")
        total_out_elements = self.batch_size * self.hidden_size
        state.add_element_count(total_out_elements)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        reads = (self.batch_size * self.input_size) * element_size
        writes = (self.batch_size * self.hidden_size) * element_size
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_tensor = torch.randn(self.batch_size, self.input_size, dtype=self.dtype, device=dev)
            out = torch.empty(self.batch_size, self.hidden_size, dtype=self.dtype, device=dev)
        return self.make_launcher(dev_id, torch.ops._C.silu_and_mul, out, input_tensor)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        N = 128
        H = 256
        input_tensor = torch.randn(N, H * 2, dtype=self.dtype, device=dev)
        out_op = torch.empty(N, H, dtype=self.dtype, device=dev)
        torch.ops._C.silu_and_mul(out_op, input_tensor)
        x, y = input_tensor.chunk(2, dim=-1)
        out_ref = F.silu(x) * y
        return self.check_diff(out_op, out_ref)