import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Mul_and_silu_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 128)
        self.hidden_size = config.get("hidden_size", 4096)
        self.dtype = getattr(torch, config.get("dtype", "float16"))

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(T{self.num_tokens}_H{self.hidden_size})")
        total_elements = self.num_tokens * self.hidden_size
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype == torch.float16 else 4
        input_size = self.num_tokens * (2 * self.hidden_size) * element_size
        output_size = self.num_tokens * self.hidden_size * element_size
        state.add_global_memory_reads(input_size)
        state.add_global_memory_writes(output_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_size = 2 * self.hidden_size
            input_tensor = torch.randn(self.num_tokens, input_size, dtype=self.dtype, device=dev)
            output_tensor = torch.empty(self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev)
        return self.make_launcher(dev_id, torch.ops._C.mul_and_silu, output_tensor, input_tensor)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        input_size = 2 * self.hidden_size
        input_tensor = torch.randn(self.num_tokens, input_size, dtype=self.dtype, device=dev)
        output_tensor = torch.empty(self.num_tokens, self.hidden_size, dtype=self.dtype, device=dev)
        torch.ops._C.mul_and_silu(output_tensor, input_tensor)
        d = self.hidden_size
        x = input_tensor[..., :d].float()
        y = input_tensor[..., d:].float()
        out_ref = x * F.silu(y)
        out_ref = out_ref.to(self.dtype)
        return self.check_diff(output_tensor, out_ref)