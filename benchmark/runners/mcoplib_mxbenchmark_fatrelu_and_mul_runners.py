import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
try:
    import mcoplib._C
except ImportError:
    pass
class Fatrelu_and_mul_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16)
        self.seq_len = config.get("seq_len", 128)
        self.hidden_size = config.get("hidden_size", 1024)
        self.threshold = config.get("threshold", 0.05)
        self.dtype = getattr(torch, config.get("dtype", "float16"))
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B{self.batch_size}_S{self.seq_len}_H{self.hidden_size})")
        total_elements = self.batch_size * self.seq_len * self.hidden_size
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype == torch.float16 else 4
        input_size = (self.batch_size * self.seq_len * 2 * self.hidden_size) * element_size
        output_size = total_elements * element_size
        state.add_global_memory_reads(input_size)
        state.add_global_memory_writes(output_size)
    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_dim = 2 * self.hidden_size
            input_tensor = torch.randn(self.batch_size, self.seq_len, input_dim, dtype=self.dtype, device=dev)
            output = torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype, device=dev)
        return self.make_launcher(
            dev_id, 
            torch.ops._C.fatrelu_and_mul, 
            output, 
            input_tensor, 
            self.threshold
        )
    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        input_dim = 2 * self.hidden_size
        input_tensor = torch.randn(self.batch_size, self.seq_len, input_dim, dtype=self.dtype, device=dev)
        output = torch.empty(self.batch_size, self.seq_len, self.hidden_size, dtype=self.dtype, device=dev)
        torch.ops._C.fatrelu_and_mul(output, input_tensor, self.threshold)
        x, y = input_tensor.chunk(2, dim=-1)
        expected = torch.where(x > self.threshold, x, torch.tensor(0.0, dtype=self.dtype, device=dev)) * y
        out_ref = expected.to(self.dtype)
        return self.check_diff(output, out_ref)