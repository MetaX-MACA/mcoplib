import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
try:
    import mcoplib._C
except ImportError:
    pass
class Gelu_quick_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 1024)
        self.hidden_size = config.get("hidden_size", 512)
    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.hidden_size})")
        total = self.batch_size * self.hidden_size
        state.add_element_count(total)
        element_size = 2 if self.dtype == torch.float16 else 4
        state.add_global_memory_reads(total * element_size)
        state.add_global_memory_writes(total * element_size)
    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.hidden_size)
            inp = torch.randn(shape, dtype=self.dtype, device=dev)
            out = torch.empty_like(inp)
        return self.make_launcher(dev_id, torch.ops._C.gelu_quick, out, inp)
    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.hidden_size)
        inp = torch.randn(shape, dtype=self.dtype, device=dev)
        out_op = torch.empty_like(inp)
        torch.ops._C.gelu_quick(out_op, inp)
        val_ref = inp.float() * torch.sigmoid(1.702 * inp.float())
        out_ref = val_ref.to(self.dtype)
        return self.check_diff(out_op, out_ref, threshold=0.999)