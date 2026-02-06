import torch
import mcoplib._C 
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Get_cuda_view_from_cpu_tensor_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.size = config.get("size", 1024 * 1024)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.size})")
        state.add_element_count(0)
        state.add_global_memory_reads(0)
        state.add_global_memory_writes(0)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        inp = torch.randn(self.size, dtype=self.dtype, device="cpu").pin_memory()
        return self.make_launcher(dev_id, torch.ops._C.get_cuda_view_from_cpu_tensor, inp)

    def run_verification(self, dev_id):
        inp = torch.randn(self.size, dtype=self.dtype, device="cpu").pin_memory()
        cuda_view = torch.ops._C.get_cuda_view_from_cpu_tensor(inp)
        check_value = 1234.5678
        cuda_view[0] = check_value
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        cpu_val = inp[0].item()
        diff = abs(cpu_val - check_value)
        return diff < 1e-4, diff