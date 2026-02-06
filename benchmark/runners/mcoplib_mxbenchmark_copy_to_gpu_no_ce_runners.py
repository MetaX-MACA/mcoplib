import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Copy_to_gpu_no_ce_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.N = config.get("N", 64)
        self.dtype = torch.int32
        if self.N not in [64, 72]:
            print(f"[Warning] copy_to_gpu_no_ce only supports N=64 or N=72. Resetting N={self.N} to 64.")
            self.N = 64

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int32")
        state.add_summary("Shape", f"(Size: {self.N})")
        total_elements = self.N
        state.add_element_count(total_elements)
        element_size = 4
        state.add_global_memory_reads(0)
        state.add_global_memory_writes(total_elements * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_cpu = torch.randint(0, 1000, (self.N,), dtype=self.dtype, device='cpu').contiguous()
            output_gpu = torch.empty((self.N,), dtype=self.dtype, device=dev).contiguous()
        return self.make_launcher(dev_id, torch.ops.sgl_kernel.copy_to_gpu_no_ce, input_cpu, output_gpu)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        input_cpu = torch.randint(-1000, 1000, (self.N,), dtype=self.dtype, device='cpu').contiguous()
        output_gpu = torch.empty((self.N,), dtype=self.dtype, device=dev).contiguous()
        torch.ops.sgl_kernel.copy_to_gpu_no_ce(input_cpu, output_gpu)
        output_ref = input_cpu.to(dev)
        return self.check_diff(output_gpu, output_ref, threshold=0.999999)