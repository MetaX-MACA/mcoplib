import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Fast_topk_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 32)
        self.vocab_size = config.get("vocab_size", 64000)
        self.k = 2048
        if self.dtype != torch.float32:
            print(f"[Warning] {name} only supports float32 in kernel source. Forcing dtype to float32.")
            self.dtype = torch.float32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32")
        state.add_summary("Shape", f"({self.batch_size} {self.vocab_size}) -> Top{self.k}")
        total_input_elements = self.batch_size * self.vocab_size
        state.add_element_count(total_input_elements)
        read_bytes = self.batch_size * self.vocab_size * 4
        write_bytes = self.batch_size * self.k * 4
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "fast_topk"):
             raise RuntimeError("Operator 'torch.ops.sgl_kernel.fast_topk' not found.")
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            score = torch.randn((self.batch_size, self.vocab_size), dtype=self.dtype, device=dev)
            indices = torch.zeros((self.batch_size, self.k), dtype=torch.int32, device=dev)
            lengths = torch.full((self.batch_size,), self.vocab_size, dtype=torch.int32, device=dev)
            row_starts = None
        return self.make_launcher(dev_id, torch.ops.sgl_kernel.fast_topk, score, indices, lengths, row_starts)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "fast_topk"):
             print("Error: torch.ops.sgl_kernel.fast_topk not available.")
             return False, 1.0
        dev = f'cuda:{dev_id}'
        score = torch.randn((self.batch_size, self.vocab_size), dtype=self.dtype, device=dev)
        indices_op = torch.zeros((self.batch_size, self.k), dtype=torch.int32, device=dev)
        lengths = torch.full((self.batch_size,), self.vocab_size, dtype=torch.int32, device=dev)
        torch.ops.sgl_kernel.fast_topk(score, indices_op, lengths, None)
        ref_values, _ = torch.topk(score, self.k, dim=1)
        op_values = torch.gather(score, 1, indices_op.to(torch.int64))
        op_values_sorted, _ = torch.sort(op_values, dim=1, descending=True)
        return self.check_diff(op_values_sorted, ref_values)