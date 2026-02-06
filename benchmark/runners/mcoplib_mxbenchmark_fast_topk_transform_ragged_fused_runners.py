import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Fast_topk_transform_ragged_fused_runner(OpBenchmarkBase):
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
        state.add_summary("Shape", f"B={self.batch_size} V={self.vocab_size} -> Top{self.k} (+Offset)")
        total_elements = self.batch_size * self.vocab_size
        state.add_element_count(total_elements)
        read_bytes = (self.batch_size * self.vocab_size * 4) + (self.batch_size * 4)
        write_bytes = self.batch_size * self.k * 4
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "fast_topk_transform_ragged_fused"):
             raise RuntimeError("Operator 'torch.ops.sgl_kernel.fast_topk_transform_ragged_fused' not found.")
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            score = torch.randn((self.batch_size, self.vocab_size), dtype=self.dtype, device=dev)
            lengths = torch.full((self.batch_size,), self.vocab_size, dtype=torch.int32, device=dev)
            topk_indices_ragged = torch.zeros((self.batch_size, self.k), dtype=torch.int32, device=dev)
            topk_indices_offset = torch.randint(0, 1000, (self.batch_size,), dtype=torch.int32, device=dev)
            row_starts = None
        return self.make_launcher(dev_id, torch.ops.sgl_kernel.fast_topk_transform_ragged_fused, 
                                  score, lengths, topk_indices_ragged, topk_indices_offset, row_starts)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "fast_topk_transform_ragged_fused"):
             return False, 1.0
        dev = f'cuda:{dev_id}'
        score = torch.randn((self.batch_size, self.vocab_size), dtype=self.dtype, device=dev)
        lengths = torch.full((self.batch_size,), self.vocab_size, dtype=torch.int32, device=dev)
        topk_indices_offset = torch.randint(0, 1000, (self.batch_size,), dtype=torch.int32, device=dev)
        op_out = torch.zeros((self.batch_size, self.k), dtype=torch.int32, device=dev)
        torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(score, lengths, op_out, topk_indices_offset, None)
        _, topk_indices = torch.topk(score, self.k, dim=1)
        offset_expanded = topk_indices_offset.unsqueeze(1).expand(-1, self.k)
        ref_out = topk_indices.int() + offset_expanded
        op_sorted, _ = torch.sort(op_out, dim=1)
        ref_sorted, _ = torch.sort(ref_out, dim=1)
        return self.check_diff(op_sorted.float(), ref_sorted.float())