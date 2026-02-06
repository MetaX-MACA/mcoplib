import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Fast_topk_transform_fused_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 32)
        self.vocab_size = config.get("vocab_size", 64000)
        self.prefill_bs = config.get("prefill_bs", 32)
        self.k = 2048
        if self.dtype != torch.float32:
            print(f"[Warning] {name} only supports float32 in kernel source. Forcing dtype to float32.")
            self.dtype = torch.float32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32")
        state.add_summary("Shape", f"B={self.batch_size} V={self.vocab_size} -> Top{self.k} (Mapped)")
        total_elements = self.batch_size * self.vocab_size
        state.add_element_count(total_elements)
        read_bytes = (self.batch_size * self.vocab_size * 4) + \
                     (self.batch_size * self.k * 4) + \
                     ((self.prefill_bs + 1) * 4)
        write_bytes = self.batch_size * self.k * 4
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "fast_topk_transform_fused"):
             raise RuntimeError("Operator 'torch.ops.sgl_kernel.fast_topk_transform_fused' not found.")
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            score = torch.randn((self.batch_size, self.vocab_size), dtype=self.dtype, device=dev)
            lengths = torch.full((self.batch_size,), self.vocab_size, dtype=torch.int32, device=dev)
            dst_page_table = torch.zeros((self.batch_size, self.k), dtype=torch.int32, device=dev)
            real_src_stride = self.vocab_size
            src_page_table = torch.randint(0, 1000, (self.prefill_bs, real_src_stride), dtype=torch.int32, device=dev)
            cu_seqlens_q = torch.linspace(0, self.batch_size, steps=self.prefill_bs + 1, dtype=torch.int32, device=dev)
            row_starts = None
        return self.make_launcher(dev_id, torch.ops.sgl_kernel.fast_topk_transform_fused, 
                                  score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "fast_topk_transform_fused"):
             return False, 1.0
        dev = f'cuda:{dev_id}'
        real_src_stride = self.vocab_size
        score = torch.randn((self.batch_size, self.vocab_size), dtype=self.dtype, device=dev)
        lengths = torch.full((self.batch_size,), self.vocab_size, dtype=torch.int32, device=dev)
        src_page_table = torch.arange(real_src_stride * self.prefill_bs, dtype=torch.int32, device=dev).view(self.prefill_bs, real_src_stride)
        cu_seqlens_q = torch.linspace(0, self.batch_size, steps=self.prefill_bs + 1, dtype=torch.int32, device=dev)
        dst_page_table_op = torch.zeros((self.batch_size, self.k), dtype=torch.int32, device=dev)
        torch.ops.sgl_kernel.fast_topk_transform_fused(score, lengths, dst_page_table_op, src_page_table, cu_seqlens_q, None)
        _, topk_indices = torch.topk(score, self.k, dim=1)
        dst_page_table_ref = torch.zeros_like(dst_page_table_op)
        cu_seq_cpu = cu_seqlens_q.cpu().numpy()
        for b in range(self.batch_size):
            p_id = -1
            for i in range(self.prefill_bs):
                if cu_seq_cpu[i] <= b < cu_seq_cpu[i+1]:
                    p_id = i
                    break
            current_indices = topk_indices[b]
            dst_page_table_ref[b] = src_page_table[p_id, current_indices]
        op_sorted, _ = torch.sort(dst_page_table_op, dim=1)
        ref_sorted, _ = torch.sort(dst_page_table_ref, dim=1)
        return self.check_diff(op_sorted.float(), ref_sorted.float())