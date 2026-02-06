import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Top_k_per_row_decode_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_rows = config.get("num_rows", 128)
        self.vocab_size = config.get("vocab_size", 32000)
        self.next_n = config.get("next_n", 1)
        self.top_k = 2048
        self.dtype = torch.float32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "float32/int32")
        state.add_summary("Shape", f"Rows={self.num_rows} Vocab={self.vocab_size} NextN={self.next_n}")
        total_elements = self.num_rows * self.vocab_size
        state.add_element_count(total_elements)
        read_bytes = (self.num_rows * self.vocab_size * 4) + (self.num_rows * 4)
        write_bytes = (self.num_rows * self.top_k * 4)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            logits = torch.randn(self.num_rows, self.vocab_size, dtype=self.dtype, device=dev)
            num_seqs = (self.num_rows + self.next_n - 1) // self.next_n
            base_len = max(self.next_n + 10, self.vocab_size - self.next_n)
            seq_lens = torch.full((num_seqs,), base_len, dtype=torch.int32, device=dev)
            indices = torch.empty((self.num_rows, self.top_k), dtype=torch.int32, device=dev)
            stride0 = logits.stride(0)
            stride1 = logits.stride(1)

        def launcher(launch):
            stream = self.as_torch_stream(launch.get_stream(), dev_id)
            with torch.cuda.stream(stream):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    self.next_n,
                    seq_lens,
                    indices,
                    self.num_rows,
                    stride0,
                    stride1
                )
        return launcher

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        logits = torch.randn(self.num_rows, self.vocab_size, dtype=self.dtype, device=dev)
        num_seqs = (self.num_rows + self.next_n - 1) // self.next_n
        safe_len = self.vocab_size - self.next_n - 1
        if safe_len < self.next_n: safe_len = self.vocab_size
        seq_lens = torch.full((num_seqs,), safe_len, dtype=torch.int32, device=dev)
        indices_op = torch.empty((self.num_rows, self.top_k), dtype=torch.int32, device=dev)
        stride0 = logits.stride(0)
        stride1 = logits.stride(1)
        torch.ops._C.top_k_per_row_decode(
            logits,
            self.next_n,
            seq_lens,
            indices_op,
            self.num_rows,
            stride0,
            stride1
        )
        ref_values_list = []
        op_values_list = []
        for r in range(self.num_rows):
            seq_idx = r // self.next_n
            s_len = seq_lens[min(seq_idx, len(seq_lens)-1)].item()
            row_end = s_len - self.next_n + (r % self.next_n) + 1
            row_end = min(max(0, row_end), self.vocab_size)
            row_logits = logits[r, :row_end]
            cur_k = min(self.top_k, row_logits.size(0))
            if cur_k == 0:
                ref_v = torch.zeros(self.top_k, device=dev, dtype=self.dtype)
                op_v = torch.zeros(self.top_k, device=dev, dtype=self.dtype)
            else:
                ref_v, _ = torch.topk(row_logits, k=cur_k, sorted=True)
                row_indices = indices_op[r, :cur_k].long()
                row_indices = row_indices.clamp(min=0, max=self.vocab_size-1)
                op_v = logits[r, row_indices]
                op_v, _ = torch.sort(op_v, descending=True)
            ref_values_list.append(ref_v)
            op_values_list.append(op_v)
        flat_ref = torch.cat([t.flatten() for t in ref_values_list])
        flat_op = torch.cat([t.flatten() for t in op_values_list])
        min_len = min(flat_ref.numel(), flat_op.numel())
        flat_ref = flat_ref[:min_len]
        flat_op = flat_op[:min_len]
        return self.check_diff(flat_op, flat_ref)