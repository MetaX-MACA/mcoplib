import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Cp_gather_indexer_k_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16)
        self.head_dim = config.get("head_dim", 128)
        self.block_size = config.get("block_size", 16)
        self.num_blocks = config.get("num_blocks", 4096)
        self.seq_len = config.get("seq_len", 256)
        self.dtype = torch.uint8

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "uint8 (forced)")
        total_tokens = self.batch_size * self.seq_len
        state.add_summary("Shape", f"({total_tokens} {self.head_dim})")
        total_elements = total_tokens * self.head_dim
        state.add_element_count(total_elements)
        element_size = 1
        state.add_global_memory_reads(total_elements * element_size)
        state.add_global_memory_writes(total_elements * element_size)

    def _prepare_data(self, dev_id):
        dev = f'cuda:{dev_id}'
        torch.manual_seed(0)
        seq_lens = torch.full((self.batch_size,), self.seq_len, dtype=torch.int32, device=dev)
        cu_seq_lens = torch.zeros(self.batch_size + 1, dtype=torch.int32, device=dev)
        torch.cumsum(seq_lens, dim=0, out=cu_seq_lens[1:])
        total_tokens = cu_seq_lens[-1].item()
        blocks_per_seq = (self.seq_len + self.block_size - 1) // self.block_size
        max_blocks = blocks_per_seq
        block_table = torch.randint(
            0, self.num_blocks,
            (self.batch_size, max_blocks),
            dtype=torch.int32, device=dev
        )
        cache_stride = self.head_dim
        kv_cache = torch.randint(
            0, 127,
            (self.num_blocks, self.block_size, cache_stride),
            dtype=self.dtype, device=dev
        )
        dst_k = torch.zeros(
            total_tokens, self.head_dim,
            dtype=self.dtype, device=dev
        )
        return kv_cache, dst_k, block_table, cu_seq_lens

    def prepare_and_get_launcher(self, dev_id, tc_s):
        kv_cache, dst_k, block_table, cu_seq_lens = self._prepare_data(dev_id)
        return self.make_launcher(
            dev_id,
            torch.ops._C_cache_ops.cp_gather_indexer_k_cache,
            kv_cache, dst_k, block_table, cu_seq_lens
        )

    def run_verification(self, dev_id):
        kv_cache, dst_k, block_table, cu_seq_lens = self._prepare_data(dev_id)
        torch.cuda.synchronize()
        try:
            torch.ops._C_cache_ops.cp_gather_indexer_k_cache(
                kv_cache, dst_k, block_table, cu_seq_lens
            )
        except Exception as e:
            print(f"\n[Error] Exec failed: {e}")
            return False, 1.0
        torch.cuda.synchronize()
        out_op = dst_k
        out_ref = torch.zeros_like(dst_k)
        cpu_cu_seq_lens = cu_seq_lens.cpu()
        for b in range(self.batch_size):
            start = cpu_cu_seq_lens[b].item()
            end = cpu_cu_seq_lens[b+1].item()
            seq_len = end - start
            if seq_len == 0: continue
            token_indices = torch.arange(seq_len, device=kv_cache.device)
            logical_block_indices = token_indices // self.block_size
            block_offsets = token_indices % self.block_size
            physical_block_ids = block_table[b][logical_block_indices]
            gathered_data = kv_cache[physical_block_ids, block_offsets, :self.head_dim]
            out_ref[start:end] = gathered_data
        return self.check_diff(out_op, out_ref)