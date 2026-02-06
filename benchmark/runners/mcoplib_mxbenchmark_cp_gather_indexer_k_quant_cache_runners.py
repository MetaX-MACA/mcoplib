import torch
import mcoplib._C 
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Cp_gather_indexer_k_quant_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 512)
        self.head_dim = config.get("head_dim", 128)
        self.quant_block_size = config.get("quant_block_size", 32)
        self.block_size = config.get("block_size", 16)
        self.seq_len = config.get("seq_len", 4096)
        self.scale_bytes_per_token = (self.head_dim // self.quant_block_size) * 4
        blocks_per_seq = (self.seq_len + self.block_size - 1) // self.block_size
        min_needed_blocks = blocks_per_seq * self.batch_size
        config_num_blocks = config.get("num_blocks", 131072)
        self.num_blocks = max(config_num_blocks, min_needed_blocks + 4096)
        self.dtype = torch.uint8

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "int8/uint8")
        state.add_summary("Shape", f"({self.batch_size}x{self.seq_len} {self.head_dim})")
        total_tokens = self.batch_size * self.seq_len
        bytes_per_token = self.head_dim + self.scale_bytes_per_token
        state.add_element_count(total_tokens * self.head_dim)
        state.add_global_memory_reads(total_tokens * bytes_per_token)
        state.add_global_memory_writes(total_tokens * bytes_per_token)

    def _prepare_data(self, dev_id):
        dev = f'cuda:{dev_id}'
        seq_lens = torch.full((self.batch_size,), self.seq_len, dtype=torch.int32, device=dev)
        cu_seq_lens = torch.zeros(self.batch_size + 1, dtype=torch.int32, device=dev)
        torch.cumsum(seq_lens, dim=0, out=cu_seq_lens[1:])
        total_tokens = cu_seq_lens[-1].item()
        blocks_per_seq = (self.seq_len + self.block_size - 1) // self.block_size
        block_table = torch.randint(
            0, self.num_blocks, 
            (self.batch_size, blocks_per_seq), 
            dtype=torch.int32, device=dev
        )
        data_area_size = self.block_size * self.head_dim
        scale_area_size = self.block_size * self.scale_bytes_per_token
        block_total_bytes = data_area_size + scale_area_size
        kv_cache_flat = torch.randint(
            0, 127, 
            (self.num_blocks, block_total_bytes), 
            dtype=self.dtype, device=dev
        )
        kv_cache_view = kv_cache_flat.view(self.num_blocks, 1, block_total_bytes).expand(-1, self.block_size, -1)
        dst_k = torch.zeros(
            total_tokens, self.head_dim, 
            dtype=self.dtype, device=dev
        )
        dst_scale = torch.zeros(
            total_tokens, self.scale_bytes_per_token, 
            dtype=self.dtype, device=dev
        )
        return kv_cache_flat, kv_cache_view, dst_k, dst_scale, block_table, cu_seq_lens

    def prepare_and_get_launcher(self, dev_id, tc_s):
        _, kv_cache_view, dst_k, dst_scale, block_table, cu_seq_lens = self._prepare_data(dev_id)
        return self.make_launcher(
            dev_id, 
            torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache, 
            kv_cache_view, dst_k, dst_scale, block_table, cu_seq_lens
        )

    def run_verification(self, dev_id):
        kv_cache_flat, kv_cache_view, dst_k, dst_scale, block_table, cu_seq_lens = self._prepare_data(dev_id)
        torch.cuda.synchronize()
        try:
            torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
                kv_cache_view, dst_k, dst_scale, block_table, cu_seq_lens
            )
        except Exception as e:
            print(f"\n[Error] Exec failed: {e}")
            return False, 1.0
        torch.cuda.synchronize()
        out_k_ref = torch.zeros_like(dst_k)
        out_scale_ref = torch.zeros_like(dst_scale)
        cpu_cu_seq_lens = cu_seq_lens.cpu()
        data_area_size = self.block_size * self.head_dim
        kv_data_reshaped = kv_cache_flat[:, :data_area_size].reshape(self.num_blocks, self.block_size, self.head_dim)
        kv_scale_reshaped = kv_cache_flat[:, data_area_size:].reshape(self.num_blocks, self.block_size, self.scale_bytes_per_token)
        for b in range(self.batch_size):
            start = cpu_cu_seq_lens[b].item()
            end = cpu_cu_seq_lens[b+1].item()
            seq_len = end - start
            if seq_len == 0: continue
            token_indices = torch.arange(seq_len, device=kv_cache_flat.device)
            logical_block_indices = token_indices // self.block_size
            block_offsets = token_indices % self.block_size
            physical_block_ids = block_table[b][logical_block_indices]
            out_k_ref[start:end] = kv_data_reshaped[physical_block_ids, block_offsets, :]
            out_scale_ref[start:end] = kv_scale_reshaped[physical_block_ids, block_offsets, :]
        num_scales = self.head_dim // self.quant_block_size
        scale_size_bytes = 4
        mask = torch.zeros(self.scale_bytes_per_token, dtype=torch.uint8, device=dst_scale.device)
        for i in range(num_scales):
            head_idx = i * self.quant_block_size
            if head_idx % 128 == 0:
                mask[i*4 : (i+1)*4] = 1
        out_scale_ref = out_scale_ref * mask.unsqueeze(0)
        passed_k, diff_k = self.check_diff(dst_k, out_k_ref)
        passed_s, diff_s = self.check_diff(dst_scale, out_scale_ref)
        if not passed_k or not passed_s:
             print(f"\n[Verify Detail] Data Diff: {diff_k:.2e}, Scale Diff: {diff_s:.2e}")
        out_op_flat = torch.cat([dst_k.flatten(), dst_scale.flatten()])
        out_ref_flat = torch.cat([out_k_ref.flatten(), out_scale_ref.flatten()])
        return self.check_diff(out_op_flat, out_ref_flat)