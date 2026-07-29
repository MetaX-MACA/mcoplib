import torch
import random
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib._C

class Paged_attention_v2_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_seqs = config.get("num_seqs", 7)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.block_size = config.get("block_size", 16)
        self.num_query_heads = config.get("num_query_heads", 32)
        self.num_blocks = config.get("num_blocks", 128)
        self.max_seq_len = config.get("max_seq_len", 256)
        self.partition_size = config.get("partition_size", 512)
        self.seed = config.get("seed", 0)
        self.x = 16
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(Seqs:{self.num_seqs} QHeads:{self.num_query_heads} HeadSize:{self.head_size})")
        total_elements = self.num_seqs * self.num_query_heads * self.head_size
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        state.add_global_memory_reads(total_elements * 3 * element_size) 
        state.add_global_memory_writes(total_elements * 1 * element_size)

    def _prepare_data(self, dev_id):
        dev = f'cuda:{dev_id}'
        dtype = self.dtype
        scale = 1.0 / (self.head_size ** 0.5)
        query = torch.randn(self.num_seqs, self.num_query_heads, self.head_size, dtype=dtype, device=dev)
        key_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.head_size // self.x, self.block_size, self.x, dtype=dtype, device=dev)
        value_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_size, dtype=dtype, device=dev)
        seq_lens_list = [random.randint(1, self.max_seq_len) for _ in range(self.num_seqs)]
        seq_lens_list[-1] = self.max_seq_len
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int, device=dev)
        max_seq_len_actual = max(seq_lens_list)
        max_num_blocks_per_seq = (max_seq_len_actual + self.block_size - 1) // self.block_size
        block_tables = torch.randint(0, self.num_blocks, (self.num_seqs, max_num_blocks_per_seq), dtype=torch.int, device=dev)
        max_num_partitions = (max_seq_len_actual + self.partition_size - 1) // self.partition_size
        output = torch.empty_like(query)
        exp_sums = torch.empty(self.num_seqs, self.num_query_heads, max_num_partitions, dtype=torch.float32, device=dev)
        max_logits = torch.empty(self.num_seqs, self.num_query_heads, max_num_partitions, dtype=torch.float32, device=dev)
        tmp_out = torch.empty(self.num_seqs, self.num_query_heads, max_num_partitions, self.head_size, dtype=dtype, device=dev)
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        return (output, exp_sums, max_logits, tmp_out, query, key_cache, value_cache, 
                scale, block_tables, seq_lens, max_seq_len_actual, k_scale, v_scale)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            data = self._prepare_data(dev_id)
            (output, exp_sums, max_logits, tmp_out, query, key_cache, value_cache, 
             scale, block_tables, seq_lens, max_seq_len_actual, k_scale, v_scale) = data
            alibi_slopes = None
            kv_cache_dtype = "auto"
            tp_rank = 0
            blocksparse_local_blocks = 0
            blocksparse_vert_stride = 0
            blocksparse_block_size = 0
            blocksparse_head_sliding_step = 0
            return self.make_launcher(dev_id, torch.ops._C.paged_attention_v2,
                                      output, exp_sums, max_logits, tmp_out, query,
                                      key_cache, value_cache, self.num_kv_heads, scale,
                                      block_tables, seq_lens, self.block_size, max_seq_len_actual,
                                      alibi_slopes, kv_cache_dtype, k_scale, v_scale,
                                      tp_rank, blocksparse_local_blocks, blocksparse_vert_stride,
                                      blocksparse_block_size, blocksparse_head_sliding_step)

    def _ref_paged_attention(self, query, key_cache, value_cache, block_tables, seq_lens, scale):
        out_list = []
        num_seqs, num_heads, head_size = query.shape
        num_kv_heads = key_cache.shape[1]
        for i in range(num_seqs):
            sl = seq_lens[i].item()
            q_i = query[i].unsqueeze(0).float()
            block_indices = block_tables[i]
            num_blocks_needed = (sl + self.block_size - 1) // self.block_size
            valid_block_indices = block_indices[:num_blocks_needed].long()
            k_blocks = key_cache[valid_block_indices]
            v_blocks = value_cache[valid_block_indices]
            k_i_full = k_blocks.permute(0, 1, 3, 2, 4).reshape(-1, num_kv_heads, head_size)
            v_i_full = v_blocks.reshape(-1, num_kv_heads, head_size)
            k_i = k_i_full[:sl].float()
            v_i = v_i_full[:sl].float()
            if num_heads != num_kv_heads:
                k_i = k_i.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v_i = v_i.repeat_interleave(num_heads // num_kv_heads, dim=1)
            attn_weights = torch.einsum("bhd,lhd->bhl", q_i, k_i) * scale
            attn_weights = torch.softmax(attn_weights, dim=-1)
            out_i = torch.einsum("bhl,lhd->bhd", attn_weights, v_i)
            out_list.append(out_i.squeeze(0))
        return torch.stack(out_list).to(query.dtype)

    def run_verification(self, dev_id):
        data = self._prepare_data(dev_id)
        (output, exp_sums, max_logits, tmp_out, query, key_cache, value_cache, 
         scale, block_tables, seq_lens, max_seq_len_actual, k_scale, v_scale) = data
        torch.ops._C.paged_attention_v2(
            output, exp_sums, max_logits, tmp_out, query,
            key_cache, value_cache, self.num_kv_heads, scale,
            block_tables, seq_lens, self.block_size, max_seq_len_actual,
            None, "auto", k_scale, v_scale,
            0, 0, 0, 0, 0
        )
        ref_out = self._ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens, scale)
        return self.check_diff(output, ref_out)