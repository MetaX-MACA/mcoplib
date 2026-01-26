import torch
import random
import math
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Paged_attention_v1_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_seqs = config.get("num_seqs", 7)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.block_size = config.get("block_size", 16)
        self.num_query_heads = config.get("num_query_heads", 32)
        self.num_blocks = config.get("num_blocks", 128)
        self.max_seq_len = config.get("max_seq_len", 256)
        self.seed = config.get("seed", 0)
        self.scale = config.get("scale", 1.0 / (self.head_size ** 0.5))

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_seqs} {self.num_query_heads} {self.head_size})")

        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        q_elements = self.num_seqs * self.num_query_heads * self.head_size
        kv_elements = self.num_blocks * self.num_kv_heads * self.block_size * self.head_size * 2

        state.add_element_count(q_elements + kv_elements)
        
        total_read_bytes = (q_elements * element_size) + (kv_elements * element_size)
        total_write_bytes = q_elements * element_size
        
        state.add_global_memory_reads(total_read_bytes)
        state.add_global_memory_writes(total_write_bytes)

    def _generate_args(self, dev_id):
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        dev = f'cuda:{dev_id}'
        
        # 1. Query
        query = torch.randn(self.num_seqs, self.num_query_heads, self.head_size, dtype=self.dtype, device=dev)
        
        # 2. Key Cache 
        # C++ Kernel layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        # x = 16 (for bfloat16/half)
        x = 16
        key_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.head_size // x, self.block_size, x, dtype=self.dtype, device=dev)
        
        # 3. Value Cache
        # C++ Kernel layout (Modified): [num_blocks, num_kv_heads, block_size, head_size]
        value_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_size, dtype=self.dtype, device=dev)
        
        # 4. Seq Lens
        seq_lens_list = [random.randint(1, self.max_seq_len) for _ in range(self.num_seqs)]
        seq_lens_list[-1] = self.max_seq_len 
        seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=dev)
        max_seq_len_actual = max(seq_lens_list)
        
        # 5. Block Tables
        max_num_blocks_per_seq = (max_seq_len_actual + self.block_size - 1) // self.block_size
        block_tables = torch.randint(0, self.num_blocks, (self.num_seqs, max_num_blocks_per_seq), dtype=torch.int32, device=dev)
        
        # 6. Output 
        output = torch.zeros_like(query)
        
        alibi_slopes = None
        kv_cache_dtype = "auto"
        
        # 修复：必须是 Tensor 类型
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        
        tp_rank = 0
        blocksparse_local_blocks = 0
        blocksparse_vert_stride = 0
        blocksparse_block_size = 0
        blocksparse_head_sliding_step = 0
        
        # 修复：必须 contiguous，否则 C++ 指针计算错误
        return (
            output.contiguous(), 
            query.contiguous(), 
            key_cache.contiguous(), 
            value_cache.contiguous(),
            self.num_kv_heads, 
            self.scale, 
            block_tables.contiguous(), 
            seq_lens.contiguous(),
            self.block_size, 
            max_seq_len_actual, 
            alibi_slopes,
            kv_cache_dtype, 
            k_scale, 
            v_scale, 
            tp_rank,
            blocksparse_local_blocks, 
            blocksparse_vert_stride,
            blocksparse_block_size, 
            blocksparse_head_sliding_step
        )

    def _ref_paged_attention(self, query, key_cache, value_cache, block_tables, seq_lens, scale):
        query_fp32 = query.float()
        
        # Key Cache Reconstruct: [Blocks, Heads, D/x, B, x] -> [Blocks, Heads, B, D]
        num_blocks, num_kv_heads, head_div_x, block_size, x = key_cache.shape
        head_size = head_div_x * x
        
        # Permute: (0, 1, 3, 2, 4) -> [Blocks, Heads, B, D/x, x] -> View
        kc = key_cache.float().permute(0, 1, 3, 2, 4).contiguous().view(num_blocks, num_kv_heads, block_size, head_size)
        
        # Value Cache: [Blocks, Heads, B, D]
        vc = value_cache.float()
        
        ref_out = torch.empty_like(query_fp32)
        
        num_seqs, num_q_heads, _ = query.shape
        groups = num_q_heads // num_kv_heads

        for i in range(num_seqs):
            sl = seq_lens[i].item()
            block_ids = block_tables[i]
            
            num_blocks_needed = (sl + self.block_size - 1) // self.block_size
            valid_block_ids = block_ids[:num_blocks_needed].long()
            
            k_blocks = kc[valid_block_ids]
            v_blocks = vc[valid_block_ids]
            
            # 展平时间维度: [Block, Head, Time, Dim] -> [TotalTime, Head, Dim]
            # 先 permute 成 [Block, Time, Head, Dim] 再 reshape
            k_seq = k_blocks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_size)
            v_seq = v_blocks.permute(0, 2, 1, 3).reshape(-1, num_kv_heads, head_size)
            
            k_seq = k_seq[:sl]
            v_seq = v_seq[:sl]
            
            for h in range(num_q_heads):
                kv_h = h // groups
                q_h = query_fp32[i, h]
                k_h = k_seq[:, kv_h, :]
                v_h = v_seq[:, kv_h, :]
                
                scores = torch.matmul(k_h, q_h) * scale
                attn = torch.softmax(scores, dim=0)
                out_h = torch.matmul(attn, v_h)
                ref_out[i, h] = out_h

        return ref_out.to(query.dtype)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            args = self._generate_args(dev_id)
        return self.make_launcher(dev_id, torch.ops._C.paged_attention_v1, *args)

    def run_verification(self, dev_id):
        args = self._generate_args(dev_id)
        output = args[0]
        
        torch.ops._C.paged_attention_v1(*args)
        
        query = args[1]
        key_cache = args[2]
        value_cache = args[3]
        block_tables = args[6]
        seq_lens = args[7]
        scale = args[5]
        
        ref_out = self._ref_paged_attention(query, key_cache, value_cache, block_tables, seq_lens, scale)
        
        # 调试信息：打印均值以检测全0输出
        # print(f"Op Mean: {output.float().abs().mean().item():.6f}, Ref Mean: {ref_out.float().abs().mean().item():.6f}")
        
        return self.check_diff(output, ref_out)