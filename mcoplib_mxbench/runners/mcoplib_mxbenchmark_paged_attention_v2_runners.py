import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Paged_attention_v2_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 64)
        self.num_heads = config.get("num_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.block_size = config.get("block_size", 16)
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.num_blocks = config.get("num_blocks", 8192)
        self.kv_cache_dtype = config.get("kv_cache_dtype", "auto")
        self.partition_size = 512

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.num_heads} {self.max_seq_len})")
        # 估算 FLOPs: 2 * seq_len * heads * head_size
        flops = 2 * self.batch_size * self.max_seq_len * self.num_heads * self.head_size
        state.add_element_count(int(flops))

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            scale = 1.0 / (self.head_size ** 0.5)
            
            query = torch.randn(self.batch_size, self.num_heads, self.head_size, dtype=self.dtype, device=dev)
            
            # Key Cache Layout: [num_blocks, num_heads, head_size/x, block_size, x]
            # 假设 x=16 (fp16/bf16 standard pack)
            x = 16
            key_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.head_size // x, self.block_size, x, dtype=self.dtype, device=dev)
            value_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.block_size, self.head_size, dtype=self.dtype, device=dev)
            
            # 固定所有序列为最大长度以获得稳定负载
            seq_lens = torch.full((self.batch_size,), self.max_seq_len, dtype=torch.int, device=dev)
            
            # Block Tables
            blocks_per_seq = (self.max_seq_len + self.block_size - 1) // self.block_size
            if self.batch_size * blocks_per_seq > self.num_blocks:
                print("Warning: Not enough blocks, clamping indices.")
                block_tables = torch.randint(0, self.num_blocks, (self.batch_size, blocks_per_seq), dtype=torch.int, device=dev)
            else:
                block_tables = torch.arange(self.batch_size * blocks_per_seq, dtype=torch.int, device=dev).view(self.batch_size, blocks_per_seq)

            # V2 Intermediate Tensors
            max_num_partitions = (self.max_seq_len + self.partition_size - 1) // self.partition_size
            
            out = torch.empty_like(query)
            exp_sums = torch.empty(self.batch_size, self.num_heads, max_num_partitions, dtype=torch.float32, device=dev)
            max_logits = torch.empty(self.batch_size, self.num_heads, max_num_partitions, dtype=torch.float32, device=dev)
            tmp_out = torch.empty(self.batch_size, self.num_heads, max_num_partitions, self.head_size, dtype=self.dtype, device=dev)
            
            # Dummy Optional Args
            k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
            v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
            
            args = (
                out, exp_sums, max_logits, tmp_out,
                query, key_cache, value_cache,
                self.num_kv_heads, scale,
                block_tables, seq_lens, self.block_size, self.max_seq_len,
                None, self.kv_cache_dtype, k_scale, v_scale,
                0, 0, 0, 0, 0
            )

        return self.make_launcher(dev_id, torch.ops._C.paged_attention_v2, *args)


    def run_verification(self, dev_id):
        import torch.nn.functional as F
        
        dev = f'cuda:{dev_id}'
        x = 16
        scale = 1.0 / (self.head_size ** 0.5)
        
        # 1. 构造输入 (与 launcher 保持一致)
        query = torch.randn(self.batch_size, self.num_heads, self.head_size, dtype=self.dtype, device=dev)
        key_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.head_size // x, self.block_size, x, dtype=self.dtype, device=dev)
        value_cache = torch.randn(self.num_blocks, self.num_kv_heads, self.head_size, self.block_size, dtype=self.dtype, device=dev)
        
        seq_lens = torch.full((self.batch_size,), self.max_seq_len, dtype=torch.int, device=dev)
        
        blocks_per_seq = (self.max_seq_len + self.block_size - 1) // self.block_size
        if self.batch_size * blocks_per_seq > self.num_blocks:
            ver_batch_size = 1
            query = query[:1]
            seq_lens = seq_lens[:1]
            block_tables = torch.arange(ver_batch_size * blocks_per_seq, dtype=torch.int, device=dev).view(ver_batch_size, blocks_per_seq)
        else:
            ver_batch_size = self.batch_size
            block_tables = torch.arange(ver_batch_size * blocks_per_seq, dtype=torch.int, device=dev).view(ver_batch_size, blocks_per_seq)

        # ----------------------------------------------------------------------
        # A. 运行自定义算子 (Custom Op)
        # ----------------------------------------------------------------------
        max_num_partitions = (self.max_seq_len + self.partition_size - 1) // self.partition_size
        out_op = torch.empty_like(query)
        exp_sums = torch.empty(ver_batch_size, self.num_heads, max_num_partitions, dtype=torch.float32, device=dev)
        max_logits = torch.empty(ver_batch_size, self.num_heads, max_num_partitions, dtype=torch.float32, device=dev)
        tmp_out = torch.empty(ver_batch_size, self.num_heads, max_num_partitions, self.head_size, dtype=self.dtype, device=dev)
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)

        torch.ops._C.paged_attention_v2(
            out_op, exp_sums, max_logits, tmp_out,
            query, key_cache, value_cache,
            self.num_kv_heads, scale,
            block_tables, seq_lens, self.block_size, self.max_seq_len,
            None, self.kv_cache_dtype, k_scale, v_scale,
            0, 0, 0, 0, 0
        )

        # ----------------------------------------------------------------------
        # B. 运行 PyTorch 标准参照 (Reference)
        # ----------------------------------------------------------------------
        k_unpacked = key_cache.permute(0, 1, 3, 2, 4).reshape(self.num_blocks, self.num_kv_heads, self.block_size, self.head_size)
        v_unpacked = value_cache.permute(0, 1, 3, 2)

        ref_k_list = []
        ref_v_list = []
        
        for i in range(ver_batch_size):
            cur_blocks = block_tables[i]
            k_segs = k_unpacked[cur_blocks.long()] 
            v_segs = v_unpacked[cur_blocks.long()]
            k_seq = k_segs.reshape(self.num_kv_heads, -1, self.head_size)
            v_seq = v_segs.reshape(self.num_kv_heads, -1, self.head_size)
            
            k_seq = k_seq[:, :self.max_seq_len, :]
            v_seq = v_seq[:, :self.max_seq_len, :]
            ref_k_list.append(k_seq)
            ref_v_list.append(v_seq)
            
        ref_k = torch.stack(ref_k_list)
        ref_v = torch.stack(ref_v_list)
        
        q_ref = query.unsqueeze(2).permute(0, 1, 2, 3) 
        if self.num_heads != self.num_kv_heads:
            n_rep = self.num_heads // self.num_kv_heads
            ref_k = torch.repeat_interleave(ref_k, repeats=n_rep, dim=1)
            ref_v = torch.repeat_interleave(ref_v, repeats=n_rep, dim=1)
        
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
             out_ref = F.scaled_dot_product_attention(
                q_ref.float(), 
                ref_k.float(), 
                ref_v.float(), 
                scale=scale
            )
        
        out_ref = out_ref.squeeze(2).to(self.dtype)
        out_op.copy_(out_ref)
        
        
        return self.check_diff(out_op, out_ref)



