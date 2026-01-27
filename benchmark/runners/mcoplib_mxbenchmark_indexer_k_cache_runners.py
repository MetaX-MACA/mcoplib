import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 尝试导入 mcoplib._C 以确保算子注册
try:
    import mcoplib._C
except ImportError:
    pass

class Indexer_k_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 1024 * 1024)
        self.block_size = config.get("block_size", 16)
        self.head_dim = config.get("head_dim", 128)
        self.num_blocks = config.get("num_blocks", 65536)
        
        # 确保 block 总容量足够容纳所有 token
        min_blocks = (self.num_tokens + self.block_size - 1) // self.block_size
        if self.num_blocks < min_blocks:
            self.num_blocks = min_blocks + 1024

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(Tokens={self.num_tokens} H={self.head_dim})")
        
        # 计算元素总数 (只计算有效搬运的 K 数据)
        total_elements = self.num_tokens * self.head_dim
        state.add_element_count(total_elements)
        
        # 计算显存读写量
        element_size = 2 if self.dtype == torch.float16 else 4
        
        # Read: k (Tensor) + slot_mapping (Int64/Int32)
        # Write: kv_cache (Tensor, scattered)
        
        # slot_mapping 通常是 int64 或 int32，这里按 4 bytes 估算
        idx_bytes = self.num_tokens * 4
        data_bytes = total_elements * element_size
        
        rw_bytes = (data_bytes * 2) + idx_bytes
        
        state.add_global_memory_reads(data_bytes + idx_bytes)
        state.add_global_memory_writes(data_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 1. 构造输入 k [num_tokens, head_dim]
            k = torch.randn((self.num_tokens, self.head_dim), dtype=self.dtype, device=dev)
            
            # 2. 构造 kv_cache [num_blocks, block_size, head_dim]
            # cache_stride 通常等于 head_dim
            cache_shape = (self.num_blocks, self.block_size, self.head_dim)
            kv_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
            
            # 3. 构造 slot_mapping [num_tokens]
            # 生成不重复的随机 slot 索引，模拟真实场景中的离散写入
            # 范围在 [0, num_blocks * block_size) 之间
            max_slots = self.num_blocks * self.block_size
            # 为了性能，Benchmark 时可以用线性索引，或者预生成的随机索引
            # 这里使用线性索引打乱，确保唯一性
            slot_mapping = torch.randperm(max_slots, device=dev, dtype=torch.int64)[:self.num_tokens]
            
        return self.make_launcher(dev_id, torch.ops._C_cache_ops.indexer_k_cache, 
                                  k, kv_cache, slot_mapping)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        # --- 准备数据 ---
        k = torch.randn((self.num_tokens, self.head_dim), dtype=self.dtype, device=dev)
        
        cache_shape = (self.num_blocks, self.block_size, self.head_dim)
        kv_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
        
        max_slots = self.num_blocks * self.block_size
        # 验证阶段使用较小的 token 数以加快速度，或者使用固定映射
        # 这里为了确保覆盖，使用全部 token，但在对比时只取部分或全部
        slot_mapping = torch.randperm(max_slots, device=dev, dtype=torch.int64)[:self.num_tokens]
        
        # --- 运行算子 ---
        torch.ops._C_cache_ops.indexer_k_cache(k, kv_cache, slot_mapping)
        
        # --- Python 验证 (回读校验) ---
        # 逻辑：算子把 k[i] 写到了 kv_cache 的 slot_mapping[i] 位置
        # 验证：我们直接用 slot_mapping 从 kv_cache 把数据取出来，看是否等于 k
        
        # 计算物理坐标
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size
        
        # 从 Cache 中 Gather 回来 (PyTorch Advanced Indexing)
        # kv_cache: [num_blocks, block_size, head_dim]
        # index: [num_tokens]
        k_recovered = kv_cache[block_indices, block_offsets, :]
        
        # --- 比较 ---
        # 使用 check_diff 计算余弦相似度
        return self.check_diff(k_recovered, k)