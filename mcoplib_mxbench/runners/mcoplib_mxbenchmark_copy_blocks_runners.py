import torch
import random
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 必须导入 _C 模块以注册 torch.ops
try:
    import mcoplib._C
except ImportError:
    pass

class Copy_blocks_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_layers = config.get("num_layers", 32)
        self.num_blocks = config.get("num_blocks", 1024)
        self.block_size = config.get("block_size", 16)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_size = config.get("head_size", 128)
        self.num_pairs = config.get("num_pairs", 128)
        
        self.capability = (f"L{self.num_layers}_B{self.block_size}_"
                           f"H{self.num_kv_heads}_D{self.head_size}_P{self.num_pairs}")

        # =================================================================
        # [核心修改] 启用强制同步模式
        # 作用：告诉 ops.py 这个算子是同步的，必须用 bench.exec_tag.sync 测量
        # 否则 nvbench 的异步计时器会因为捕获不到 GPU 结束信号而死锁
        # =================================================================
        self._force_sync = True 

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", self.capability)
        
        # [修改点 1] 去掉 "torch." 前缀
        dtype_str = str(self.dtype).replace("torch.", "")
        state.add_summary("dtype", dtype_str)
        
        state.add_element_count(1)
        
        # [修改点 2] 添加带宽统计代码
        # 1. 获取元素大小
        element_size = torch.tensor([], dtype=self.dtype).element_size()

        # 2. 计算单个 Cache (仅 Key) 的搬运元素量
        # 公式: 层数 * 复制对数 * 块大小 * KV头数 * Head维度
        elements_per_cache = self.num_layers * self.num_pairs * self.block_size * self.num_kv_heads * self.head_size
        
        # 3. 计算 Key + Value 的总字节数
        # copy_blocks 同时搬运 Key 和 Value，所以要 * 2
        total_bytes = elements_per_cache * 2 * element_size

        # 4. 注册读写量
        state.add_global_memory_reads(total_bytes)
        state.add_global_memory_writes(total_bytes)

        # 禁用死锁检测
        try:
            state.set_blocking_kernel_timeout(-1.0)
        except Exception:
            pass

    def _prepare_data(self, dev_id):
        dev = f'cuda:{dev_id}'
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        x = 16 // element_size
        cache_shape = (self.num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        
        key_caches = [torch.randn(cache_shape, dtype=self.dtype, device=dev) 
                      for _ in range(self.num_layers)]
        value_caches = [torch.randn(cache_shape, dtype=self.dtype, device=dev) 
                        for _ in range(self.num_layers)]
        
        all_indices = list(range(self.num_blocks))
        src_indices = [random.choice(all_indices) for _ in range(self.num_pairs)]
        dst_indices = [random.choice(all_indices) for _ in range(self.num_pairs)]
        
        mapping_data = list(zip(src_indices, dst_indices))
        block_mapping = torch.tensor(mapping_data, dtype=torch.int64, device=dev)
        
        return key_caches, value_caches, block_mapping

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            key_caches, value_caches, block_mapping = self._prepare_data(dev_id)
            
        def op_launcher(*args):
             torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)

        return self.make_launcher(dev_id, op_launcher)

    def run_verification(self, dev_id):
        key_caches, value_caches, block_mapping = self._prepare_data(dev_id)
        
        ref_key_caches = [k.clone() for k in key_caches]
        ref_val_caches = [v.clone() for v in value_caches]

        torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, block_mapping)
        
        mapping_cpu = block_mapping.cpu().numpy()
        for layer_idx in range(self.num_layers):
            for src_idx, dst_idx in mapping_cpu:
                ref_key_caches[layer_idx][dst_idx].copy_(ref_key_caches[layer_idx][src_idx])
                ref_val_caches[layer_idx][dst_idx].copy_(ref_val_caches[layer_idx][src_idx])
        
        passed_k, diff_k = self.check_diff(key_caches[0], ref_key_caches[0])
        if not passed_k:
            return False, diff_k
            
        passed_v, diff_v = self.check_diff(value_caches[0], ref_val_caches[0])
        
        max_diff = max(diff_k, diff_v)
        return (passed_k and passed_v), max_diff