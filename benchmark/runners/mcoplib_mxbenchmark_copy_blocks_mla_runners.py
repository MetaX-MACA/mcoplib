import torch
import random
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 尝试导入 _C 模块以注册 torch.ops
try:
    import mcoplib._C
except ImportError:
    pass

class Copy_blocks_mla_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_layers = config.get("num_layers", 32)
        self.num_blocks = config.get("num_blocks", 4096)
        self.block_size = config.get("block_size", 16)
        # MLA 通常使用联合的 Hidden Dim (例如 576)，而不是标准的 num_heads * head_size
        self.hidden_dim = config.get("hidden_dim", 576)
        self.num_pairs = config.get("num_pairs", 128)
        
        self.capability = (f"L{self.num_layers}_B{self.block_size}_"
                           f"D{self.hidden_dim}_P{self.num_pairs}")

        # [同步模式] 
        # copy 类算子涉及 D2D 拷贝和 CPU-GPU 交互，建议强制同步以获得准确时间并防止死锁
        self._force_sync = True 

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", self.capability)
        
        # [修改点 1] 移除 'torch.' 前缀
        dtype_str = str(self.dtype).replace("torch.", "")
        state.add_summary("dtype", dtype_str)
        
        state.add_element_count(1)
        
        # [修改点 2] 带宽计算 (针对 Copy Blocks MLA)
        # 1. 获取元素大小
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        
        # 2. 计算总搬运元素数
        # 公式: 层数 * 复制对数 * 块大小 * 隐藏层维度
        total_elements = self.num_layers * self.num_pairs * self.block_size * self.hidden_dim
        
        # 3. 计算总字节数
        total_bytes = total_elements * element_size
        
        # 4. 注册读写量 (读一次 + 写一次)
        state.add_global_memory_reads(total_bytes)
        state.add_global_memory_writes(total_bytes)

        # 禁用死锁检测
        try:
            state.set_blocking_kernel_timeout(-1.0)
        except Exception:
            pass

    def _prepare_data(self, dev_id):
        dev = f'cuda:{dev_id}'
        # MLA Kernel 假设 tensor 形状为 [num_blocks, block_size, hidden_dim]
        # 且使用 stride(0) 计算 block 的 footprint，因此必须是 contiguous 的
        cache_shape = (self.num_blocks, self.block_size, self.hidden_dim)
        
        kv_caches = [torch.randn(cache_shape, dtype=self.dtype, device=dev).contiguous() 
                     for _ in range(self.num_layers)]
        
        # 构造随机的拷贝对
        all_indices = list(range(self.num_blocks))
        src_indices = [random.choice(all_indices) for _ in range(self.num_pairs)]
        dst_indices = [random.choice(all_indices) for _ in range(self.num_pairs)]
        
        mapping_data = list(zip(src_indices, dst_indices))
        block_mapping = torch.tensor(mapping_data, dtype=torch.int64, device=dev)
        
        return kv_caches, block_mapping

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            kv_caches, block_mapping = self._prepare_data(dev_id)
            
        def op_launcher(*args):
            # 调用 C++ 注册的 copy_blocks_mla 算子
            torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)

        return self.make_launcher(dev_id, op_launcher)

    def run_verification(self, dev_id):
        kv_caches, block_mapping = self._prepare_data(dev_id)
        
        # 创建用于 CPU 验证的副本
        ref_kv_caches = [k.clone() for k in kv_caches]

        # 执行算子
        torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)
        
        # 执行 CPU 端的参考逻辑
        # 逻辑：将 src_idx 的 block 内容复制到 dst_idx
        mapping_cpu = block_mapping.cpu().numpy()
        for layer_idx in range(self.num_layers):
            for src_idx, dst_idx in mapping_cpu:
                ref_kv_caches[layer_idx][dst_idx].copy_(ref_kv_caches[layer_idx][src_idx])
        
        # 验证精度
        passed, diff = self.check_diff(kv_caches[0], ref_kv_caches[0])
        
        return passed, diff