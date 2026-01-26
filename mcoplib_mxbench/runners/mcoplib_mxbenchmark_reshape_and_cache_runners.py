import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Reshape_and_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        # 读取配置参数
        self.num_tokens = config.get("num_tokens", 128)
        self.num_heads = config.get("num_heads", 32)
        self.head_size = config.get("head_size", 128)
        self.block_size = config.get("block_size", 16)
        self.num_blocks = config.get("num_blocks", 1024)
        self.x = config.get("x", 16)
        self.kv_cache_dtype = config.get("kv_cache_dtype", "auto")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens} {self.num_heads} {self.head_size})")
        
        # 计算涉及的元素总数
        input_elements = self.num_tokens * self.num_heads * self.head_size
        total_elements = input_elements * 2
        state.add_element_count(total_elements)

        # 估算显存带宽 (Global Memory Access)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        
        # Reads: Key + Value
        read_bytes = total_elements * element_size
        # Reads: Slot Mapping (int64)
        read_bytes += self.num_tokens * 8
        
        # Writes: KeyCache + ValueCache
        write_bytes = total_elements * element_size

        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        """准备数据并返回 Launcher"""
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = self._generate_inputs(dev_id)
        
        op_func = torch.ops._C_cache_ops.reshape_and_cache

        return self.make_launcher(dev_id, op_func,
                                  key, value, key_cache, value_cache,
                                  slot_mapping, self.kv_cache_dtype, k_scale, v_scale)

    def run_verification(self, dev_id):
        """执行一次算子并验证结果"""
        # 1. 准备数据
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = self._generate_inputs(dev_id)
        
        # 2. 运行算子
        torch.ops._C_cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache,
            slot_mapping, self.kv_cache_dtype, k_scale, v_scale
        )
        
        # 3. 验证逻辑 (Reconstruction Check)
        # 修复：使用更稳健的 Flatten + Gather 方式，避免 Advanced Indexing 的维度歧义
        
        # ---------------- 验证 Value Cache ----------------
        # Layout: [num_blocks, num_heads, head_size, block_size]
        # 目标: 将其视为 [Total_Slots, num_heads, head_size] 并按 slot_mapping 提取
        
        # 1. Permute: 把 block 和 block_size 移到前面相邻 -> [num_blocks, block_size, num_heads, head_size]
        vc_permuted = value_cache.permute(0, 3, 1, 2)
        # 2. Flatten: 合并前两维 (Block * BlockSize = TotalSlots) -> [total_slots, num_heads, head_size]
        vc_flat = vc_permuted.flatten(0, 1)
        # 3. Gather: 直接使用 slot_mapping 提取对应的 Slot 数据 -> [num_tokens, num_heads, head_size]
        value_recon = vc_flat[slot_mapping]

        # ---------------- 验证 Key Cache ----------------
        # Layout: [num_blocks, num_heads, head_size/x, block_size, x]
        # 目标: 还原为 [num_tokens, num_heads, head_size]

        # 1. Permute: 把 block 和 block_size 移到前面 -> [num_blocks, block_size, num_heads, head_size/x, x]
        kc_permuted = key_cache.permute(0, 3, 1, 2, 4)
        # 2. Flatten: 合并前两维 -> [total_slots, num_heads, head_size/x, x]
        kc_flat = kc_permuted.flatten(0, 1)
        # 3. Gather: 提取对应 slot -> [num_tokens, num_heads, head_size/x, x]
        kc_gathered = kc_flat[slot_mapping]
        # 4. Flatten Last 2 Dims: 合并 (head_size/x) 和 x 还原为 head_size -> [num_tokens, num_heads, head_size]
        key_recon = kc_gathered.flatten(-2, -1)
        
        # 4. 对比
        # 转换成 float 进行对比，避免 FP16/BF16 的精度累积误差
        passed_k, diff_k = self.check_diff(key_recon, key.to(key_recon.dtype))
        passed_v, diff_v = self.check_diff(value_recon, value.to(value_recon.dtype))
        
        total_diff = max(diff_k, diff_v)
        return (passed_k and passed_v), total_diff

    def _generate_inputs(self, dev_id):
        """辅助函数：生成符合形状的随机输入"""
        dev = f'cuda:{dev_id}'
        
        input_shape = (self.num_tokens, self.num_heads, self.head_size)
        
        key = torch.randn(input_shape, dtype=self.dtype, device=dev)
        value = torch.randn(input_shape, dtype=self.dtype, device=dev)
        
        # Key Cache Layout: [num_blocks, num_heads, head_size/x, block_size, x]
        kc_shape = (self.num_blocks, self.num_heads, self.head_size // self.x, self.block_size, self.x)
        key_cache = torch.zeros(kc_shape, dtype=self.dtype, device=dev)
        
        # Value Cache Layout: [num_blocks, num_heads, head_size, block_size]
        vc_shape = (self.num_blocks, self.num_heads, self.head_size, self.block_size)
        value_cache = torch.zeros(vc_shape, dtype=self.dtype, device=dev)
        
        # Slot Mapping
        total_slots = self.num_blocks * self.block_size
        
        # [Critical Fix] 确保 slot 不重复！
        # 原来的 randint 会导致 collision (写入冲突)，使得数据被覆盖，从而导致验证失败
        if total_slots < self.num_tokens:
            raise ValueError(f"Total slots ({total_slots}) < num_tokens ({self.num_tokens}), cannot allocate unique slots.")
        
        # 使用 randperm 生成不重复的随机序列
        slot_mapping = torch.randperm(total_slots, dtype=torch.long, device=dev)[:self.num_tokens]
        
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        
        return key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale