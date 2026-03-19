import torch
import numpy as np
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Transfer_kv_all_layer_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_layers = config.get("num_layers", 32)
        self.num_tokens = config.get("num_tokens", 1024)
        self.head_num = config.get("head_num", 8)
        self.head_dim = config.get("head_dim", 128)
        self.block_quota = config.get("block_quota", 128)
        self.num_warps_per_block = config.get("num_warps_per_block", 4)
        
        self.elements_per_item = self.head_num * self.head_dim
        self.element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        self.item_size = self.elements_per_item * self.element_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_layers}L, {self.num_tokens}T)")
        total_elements = self.num_layers * self.num_tokens * self.elements_per_item * 2
        state.add_element_count(total_elements)
        state.add_global_memory_reads(total_elements * self.element_size)
        state.add_global_memory_writes(total_elements * self.element_size)

    def _to_uint64_ptr(self, tensors, dev):
        """核心修复点：将Tensor的内存地址转换为底层要求的UInt64指针张量"""
        return torch.tensor(
            np.array([sub.data_ptr() for sub in tensors], dtype=np.uint64),
            device=dev, dtype=torch.uint64
        )

    def _create_tensors(self, dev):
        src_k_list = [torch.randn((self.num_tokens, self.elements_per_item), dtype=self.dtype, device=dev) for _ in range(self.num_layers)]
        src_v_list = [torch.randn((self.num_tokens, self.elements_per_item), dtype=self.dtype, device=dev) for _ in range(self.num_layers)]
        dst_k_list = [torch.zeros_like(t) for t in src_k_list]
        dst_v_list = [torch.zeros_like(t) for t in src_v_list]
        
        src_k_ptrs = self._to_uint64_ptr(src_k_list, dev)
        src_v_ptrs = self._to_uint64_ptr(src_v_list, dev)
        dst_k_ptrs = self._to_uint64_ptr(dst_k_list, dev)
        dst_v_ptrs = self._to_uint64_ptr(dst_v_list, dev)
        
        src_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        dst_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        
        return src_k_list, src_v_list, dst_k_list, dst_v_list, src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, src_indices, dst_indices

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        # 挂载为成员变量，防止GC回收导致悬空指针
        self.tensors = self._create_tensors(dev)
        _, _, _, _, src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, src_indices, dst_indices = self.tensors
        
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.transfer_kv_all_layer,
            src_k_ptrs, dst_k_ptrs, src_v_ptrs, dst_v_ptrs,
            src_indices, dst_indices,
            self.item_size, self.num_layers,
            self.block_quota, self.num_warps_per_block
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        src_k_list, src_v_list, dst_k_list, dst_v_list, src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, src_indices, dst_indices = self._create_tensors(dev)
        
        torch.ops.sgl_kernel.transfer_kv_all_layer(
            src_k_ptrs, dst_k_ptrs, src_v_ptrs, dst_v_ptrs,
            src_indices, dst_indices,
            self.item_size, self.num_layers,
            self.block_quota, self.num_warps_per_block
        )
        
        is_passed = True
        max_diff = 0.0
        for i in range(self.num_layers):
            # 检查第 i 层的目标 K 和 V 是否等于源 V
            pass_k, diff_k = self.check_diff(dst_k_list[i], src_k_list[i])
            pass_v, diff_v = self.check_diff(dst_v_list[i], src_v_list[i])
            is_passed = is_passed and pass_k and pass_v
            max_diff = max(max_diff, diff_k, diff_v)
            
        return is_passed, max_diff