import torch
import numpy as np
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Transfer_kv_all_layer_mla_lf_pf_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_layers = config.get("num_layers", 32)
        self.num_tokens = config.get("num_tokens", 8192)
        self.head_num = config.get("head_num", 32)
        self.head_dim = config.get("head_dim", 128)
        self.block_quota = config.get("block_quota", 256)
        self.num_warps_per_block = config.get("num_warps_per_block", 4)
        
        self.elements_per_item = self.head_num * self.head_dim
        self.element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        # 传递给底层的都要是 Byte Size
        self.item_byte_size = self.elements_per_item * self.element_size
        self.dst_layout_byte_dim = self.num_layers * self.item_byte_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_layers}L, {self.num_tokens}T)")
        total_elements = self.num_layers * self.num_tokens * self.elements_per_item
        state.add_element_count(total_elements)
        state.add_global_memory_reads(total_elements * self.element_size)
        state.add_global_memory_writes(total_elements * self.element_size)

    def _to_uint64_ptr(self, tensors, dev):
        return torch.tensor(
            np.array([sub.data_ptr() for sub in tensors], dtype=np.uint64),
            device=dev, dtype=torch.uint64
        )

    def _create_tensors(self, dev):
        # 优化 1：Bulk Allocation (大块分配)。降低 Allocator 压力
        src_bulk = torch.randn((self.num_layers, self.num_tokens, self.elements_per_item), dtype=self.dtype, device=dev)
        
        # 使用视图切片代替独立生成张量
        src_layers = [src_bulk[i] for i in range(self.num_layers)]
        
        # dst 已经是连续块
        dst = torch.zeros((self.num_tokens, self.num_layers * self.elements_per_item), dtype=self.dtype, device=dev)
        
        src_ptrs = self._to_uint64_ptr(src_layers, dev)
        
        src_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        dst_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        
        return src_bulk, src_layers, dst, src_ptrs, src_indices, dst_indices

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        self.tensors = self._create_tensors(dev)
        # 解包 (忽略前两个 bulk/list 引用)
        _, _, dst, src_ptrs, src_indices, dst_indices = self.tensors
        
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf,
            src_ptrs, dst, src_indices, dst_indices,
            self.item_byte_size, self.dst_layout_byte_dim, self.num_layers,
            self.block_quota, self.num_warps_per_block
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        src_bulk, src_layers, dst, src_ptrs, src_indices, dst_indices = self._create_tensors(dev)
        
        torch.ops.sgl_kernel.transfer_kv_all_layer_mla_lf_pf(
            src_ptrs, dst, src_indices, dst_indices,
            self.item_byte_size, self.dst_layout_byte_dim, self.num_layers,
            self.block_quota, self.num_warps_per_block
        )
        
        is_passed = True
        max_diff = 0.0
        
        dst_view = dst.view(self.num_tokens, self.num_layers, self.elements_per_item)
        
        for i in range(self.num_layers):
            pass_k, diff_k = self.check_diff(dst_view[:, i, :], src_layers[i])
            is_passed = is_passed and pass_k
            max_diff = max(max_diff, diff_k)
            
        # 优化 2：强制垃圾回收和清空显存缓存，防止 10 次 Warmup 报错
        del src_bulk, src_layers, dst, src_ptrs, src_indices, dst_indices, dst_view
        torch.cuda.empty_cache()
            
        return is_passed, max_diff