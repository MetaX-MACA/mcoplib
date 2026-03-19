import torch
import numpy as np
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Transfer_kv_all_layer_lf_ph_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_layers = config.get("num_layers", 32)
        self.num_tokens = config.get("num_tokens", 1024)
        self.page_size = config.get("page_size", 16)
        self.head_num = config.get("head_num", 8)
        self.head_dim = config.get("head_dim", 128)
        self.block_quota = config.get("block_quota", 128)
        self.num_warps_per_block = config.get("num_warps_per_block", 4)
        
        self.elements_per_item = self.head_num * self.head_dim
        self.element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        self.item_size = self.elements_per_item * self.element_size
        self.dst_layout_dim = self.num_layers * self.item_size
        
        self.num_pages_alloc = (self.num_tokens + self.page_size - 1) // self.page_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_layers}L, {self.num_tokens}T)")
        total_elements = self.num_layers * self.num_tokens * self.elements_per_item * 2
        state.add_element_count(total_elements)
        state.add_global_memory_reads(total_elements * self.element_size)
        state.add_global_memory_writes(total_elements * self.element_size)

    def _to_uint64_ptr(self, tensors, dev):
        return torch.tensor(
            np.array([sub.data_ptr() for sub in tensors], dtype=np.uint64),
            device=dev, dtype=torch.uint64
        )

    def _create_tensors(self, dev):
        src_k_list = [torch.randn((self.num_tokens, self.elements_per_item), dtype=self.dtype, device=dev) for _ in range(self.num_layers)]
        src_v_list = [torch.randn((self.num_tokens, self.elements_per_item), dtype=self.dtype, device=dev) for _ in range(self.num_layers)]
        
        # PH layout: [page_num, head_num, page_size, layer_num, head_dim]
        dst_k = torch.zeros((self.num_pages_alloc, self.head_num, self.page_size, self.num_layers, self.head_dim), dtype=self.dtype, device=dev)
        dst_v = torch.zeros_like(dst_k)
        
        src_k_ptrs = self._to_uint64_ptr(src_k_list, dev)
        src_v_ptrs = self._to_uint64_ptr(src_v_list, dev)
        
        src_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        dst_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        
        return src_k_list, src_v_list, dst_k, dst_v, src_k_ptrs, src_v_ptrs, src_indices, dst_indices

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        self.tensors = self._create_tensors(dev)
        _, _, dst_k, dst_v, src_k_ptrs, src_v_ptrs, src_indices, dst_indices = self.tensors
        
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph,
            src_k_ptrs, dst_k, src_v_ptrs, dst_v,
            src_indices, dst_indices,
            self.item_size, self.dst_layout_dim, self.num_layers,
            self.page_size, self.head_num,
            self.block_quota, self.num_warps_per_block
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        src_k_list, src_v_list, dst_k, dst_v, src_k_ptrs, src_v_ptrs, src_indices, dst_indices = self._create_tensors(dev)
        
        torch.ops.sgl_kernel.transfer_kv_all_layer_lf_ph(
            src_k_ptrs, dst_k, src_v_ptrs, dst_v,
            src_indices, dst_indices,
            self.item_size, self.dst_layout_dim, self.num_layers,
            self.page_size, self.head_num,
            self.block_quota, self.num_warps_per_block
        )
        
        is_passed = True
        max_diff = 0.0
        
        for i in range(self.num_layers):
            reshaped_src_k = src_k_list[i].view(-1, self.head_num, self.head_dim)
            reshaped_src_v = src_v_list[i].view(-1, self.head_num, self.head_dim)
            
            # 转置回 [page_num, page_size, head_num, layer_num, head_dim] 后拉平
            dst_k_permuted = dst_k.permute(0, 2, 1, 3, 4).contiguous()
            dst_v_permuted = dst_v.permute(0, 2, 1, 3, 4).contiguous()
            
            dst_k_flat = dst_k_permuted.view(-1, self.head_num, self.num_layers, self.head_dim)
            dst_v_flat = dst_v_permuted.view(-1, self.head_num, self.num_layers, self.head_dim)
            
            valid_dst_k = dst_k_flat[:self.num_tokens, :, i, :]
            valid_dst_v = dst_v_flat[:self.num_tokens, :, i, :]
            
            pass_k, diff_k = self.check_diff(valid_dst_k, reshaped_src_k)
            pass_v, diff_v = self.check_diff(valid_dst_v, reshaped_src_v)
            
            is_passed = is_passed and pass_k and pass_v
            max_diff = max(max_diff, diff_k, diff_v)
            
        return is_passed, max_diff