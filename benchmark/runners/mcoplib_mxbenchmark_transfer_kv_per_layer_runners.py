import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Transfer_kv_per_layer_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.total_pool_blocks = config.get("total_pool_blocks", 65536)
        self.num_blocks_to_copy = config.get("num_blocks_to_copy", 16384)
        self.elements_per_block = config.get("elements_per_block", 4096)
        self.block_quota = config.get("block_quota", 256)
        self.num_warps_per_block = config.get("num_warps_per_block", 4)
        
        test_tensor = torch.empty(0, dtype=self.dtype)
        self.element_size = test_tensor.element_size()
        self.item_size_bytes = self.elements_per_block * self.element_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"({self.num_blocks_to_copy}, {self.elements_per_block})")
        
        total_elements_transferred = self.num_blocks_to_copy * self.elements_per_block * 2
        state.add_element_count(total_elements_transferred)
        
        transferred_bytes = total_elements_transferred * self.element_size
        state.add_global_memory_reads(transferred_bytes)
        state.add_global_memory_writes(transferred_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            total_elements = self.total_pool_blocks * self.elements_per_block
            
            src_k = torch.randn(total_elements, dtype=self.dtype, device=dev)
            src_v = torch.randn(total_elements, dtype=self.dtype, device=dev)
            dst_k = torch.zeros(total_elements, dtype=self.dtype, device=dev)
            dst_v = torch.zeros(total_elements, dtype=self.dtype, device=dev)

            all_indices_src = torch.randperm(self.total_pool_blocks, device=dev)
            src_indices = all_indices_src[:self.num_blocks_to_copy].to(torch.int64)
            
            all_indices_dst = torch.randperm(self.total_pool_blocks, device=dev)
            dst_indices = all_indices_dst[:self.num_blocks_to_copy].to(torch.int64)

        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.transfer_kv_per_layer, 
            src_k, dst_k, src_v, dst_v, 
            src_indices, dst_indices, 
            self.item_size_bytes, self.block_quota, self.num_warps_per_block
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        total_elements = self.total_pool_blocks * self.elements_per_block
        
        src_k = torch.randn(total_elements, dtype=self.dtype, device=dev)
        src_v = torch.randn(total_elements, dtype=self.dtype, device=dev)
        dst_k = torch.zeros(total_elements, dtype=self.dtype, device=dev)
        dst_v = torch.zeros(total_elements, dtype=self.dtype, device=dev)

        all_indices_src = torch.randperm(self.total_pool_blocks, device=dev)
        src_indices = all_indices_src[:self.num_blocks_to_copy].to(torch.int64)
        
        all_indices_dst = torch.randperm(self.total_pool_blocks, device=dev)
        dst_indices = all_indices_dst[:self.num_blocks_to_copy].to(torch.int64)

        expected_dst_k = dst_k.clone()
        expected_dst_v = dst_v.clone()
        
        src_k_2d = src_k.view(self.total_pool_blocks, self.elements_per_block)
        src_v_2d = src_v.view(self.total_pool_blocks, self.elements_per_block)
        exp_dst_k_2d = expected_dst_k.view(self.total_pool_blocks, self.elements_per_block)
        exp_dst_v_2d = expected_dst_v.view(self.total_pool_blocks, self.elements_per_block)

        exp_dst_k_2d[dst_indices] = src_k_2d[src_indices]
        exp_dst_v_2d[dst_indices] = src_v_2d[src_indices]

        torch.ops.sgl_kernel.transfer_kv_per_layer(
            src_k, dst_k, src_v, dst_v, 
            src_indices, dst_indices, 
            self.item_size_bytes, self.block_quota, self.num_warps_per_block
        )

        pass_k, diff_k = self.check_diff(dst_k, expected_dst_k)
        pass_v, diff_v = self.check_diff(dst_v, expected_dst_v)

        is_passed = pass_k and pass_v
        max_diff = max(diff_k, diff_v)

        return is_passed, max_diff