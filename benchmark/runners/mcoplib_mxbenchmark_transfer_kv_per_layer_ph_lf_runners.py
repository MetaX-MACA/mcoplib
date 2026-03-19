import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Transfer_kv_per_layer_ph_lf_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 32768)
        self.num_pages = config.get("num_pages", 131072)
        self.item_elements = config.get("item_elements", 1024)
        self.num_layers = config.get("num_layers", 32)
        self.head_num = config.get("head_num", 8)
        self.page_size = config.get("page_size", 16)
        self.block_quota = config.get("block_quota", 128)
        self.num_warps_per_block = config.get("num_warps_per_block", 4)
        self.layer_id = 0

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(Tokens:{self.num_tokens} H:{self.head_num} PgSz:{self.page_size})")
        
        total_elements = self.num_tokens * self.item_elements * 2
        state.add_element_count(total_elements)
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        transfer_bytes = total_elements * element_size
        state.add_global_memory_reads(transfer_bytes)
        state.add_global_memory_writes(transfer_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        item_size_bytes = self.item_elements * element_size
        src_layout_dim_bytes = self.num_layers * item_size_bytes

        with torch.cuda.stream(tc_s):
            ph_size = self.num_pages * self.num_layers * self.item_elements
            src_k = torch.randn(ph_size, dtype=self.dtype, device=dev)
            src_v = torch.randn(ph_size, dtype=self.dtype, device=dev)
            
            dst_k = torch.zeros((self.num_pages, self.item_elements), dtype=self.dtype, device=dev)
            dst_v = torch.zeros((self.num_pages, self.item_elements), dtype=self.dtype, device=dev)
            
            src_indices = torch.randint(0, self.num_pages, (self.num_tokens,), dtype=torch.long, device=dev)
            dst_indices = torch.randperm(self.num_pages, dtype=torch.long, device=dev)[:self.num_tokens]

        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf, 
            src_k, dst_k, src_v, dst_v, src_indices, dst_indices, 
            self.layer_id, item_size_bytes, src_layout_dim_bytes, 
            self.page_size, self.head_num, self.block_quota, self.num_warps_per_block
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        item_size_bytes = self.item_elements * element_size
        src_layout_dim_bytes = self.num_layers * item_size_bytes
        
        page_dim_elements = self.num_layers * self.item_elements
        head_elements = self.item_elements // self.head_num

        ph_size = self.num_pages * self.num_layers * self.item_elements
        src_k = torch.randn(ph_size, dtype=self.dtype, device=dev)
        src_v = torch.randn(ph_size, dtype=self.dtype, device=dev)
        
        dst_k = torch.zeros((self.num_pages, self.item_elements), dtype=self.dtype, device=dev)
        dst_v = torch.zeros((self.num_pages, self.item_elements), dtype=self.dtype, device=dev)
        
        src_indices = torch.randint(0, self.num_pages, (self.num_tokens,), dtype=torch.long, device=dev)
        dst_indices = torch.randperm(self.num_pages, dtype=torch.long, device=dev)[:self.num_tokens]

        torch.ops.sgl_kernel.transfer_kv_per_layer_ph_lf(
            src_k, dst_k, src_v, dst_v, src_indices, dst_indices, 
            self.layer_id, item_size_bytes, src_layout_dim_bytes, 
            self.page_size, self.head_num, self.block_quota, self.num_warps_per_block
        )

        dst_k_ref = torch.zeros_like(dst_k)
        dst_v_ref = torch.zeros_like(dst_v)
        
        for i in range(self.num_tokens):
            s_idx = src_indices[i].item()
            d_idx = dst_indices[i].item()

            for head_id in range(self.head_num):
                s_offset = ((s_idx // self.page_size) * self.page_size * page_dim_elements) + \
                           ((page_dim_elements // self.head_num) * head_id * self.page_size) + \
                           ((s_idx % self.page_size) * (page_dim_elements // self.head_num)) + \
                           (self.layer_id * head_elements)
                
                d_offset = (d_idx * self.item_elements) + (head_id * head_elements)

                dst_k_ref.view(-1)[d_offset : d_offset + head_elements] = src_k.view(-1)[s_offset : s_offset + head_elements]
                dst_v_ref.view(-1)[d_offset : d_offset + head_elements] = src_v.view(-1)[s_offset : s_offset + head_elements]

        pass_k, diff_k = self.check_diff(dst_k, dst_k_ref)
        pass_v, diff_v = self.check_diff(dst_v, dst_v_ref)
        
        is_passed = pass_k and pass_v
        max_diff = max(diff_k, diff_v)
        return is_passed, max_diff