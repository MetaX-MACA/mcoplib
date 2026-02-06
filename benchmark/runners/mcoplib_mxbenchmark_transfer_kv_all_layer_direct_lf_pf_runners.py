import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Transfer_kv_all_layer_direct_lf_pf_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_layers = config.get("num_layers", 32)
        self.page_size = config.get("page_size", 16)
        self.head_dim = config.get("head_dim", 128)
        self.num_heads = config.get("num_heads", 8)
        self.total_pages = config.get("total_pages", 4096)
        self.copy_pages = config.get("copy_pages", 128)
        self._force_sync = True
        if self.dtype == torch.float32:
             self.dtype = torch.float16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        dtype_str = str(self.dtype).replace("torch.", "")
        state.add_summary("dtype", dtype_str)
        state.add_summary("Shape", f"L{self.num_layers}_P{self.page_size}_H{self.num_heads}_D{self.head_dim}_CP{self.copy_pages}")
        state.add_element_count(1)
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        num_indices = self.copy_pages * self.page_size
        total_elements = num_indices * self.num_layers * self.num_heads * self.head_dim
        total_bytes = total_elements * element_size
        state.add_global_memory_reads(total_bytes)
        state.add_global_memory_writes(total_bytes)
        try:
            state.set_blocking_kernel_timeout(-1.0)
        except Exception:
            pass

    def _prepare_data(self, dev_id):
        dev = f'cuda:{dev_id}'
        flat_dim = self.total_pages * self.page_size
        src_layers = []
        for _ in range(self.num_layers):
            s = torch.randn(flat_dim, self.num_heads, self.head_dim, dtype=self.dtype, device=dev)
            src_layers.append(s)
        dst_fused = torch.zeros(
            self.total_pages, self.num_layers, self.page_size, self.num_heads, self.head_dim,
            dtype=self.dtype, device=dev
        )
        dst_ptrs = [dst_fused]
        perm = torch.randperm(self.total_pages, device=dev)
        src_page_ids = perm[:self.copy_pages]
        dst_page_ids = perm[self.copy_pages:2*self.copy_pages]
        page_offsets = torch.arange(self.page_size, device=dev).unsqueeze(0)
        src_indices = ((src_page_ids * self.page_size).unsqueeze(1) + page_offsets).flatten().to(torch.int64)
        dst_indices = ((dst_page_ids * self.page_size).unsqueeze(1) + page_offsets).flatten().to(torch.int64)
        return src_layers, dst_ptrs, src_indices, dst_indices, dst_page_ids

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "transfer_kv_all_layer_direct_lf_pf"):
             raise RuntimeError("Operator 'torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf' not found.")
        with torch.cuda.stream(tc_s):
            src_layers, dst_ptrs, src_indices, dst_indices, _ = self._prepare_data(dev_id)
        def op_launcher(*args):
            torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf(
                src_layers, dst_ptrs, src_indices, dst_indices, self.page_size)
        return self.make_launcher(dev_id, op_launcher)

    def run_verification(self, dev_id):
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(torch.ops.sgl_kernel, "transfer_kv_all_layer_direct_lf_pf"):
             return False, 1.0
        src_layers, dst_ptrs, src_indices, dst_indices, dst_page_ids = self._prepare_data(dev_id)
        torch.ops.sgl_kernel.transfer_kv_all_layer_direct_lf_pf(src_layers, dst_ptrs, src_indices, dst_indices, self.page_size)
        dst_fused = dst_ptrs[0]
        dst_val_block = dst_fused[dst_page_ids, 0, :, :, :]
        dst_val_flat = dst_val_block.reshape(-1, self.num_heads, self.head_dim)
        src_val_flat = src_layers[0][src_indices]
        return self.check_diff(dst_val_flat, src_val_flat)