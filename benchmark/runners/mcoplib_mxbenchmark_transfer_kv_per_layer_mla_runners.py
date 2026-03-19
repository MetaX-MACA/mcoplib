import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Transfer_kv_per_layer_mla_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 8192)
        self.head_num = config.get("head_num", 32)
        self.head_dim = config.get("head_dim", 128)
        self.block_quota = config.get("block_quota", 256)
        self.num_warps_per_block = config.get("num_warps_per_block", 4)
        
        self.elements_per_item = self.head_num * self.head_dim
        self.element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        # 修复：底层 C++ 要求的 item_size 是 Byte Size
        self.item_byte_size = self.elements_per_item * self.element_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(1L, {self.num_tokens}T)")
        total_elements = self.num_tokens * self.elements_per_item
        state.add_element_count(total_elements)
        state.add_global_memory_reads(total_elements * self.element_size)
        state.add_global_memory_writes(total_elements * self.element_size)

    def _create_tensors(self, dev):
        src = torch.randn((self.num_tokens, self.elements_per_item), dtype=self.dtype, device=dev)
        dst = torch.zeros_like(src)
        
        src_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        dst_indices = torch.arange(self.num_tokens, dtype=torch.long, device=dev)
        
        return src, dst, src_indices, dst_indices

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f'cuda:{dev_id}'
        self.tensors = self._create_tensors(dev)
        src, dst, src_indices, dst_indices = self.tensors
        
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.transfer_kv_per_layer_mla,
            src, dst, src_indices, dst_indices,
            self.item_byte_size, self.block_quota, self.num_warps_per_block
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        src, dst, src_indices, dst_indices = self._create_tensors(dev)
        
        torch.ops.sgl_kernel.transfer_kv_per_layer_mla(
            src, dst, src_indices, dst_indices,
            self.item_byte_size, self.block_quota, self.num_warps_per_block
        )
        
        return self.check_diff(dst, src)