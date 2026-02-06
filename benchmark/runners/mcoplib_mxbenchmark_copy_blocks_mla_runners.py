import torch
import random
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

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
        self.hidden_dim = config.get("hidden_dim", 576)
        self.num_pairs = config.get("num_pairs", 128)
        self.capability = (f"L{self.num_layers}_B{self.block_size}_"
                           f"D{self.hidden_dim}_P{self.num_pairs}")
        self._force_sync = True

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", self.capability)
        dtype_str = str(self.dtype).replace("torch.", "")
        state.add_summary("dtype", dtype_str)
        state.add_element_count(1)
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        total_elements = self.num_layers * self.num_pairs * self.block_size * self.hidden_dim
        total_bytes = total_elements * element_size
        state.add_global_memory_reads(total_bytes)
        state.add_global_memory_writes(total_bytes)
        try:
            state.set_blocking_kernel_timeout(-1.0)
        except Exception:
            pass

    def _prepare_data(self, dev_id):
        dev = f'cuda:{dev_id}'
        cache_shape = (self.num_blocks, self.block_size, self.hidden_dim)
        kv_caches = [torch.randn(cache_shape, dtype=self.dtype, device=dev).contiguous() 
                     for _ in range(self.num_layers)]
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
            torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)
        return self.make_launcher(dev_id, op_launcher)

    def run_verification(self, dev_id):
        kv_caches, block_mapping = self._prepare_data(dev_id)
        ref_kv_caches = [k.clone() for k in kv_caches]
        torch.ops._C_cache_ops.copy_blocks_mla(kv_caches, block_mapping)
        mapping_cpu = block_mapping.cpu().numpy()
        for layer_idx in range(self.num_layers):
            for src_idx, dst_idx in mapping_cpu:
                ref_kv_caches[layer_idx][dst_idx].copy_(ref_kv_caches[layer_idx][src_idx])
        passed, diff = self.check_diff(kv_caches[0], ref_kv_caches[0])
        return passed, diff