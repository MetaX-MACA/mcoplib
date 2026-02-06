import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Reshape_and_cache_flash_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 128)
        self.num_heads = config.get("num_heads", 32)
        self.head_size = config.get("head_size", 128)
        self.block_size = config.get("block_size", 16)
        self.num_blocks = config.get("num_blocks", 1024)
        self.kv_cache_dtype = config.get("kv_cache_dtype", "auto")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens} {self.num_heads} {self.head_size})")
        input_elements = self.num_tokens * self.num_heads * self.head_size
        total_elements = input_elements * 2 
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        read_bytes = total_elements * element_size + self.num_tokens * 8
        write_bytes = total_elements * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = self._generate_inputs(dev_id)
        op_func = torch.ops._C_cache_ops.reshape_and_cache_flash
        return self.make_launcher(dev_id, op_func, 
                                  key, value, key_cache, value_cache, 
                                  slot_mapping, self.kv_cache_dtype, k_scale, v_scale)

    def run_verification(self, dev_id):
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = self._generate_inputs(dev_id)
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key, value, key_cache, value_cache, 
            slot_mapping, self.kv_cache_dtype, k_scale, v_scale
        )
        slot_mapping_long = slot_mapping.long()
        block_idxs = slot_mapping_long // self.block_size
        block_offsets = slot_mapping_long % self.block_size
        key_recon = key_cache[block_idxs, block_offsets]
        value_recon = value_cache[block_idxs, block_offsets]
        passed_k, diff_k = self.check_diff(key_recon.to(key.dtype), key)
        passed_v, diff_v = self.check_diff(value_recon.to(value.dtype), value)
        total_diff = max(diff_k, diff_v)
        return (passed_k and passed_v), total_diff

    def _generate_inputs(self, dev_id):
        dev = f'cuda:{dev_id}'
        input_shape = (self.num_tokens, self.num_heads, self.head_size)
        key = torch.randn(input_shape, dtype=self.dtype, device=dev)
        value = torch.randn(input_shape, dtype=self.dtype, device=dev)
        cache_shape = (self.num_blocks, self.block_size, self.num_heads, self.head_size)
        key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
        value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=dev)
        total_slots = self.num_blocks * self.block_size
        if total_slots < self.num_tokens:
            raise ValueError(f"Cache size ({total_slots}) is smaller than num_tokens ({self.num_tokens})")
        slot_mapping = torch.randperm(total_slots, dtype=torch.long, device=dev)[:self.num_tokens]
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        return key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale