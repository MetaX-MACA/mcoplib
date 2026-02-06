import torch
import mcoplib._C
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Reshape_and_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
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
        input_elements = self.num_tokens * self.num_heads * self.head_size
        total_elements = input_elements * 2
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        read_bytes = total_elements * element_size
        read_bytes += self.num_tokens * 8
        write_bytes = total_elements * element_size
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = self._generate_inputs(dev_id)
        op_func = torch.ops._C_cache_ops.reshape_and_cache
        return self.make_launcher(dev_id, op_func,
                                  key, value, key_cache, value_cache,
                                  slot_mapping, self.kv_cache_dtype, k_scale, v_scale)

    def run_verification(self, dev_id):
        key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale = self._generate_inputs(dev_id)
        torch.ops._C_cache_ops.reshape_and_cache(
            key, value, key_cache, value_cache,
            slot_mapping, self.kv_cache_dtype, k_scale, v_scale
        )
        vc_permuted = value_cache.permute(0, 3, 1, 2)
        vc_flat = vc_permuted.flatten(0, 1)
        value_recon = vc_flat[slot_mapping]
        kc_permuted = key_cache.permute(0, 3, 1, 2, 4)
        kc_flat = kc_permuted.flatten(0, 1)
        kc_gathered = kc_flat[slot_mapping]
        key_recon = kc_gathered.flatten(-2, -1)
        passed_k, diff_k = self.check_diff(key_recon, key.to(key_recon.dtype))
        passed_v, diff_v = self.check_diff(value_recon, value.to(value_recon.dtype))
        total_diff = max(diff_k, diff_v)
        return (passed_k and passed_v), total_diff

    def _generate_inputs(self, dev_id):
        dev = f'cuda:{dev_id}'
        input_shape = (self.num_tokens, self.num_heads, self.head_size)
        key = torch.randn(input_shape, dtype=self.dtype, device=dev)
        value = torch.randn(input_shape, dtype=self.dtype, device=dev)
        kc_shape = (self.num_blocks, self.num_heads, self.head_size // self.x, self.block_size, self.x)
        key_cache = torch.zeros(kc_shape, dtype=self.dtype, device=dev)
        vc_shape = (self.num_blocks, self.num_heads, self.head_size, self.block_size)
        value_cache = torch.zeros(vc_shape, dtype=self.dtype, device=dev)
        total_slots = self.num_blocks * self.block_size
        if total_slots < self.num_tokens:
            raise ValueError(f"Total slots ({total_slots}) < num_tokens ({self.num_tokens})")
        slot_mapping = torch.randperm(total_slots, dtype=torch.long, device=dev)[:self.num_tokens]
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=dev)
        return key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale