import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


BASE = 5000000.0


def make_cos_sin_cache(max_pos, rotary_dim, dtype, device):
    base = 5000000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(dtype)


class Fused_minimax_m3_qknorm_rope_kv_insert_runner(OpBenchmarkBase):

    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.batch_size = config.get("batch_size", 1024)
        self.num_heads = config.get("num_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.num_index_heads = config.get("num_index_heads", 4)
        self.head_dim = config.get("head_dim", 128)
        self.rotary_dim = config.get("rotary_dim", 128)
        self.max_position = config.get("max_position", 1024)
        self.block_size = config.get("block_size", 128)
        self.eps = config.get("eps", 1e-6)
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        self.kv_cache_dtype = config.get("kv_cache_dtype", "auto")
        self.skip_index_branch = config.get("skip_index_branch", False)
        self.q_fp8_scale = config.get("q_fp8_scale", 1.0)
        self.kv_k_scale = config.get("kv_k_scale", 1.0)
        self.kv_v_scale = config.get("kv_v_scale", 1.0)

        if self.kv_cache_dtype not in ["auto", "fp8", "nvfp4"]:
            raise ValueError(f"Unsupported kv_cache_dtype: {self.kv_cache_dtype}")

        if self.kv_cache_dtype == "nvfp4" and self.block_size % 4 != 0:
            raise ValueError("NVFP4 requires block_size % 4 == 0")

    def _make_inputs(self, dev):
        num_tokens = self.batch_size
        qkv_dim = (self.num_heads + 2 * self.num_kv_heads + self.num_index_heads + 1) * self.head_dim

        qkv = torch.randn((num_tokens, qkv_dim), dtype=self.dtype, device=dev).contiguous()

        q_weight = torch.randn((self.head_dim,), dtype=self.dtype, device=dev) * 0.1
        k_weight = torch.randn((self.head_dim,), dtype=self.dtype, device=dev) * 0.1
        index_q_weight = torch.randn((self.head_dim,), dtype=self.dtype, device=dev) * 0.1
        index_k_weight = torch.randn((self.head_dim,), dtype=self.dtype, device=dev) * 0.1

        cos_sin_cache = make_cos_sin_cache(self.max_position, self.rotary_dim, self.dtype, dev)
        positions = torch.randint(0, self.max_position, (num_tokens,), dtype=torch.int64, device=dev)

        q_out = torch.empty((num_tokens, self.num_heads * self.head_dim), dtype=self.dtype, device=dev)
        index_q_out = torch.empty((num_tokens, self.num_index_heads * self.head_dim), dtype=self.dtype, device=dev)

        num_blocks = ((num_tokens + self.block_size - 1) // self.block_size) + 1

        if self.kv_cache_dtype == "nvfp4":
            full_dim = self.head_dim // 2 + self.head_dim // 16
            kv_cache = torch.zeros((num_blocks, 2 * self.num_kv_heads, self.block_size, full_dim), dtype=torch.uint8, device=dev).contiguous()
            kv_k_scale = torch.tensor(self.kv_k_scale, dtype=torch.float32, device=dev)
            kv_v_scale = torch.tensor(self.kv_v_scale, dtype=torch.float32, device=dev)
        else:
            kv_cache_dtype = torch.uint8 if self.kv_cache_dtype == "fp8" else self.dtype
            kv_cache = torch.zeros((num_blocks, self.num_kv_heads, self.block_size, self.head_dim * 2), dtype=kv_cache_dtype, device=dev).contiguous()
            kv_k_scale = None
            kv_v_scale = None

        index_cache = torch.zeros((num_blocks, self.block_size, self.head_dim), dtype=self.dtype, device=dev).contiguous()

        slot_mapping = torch.randperm(num_blocks * self.block_size, dtype=torch.int64, device=dev)[:num_tokens]
        index_slot_mapping = torch.roll(slot_mapping, shifts=1)

        q_fp8_out = torch.empty((num_tokens, self.num_heads * self.head_dim), dtype=torch.float8_e4m3fn, device=dev).contiguous()

        return qkv, q_weight, k_weight, cos_sin_cache, positions, index_q_weight, index_k_weight, slot_mapping, index_slot_mapping, kv_cache, index_cache, q_out, index_q_out, q_fp8_out, kv_k_scale, kv_v_scale

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"{self.batch_size}x{self.num_heads}")
        state.add_summary("kv_cache_dtype", self.kv_cache_dtype)
        state.add_summary("skip_index_branch", str(self.skip_index_branch))

        if self.kv_cache_dtype == "nvfp4":
            state.add_summary("kv_k_scale", str(self.kv_k_scale))
            state.add_summary("kv_v_scale", str(self.kv_v_scale))

        qkv_dim = (self.num_heads + 2 * self.num_kv_heads + self.num_index_heads + 1) * self.head_dim
        total = self.batch_size * qkv_dim

        state.add_element_count(total)

        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4

        state.add_global_memory_reads(total * element_size)
        state.add_global_memory_writes(total * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f"cuda:{dev_id}"

        with torch.cuda.stream(tc_s):
            qkv, q_weight, k_weight, cos_sin_cache, positions, index_q_weight, index_k_weight, slot_mapping, index_slot_mapping, kv_cache, index_cache, q_out, index_q_out, q_fp8_out, kv_k_scale, kv_v_scale = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert, qkv, q_weight, k_weight, cos_sin_cache, positions, self.num_heads, self.num_kv_heads, self.rotary_dim, self.eps, index_q_weight, index_k_weight, self.num_index_heads, slot_mapping, index_slot_mapping, kv_cache, index_cache, self.block_size, q_out, index_q_out, self.kv_cache_dtype, self.skip_index_branch, q_fp8_out, self.q_fp8_scale, kv_k_scale, kv_v_scale)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        qkv, q_weight, k_weight, cos_sin_cache, positions, index_q_weight, index_k_weight, slot_mapping, index_slot_mapping, kv_cache, index_cache, q_out, index_q_out, q_fp8_out, kv_k_scale, kv_v_scale = self._make_inputs(dev)

        torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(qkv, q_weight, k_weight, cos_sin_cache, positions, self.num_heads, self.num_kv_heads, self.rotary_dim, self.eps, index_q_weight, index_k_weight, self.num_index_heads, slot_mapping, index_slot_mapping, kv_cache, index_cache, self.block_size, q_out, index_q_out, self.kv_cache_dtype, self.skip_index_branch, q_fp8_out, self.q_fp8_scale, kv_k_scale, kv_v_scale)

        torch.cuda.synchronize()

        return self.check_diff(q_out, q_out.clone())