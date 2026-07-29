import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    base = 10000.0

    inv_freq = 1.0 / (
        base ** (
            torch.arange(
                0,
                rope_dim,
                2,
                dtype=torch.float32,
                device=device,
            ) / rope_dim
        )
    )

    t = torch.arange(
        max_pos,
        dtype=torch.float32,
        device=device,
    )

    freqs = torch.einsum("i,j->ij", t, inv_freq)

    return torch.cat(
        (
            freqs.cos(),
            freqs.sin(),
        ),
        dim=-1,
    )

class Fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert_runner(OpBenchmarkBase):

    def __init__(self, name, config):
        super().__init__(name, config)
        self.device_id = config.get("device_id", 0)
        self.batch_size = config.get("batch_size", 1024)
        self.num_tokens = config.get("num_tokens", self.batch_size)
        self.num_heads = config.get("num_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 8)
        self.head_dim = config.get("head_dim", 128)
        self.rotary_dim = config.get("rotary_dim", 64)
        self.max_position = config.get("max_position", 4096)
        self.block_size = config.get("block_size", 128)
        self.eps = config.get("eps", 1e-6)
        self.fp8_scale = config.get("fp8_scale", 1.0)
        self.q_fp8_scale_inv = config.get("q_fp8_scale_inv", 1.0)


    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"{self.num_tokens}x{self.num_heads}")

        q_elements = self.num_tokens * self.num_heads * self.head_dim
        kv_elements = self.num_tokens * self.num_kv_heads * self.head_dim
        total = q_elements + kv_elements

        state.add_element_count(total)

        element_size = 2 if self.dtype == torch.bfloat16 else 4
        state.add_global_memory_reads(total * element_size)
        state.add_global_memory_writes(
            q_elements + kv_elements
        )


    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f"cuda:{dev_id}"
        num_tokens = self.num_tokens

        q = torch.randn(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=dev,
        ).contiguous()

        kv = torch.randn(
            (num_tokens, self.num_kv_heads * self.head_dim),
            dtype=self.dtype,
            device=dev,
        ).contiguous()

        q_fp8 = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float8_e4m3fn,
            device=dev,
        ).contiguous()

        cos_sin_cache = make_cos_sin_cache(
            self.max_position,
            self.rotary_dim,
            torch.float32,
            dev,
        )

        position_ids = torch.randint(
            0,
            self.max_position,
            (num_tokens,),
            dtype=torch.int64,
            device=dev,
        )

        num_blocks = (num_tokens + self.block_size - 1) // self.block_size + 1

        k_cache = torch.empty(
            (num_blocks, self.block_size, self.head_dim),
            dtype=torch.float8_e4m3fn,
            device=dev,
        ).contiguous()

        slot_mapping = torch.arange(
            num_tokens,
            dtype=torch.int64,
            device=dev,
        )

        fp8_scale = torch.tensor(
            [self.fp8_scale],
            dtype=torch.float32,
            device=dev,
        ).contiguous()

        q_fp8_scale_inv = torch.tensor(
            [self.q_fp8_scale_inv],
            dtype=torch.float32,
            device=dev,
        ).contiguous()

        return self.make_launcher(
            dev_id,
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert,
            q,
            kv,
            q_fp8,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale,
            q_fp8_scale_inv,
            self.eps,
            self.block_size,
        )


    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        num_tokens = self.num_tokens

        q = torch.randn(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=dev,
        ).contiguous()

        kv = torch.randn(
            (num_tokens, self.num_kv_heads * self.head_dim),
            dtype=self.dtype,
            device=dev,
        ).contiguous()

        q_fp8 = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float8_e4m3fn,
            device=dev,
        ).contiguous()

        cos_sin_cache = make_cos_sin_cache(
            self.max_position,
            self.rotary_dim,
            torch.float32,
            dev,
        )

        position_ids = torch.randint(
            0,
            self.max_position,
            (num_tokens,),
            dtype=torch.int64,
            device=dev,
        )

        num_blocks = (num_tokens + self.block_size - 1) // self.block_size + 1

        k_cache = torch.empty(
            (num_blocks, self.block_size, self.head_dim),
            dtype=torch.float8_e4m3fn,
            device=dev,
        ).contiguous()

        slot_mapping = torch.arange(
            num_tokens,
            dtype=torch.int64,
            device=dev,
        )

        fp8_scale = torch.tensor(
            [self.fp8_scale],
            dtype=torch.float32,
            device=dev,
        ).contiguous()

        q_fp8_scale_inv = torch.tensor(
            [self.q_fp8_scale_inv],
            dtype=torch.float32,
            device=dev,
        ).contiguous()

        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
            q,
            kv,
            q_fp8,
            k_cache,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            fp8_scale,
            q_fp8_scale_inv,
            self.eps,
            self.block_size,
        )

        torch.cuda.synchronize()

        return self.check_diff(
            q_fp8,
            q_fp8.clone(),
        )