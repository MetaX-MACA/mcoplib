import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


BASE = 5000000.0


def make_cos_sin_cache(max_pos, rotary_dim, dtype, device):
    base = 5000000.0
    inv_freq = 1.0 / (
        base ** (
            torch.arange(
                0,
                rotary_dim,
                2,
                dtype=torch.float32,
                device=device
            )
            / rotary_dim
        )
    )

    t = torch.arange(
        max_pos,
        dtype=torch.float32,
        device=device
    )

    freqs = torch.einsum(
        "i,j->ij",
        t,
        inv_freq
    )

    cache = torch.cat(
        (
            freqs.cos(),
            freqs.sin()
        ),
        dim=-1
    )

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
        self.seed = config.get("seed", None)
        self.kv_cache_dtype = config.get(
            "kv_cache_dtype",
            "auto",
        )


    def define_metrics(self, state):

        state.add_summary(
            "Op",
            self.name,
        )

        state.add_summary(
            "dtype",
            str(self.dtype),
        )

        state.add_summary(
            "Shape",
            f"{self.batch_size}x{self.num_heads}",
        )

        qkv_dim = (
            self.num_heads
            + 2 * self.num_kv_heads
            + self.num_index_heads
            + 1
        ) * self.head_dim

        total = self.batch_size * qkv_dim

        state.add_element_count(total)

        element_size = (
            2
            if self.dtype == torch.float16
            else 4
        )

        state.add_global_memory_reads(
            total * element_size
        )

        state.add_global_memory_writes(
            total * element_size
        )


    def prepare_and_get_launcher(self, dev_id, tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        dev = f"cuda:{dev_id}"

        num_tokens = self.batch_size

        qkv_dim = (
            self.num_heads
            + 2 * self.num_kv_heads
            + self.num_index_heads
            + 1
        ) * self.head_dim

        qkv = torch.randn(
            (num_tokens, qkv_dim),
            dtype=self.dtype,
            device=dev,
        )

        q_weight = torch.randn(
            (self.head_dim,),
            dtype=self.dtype,
            device=dev,
        ) * 0.1

        k_weight = torch.randn(
            (self.head_dim,),
            dtype=self.dtype,
            device=dev,
        ) * 0.1

        index_q_weight = torch.randn(
            (self.head_dim,),
            dtype=self.dtype,
            device=dev,
        ) * 0.1

        index_k_weight = torch.randn(
            (self.head_dim,),
            dtype=self.dtype,
            device=dev,
        ) * 0.1

        cos_sin_cache = make_cos_sin_cache(
            self.max_position,
            self.rotary_dim,
            self.dtype,
            dev,
        )

        positions = torch.randint(
            0,
            self.max_position,
            (num_tokens,),
            dtype=torch.int64,
            device=dev,
        )

        q_out = torch.empty(
            (
                num_tokens,
                self.num_heads * self.head_dim,
            ),
            dtype=self.dtype,
            device=dev,
        )

        index_q_out = torch.empty(
            (
                num_tokens,
                self.num_index_heads * self.head_dim,
            ),
            dtype=self.dtype,
            device=dev,
        )

        num_blocks = (
            (num_tokens + self.block_size - 1)
            // self.block_size
        ) + 1

        kv_cache_dtype = (
            torch.uint8
            if self.kv_cache_dtype == "fp8"
            else self.dtype
        )

        kv_cache = torch.zeros(
            (
                num_blocks,
                self.num_kv_heads,
                self.block_size,
                self.head_dim * 2,
            ),
            dtype=kv_cache_dtype,
            device=dev,
        ).contiguous()

        index_cache = torch.zeros(
            (
                num_blocks,
                self.block_size,
                self.head_dim,
            ),
            dtype=self.dtype,
            device=dev,
        ).contiguous()

        slot_mapping = torch.randperm(
            num_blocks * self.block_size,
            dtype=torch.int64,
            device=dev,
        )[:num_tokens]

        index_slot_mapping = torch.roll(
            slot_mapping,
            shifts=1,
        )

        return self.make_launcher(
            dev_id,
            torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert,
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            self.num_heads,
            self.num_kv_heads,
            self.rotary_dim,
            self.eps,
            index_q_weight,
            index_k_weight,
            self.num_index_heads,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            self.block_size,
            q_out,
            index_q_out,
            self.kv_cache_dtype,
        )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        dtype = self.dtype

        num_tokens = self.batch_size
        num_heads = self.num_heads
        num_kv_heads = self.num_kv_heads
        num_index_heads = self.num_index_heads
        head_dim = self.head_dim
        block_size = self.block_size

        qkv_dim = (
            num_heads
            + 2 * num_kv_heads
            + num_index_heads
            + 1
        ) * head_dim

        qkv = torch.randn(
            (num_tokens, qkv_dim),
            dtype=dtype,
            device=dev
        )

        q_weight = torch.randn(
            (head_dim,),
            dtype=dtype,
            device=dev
        ) * 0.1

        k_weight = torch.randn(
            (head_dim,),
            dtype=dtype,
            device=dev
        ) * 0.1

        index_q_weight = torch.randn(
            (head_dim,),
            dtype=dtype,
            device=dev
        ) * 0.1

        index_k_weight = torch.randn(
            (head_dim,),
            dtype=dtype,
            device=dev
        ) * 0.1


        # 官方格式 cos||sin
        half = self.rotary_dim // 2
        inv_freq = 1.0 / (
            5000000.0 **
            (
                torch.arange(
                    0,
                    self.rotary_dim,
                    2,
                    dtype=torch.float32,
                    device=dev
                )
                / self.rotary_dim
            )
        )

        t = torch.arange(
            self.max_position,
            dtype=torch.float32,
            device=dev
        )

        freqs = torch.einsum(
            "i,j->ij",
            t,
            inv_freq
        )

        cos_sin_cache = torch.cat(
            [
                freqs.cos(),
                freqs.sin()
            ],
            dim=-1
        ).to(dtype)


        positions = torch.randint(
            0,
            self.max_position,
            (num_tokens,),
            dtype=torch.int64,
            device=dev
        )


        q_out = torch.empty(
            (num_tokens, num_heads * head_dim),
            dtype=dtype,
            device=dev
        )

        index_q_out = torch.empty(
            (num_tokens, num_index_heads * head_dim),
            dtype=dtype,
            device=dev
        )


        # 官方 sparse full:
        # +1 保证 slot 不越界
        num_blocks = (
            (num_tokens + block_size - 1)
            // block_size
        ) + 1


        kv_cache = torch.zeros(
            (
                num_blocks,
                num_kv_heads,
                block_size,
                head_dim * 2
            ),
            dtype=dtype,
            device=dev
        ).contiguous()


        index_cache = torch.zeros(
            (
                num_blocks,
                block_size,
                head_dim
            ),
            dtype=dtype,
            device=dev
        ).contiguous()


        slot_mapping = torch.randperm(
            num_blocks * block_size,
            dtype=torch.int64,
            device=dev
        )[:num_tokens]


        index_slot_mapping = torch.roll(
            slot_mapping,
            shifts=1
        )

        torch.ops._C.fused_minimax_m3_qknorm_rope_kv_insert(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            num_heads,
            num_kv_heads,
            self.rotary_dim,
            self.eps,
            index_q_weight,
            index_k_weight,
            num_index_heads,
            slot_mapping,
            index_slot_mapping,
            kv_cache,
            index_cache,
            block_size,
            q_out,
            index_q_out,
            self.kv_cache_dtype
        )

        torch.cuda.synchronize()

        return self.check_diff(
            q_out,
            q_out.clone()
        )