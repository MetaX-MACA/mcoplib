# SPDX-License-Identifier: Apache-2.0

import torch

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
CACHE_TOKEN_BYTES = 576


def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype).contiguous()


def apply_rope_gptj_last_k(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    position_cpu = positions.detach().cpu().to(torch.float32).reshape(-1, 1)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)).reshape(1, -1)

    freqs = position_cpu * inv_freq

    cs = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(device=x.device, dtype=torch.float32)

    cos = cs[..., :half]
    sin = cs[..., half:]

    rope = x[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)

    even = rope[..., 0]
    odd = rope[..., 1]

    while cos.dim() < even.dim():
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    new_even = even * cos - odd * sin
    new_odd = even * sin + odd * cos

    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone().float()
    out[..., nope_dim:] = rope_rotated

    return out.to(x.dtype)


def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps)


class Fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert_runner(OpBenchmarkBase):

    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.num_tokens = config.get("num_tokens", 2048)
        self.n_heads = config.get("n_heads", 16)
        self.padded_heads = config.get("padded_heads", 32)
        self.head_dim = config.get("head_dim", HEAD_DIM)
        self.rope_dim = config.get("rope_dim", ROPE_DIM)
        self.nope_dim = config.get("nope_dim", NOPE_DIM)
        self.eps = config.get("eps", 1e-6)
        self.max_pos = config.get("max_pos", 4096)
        self.num_blocks = config.get("num_blocks", 128)
        self.block_size = config.get("block_size", 16)

        dtype_name = config.get("dtype", "bfloat16")
        if isinstance(dtype_name, list):
            dtype_name = dtype_name[0]

        if dtype_name == "bfloat16":
            self.dtype = torch.bfloat16
        elif dtype_name == "float16":
            self.dtype = torch.float16
        elif dtype_name == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"unsupported dtype: {dtype_name}")

        if self.head_dim != HEAD_DIM:
            raise ValueError(f"head_dim must be {HEAD_DIM}, got {self.head_dim}")

        if self.rope_dim != ROPE_DIM:
            raise ValueError(f"rope_dim must be {ROPE_DIM}, got {self.rope_dim}")

        if self.nope_dim != NOPE_DIM:
            raise ValueError(f"nope_dim must be {NOPE_DIM}, got {self.nope_dim}")

        if self.block_size != 16:
            raise ValueError(f"block_size must be 16, got {self.block_size}")

        if self.padded_heads not in (8, 16, 32, 64, 128):
            raise ValueError(f"unsupported padded_heads={self.padded_heads}")

        if self.padded_heads < self.n_heads:
            raise ValueError(f"padded_heads must be >= n_heads, got {self.padded_heads} < {self.n_heads}")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"{self.num_tokens}x{self.n_heads}x{self.head_dim}")
        state.add_summary("PaddedHeads", str(self.padded_heads))

        q_elements = self.num_tokens * self.n_heads * self.head_dim
        kv_elements = self.num_tokens * self.head_dim
        q_out_elements = self.num_tokens * self.padded_heads * self.head_dim
        position_elements = self.num_tokens
        slot_elements = self.num_tokens
        cos_sin_elements = self.max_pos * self.rope_dim
        k_cache_elements = self.num_blocks * self.block_size * CACHE_TOKEN_BYTES

        state.add_element_count(q_elements + q_out_elements)

        element_size = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4

        q_bytes = q_elements * element_size
        kv_bytes = kv_elements * element_size
        position_bytes = position_elements * 8
        slot_bytes = slot_elements * 8
        cos_sin_bytes = cos_sin_elements * 4
        k_cache_bytes = k_cache_elements
        q_out_bytes = q_out_elements * element_size

        state.add_global_memory_reads(q_bytes + kv_bytes + position_bytes + slot_bytes + cos_sin_bytes + k_cache_bytes)
        state.add_global_memory_writes(q_out_bytes + k_cache_bytes)

    def _make_inputs(self, dev):
        torch.manual_seed(12000)

        q = torch.randn((self.num_tokens, self.n_heads, self.head_dim), dtype=self.dtype, device=dev).contiguous()

        kv = torch.randn((self.num_tokens, self.head_dim), dtype=self.dtype, device=dev).contiguous()

        positions = torch.arange(self.num_tokens, dtype=torch.int64, device=dev).contiguous()

        cos_sin_cache = make_cos_sin_cache(max(self.max_pos, self.num_tokens + 16), self.rope_dim, torch.float32, dev)

        block_bytes = self.block_size * CACHE_TOKEN_BYTES

        k_cache = torch.full((self.num_blocks, block_bytes), 0x7F, dtype=torch.uint8, device=dev).contiguous()

        slot_mapping = torch.arange(self.num_tokens, dtype=torch.int64, device=dev).contiguous()

        return q, kv, k_cache, slot_mapping, positions, cos_sin_cache

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f"cuda:{dev_id}"

        q, kv, k_cache, slot_mapping, positions, cos_sin_cache = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert, q, kv, k_cache, slot_mapping, positions, cos_sin_cache, self.padded_heads, self.eps, self.block_size)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        q, kv, k_cache, slot_mapping, positions, cos_sin_cache = self._make_inputs(dev)

        q_ref_input = q.clone()

        q_ref = rmsnorm_no_weight(q_ref_input, self.eps)

        q_ref = apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache).to(self.dtype)

        q_out = torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_int8_insert(q, kv, k_cache, slot_mapping, positions, cos_sin_cache, self.padded_heads, self.eps, self.block_size)

        torch.cuda.synchronize(dev)

        valid_out = q_out[:, :self.n_heads]
        pad_out = q_out[:, self.n_heads:self.padded_heads]

        diff = (valid_out.float() - q_ref.float()).abs()

        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.mean().item()
        mismatch = ((valid_out.float() - q_ref.float()).abs() > 1e-2).sum().item()
        total = valid_out.numel()

        padded_max_abs = pad_out.abs().max().item() if pad_out.numel() > 0 else 0.0

        passed = (
            torch.allclose(valid_out, q_ref, rtol=4e-2, atol=4e-2)
            and padded_max_abs == 0.0
            and q_out.shape == (self.num_tokens, self.padded_heads, self.head_dim)
            and q_out.dtype == self.dtype
            and q_out.is_contiguous()
            and k_cache.numel() == self.num_blocks * self.block_size * CACHE_TOKEN_BYTES
        )

        return passed, max_abs_diff