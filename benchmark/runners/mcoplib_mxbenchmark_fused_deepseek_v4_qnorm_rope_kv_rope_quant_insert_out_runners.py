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
QUANT_BLOCK = 64
HEAD_BYTES = NOPE_DIM + ROPE_DIM * 2 + 8


def make_cos_sin_cache(max_pos: int, rope_dim: int, dtype, device):
    base = 10000.0

    inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim))

    t = torch.arange(max_pos, dtype=torch.float32, device=device)

    freqs = torch.einsum("i,j->ij", t, inv_freq)

    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def apply_rope_gptj_last_k(x: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    rope_dim = cos_sin_cache.shape[-1]
    half = rope_dim // 2
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim

    cs = cos_sin_cache[positions.long()].to(torch.float32)
    cos = cs[..., :half]
    sin = cs[..., half:]

    rope = x[..., nope_dim:].float()
    shape = rope.shape
    rope = rope.reshape(*shape[:-1], half, 2)
    even = rope[..., 0]
    odd = rope[..., 1]

    for _ in range(rope.ndim - 3):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    new_even = torch.addcmul(-odd * sin, even, cos)
    new_odd = torch.addcmul(odd * cos, even, sin)

    rope_rotated = torch.stack((new_even, new_odd), dim=-1).reshape(shape)

    out = x.clone().float()
    out[..., nope_dim:] = rope_rotated

    return out.to(x.dtype)


def rmsnorm_no_weight(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(variance + eps)


class Fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out_runner(OpBenchmarkBase):

    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.num_tokens = config.get("num_tokens", 2048)
        self.n_heads = config.get("n_heads", 16)
        self.padded_heads = config.get("padded_heads", 32)
        self.head_dim = config.get("head_dim", HEAD_DIM)
        self.rope_dim = config.get("rope_dim", ROPE_DIM)
        self.nope_dim = config.get("nope_dim", NOPE_DIM)
        self.head_bytes = config.get("head_bytes", HEAD_BYTES)
        self.eps = config.get("eps", 1e-6)
        self.max_pos = config.get("max_pos", 4096)
        self.num_blocks = config.get("num_blocks", 2)
        self.block_size = config.get("block_size", 16)

        if self.head_dim != HEAD_DIM:
            raise ValueError(f"head_dim must be {HEAD_DIM}, got {self.head_dim}")

        if self.rope_dim != ROPE_DIM:
            raise ValueError(f"rope_dim must be {ROPE_DIM}, got {self.rope_dim}")

        if self.nope_dim != NOPE_DIM:
            raise ValueError(f"nope_dim must be {NOPE_DIM}, got {self.nope_dim}")

        if self.head_bytes != HEAD_BYTES:
            raise ValueError(f"head_bytes must be {HEAD_BYTES}, got {self.head_bytes}")

        if self.block_size != 16:
            raise ValueError(f"block_size must be 16, got {self.block_size}")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"{self.num_tokens}x{self.n_heads}x{self.head_dim}")

        q_elements = self.num_tokens * self.n_heads * self.head_dim
        q_out_elements = self.num_tokens * self.padded_heads * self.head_dim
        kv_elements = self.num_tokens * self.head_dim
        position_elements = self.num_tokens
        slot_elements = self.num_tokens
        cos_sin_elements = self.max_pos * self.rope_dim
        k_cache_elements = self.num_blocks * self.block_size * self.head_bytes

        total = q_elements + q_out_elements

        state.add_element_count(total)

        element_size = 2 if self.dtype == torch.bfloat16 else 4

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
        torch.manual_seed(0)

        q = torch.randn((self.num_tokens, self.n_heads, self.head_dim), dtype=self.dtype, device=dev).contiguous()

        kv = torch.zeros((self.num_tokens, self.head_dim), dtype=self.dtype, device=dev).contiguous()

        positions = torch.arange(self.num_tokens, dtype=torch.int64, device=dev)

        cos_sin_cache = make_cos_sin_cache(self.max_pos, self.rope_dim, torch.float32, dev)

        k_cache = torch.zeros((self.num_blocks, self.block_size * self.head_bytes), dtype=torch.uint8, device=dev).contiguous()

        slot_mapping = torch.full((self.num_tokens,), -1, dtype=torch.int64, device=dev)

        q_out = torch.empty((self.num_tokens, self.padded_heads, self.head_dim), dtype=self.dtype, device=dev).contiguous()

        return q, kv, q_out, k_cache, slot_mapping, positions, cos_sin_cache

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = f"cuda:{dev_id}"
        q, kv, q_out, k_cache, slot_mapping, positions, cos_sin_cache = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out, q, kv, q_out, k_cache, slot_mapping, positions, cos_sin_cache, self.padded_heads, self.eps, self.block_size)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        q, kv, q_out, k_cache, slot_mapping, positions, cos_sin_cache = self._make_inputs(dev)

        q_ref = rmsnorm_no_weight(q, self.eps)
        q_ref = apply_rope_gptj_last_k(q_ref, positions, cos_sin_cache).to(self.dtype)

        torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(q, kv, q_out, k_cache, slot_mapping, positions, cos_sin_cache, self.padded_heads, self.eps, self.block_size)

        torch.cuda.synchronize(dev)

        valid_out = q_out[:, :self.n_heads]
        pad_out = q_out[:, self.n_heads:self.padded_heads]

        diff = (valid_out.float() - q_ref.float()).abs()
        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.mean().item()
        mismatch = ((valid_out.float() - q_ref.float()).abs() > 1e-2).sum().item()
        total = valid_out.numel()

        padded_max_abs = pad_out.abs().max().item()

        passed = (
            torch.allclose(valid_out, q_ref, rtol=1e-2, atol=1e-2)
            and padded_max_abs == 0.0
            and q_out.shape == (self.num_tokens, self.padded_heads, self.head_dim)
            and k_cache.numel() == self.num_blocks * self.block_size * self.head_bytes
        )

        return passed, max_abs_diff