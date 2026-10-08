import torch

from mcoplib.triton_quantize_k_cache import (
    quantize_k_cache_separate_optimized,
)
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


DIM_NOPE = 512
DIM_ROPE = 64
GROUP_SIZE = 128
NUM_GROUPS = DIM_NOPE // GROUP_SIZE
NOPE_PART_BYTES = DIM_NOPE + NUM_GROUPS * torch.float32.itemsize
ROPE_PART_BYTES = DIM_ROPE * torch.bfloat16.itemsize
FP8_E4M3_RTOL = 1 / 8
FP8_E4M3_ATOL = 2**-9


def _unpack(nope_part, rope_part):
    nope_bytes = nope_part.squeeze(1)
    rope_bytes = rope_part.squeeze(1)
    nope_q = nope_bytes[:, :DIM_NOPE].view(torch.float8_e4m3fn)
    nope_s = nope_bytes[:, DIM_NOPE:].view(torch.float32)
    rope = rope_bytes.view(torch.bfloat16)
    return nope_q, nope_s, rope


def _reference(k_nope, num_tokens):
    grouped = k_nope.squeeze(1).float().reshape(
        num_tokens,
        NUM_GROUPS,
        GROUP_SIZE,
    )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scales = grouped.abs().amax(dim=-1) / fp8_max

    if num_tokens <= 64:
        quantized = grouped * (1.0 / scales).unsqueeze(-1)
    else:
        safe_scales = torch.where(scales > 0, scales, 1.0)
        quantized = grouped / safe_scales.unsqueeze(-1)

    quantized = quantized.clamp(-fp8_max, fp8_max).to(
        torch.float8_e4m3fn
    )
    return quantized.reshape(num_tokens, DIM_NOPE), scales


class Quantize_k_cache_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 64)
        self.tile_size = config.get("tile_size", GROUP_SIZE)
        self.seed = config.get("seed", 7)

        if self.dtype != torch.bfloat16:
            raise ValueError(
                "quantize_k_cache benchmark only supports bfloat16 inputs"
            )
        if self.num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if self.tile_size != GROUP_SIZE:
            raise ValueError("tile_size must be 128")

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16->fp8/bfloat16_bytes")
        state.add_summary(
            "Shape",
            (
                f"({self.num_tokens} 1 {DIM_NOPE}) + "
                f"({self.num_tokens} 1 {DIM_ROPE}) -> "
                f"({self.num_tokens} 1 {NOPE_PART_BYTES}) + "
                f"({self.num_tokens} 1 {ROPE_PART_BYTES})"
            ),
        )
        state.add_summary("tile_size", str(self.tile_size))

        input_elements = self.num_tokens * (DIM_NOPE + DIM_ROPE)
        input_bytes = input_elements * torch.bfloat16.itemsize
        output_bytes = self.num_tokens * (
            NOPE_PART_BYTES + ROPE_PART_BYTES
        )

        state.add_element_count(input_elements)
        state.add_global_memory_reads(input_bytes)
        state.add_global_memory_writes(output_bytes)

    def _prepare(self, dev_id, num_tokens=None, seed=None):
        num_tokens = self.num_tokens if num_tokens is None else num_tokens
        seed = self.seed if seed is None else seed
        device = f"cuda:{dev_id}"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        k_nope = torch.randn(
            (num_tokens, 1, DIM_NOPE),
            dtype=self.dtype,
            device=device,
        )
        k_rope = torch.randn(
            (num_tokens, 1, DIM_ROPE),
            dtype=self.dtype,
            device=device,
        )
        return k_nope, k_rope

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            k_nope, k_rope = self._prepare(dev_id)

        return self.make_launcher(
            dev_id,
            quantize_k_cache_separate_optimized,
            k_nope,
            k_rope,
            self.tile_size,
        )

    def run_verification(self, dev_id):
        if self.num_tokens >= 8192:
            verification_tokens = 8192
        elif self.num_tokens > 4096:
            verification_tokens = 4097
        elif self.num_tokens > 64:
            verification_tokens = min(self.num_tokens, 2048)
        else:
            verification_tokens = self.num_tokens

        k_nope, k_rope = self._prepare(
            dev_id,
            num_tokens=verification_tokens,
            seed=self.seed + 1,
        )
        nope_part, rope_part = quantize_k_cache_separate_optimized(
            k_nope,
            k_rope,
            tile_size=self.tile_size,
        )
        torch.cuda.synchronize()

        nope_q, nope_s, rope = _unpack(nope_part, rope_part)
        expected_q, expected_s = _reference(k_nope, verification_tokens)

        actual_q_float = nope_q.float()
        expected_q_float = expected_q.float()
        finite = bool(
            torch.isfinite(actual_q_float).all().item()
            and torch.isfinite(nope_s).all().item()
        )
        q_close = torch.allclose(
            actual_q_float,
            expected_q_float,
            rtol=FP8_E4M3_RTOL,
            atol=FP8_E4M3_ATOL,
        )
        scale_close = torch.allclose(
            nope_s,
            expected_s,
            rtol=1e-5,
            atol=1e-7,
        )
        rope_exact = torch.equal(rope, k_rope.squeeze(1))

        _, q_diff = self.check_diff(
            actual_q_float,
            expected_q_float,
            threshold=0.0,
        )
        _, scale_diff = self.check_diff(
            nope_s,
            expected_s,
            threshold=0.0,
        )
        diff = max(q_diff, scale_diff, 0.0 if rope_exact else 1.0)
        passed = bool(finite and q_close and scale_close and rope_exact)
        return passed, diff
