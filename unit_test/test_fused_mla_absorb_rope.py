import os
import time
import sys
import time
import torch
import torch.nn as nn
import unittest
import math
from typing import Any, Dict, List, Optional, Tuple, Union
# from mcoplib.fused_mla import fused_mla_normal_rotary_emb as fused_mla_normal_rotary_emb
# from sgl_kernel import fused_mla_absorb_rotary_emb
from measure_cuda import measure_cuda
# from sglang.srt.layers.rotary_embedding import get_rope_wrapper
# from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from torch.profiler import profile, record_function, ProfilerActivity
import argparse
import mcoplib.sgl_kernel as sgl
# from sgl_kernel import torch.ops.sgl_kernel.
def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

class CustomOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        return self.forward_native

def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )

# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: torch.dtype
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
        device: Optional[str] = "cuda",
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        self.device = device
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device)
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq_mask = inv_freq_mask.to(self.device)
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        print("Cache shape", cache.shape)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        # print("cos", cos.shape, cos.dtype, cos)
        # print("sin", sin.shape, sin.dtype, sin)
        # print("query_rot", query_rot.shape, query_rot)
        # print("rotate_fn(query_rot)", rotate_fn(query_rot).shape, rotate_fn(query_rot))
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}

def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if "rope_type" in rope_scaling:
        scaling_type = rope_scaling["rope_type"]
    elif "type" in rope_scaling:
        scaling_type = rope_scaling["type"]
    else:
        raise ValueError("Unknown RoPE scaling type")

    if scaling_type == "deepseek_yarn":
        scaling_factor = rope_scaling["factor"]
        original_max_position = rope_scaling["original_max_position_embeddings"]
        # assert max_position == original_max_position * scaling_factor
        extra_kwargs = {
            k: v
            for k, v in rope_scaling.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        }
        rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )
    else:
        raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb

def get_rope_wrapper(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    device: Optional[str] = None,
):
    if device != "cpu":
        return get_rope(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor,
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.variance_epsilon = eps
        self.weight = nn.Parameter(torch.ones(hidden_size).to(torch.bfloat16))

    def forward(
            self,
            x: torch.Tensor,
            residual= None,
        ) :
            orig_dtype = x.dtype
            x = x.to(torch.float32)
            if residual is not None:
                x = x + residual.to(torch.float32)
                residual = x.to(orig_dtype)

            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
            x = x.to(orig_dtype) * self.weight
            if residual is None:
                return x
            else:
                return x, residual

def fused_forward_absorb(
    q:torch.tensor, # [bs, 128, 192], dtype=bf16
    w_kc:torch.tensor, # [128, 128, 512], dtype=bf16
    latent_cache:torch.tensor, # [bs, 576], dtype=bf16
    cos_sin_cache:torch.tensor, # [max_position_embeddings, 64], dtype=bf16
    positions:torch.tensor, # [bs], dtype=int64
    norm_weight:torch.tensor, # [512], dtype=bf16
    q_input:torch.tensor, #[bs, 128, 576], dtype=bf16
    k_input:torch.tensor, #[bs, 1, 576], dtype=bf16
    v_input:torch.tensor, #[bs, 1, 512]
    q_len:int, #16
    num_local_heads:int, #128,
    kv_lora_rank:int, # 512
    qk_rope_head_dim:int, #64
    qk_nope_head_dim:int, #128
):
    out = torch.ops.sgl_kernel.fused_mla_absorb_rotary_emb(q, w_kc, latent_cache, cos_sin_cache, positions, norm_weight, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
    if out != 0:
        print("Failed to call FusedMLA.fused_forward_absorb")
    return q_input, k_input, v_input

def mla_absorb_rotary_emb(
    kv_a_layernorm,
    rotary_emb,
    q:torch.tensor, # [bs, 128, 192], dtype=bf16
    w_kc:torch.tensor, # [128, 128, 512], dtype=bf16
    latent_cache:torch.tensor, # [bs, 576], dtype=bf16
    positions:torch.tensor, # [bs], dtype=int64
    q_input:torch.tensor, #[bs, 128, 576], dtype=bf16
    k_input:torch.tensor, #[bs, 1, 576], dtype=bf16
    v_input:torch.tensor, #[bs, 1, 512]
    q_len:int, #16
    num_local_heads:int, #128,
    kv_lora_rank:int, # 512
    qk_rope_head_dim:int, #64
    qk_nope_head_dim:int, #128
):
    q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    # q_input = torch.zeros(
    #     q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim
    # )
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc)
    q_input[..., : kv_lora_rank] = q_nope_out.transpose(0, 1)
# Input: q_nope_out [128, bs, 512], q_input [16, 128, 576]
# Operation: 1. q_nope_out.transpose -> [bs, 128, 512]
#            2. value q_input[:, :, 0:512] = [bs, 128, 512]
# Output: 
    v_input = latent_cache[..., : kv_lora_rank] # v_input [bs, 512]
# Input: v_input = [bs, 512]
# Operation: v_input = latent_cache[:, 0:512]
    v_input = kv_a_layernorm(v_input.contiguous()).unsqueeze(1) #[bs, 1, 512]
# Input: v_input = [bs, 512]
# Operation: RMSNormal 
# Output: v_input = [bs, 1, 512]
    k_input = latent_cache.unsqueeze(1) # [bs, 1, 512+64]
# Input: latent_cache [bs, 576]
# Output: k_input = [bs, 1, 576]
    k_input[..., : kv_lora_rank] = v_input #
# Input: k_input [bs, 1, 576], v_input: 
# Operation: k_input[:, :, 0:512] = v_input

    k_pe = k_input[..., kv_lora_rank :] # [bs, 1, 64]
# Input: k_input [bs, 1, 576]
# Operation: k_pe [bs, 1, 64] = k_input[:, :, 512:576]

    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
# Input: positions = [bs], q_pe = [bs, 128, 64], k_pe = [bs, 1, 64]
# Operation: rotary_emb
# Output: q_pe = [bs, 128, 64], k_pe = [bs, 1, 64]

    q_input[..., kv_lora_rank :] = q_pe   # [bs, 128, 512+64]
    k_input[..., kv_lora_rank :] = k_pe   # [bs, 1, 512+64]
# q_input[bs, 128, 576], k_input[bs, 1, 576]
# q_input[:, :, 512:576] = q_pe[:, :, :], k_input[:, :, 512:576] = k_pe[:, :, :]
# Input: q_pe, k_pe
    return q_input, k_input, v_input

with_profile=False

def show_error(golden, v, tag="DIFF ERROR"):
    errors = torch.abs(golden - v)

    errors_max = torch.max(errors)
    errors_ave = torch.sum(errors) / errors.numel()

    max_idx_flat = torch.argmax(errors)
    max_idx = torch.unravel_index(max_idx_flat, errors.shape)

    golden_val = golden[max_idx]
    v_val = v[max_idx]

    print(f"{tag}: error_max={errors_max}, error_ave={errors_ave}, max_error_idx={max_idx}")
    print(f"golden[{max_idx}]={golden_val}, v[{max_idx}]={v_val}")

def print_profiler_summary(prof, max_key_len=50):
    events = prof.key_averages()
    events = sorted(events, key=lambda x: (x.device_time_total / x.count) if x.count > 0 else 0, reverse=True)

    print(f"{'Name':{max_key_len}} | {'CPU Time Avg (us)':>20} | {'CUDA Time Avg (us)':>20} | {'Count':>10}")

    total_cpu_time = 0.0
    total_cuda_time = 0.0

    for evt in events:
        if evt.count == 0:
            continue

        cpu_time_avg = evt.cpu_time_total / evt.count
        cuda_time_avg = evt.device_time_total / evt.count
        key_str = evt.key

        if len(key_str) > max_key_len:
            key_str = key_str[:max_key_len-3] + '...'

        print(f"{key_str.ljust(max_key_len)} | {cpu_time_avg:20.2f} | {cuda_time_avg:20.2f} | {evt.count:10}")

        total_cpu_time += cpu_time_avg
        total_cuda_time += cuda_time_avg

    print("=" * (max_key_len + 55))
    print(f"{'Total'.ljust(max_key_len)} | {total_cpu_time:20.2f} | {total_cuda_time:20.2f} |")

def save_tensor_to_bin(tensor, string_name):
    tensor_cpu = tensor.contiguous().cpu()
    with open(string_name, "wb") as f:
        f.write(tensor_cpu.view(torch.uint16).numpy().tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="profile", choices=["profile", "acc"])
    args = parser.parse_args()

    # q_len = 16 # bs
    num_local_heads = 128
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    hidden_size = 7168

    # set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


    kv_a_layernorm = RMSNorm(kv_lora_rank, eps=1e-06).cuda()
    max_position_embeddings = 163840
    rope_theta = 10000
    rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
    }
    rope_scaling["rope_type"] = "deepseek_yarn"
    # import pdb
    # pdb.set_trace()
    rotary_emb = get_rope_wrapper(
        qk_rope_head_dim,
        rotary_dim=qk_rope_head_dim,
        max_position=max_position_embeddings,
        base=rope_theta,
        rope_scaling=rope_scaling,
        is_neox_style=False,
        dtype=torch.float32,
        device="cuda",
    )

    cos_sin_cache = rotary_emb.cos_sin_cache

    # for q_len in [2, 3, 4, 6, 8, 12, 16, 20, 24,30, 32,36, 48, 64, 96]: 
    for q_len in [32]: 
        print(f"\n\n================= Profiling q_len={q_len} =================")
        q = (torch.rand(q_len, num_local_heads, qk_nope_head_dim+qk_rope_head_dim, dtype=torch.bfloat16).cuda() - 0.5)/10
        w_kc = (torch.rand(num_local_heads, num_local_heads, kv_lora_rank, dtype=torch.bfloat16).cuda() - 0.5) / 10

        shape = (q_len, kv_lora_rank+qk_rope_head_dim)
        strides = (2112, 1)
        storage_size = (shape[0] - 1) * strides[0] + (shape[1] - 1) * strides[1] + 1
        latent_cache_storage = (torch.rand(storage_size, dtype=torch.bfloat16).cuda() - 0.5) / 10
        latent_cache = torch.as_strided(latent_cache_storage, size=shape, stride=strides)
        latent_cache2_storage = latent_cache_storage.clone().detach()
        latent_cache2 = torch.as_strided(latent_cache2_storage, size=shape, stride=strides)
        q_input = torch.zeros(q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        k_input = torch.zeros(q_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        v_input = torch.zeros(q_len, 1, kv_lora_rank, dtype=torch.bfloat16).cuda()
        fused_q_input = torch.zeros(q_len, num_local_heads, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        fused_k_input = torch.zeros(q_len, 1, kv_lora_rank + qk_rope_head_dim, dtype=torch.bfloat16).cuda()
        fused_v_input = torch.zeros(q_len, 1, kv_lora_rank, dtype=torch.bfloat16).cuda()
        positions = torch.arange(1, q_len + 1, dtype=torch.int64).cuda()
        print("w_kc stride:", w_kc.stride())
        w_kc_1 = w_kc.clone().detach()
        print("w_kc1.stride:", w_kc_1.transpose(1, 2).contiguous().transpose(1, 2).stride())
        latent_cache3 = latent_cache.clone().detach()

        print("latent_cache stride:", latent_cache.stride())
        print("latent_cache2 stride:", latent_cache2.stride())
        print("latent_cache3 stride:", latent_cache3.contiguous().stride())

        if args.mode == "profile":
            print("Profiling mla_absorb_rotary_emb")
            with profile(
                activities=[ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=5, active=5, repeat=1),
                # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                # record_shapes=True,
                # profile_memory=True,
                # with_stack=True
            ) as prof:
                for step in range(12):
                    mla_absorb_rotary_emb(kv_a_layernorm, rotary_emb, q, w_kc, latent_cache, positions, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
                    prof.step()

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
            print_profiler_summary(prof)
            print("Profiling fused_forward_absorb")
            with profile(
                activities=[ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=5, active=5, repeat=1),
                # on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
                # record_shapes=True,
                # profile_memory=True,
                # with_stack=True
            ) as prof:
                for step in range(12):
                    fused_forward_absorb(q, w_kc, latent_cache2, cos_sin_cache, positions, kv_a_layernorm.weight, fused_q_input, fused_k_input, fused_v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
                    prof.step()

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
            print_profiler_summary(prof)
        else:
            legacy_q_input, legacy_k_input, legacy_v_input = mla_absorb_rotary_emb(kv_a_layernorm, rotary_emb, q, w_kc, latent_cache, positions, q_input, k_input, v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
            # save_tensor_to_bin(q, "./data/q.bin")
            # save_tensor_to_bin(w_kc, "./data/w_kc.bin")
            # save_tensor_to_bin(latent_cache2, "./data/latent_cache2.bin")
            # save_tensor_to_bin(cos_sin_cache, "./data/cos_sin_cache.bin")
            # save_tensor_to_bin(positions, "./data/positions.bin")
            # save_tensor_to_bin(kv_a_layernorm.weight, "./data/weight.bin")
            fused_forward_absorb(q, w_kc, latent_cache2, cos_sin_cache, positions, kv_a_layernorm.weight, fused_q_input, fused_k_input, fused_v_input, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
            # save_tensor_to_bin(fused_q_input, "./data/fused_q_input.bin")
            # save_tensor_to_bin(fused_k_input, "./data/fused_k_input.bin")
            # save_tensor_to_bin(fused_v_input, "./data/fused_v_input.bin")
            print(f"{q_len=}, {num_local_heads=}, {kv_lora_rank=}, {qk_rope_head_dim=}, {qk_nope_head_dim=}, {latent_cache2.stride(0)=}")
            show_error(legacy_q_input, fused_q_input, "DIFF ERROR OF Q_INPUT")
            show_error(legacy_k_input, fused_k_input, "DIFF ERROR OF K_INPUT")
            show_error(legacy_v_input, fused_v_input, "DIFF ERROR OF V_INPUT")