# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError as e:
    raise RuntimeError(f"Failed to import mcoplib._C: {e}") from e


OP_NAME = "fused_gdn_decode_post_conv_mtp"

K = 128
V = 128
RATIO = 8


class Fused_gdn_decode_post_conv_mtp_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self.device_id = config.get("device_id", 0)
        self.num_tokens = config.get("num_tokens", 8)
        self.tp_size = config.get("tp_size", 16)
        self.state_width = config.get("state_width", 8)
        self.state_dtype = self._parse_dtype(config.get("state_dtype", "float32"))
        self.norm_dtype = self._parse_dtype(config.get("norm_dtype", "bfloat16"))
        self.dt_bias_dtype = self._parse_dtype(config.get("dt_bias_dtype", "float16"))
        self.seed = config.get("seed", 0)
        self.scale = config.get("scale", 1.0 / math.sqrt(K))
        self.norm_eps = config.get("norm_eps", 1.0e-6)
        self.num_requests = 1
        self.query_lengths = [self.num_tokens]

        self.H = self.tp_size
        self.HV = self.H * RATIO

        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {self.tp_size}")

        if self.num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {self.num_tokens}")

        if self.num_tokens > 8:
            raise ValueError(f"num_tokens must be <= 8 for this op, got {self.num_tokens}")

        if self.state_width <= 0 or self.state_width > 8:
            raise ValueError(f"state_width must be in [1, 8], got {self.state_width}")

        if self.state_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"state_dtype must be float32 or bfloat16, got {self.state_dtype}")

        if self.norm_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"norm_dtype must be float32 or bfloat16, got {self.norm_dtype}")

        if self.dt_bias_dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError(f"dt_bias_dtype must be float32, bfloat16 or float16, got {self.dt_bias_dtype}")

        if not hasattr(torch.ops._C, OP_NAME):
            raise RuntimeError(f"torch.ops._C.{OP_NAME} is not registered")

    @staticmethod
    def _parse_dtype(dtype):
        if isinstance(dtype, torch.dtype):
            return dtype

        dtype_map = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }

        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")

        return dtype_map[dtype]

    def _make_inputs(self, dev):
        torch.manual_seed(self.seed)

        num_slots = max(32, self.num_requests * max(max(self.query_lengths), self.state_width) + 8)
        mixed_qkv_dim = 2 * self.H * K + self.HV * V

        mixed_qkv = torch.randn((self.num_tokens, mixed_qkv_dim), dtype=torch.bfloat16, device=dev).contiguous()

        ba = torch.randn((self.num_tokens, 2 * self.HV), dtype=torch.bfloat16, device=dev).contiguous()
        b, a = ba.chunk(2, dim=-1)

        A_log = (0.5 * torch.randn((self.HV,), dtype=torch.float32, device=dev)).contiguous()

        dt_bias = (0.1 * torch.randn((self.HV,), dtype=self.dt_bias_dtype, device=dev)).contiguous()

        output_gate = torch.randn((self.num_tokens, self.HV, V), dtype=torch.bfloat16, device=dev).contiguous()

        norm_weight = torch.randn((V,), dtype=self.norm_dtype, device=dev).contiguous()

        state_ref = (0.01 * torch.randn((num_slots, self.HV, V, K), dtype=torch.float32, device=dev)).to(self.state_dtype)

        state_actual = state_ref.clone()

        state_indices = torch.zeros((self.num_requests, self.state_width), dtype=torch.int32, device=dev).contiguous()

        next_slot = 1

        for req_idx, req_len in enumerate(self.query_lengths):
            for token_idx in range(min(req_len, self.state_width)):
                state_indices[req_idx, token_idx] = next_slot
                next_slot += 1

        cu_seqlens = torch.tensor([0, self.num_tokens], dtype=torch.int32, device=dev)

        num_accepted_tokens = torch.full((self.num_requests,), min(self.num_tokens, self.state_width), dtype=torch.int32, device=dev)

        out = torch.zeros_like(output_gate)

        return mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, out

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", f"qkv:bf16 state:{self.state_dtype} norm:{self.norm_dtype} dt_bias:{self.dt_bias_dtype}")
        state.add_summary("Shape", f"({self.num_tokens} {self.H} {self.HV} {K} {V})")

        mixed_qkv_elements = self.num_tokens * (2 * self.H * K + self.HV * V)
        a_elements = self.num_tokens * self.HV
        b_elements = self.num_tokens * self.HV
        A_log_elements = self.HV
        dt_bias_elements = self.HV
        output_gate_elements = self.num_tokens * self.HV * V
        norm_weight_elements = V
        output_elements = self.num_tokens * self.HV * V
        state_elements_per_slot = self.HV * V * K

        input_bytes = mixed_qkv_elements * 2
        input_bytes += a_elements * 2
        input_bytes += b_elements * 2
        input_bytes += A_log_elements * 4
        input_bytes += dt_bias_elements * self._dtype_size(self.dt_bias_dtype)
        input_bytes += output_gate_elements * 2
        input_bytes += norm_weight_elements * self._dtype_size(self.norm_dtype)
        input_bytes += state_elements_per_slot * self._dtype_size(self.state_dtype)

        output_bytes = output_elements * 2
        output_bytes += self.num_tokens * state_elements_per_slot * self._dtype_size(self.state_dtype)

        state.add_element_count(output_elements)
        state.add_global_memory_reads(input_bytes)
        state.add_global_memory_writes(output_bytes)

    @staticmethod
    def _dtype_size(dtype):
        if dtype == torch.float32:
            return 4
        if dtype in (torch.float16, torch.bfloat16):
            return 2
        return torch.tensor([], dtype=dtype).element_size()

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, out = self._make_inputs(dev)

        return self.make_launcher(dev_id, torch.ops._C.fused_gdn_decode_post_conv_mtp, mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state_actual, output_gate, norm_weight, out, self.scale, self.norm_eps)

    def _rmsnorm_reference(self, x, weight, z=None):
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(variance + self.norm_eps)
        x_norm = x_norm * weight.float()

        if z is not None:
            x_norm = x_norm * F.silu(z.float())

        return x_norm.to(x.dtype)

    def _gdn_update_reference(self, A_log, a, b, q, k, v, initial_state, state_indices, cu_seqlens, num_accepted_tokens):
        q = q.float()
        k = k.float()
        v = v.float()
        a = a.float()
        b = b.float()
        A_log = A_log.float()
        dt_bias = self._cached_dt_bias.float()

        num_tokens = q.shape[0]
        num_heads = q.shape[1]
        num_kv_heads = v.shape[1]

        repeat_ratio = num_kv_heads // num_heads

        q_square = q.pow(2).sum(dim=-1, keepdim=True)
        k_square = k.pow(2).sum(dim=-1, keepdim=True)

        q = q * torch.rsqrt(q_square + 1.0e-6)
        k = k * torch.rsqrt(k_square + 1.0e-6)
        q = q * self.scale

        x = a + dt_bias.view(1, num_kv_heads)
        beta_sp = torch.where(x > 20.0, x, torch.log1p(torch.exp(x)))

        g = -torch.exp(A_log).view(1, num_kv_heads) * beta_sp
        decay = torch.exp(g)

        beta = 1.0 / (1.0 + torch.exp(-b))

        state_dtype = initial_state.dtype
        state = initial_state.float().clone()

        output = torch.zeros((num_tokens, num_kv_heads, V), dtype=torch.float32, device=q.device)

        num_requests = state_indices.shape[0]
        state_indices_width = state_indices.shape[1]

        for req_idx in range(num_requests):
            bos = int(cu_seqlens[req_idx].item())
            eos = int(cu_seqlens[req_idx + 1].item())
            req_tokens = eos - bos

            if req_tokens <= 0:
                continue

            accepted = int(num_accepted_tokens[req_idx].item())

            if accepted <= 0 or accepted > state_indices_width:
                output[bos:eos].zero_()
                continue

            source_slot = int(state_indices[req_idx, accepted - 1].item())

            if source_slot <= 0 or req_tokens > 8:
                output[bos:eos].zero_()
                continue

            for value_head in range(num_kv_heads):
                key_head = value_head // repeat_ratio
                h = state[source_slot, value_head].clone()

                for token_offset in range(req_tokens):
                    token = bos + token_offset

                    q_vec = q[token, key_head]
                    k_vec = k[token, key_head]
                    v_vec = v[token, value_head]

                    h = h * decay[token, value_head]

                    state_k = torch.mv(h, k_vec)
                    delta = (v_vec - state_k) * beta[token, value_head]

                    h = h + torch.outer(delta, k_vec)

                    output[token, value_head] = torch.mv(h, q_vec)

                    if token_offset < state_indices_width:
                        destination_slot = int(state_indices[req_idx, token_offset].item())

                        if destination_slot > 0:
                            if state_dtype == torch.bfloat16:
                                state[destination_slot, value_head].copy_(h.to(torch.bfloat16).float())
                            else:
                                state[destination_slot, value_head].copy_(h)

        return output, state.to(state_dtype)

    def _build_reference(self, mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight):
        self._cached_dt_bias = dt_bias

        query, key, value = torch.split(mixed_qkv, [self.H * K, self.H * K, self.HV * V], dim=-1)

        query = query.view(self.num_tokens, self.H, K)
        key = key.view(self.num_tokens, self.H, K)
        value = value.view(self.num_tokens, self.HV, V)

        raw_ref, state_out = self._gdn_update_reference(A_log, a, b, query, key, value, state_ref, state_indices, cu_seqlens, num_accepted_tokens)

        raw_ref = raw_ref.to(torch.bfloat16)

        expected = self._rmsnorm_reference(raw_ref, norm_weight, z=output_gate)

        return expected, state_out

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        mixed_qkv, a, b, A_log, dt_bias, state_ref, state_actual, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight, out = self._make_inputs(dev)

        self._cached_dt_bias = dt_bias

        expected, state_expected = self._build_reference(mixed_qkv, a, b, A_log, dt_bias, state_ref, state_indices, cu_seqlens, num_accepted_tokens, output_gate, norm_weight)

        result = torch.ops._C.fused_gdn_decode_post_conv_mtp(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, state_indices=state_indices, cu_seqlens=cu_seqlens, num_accepted_tokens=num_accepted_tokens, state=state_actual, output_gate=output_gate, norm_weight=norm_weight, out=out, scale=self.scale, norm_eps=self.norm_eps)

        if result is not None:
            return False, float("inf")

        torch.cuda.synchronize(dev)

        if out.shape != (self.num_tokens, self.HV, V):
            return False, float("inf")

        if out.dtype != torch.bfloat16:
            return False, float("inf")

        if not torch.isfinite(out).all():
            return False, float("inf")

        if not torch.isfinite(state_actual).all():
            return False, float("inf")

        output_error = (out.float() - expected.float()).norm()
        output_relative_l2 = output_error / expected.float().norm().clamp_min(1.0e-20)

        if output_relative_l2.item() >= 5e-4:
            return False, output_relative_l2.item()

        state_error = (state_actual.float() - state_expected.float()).norm()
        state_relative_l2 = state_error / state_expected.float().norm().clamp_min(1.0e-20)

        if self.state_dtype == torch.float32:
            state_atol = 5e-5
            state_rtol = 5e-5
        else:
            state_atol = 5e-3
            state_rtol = 5e-3

        if not torch.allclose(state_actual, state_expected, rtol=state_rtol, atol=state_atol):
            return False, max(output_relative_l2.item(), state_relative_l2.item())

        return True, max(output_relative_l2.item(), state_relative_l2.item())