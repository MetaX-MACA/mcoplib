import torch
import torch.nn.functional as F

from mcoplib.triton_fused_sigmoid_gating_delta_rule_update import (
    _select_execution_strategy,
    fused_sigmoid_gating_delta_rule_update,
)
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Fused_sigmoid_gating_delta_rule_update_runner(OpBenchmarkBase):
    """mxbench runner for the Kimi-K3 KDA recurrent update."""

    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 32)
        self.num_heads = config.get("num_heads", 12)
        self.num_steps = config.get("num_steps", 8)
        self.head_dim = config.get("head_dim", 128)
        self.mode = config.get("mode", "verify")
        self.lower_bound = config.get("lower_bound", -5.0)
        self.seed = config.get("seed", 42)
        self.atol = config.get("atol", 2.0e-2)
        self.rtol = config.get("rtol", 1.0e-2)

        if self.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("dtype must be float16 or bfloat16")
        if self.mode not in ("decode", "verify"):
            raise ValueError("mode must be 'decode' or 'verify'")
        if min(self.batch_size, self.num_heads, self.num_steps) <= 0:
            raise ValueError("batch_size, num_heads, and num_steps must be positive")
        if self.head_dim != 128:
            raise ValueError("the optimized KDA benchmark requires head_dim == 128")
        if self.mode == "decode":
            self.num_steps = 1

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            (
                f"mode={self.mode}, B={self.batch_size}, T={self.num_steps}, "
                f"H={self.num_heads}, K=V={self.head_dim}"
            ),
        )

        bsz, steps, heads, dim = (
            self.batch_size,
            self.num_steps,
            self.num_heads,
            self.head_dim,
        )
        token_heads = bsz * steps * heads
        input_bytes = torch.empty((), dtype=self.dtype).element_size()
        state_bytes = torch.empty((), dtype=torch.float32).element_size()

        # q, k, raw per-K gate, v, and raw beta.
        reads = token_heads * (3 * dim + dim + 1) * input_bytes
        reads += heads * (dim + 1) * state_bytes  # dt_bias and A_log
        reads += bsz * heads * dim * dim * state_bytes
        writes = token_heads * dim * input_bytes  # output
        if self.mode == "decode":
            writes += bsz * heads * dim * dim * state_bytes
        else:
            # ReplaySSM raw-v/raw-k/g/beta ring writes.
            writes += token_heads * (
                2 * dim * input_bytes + (dim + 1) * state_bytes
            )

        strategy = _select_execution_strategy(
            num_sequences=bsz,
            value_heads=heads,
            tokens=steps,
            key_dim=dim,
            value_dim=dim,
        )
        if strategy == "dual_opt":
            # gate_decay and beta are produced then consumed; verify also keeps
            # the un-exponentiated gate for the ReplaySSM ring.
            intermediate_elements = token_heads * (dim + 1)
            if self.mode == "verify":
                intermediate_elements += token_heads * dim
            reads += intermediate_elements * state_bytes
            writes += intermediate_elements * state_bytes

        state.add_element_count(token_heads * dim)
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _make_case(self, dev_id, *, batch_size=None, num_heads=None, num_steps=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        num_heads = self.num_heads if num_heads is None else num_heads
        num_steps = self.num_steps if num_steps is None else num_steps
        if self.mode == "decode":
            num_steps = 1

        device = f"cuda:{dev_id}"
        dim = self.head_dim
        total_tokens = batch_size * num_steps
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        q = torch.randn(
            1, total_tokens, num_heads, dim, device=device, dtype=self.dtype
        )
        k = torch.randn_like(q)
        v = 0.1 * torch.randn_like(q)
        if self.mode == "decode":
            a = torch.randn(
                batch_size, num_heads * dim, device=device, dtype=self.dtype
            )
        else:
            a = torch.randn(
                1,
                total_tokens,
                num_heads,
                dim,
                device=device,
                dtype=self.dtype,
            )
        a.mul_(0.5).add_(-1.0)
        b = 0.5 * torch.randn(
            1, total_tokens, num_heads, device=device, dtype=self.dtype
        )
        A_log = 0.2 * torch.randn(num_heads, device=device, dtype=torch.float32)
        dt_bias = 0.1 * torch.randn(
            num_heads * dim, device=device, dtype=torch.float32
        )
        state = 0.01 * torch.randn(
            batch_size,
            num_heads,
            dim,
            dim,
            device=device,
            dtype=torch.float32,
        )
        state_indices = torch.arange(
            batch_size, device=device, dtype=torch.int32
        )
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            num_steps,
            device=device,
            dtype=torch.int32,
        )

        kwargs = dict(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=state_indices,
            scale=dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            is_kda=True,
            lower_bound=self.lower_bound,
            disable_state_update=self.mode == "verify",
            cache_ring=self.mode == "verify",
        )
        if self.mode == "verify":
            kwargs.update(
                replayssm_rawv=torch.empty(
                    batch_size,
                    num_heads,
                    num_steps,
                    dim,
                    device=device,
                    dtype=self.dtype,
                ),
                replayssm_rawk=torch.empty(
                    batch_size,
                    num_heads,
                    num_steps,
                    dim,
                    device=device,
                    dtype=self.dtype,
                ),
                replayssm_g=torch.empty(
                    batch_size,
                    num_heads,
                    num_steps,
                    dim,
                    device=device,
                    dtype=torch.float32,
                ),
                replayssm_beta=torch.empty(
                    batch_size,
                    num_heads,
                    num_steps,
                    device=device,
                    dtype=torch.float32,
                ),
            )
        return kwargs

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            kwargs = self._make_case(dev_id)

            def op_closure():
                fused_sigmoid_gating_delta_rule_update(**kwargs)

        return self.make_launcher(dev_id, op_closure)

    def _reference(self, args, batch_size, num_heads, num_steps):
        """Compute a small FP32 reference without invoking the old kernel."""
        dim = self.head_dim
        q = args["q"][0].reshape(batch_size, num_steps, num_heads, dim).float()
        raw_k = (
            args["k"][0]
            .reshape(batch_size, num_steps, num_heads, dim)
            .float()
        )
        v = args["v"][0].reshape(batch_size, num_steps, num_heads, dim).float()
        raw_a = args["a"].reshape(batch_size, num_steps, num_heads, dim).float()
        raw_beta = (
            args["b"][0]
            .reshape(batch_size, num_steps, num_heads)
            .float()
        )
        A_log = args["A_log"].reshape(num_heads).float()
        dt_bias = args["dt_bias"].reshape(num_heads, dim).float()

        q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1.0e-6)
        k = raw_k * torch.rsqrt(
            torch.sum(raw_k * raw_k, dim=-1, keepdim=True) + 1.0e-6
        )
        gate_x = raw_a + dt_bias[None, None, :, :]
        if self.lower_bound is None:
            gate = -torch.exp(A_log)[None, None, :, None] * F.softplus(
                gate_x,
                beta=args["softplus_beta"],
                threshold=args["softplus_threshold"],
            )
        else:
            gate = self.lower_bound * torch.sigmoid(
                torch.exp(A_log)[None, None, :, None] * gate_x
            )
        beta = torch.sigmoid(raw_beta)

        state_before = args["initial_state_source"].clone()
        state = state_before.index_select(
            0, args["initial_state_indices"].long()
        )
        outputs = []
        for step in range(num_steps):
            state = state * torch.exp(gate[:, step]).unsqueeze(-2)
            delta = v[:, step] - torch.sum(
                state * k[:, step].unsqueeze(-2), dim=-1
            )
            delta = delta * beta[:, step].unsqueeze(-1)
            state = state + delta.unsqueeze(-1) * k[:, step].unsqueeze(-2)
            outputs.append(
                torch.sum(
                    state
                    * (q[:, step] * args["scale"]).unsqueeze(-2),
                    dim=-1,
                )
            )

        output = (
            torch.stack(outputs, dim=1)
            .reshape(1, batch_size * num_steps, num_heads, dim)
            .to(args["q"].dtype)
        )
        expected_state = state_before if self.mode == "verify" else state
        expected_rings = {}
        if self.mode == "verify":
            expected_rings = {
                "replayssm_rawv": v.transpose(1, 2).to(args["v"].dtype),
                "replayssm_rawk": raw_k.transpose(1, 2).to(args["k"].dtype),
                "replayssm_g": gate.transpose(1, 2),
                "replayssm_beta": beta.transpose(1, 2),
            }
        return output, expected_state, expected_rings

    @torch.inference_mode()
    def run_verification(self, dev_id):
        # Cap verification memory while preserving the selected strategy.
        strategy = _select_execution_strategy(
            num_sequences=self.batch_size,
            value_heads=self.num_heads,
            tokens=self.num_steps,
            key_dim=self.head_dim,
            value_dim=self.head_dim,
        )
        if strategy == "original":
            verify_batch, verify_heads = 1, 6
        elif strategy == "single_opt":
            verify_batch, verify_heads = 8, 12
        else:
            verify_batch, verify_heads = 8, 24
        verify_steps = min(self.num_steps, 4)

        optimized_args = self._make_case(
            dev_id,
            batch_size=verify_batch,
            num_heads=verify_heads,
            num_steps=verify_steps,
        )
        expected, expected_state, expected_rings = self._reference(
            optimized_args, verify_batch, verify_heads, verify_steps
        )
        actual = fused_sigmoid_gating_delta_rule_update(**optimized_args)
        torch.cuda.synchronize(dev_id)

        finite = bool(torch.isfinite(actual).all().item())
        close = torch.allclose(
            actual.float(), expected.float(), atol=self.atol, rtol=self.rtol
        )
        max_diff = (actual.float() - expected.float()).abs().max().item()

        state_actual = optimized_args["initial_state_source"]
        close &= torch.allclose(
            state_actual, expected_state, atol=2.0e-3, rtol=1.0e-2
        )
        max_diff = max(
            max_diff, (state_actual - expected_state).abs().max().item()
        )
        for name, rhs in expected_rings.items():
            if name in optimized_args:
                lhs = optimized_args[name]
                close &= torch.allclose(lhs, rhs, atol=2.0e-3, rtol=1.0e-2)
                max_diff = max(
                    max_diff,
                    (lhs.float() - rhs.float()).abs().max().item(),
                )

        return bool(finite and close), max_diff
