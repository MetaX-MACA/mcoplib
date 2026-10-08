from itertools import accumulate

import torch
import torch.nn.functional as F

from mcoplib.triton_causal_conv1d_fwd import causal_conv1d_fwd
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase


class Causal_conv1d_fwd_runner(OpBenchmarkBase):
    """Benchmark the optimized width-4 continuous-batching implementation."""

    def __init__(self, name, config):
        super().__init__(name, config)
        self.tp = config.get("tp", 8)
        self.tokens = config.get("tokens", 2048)
        self.model_num_heads = config.get("model_num_heads", 96)
        self.head_dim = config.get("head_dim", 128)
        self.kernel_width = config.get("kernel_width", 4)
        self.request_token_cap = config.get("request_token_cap", 3072)
        self.num_cache_lines = config.get("num_cache_lines", 212)
        self.activation = config.get("activation", "silu")
        self.seed = config.get("seed", 1234)
        self.atol = config.get("atol", 2.0e-2)
        self.rtol = config.get("rtol", 2.0e-2)

        if self.dtype != torch.bfloat16:
            raise ValueError("the optimized workload requires dtype=bfloat16")
        if self.tp <= 0 or self.model_num_heads % self.tp != 0:
            raise ValueError("tp must be a positive divisor of model_num_heads")
        if self.tokens <= 0 or self.request_token_cap <= 0:
            raise ValueError("tokens and request_token_cap must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.kernel_width != 4:
            raise ValueError("causal_conv1d_fwd optimization requires kernel_width=4")
        if self.activation not in (None, "silu", "swish"):
            raise ValueError("activation must be null, 'silu', or 'swish'")

        self.dim = (self.model_num_heads // self.tp) * self.head_dim
        self.seq_lens = self._request_lengths(self.tokens)
        self.request_count = len(self.seq_lens)
        if self.num_cache_lines < self.request_count:
            raise ValueError("num_cache_lines must cover every packed request")

    def _request_lengths(self, tokens):
        full_requests, remainder = divmod(tokens, self.request_token_cap)
        lengths = [self.request_token_cap] * full_requests
        if remainder:
            lengths.append(remainder)
        return lengths

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary(
            "Shape",
            (
                f"TP={self.tp} tokens={self.tokens} requests={self.request_count} "
                f"dim={self.dim} width={self.kernel_width}"
            ),
        )

        data_size = torch.empty((), dtype=self.dtype).element_size()
        weight_size = torch.empty((), dtype=torch.float32).element_size()
        output_elements = self.dim * self.tokens
        state_elements = (
            self.request_count * self.dim * (self.kernel_width - 1)
        )

        # Useful tensor traffic: packed input and weights are read, while the
        # result and the selected cache lines are written. Metadata is int32.
        reads = output_elements * data_size
        reads += self.dim * self.kernel_width * weight_size
        reads += (2 * self.request_count + 1) * 4
        writes = (output_elements + state_elements) * data_size
        state.add_element_count(output_elements)
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _make_strided_x(self, device):
        # Match the transposed view of one region in a fused four-region QKV
        # buffer: shape (dim, tokens), stride (1, 4 * dim).
        fused = torch.empty(
            (self.tokens, 4 * self.dim),
            device=device,
            dtype=self.dtype,
        )
        fused.normal_(mean=0.0, std=0.5)
        return fused[:, : self.dim].transpose(0, 1)

    def _prepare(self, dev_id, seed=None):
        device = f"cuda:{dev_id}"
        seed = self.seed if seed is None else seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        x = self._make_strided_x(device)
        weight = 0.25 * torch.randn(
            (self.dim, self.kernel_width),
            device=device,
            dtype=torch.float32,
        )
        conv_states = torch.randn(
            (self.num_cache_lines, self.dim, self.kernel_width - 1),
            device=device,
            dtype=self.dtype,
        )
        boundaries = [0, *accumulate(self.seq_lens)]
        query_start_loc = torch.tensor(
            boundaries,
            device=device,
            dtype=torch.int32,
        )
        cache_indices = torch.arange(
            self.request_count,
            device=device,
            dtype=torch.int32,
        )
        return x, weight, conv_states, query_start_loc, cache_indices

    def _call(
        self,
        x,
        weight,
        conv_states,
        query_start_loc,
        cache_indices,
        *,
        validate_data=False,
    ):
        return causal_conv1d_fwd(
            x=x,
            weight=weight,
            bias=None,
            conv_states=conv_states,
            query_start_loc=query_start_loc,
            seq_lens_cpu=self.seq_lens,
            cache_indices=cache_indices,
            has_initial_state=None,
            activation=self.activation,
            validate_data=validate_data,
        )

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            args = self._prepare(dev_id)

            def op_closure():
                self._call(*args)

        return self.make_launcher(dev_id, op_closure)

    def _reference(self, x, weight, query_start_loc):
        outputs = []
        boundaries = query_start_loc.cpu().tolist()
        weight_fp32 = weight.float()
        for start, end in zip(boundaries, boundaries[1:]):
            sequence = x[:, start:end].float()
            output = torch.zeros_like(sequence)
            for tap in range(self.kernel_width):
                history = self.kernel_width - 1 - tap
                if sequence.shape[1] > history:
                    output[:, history:] += (
                        sequence[:, : sequence.shape[1] - history]
                        * weight_fp32[:, tap, None]
                    )
            if self.activation in ("silu", "swish"):
                output = F.silu(output)
            outputs.append(output)
        return torch.cat(outputs, dim=1).to(x.dtype)

    @torch.inference_mode()
    def run_verification(self, dev_id):
        x, weight, conv_states, query_start_loc, cache_indices = self._prepare(
            dev_id, seed=self.seed + 1
        )
        expected = self._reference(x, weight, query_start_loc)
        actual = self._call(
            x,
            weight,
            conv_states,
            query_start_loc,
            cache_indices,
            validate_data=True,
        )
        torch.cuda.synchronize(dev_id)

        finite = bool(torch.isfinite(actual).all().item())
        close = torch.allclose(
            actual.float(), expected.float(), atol=self.atol, rtol=self.rtol
        )
        max_diff = (actual.float() - expected.float()).abs().max().item()

        # With no initial state, each selected line must contain the final
        # width-1 raw inputs, left-padded with zeros for a short request.
        boundaries = query_start_loc.cpu().tolist()
        for request_id, (start, end) in enumerate(
            zip(boundaries, boundaries[1:])
        ):
            sequence = x[:, start:end]
            zero_history = torch.zeros(
                (self.dim, self.kernel_width - 1),
                device=x.device,
                dtype=x.dtype,
            )
            expected_state = torch.cat((zero_history, sequence), dim=1)[
                :, -(self.kernel_width - 1) :
            ]
            actual_state = conv_states[cache_indices[request_id]]
            state_close = torch.equal(actual_state, expected_state)
            close = bool(close and state_close)
            max_diff = max(
                max_diff,
                (actual_state.float() - expected_state.float()).abs().max().item(),
            )

        return bool(finite and close), max_diff
