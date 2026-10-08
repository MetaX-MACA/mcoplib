import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_kda_decode_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_heads = config.get("num_heads", 12)
        self.num_seqs = config.get("num_seqs", 1)
        self.head_dim = config.get("head_dim", 128)
        self.width = config.get("width", 4)
        self.lower_bound = config.get("lower_bound", -5.0)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"heads={self.num_heads},seqs={self.num_seqs},dim={self.head_dim}")

        dim = self.num_heads * self.head_dim
        elements = self.num_seqs * dim * 3
        bytes_size = elements * 2

        state.add_element_count(elements)
        state.add_global_memory_reads(bytes_size)
        state.add_global_memory_writes(bytes_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            D = self.head_dim
            H = self.num_heads
            dim = H * D
            W = self.width
            slots = self.num_seqs + 2
            x = torch.randn(self.num_seqs, 3 * dim, dtype=self.dtype, device=dev)
            weight = torch.randn(3, W, dim, dtype=torch.float32, device=dev)
            conv_state = torch.randn(slots, W - 1, 3 * dim, dtype=self.dtype, device=dev).transpose(1, 2)
            raw_g = torch.randn(1, self.num_seqs, H, D, dtype=self.dtype, device=dev)
            raw_beta = torch.randn(1, self.num_seqs, H, dtype=self.dtype, device=dev)
            A_log = torch.randn(H, dtype=torch.float32, device=dev)
            dt_bias = torch.randn(dim, dtype=torch.float32, device=dev)
            state_indices = torch.arange(self.num_seqs, 0, -1, dtype=torch.int32, device=dev)
            state = torch.randn(slots, H, D, D, dtype=torch.float32, device=dev)

            out = torch.empty(1, self.num_seqs, H, D, dtype=self.dtype, device=dev)

        return self.make_launcher(
            dev_id,
            torch.ops._C.fused_kda_decode,
            x,
            weight,
            None,
            conv_state,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            state_indices,
            state,
            out,
            self.lower_bound,
            None,
            None,
            1e-5,
        )

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        num_heads = 12
        num_seqs = 1
        D = 128
        W = 4
        dim = num_heads * D
        dtype = torch.bfloat16

        torch.manual_seed(42)

        x = torch.randn(num_seqs, 3 * dim, dtype=dtype, device=dev)

        weight = torch.randn(3, W, dim, dtype=torch.float32, device=dev)

        conv_state = torch.randn(num_seqs + 2, W - 1, 3 * dim, dtype=torch.bfloat16, device=dev).transpose(1, 2)

        raw_g = torch.randn(1, num_seqs, num_heads, D, dtype=dtype, device=dev)

        raw_beta = torch.randn(1, num_seqs, num_heads, dtype=dtype, device=dev)

        A_log = torch.randn(num_heads, dtype=torch.float32, device=dev)

        dt_bias = torch.randn(dim, dtype=torch.float32, device=dev)

        state_indices = torch.arange(num_seqs, dtype=torch.int32, device=dev)

        state = torch.randn(num_seqs + 2, num_heads, D, D, dtype=torch.float32, device=dev)

        out = torch.empty(1, num_seqs, num_heads, D, dtype=dtype, device=dev)

        torch.ops._C.fused_kda_decode(
            x=x,
            weight=weight,
            bias=None,
            conv_state=conv_state,
            raw_g=raw_g,
            raw_beta=raw_beta,
            A_log=A_log,
            dt_bias=dt_bias,
            state_indices=state_indices,
            state=state,
            out=out,
            lower_bound=None,
            output_gate=None,
            norm_weight=None,
            norm_eps=1e-5,
        )

        if not torch.isfinite(out).all():
            return False, float("inf")

        return True, 0.0