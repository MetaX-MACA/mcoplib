import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Swigluoai_and_mul_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 16)
        self.seq_len = config.get("seq_len", 128)
        self.hidden_dim = config.get("hidden_dim", 4096)
        self.alpha = config.get("alpha", 1.702)
        self.limit = config.get("limit", 7.0)
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(B{self.batch_size}_S{self.seq_len}_H{self.hidden_dim})")
        total_elements = self.batch_size * self.seq_len * self.hidden_dim
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        input_size = (self.batch_size * self.seq_len * 2 * self.hidden_dim) * element_size
        output_size = total_elements * element_size
        state.add_global_memory_reads(input_size)
        state.add_global_memory_writes(output_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_dim = 2 * self.hidden_dim
            input_tensor = torch.randn(self.batch_size, self.seq_len, input_dim, dtype=self.dtype, device=dev)
            output_tensor = torch.empty(self.batch_size, self.seq_len, self.hidden_dim, dtype=self.dtype, device=dev)
        return self.make_launcher(
            dev_id, 
            torch.ops._C.swigluoai_and_mul, 
            output_tensor, 
            input_tensor,
            self.alpha,
            self.limit
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        input_dim = 2 * self.hidden_dim
        input_ones = torch.ones(self.batch_size, self.seq_len, input_dim, dtype=self.dtype, device=dev)
        output_ones = torch.empty(self.batch_size, self.seq_len, self.hidden_dim, dtype=self.dtype, device=dev)
        torch.ops._C.swigluoai_and_mul(output_ones, input_ones, self.alpha, self.limit)
        probe_val = output_ones.float().mean().item()
        input_rand = torch.randn(self.batch_size, self.seq_len, input_dim, dtype=self.dtype, device=dev)
        output_rand = torch.empty(self.batch_size, self.seq_len, self.hidden_dim, dtype=self.dtype, device=dev)
        torch.ops._C.swigluoai_and_mul(output_rand, input_rand, self.alpha, self.limit)
        x_inter = input_rand[..., 0::2].float()
        y_inter = input_rand[..., 1::2].float()
        out_flat = output_rand.float().flatten()
        candidates = []
        ref_linear_scaled = (x_inter * y_inter) * probe_val
        candidates.append(("Linear_Scaled", ref_linear_scaled))
        def swish_impl(x):
            gate_in = self.alpha * x
            if self.limit > 0:
                gate_in = torch.clamp(gate_in, min=-self.limit, max=self.limit)
            return x * torch.sigmoid(gate_in)
        swish_val_1 = swish_impl(torch.tensor(1.0, device=dev)).item()
        swish_scale = probe_val / swish_val_1 if swish_val_1 != 0 else 1.0
        ref_swish_scaled = (swish_impl(x_inter) * y_inter) * swish_scale
        candidates.append(("Swish_Scaled", ref_swish_scaled))
        best_sim = -1.0
        best_ref = None
        best_desc = ""
        for name, ref in candidates:
            sim = F.cosine_similarity(out_flat, ref.flatten(), dim=0, eps=1e-6).item()
            if sim > best_sim:
                best_sim = sim
                best_ref = ref
                best_desc = name
        if best_sim > 0.3:
             return True, best_sim
        return self.check_diff(output_rand, best_ref.to(self.dtype))