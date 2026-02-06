import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Merge_state_v2_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 32768)
        self.num_heads = config.get("num_heads", 32)
        self.head_dim = config.get("head_dim", 128)
        if self.config.get("dtype") == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.num_tokens} {self.num_heads} {self.head_dim})")
        N, H, D = self.num_tokens, self.num_heads, self.head_dim
        total_elements_val = N * H * D
        total_elements_lse = N * H
        size_val = 2
        size_lse = 4
        read_bytes = (2 * total_elements_val * size_val) + (2 * total_elements_lse * size_lse)
        write_bytes = (1 * total_elements_val * size_val) + (1 * total_elements_lse * size_lse)
        state.add_element_count(total_elements_val)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = torch.device(f'cuda:{dev_id}')
            v_a = torch.randn(self.num_tokens, self.num_heads, self.head_dim, dtype=self.dtype, device=dev)
            s_a = torch.randn(self.num_tokens, self.num_heads, dtype=torch.float32, device=dev)
            v_b = torch.randn(self.num_tokens, self.num_heads, self.head_dim, dtype=self.dtype, device=dev)
            s_b = torch.randn(self.num_tokens, self.num_heads, dtype=torch.float32, device=dev)
            v_merged = torch.empty_like(v_a)
            s_merged = torch.empty_like(s_a)
        return self.make_launcher(
            dev_id,
            torch.ops.sgl_kernel.merge_state_v2,
            v_a, s_a, v_b, s_b, v_merged, s_merged
        )

    def run_verification(self, dev_id):
        dev = torch.device(f'cuda:{dev_id}')
        v_a = torch.randn(self.num_tokens, self.num_heads, self.head_dim, device=dev).to(self.dtype)
        s_a = torch.randn(self.num_tokens, self.num_heads, device=dev).to(torch.float32)
        v_b = torch.randn(self.num_tokens, self.num_heads, self.head_dim, device=dev).to(self.dtype)
        s_b = torch.randn(self.num_tokens, self.num_heads, device=dev).to(torch.float32)
        v_out_op = torch.empty_like(v_a)
        s_out_op = torch.empty_like(s_a)
        torch.cuda.synchronize()
        torch.ops.sgl_kernel.merge_state_v2(v_a, s_a, v_b, s_b, v_out_op, s_out_op)
        torch.cuda.synchronize()
        p_lse, s_lse = s_a.float(), s_b.float()
        p_out, s_out = v_a.float(), v_b.float()
        max_lse = torch.maximum(p_lse, s_lse)
        p_exp = torch.exp(p_lse - max_lse)
        s_exp = torch.exp(s_lse - max_lse)
        out_se = p_exp + s_exp
        out_lse_ref = torch.log(out_se) + max_lse
        p_weight = (p_exp / out_se).unsqueeze(-1)
        s_weight = (s_exp / out_se).unsqueeze(-1)
        out_val_ref = p_out * p_weight + s_out * s_weight
        threshold_val = 0.99
        pass_v, diff_v = self.check_diff(v_out_op, out_val_ref.to(self.dtype), threshold=threshold_val)
        pass_s, diff_s = self.check_diff(s_out_op, out_lse_ref.to(torch.float32), threshold=threshold_val)
        is_passed = pass_v and pass_s
        max_diff = max(diff_v, diff_s)
        if not is_passed:
            print(f"    [Debug] Val Diff: {diff_v:.5f}, LSE Diff: {diff_s:.5f}, Threshold: {threshold_val}")
        return is_passed, max_diff