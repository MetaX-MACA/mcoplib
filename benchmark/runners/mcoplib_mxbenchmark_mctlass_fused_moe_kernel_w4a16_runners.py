import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_moe_fused_w4a16 as target_lib
except ImportError:
    try:
        import mcoplib.op as target_lib
    except ImportError:
        target_lib = None

class Mctlass_fused_moe_kernel_w4a16_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 4096)
        self.num_experts = config.get("num_experts", 8)
        self.topk = config.get("topk", 2)
        self.K = config.get("K", 4096)
        self.N = config.get("N", 11008)
        self.group_size = config.get("group_size", 64)
        if self.dtype != torch.bfloat16:
            print(f"Warning: {name} requires bfloat16. Overriding config.")
            self.dtype = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16")
        state.add_summary("Shape", f"({self.num_tokens} {self.num_experts} {self.topk})")
        num_valid_tokens = self.num_tokens * self.topk
        flops = 2 * num_valid_tokens * self.N * self.K
        state.add_element_count(int(flops)) 
        size_a = self.num_tokens * self.K * 2
        size_b_total = self.num_experts * self.N * (self.K // 2)
        size_c = self.num_tokens * self.N * 2
        state.add_global_memory_reads(size_a + size_b_total) 
        state.add_global_memory_writes(size_c)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        args = self._generate_args(dev_id, for_verification=False)
        return self.make_launcher(dev_id, target_lib.mctlass_fused_moe_kernel_w4a16, *args)

    def _generate_args(self, dev_id, for_verification=False):
        dev = f'cuda:{dev_id}'
        n_tokens = 128 if for_verification else self.num_tokens
        k = self.K
        n = self.N
        a = torch.randn((n_tokens, k), dtype=self.dtype, device=dev)
        if for_verification:
            b = torch.full((self.num_experts, n, k // 2), 17, dtype=torch.uint8, device=dev)
        else:
            b = torch.randint(0, 255, (self.num_experts, n, k // 2), dtype=torch.uint8, device=dev)
        c = torch.zeros((n_tokens, n), dtype=self.dtype, device=dev)
        scale_shape = (self.num_experts, n, k // self.group_size)
        if for_verification:
            b_scales = torch.ones(scale_shape, dtype=self.dtype, device=dev)
            zp_b = torch.zeros(scale_shape, dtype=torch.uint8, device=dev)
        else:
            b_scales = torch.randn(scale_shape, dtype=self.dtype, device=dev)
            zp_b = torch.randint(0, 255, scale_shape, dtype=torch.uint8, device=dev)
        num_valid_tokens = n_tokens * self.topk
        if for_verification:
            expert_ids = torch.arange(num_valid_tokens, dtype=torch.int32, device=dev)
            tokens_per_expert = (num_valid_tokens + self.num_experts - 1) // self.num_experts
            expert_ids = (expert_ids // tokens_per_expert).clamp(0, self.num_experts - 1)
            token_ids = torch.arange(num_valid_tokens, dtype=torch.int32, device=dev) % n_tokens
        else:
            token_ids = torch.arange(num_valid_tokens, dtype=torch.int32, device=dev) % n_tokens
            expert_ids = torch.randint(0, self.num_experts, (num_valid_tokens,), dtype=torch.int32, device=dev)
            expert_ids, sort_idx = torch.sort(expert_ids)
            token_ids = token_ids[sort_idx]
        moe_weight = torch.rand((n_tokens, self.topk), dtype=torch.float32, device=dev)
        num_tokens_post_padded = torch.tensor([num_valid_tokens], dtype=torch.int32, device=dev)
        EM = n 
        mul_routed_weight = False 
        return [
            a, b, c, b_scales, zp_b, 
            moe_weight, token_ids, expert_ids, num_tokens_post_padded,
            n, k, EM, num_valid_tokens, self.topk, mul_routed_weight
        ]

    def run_verification(self, dev_id):
        args = self._generate_args(dev_id, for_verification=True)
        target_lib.mctlass_fused_moe_kernel_w4a16(*args)
        out_op = args[2]
        a = args[0]
        token_ids = args[6]
        a_float = a.float()
        row_sums = a_float.sum(dim=1)
        out_ref = torch.zeros_like(out_op, dtype=torch.float32)
        for i in range(token_ids.size(0)):
            tid = token_ids[i].item()
            out_ref[tid] += row_sums[tid]
        out_ref = out_ref.to(self.dtype)
        return self.check_diff(out_op, out_ref, threshold=0.99)