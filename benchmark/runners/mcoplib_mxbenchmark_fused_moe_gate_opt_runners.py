import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Fused_moe_gate_opt_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.batch_size = config.get("batch_size", 4096)
        self.num_experts = config.get("num_experts", 320)
        self.num_expert_group = config.get("num_expert_group", 1)
        self.topk_group = config.get("topk_group", 1)
        self.topk = config.get("topk", 8)
        self.renormalize = config.get("renormalize", True)
        self.num_fused_shared_experts = config.get("num_fused_shared_experts", 0)
        self.routed_scaling_factor = config.get("routed_scaling_factor", 1.0)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.num_experts})")
        
        total_elements = self.batch_size * self.num_experts
        state.add_element_count(total_elements)
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        read_bytes = (self.batch_size * self.num_experts * element_size) + (self.num_experts * element_size)
        write_bytes = (self.batch_size * self.topk * 4) + (self.batch_size * self.topk * 4)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            gating_outputs = torch.randn((self.batch_size, self.num_experts), dtype=self.dtype, device=dev)
            correction_bias = torch.randn(self.num_experts, dtype=self.dtype, device=dev)
            
            out_routing_weights = torch.empty((self.batch_size, self.topk), dtype=torch.float32, device=dev)
            out_selected_experts = torch.empty((self.batch_size, self.topk), dtype=torch.int32, device=dev)

        return self.make_launcher(
            dev_id, 
            torch.ops.sgl_kernel.fused_moe_gate_opt, 
            gating_outputs, 
            correction_bias, 
            out_routing_weights, 
            out_selected_experts, 
            self.topk, 
            self.renormalize, 
            self.num_expert_group, 
            self.topk_group, 
            self.num_fused_shared_experts, 
            self.routed_scaling_factor
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        gating_outputs = torch.randn((self.batch_size, self.num_experts), dtype=self.dtype, device=dev)
        correction_bias = torch.randn(self.num_experts, dtype=self.dtype, device=dev)
        
        out_routing_weights = torch.empty((self.batch_size, self.topk), dtype=torch.float32, device=dev)
        out_selected_experts = torch.empty((self.batch_size, self.topk), dtype=torch.int32, device=dev)

        gate_ref = gating_outputs.clone().float()
        bias_ref = correction_bias.clone().float()

        torch.ops.sgl_kernel.fused_moe_gate_opt(
            gating_outputs, 
            correction_bias, 
            out_routing_weights, 
            out_selected_experts, 
            self.topk, 
            self.renormalize, 
            self.num_expert_group, 
            self.topk_group, 
            self.num_fused_shared_experts, 
            self.routed_scaling_factor
        )

        gate_f32 = gating_outputs.float()
        gate_sigmoid_bf16 = torch.sigmoid(gate_f32).to(self.dtype)
        bias_bf16 = correction_bias.to(self.dtype)
        score_bf16 = gate_sigmoid_bf16 + bias_bf16

        score_f32 = score_bf16.float()
        indices_penalty = torch.arange(self.num_experts, device=dev, dtype=torch.float32) * 1e-7
        score_f32_tied = score_f32 - indices_penalty
        #选出top_k个
        _, expected_idx = torch.topk(score_f32_tied, self.topk, dim=-1)
        
        expected_w_f32 = torch.gather(gate_sigmoid_bf16, -1, expected_idx).float()
        #归一化，k个专家权重之和为1
        weight_sum = expected_w_f32.sum(dim=-1, keepdim=True)
        expected_w_f32 = expected_w_f32 / weight_sum
        #缩放
        if self.renormalize:
            expected_w_f32 = expected_w_f32 * self.routed_scaling_factor

        pass_w, diff_w = self.check_diff(out_routing_weights, expected_w_f32, threshold=0.999)
        
        idx_diff = (out_selected_experts != expected_idx.int()).float().mean().item()
        pass_idx = (idx_diff < 0.05) 

        is_passed = pass_w and pass_idx
        return is_passed, diff_w