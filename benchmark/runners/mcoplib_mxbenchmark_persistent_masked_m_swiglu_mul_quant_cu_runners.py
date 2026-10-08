import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Persistent_masked_m_swiglu_mul_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_experts = config.get("num_experts", 8)
        self.num_tokens = config.get("num_tokens", 64)
        self.hidden_size = config.get("hidden_size", 4096)
        self.input_dtype = getattr(torch, config.get("input_dtype", "bfloat16"))
        self.output_dtype = getattr(torch, config.get("output_dtype", "float8_e4m3fn"))
        self.use_ue8m0 = config.get("use_ue8m0", False)
        self.limit = config.get("limit", 6.0)
        self.seed = config.get("seed", None)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", f"{self.input_dtype} -> {self.output_dtype}")
        state.add_summary("Shape", f"(E{self.num_experts}_T{self.num_tokens}_H{self.hidden_size})")

        total_elements = self.num_experts * self.num_tokens * self.hidden_size
        state.add_element_count(total_elements)
        
        in_elem_size = 2 if self.input_dtype == torch.bfloat16 else 4
        out_elem_size = 1
        
        input_size = self.num_experts * self.num_tokens * 2 * self.hidden_size * in_elem_size
        output_q_size = total_elements * out_elem_size
        num_groups = self.hidden_size // 128
        scale_elem_size = 4
        output_s_size = self.num_experts * self.num_tokens * num_groups * scale_elem_size
        counts_size = self.num_experts * 4
        limit_size = 2 if self.input_dtype == torch.bfloat16 else 4
        
        state.add_global_memory_reads(input_size + counts_size + limit_size)
        state.add_global_memory_writes(output_q_size + output_s_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            input_tensor = torch.randn(
                self.num_experts, self.num_tokens, 2 * self.hidden_size,
                dtype=self.input_dtype, device=dev
            )
            
            tokens_per_expert = torch.tensor(
                [self.num_tokens] * self.num_experts,
                dtype=torch.int32, device=dev
            )
            
            limit = torch.tensor([self.limit], dtype=self.input_dtype, device=dev)
            
            y_q = torch.empty(
                self.num_experts, self.num_tokens, self.hidden_size,
                dtype=self.output_dtype, device=dev
            )
            
            num_groups = self.hidden_size // 128
            scale_dtype = torch.int32 if self.use_ue8m0 else torch.float32
            y_s = torch.empty(
                self.num_experts, self.num_tokens, num_groups,
                dtype=scale_dtype, device=dev
            )
            
        return self.make_launcher(
            dev_id,
            torch.ops._C.persistent_masked_m_swiglu_mul_quant,
            input_tensor,
            tokens_per_expert,
            y_q,
            y_s,
            limit,
            self.use_ue8m0
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        E, T, H = 2, 4, 256
        
        input_tensor = torch.randn(E, T, 2 * H, dtype=self.input_dtype, device=dev)
        tokens_per_expert = torch.tensor([T] * E, dtype=torch.int32, device=dev)
        limit = torch.tensor([self.limit], dtype=self.input_dtype, device=dev)
        
        y_q = torch.empty(E, T, H, dtype=self.output_dtype, device=dev)
        scale_dtype = torch.int32 if self.use_ue8m0 else torch.float32
        y_s = torch.empty(E, T, H // 128, dtype=scale_dtype, device=dev)
        
        torch.ops._C.persistent_masked_m_swiglu_mul_quant(
            input_tensor, tokens_per_expert, y_q, y_s, limit, self.use_ue8m0
        )
        
        y_q_float = y_q.float()
        fp8_max = 448.0 if self.output_dtype == torch.float8_e4m3fn else 240.0
        
        if torch.any(torch.abs(y_q_float) > fp8_max * 1.1):
            return False, f"Output values exceed FP8 range: {y_q_float.abs().max()}"
        
        if not self.use_ue8m0:
            if torch.any(y_s <= 0):
                return False, f"Scale values are not positive: {y_s.min()}"
        
        return True, 0.0