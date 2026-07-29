import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass

class Dsv3RouterGemmRunner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 16)
        self.num_experts = config.get("num_experts", 256)
        self.hidden_dim = config.get("hidden_dim", 7168)
        self.dtype = torch.bfloat16
        self.out_dtype = torch.float32

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        shape = (
            f"tokens={self.num_tokens},"
            f"experts={self.num_experts},"
            f"hidden={self.hidden_dim}"
        )
        state.add_summary(
            "Shape",
            shape
        )
        # mat_a
        # [tokens, hidden]
        #
        # mat_b
        # [experts, hidden]
        #
        # output
        # [tokens, experts]
        input_elements = (self.num_tokens * self.hidden_dim + self.num_experts * self.hidden_dim)
        output_elements = (self.num_tokens * self.num_experts)
        state.add_element_count(input_elements + output_elements)
        # bf16 input
        read_bytes = (input_elements * 2)
        # float output
        write_bytes = (output_elements * 4)
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"
            mat_a = torch.randn(self.num_tokens, self.hidden_dim, dtype=torch.bfloat16, device=dev,).contiguous()
            mat_b = torch.randn(self.num_experts, self.hidden_dim, dtype=torch.bfloat16, device=dev,).contiguous()
            output = torch.empty(self.num_tokens, self.num_experts, dtype=torch.float32, device=dev,).contiguous()
        return self.make_launcher(dev_id, torch.ops._moe_C.dsv3_router_gemm, output, mat_a, mat_b,)

    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"
        torch.manual_seed(0)
        mat_a = torch.randn(self.num_tokens, self.hidden_dim, dtype=torch.bfloat16, device=dev,)
        mat_b = torch.randn(self.num_experts, self.hidden_dim, dtype=torch.bfloat16, device=dev,)
        output = torch.empty(self.num_tokens, self.num_experts, dtype=torch.float32,device=dev,)
        torch.ops._moe_C.dsv3_router_gemm(output, mat_a, mat_b,)
        ref = torch.matmul(mat_a, mat_b.transpose(0,1), ).float()
        diff = torch.max(torch.abs(output.float() - ref))
        passed = torch.allclose(output.float(), ref, atol=2e-2, rtol=2e-2)
        return bool(passed), float(diff)