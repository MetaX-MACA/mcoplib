import time
import torch
import mcoplib._moe_C


class Dsv3RouterGemmRunner:

    def __init__(self, num_tokens=16, num_experts=256, hidden_dim=7168, dtype=torch.bfloat16, out_dtype=torch.float32, device="cuda",):
        self.num_tokens = num_tokens
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.out_dtype = out_dtype
        self.device = torch.device(device)

    def make_inputs(self):
        mat_a = torch.randn(
            self.num_tokens,
            self.hidden_dim,
            dtype=self.dtype,
            device=self.device,
        ).contiguous()
        mat_b = torch.randn(
            self.num_experts,
            self.hidden_dim,
            dtype=self.dtype,
            device=self.device,
        ).contiguous()
        output = torch.empty(
            self.num_tokens,
            self.num_experts,
            dtype=self.out_dtype,
            device=self.device,
        ).contiguous()
        return output, mat_a, mat_b

    def torch_reference(self, mat_a, mat_b):
        ref = torch.matmul(
            mat_a,
            mat_b.transpose(0, 1),
        )
        if self.out_dtype == torch.float32:
            ref = ref.float()
        elif self.out_dtype == torch.bfloat16:
            ref = ref.to(torch.bfloat16)
        return ref

    def run_kernel(self):
        output, mat_a, mat_b = self.make_inputs()
        torch.ops._moe_C.dsv3_router_gemm(
            output,
            mat_a,
            mat_b,
        )
        torch.cuda.synchronize()
        return output, mat_a, mat_b

    def verify(self):
        output, mat_a, mat_b = self.run_kernel()
        ref = self.torch_reference(
            mat_a,
            mat_b,
        )
        torch.testing.assert_close(
            output.float(),
            ref.float(),
            atol=2e-2,
            rtol=2e-2,
        )
        print(
            "Correctness PASS:",
            f"tokens={self.num_tokens},",
            f"experts={self.num_experts},",
            f"hidden={self.hidden_dim}",
        )

    def benchmark(self, warmup=20, iters=100,):
        output, mat_a, mat_b = self.make_inputs()
        for _ in range(warmup):
            torch.ops._moe_C.dsv3_router_gemm(
                output,
                mat_a,
                mat_b,
            )
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            torch.ops._moe_C.dsv3_router_gemm(
                output,
                mat_a,
                mat_b,
            )
        torch.cuda.synchronize()
        end = time.perf_counter()
        latency_ms = ((end - start) * 1000 / iters)
        flops = (2 * self.num_tokens * self.num_experts * self.hidden_dim)
        tflops = (flops / (latency_ms / 1000) / 1e12)
        print(
            "Benchmark:",
            f"tokens={self.num_tokens},",
            f"experts={self.num_experts},",
            f"hidden={self.hidden_dim}",
        )
        print(
            f"Latency : {latency_ms:.4f} ms"
        )
        print(
            f"TFLOPS  : {tflops:.3f}"
        )

    def run(self):
        self.verify()
        self.benchmark()



def test_all():

    # 与 C++ TORCH_CHECK 对应
    #
    # hidden_dim:
    #   DEFAULT_HIDDEN_DIM = 7168
    #   GLM_5_HIDDEN_DIM   = 6144
    #
    # experts:
    #   DEFAULT_NUM_EXPERTS = 256
    #   KIMI_K2_NUM_EXPERTS = 384
    #
    # 但是:
    #   hidden_dim=6144 只能 num_experts=256
    test_cases = [
        # DeepSeek V3
        # hidden=7168 experts=256
        (1, 256, 7168),
        (2, 256, 7168),
        (4, 256, 7168),
        (8, 256, 7168),
        (16, 256, 7168),

        # Kimi K2
        # hidden=7168 experts=384
        (1, 384, 7168),
        (8, 384, 7168),
        (16, 384, 7168),

        # GLM5
        # hidden=6144 experts=256
        (1, 256, 6144),
        (8, 256, 6144),
        (16, 256, 6144),
    ]

    for (num_tokens, num_experts, hidden_dim,) in test_cases:
        print(
            "\n=============================="
        )
        print(
            f"tokens={num_tokens}, "
            f"experts={num_experts}, "
            f"hidden={hidden_dim}"
        )
        runner = Dsv3RouterGemmRunner(
            num_tokens=num_tokens,
            num_experts=num_experts,
            hidden_dim=hidden_dim,
            dtype=torch.bfloat16,
            out_dtype=torch.float32,
        )
        runner.verify()

if __name__ == "__main__":
    runner = Dsv3RouterGemmRunner(
        num_tokens=16,
        num_experts=256,
        hidden_dim=7168,
        dtype=torch.bfloat16,
        out_dtype=torch.float32,
    )
    runner.run()
    test_all()