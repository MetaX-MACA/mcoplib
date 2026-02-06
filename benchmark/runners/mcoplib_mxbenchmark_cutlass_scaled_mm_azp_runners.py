import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

func_name = "cutlass_scaled_mm_azp"
_kernel_func = None

try:
    import mcoplib._C
    if hasattr(mcoplib._C, func_name):
        _kernel_func = getattr(mcoplib._C, func_name)
except ImportError:
    pass

if _kernel_func is None:
    try:
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, func_name):
            _kernel_func = getattr(torch.ops._C, func_name)
    except:
        pass

class Cutlass_scaled_mm_azp_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.m = config.get("m", 4096)
        self.n = config.get("n", 4096)
        self.k = config.get("k", 4096)
        self.dtype = getattr(torch, config.get("dtype", "bfloat16"))
        assert self.k % 16 == 0, "K dimension must be a multiple of 16 for alignment"
        assert self.n % 16 == 0, "N dimension must be a multiple of 16 for alignment"

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", f"M={self.m} N={self.n} K={self.k}")
        state.add_element_count(self.m * self.n)
        size_a = self.m * self.k * 1
        size_b = self.k * self.n * 1
        size_scales = (self.m * 4) + (self.n * 4)
        size_azp = (self.n * 4) + (self.m * 4)
        element_size = 2 if self.dtype == torch.bfloat16 or self.dtype == torch.float16 else 4
        size_bias = self.n * element_size
        size_c = self.m * self.n * element_size
        total_read = size_a + size_b + size_scales + size_azp + size_bias
        total_write = size_c
        state.add_global_memory_reads(total_read)
        state.add_global_memory_writes(total_write)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if _kernel_func is None:
            raise RuntimeError(f"算子 {func_name} 未找到，请检查 mcoplib 安装。")
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            a = torch.randint(-127, 127, (self.m, self.k), dtype=torch.int8, device=dev)
            b_storage = torch.randint(-127, 127, (self.n, self.k), dtype=torch.int8, device=dev)
            b = b_storage.t()
            c = torch.empty((self.m, self.n), dtype=self.dtype, device=dev)
            a_scales = torch.randn(self.m, 1, dtype=torch.float32, device=dev)
            b_scales = torch.randn(1, self.n, dtype=torch.float32, device=dev)
            azp_adj = torch.randint(-1000, 1000, (self.n,), dtype=torch.int32, device=dev)
            azp = torch.randint(-10, 10, (self.m,), dtype=torch.int32, device=dev)
            bias = torch.randn(self.n, dtype=self.dtype, device=dev)
        return self.make_launcher(dev_id, _kernel_func, c, a, b, a_scales, b_scales, azp_adj, azp, bias)

    def run_verification(self, dev_id):
        if _kernel_func is None:
            return False, 1.0
        dev = f'cuda:{dev_id}'
        a = torch.randint(-5, 5, (self.m, self.k), dtype=torch.int8, device=dev)
        b_storage = torch.randint(-5, 5, (self.n, self.k), dtype=torch.int8, device=dev)
        b = b_storage.t()
        a_scales = torch.rand(self.m, 1, dtype=torch.float32, device=dev)
        b_scales = torch.rand(1, self.n, dtype=torch.float32, device=dev)
        azp_adj = torch.randint(-10, 10, (self.n,), dtype=torch.int32, device=dev)
        azp = torch.randint(-2, 2, (self.m,), dtype=torch.int32, device=dev)
        bias = torch.randn(self.n, dtype=self.dtype, device=dev)
        c_op = torch.empty((self.m, self.n), dtype=self.dtype, device=dev)
        _kernel_func(c_op, a, b, a_scales, b_scales, azp_adj, azp, bias)
        a_f = a.float()
        b_f = b.float()
        matmul_res = torch.matmul(a_f, b_f)
        azp_term = azp.float().unsqueeze(1)
        azp_adj_term = azp_adj.float().unsqueeze(0)
        correction = azp_term * azp_adj_term
        accum = matmul_res + correction
        res = accum * a_scales * b_scales
        res = res + bias.float()
        out_ref = res.to(self.dtype)
        return self.check_diff(c_op, out_ref, threshold=0.99)