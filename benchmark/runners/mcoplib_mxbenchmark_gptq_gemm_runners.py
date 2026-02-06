import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass

class Gptq_gemm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.bits = config.get("bits", 4)
        self.groupsize = config.get("groupsize", 128)
        self.m = config.get("m", 128)
        self.k = config.get("k", 1024)
        self.n = config.get("n", 4096)
        self.dtype = getattr(torch, config.get("dtype", "float16"))
        self.pack_factor = 32 // self.bits
        self.k_packed = self.k // self.pack_factor
        self.num_groups = (self.k + self.groupsize - 1) // self.groupsize
        self.use_exllama = False

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"(M{self.m}_K{self.k}_N{self.n}_G{self.groupsize}_B{self.bits})")
        total_elements = self.m * self.n
        state.add_element_count(total_elements)
        element_size = 2 if self.dtype == torch.float16 else 4
        read_vol = (self.m * self.k * element_size) + \
                   (self.k_packed * self.n * 4) + \
                   (self.num_groups * self.n * 2) + \
                   (self.num_groups * (self.n // self.pack_factor) * 4) + \
                   (self.k * 4)       
        write_vol = self.m * self.n * element_size
        state.add_global_memory_reads(read_vol)
        state.add_global_memory_writes(write_vol)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            a = torch.randn(self.m, self.k, dtype=self.dtype, device=dev)
            b_q_weight = torch.randint(-2000000000, 2000000000, (self.k_packed, self.n), dtype=torch.int32, device=dev)
            b_gptq_scales = torch.ones(self.num_groups, self.n, dtype=torch.float16, device=dev)
            b_gptq_qzeros = torch.randint(0, 2000000000, (self.num_groups, self.n // self.pack_factor), dtype=torch.int32, device=dev)
            b_g_idx = torch.arange(self.k, dtype=torch.int32, device=dev) // self.groupsize
            perm_space = torch.empty((self.n,), dtype=torch.int32, device=dev)
            temp_space = torch.empty((self.m, self.n), dtype=torch.float16, device=dev)
            is_bf16 = (self.dtype == torch.bfloat16)

        return self.make_launcher(
            dev_id, 
            torch.ops._C.gptq_gemm, 
            a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, 
            self.use_exllama, self.bits, self.groupsize, 
            perm_space, temp_space, is_bf16
        )

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        b_q_weight = torch.randint(-2000000000, 2000000000, (self.k_packed, self.n), dtype=torch.int32, device=dev)
        b_gptq_scales = torch.randn(self.num_groups, self.n, dtype=torch.float16, device=dev)
        b_gptq_qzeros = torch.randint(0, 2000000000, (self.num_groups, self.n // self.pack_factor), dtype=torch.int32, device=dev)
        b_g_idx = torch.arange(self.k, dtype=torch.int32, device=dev) // self.groupsize
        perm_space = torch.empty((self.n,), dtype=torch.int32, device=dev)
        temp_space = torch.empty((self.m, self.n), dtype=torch.float16, device=dev)
        is_bf16 = (self.dtype == torch.bfloat16)
        a1 = torch.randn(self.m, self.k, dtype=self.dtype, device=dev)
        scale_factor = 2.0
        a2 = (a1 * scale_factor).to(self.dtype)
        out1 = torch.ops._C.gptq_gemm(
            a1, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, 
            self.use_exllama, self.bits, self.groupsize, 
            perm_space, temp_space, is_bf16
        )
        out2 = torch.ops._C.gptq_gemm(
            a2, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, 
            self.use_exllama, self.bits, self.groupsize, 
            perm_space, temp_space, is_bf16
        )
        ref_out2 = out1 * scale_factor
        if out1.abs().sum() == 0:
            print("[Verify Error] Output is all zeros.")
            return self.check_diff(out1, torch.ones_like(out1))
        return self.check_diff(out2, ref_out2)