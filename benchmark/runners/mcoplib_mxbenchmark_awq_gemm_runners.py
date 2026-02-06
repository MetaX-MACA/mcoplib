import torch
import torch.nn.functional as F

try:
    import mcoplib._C as op
    if not hasattr(op, "awq_gemm"):
        op = None
except ImportError:
    op = None

from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

class Awq_gemm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.M = config.get("M", 16)
        self.N = config.get("N", 4096)
        self.K = config.get("K", 4096)
        self.group_size = config.get("group_size", 128)
        self.pack_factor = 8 
        self.dtype_bf16 = (self.dtype == torch.bfloat16)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"M={self.M} N={self.N} K={self.K}")
        state.add_element_count(int(2 * self.M * self.N * self.K))

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            input_tensor = torch.randn(self.M, self.K, dtype=self.dtype, device=dev)
            packed_kernel = torch.randint(0, 2**31, (self.N, self.K // self.pack_factor), dtype=torch.int32, device=dev)
            
            num_groups = self.K // self.group_size
            scaling_factors = torch.randn(num_groups, self.N, dtype=self.dtype, device=dev)
            packed_zeros = torch.randint(0, 2**31, (num_groups, self.N // self.pack_factor), dtype=torch.int32, device=dev)
            
            temp_space = torch.empty(self.M * self.N, dtype=torch.float32, device=dev)
            func = torch.ops._C.awq_gemm

        return self.make_launcher(
            dev_id, func, 
            input_tensor, packed_kernel, scaling_factors, packed_zeros, 
            1, temp_space, self.dtype_bf16
        )

    def unpack_tensor(self, packed, shifts, reorder_indices=None):
        unpacked_list = []
        for sh in shifts:
            val = (packed.to(torch.int64) >> sh) & 0xF
            unpacked_list.append(val)
        
        result = torch.stack(unpacked_list, dim=-1)
        
        if reorder_indices is not None:
            result = result[..., reorder_indices]
            
        return result.flatten(-2)

    def run_verification(self, dev_id):
        M, N, K = 32, 256, 1024 
        group_size = 128
        pack_factor = 8
        groups = K // group_size
        
        dev = f'cuda:{dev_id}'
        dtype = self.dtype

        input_tensor = torch.randn(M, K, dtype=dtype, device=dev)
        
        packed_kernel_flat = torch.randint(
            -2**31, 2**31-1, (N * (K // pack_factor),), dtype=torch.int32, device=dev
        )
        packed_kernel = packed_kernel_flat.view(N, K // pack_factor)

        scales = torch.randn(groups, N, dtype=dtype, device=dev)

        packed_zeros = torch.randint(
            -2**31, 2**31-1, (groups, N // pack_factor), dtype=torch.int32, device=dev
        )

        temp_space = torch.empty(M * N, dtype=torch.float32, device=dev)

        func = torch.ops._C.awq_gemm
        try:
            op_out = func(
                input_tensor, packed_kernel, scales, packed_zeros, 
                1, temp_space, self.dtype_bf16
            )
        except RuntimeError as e:
            print(f"[Verify] Op Runtime Error: {e}")
            return False, 1.0

        shifts = [0, 4, 8, 12, 16, 20, 24, 28]
        
        awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)

        k_view = packed_kernel.view(-1).view(K // 8, N) 
        k_unpacked = self.unpack_tensor(k_view, shifts, reorder_indices=awq_order).to(dtype)
        k_unpacked_3d = k_unpacked.view(K // 8, N, 8)
        w_int = k_unpacked_3d.permute(0, 2, 1).reshape(K, N)

        z_unpacked = self.unpack_tensor(packed_zeros, shifts, reorder_indices=awq_order).to(dtype)
        
        s_expanded = scales.repeat_interleave(group_size, dim=0)
        z_expanded = z_unpacked.repeat_interleave(group_size, dim=0)

        w_fp = (w_int - z_expanded) * s_expanded

        ref_out = torch.matmul(input_tensor, w_fp)

        return self.check_diff(op_out, ref_out, threshold=0.99)