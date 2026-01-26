import torch
import torch.nn.functional as F

# 尝试导入算子库
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
            # 真实 Bench 时的形状
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

    # === 通用解包 + 重排函数 ===
    def unpack_tensor(self, packed, shifts, reorder_indices=None):
        unpacked_list = []
        # 标准提取: 得到 [PackDim, 8] (按位移顺序 0, 4, 8...)
        for sh in shifts:
            val = (packed.to(torch.int64) >> sh) & 0xF
            unpacked_list.append(val)
        
        result = torch.stack(unpacked_list, dim=-1)
        
        # 应用 AWQ 的特殊重排 [0, 2, 4, 6, 1, 3, 5, 7]
        # 如果不重排，顺序是 [Bit0-3, Bit4-7, Bit8-11...] (即 Logical 0, 4, 1, 5...)
        if reorder_indices is not None:
            result = result[..., reorder_indices]
            
        return result.flatten(-2)

    def run_verification(self, dev_id):
        # 1. 基础参数设置
        # 注意：Kernel 内部把 input reshape 成了 (K/8, N)，所以 N 必须能被 8 整除
        M, N, K = 32, 256, 1024 
        group_size = 128
        pack_factor = 8
        groups = K // group_size
        
        dev = f'cuda:{dev_id}'
        dtype = self.dtype

        # 2. 构造数据
        # Input: [M, K]
        input_tensor = torch.randn(M, K, dtype=dtype, device=dev)
        
        # Packed Kernel: 物理内存上是 N * (K/8) 个 int32
        # 【陷阱解析】虽然形状叫 (N, K/8)，但 Kernel 内部是按 (K/8, N) 访问的
        # 为了避免随机数生成的“转置”歧义，我们生成 flat 数据，让 memory layout 保持线性
        packed_kernel_flat = torch.randint(
            -2**31, 2**31-1, (N * (K // pack_factor),), dtype=torch.int32, device=dev
        )
        # 算子要求输入是 (N, K/8) 才能拿到正确的 size(0)=N
        packed_kernel = packed_kernel_flat.view(N, K // pack_factor)

        # Scales: [Groups, N]
        scales = torch.randn(groups, N, dtype=dtype, device=dev)

        # Zeros: [Groups, N/8] -> 同样按 (Groups, N/8) 线性存储
        packed_zeros = torch.randint(
            -2**31, 2**31-1, (groups, N // pack_factor), dtype=torch.int32, device=dev
        )

        # Temp Space
        temp_space = torch.empty(M * N, dtype=torch.float32, device=dev)

        # 3. 运行 CUDA 算子
        func = torch.ops._C.awq_gemm
        try:
            op_out = func(
                input_tensor, packed_kernel, scales, packed_zeros, 
                1, temp_space, self.dtype_bf16
            )
        except RuntimeError as e:
            print(f"[Verify] Op Runtime Error: {e}")
            return False, 1.0

        # 位解包的取数的位置，8个4-bit
        shifts = [0, 4, 8, 12, 16, 20, 24, 28]
        
        #awq特殊的非顺序打包，前半段在每个字节的低四位，后半段都放在高四位
        awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)

        # A. 解包 Kernel
        k_view = packed_kernel.view(-1).view(K // 8, N) 
        k_unpacked = self.unpack_tensor(k_view, shifts, reorder_indices=awq_order).to(dtype)
        k_unpacked_3d = k_unpacked.view(K // 8, N, 8)
        w_int = k_unpacked_3d.permute(0, 2, 1).reshape(K, N)

        # B. 解包 Zeros
        # Zeros 是 [Groups, N/8] -> 解包后 [Groups, N]
        # 源码显示 Zeros 也用了同样的 unpack 逻辑
        z_unpacked = self.unpack_tensor(packed_zeros, shifts, reorder_indices=awq_order).to(dtype)
        # 这里 flatten 后是 [Groups, N/8 * 8] = [Groups, N]
        
        # C. 广播与计算
        # Scales: [Groups, N] -> 广播到 [K, N]
        s_expanded = scales.repeat_interleave(group_size, dim=0)
        z_expanded = z_unpacked.repeat_interleave(group_size, dim=0)

        # D. 反量化: W_fp = (W_int - Z) * S
        w_fp = (w_int - z_expanded) * s_expanded

        # E. 标准矩阵乘法: [M, K] @ [K, N]
        ref_out = torch.matmul(input_tensor, w_fp)

        return self.check_diff(op_out, ref_out, threshold=0.99)