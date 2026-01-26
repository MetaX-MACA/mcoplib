import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 尝试导入算子库
try:
    import mcoplib._C as op
    if not hasattr(op, "awq_dequantize"):
        op = None
except ImportError:
    op = None

class Awq_dequantize_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.in_c = config.get("in_c", 4096)
        self.out_c = config.get("out_c", 4096)
        self.group_size = config.get("group_size", 128)
        self.split_k_iters = config.get("split_k_iters", 0)
        self.thx = config.get("thx", 0)
        self.thy = config.get("thy", 0)
        
        # 校验参数约束
        assert self.out_c % 8 == 0, "out_c must be divisible by 8"
        assert self.in_c % self.group_size == 0, "in_c must be divisible by group_size"
        
        self.qout_c = self.out_c // 8
        self.num_groups = self.in_c // self.group_size

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"({self.in_c}x{self.out_c} G={self.group_size})")
        
        # 计算元素总量
        total_elements = self.in_c * self.out_c
        state.add_element_count(total_elements)
        
        # 计算显存读写量 (Bytes)
        # 1. Read Kernel (Int32, packed)
        read_bytes = self.in_c * self.qout_c * 4
        # 2. Read Zeros (Int32, packed)
        read_bytes += self.num_groups * self.qout_c * 4
        # 3. Read Scales (FP16)
        read_bytes += self.num_groups * self.out_c * 2
        
        # 4. Write Output (FP16)
        write_bytes = total_elements * 2
        
        state.add_global_memory_reads(int(read_bytes))
        state.add_global_memory_writes(int(write_bytes))

    def prepare_and_get_launcher(self, dev_id, tc_s):
        """准备数据并返回执行闭包"""
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            # 构造 Int32 的随机权重和零点
            kernel = torch.randint(-2**31, 2**31-1, (self.in_c, self.qout_c), dtype=torch.int32, device=dev)
            zeros = torch.randint(-2**31, 2**31-1, (self.num_groups, self.qout_c), dtype=torch.int32, device=dev)
            # 构造 FP16 的 Scales
            scales = torch.randn(self.num_groups, self.out_c, dtype=self.dtype, device=dev)
            
            # 优先使用 mcoplib._C 中的算子，否则尝试 torch.ops._C
            if op and hasattr(op, "awq_dequantize"):
                func = op.awq_dequantize
            else:
                func = torch.ops._C.awq_dequantize

        return self.make_launcher(dev_id, func, 
                                  kernel, scales, zeros, 
                                  self.split_k_iters, self.thx, self.thy)

    # === [关键复用] 通用解包 + 重排函数 (同 awq_gemm) ===
    def unpack_tensor(self, packed, shifts, reorder_indices=None):
        """
        通用的 Int32 -> 8x4bit 解包函数
        packed: [..., PackedDim]
        Output: [..., PackedDim * 8]
        """
        unpacked_list = []
        # 标准提取: 得到 [..., PackedDim, 8]
        for sh in shifts:
            val = (packed.to(torch.int64) >> sh) & 0xF
            unpacked_list.append(val)
        
        result = torch.stack(unpacked_list, dim=-1)
        
        # 应用 AWQ 的特殊重排 [0, 4, 1, 5, 2, 6, 3, 7]
        # 这是为了对齐 Tensor Core 的特殊 Layout
        if reorder_indices is not None:
            result = result[..., reorder_indices]
            
        return result.flatten(-2)

    def run_verification(self, dev_id):
        # 1. 缩小规模以加快验证，同时保证能整除
        in_c, out_c = 1024, 256
        group_size = 128
        groups = in_c // group_size
        qout_c = out_c // 8
        
        dev = f'cuda:{dev_id}'
        dtype = self.dtype

        # 2. 构造数据
        kernel = torch.randint(-2**31, 2**31-1, (in_c, qout_c), dtype=torch.int32, device=dev)
        zeros = torch.randint(-2**31, 2**31-1, (groups, qout_c), dtype=torch.int32, device=dev)
        scales = torch.randn(groups, out_c, dtype=dtype, device=dev)

        # 3. 运行算子
        func = None
        if op and hasattr(op, "awq_dequantize"):
            func = op.awq_dequantize
        else:
            try:
                func = torch.ops._C.awq_dequantize
            except:
                print("[Verify] Op not found."); return False, 1.0

        try:
            # 参数: kernel, scales, zeros, split_k_iters, thx, thy
            op_out = func(kernel, scales, zeros, 0, 0, 0)
        except RuntimeError as e:
            print(f"[Verify] Op Error: {e}"); return False, 1.0

        # 4. 运行 Reference (PyTorch 实现)
        # 定义移位和重排顺序 (从 awq_gemm 调试中确认的正确顺序)
        shifts = [0, 4, 8, 12, 16, 20, 24, 28]
        awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)

        # A. 解包权重
        # Input: [In_C, QOut_C] -> Unpack -> [In_C, Out_C]
        w_int = self.unpack_tensor(kernel, shifts, reorder_indices=awq_order).to(dtype)
        
        # B. 解包零点
        # Input: [Groups, QOut_C] -> Unpack -> [Groups, Out_C]
        z_unpacked = self.unpack_tensor(zeros, shifts, reorder_indices=awq_order).to(dtype)
        
        # C. 维度对齐 (广播)
        # Scales/Zeros: [Groups, Out_C] -> [In_C, Out_C]
        # 假设 Groups 维度对应 In_C 的分块
        s_expanded = scales.repeat_interleave(group_size, dim=0)
        z_expanded = z_unpacked.repeat_interleave(group_size, dim=0)
        
        # D. 计算: (W - Z) * S
        ref_out = (w_int - z_expanded) * s_expanded
        
        # 5. 比较
        # 允许魔法数反量化带来的微小误差 (1e-3 级别)
        return self.check_diff(op_out, ref_out, threshold=0.99)