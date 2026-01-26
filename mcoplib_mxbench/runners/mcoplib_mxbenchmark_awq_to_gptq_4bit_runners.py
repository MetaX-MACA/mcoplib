import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
# 尝试导入算子库
try:
    import mcoplib._C as op
    if not hasattr(op, "awq_gemm"):
        op = None
except ImportError:
    op = None

class Awq_to_gptq_4bit_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.in_c = config.get("in_c", 4096)
        self.out_c = config.get("out_c", 4096)
        self.pack_factor = 8 # 源码中 COMPACT_FACTOR = 8
        
        # 校验参数
        assert self.out_c % self.pack_factor == 0, "out_c must be divisible by 8"
        
        # === 维度计算 (参考 C++ Host 代码) ===
        # AWQ 输入: [num_in_channels, num_out_channels / 8]
        self.qout_c = self.out_c // self.pack_factor
        
        # GPTQ 输出: [num_out_channels, compact_output_k]
        # 源码逻辑: (num_in_channels + 8 - 1) / 8
        self.compact_k = (self.in_c + self.pack_factor - 1) // self.pack_factor

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"IC={self.in_c} OC={self.out_c}")

        
        # 计算显存读写量
        input_bytes = self.in_c * self.qout_c * 4
        output_bytes = self.out_c * self.compact_k * 4
        
        state.add_global_memory_reads(int(input_bytes))
        state.add_global_memory_writes(int(output_bytes))
        state.add_element_count(int(self.in_c * self.out_c))

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 构造 AWQ 输入 [IC, OC/8]
            qweight = torch.randint(
                -2**31, 2**31-1, 
                (self.in_c, self.qout_c), 
                dtype=torch.int32, 
                device=dev
            )
            
            if op and hasattr(op, "awq_to_gptq_4bit"):
                func = op.awq_to_gptq_4bit
            else:
                func = torch.ops._C.awq_to_gptq_4bit

        return self.make_launcher(dev_id, func, qweight)

    # === 通用解包函数 ===
    def unpack_tensor(self, packed, shifts, reorder_indices=None):
        unpacked_list = []
        # 将 int32 拆成 8 个 4-bit
        for sh in shifts:
            val = (packed.to(torch.int64) >> sh) & 0xF
            unpacked_list.append(val)
        
        result = torch.stack(unpacked_list, dim=-1)
        
        # 应用重排
        if reorder_indices is not None:
            result = result[..., reorder_indices]
            
        return result.flatten(-2)

    def run_verification(self, dev_id):
        # 1. 设置验证规模
        # AWQ 输入: [IC, OC_packed] -> 逻辑上是 [IC, OC]
        # 这里的 out_c 必须是 8 的倍数以适应 packing
        in_c, out_c = 128, 256
        qout_c = out_c // 8  # 32
        
        dev = f'cuda:{dev_id}'
        dtype = torch.int32 

        # 2. 构造 AWQ 输入 [128, 32]
        # 这里的每个 int32 包含 8 个 4-bit 权重，按照 AWQ 格式排列
        qweight = torch.randint(-2**31, 2**31-1, (in_c, qout_c), dtype=dtype, device=dev)

        # 3. 运行 CUDA 算子
        # 接口原型: torch::Tensor awq_to_gptq_4bit(torch::Tensor qweight);
        try:
            # 直接调用底层注册的算子
            # Output Shape预期: [out_c, in_c // 8] = [256, 16]
            # 算子不仅做了转置，还将 packing 的方向从“横向”变成了“纵向”
            op_out = torch.ops._C.awq_to_gptq_4bit(qweight)
            
        except AttributeError:
            print("[Verify] Op not found in torch.ops._C."); return False, 1.0
        except RuntimeError as e:
            print(f"[Verify] Op Runtime Error: {e}"); return False, 1.0

        # 4. 验证逻辑 (Verify Logic)
        
        # 定义位移: 提取 int32 内部的 8 个 4-bit
        shifts = [0, 4, 8, 12, 16, 20, 24, 28]
        
        # AWQ 的打包顺序通常是交错的: [0, 4, 1, 5, 2, 6, 3, 7]
        awq_order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=dev)
        
        # --- Path A: 计算预期真值 (Python 模拟) ---
        # 1. 解包 AWQ 输入
        # qweight [IC, OC/8] -> unpack -> [IC, OC]
        ref_unpacked = self.unpack_tensor(qweight, shifts, reorder_indices=awq_order)
        
        # 2. 执行逻辑转置 (因为 GPTQ/Marlin 需要 Out-Channel Major)
        # [IC, OC] -> [OC, IC]
        ref_transposed = ref_unpacked.t()

        # --- Path B: 处理算子输出 ---
        # op_out [OC, IC/8] -> unpack -> [OC, IC]
        op_out_unpacked = self.unpack_tensor(op_out, shifts, reorder_indices=awq_order)
        
        # 5. 比较
        # 检查：(输入 -> 解包 -> 转置) 是否等于 (输出 -> 解包)
        return self.check_diff(op_out_unpacked.float(), ref_transposed.float(), threshold=0.1)

    # 辅助函数：解包 Tensor
    def unpack_tensor(self, packed_tensor, shifts, reorder_indices=None):
        # packed_tensor: [Rows, Cols] (int32)
        # return: [Rows, Cols * 8] (int32/float values)
        
        rows, cols = packed_tensor.shape
        # 扩展维度以进行位运算广播: [Rows, Cols, 1]
        data = packed_tensor.unsqueeze(-1)
        
        # 提取 8 个 4-bit: [Rows, Cols, 8]
        # ((data >> shift) & 0xF)
        unpacked = torch.stack([(data >> s) & 0xF for s in shifts], dim=-1).squeeze(-2)
        
        # 如果需要重排位顺序 (AWQ specific)
        if reorder_indices is not None:
            unpacked = unpacked.index_select(-1, reorder_indices)
            
        # 展平最后两维: [Rows, Cols * 8]
        return unpacked.view(rows, cols * 8)