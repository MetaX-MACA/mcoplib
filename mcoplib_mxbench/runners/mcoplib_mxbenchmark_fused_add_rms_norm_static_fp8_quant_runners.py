import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 尝试导入算子库
try:
    import mcoplib._C as op
except ImportError:
    pass

class Fused_add_rms_norm_static_fp8_quant_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        # 按照要求，这里使用 batch_size 代表总 token 数
        self.batch_size = config.get("batch_size", 32768)
        self.hidden_size = config.get("hidden_size", 4096)
        self.epsilon = config.get("epsilon", 1e-6)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))
        state.add_summary("Shape", f"({self.batch_size} {self.hidden_size})")
        
        total_elements = self.batch_size * self.hidden_size
        state.add_element_count(total_elements)

        # 显存带宽估算 (Bytes per element):
        # Read: Input (2B) + Residual (2B)
        # Write: Residual (2B, inplace) + Out (1B, FP8)
        # Total: 7 Bytes
        element_size = 2 # bf16/fp16
        fp8_size = 1
        
        read_bytes = total_elements * (2 * element_size) 
        write_bytes = total_elements * (element_size + fp8_size)
        
        state.add_global_memory_reads(read_bytes)
        state.add_global_memory_writes(write_bytes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            shape = (self.batch_size, self.hidden_size)
            
            input_tensor = torch.randn(shape, dtype=self.dtype, device=dev)
            residual_tensor = torch.randn(shape, dtype=self.dtype, device=dev)
            weight = torch.ones(self.hidden_size, dtype=self.dtype, device=dev)
            scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
            
            # 输出为 FP8 (E4M3)
            out_tensor = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)
            
            # 获取算子函数
            func = getattr(op, "fused_add_rms_norm_static_fp8_quant", None)
            if func is None:
                func = torch.ops._C.fused_add_rms_norm_static_fp8_quant

        return self.make_launcher(dev_id, func, out_tensor, input_tensor, residual_tensor, weight, scale, self.epsilon)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        shape = (self.batch_size, self.hidden_size)
        
        input_tensor = torch.randn(shape, dtype=self.dtype, device=dev)
        residual_tensor = torch.randn(shape, dtype=self.dtype, device=dev)
        weight = torch.ones(self.hidden_size, dtype=self.dtype, device=dev)
        scale = torch.tensor([1.0], dtype=torch.float32, device=dev)
        out_op = torch.empty(shape, dtype=torch.float8_e4m3fn, device=dev)

        # 备份 Ref 输入
        input_ref = input_tensor.clone().float()
        residual_ref = residual_tensor.clone().float()
        weight_ref = weight.clone().float()
        
        func = getattr(op, "fused_add_rms_norm_static_fp8_quant", None)
        if func is None:
            func = torch.ops._C.fused_add_rms_norm_static_fp8_quant
            
        func(out_op, input_tensor, residual_tensor, weight, scale, self.epsilon)

        # Python Ref 实现
        # 1. Add
        x_sum = input_ref + residual_ref
        # 2. RMS Norm
        variance = x_sum.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + self.epsilon)
        y = x_sum * rsqrt * weight_ref
        # 3. Scale & Quant
        out_ref = (y / 1.0).to(torch.float8_e4m3fn).float()
        
        # 比较 (FP8 精度损失较大，阈值设低一点)
        return self.check_diff(out_op.float(), out_ref, threshold=0.90)