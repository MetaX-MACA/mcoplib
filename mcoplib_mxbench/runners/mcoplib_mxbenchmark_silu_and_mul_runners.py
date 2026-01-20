import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    # 触发库加载以注册 torch.ops._C
    import mcoplib._C
except ImportError:
    pass

class Silu_and_mul_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        # 典型 Llama-2-70B FFN 维度参考
        self.batch_size = config.get("batch_size", 4096)
        self.hidden_size = config.get("hidden_size", 11008)
        # 输入维度是输出维度的2倍 (Gate + Value)
        self.input_size = self.hidden_size * 2

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"({self.batch_size} {self.input_size}) -> ({self.batch_size} {self.hidden_size})")
        
        # 元素总数以输出为基准
        total_out_elements = self.batch_size * self.hidden_size
        state.add_element_count(total_out_elements)
        
        # 估算显存读写 (Bytes)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        # Reads: Input (2 * hidden_size)
        reads = (self.batch_size * self.input_size) * element_size
        # Writes: Output (1 * hidden_size)
        writes = (self.batch_size * self.hidden_size) * element_size
        
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # Input shape: [batch_size, 2 * d]
            input_tensor = torch.randn(self.batch_size, self.input_size, dtype=self.dtype, device=dev)
            
            # Output shape: [batch_size, d]
            out = torch.empty(self.batch_size, self.hidden_size, dtype=self.dtype, device=dev)
            
        # 注意：根据接口原型，参数顺序为 (out, input)
        return self.make_launcher(dev_id, torch.ops._C.silu_and_mul, out, input_tensor)

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        # 使用较小的 Shape 进行验证
        N = 128
        H = 256
        input_tensor = torch.randn(N, H * 2, dtype=self.dtype, device=dev)
        out_op = torch.empty(N, H, dtype=self.dtype, device=dev)
        
        # 运行算子
        torch.ops._C.silu_and_mul(out_op, input_tensor)
        
        # PyTorch 原生实现验证
        # 切分 input: 前半部分是 x (gate), 后半部分是 y (value)
        x, y = input_tensor.chunk(2, dim=-1)
        # SiLU(x) * y
        out_ref = F.silu(x) * y
        
        return self.check_diff(out_op, out_ref)