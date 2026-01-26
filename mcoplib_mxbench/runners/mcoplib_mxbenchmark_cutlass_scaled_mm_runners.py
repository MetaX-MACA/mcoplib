import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 1. 动态获取算子函数
func_name = "cutlass_scaled_mm"
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

class Cutlass_scaled_mm_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.m = config.get("m", 4096)
        self.n = config.get("n", 4096)
        self.k = config.get("k", 4096)
        
        # 显式声明该算子固有的精度特性，覆盖 config 中的通用设置（如果需要）
        # 或者直接使用 config["dtype"] 用于显示
        self.input_dtype = torch.int8
        self.output_dtype = torch.bfloat16
        
        # 约束检查
        assert self.k % 16 == 0, "K dimension must be a multiple of 16 for alignment"
        assert self.n % 16 == 0, "N dimension must be a multiple of 16 for alignment"
        
        self.a = None
        self.b = None
        self.c = None
        self.a_scales = None
        self.b_scales = None
        self.bias = None

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        # 在 Shape 中也可以补充说明精度流，或者依赖基类处理 dtype 列
        state.add_summary("Shape", f"M={self.m} N={self.n} K={self.k}")
        
        # 计算显存读写 (Global Memory)
        size_a = self.m * self.k * 1  # int8 = 1 byte
        size_b = self.k * self.n * 1  # int8 = 1 byte
        size_c = self.m * self.n * 2  # bf16 = 2 bytes
        
        size_scales = (self.m * 4) + (self.n * 4) # fp32 = 4 bytes
        size_bias = self.n * 2        # bf16 = 2 bytes
        
        state.add_global_memory_reads(size_a + size_b + size_scales + size_bias)
        state.add_global_memory_writes(size_c)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if _kernel_func is None:
            raise RuntimeError(f"算子 {func_name} 未找到，请检查 mcoplib 安装。")

        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 严格按照文档要求的类型构造数据
            self.a = torch.randint(-127, 127, (self.m, self.k), dtype=self.input_dtype, device=dev)
            
            # Input B 需处理为 Column Major
            b_storage = torch.randint(-127, 127, (self.n, self.k), dtype=self.input_dtype, device=dev)
            self.b = b_storage.t() 
            
            self.c = torch.empty((self.m, self.n), dtype=self.output_dtype, device=dev)
            
            self.a_scales = torch.randn(self.m, 1, dtype=torch.float32, device=dev)
            self.b_scales = torch.randn(1, self.n, dtype=torch.float32, device=dev)
            self.bias = torch.randn(self.n, dtype=self.output_dtype, device=dev)
            
        return self.make_launcher(dev_id, _kernel_func, 
                                  self.c, self.a, self.b, 
                                  self.a_scales, self.b_scales, self.bias)

    def run_verification(self, dev_id):
        if _kernel_func is None:
            return False, 1.0

        dev = f'cuda:{dev_id}'
        
        # 验证数据
        a = torch.randint(-10, 10, (self.m, self.k), dtype=self.input_dtype, device=dev)
        b_storage = torch.randint(-10, 10, (self.n, self.k), dtype=self.input_dtype, device=dev)
        b = b_storage.t()
        
        a_scales = torch.rand(self.m, 1, dtype=torch.float32, device=dev)
        b_scales = torch.rand(1, self.n, dtype=torch.float32, device=dev)
        bias = torch.randn(self.n, dtype=self.output_dtype, device=dev)
        
        c_op = torch.empty((self.m, self.n), dtype=self.output_dtype, device=dev)
        
        _kernel_func(c_op, a, b, a_scales, b_scales, bias)
        
        # Reference 计算
        a_f = a.float()
        b_f = b.float()
        res = torch.matmul(a_f, b_f)
        res = res * a_scales * b_scales
        res = res + bias
        out_ref = res.to(self.output_dtype)
        
        return self.check_diff(c_op, out_ref, threshold=0.99)