import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 1. 动态获取算子
func_name = "cutlass_scaled_mm_supports_fp8"
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

class Cutlass_scaled_mm_supports_fp8_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        # 该参数用于模拟传入 CUDA Capability，例如 80 (A100), 90 (H100)
        self.capability = config.get("cuda_device_capability", 90)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        
        # 1. 明确把 Shape 设置为参数
        state.add_summary("Shape", f"Cap={self.capability}")
        # 2. 【新增】明确把 dtype 设置为空字符串
        # 这样能确保生成的 Key 里 dtype 这一项是 ""，能匹配上 CSV 里的空列
        state.add_summary("dtype", "") 
        state.add_element_count(1)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if _kernel_func is None:
            raise RuntimeError(f"算子 {func_name} 未找到")

        # 为了保持框架兼容性，依然在 stream 上下文中，尽管该函数可能只在 CPU 运行
        with torch.cuda.stream(tc_s):
            pass
            
        return self.make_launcher(dev_id, _kernel_func, self.capability)

    def run_verification(self, dev_id):
        if _kernel_func is None:
            return False, 1.0

        # 调用算子
        ret = _kernel_func(self.capability)
        
        # 验证逻辑：
        # 根据 C++ 源码，该函数目前被硬编码为 return false
        # 因此我们验证返回值是否为 False (或者仅验证能否成功调用)
        
        expected = False # 源码中是 return false
        
        # 只要能跑通且返回布尔值即视为 Pass
        passed = (ret == expected)
        
        if passed:
            return True, 0.0
        else:
            # 如果源码逻辑改变导致返回 True，这里会提示
            print(f"Warning: Expected {expected} but got {ret}")
            return False, 1.0