import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

# 1. 动态获取算子
func_name = "cutlass_group_gemm_supported"
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

class Cutlass_group_gemm_supported_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.capability = config.get("cuda_device_capability", 90)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", f"Cap={self.capability}") 
        state.add_element_count(1)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if _kernel_func is None:
            raise RuntimeError(f"算子 {func_name} 未找到")

        # 这是一个简单的 Check 函数，无需复杂的 Tensor 准备
        with torch.cuda.stream(tc_s):
            pass
            
        return self.make_launcher(dev_id, _kernel_func, self.capability)

    def run_verification(self, dev_id):
        if _kernel_func is None:
            return False, 1.0

        # 调用 C++ 算子
        ret = _kernel_func(self.capability)
        
        # 验证逻辑：
        # 根据提供的 scaled_mm_entry.cu，该函数目前在 C++ 中硬编码返回 false
        # 或者根据文档描述，非 SM100/SM90+HighCUDA 环境通常不支持
        expected = False 
        
        # 如果环境确实支持（例如 SM90 + CUDA 12.3+），此处逻辑可能需要调整为动态判断
        # 但基于提供的源码 return false，此处默认期望为 False
        if ret == expected:
            return True, 0.0
        else:
            print(f"Warning: Expected {expected} but got {ret}")
            return False, 1.0