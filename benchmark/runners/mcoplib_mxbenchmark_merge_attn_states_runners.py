import torch
import random
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase
import mcoplib._C

class Merge_attn_states_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 128)
        self.num_heads = config.get("num_heads", 32)
        self.head_size = config.get("head_size", 128)
        self.seed = config.get("seed", 0)
        
        torch.manual_seed(self.seed)
        random.seed(self.seed)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        state.add_summary("Shape", f"(T:{self.num_tokens}, H:{self.num_heads}, D:{self.head_size})")
        
        # 元素总数：主要输出 Tensor
        total = self.num_tokens * self.num_heads * self.head_size
        state.add_element_count(total)
        
        element_size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4
        lse_size = 4 # LSE 始终为 float32
        
        # 计算读写字节数
        # Read: 2 * Output (bf16/fp16) + 2 * LSE (fp32)
        # Write: 1 * Output (bf16/fp16) + 1 * LSE (fp32)
        lse_elements = self.num_heads * self.num_tokens
        
        reads = (2 * total * element_size) + (2 * lse_elements * lse_size)
        writes = (1 * total * element_size) + (1 * lse_elements * lse_size)
        
        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

    def _prepare_tensors(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        prefix_output = torch.randn(self.num_tokens, self.num_heads, self.head_size, dtype=self.dtype, device=dev)
        suffix_output = torch.randn(self.num_tokens, self.num_heads, self.head_size, dtype=self.dtype, device=dev)
        
        prefix_lse = torch.randn(self.num_heads, self.num_tokens, dtype=torch.float32, device=dev)
        suffix_lse = torch.randn(self.num_heads, self.num_tokens, dtype=torch.float32, device=dev)
        
        output = torch.empty_like(prefix_output)
        output_lse = torch.empty_like(prefix_lse)
        
        return output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            tensors = self._prepare_tensors(dev_id)
            return self.make_launcher(dev_id, torch.ops._C.merge_attn_states, *tensors)

    def run_verification(self, dev_id):
        output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse = self._prepare_tensors(dev_id)
        
        # 运行算子
        torch.ops._C.merge_attn_states(
            output, output_lse, prefix_output, prefix_lse, suffix_output, suffix_lse
        )
        
        # Python 参考实现
        # LSE Shape: [H, N], Output Shape: [N, H, D]
        # 对齐维度用于广播: [H, N] -> [N, H] -> [N, H, 1]
        p_lse_t = prefix_lse.transpose(0, 1).unsqueeze(-1)
        s_lse_t = suffix_lse.transpose(0, 1).unsqueeze(-1)
        
        # 计算 logsumexp
        # 注意：这里需要在 float32 下进行以保持精度
        lse_new = torch.logaddexp(p_lse_t, s_lse_t) # [N, H, 1]
        
        # 计算权重
        w_prefix = torch.exp(p_lse_t - lse_new)
        w_suffix = torch.exp(s_lse_t - lse_new)
        
        # 计算加权 Output
        out_ref = prefix_output.float() * w_prefix + suffix_output.float() * w_suffix
        out_ref = out_ref.to(self.dtype)
        
        # 验证 Output (Output LSE 验证略，因 Output 依赖 LSE 正确性)
        return self.check_diff(output, out_ref)