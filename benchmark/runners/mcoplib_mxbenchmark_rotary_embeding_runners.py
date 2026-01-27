import torch
import sys
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.op as op
except ImportError:
    op = None

class Rotary_embedding_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        # 从配置中读取参数，如果不存在则使用默认值
        self.batch_size_list = config.get("batch_size_list", [128, 64, 256, 32])
        self.q_head_num = config.get("q_head_num", 32)
        self.kv_head_num = config.get("kv_head_num", 8)
        self.head_size = config.get("head_size", 128)
        self.max_seq_len = config.get("max_seq_len", 2048)
        self.rope_offset = config.get("rope_offset", 0)
        
        # 计算衍生参数
        self.num_seqs = len(self.batch_size_list)
        self.batch_size = sum(self.batch_size_list)
        self.total_head_num = self.q_head_num + 2 * self.kv_head_num

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", self.config.get("dtype", str(self.dtype)))
        
        # 记录形状信息
        shape_str = f"(Tokens:{self.batch_size} Heads:{self.total_head_num} Dim:{self.head_size})"
        state.add_summary("Shape", shape_str)
        
        # 计算元素总数 (packed_qkv的大小)
        qkv_elems = self.batch_size * self.total_head_num * self.head_size
        state.add_element_count(qkv_elems)
        
        # 计算显存读写量
        # 读: packed_qkv + cos + sin + indices (忽略 indices 因为较小)
        # 写: out (大小等于 packed_qkv)
        element_size = 2 if self.dtype == torch.float16 or self.dtype == torch.bfloat16 else 4
        
        # 注意：cos/sin 通常是 float32，但在算子内部可能被转换。这里按配置的 dtype 估算带宽
        rw_bytes = (qkv_elems * 2) * element_size # Read input + Write output
        # 加上 cos/sin 的读取 (假设 broadcast 或对齐读取)
        cos_sin_elems = self.max_seq_len * self.head_size * 2
        rw_bytes += cos_sin_elems * 4 # cos/sin 通常是 float32
        
        state.add_global_memory_reads(qkv_elems * element_size + cos_sin_elems * 4) 
        state.add_global_memory_writes(qkv_elems * element_size)

    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev = f'cuda:{dev_id}'
            
            # 创建输入 Tensor
            packed_qkv = torch.randn(self.batch_size, self.total_head_num, self.head_size, dtype=self.dtype, device=dev)
            
            # 原始脚本中 cos/sin 是 float32
            cos = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
            sin = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
            
            # 创建输出 Tensor
            out = torch.empty_like(packed_qkv)
            
            # 创建辅助 Tensor
            q_len = torch.tensor(self.batch_size_list, dtype=torch.int32, device=dev)
            accum_q_lens = torch.tensor([0] + self.batch_size_list, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32)
            cache_lens = torch.zeros(self.num_seqs, dtype=torch.int32, device=dev)
            
        # 返回 launcher lambda
        return self.make_launcher(dev_id, op.rotary_embedding, 
                                  packed_qkv, 
                                  q_len, 
                                  accum_q_lens, 
                                  cache_lens, 
                                  cos, 
                                  sin, 
                                  out, 
                                  self.q_head_num, 
                                  self.kv_head_num, 
                                  self.rope_offset)

    def _torch_rope_impl(self, packed_qkv, cos, sin, seq_lens):
        """
        PyTorch 参考实现。
        注意：这里假设 packed_qkv 的布局为 [Q, K, V]，其中 Q=q_head_num, K=kv_head_num, V=kv_head_num
        """
        out = torch.empty_like(packed_qkv)
        start_token_idx = 0
        
        # 辅助函数：旋转逻辑
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        for i, seq_len in enumerate(seq_lens):
            end_token_idx = start_token_idx + seq_len
            
            # 提取当前序列的 token
            # shape: [seq_len, total_heads, head_dim]
            current_tokens = packed_qkv[start_token_idx:end_token_idx]
            
            # 切分 Q, K, V
            # 假设前 q_head_num 是 Q，接下来 kv_head_num 是 K，最后是 V
            q = current_tokens[:, :self.q_head_num, :].float()
            k = current_tokens[:, self.q_head_num:self.q_head_num + self.kv_head_num, :].float()
            v = current_tokens[:, self.q_head_num + self.kv_head_num:, :].float()
            
            # 获取对应的 cos/sin (假设从 0 开始到 seq_len)
            # shape: [seq_len, head_dim] -> [seq_len, 1, head_dim] 用于广播
            cur_cos = cos[:seq_len, :].unsqueeze(1)
            cur_sin = sin[:seq_len, :].unsqueeze(1)
            
            # 应用 RoPE 到 Q 和 K
            q_out = (q * cur_cos) + (rotate_half(q) * cur_sin)
            k_out = (k * cur_cos) + (rotate_half(k) * cur_sin)
            
            # 拼接回原来的形状 (Q_out, K_out, V)
            # 注意需要转回原来的 dtype
            rotated_part = torch.cat([q_out, k_out, v], dim=1).to(packed_qkv.dtype)
            
            out[start_token_idx:end_token_idx] = rotated_part
            start_token_idx = end_token_idx
            
        return out

    def run_verification(self, dev_id):
        dev = f'cuda:{dev_id}'
        
        # 准备数据
        packed_qkv = torch.randn(self.batch_size, self.total_head_num, self.head_size, dtype=self.dtype, device=dev)
        cos = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
        sin = torch.randn(self.max_seq_len, self.head_size, dtype=torch.float32, device=dev)
        out_op = torch.empty_like(packed_qkv)
        
        q_len = torch.tensor(self.batch_size_list, dtype=torch.int32, device=dev)
        accum_q_lens = torch.tensor([0] + self.batch_size_list, dtype=torch.int32, device=dev).cumsum(0, dtype=torch.int32)
        cache_lens = torch.zeros(self.num_seqs, dtype=torch.int32, device=dev)

        # 运行算子
        op.rotary_embedding(
            packed_qkv,
            q_len,
            accum_q_lens,
            cache_lens,
            cos,
            sin,
            out_op,
            self.q_head_num,
            self.kv_head_num,
            self.rope_offset
        )
        
        # 运行参考实现
        # 注意：这里传入 clone 的输入以防算子原地修改（虽然这个算子有 out 参数）
        out_ref = self._torch_rope_impl(packed_qkv.clone(), cos, sin, self.batch_size_list)
        
        return self.check_diff(out_op, out_ref)