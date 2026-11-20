import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


class Fused_mla_RMS_rotary_emb_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.q_len = config.get("q_len", 4096)
        self.num_local_heads = config.get("num_local_heads", 128)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        self.qk_nope_head_dim = config.get("qk_nope_head_dim", 128)
        self.max_pos = config.get("max_position_embeddings", 4096)
        if self.dtype != torch.bfloat16:
            self.dtype = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16")
        shape_str = f"({self.q_len} {self.num_local_heads} {self.kv_lora_rank} {self.qk_nope_head_dim} {self.qk_rope_head_dim})"
        state.add_summary("Shape", shape_str)
        # 读：q 仅 rope 部分（nope 部分本算子不触碰全局内存）+ latent_cache 全量 + norm_weight + positions + cos_sin
        size_q_rope = self.q_len * self.num_local_heads * self.qk_rope_head_dim
        size_latent = self.q_len * (self.kv_lora_rank + self.qk_rope_head_dim)
        size_norm = self.kv_lora_rank
        size_pos = self.q_len * 8
        size_cos_sin = self.q_len * self.qk_rope_head_dim * 4
        # 写：q rope 部分（原地）+ latent_cache 全量（原地）+ kv_a
        size_out = size_q_rope + size_latent + self.q_len * self.kv_lora_rank
        state.add_global_memory_reads(int((size_q_rope + size_latent + size_norm) * 2 + size_pos + size_cos_sin))
        state.add_global_memory_writes(int(size_out * 2))
        state.add_element_count(int(size_out))

    def _prepare(self, dev, q_len, heads=None):
        if heads is None:
            heads = self.num_local_heads
        q = torch.randn((q_len, heads, self.qk_nope_head_dim + self.qk_rope_head_dim),
                        dtype=torch.bfloat16, device=dev)
        latent_cache = torch.randn((q_len, self.kv_lora_rank + self.qk_rope_head_dim),
                                   dtype=torch.bfloat16, device=dev)
        norm_weight = torch.randn((self.kv_lora_rank,), dtype=torch.bfloat16, device=dev)
        cos_sin_cache = torch.randn((self.max_pos, self.qk_rope_head_dim), dtype=torch.float32, device=dev)
        positions = torch.randint(0, self.max_pos, (q_len,), dtype=torch.int64, device=dev)
        kv_a = torch.empty((q_len, self.kv_lora_rank), dtype=torch.bfloat16, device=dev)
        return q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = torch.device(f'cuda:{dev_id}')
        q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a = self._prepare(dev, self.q_len)
        def launcher(launch):
            stream_ptr = launch.get_stream()
            stream = self.as_torch_stream(stream_ptr, dev_id)
            with torch.cuda.stream(stream):
                torch.ops.sgl_kernel.fused_mla_RMS_rotary_emb(
                    q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a,
                    self.q_len, self.num_local_heads, self.kv_lora_rank,
                    self.qk_rope_head_dim, self.qk_nope_head_dim
                )
        return launcher

    def run_verification(self, dev_id):
        dev = torch.device(f'cuda:{dev_id}')
        v_rank = 512
        v_nope = 128
        v_rope = 64
        # 覆盖：常规形状、q_len=3（branch 二尾块保护 j*4+k < q_len）、heads=8（lane 级保护）
        v_cases = [(4, 128), (3, 128), (4, 8), (3, 8)]
        all_pass = True
        max_diff = 0.0
        for v_q_len, v_heads in v_cases:
            q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a = self._prepare(dev, v_q_len, v_heads)
            # 保存原始输入用于参考计算
            q_orig = q.clone()
            latent_orig = latent_cache.clone()
            torch.ops.sgl_kernel.fused_mla_RMS_rotary_emb(
                q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a,
                v_q_len, v_heads, v_rank, v_rope, v_nope
            )
            torch.cuda.synchronize()
            # ===== 参考实现 =====
            q_f = q_orig.float()
            latent_f = latent_orig.float()
            norm_w_f = norm_weight.float()
            eps = 1e-6
            half_dim = v_rope // 2
            cos_sin_batch = cos_sin_cache[positions]
            cos = cos_sin_batch[:, :half_dim]
            sin = cos_sin_batch[:, half_dim:]

            def apply_rope_interleaved(x_in, c, s):
                x_pairs = x_in.view(*x_in.shape[:-1], half_dim, 2)
                x0 = x_pairs[..., 0]
                x1 = x_pairs[..., 1]
                while c.ndim < x0.ndim:
                    c = c.unsqueeze(1)
                    s = s.unsqueeze(1)
                res0 = x0 * c - x1 * s
                res1 = x1 * c + x0 * s
                return torch.stack([res0, res1], dim=-1).flatten(-2)

            # 1) q 的 rope 部分原地做 rotary，nope 部分不变
            q_ref = q_f.clone()
            q_rope_ref = apply_rope_interleaved(q_f[:, :, v_nope:], cos, sin)
            q_ref[:, :, v_nope:] = q_rope_ref
            # 2) latent_cache 的 rank 部分做 RMSNorm，结果写回 latent_cache 并拷贝到 kv_a
            latent_rank = latent_f[:, :v_rank]
            mean_sq = latent_rank.pow(2).mean(dim=-1, keepdim=True)
            normed_latent = latent_rank * torch.rsqrt(mean_sq + eps) * norm_w_f
            kv_a_ref = normed_latent
            # 3) latent_cache 的 rope 部分原地做 rotary
            latent_rope_ref = apply_rope_interleaved(latent_f[:, v_rank:], cos, sin)
            latent_ref = torch.cat([normed_latent, latent_rope_ref], dim=-1)
            # ===== 比对 =====
            q_ref = q_ref.to(torch.bfloat16)
            latent_ref = latent_ref.to(torch.bfloat16)
            kv_a_ref = kv_a_ref.to(torch.bfloat16)
            pass_q, diff_q = self.check_diff(q, q_ref, threshold=0.98)
            pass_latent, diff_latent = self.check_diff(latent_cache, latent_ref, threshold=0.98)
            pass_kv, diff_kv = self.check_diff(kv_a, kv_a_ref, threshold=0.99)
            all_pass = all_pass and pass_q and pass_latent and pass_kv
            max_diff = max(max_diff, diff_q, diff_latent, diff_kv)
        return all_pass, max_diff
