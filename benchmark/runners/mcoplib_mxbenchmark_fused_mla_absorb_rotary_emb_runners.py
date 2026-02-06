import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass

class Fused_mla_absorb_rotary_emb_runner(OpBenchmarkBase):
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
        shape_str = f"({self.q_len}, {self.num_local_heads}, {self.kv_lora_rank}, {self.qk_nope_head_dim}, {self.qk_rope_head_dim})"
        state.add_summary("Shape", shape_str)
        size_q = self.q_len * self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        size_w_kc = self.num_local_heads * self.qk_nope_head_dim * self.kv_lora_rank
        size_latent = self.q_len * (self.kv_lora_rank + self.qk_rope_head_dim)
        size_norm = self.kv_lora_rank
        size_pos = self.q_len * 8
        size_cos_sin = self.q_len * self.qk_rope_head_dim * 4
        size_out = self.q_len * (self.num_local_heads + 1) * (self.kv_lora_rank + self.qk_rope_head_dim) + self.q_len * self.kv_lora_rank
        state.add_global_memory_reads(int((size_q + size_w_kc + size_latent + size_norm)*2 + size_pos + size_cos_sin))
        state.add_global_memory_writes(int(size_out * 2))
        state.add_element_count(int(size_out))

    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev = torch.device(f'cuda:{dev_id}')
        q = torch.randn((self.q_len, self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim), dtype=torch.bfloat16, device=dev)
        latent_cache = torch.randn((self.q_len, self.kv_lora_rank + self.qk_rope_head_dim), dtype=torch.bfloat16, device=dev)
        w_kc = torch.randn((self.num_local_heads, self.qk_nope_head_dim, self.kv_lora_rank), dtype=torch.bfloat16, device=dev)
        norm_weight = torch.randn((self.kv_lora_rank,), dtype=torch.bfloat16, device=dev)
        cos_sin_cache = torch.randn((self.max_pos, self.qk_rope_head_dim), dtype=torch.float32, device=dev)
        positions = torch.randint(0, self.max_pos, (self.q_len,), dtype=torch.int64, device=dev)
        q_out = torch.empty((self.q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim), dtype=torch.bfloat16, device=dev)
        k_out = torch.empty((self.q_len, 1, self.kv_lora_rank + self.qk_rope_head_dim), dtype=torch.bfloat16, device=dev)
        v_out = torch.empty((self.q_len, 1, self.kv_lora_rank), dtype=torch.bfloat16, device=dev)
        def launcher(launch):
            stream_ptr = launch.get_stream()
            stream = self.as_torch_stream(stream_ptr, dev_id)
            with torch.cuda.stream(stream):
                torch.ops.sgl_kernel.fused_mla_absorb_rotary_emb(
                    q, w_kc, latent_cache, cos_sin_cache, positions, norm_weight,
                    q_out, k_out, v_out,
                    self.q_len, self.num_local_heads, self.kv_lora_rank,
                    self.qk_rope_head_dim, self.qk_nope_head_dim
                )
        return launcher

    def run_verification(self, dev_id):
        dev = torch.device(f'cuda:{dev_id}')
        v_q_len = 4
        v_heads = 128
        v_rank = 512
        v_nope = 128
        v_rope = 64
        q = torch.randn((v_q_len, v_heads, v_nope + v_rope), dtype=torch.bfloat16, device=dev)
        latent_cache = torch.randn((v_q_len, v_rank + v_rope), dtype=torch.bfloat16, device=dev)
        w_kc = torch.randn((v_heads, v_nope, v_rank), dtype=torch.bfloat16, device=dev)
        norm_weight = torch.randn((v_rank,), dtype=torch.bfloat16, device=dev)
        cos_sin_cache = torch.randn((self.max_pos, v_rope), dtype=torch.float32, device=dev)
        positions = torch.randint(0, self.max_pos, (v_q_len,), dtype=torch.int64, device=dev)
        q_out = torch.empty((v_q_len, v_heads, v_rank + v_rope), dtype=torch.bfloat16, device=dev)
        k_out = torch.empty((v_q_len, 1, v_rank + v_rope), dtype=torch.bfloat16, device=dev)
        v_out = torch.empty((v_q_len, 1, v_rank), dtype=torch.bfloat16, device=dev)
        torch.ops.sgl_kernel.fused_mla_absorb_rotary_emb(
            q, w_kc, latent_cache, cos_sin_cache, positions, norm_weight,
            q_out, k_out, v_out,
            v_q_len, v_heads, v_rank, v_rope, v_nope
        )
        q_f = q.float()
        latent_f = latent_cache.float()
        norm_w_f = norm_weight.float()
        cos_sin_f = cos_sin_cache
        pos_f = positions
        latent_rank = latent_f[:, :v_rank]
        latent_rope = latent_f[:, v_rank:]
        eps = 1e-6
        mean_sq = latent_rank.pow(2).mean(dim=-1, keepdim=True)
        rsqrt = torch.rsqrt(mean_sq + eps)
        normed_latent = latent_rank * rsqrt * norm_w_f
        v_ref = normed_latent.unsqueeze(1)
        half_dim = v_rope // 2
        cos_sin_batch = cos_sin_f[pos_f]
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
        k_rope_ref = apply_rope_interleaved(latent_rope, cos, sin)
        k_ref = torch.cat([normed_latent, k_rope_ref], dim=-1).unsqueeze(1)
        q_nope = q_f[:, :, :v_nope]
        q_rope = q_f[:, :, v_nope:]
        q_absorbed = torch.einsum('bhn,hnr->bhr', q_nope, w_kc.float())
        q_rope_ref = apply_rope_interleaved(q_rope, cos, sin)
        q_ref = torch.cat([q_absorbed, q_rope_ref], dim=-1)
        q_ref = q_ref.to(dev).to(torch.bfloat16)
        k_ref = k_ref.to(dev).to(torch.bfloat16)
        v_ref = v_ref.to(dev).to(torch.bfloat16)
        pass_q, diff_q = self.check_diff(q_out, q_ref, threshold=0.98)
        pass_v, diff_v = self.check_diff(v_out, v_ref, threshold=0.99)
        pass_k, diff_k = self.check_diff(k_out, k_ref, threshold=0.98)
        return (pass_q and pass_k and pass_v), max(diff_q, diff_k, diff_v)