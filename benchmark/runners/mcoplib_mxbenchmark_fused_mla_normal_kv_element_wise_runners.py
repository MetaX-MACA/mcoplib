import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib.sgl_kernel
except ImportError:
    pass


def _ref_fused_mla_normal_kv_element_wise(kv, latent_cache, num_local_heads,
                                          kv_lora_rank, qk_nope_head_dim,
                                          qk_rope_head_dim, v_head_dim):
    """参考实现：纯 PyTorch 复现算子的拆分/重排语义，供精度验证对比。

    输入:
        kv:           [q_len, num_local_heads * (qk_nope_head_dim + v_head_dim)]
        latent_cache: [q_len, kv_lora_rank + qk_rope_head_dim]
    输出:
        k_ref: [q_len, num_local_heads, qk_nope_head_dim + qk_rope_head_dim]
        v_ref: [q_len, num_local_heads, v_head_dim]
    """
    q_len = kv.size(0)
    # ① k_nope: kv 每头前 qk_nope_head_dim 维
    kv_view = kv.view(q_len, num_local_heads, qk_nope_head_dim + v_head_dim)
    k_nope = kv_view[..., :qk_nope_head_dim]
    # ② v: kv 每头后 v_head_dim 维
    v_ref = kv_view[..., qk_nope_head_dim:]
    # ③ k_pe: latent_cache 后 qk_rope_head_dim 维, 广播复制到所有头
    k_pe = latent_cache[:, kv_lora_rank:].unsqueeze(1)  # [q_len, 1, rope]
    k_ref = torch.cat(
        [k_nope, k_pe.expand(-1, num_local_heads, -1)], dim=-1
    )
    return k_ref.contiguous(), v_ref.contiguous()


class Fused_mla_normal_kv_element_wise_runner(OpBenchmarkBase):
    # kernel 模板特化支持的全部头数档位, 精度验证需逐一覆盖
    SUPPORTED_NUM_LOCAL_HEADS = [8, 16, 32, 64, 128]

    def __init__(self, name, config):
        super().__init__(name, config)
        self.q_len = config.get("q_len", 4096)
        self.num_local_heads = config.get("num_local_heads", 128)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.qk_nope_head_dim = config.get("qk_nope_head_dim", 128)
        self.qk_rope_head_dim = config.get("qk_rope_head_dim", 64)
        self.v_head_dim = config.get("v_head_dim", 128)
        # kernel 中硬编码 scalar_t = nv_bfloat16, 仅支持 bf16
        if self.dtype != torch.bfloat16:
            self.dtype = torch.bfloat16

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", "bfloat16")
        shape_str = (
            f"({self.q_len} {self.num_local_heads} {self.kv_lora_rank} "
            f"{self.qk_nope_head_dim} {self.qk_rope_head_dim} {self.v_head_dim})"
        )
        state.add_summary("Shape", shape_str)
        # ---- 真实 DRAM 数据流量 (bf16 = 2 字节/元素) ----
        # 读侧:
        #   kv 全量:        q_len * num_local_heads * (qk_nope_head_dim + v_head_dim)
        #   latent_cache:   仅后 qk_rope_head_dim 维被读取, 且所有头共享同一份
        #                   源数据(实际由 L2 缓存命中), DRAM 只读一次
        size_kv_read = self.q_len * self.num_local_heads * (self.qk_nope_head_dim + self.v_head_dim)
        size_k_pe_read = self.q_len * self.qk_rope_head_dim
        # 写侧:
        #   k 全量(nope + pe): q_len * num_local_heads * (qk_nope_head_dim + qk_rope_head_dim)
        #   v 全量:            q_len * num_local_heads * v_head_dim
        size_k_write = self.q_len * self.num_local_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim)
        size_v_write = self.q_len * self.num_local_heads * self.v_head_dim
        state.add_global_memory_reads(int((size_kv_read + size_k_pe_read) * 2))
        state.add_global_memory_writes(int((size_k_write + size_v_write) * 2))
        state.add_element_count(int(size_k_write + size_v_write))

    def _prepare(self, dev_id):
        dev = torch.device(f'cuda:{dev_id}')
        kv = torch.randn(
            (self.q_len, self.num_local_heads * (self.qk_nope_head_dim + self.v_head_dim)),
            dtype=torch.bfloat16, device=dev)
        latent_cache = torch.randn(
            (self.q_len, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=torch.bfloat16, device=dev)
        k = torch.empty(
            (self.q_len, self.num_local_heads, self.qk_nope_head_dim + self.qk_rope_head_dim),
            dtype=torch.bfloat16, device=dev)
        v = torch.empty(
            (self.q_len, self.num_local_heads, self.v_head_dim),
            dtype=torch.bfloat16, device=dev)
        return kv, latent_cache, k, v

    def prepare_and_get_launcher(self, dev_id, tc_s):
        kv, latent_cache, k, v = self._prepare(dev_id)

        def launcher(launch):
            stream_ptr = launch.get_stream()
            stream = self.as_torch_stream(stream_ptr, dev_id)
            with torch.cuda.stream(stream):
                torch.ops.sgl_kernel.fused_mla_normal_kv_element_wise(
                    kv, latent_cache, k, v,
                    self.q_len, self.num_local_heads, self.kv_lora_rank,
                    self.qk_nope_head_dim, self.qk_rope_head_dim, self.v_head_dim
                )
        return launcher

    def run_verification(self, dev_id):
        dev = torch.device(f'cuda:{dev_id}')
        # 遍历 kernel 全部头数模板特化, 逐一验证
        v_q_len = 4
        v_rank = 512
        v_nope = 128
        v_rope = 64
        v_vdim = 128
        all_passed = True
        max_diff = 0.0
        for v_heads in self.SUPPORTED_NUM_LOCAL_HEADS:
            kv = torch.randn(
                (v_q_len, v_heads * (v_nope + v_vdim)), dtype=torch.bfloat16, device=dev)
            latent_cache = torch.randn(
                (v_q_len, v_rank + v_rope), dtype=torch.bfloat16, device=dev)
            k = torch.empty(
                (v_q_len, v_heads, v_nope + v_rope), dtype=torch.bfloat16, device=dev)
            v = torch.empty(
                (v_q_len, v_heads, v_vdim), dtype=torch.bfloat16, device=dev)
            torch.ops.sgl_kernel.fused_mla_normal_kv_element_wise(
                kv, latent_cache, k, v,
                v_q_len, v_heads, v_rank, v_nope, v_rope, v_vdim
            )
            torch.cuda.synchronize()
            k_ref, v_ref = _ref_fused_mla_normal_kv_element_wise(
                kv, latent_cache, v_heads, v_rank, v_nope, v_rope, v_vdim
            )
            # 纯拷贝算子: 无数学计算, 理想情况逐位相等, 用 torch.equal 零容差判定
            k_match = torch.equal(k, k_ref)
            v_match = torch.equal(v, v_ref)
            if not (k_match and v_match):
                all_passed = False
                max_diff = 1.0
        return all_passed, max_diff
