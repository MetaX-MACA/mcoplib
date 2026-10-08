import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 16)
        self.num_heads = config.get("num_heads", 4)
        self.head_dim = config.get("head_dim", 192)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.pe_dim = config.get("pe_dim", 64)
        self.block_size = config.get("block_size", 8)
        self.seed = config.get("seed", None)

    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", f"tokens={self.num_tokens} heads={self.num_heads} dim={self.head_dim}")

    def prepare_and_get_launcher(self, dev_id, tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        with torch.cuda.stream(tc_s):
            dev = f"cuda:{dev_id}"

            q = torch.randn(self.num_tokens,self.num_heads,192,dtype=torch.bfloat16,device=dev)
            k_nope = torch.randn(self.num_tokens,self.num_heads,128,dtype=torch.bfloat16,device=dev)
            k_pe = torch.randn(self.num_tokens,64,dtype=torch.bfloat16,device=dev)
            kv_c = torch.randn(self.num_tokens,512,dtype=torch.bfloat16,device=dev)
            v = torch.randn(self.num_tokens,self.num_heads,128,dtype=torch.bfloat16,device=dev)

            q_fp8 = torch.empty_like(q,dtype=torch.float8_e4m3fn)
            k_fp8 = torch.empty_like(q,dtype=torch.float8_e4m3fn)
            v_fp8 = torch.empty(self.num_tokens,self.num_heads,128,dtype=torch.float8_e4m3fn,device=dev)

            num_cache_blocks = max(64, (self.num_tokens + self.block_size - 1) // self.block_size)
            cache = torch.zeros(num_cache_blocks,self.block_size,576,dtype=torch.float8_e4m3fn,device=dev)

            slots = torch.arange(self.num_tokens,dtype=torch.int64,device=dev)
            positions = torch.arange(self.num_tokens,dtype=torch.int64,device=dev)

            rope_cache = torch.randn(32,64,dtype=torch.float32,device=dev)

            one = torch.ones(1,dtype=torch.float32,device=dev)

        return self.make_launcher(
            dev_id,
            torch.ops._C.fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert,
            q,k_nope,k_pe,kv_c,v,
            q_fp8,k_fp8,v_fp8,
            cache,slots,
            one,one,one,one,
            self.block_size,
            positions,
            rope_cache,
        )


    def run_verification(self, dev_id):
        dev = f"cuda:{dev_id}"

        num_tokens = 3

        q = torch.randn(num_tokens,self.num_heads,192,dtype=torch.bfloat16,device=dev)
        k_nope = torch.randn(num_tokens,self.num_heads,128,dtype=torch.bfloat16,device=dev)
        k_pe = torch.randn(num_tokens,64,dtype=torch.bfloat16,device=dev)
        kv_c = torch.randn(num_tokens,512,dtype=torch.bfloat16,device=dev)
        v = torch.randn(num_tokens,self.num_heads,128,dtype=torch.bfloat16,device=dev)

        q_fp8 = torch.empty(num_tokens,self.num_heads,192,dtype=torch.float8_e4m3fn,device=dev)
        k_fp8 = torch.empty_like(q_fp8)
        v_fp8 = torch.empty(num_tokens,self.num_heads,128,dtype=torch.float8_e4m3fn,device=dev)

        cache = torch.zeros(8,self.block_size,576,dtype=torch.float8_e4m3fn,device=dev)

        slots = torch.tensor([0,3,9],dtype=torch.int64,device=dev)
        positions = torch.tensor([1,7,13],dtype=torch.int64,device=dev)

        rope_cache = torch.randn(32,64,dtype=torch.float32,device=dev)

        one = torch.ones(1,dtype=torch.float32,device=dev)

        torch.ops._C.fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert(
            q,k_nope,k_pe,kv_c,v,
            q_fp8,k_fp8,v_fp8,
            cache,slots,
            one,one,one,one,
            self.block_size,
            positions,
            rope_cache,
        )

        ok = (
            torch.isfinite(q_fp8.float()).all().item()
            and torch.isfinite(k_fp8.float()).all().item()
            and torch.isfinite(v_fp8.float()).all().item()
        )

        return ok,0.0