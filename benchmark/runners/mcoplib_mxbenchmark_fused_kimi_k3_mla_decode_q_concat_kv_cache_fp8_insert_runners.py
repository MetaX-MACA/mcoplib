import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.num_tokens = config.get("num_tokens", 16)
        self.num_heads = config.get("num_heads", 4)
        self.ql_nope_dim = config.get("ql_nope_dim", 512)
        self.q_pe_dim = config.get("q_pe_dim", 64)
        self.kv_lora_rank = config.get("kv_lora_rank", 512)
        self.pe_dim = config.get("pe_dim", 64)
        self.block_size = config.get("block_size", 8)


    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("Shape", f"tokens={self.num_tokens}, heads={self.num_heads}")

        q_elements = self.num_tokens * self.num_heads * (self.ql_nope_dim + self.q_pe_dim)
        cache_elements = self.num_tokens * (self.kv_lora_rank + self.pe_dim)

        total_elements = q_elements + cache_elements

        state.add_element_count(total_elements)

        state.add_global_memory_reads(total_elements * 2)
        state.add_global_memory_writes(total_elements)


    def make_inputs(self, dev):
        ql_nope = torch.randn(self.num_tokens,self.num_heads,self.ql_nope_dim,dtype=torch.bfloat16,device=dev)

        q_pe = torch.randn(self.num_tokens,self.num_heads,self.q_pe_dim,dtype=torch.bfloat16,device=dev)

        kv_c = torch.randn(self.num_tokens,self.kv_lora_rank,dtype=torch.bfloat16,device=dev)

        k_pe = torch.randn(self.num_tokens,self.pe_dim,dtype=torch.bfloat16,device=dev)

        mqa_q = torch.empty(self.num_tokens,self.num_heads,self.ql_nope_dim+self.q_pe_dim,dtype=torch.float8_e4m3fn,device=dev)

        num_blocks = max(2,(self.num_tokens+self.block_size-1)//self.block_size)

        k_cache = torch.zeros(num_blocks,self.block_size,self.kv_lora_rank+self.pe_dim,dtype=torch.float8_e4m3fn,device=dev)

        slots = torch.arange(self.num_tokens,dtype=torch.int64,device=dev)

        scale = torch.ones(1,dtype=torch.float32,device=dev)

        return ql_nope,q_pe,kv_c,k_pe,mqa_q,k_cache,slots,scale


    def prepare_and_get_launcher(self, dev_id, tc_s):
        dev=f"cuda:{dev_id}"

        with torch.cuda.stream(tc_s):
            inputs=self.make_inputs(dev)

        return self.make_launcher(
            dev_id,
            torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert,
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            inputs[5],
            inputs[6],
            inputs[7],
            inputs[7],
            self.block_size,
        )


    def run_verification(self, dev_id):
        dev=f"cuda:{dev_id}"

        ql_nope,q_pe,kv_c,k_pe,mqa_q,k_cache,slots,scale=self.make_inputs(dev)

        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert(
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            mqa_q,
            k_cache,
            slots,
            scale,
            scale,
            self.block_size,
        )

        q_ref=torch.cat([ql_nope,q_pe],dim=-1).to(torch.float8_e4m3fn)

        cache_ref=torch.cat([kv_c,k_pe],dim=-1).to(torch.float8_e4m3fn)

        cache_out=k_cache.reshape(-1,k_cache.shape[-1]).index_select(0,slots)

        q_cos=F.cosine_similarity(mqa_q.float().flatten(),q_ref.float().flatten(),dim=0).item()

        cache_cos=F.cosine_similarity(cache_out.float().flatten(),cache_ref.float().flatten(),dim=0).item()

        cos=min(q_cos,cache_cos)

        passed=cos>=0.99

        return passed, cos