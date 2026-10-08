import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_kimi_k3_mla_decode_q_concat_ds_mla_insert_runner(OpBenchmarkBase):
    def __init__(self,name,config):
        super().__init__(name,config)
        self.num_tokens=config.get("num_tokens",16)
        self.num_heads=config.get("num_heads",4)
        self.block_size=config.get("block_size",8)
        self.seed = config.get("seed", None)

    def define_metrics(self,state):
        state.add_summary("Op",self.name)
        state.add_summary("Shape",f"tokens={self.num_tokens} heads={self.num_heads}")
        elements=self.num_tokens*(512+64+self.num_heads*576)
        state.add_element_count(elements)

    def prepare_and_get_launcher(self,dev_id,tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"

            ql_nope=torch.randn(self.num_tokens,self.num_heads,512,dtype=torch.bfloat16,device=dev)
            q_pe=torch.randn(self.num_tokens,self.num_heads,64,dtype=torch.bfloat16,device=dev)
            kv_c=torch.randn(self.num_tokens,512,dtype=torch.bfloat16,device=dev)
            k_pe=torch.randn(self.num_tokens,64,dtype=torch.bfloat16,device=dev)

            mqa_q=torch.empty(self.num_tokens,self.num_heads,576,dtype=torch.bfloat16,device=dev)

            k_cache=torch.zeros(64,self.block_size,656,dtype=torch.uint8,device=dev)

            slots=torch.arange(self.num_tokens,dtype=torch.int64,device=dev)
            positions=torch.arange(self.num_tokens,dtype=torch.int64,device=dev)

            rope_cache=self.make_rope_cache(dev)

        return self.make_launcher(
            dev_id,
            torch.ops._C.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert,
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            mqa_q,
            k_cache,
            slots,
            self.block_size,
            positions,
            rope_cache,
        )


    def make_rope_cache(self,dev):
        inv_freq=1.0/(50000**(torch.arange(0,64,2,dtype=torch.float32,device=dev)/64))
        pos=torch.arange(32,dtype=torch.float32,device=dev)
        freq=torch.outer(pos,inv_freq)
        return torch.cat([freq.cos(),freq.sin()],dim=-1)


    def run_verification(self,dev_id):
        dev=f"cuda:{dev_id}"

        num_tokens=3

        ql_nope=torch.randn(num_tokens,self.num_heads,512,dtype=torch.bfloat16,device=dev)
        q_pe=torch.randn(num_tokens,self.num_heads,64,dtype=torch.bfloat16,device=dev)

        kv_c=torch.randn(num_tokens,512,dtype=torch.bfloat16,device=dev)
        k_pe=torch.randn(num_tokens,64,dtype=torch.bfloat16,device=dev)

        mqa_q=torch.empty(num_tokens,self.num_heads,576,dtype=torch.bfloat16,device=dev)

        k_cache=torch.zeros(8,self.block_size,656,dtype=torch.uint8,device=dev)

        slots=torch.tensor([0,3,9],dtype=torch.int64,device=dev)
        positions=torch.tensor([1,7,13],dtype=torch.int64,device=dev)

        rope_cache=self.make_rope_cache(dev)

        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
            ql_nope,
            q_pe,
            kv_c,
            k_pe,
            mqa_q,
            k_cache,
            slots,
            self.block_size,
            positions,
            rope_cache,
        )

        ok=torch.isfinite(mqa_q).all().item()

        return ok,0.0