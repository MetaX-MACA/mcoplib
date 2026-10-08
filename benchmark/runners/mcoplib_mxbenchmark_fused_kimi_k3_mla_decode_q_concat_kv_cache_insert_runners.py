import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_kimi_k3_mla_decode_q_concat_kv_cache_insert_runner(OpBenchmarkBase):

    def __init__(self,name,config):
        super().__init__(name,config)
        self.num_tokens=config.get("num_tokens",3)
        self.num_heads=config.get("num_heads",4)
        self.block_size=config.get("block_size",8)
        self.seed=config.get("seed",None)

    def define_metrics(self,state):
        state.add_summary("Op",self.name)
        state.add_summary("dtype",str(self.dtype))
        state.add_summary("Shape",f"tokens={self.num_tokens},heads={self.num_heads}")
        elements=self.num_tokens*self.num_heads*576
        state.add_element_count(elements)
        state.add_global_memory_reads(self.num_tokens*(512+self.num_heads*64+512)*2)
        state.add_global_memory_writes(self.num_tokens*(self.num_heads*576)*2)

    def make_rope_cache(self,dev):
        inv_freq=1.0/(50000**(torch.arange(0,64,2,device=dev,dtype=torch.float32)/64))
        pos=torch.arange(2048,device=dev,dtype=torch.float32)
        freq=torch.outer(pos,inv_freq)
        return torch.cat([freq.cos(),freq.sin()],dim=-1)

    def prepare_and_get_launcher(self,dev_id,tc_s):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"

            ql_nope=torch.randn(self.num_tokens,self.num_heads,512,dtype=self.dtype,device=dev)
            q_pe=torch.randn(self.num_tokens,self.num_heads,64,dtype=self.dtype,device=dev)
            kv_c=torch.randn(self.num_tokens,512,dtype=self.dtype,device=dev)
            k_pe=torch.randn(self.num_tokens,64,dtype=self.dtype,device=dev)

            mqa_q=torch.empty(self.num_tokens,self.num_heads,576,dtype=self.dtype,device=dev)

            num_cache_blocks=max(8,(self.num_tokens+self.block_size-1)//self.block_size)
            k_cache=torch.zeros(num_cache_blocks,self.block_size,576,dtype=self.dtype,device=dev)

            slots=torch.arange(self.num_tokens,dtype=torch.int64,device=dev)
            positions=torch.arange(self.num_tokens,dtype=torch.int64,device=dev)

            rope_cache=self.make_rope_cache(dev)

        return self.make_launcher(dev_id,torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert,ql_nope,q_pe,kv_c,k_pe,mqa_q,k_cache,slots,self.block_size,positions,rope_cache)


    def run_verification(self,dev_id):
        dev=f"cuda:{dev_id}"

        num_tokens=3
        num_heads=4
        block_size=8

        ql_nope=torch.randn(num_tokens,num_heads,512,dtype=self.dtype,device=dev)
        q_pe=torch.randn(num_tokens,num_heads,64,dtype=self.dtype,device=dev)
        kv_c=torch.randn(num_tokens,512,dtype=self.dtype,device=dev)
        k_pe=torch.randn(num_tokens,64,dtype=self.dtype,device=dev)

        mqa_q=torch.empty(num_tokens,num_heads,576,dtype=self.dtype,device=dev)

        k_cache=torch.zeros(2,block_size,576,dtype=self.dtype,device=dev)

        slots=torch.tensor([0,3,9],dtype=torch.int64,device=dev)
        positions=torch.tensor([1,7,13],dtype=torch.int64,device=dev)

        rope_cache=self.make_rope_cache(dev)

        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(ql_nope,q_pe,kv_c,k_pe,mqa_q,k_cache,slots,block_size,positions,rope_cache)

        cache=rope_cache.index_select(0,positions)
        cos,sin=cache.chunk(2,dim=-1)

        q1=q_pe.float()[...,::2]
        q2=q_pe.float()[...,1::2]

        q_pe_ref=torch.stack([q1*cos.unsqueeze(1)-q2*sin.unsqueeze(1),q2*cos.unsqueeze(1)+q1*sin.unsqueeze(1)],dim=-1).flatten(-2).to(self.dtype)

        q_ref=torch.cat([ql_nope,q_pe_ref],dim=-1)

        torch.testing.assert_close(mqa_q,q_ref,atol=2e-2,rtol=2e-2)

        k1=k_pe.float()[...,::2]
        k2=k_pe.float()[...,1::2]

        k_pe_ref=torch.stack([k1*cos-k2*sin,k2*cos+k1*sin],dim=-1).flatten(-2).to(self.dtype)

        cache_ref=torch.cat([kv_c,k_pe_ref],dim=-1)

        cache_out=k_cache.reshape(-1,576).index_select(0,slots)

        torch.testing.assert_close(cache_out,cache_ref,atol=2e-2,rtol=2e-2)

        cos_sim=F.cosine_similarity(mqa_q.flatten().float(),q_ref.flatten().float(),dim=0).item()

        return True,cos_sim