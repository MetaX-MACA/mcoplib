import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Fused_kimi_k3_mla_key_concat_ds_mla_insert_runner(OpBenchmarkBase):

    def __init__(self,name,config):
        super().__init__(name,config)
        self.num_tokens=config.get("num_tokens",32)
        self.num_heads=config.get("num_heads",4)
        self.block_size=config.get("block_size",8)

    def define_metrics(self,state):
        state.add_summary("Op",self.name)
        state.add_summary("dtype",str(self.dtype))
        state.add_summary("Shape",f"tokens={self.num_tokens} heads={self.num_heads}")
        elements=self.num_tokens*self.num_heads*192
        state.add_element_count(elements)
        # 4-head light warps (kernel's KIMI_K3_HEADS_PER_WARP=4): k_pe is read
        # once per light warp (shared by its 4 head slots) plus once by the
        # cache order.
        light_orders=(self.num_heads+3)//4
        # reads (bf16, x2 bytes): q rope seg per head (H*64) + k_nope per head
        # (H*128) + k_pe per light order (light_orders*64) + k_pe for the cache
        # row (64) + kv_c (512); plus cos/sin row (64 floats = 256B) per token
        state.add_global_memory_reads(self.num_tokens*(self.num_heads*(64+128)+light_orders*64+64+512)*2 + self.num_tokens*256)
        # writes: q rope seg per head (H*64, bf16 x2) + k_out per head (H*192,
        # bf16 x2) + 656B fp8_ds_mla cache row per token
        state.add_global_memory_writes(self.num_tokens*(self.num_heads*(64+192))*2 + self.num_tokens*656)

    def make_rope_cache(self,dev,max_position=2048):
        inv_freq=1.0/(50000**(torch.arange(0,64,2,device=dev,dtype=torch.float32)/64))
        pos=torch.arange(max_position,device=dev,dtype=torch.float32)
        freq=torch.outer(pos,inv_freq)
        return torch.cat([freq.cos(),freq.sin()],dim=-1)

    def prepare_and_get_launcher(self,dev_id,tc_s):
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"

            q=torch.randn(self.num_tokens,self.num_heads,192,dtype=self.dtype,device=dev)
            k_nope=torch.randn(self.num_tokens,self.num_heads,128,dtype=self.dtype,device=dev)
            k_pe=torch.randn(self.num_tokens,64,dtype=self.dtype,device=dev)
            kv_c=torch.randn(self.num_tokens,512,dtype=self.dtype,device=dev)

            k_out=torch.empty(self.num_tokens,self.num_heads,192,dtype=self.dtype,device=dev)

            num_blocks=(self.num_tokens + self.block_size - 1) // self.block_size
            k_cache=torch.zeros(num_blocks,self.block_size,656,dtype=torch.uint8,device=dev)

            slots=torch.arange(self.num_tokens,dtype=torch.int64,device=dev)
            positions=torch.arange(self.num_tokens,dtype=torch.int64,device=dev)

            # positions 最大到 num_tokens-1，rope_cache 必须覆盖同样多的位置，
            # 否则 kernel 越界读 cos_sin_cache 触发 ATU 错误
            rope_cache=self.make_rope_cache(dev,max_position=max(self.num_tokens,2048))

        return self.make_launcher(dev_id,torch.ops._C.fused_kimi_k3_mla_key_concat_ds_mla_insert,q,k_nope,k_pe,kv_c,k_out,k_cache,slots,self.block_size,positions,rope_cache)

    def run_verification(self,dev_id):
        dev=f"cuda:{dev_id}"

        num_tokens=3
        num_heads=4
        block_size=8

        q=torch.randn(num_tokens,num_heads,192,dtype=self.dtype,device=dev)
        k_nope=torch.randn(num_tokens,num_heads,128,dtype=self.dtype,device=dev)
        k_pe=torch.randn(num_tokens,64,dtype=self.dtype,device=dev)
        kv_c=torch.randn(num_tokens,512,dtype=self.dtype,device=dev)

        k_out=torch.empty(num_tokens,num_heads,192,dtype=self.dtype,device=dev)

        k_cache=torch.zeros(2,block_size,656,dtype=torch.uint8,device=dev)

        slots=torch.tensor([0,3,9],dtype=torch.int64,device=dev)
        positions=torch.tensor([1,7,13],dtype=torch.int64,device=dev)

        rope_cache=self.make_rope_cache(dev)

        torch.ops._C.fused_kimi_k3_mla_key_concat_ds_mla_insert(q,k_nope,k_pe,kv_c,k_out,k_cache,slots,block_size,positions,rope_cache)

        cache=rope_cache.index_select(0,positions)
        cos,sin=cache.chunk(2,dim=-1)

        x1=k_pe.float()[...,::2]
        x2=k_pe.float()[...,1::2]

        k_pe_ref=torch.stack([x1*cos-x2*sin,x2*cos+x1*sin],dim=-1).flatten(-2).to(self.dtype)

        k_ref=torch.cat([k_nope,k_pe_ref[:,None,:].expand(-1,num_heads,-1)],dim=-1)

        torch.testing.assert_close(k_out,k_ref,atol=2e-2,rtol=2e-2)

        assert k_cache.shape==(2,block_size,656)

        cos_sim=F.cosine_similarity(k_out.flatten().float(),k_ref.flatten().float(),dim=0).item()

        return True,cos_sim