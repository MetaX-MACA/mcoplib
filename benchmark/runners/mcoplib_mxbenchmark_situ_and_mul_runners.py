import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Situ_and_mul_runner(OpBenchmarkBase):
    def __init__(self,name,config):
        super().__init__(name,config)
        self.tokens=config.get("tokens",4096)
        self.hidden=config.get("hidden_size",4096)
        self.beta=config.get("beta",1.0)
        self.linear_beta=config.get("linear_beta",-1.0)

    def define_metrics(self,state):
        state.add_summary("Op",self.name)
        state.add_summary("dtype",self.config.get("dtype",str(self.dtype)))
        state.add_summary("Shape",f"({self.tokens} {self.hidden})")
        elements=self.tokens*self.hidden
        state.add_element_count(elements)
        element_size=2 if self.dtype in [torch.float16,torch.bfloat16] else 4
        state.add_global_memory_reads(self.tokens*self.hidden*2*element_size)
        state.add_global_memory_writes(self.tokens*self.hidden*element_size)

    def prepare_and_get_launcher(self,dev_id,tc_s):
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"
            input=torch.randn(self.tokens,2*self.hidden,dtype=self.dtype,device=dev)
            output=torch.empty(self.tokens,self.hidden,dtype=self.dtype,device=dev)
        return self.make_launcher(dev_id,torch.ops._C.situ_and_mul,output,input,self.beta,self.linear_beta)

    def run_verification(self,dev_id):
        dev=f"cuda:{dev_id}"
        tokens=17
        hidden=512
        input=torch.randn(tokens,2*hidden,dtype=self.dtype,device=dev)
        output=torch.empty(tokens,hidden,dtype=self.dtype,device=dev)

        torch.ops._C.situ_and_mul(output,input,self.beta,self.linear_beta)

        gate,up=input.float().chunk(2,dim=-1)
        gate_out=self.beta*torch.tanh(gate/self.beta)*torch.sigmoid(gate)

        if self.linear_beta>0:
            up=self.linear_beta*torch.tanh(up/self.linear_beta)

        expected=(gate_out*up).to(self.dtype)
        diff=(output.float()-expected.float()).abs().max().item()

        return diff<3e-2,diff