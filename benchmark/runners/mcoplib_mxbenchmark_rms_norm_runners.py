import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Rms_norm_runner(OpBenchmarkBase):
    def __init__(self,name,config):
        super().__init__(name,config)

        self.B=config.get("batch_size")
        self.S=config.get("seq_len")
        self.H=config.get("hidden_size")
        self.heads=config.get("heads",0)

        self.var_epsilon=config.get("var_epsilon",1e-6)

        self.is_4d=self.heads>0

        if self.is_4d:
            self.head_dim=self.H//self.heads
            self.shape=(self.B,self.S,self.heads,self.head_dim)
            self.weight_dim=self.head_dim
        else:
            self.shape=(self.B,self.S,self.H)
            self.weight_dim=self.H


    def define_metrics(self,state):
        state.add_summary("Op",self.name)
        state.add_summary("dtype",self.config.get("dtype",str(self.dtype)))
        state.add_summary("Shape","("+" ".join(map(str,self.shape))+")")

        total_elements=1
        for x in self.shape:
            total_elements*=x

        state.add_element_count(total_elements)

        element_size=2 if self.dtype in [torch.float16,torch.bfloat16] else 4

        state.add_global_memory_reads((total_elements+self.weight_dim)*element_size)
        state.add_global_memory_writes(total_elements*element_size)


    def prepare_and_get_launcher(self,dev_id,tc_s):
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"

            input_tensor=torch.randn(*self.shape,dtype=self.dtype,device=dev)
            weight=torch.randn(self.weight_dim,dtype=self.dtype,device=dev)
            output=torch.empty_like(input_tensor)

        return self.make_launcher(dev_id,torch.ops._C.rms_norm,output,input_tensor,weight,self.var_epsilon)


    def run_verification(self,dev_id):
        dev=f"cuda:{dev_id}"

        input_tensor=torch.randn(*self.shape,dtype=self.dtype,device=dev)
        weight=torch.randn(self.weight_dim,dtype=self.dtype,device=dev)
        output=torch.empty_like(input_tensor)

        torch.ops._C.rms_norm(output,input_tensor,weight,self.var_epsilon)

        x=input_tensor.float()
        w=weight.float()

        mean_square=torch.mean(x*x,dim=-1,keepdim=True)
        ref=x*torch.rsqrt(mean_square+self.var_epsilon)*w

        return self.check_diff(output,ref.to(self.dtype))