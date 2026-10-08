import torch
import torch.nn.functional as F
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._C
except ImportError:
    pass


class Masked_situ_and_mul_runner(OpBenchmarkBase):
    def __init__(self,name,config):
        super().__init__(name,config)
        self.num_experts=config.get("num_experts",64)
        self.max_num_tokens=config.get("max_num_tokens",128)
        self.hidden=config.get("hidden_size",4096)
        self.beta=config.get("beta",1.0)
        self.linear_beta=config.get("linear_beta",-1.0)

    def define_metrics(self,state):
        state.add_summary("Op",self.name)
        state.add_summary("dtype",self.config.get("dtype",str(self.dtype)))
        state.add_summary("Shape",f"({self.num_experts} {self.max_num_tokens} {self.hidden})")
        elements=self.num_experts*self.max_num_tokens*self.hidden
        state.add_element_count(elements)
        bytes_per=2 if self.dtype in [torch.float16,torch.bfloat16] else 4
        state.add_global_memory_reads(self.num_experts*self.max_num_tokens*2*self.hidden*bytes_per)
        state.add_global_memory_writes(elements*bytes_per)

    def prepare_and_get_launcher(self,dev_id,tc_s):
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"

            input=torch.randn(
                self.num_experts,
                self.max_num_tokens,
                2*self.hidden,
                dtype=self.dtype,
                device=dev,
            )

            output=torch.zeros(
                self.num_experts,
                self.max_num_tokens,
                self.hidden,
                dtype=self.dtype,
                device=dev,
            )

            expert_num_tokens=torch.full(
                (self.num_experts,),
                self.max_num_tokens,
                dtype=torch.int32,
                device=dev,
            )

        return self.make_launcher(
            dev_id,
            torch.ops._C.masked_situ_and_mul,
            output,
            input,
            expert_num_tokens,
            self.beta,
            self.linear_beta,
        )

    def run_verification(self,dev_id):
        dev=f"cuda:{dev_id}"

        num_experts=4
        max_num_tokens=7
        hidden=512

        input=torch.randn(
            num_experts,
            max_num_tokens,
            2*hidden,
            dtype=self.dtype,
            device=dev,
        )

        expert_num_tokens=torch.tensor(
            [0,1,4,7],
            dtype=torch.int32,
            device=dev,
        )

        output=torch.zeros(
            num_experts,
            max_num_tokens,
            hidden,
            dtype=self.dtype,
            device=dev,
        )

        torch.ops._C.masked_situ_and_mul(
            output,
            input,
            expert_num_tokens,
            self.beta,
            self.linear_beta,
        )

        gate,up=input.float().chunk(2,dim=-1)

        gate_out=self.beta*torch.tanh(gate/self.beta)*torch.sigmoid(gate)

        expected=(gate_out*up).to(self.dtype)

        for expert,n in enumerate(expert_num_tokens.tolist()):
            if n > 0:
                diff=(output[expert,:n].float()-expected[expert,:n].float()).abs().max().item()
                if diff > 3e-2:
                    return False,diff

            if torch.count_nonzero(output[expert,n:]) != 0:
                return False,float("inf")

        return True,0.0