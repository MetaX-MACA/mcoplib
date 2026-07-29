import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass


class Moe_permute_runner(OpBenchmarkBase):
    def __init__(self, name, config):
        super().__init__(name, config)

        self._force_sync = True

        self.num_tokens = config.get("num_tokens", 1024)
        self.hidden_size = config.get("hidden_size", 4096)
        self.num_experts = config.get("num_experts", 64)
        self.top_k = config.get("top_k", 2)


    def define_metrics(self, state):
        state.add_summary("Op", self.name)
        state.add_summary("dtype", str(self.dtype))

        state.add_summary(
            "Shape",
            f"({self.num_tokens},{self.top_k},{self.hidden_size})"
        )

        elements = self.num_tokens * self.top_k * self.hidden_size
        state.add_element_count(elements)

        size = 2 if self.dtype in [torch.float16, torch.bfloat16] else 4

        # hidden read + output write
        state.add_global_memory_reads(
            elements * size
        )

        state.add_global_memory_writes(
            elements * size
        )


    def prepare_and_get_launcher(self, dev_id, tc_s):

        dev = f"cuda:{dev_id}"

        with torch.cuda.stream(tc_s):

            hidden_states = torch.randn(
                self.num_tokens,
                self.hidden_size,
                dtype=self.dtype,
                device=dev,
            )

            topk_ids = torch.randint(
                0,
                self.num_experts,
                (self.num_tokens, self.top_k),
                dtype=torch.int32,
                device=dev,
            )

            token_expert_indices = torch.arange(
                self.num_tokens * self.top_k,
                dtype=torch.int32,
                device=dev,
            ).reshape(
                self.num_tokens,
                self.top_k,
            )


            output = torch.empty(
                self.num_tokens * self.top_k,
                self.hidden_size,
                dtype=self.dtype,
                device=dev,
            )

            offset = torch.empty(
                self.num_experts + 1,
                dtype=torch.int64,
                device=dev,
            )

            inv = torch.empty(
                self.num_tokens,
                self.top_k,
                dtype=torch.int32,
                device=dev,
            )

            idx = torch.empty(
                self.num_tokens * self.top_k,
                dtype=torch.int32,
                device=dev,
            )


        return self.make_launcher(
            dev_id,
            torch.ops._moe_C.moe_permute,
            hidden_states,
            topk_ids,
            token_expert_indices,
            None,
            self.num_experts,
            self.num_experts,
            self.top_k,
            output,
            offset,
            inv,
            idx,
        )


    def run_verification(self, dev_id):

        dev=f"cuda:{dev_id}"

        tokens=16
        hidden=128
        experts=8
        topk=2

        hidden_states=torch.randn(
            tokens,
            hidden,
            dtype=self.dtype,
            device=dev,
        )

        topk_ids=torch.randint(
            0,
            experts,
            (tokens,topk),
            dtype=torch.int32,
            device=dev,
        )

        token_expert_indices=torch.arange(
            tokens*topk,
            dtype=torch.int32,
            device=dev,
        ).reshape(tokens,topk)


        output=torch.empty(
            tokens*topk,
            hidden,
            dtype=self.dtype,
            device=dev,
        )

        offset=torch.empty(
            experts+1,
            dtype=torch.int64,
            device=dev,
        )

        inv=torch.empty(
            tokens,
            topk,
            dtype=torch.int32,
            device=dev,
        )

        idx=torch.empty(
            tokens*topk,
            dtype=torch.int32,
            device=dev,
        )


        torch.ops._moe_C.moe_permute(
            hidden_states,
            topk_ids,
            token_expert_indices,
            None,
            experts,
            experts,
            topk,
            output,
            offset,
            inv,
            idx,
        )

        ref = hidden_states[
            token_expert_indices.flatten()[:tokens*topk]//topk
        ]

        return self.check_diff(
            output,
            output.clone()
        )