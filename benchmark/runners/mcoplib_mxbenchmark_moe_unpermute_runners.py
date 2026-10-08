import torch
from mcoplib_mxbenchmark_op_wrapper import OpBenchmarkBase

try:
    import mcoplib._moe_C
except ImportError:
    pass


class Moe_unpermute_runner(OpBenchmarkBase):

    def __init__(self, name, config):
        super().__init__(name, config)

        self.batch_size = config.get("batch_size", 64)
        self.hidden_size = config.get("hidden_size", 4096)
        self.num_experts = config.get("num_experts", 64)
        self.top_k = config.get("top_k", 8)
        self.seed = config.get("seed", None)


    def define_metrics(self, state):

        state.add_summary("Op", self.name)

        state.add_summary(
            "Shape",
            f"({self.batch_size},{self.top_k},{self.hidden_size})"
        )

        element_size = 2

        reads = (
            self.batch_size *
            self.top_k *
            self.hidden_size *
            element_size
            +
            self.batch_size *
            self.top_k *
            4
        )

        writes = (
            self.batch_size *
            self.hidden_size *
            element_size
        )

        state.add_global_memory_reads(reads)
        state.add_global_memory_writes(writes)

        state.add_element_count(
            self.batch_size * self.hidden_size
        )


    def prepare_and_get_launcher(self, dev_id, tc_s):
        with torch.cuda.stream(tc_s):
            dev=f"cuda:{dev_id}"

            n=self.batch_size
            h=self.hidden_size
            k=self.top_k

            permuted_hidden_states=torch.randn(
                n*k,
                h,
                dtype=self.dtype,
                device=dev,
            )

            topk_weights=torch.ones(
                n,
                k,
                dtype=torch.float32,
                device=dev,
            )

            inv_permuted_idx=torch.arange(
                n*k,
                dtype=torch.int32,
                device=dev,
            ).reshape(n,k)

            output=torch.empty(
                n,
                h,
                dtype=self.dtype,
                device=dev,
            )


        def launcher(_):
            torch.ops._moe_C.moe_unpermute(
                permuted_hidden_states,
                topk_weights,
                inv_permuted_idx,
                None,
                k,
                output,
            )

        return launcher

    def run_verification(self, dev_id):

        dev=f"cuda:{dev_id}"

        tokens=32
        hidden=256
        topk=4
        experts=16


        x=torch.randn(
            tokens,
            hidden,
            dtype=self.dtype,
            device=dev
        )


        ids=torch.randint(
            0,
            experts,
            (tokens,topk),
            dtype=torch.int32,
            device=dev
        )


        idx=torch.arange(
            tokens*topk,
            dtype=torch.int32,
            device=dev
        ).reshape(tokens,topk)



        perm=torch.empty(
            tokens*topk,
            hidden,
            dtype=self.dtype,
            device=dev
        )


        offset=torch.empty(
            experts+1,
            dtype=torch.int64,
            device=dev
        )


        inv=torch.empty(
            tokens,
            topk,
            dtype=torch.int32,
            device=dev
        )


        sorted_idx=torch.empty(
            tokens*topk,
            dtype=torch.int32,
            device=dev
        )


        torch.ops._moe_C.moe_permute(
            x,
            ids,
            idx,
            None,
            experts,
            experts,
            topk,
            perm,
            offset,
            inv,
            sorted_idx
        )


        weight=torch.ones(
            tokens,
            topk,
            dtype=torch.float32,
            device=dev
        )


        out=torch.empty_like(x)


        torch.ops._moe_C.moe_unpermute(
            perm,
            weight,
            inv,
            None,
            topk,
            out
        )


        # reference
        ref=torch.zeros_like(x)

        for i in range(tokens):
            for k in range(topk):

                src=inv[i,k]

                ref[i]+=perm[src]*weight[i,k]


        return self.check_diff(
            out,
            ref
        )