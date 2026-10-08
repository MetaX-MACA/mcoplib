"""Test for `send_to_attention_node_pre_process`
(op/send_to_attention_node_pre_process.cu).

The kernel scatters weighted MoE hidden states back to their source rows
(part of the DeepEP "send" path). For each valid entry ``bidx``:

    src_expert = new_index[bidx * 2]     (== local expert id)
    src_row    = new_index[bidx * 2 + 1]
    tar_row    = ori_index[bidx * 2 + 1]

    output[tar_row] += moe_hidden_status[src_expert, src_row] *
                       deepep_topk_weights[src_expert * num_tokens + src_row]

The kernel zeros ``output`` first, then accumulates with atomicAdd, so several
experts may contribute to the same row. Only bfloat16 is supported.
"""

import torch

import mcoplib.op as ops


def ref_send(moe_hidden_status, deepep_topk_weights, ori_index, new_index,
             num_tokens, num_valid):
    moe = moe_hidden_status.float().cpu()
    w = deepep_topk_weights.cpu()
    oi = ori_index.cpu()
    ni = new_index.cpu()
    hidden_size = moe.shape[-1]
    out = torch.zeros(num_tokens, hidden_size, dtype=torch.float32)
    for bidx in range(num_valid):
        src_expert = int(ni[bidx * 2])
        src_row = int(ni[bidx * 2 + 1])
        tar_row = int(oi[bidx * 2 + 1])
        out[tar_row] += moe[src_expert, src_row] * float(w[src_expert * num_tokens + src_row])
    return out


def run_case(num_tokens, hidden_size, num_local_experts):
    device = "cuda"
    moe = torch.randn(num_local_experts, num_tokens, hidden_size,
                      dtype=torch.bfloat16, device=device)
    deepep_weights = torch.rand(num_local_experts * num_tokens,
                                dtype=torch.float32, device=device)

    # Route each token to exactly one expert (expert = row % E), identity
    # target row. This keeps the reference collision-free and deterministic.
    num_valid = num_tokens
    new_index = torch.zeros(num_valid * 2, dtype=torch.int32, device=device)
    ori_index = torch.zeros(num_valid * 2, dtype=torch.int32, device=device)
    for bidx in range(num_valid):
        new_index[bidx * 2] = bidx % num_local_experts
        new_index[bidx * 2 + 1] = bidx
        ori_index[bidx * 2 + 1] = bidx

    valid_idx_size = torch.tensor([num_valid], dtype=torch.int32, device=device)
    output = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16,
                         device=device)

    ops.send_to_attention_node_pre_process(moe, deepep_weights, ori_index,
                                           new_index, output, valid_idx_size,
                                           num_valid)
    torch.cuda.synchronize()

    ref = ref_send(moe, deepep_weights, ori_index, new_index, num_tokens,
                   num_valid)
    torch.testing.assert_close(output, ref.to(torch.bfloat16).cuda(),
                               atol=0.05, rtol=0.05)


def test_send_to_attention_node_pre_process():
    torch.manual_seed(0)
    run_case(num_tokens=4, hidden_size=8, num_local_experts=2)
    run_case(num_tokens=16, hidden_size=32, num_local_experts=4)


if __name__ == "__main__":
    test_send_to_attention_node_pre_process()
    print("test_send_to_attention_node_pre_process PASS")
