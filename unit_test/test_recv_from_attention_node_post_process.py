"""Test for `recv_from_attention_node_post_process`
(op/recv_from_attention_node_post_process.cu).

The kernel is the DeepEP "recv" companion: it partitions top-k routing results
by local expert and gathers each expert's tokens' hidden states into a packed
buffer. On output it fills:

    expert_cnt[e]            = number of (token, topk) entries routed to expert e
    valid_idx_size[0]        = sum of expert_cnt (total routed entries)
    deepep_topk_weights      = top-k weights, packed per expert
    deepep_hidden_status     = hidden states, packed per expert
                               (layout [num_local_experts * batch_size * work_count, hidden])

Only bfloat16 is supported.
"""

import torch

import mcoplib.op as ops


def ref_recv(hidden_status, topk_idx, topk_weights, begin_expert_id,
             num_local_experts, batch_size, work_count, topk):
    num_tokens = batch_size * work_count
    hidden_size = hidden_status.shape[-1]

    expert_cnt = [0] * num_local_experts
    rows = [[] for _ in range(num_local_experts)]
    weights = [[] for _ in range(num_local_experts)]

    # The kernel walks the flattened (token, topk) pairs in ascending order,
    # so replicating that order reproduces the exact per-expert packing.
    for t in range(num_tokens):
        for k in range(topk):
            e = int(topk_idx[t, k]) - begin_expert_id
            if 0 <= e < num_local_experts:
                rows[e].append(t)
                weights[e].append(float(topk_weights[t, k]))
                expert_cnt[e] += 1

    deepep_hidden = torch.zeros(num_local_experts * num_tokens, hidden_size,
                                dtype=torch.bfloat16)
    deepep_weights = torch.zeros(num_local_experts * num_tokens,
                                 dtype=torch.float32)
    for e in range(num_local_experts):
        base = e * num_tokens
        for i, t in enumerate(rows[e]):
            deepep_hidden[base + i] = hidden_status[t]
            deepep_weights[base + i] = weights[e][i]

    return expert_cnt, deepep_hidden, deepep_weights


def run_case(batch_size, work_count, topk, num_local_experts, hidden_size,
             begin_expert_id=0):
    device = "cuda"
    num_tokens = batch_size * work_count

    hidden_status = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16,
                                device=device)
    # Route every token to every local expert, so all experts get all tokens.
    topk_idx = torch.arange(begin_expert_id,
                            begin_expert_id + num_local_experts,
                            dtype=torch.int32, device=device)
    topk_idx = topk_idx.expand(num_tokens, topk).contiguous()
    topk_weights = torch.rand(num_tokens, topk, dtype=torch.float32,
                              device=device)

    max_index_size = num_tokens * topk
    ori_index = torch.zeros(max_index_size * 2, dtype=torch.int32, device=device)
    new_index = torch.zeros(max_index_size * 2, dtype=torch.int32, device=device)
    deepep_hidden = torch.zeros(num_local_experts * num_tokens, hidden_size,
                                dtype=torch.bfloat16, device=device)
    deepep_weights = torch.zeros(num_local_experts * num_tokens,
                                 dtype=torch.float32, device=device)
    expert_cnt = torch.zeros(num_local_experts, dtype=torch.int32, device=device)
    valid_idx_size = torch.zeros(1, dtype=torch.int32, device=device)

    ops.recv_from_attention_node_post_process(
        hidden_status, topk_idx, topk_weights, ori_index, new_index,
        deepep_hidden, deepep_weights, expert_cnt, valid_idx_size,
        begin_expert_id, num_local_experts, max_index_size, work_count)
    torch.cuda.synchronize()

    ref_cnt, ref_hidden, ref_weights = ref_recv(
        hidden_status, topk_idx, topk_weights, begin_expert_id,
        num_local_experts, batch_size, work_count, topk)

    # Every token routes to every expert.
    assert expert_cnt.cpu().tolist() == ref_cnt
    assert valid_idx_size.cpu().item() == sum(ref_cnt)
    torch.testing.assert_close(deepep_weights, ref_weights.cuda())
    # deepep_hidden is a pure copy of bf16 rows -> exact match.
    torch.testing.assert_close(deepep_hidden, ref_hidden.cuda())


def test_recv_from_attention_node_post_process():
    torch.manual_seed(0)
    run_case(batch_size=2, work_count=1, topk=2, num_local_experts=2,
             hidden_size=8)
    run_case(batch_size=4, work_count=1, topk=4, num_local_experts=4,
             hidden_size=16)


if __name__ == "__main__":
    test_recv_from_attention_node_post_process()
    print("test_recv_from_attention_node_post_process PASS")
