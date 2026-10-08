import os
import itertools
import torch
import torch.nn.functional as F
import mcoplib.op as ops

# ---- config from task spec ----
NUM_EXPERTS = 128
TOPK = 8
EP_SIZE = 8
EP_RANK = 0
HIDDEN_SIZE = 1536
DTYPE = torch.bfloat16
RES_SCALE = 1.0
SP_RANK = 0

# (sp_size, num_tokens) pairs — all 18 shapes from config
SHAPES = [
    (1, 48), (1, 128), (1, 1024), (1, 2048), (1, 4096),
    (1, 8192), (1, 10240), (1, 16384), (1, 32768),
    (8, 48), (8, 128), (8, 1024), (8, 2048), (8, 4096),
    (8, 8192), (8, 10240), (8, 16384), (8, 32768),
]


def cosine_similarity(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def benchmark(func, warmup=20, repeat=100):
    for _ in range(warmup):
        func()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(repeat):
        start.record()
        func()
        end.record()
        torch.cuda.synchronize()
        total += start.elapsed_time(end)
    return total / repeat


def get_moe_tokens_info(num_tokens, num_experts, topk, ep_size=1, ep_rank=0):
    """Exact replica of the benchmark framework's dispatch generator."""
    num_scatter_tokens = num_tokens * topk
    num_scatter_tokens_per_rank = num_scatter_tokens // ep_size
    num_experts_per_rank = num_experts // ep_size

    experts_start_idx = ep_rank * num_experts_per_rank
    experts_end_idx = experts_start_idx + num_experts_per_rank

    experts_idx_for_each_rank = []
    for rank_idx in range(ep_size):
        start_idx = rank_idx * num_experts_per_rank
        end_idx = start_idx + num_experts_per_rank
        experts_idx_for_each_rank.append(list(range(start_idx, end_idx)))
    transpose_experts = [list(row) for row in zip(*experts_idx_for_each_rank)]
    experts_array = [num for row in transpose_experts for num in row]

    all_select_experts = []
    all_select_weights = []
    cur_expert = 0
    for token_idx in range(num_tokens):
        cur_token_selections = []
        for topk_idx in range(topk):
            cur_token_selections.append(experts_array[cur_expert])
            cur_expert += 1
            if cur_expert >= num_experts:
                cur_expert = 0
        all_select_experts.append(cur_token_selections)
        all_select_weights.append([1 / topk for _ in range(topk)])

    cur_rank_tokens = {}
    cur_rank_weights = {}
    dispatch_tokens = 0
    for token_idx in range(num_tokens):
        cur_token_dispatch_experts = []
        cur_token_dispatch_weights = []
        for expert_idx, expert_weight in zip(all_select_experts[token_idx], all_select_weights[token_idx]):
            if expert_idx >= experts_start_idx and expert_idx < experts_end_idx:
                cur_token_dispatch_experts.append(expert_idx)
                cur_token_dispatch_weights.append(expert_weight)
        if cur_token_dispatch_experts:
            cur_rank_tokens[token_idx] = cur_token_dispatch_experts
            cur_rank_weights[token_idx] = cur_token_dispatch_weights
            dispatch_tokens += len(cur_token_dispatch_experts)

    expert_dispatch_tokens = [[] for _ in range(experts_start_idx, experts_end_idx)]
    expert_dispatch_weights = [[] for _ in range(experts_start_idx, experts_end_idx)]
    for token_idx in cur_rank_tokens:
        for expert_idx, weight in zip(cur_rank_tokens[token_idx], cur_rank_weights[token_idx]):
            expert_dispatch_tokens[expert_idx - experts_start_idx].append(token_idx)
            expert_dispatch_weights[expert_idx - experts_start_idx].append(weight)

    scatter_token_id = []
    scatter_token_weight = []
    for expert_idx, tokens in enumerate(expert_dispatch_tokens):
        weights = expert_dispatch_weights[expert_idx]
        for target_token, target_weight in zip(tokens, weights):
            scatter_token_id.append(target_token)
            scatter_token_weight.append(target_weight)

    return dispatch_tokens, scatter_token_id, scatter_token_weight


def make_inputs(sp_size, num_tokens, hidden_size, device="cuda"):
    dispatch_tokens, scatter_id_list, scatter_w_list = get_moe_tokens_info(
        num_tokens, NUM_EXPERTS, TOPK, ep_size=EP_SIZE, ep_rank=EP_RANK)

    scatter_tokens = torch.randn(dispatch_tokens, hidden_size, dtype=DTYPE, device=device)
    scatter_token_id = torch.tensor(scatter_id_list, dtype=torch.int32, device=device)
    scatter_token_weight = torch.tensor(scatter_w_list, dtype=torch.float32, device=device)

    if sp_size > 1:
        num_res = (num_tokens + sp_size - 1) // sp_size
        res_token_start = SP_RANK * num_res
        res_token_end = min(res_token_start + num_res, num_tokens)
    else:
        num_res = 1
        res_token_start = 0
        res_token_end = num_tokens

    residual = torch.randn(num_res, hidden_size, dtype=DTYPE, device=device)
    return (scatter_tokens, scatter_token_id, scatter_token_weight, residual,
            dispatch_tokens, num_res, res_token_start, res_token_end)


def reference(scatter_tokens, scatter_token_id, scatter_token_weight, residual,
              num_tokens, hidden_size, res_token_start, res_token_end, res_scale):
    convergent = torch.zeros(num_tokens, hidden_size, dtype=DTYPE, device=scatter_tokens.device)
    convergent[res_token_start:res_token_end] += residual * res_scale
    convergent.index_add_(
        0, scatter_token_id,
        (scatter_tokens * scatter_token_weight.unsqueeze(-1)).to(DTYPE))
    return convergent


def run_case(sp_size, num_tokens):
    (scatter_tokens, scatter_token_id, scatter_token_weight, residual,
     dispatch_tokens, num_res, res_token_start, res_token_end) = make_inputs(
        sp_size, num_tokens, HIDDEN_SIZE)

    convergent = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=DTYPE, device="cuda")

    def run():
        ops.moe_gather(scatter_tokens, scatter_token_id, scatter_token_weight,
                       convergent, residual, RES_SCALE, res_token_start)

    # accuracy: fresh zero output, single call
    convergent.zero_()
    run()
    torch.cuda.synchronize()
    expected = reference(scatter_tokens, scatter_token_id, scatter_token_weight, residual,
                         num_tokens, HIDDEN_SIZE, res_token_start, res_token_end, RES_SCALE)
    cos = cosine_similarity(convergent, expected)

    time_ms = benchmark(run)

    esz = torch.tensor([], dtype=DTYPE).element_size()
    scatter_tokens_bytes = dispatch_tokens * HIDDEN_SIZE * esz
    id_bytes = dispatch_tokens * 4
    w_bytes = dispatch_tokens * 4
    res_bytes = num_res * HIDDEN_SIZE * esz
    read_bytes = scatter_tokens_bytes + id_bytes + w_bytes + res_bytes   # all inputs
    write_bytes = scatter_tokens_bytes                                    # reference 口径
    io_bytes = read_bytes + write_bytes
    gbps = io_bytes / (time_ms * 1e-3) / 1e9

    status = "PASS" if cos > 0.9999 else "FAIL"
    print(f"[{status}] sp={sp_size:<2} num_tokens={num_tokens:<6} D={dispatch_tokens:<7} "
          f"H={HIDDEN_SIZE} num_res={num_res:<6} "
          f"cos={cos:.8f} time={time_ms:.4f}ms BW={gbps:.1f} GB/s")
    return cos, gbps


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"Device: {torch.cuda.get_device_name(0)}  visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print("=" * 105)
    best_bw = 0.0
    best_cfg = None
    all_pass = True
    for sp_size, num_tokens in SHAPES:
        cos, gbps = run_case(sp_size, num_tokens)
        if cos <= 0.9999:
            all_pass = False
        if gbps > best_bw:
            best_bw = gbps
            best_cfg = (sp_size, num_tokens)
    print("=" * 105)
    print(f"ALL_PASS={all_pass}  BEST_BW={best_bw:.1f} GB/s @ sp={best_cfg[0]} num_tokens={best_cfg[1]}")
    TARGET = 1190.0
    print(f"TARGET={TARGET} GB/s  {'MET' if best_bw >= TARGET else 'NOT MET'}")


if __name__ == "__main__":
    main()
