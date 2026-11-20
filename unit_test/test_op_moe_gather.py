import os
import torch
import torch.nn.functional as F
import mcoplib.op as ops

# ---- config from task spec ----
NUM_EXPERTS = 128
TOPK = 8
EP_SIZE = 8
EP_RANK = 0
HIDDEN_SIZE = 1536
OUT_DTYPE = torch.bfloat16
RES_SCALE = 1.0
SP_RANK = 0
COS_THRESH = 0.9999
TARGET = 1190.0

# (sp_size, num_tokens) pairs — all 18 shapes from config
SHAPES = [
    (1, 48), (1, 128), (1, 1024), (1, 2048), (1, 4096),
    (1, 8192), (1, 10240), (1, 16384), (1, 32768),
    (8, 48), (8, 128), (8, 1024), (8, 2048), (8, 4096),
    (8, 8192), (8, 10240), (8, 16384), (8, 32768),
]

# name -> (torch dtype, qmax or None for non-quantized)
IN_DTYPES = {
    "bf16": (torch.bfloat16, None),
    "int8": (torch.int8, 127.0),
    "fp8":  (torch.float8_e4m3fn, 448.0),
}


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
            if experts_start_idx <= expert_idx < experts_end_idx:
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


def quantize_per_row(x, in_dtype, qmax):
    """Symmetric per-row quantization. Returns (q, scale[D]) with x ≈ q * scale."""
    xf = x.float()
    amax = xf.abs().amax(dim=-1).clamp(min=1e-6)
    scale = (amax / qmax).contiguous()
    y = xf / scale.unsqueeze(-1)
    if in_dtype == torch.int8:
        q = y.round().clamp(-qmax, qmax).to(torch.int8)
    else:
        q = y.clamp(-qmax, qmax).to(in_dtype)
    return q.contiguous(), scale.to(torch.float32)


def make_inputs(sp_size, num_tokens, hidden_size, in_name, device="cuda"):
    in_dtype, qmax = IN_DTYPES[in_name]
    dispatch_tokens, scatter_id_list, scatter_w_list = get_moe_tokens_info(
        num_tokens, NUM_EXPERTS, TOPK, ep_size=EP_SIZE, ep_rank=EP_RANK)

    base = torch.randn(dispatch_tokens, hidden_size, dtype=torch.bfloat16, device=device)
    if qmax is None:
        scatter_tokens, scatter_scale = base, None
    else:
        scatter_tokens, scatter_scale = quantize_per_row(base, in_dtype, qmax)

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

    residual = torch.randn(num_res, hidden_size, dtype=OUT_DTYPE, device=device)
    return (scatter_tokens, scatter_scale, scatter_token_id, scatter_token_weight, residual,
            dispatch_tokens, num_res, res_token_start, res_token_end)


def reference(scatter_tokens, scatter_scale, scatter_token_id, scatter_token_weight, residual,
              num_tokens, hidden_size, res_token_start, res_token_end, res_scale):
    convergent = torch.zeros(num_tokens, hidden_size, dtype=OUT_DTYPE, device=scatter_tokens.device)
    convergent[res_token_start:res_token_end] += residual * res_scale
    x = scatter_tokens.float()
    if scatter_scale is not None:
        x = x * scatter_scale.unsqueeze(-1)
    convergent.index_add_(
        0, scatter_token_id,
        (x * scatter_token_weight.unsqueeze(-1)).to(OUT_DTYPE))
    return convergent


def run_case(in_name, sp_size, num_tokens):
    (scatter_tokens, scatter_scale, scatter_token_id, scatter_token_weight, residual,
     dispatch_tokens, num_res, res_token_start, res_token_end) = make_inputs(
        sp_size, num_tokens, HIDDEN_SIZE, in_name)

    convergent = torch.zeros(num_tokens, HIDDEN_SIZE, dtype=OUT_DTYPE, device="cuda")

    if scatter_scale is None:
        def run():
            ops.moe_gather(scatter_tokens, scatter_token_id, scatter_token_weight,
                           convergent, residual, RES_SCALE, res_token_start)
    else:
        def run():
            ops.moe_gather(scatter_tokens, scatter_token_id, scatter_token_weight,
                           convergent, residual, RES_SCALE, res_token_start, scatter_scale)

    # accuracy: fresh zero output, single call
    convergent.zero_()
    run()
    torch.cuda.synchronize()
    expected = reference(scatter_tokens, scatter_scale, scatter_token_id, scatter_token_weight,
                         residual, num_tokens, HIDDEN_SIZE, res_token_start, res_token_end,
                         RES_SCALE)
    cos = cosine_similarity(convergent, expected)

    time_ms = benchmark(run)

    in_esz = scatter_tokens.element_size()
    out_esz = torch.tensor([], dtype=OUT_DTYPE).element_size()
    scatter_tokens_bytes = dispatch_tokens * HIDDEN_SIZE * in_esz
    scale_bytes = dispatch_tokens * 4 if scatter_scale is not None else 0
    id_bytes = dispatch_tokens * 4
    w_bytes = dispatch_tokens * 4
    res_bytes = num_res * HIDDEN_SIZE * out_esz
    read_bytes = scatter_tokens_bytes + scale_bytes + id_bytes + w_bytes + res_bytes
    write_bytes = dispatch_tokens * HIDDEN_SIZE * out_esz   # reference 口径 (bf16)
    io_bytes = read_bytes + write_bytes
    gbps = io_bytes / (time_ms * 1e-3) / 1e9

    status = "PASS" if cos > COS_THRESH else "FAIL"
    print(f"[{status}] in={in_name:<4} sp={sp_size:<2} num_tokens={num_tokens:<6} "
          f"D={dispatch_tokens:<7} H={HIDDEN_SIZE} num_res={num_res:<6} "
          f"cos={cos:.8f} time={time_ms:.4f}ms BW={gbps:.1f} GB/s")
    return cos, gbps


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"Device: {torch.cuda.get_device_name(0)}  visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    in_names = [s.strip() for s in os.environ.get("DTYPES", "bf16,int8,fp8").split(",") if s.strip()]
    for n in in_names:
        assert n in IN_DTYPES, f"unknown dtype {n}, choose from {list(IN_DTYPES)}"

    torch.manual_seed(0)
    summary = {}
    for in_name in in_names:
        print("=" * 115)
        print(f"Input dtype: {in_name}")
        print("-" * 115)
        best_bw, best_cfg, all_pass = 0.0, None, True
        for sp_size, num_tokens in SHAPES:
            cos, gbps = run_case(in_name, sp_size, num_tokens)
            if not cos > COS_THRESH:
                all_pass = False
            if gbps > best_bw:
                best_bw, best_cfg = gbps, (sp_size, num_tokens)
        summary[in_name] = (all_pass, best_bw, best_cfg)

    print("=" * 115)
    overall = True
    for in_name, (all_pass, best_bw, best_cfg) in summary.items():
        overall &= all_pass
        print(f"[{in_name:<4}] ALL_PASS={all_pass}  BEST_BW={best_bw:.1f} GB/s "
              f"@ sp={best_cfg[0]} num_tokens={best_cfg[1]}  "
              f"TARGET={TARGET} GB/s {'MET' if best_bw >= TARGET else 'NOT MET'}")
    print(f"OVERALL_PASS={overall}")


if __name__ == "__main__":
    main()
