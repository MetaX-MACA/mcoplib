"""Unit test + bandwidth benchmark for mcoplib.op.store_kv_cache_cuda_interface.

Cache layout (4D): k_cache[num_blocks, kv_head_num, block_size, head_dim]

Run:
  CUDA_VISIBLE_DEVICES='0' python unit_test/test_op_store_kv_cache_cuda_interface.py
"""
import os, torch
import mcoplib.op as ops

DEVICE = "cuda"
COS_SIM_THRESHOLD = 0.9999
TARGET_GBPS = 1200.0
Q_HEAD_NUM = 80
KV_HEAD_NUM = 8
HEAD_DIM = 128
BLOCK_SIZE = 512


def float_to_int8_rn_vec(x):
    return torch.clamp(torch.round(torch.clamp(x, -127.0, 127.0)), -127, 127).to(torch.int8)


def reference_store_kv(packed_qkv, q_lens, accum_q_lens, cache_lens,
                       cache_slot_ids, k_cache, v_cache, k_scale, v_scale,
                       batch_size, q_head_num, kv_head_num):
    """Vectorized reference on CPU to avoid GPU OOM."""
    packed_qkv = packed_qkv.cpu()
    k_scale = k_scale.cpu()
    v_scale = v_scale.cpu()
    q_lens = q_lens.cpu()
    accum_q_lens = accum_q_lens.cpu()
    cache_lens = cache_lens.cpu()
    cache_slot_ids = cache_slot_ids.cpu()

    for b in range(batch_size):
        q_len = int(q_lens[b].item())
        q_offset = int(accum_q_lens[b].item())
        cur_cache_len = int(cache_lens[b].item())
        cur_slot_id = int(cache_slot_ids[b].item())
        if q_len == 0:
            continue

        k_src = packed_qkv[q_offset:q_offset + q_len,
                           q_head_num:q_head_num + kv_head_num, :].float()
        v_src = packed_qkv[q_offset:q_offset + q_len,
                           q_head_num + kv_head_num:q_head_num + 2 * kv_head_num, :].float()

        k_int8 = float_to_int8_rn_vec(k_src * k_scale.unsqueeze(0))
        v_int8 = float_to_int8_rn_vec(v_src * v_scale.unsqueeze(0))

        k_cache[cur_slot_id, :, cur_cache_len:cur_cache_len + q_len, :] = k_int8.permute(1, 0, 2)
        v_cache[cur_slot_id, :, cur_cache_len:cur_cache_len + q_len, :] = v_int8.permute(1, 0, 2)


def make_test_data(batch_size, cache_len, q_len, seed=42):
    torch.manual_seed(seed)
    total_tokens = q_len + 10
    packed_qkv = torch.randn(total_tokens, Q_HEAD_NUM + 2 * KV_HEAD_NUM,
                             HEAD_DIM, dtype=torch.bfloat16, device=DEVICE)
    k_scale = (torch.rand(KV_HEAD_NUM, HEAD_DIM, dtype=torch.float32, device=DEVICE) + 0.1).abs()
    v_scale = (torch.rand(KV_HEAD_NUM, HEAD_DIM, dtype=torch.float32, device=DEVICE) + 0.1).abs()

    q_lens = torch.full((batch_size,), q_len, dtype=torch.int32, device=DEVICE)
    accum_q_lens = torch.zeros(batch_size, dtype=torch.int32, device=DEVICE)
    cache_lens = torch.full((batch_size,), cache_len, dtype=torch.int32, device=DEVICE)
    cache_slot_ids = torch.arange(batch_size, dtype=torch.int32, device=DEVICE)

    max_pos = cache_len + q_len
    num_blocks = max((max_pos + BLOCK_SIZE - 1) // BLOCK_SIZE, batch_size)
    cb = max(num_blocks * BLOCK_SIZE, max_pos)

    k_cache = torch.zeros(num_blocks, KV_HEAD_NUM, cb, HEAD_DIM,
                          dtype=torch.int8, device=DEVICE)
    v_cache = torch.zeros(num_blocks, KV_HEAD_NUM, cb, HEAD_DIM,
                          dtype=torch.int8, device=DEVICE)

    return dict(packed_qkv=packed_qkv, q_lens=q_lens, accum_q_lens=accum_q_lens,
                cache_lens=cache_lens, cache_slot_ids=cache_slot_ids,
                k_cache=k_cache, v_cache=v_cache, k_scale=k_scale, v_scale=v_scale,
                batch_size=batch_size, cache_len=cache_len, q_len=q_len)


def run_kernel(data):
    ops.store_kv_cache_cuda_interface(
        data["packed_qkv"], data["q_lens"], data["accum_q_lens"],
        data["cache_lens"], data["cache_slot_ids"],
        data["k_cache"], data["v_cache"],
        data["k_scale"], data["v_scale"],
        data["batch_size"], Q_HEAD_NUM, KV_HEAD_NUM)


def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0).item()


def compute_effective_bytes(batch_size, q_len):
    return (batch_size * q_len * KV_HEAD_NUM * 2 * HEAD_DIM * 2 +
            2 * KV_HEAD_NUM * HEAD_DIM * 4 +
            2 * batch_size * q_len * KV_HEAD_NUM * HEAD_DIM * 1)


def bench_kernel(data, warmup=10, rep=100):
    for _ in range(warmup):
        run_kernel(data)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        run_kernel(data)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median_ms = times[len(times) // 2]
    eff = compute_effective_bytes(data["batch_size"], data["q_len"])
    return median_ms, eff / (median_ms * 1e-3) / 1e9


def run_test_case(batch_size, cache_len, q_len, case_idx=0):
    data = make_test_data(batch_size, cache_len, q_len, seed=42 + case_idx)

    k_ref = data["k_cache"].cpu().clone()
    v_ref = data["v_cache"].cpu().clone()
    reference_store_kv(data["packed_qkv"], data["q_lens"], data["accum_q_lens"],
                       data["cache_lens"], data["cache_slot_ids"],
                       k_ref, v_ref, data["k_scale"], data["v_scale"],
                       batch_size, Q_HEAD_NUM, KV_HEAD_NUM)

    # Only compare batch 0's written region
    cur_cl = int(data["cache_lens"][0].item())
    ql = data["q_len"]
    slot0 = int(data["cache_slot_ids"][0].item())
    k_written = k_ref[slot0, :, cur_cl:cur_cl+ql, :]
    v_written = v_ref[slot0, :, cur_cl:cur_cl+ql, :]

    run_kernel(data)
    torch.cuda.synchronize()

    k_written_gpu = data["k_cache"][slot0, :, cur_cl:cur_cl+ql, :]
    v_written_gpu = data["v_cache"][slot0, :, cur_cl:cur_cl+ql, :]

    k_sim = cosine_similarity(k_written.cpu(), k_written_gpu.cpu())
    v_sim = cosine_similarity(v_written.cpu(), v_written_gpu.cpu())
    sim = min(k_sim, v_sim)

    median_ms, gbps = bench_kernel(data)
    ok = "OK" if sim >= COS_SIM_THRESHOLD else "FAIL"
    print(f"[store_kv] B={batch_size:2d} C={cache_len:6d} Q={q_len:5d} "
          f"cos_sim={sim:.6f} {ok}  {median_ms:8.4f} ms  {gbps:8.1f} GB/s")
    del k_ref, v_ref, k_written, v_written, k_written_gpu, v_written_gpu
    torch.cuda.empty_cache()
    return sim, gbps


def main():
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    print(f"CUDA_VISIBLE_DEVICES={vis!r}  device_count={torch.cuda.device_count()}")
    assert torch.cuda.is_available()
    print(f"config: q_head_num={Q_HEAD_NUM} kv_head_num={KV_HEAD_NUM} head_dim={HEAD_DIM}")
    print("=" * 96)
    max_gbps, all_pass, case_idx = 0.0, True, 0

    print("--- Prefill mode ---")
    for bs, cl, ql in [(1,0,1024),(1,0,2048),(1,0,4096),(1,0,8192),
                        (1,0,10240),(1,0,16384),(1,0,32768)]:
        s, g = run_test_case(bs, cl, ql, case_idx)
        max_gbps = max(max_gbps, g); all_pass &= (s >= COS_SIM_THRESHOLD); case_idx += 1
    for bs, cl, ql in [(1,0,1024),(1,0,2048),(1,0,4096),(1,0,8192),
                        (1,0,10240),(1,0,16384),(1,0,32768),
                        (1,4096,4096),(1,5120,5120),(1,8192,8192),
                        (1,16384,4096),(1,28672,4096)]:
        s, g = run_test_case(bs, cl, ql, case_idx)
        max_gbps = max(max_gbps, g); all_pass &= (s >= COS_SIM_THRESHOLD); case_idx += 1

    print("--- Decode mode ---")
    for bs, cl, ql in [(16,1024,1),(16,2048,1),(16,4096,1),(16,8192,1),
                        (16,10240,1),(16,16384,1),(16,32768,1),
                        (16,1024,4),(16,4096,4),(16,8192,4),
                        (16,10240,4),(16,16384,4),(16,32768,4)]:
        s, g = run_test_case(bs, cl, ql, case_idx)
        max_gbps = max(max_gbps, g); all_pass &= (s >= COS_SIM_THRESHOLD); case_idx += 1

    print("=" * 96)
    print(f"Accuracy: {'ALL PASS' if all_pass else 'FAILURES'} (cos_sim >= {COS_SIM_THRESHOLD})")
    print(f"Peak bandwidth: {max_gbps:.1f} GB/s (target {TARGET_GBPS:.0f} -> "
          f"{'REACHED' if max_gbps >= TARGET_GBPS else 'NOT reached'})")
    assert all_pass, "accuracy check failed"


if __name__ == "__main__":
    main()
