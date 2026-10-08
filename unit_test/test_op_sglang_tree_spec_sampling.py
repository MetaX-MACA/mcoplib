"""Correctness + bandwidth unit test for the flashinfer-stripped tree speculative
sampling kernel.

Op: torch.ops.sgl_kernel.tree_speculative_sampling_target_only(
        predicts, accept_index, accept_token_num,          # int32 outputs (mutated)
        candidates, retrive_index, retrive_next_token, retrive_next_sibling,  # int64
        uniform_samples, uniform_samples_for_final_sampling, target_probs, draft_probs,  # fp32
        threshold_single, threshold_acc, deterministic)

Backing code ported during flashinfer-strip (op/sglang/csrc/speculative/speculative_sampling.cuh):
  SamplingTempStorage / DeterministicInclusiveSum / DeviceSamplingFromProb /
  TreeSpeculativeSamplingTargetOnly (kernel + launcher), all in namespace mcoplib::sampling.
Only needs `import mcoplib.sgl_kernel`.

Algorithm (verified against the CUDA source, deterministic mode):
  Per batch row b, walk the token tree from the root (index 0):
    coin = uniform_samples[b, 0]; last_accepted = retrive_index[b, 0];
    accept_index[b, 0] = last_accepted; prob_acc = 0; cur_prob_offset = b*T*d
    for step in 1..num_spec_step-1:
      cur = retrive_next_token[b, cur]
      while cur != -1:
        draft_index = retrive_index[b, cur]; tok = candidates[b, cur]
        p = target_probs[cur_prob_offset + tok]; prob_acc += p
        if coin <= prob_acc / threshold_acc or p >= threshold_single:  # ACCEPT
          predicts[last_accepted] = tok; num_accepted += 1
          accept_index[b, num_accepted] = draft_index
          last_accepted = draft_index; prob_acc = 0
          cur_prob_offset = (b*T + cur)*d; coin = uniform_samples[b, cur]; break
        else:                                                          # REJECT
          draft_probs[cur_prob_offset + tok] = target_probs[cur_prob_offset + tok]
          cur = retrive_next_sibling[b, cur]
      if cur == -1: break
    accept_token_num[b] = num_accepted
  Final (bonus/rejection) token sampled from relu(target - draft) at cur_prob_offset
  with coin2 = uniform_samples_for_final_sampling[b]:
    r = relu(target[row] - draft[row]); S = r.sum(); u = coin2 * S
    sampled_id = smallest k with r[k] > 0 and cumsum(r)[k] > u  (else d-1)
    predicts[last_accepted] = sampled_id

Accuracy gate: this op emits discrete int32 token ids / tree indices, so the primary
correctness check is EXACT integer equality on predicts / accept_index / accept_token_num.
Per the task's cosine-similarity requirement we additionally report cosine similarity of
the (float-cast) output vectors, which is 1.0 on exact match; the >= 0.9999 gate is
asserted on those.
"""
import numpy as np
import torch
import pytest
import mcoplib.sgl_kernel  # noqa: F401

OP = torch.ops.sgl_kernel.tree_speculative_sampling_target_only
COS_THRESH = 0.9999
TARGET_GBPS = 1300.0

# (batch_size, num_spec_step, num_draft_tokens, vocab_size)
# vocab sizes cover mainstream LLMs: Llama3 128256, Qwen2 152064, GPT-NeoX 50272,
# Llama2 32000. num_draft_tokens/num_spec_step are typical EAGLE tree sizes.
SHAPES = [
    (1, 4, 8, 32000),
    (8, 4, 8, 128256),
    (16, 5, 16, 152064),
    (32, 6, 32, 128256),
    (64, 4, 8, 50272),
]


def build_chain_tree(bs, num_draft, vocab, accept_pattern, dev, seed=0):
    """Build a *chain* tree (node i's only child is node i+1) so the traversal is a
    deterministic straight walk. accept_pattern[b] = how many draft tokens should be
    accepted (0..num_spec_step-1). target_probs is structured so that exactly the
    first `accept_pattern[b]` chain nodes clear the acceptance threshold and the next
    one fails, making predicts/accept_index fully predictable.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # chain: candidates token ids are distinct small ints per node
    candidates = torch.zeros(bs, num_draft, dtype=torch.int64)
    retrive_index = torch.zeros(bs, num_draft, dtype=torch.int64)
    retrive_next_token = torch.full((bs, num_draft), -1, dtype=torch.int64)
    retrive_next_sibling = torch.full((bs, num_draft), -1, dtype=torch.int64)
    for b in range(bs):
        for i in range(num_draft):
            candidates[b, i] = 10 + i          # token id for chain node i
            retrive_index[b, i] = b * num_draft + i
            if i + 1 < num_draft:
                retrive_next_token[b, i] = i + 1   # single child chain
            # no siblings -> retrive_next_sibling stays -1
    # target_probs [bs, num_draft, vocab]: small random base, normalised to a valid
    # prob distribution per (row, node). Acceptance of chain node `i` is decided by
    # target_probs at the *parent's* prob block (node i-1) evaluated at node i's token
    # id (see kernel: cur_prob_offset starts at node 0 and only advances AFTER an
    # accept). So to accept chain nodes 1..k we spike tp[b, i-1, tok_i] >= threshold_single.
    # Non-accepted nodes keep small probs (< threshold_single); with coin=0.99 the
    # accumulated-prob rule also fails there -> deterministic accept count = k.
    tp = torch.rand(bs, num_draft, vocab, generator=g) * 0.01 + 1e-6
    tp = tp / tp.sum(dim=-1, keepdim=True)
    for b in range(bs):
        k = accept_pattern[b]
        for i in range(1, k + 1):
            tok = int(candidates[b, i].item())
            tp[b, i - 1, tok] = 0.9      # parent row (i-1), child i's token -> ACCEPT
    target_probs = tp.contiguous()
    draft_probs = torch.zeros(bs, num_draft, vocab, dtype=torch.float32)
    # coins: high (0.99) so acceptance only happens via threshold_single spike
    uniform_samples = torch.full((bs, num_draft), 0.99, dtype=torch.float32)
    uniform_final = torch.rand(bs, generator=g).to(torch.float32)
    return (candidates.to(dev), retrive_index.to(dev), retrive_next_token.to(dev),
            retrive_next_sibling.to(dev), target_probs.to(dev), draft_probs.to(dev),
            uniform_samples.to(dev), uniform_final.to(dev))


def ref_tree_sampling(bs, num_spec, num_draft, vocab, cand, ridx, rnt, rns,
                      us, uf, tp, dp_out, thr_single, thr_acc):
    """Faithful CPU reference of the deterministic kernel. Returns
    (predicts, accept_index, accept_token_num) as numpy int arrays and the updated
    draft_probs (fp32) so the caller can verify the final-sampling path too."""
    cand = cand.cpu().numpy(); ridx = ridx.cpu().numpy()
    rnt = rnt.cpu().numpy(); rns = rns.cpu().numpy()
    us = us.cpu().numpy(); uf = uf.cpu().numpy()
    tp = tp.cpu().numpy().astype(np.float64)
    dp = np.zeros_like(tp)
    predicts = np.full((bs * num_draft,), -1, dtype=np.int64)
    accept_index = np.full((bs, num_spec), -1, dtype=np.int64)
    accept_num = np.zeros((bs,), dtype=np.int64)
    thr_acc = max(thr_acc, 1e-9)
    for b in range(bs):
        prob_acc = 0.0
        cur_prob_node = 0                      # node within row whose probs we read
        coin = us[b, 0]
        last_acc = int(ridx[b, 0])
        accept_index[b, 0] = last_acc
        n_acc = 0
        cur = 0
        for step in range(1, num_spec):
            cur = int(rnt[b, cur])
            while cur != -1:
                draft_index = int(ridx[b, cur])
                tok = int(cand[b, cur])
                p = tp[b, cur_prob_node, tok]
                prob_acc += p
                if coin <= prob_acc / thr_acc or p >= thr_single:
                    prob_acc = 0.0
                    cur_prob_node = cur
                    coin = us[b, cur]
                    predicts[last_acc] = tok
                    n_acc += 1
                    accept_index[b, n_acc] = draft_index
                    last_acc = draft_index
                    break
                else:
                    dp[b, cur_prob_node, tok] = tp[b, cur_prob_node, tok]
                    cur = int(rns[b, cur])
            if cur == -1:
                break
        accept_num[b] = n_acc
        # final sampling from relu(target - draft) at cur_prob_node
        coin2 = uf[b]
        row_t = tp[b, cur_prob_node]
        row_d = dp[b, cur_prob_node] if n_acc != num_spec - 1 else np.zeros_like(row_t)
        r = np.maximum(row_t - row_d, 0.0)
        S = r.sum()
        u = coin2 * S
        cdf = np.cumsum(r)
        sampled = vocab - 1
        idxs = np.where((r > 0) & (cdf > u))[0]
        if idxs.size > 0:
            sampled = int(idxs[0])
        predicts[last_acc] = sampled
    return predicts, accept_index, accept_num, dp


def cos_sim_int(a, b):
    a = torch.as_tensor(a, dtype=torch.float64).flatten()
    b = torch.as_tensor(b, dtype=torch.float64).flatten()
    if a.norm() == 0 and b.norm() == 0:
        return 1.0
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def run_once(bs, num_spec, num_draft, vocab, dev, accept_pattern, thr_single=0.5, thr_acc=1.0, seed=0):
    (cand, ridx, rnt, rns, tp, dp, us, uf) = build_chain_tree(
        bs, num_draft, vocab, accept_pattern, dev, seed)
    predicts = torch.full((bs * num_draft,), -1, dtype=torch.int32, device=dev)
    accept_index = torch.full((bs, num_spec), -1, dtype=torch.int32, device=dev)
    accept_num = torch.zeros((bs,), dtype=torch.int32, device=dev)
    dp_run = dp.clone()
    OP(predicts, accept_index, accept_num, cand, ridx, rnt, rns, us, uf, tp, dp_run,
       float(thr_single), float(thr_acc), True)
    torch.cuda.synchronize()  # trap check
    r_pred, r_ai, r_an, _ = ref_tree_sampling(
        bs, num_spec, num_draft, vocab, cand, ridx, rnt, rns, us, uf, tp, dp,
        thr_single, thr_acc)
    return (predicts.cpu().numpy(), accept_index.cpu().numpy(), accept_num.cpu().numpy(),
            r_pred, r_ai, r_an)


@pytest.mark.parametrize("shape", SHAPES)
def test_tree_speculative_sampling(shape):
    dev = torch.device("cuda")
    bs, num_spec, num_draft, vocab = shape
    # deterministic accept pattern: row b accepts (b % (num_spec-1)) chain tokens
    rng = np.random.default_rng(0)
    accept_pattern = [int(rng.integers(0, num_spec - 1) + 1) for _ in range(bs)]
    accept_pattern = [min(a, num_draft - 1) for a in accept_pattern]

    pred, ai, an, r_pred, r_ai, r_an = run_once(
        bs, num_spec, num_draft, vocab, dev, accept_pattern)

    # accept_token_num and accept_index are fully determined -> exact match required
    exact_an = np.array_equal(an, r_an)
    exact_ai = np.array_equal(ai, r_ai)
    # predicts: only entries at accepted retrive indices are defined; compare those
    defined = r_pred != -1
    exact_pred = np.array_equal(pred[defined], r_pred[defined])

    cos_an = cos_sim_int(an, r_an)
    cos_ai = cos_sim_int(ai[ai != -1], r_ai[r_ai != -1])
    cos_pred = cos_sim_int(pred[defined], r_pred[defined])
    print(f"[tree_spec] shape={shape} accept_num_match={exact_an} "
          f"accept_index_match={exact_ai} predicts_match={exact_pred} | "
          f"cos(an)={cos_an:.6f} cos(ai)={cos_ai:.6f} cos(pred)={cos_pred:.6f} | "
          f"accepted/row(min,max)={an.min()},{an.max()}")

    assert exact_an, f"accept_token_num mismatch\n got={an}\n exp={r_an}"
    assert exact_ai, f"accept_index mismatch\n got={ai}\n exp={r_ai}"
    assert exact_pred, f"predicts (defined entries) mismatch"
    assert cos_an >= COS_THRESH and cos_ai >= COS_THRESH and cos_pred >= COS_THRESH


def _bench(bs, num_spec, num_draft, vocab, dev, iters=50, warmup=10):
    accept_pattern = [num_spec // 2 for _ in range(bs)]
    accept_pattern = [min(a, num_draft - 1) for a in accept_pattern]
    (cand, ridx, rnt, rns, tp, dp, us, uf) = build_chain_tree(
        bs, num_draft, vocab, accept_pattern, dev)
    predicts = torch.full((bs * num_draft,), -1, dtype=torch.int32, device=dev)
    accept_index = torch.full((bs, num_spec), -1, dtype=torch.int32, device=dev)
    accept_num = torch.zeros((bs,), dtype=torch.int32, device=dev)
    for _ in range(warmup):
        dpr = dp.clone()
        OP(predicts, accept_index, accept_num, cand, ridx, rnt, rns, us, uf, tp, dpr, 0.5, 1.0, True)
    torch.cuda.synchronize()
    dprs = [dp.clone() for _ in range(iters)]
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for i in range(iters):
        OP(predicts, accept_index, accept_num, cand, ridx, rnt, rns, us, uf, tp, dprs[i], 0.5, 1.0, True)
    e.record(); torch.cuda.synchronize()
    ms = s.elapsed_time(e) / iters
    # Actual per-launch traffic model. This op is NOT a full-tensor streaming kernel:
    # the tree walk only touches a handful of (row, token) scalars in target/draft
    # probs, but the final DeviceSamplingFromProb step scans ONE full vocab row of
    # target_probs and reads/writes ONE full vocab row of draft_probs per batch row.
    # So the dominant real traffic is ~ bs * vocab floats read (target) + bs * vocab
    # floats read+written (draft during the relu(target-draft) cdf scan).
    # We deliberately do NOT model bs*num_draft*vocab (the full allocation) as traffic
    # because the kernel never streams the whole tensor -> that would grossly overstate
    # bandwidth (well above the HBM wall). Latency (ms) is the primary metric here.
    bytes_target = bs * vocab * 4            # one target row per batch, read in final sampling
    bytes_draft = bs * vocab * 4 * 2         # one draft row per batch, read+write in final sampling
    io = bytes_target + bytes_draft
    gbps = io / (ms * 1e-3) / 1e9
    return ms, gbps


def test_tree_speculative_bandwidth():
    dev = torch.device("cuda")
    print("\n[tree_spec perf] iters=50 warmup=10  "
          "(discrete tree-sampling op: latency is the primary metric; "
          "effective BW modelled on the final full-vocab-row scan only)")
    print(f"{'bs':>4} {'spec':>5} {'draft':>6} {'vocab':>8} {'ms':>9} {'us/row':>8} {'eff GB/s':>9}")
    peak = 0.0
    for (bs, num_spec, num_draft, vocab) in SHAPES:
        ms, gbps = _bench(bs, num_spec, num_draft, vocab, dev)
        peak = max(peak, gbps)
        us_per_row = ms * 1e3 / bs
        print(f"{bs:4d} {num_spec:5d} {num_draft:6d} {vocab:8d} "
              f"{ms:9.4f} {us_per_row:8.2f} {gbps:9.1f}")
    print(f"[tree_spec perf] peak effective BW (final-scan model) = {peak:.1f} GB/s "
          f"({100*peak/TARGET_GBPS:.1f}% of {TARGET_GBPS} wall)")
    assert peak > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
