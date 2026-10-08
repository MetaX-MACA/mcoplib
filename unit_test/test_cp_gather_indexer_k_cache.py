"""Unit test for cp_gather_indexer_k_cache / cp_gather_indexer_k_quant_cache.

Purpose
-------
These two CUDA kernels gather a paged (block-table organized) indexer-K cache
into a contiguous [num_tokens, head_dim] buffer for DeepSeek/GLM-style sparse
attention (Lightning Indexer). They use a __shared__ batch_idx[BLOCK_Y_SIZE]
array: the batch-search loop writes batch_idx[y] from ONE x-lane and ALL x-lanes
of that y later read it. Publishing that write across lanes needs a barrier that
both reconverges the threads and fences shared memory.

The original code used __syncwarp() (default 32-bit mask). On MetaX C600-U the
physical warp is 64 lanes while WARP_SIZE is #defined 32, so __syncwarp() does
not reliably synchronize/fence all participating lanes. For BLOCK_Y_SIZE==32
(chosen by the host launcher when num_tokens>=512, e.g. GLM inputs >2k) each
block has 8*32=256 threads = 4 physical warps, and readers can observe a stale
batch_idx -> wrong batch -> wrong block_idx -> wrong token gathered -> precision
failure.

This test constructs inputs that make any wrong-batch gather DETECTABLE:
  * multi-batch, num_tokens spanning every BLOCK_Y_SIZE bucket incl. >=512 and
    a large >2k case (the regime that fails);
  * sequences that cross cache_block_size boundaries (exercise the two-level
    reverse-paging indirection);
  * a SHUFFLED block_table and per-(block,token) distinct byte patterns so a
    stale batch_idx maps to a physically different, unequal source region.

Because the kernel is a pure gather/copy (no arithmetic), the correct precision
criterion is BIT-EXACT equality against a CPU reference that reproduces the same
block_table + cu_seq_lens indirection. Any mismatch => the race fired.

Run:
  export CUDA_VISIBLE_DEVICES=<idle gpu from mx-smi>
  python unit_test/test_cp_gather_indexer_k_cache.py     # standalone
  pytest  unit_test/test_cp_gather_indexer_k_cache.py    # or via pytest
"""
import time

import pytest
import torch

try:
    import mcoplib._C  # noqa: F401  (registers the cache ops)
except ImportError:
    pass

VEC_BYTES = 16  # kernel gathers float4 (16 bytes) per vector step; head_dim%16==0


# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------
def _nonquant_available():
    return hasattr(torch.ops, "_C_cache_ops") and hasattr(
        torch.ops._C_cache_ops, "cp_gather_indexer_k_cache")


def _quant_available():
    return hasattr(torch.ops, "_C_cache_ops") and hasattr(
        torch.ops._C_cache_ops, "cp_gather_indexer_k_quant_cache")


# ---------------------------------------------------------------------------
# Input construction
# ---------------------------------------------------------------------------
def _make_layout(batch_size, seq_lens, cache_block_size, head_dim, seed):
    """Build cu_seq_lens + a shuffled block_table sized to hold every sequence.

    Returns a dict with cu_seq_lens (int32, len batch+1), block_table
    (int32 [batch, max_blocks_per_seq]), num_tokens, num_blocks (physical),
    and the per-batch block counts.
    """
    g = torch.Generator().manual_seed(seed)
    assert len(seq_lens) == batch_size
    cu = [0]
    for L in seq_lens:
        cu.append(cu[-1] + L)
    num_tokens = cu[-1]

    blocks_per_seq = [(L + cache_block_size - 1) // cache_block_size
                      for L in seq_lens]
    max_bps = max(blocks_per_seq)
    total_needed = sum(blocks_per_seq)
    # Extra physical blocks so the mapping is genuinely sparse/shuffled.
    num_blocks = total_needed + max(4, total_needed // 2)

    # Assign a UNIQUE shuffled physical block to every (batch, logical_block).
    perm = torch.randperm(num_blocks, generator=g).tolist()
    block_table = torch.full((batch_size, max_bps), -1, dtype=torch.int32)
    p = 0
    for b in range(batch_size):
        for j in range(blocks_per_seq[b]):
            block_table[b, j] = perm[p]
            p += 1

    cu_seq_lens = torch.tensor(cu, dtype=torch.int32)
    return {
        "cu_seq_lens": cu_seq_lens,
        "block_table": block_table,
        "num_tokens": num_tokens,
        "num_blocks": num_blocks,
        "blocks_per_seq": blocks_per_seq,
        "seq_lens": list(seq_lens),
    }


def _fill_data_cache(num_blocks, cache_block_size, head_dim, seed, device,
                     dtype=torch.uint8):
    """Physical data cache [num_blocks, cache_block_size, head_dim].

    Every element is a deterministic function of (physical_block, in_block_token,
    dim) so that gathering the WRONG physical block/token yields different values.
    dtype is parametrized: uint8 is the ONE dtype where element==byte, so it hides
    the element/byte unit bug. Multi-byte dtypes (fp16/bf16/fp32) are the real
    network dtypes and expose it.
    """
    g = torch.Generator().manual_seed(seed + 1)
    if dtype == torch.uint8:
        data = torch.randint(
            0, 256, (num_blocks, cache_block_size, head_dim),
            generator=g, dtype=torch.int32).to(torch.uint8)
    else:
        data = torch.randn(
            (num_blocks, cache_block_size, head_dim),
            generator=g, dtype=torch.float32).to(dtype)
    return data.contiguous().to(device)


def _cpu_reference(data_cpu, block_table_cpu, cu_cpu, cache_block_size):
    """Gather [num_tokens, head_dim] exactly like the kernel indirection."""
    batch_size = block_table_cpu.shape[0]
    head_dim = data_cpu.shape[2]
    num_tokens = int(cu_cpu[-1].item())
    out = torch.zeros((num_tokens, head_dim), dtype=data_cpu.dtype)
    for b in range(batch_size):
        s = int(cu_cpu[b].item())
        e = int(cu_cpu[b + 1].item())
        for t in range(e - s):
            phys = int(block_table_cpu[b, t // cache_block_size].item())
            off = t % cache_block_size
            out[s + t] = data_cpu[phys, off]
    return out


# ---------------------------------------------------------------------------
# Kernel runners
# ---------------------------------------------------------------------------
def _run_nonquant(layout, data_cache, head_dim, device):
    cache_block_size = data_cache.shape[1]
    # kv_cache is char [num_blocks, cache_block_size, head_dim]; the kernel reads
    # block_stride=stride(0) and cache_block_size=size(1) from this 3-D tensor.
    kv_cache = data_cache
    dst_k = torch.zeros((layout["num_tokens"], head_dim),
                        dtype=data_cache.dtype, device=device)
    block_table = layout["block_table"].to(device)
    cu = layout["cu_seq_lens"].to(device)
    torch.ops._C_cache_ops.cp_gather_indexer_k_cache(
        kv_cache, dst_k, block_table, cu)
    torch.cuda.synchronize()
    return dst_k


# Quant-cache scale contract (verified empirically against the kernel):
#   * quant_block_size is derived by the host as qbs = head_dim*4/dst_scale.size(1).
#   * the kernel writes ONE fp32 scale per token (only threadIdx.x==0 / head_idx==0
#     participates), taken from the block's scale trailer at group 0's slot, into
#     dst_scale flat index token*head_dim/qbs.
#   * For a dense per-token write (flat index == token) we need head_dim/qbs == 1,
#     i.e. qbs == head_dim, i.e. dst_scale.size(1) == 4. That is the real caller's
#     layout: one scale per token. The scale trailer therefore holds one fp32 per
#     (block, in-block token): [cache_block_size] fp32 per block.
SCALE_COLS = 4  # dst_scale.size(1) -> derived qbs == head_dim -> 1 scale/token


def _build_quant_cache(layout, cache_block_size, head_dim, seed, device):
    """Physical quant cache: per block, [cache_block_size*head_dim] data bytes
    followed by a scale trailer of [cache_block_size] fp32 (one scale per token,
    group 0). Returns (kv_cache_char[num_blocks, block_stride_bytes],
    data_ref[num_blocks,cbs,head_dim] uint8, scale_ref[num_blocks,cbs] fp32,
    block_stride).
    """
    g = torch.Generator().manual_seed(seed + 7)
    num_blocks = layout["num_blocks"]
    data_bytes = cache_block_size * head_dim
    # Trailer must be large enough for the group-0 fp32 of every in-block token.
    # The kernel indexes group 0 at byte (off*head_dim)*4/qbs = off*4 (qbs==hd),
    # so one fp32 per token is exactly cache_block_size*4 bytes.
    scale_bytes = cache_block_size * 4
    block_stride = data_bytes + scale_bytes

    data_ref = torch.randint(
        0, 256, (num_blocks, cache_block_size, head_dim),
        generator=g, dtype=torch.int32).to(torch.uint8)
    scale_ref = (torch.rand((num_blocks, cache_block_size),
                            generator=g, dtype=torch.float32) * 2.0 + 0.1)

    kv = torch.zeros((num_blocks, block_stride), dtype=torch.uint8)
    kv[:, :data_bytes] = data_ref.reshape(num_blocks, data_bytes)
    kv[:, data_bytes:block_stride] = scale_ref.reshape(
        num_blocks, cache_block_size).view(torch.uint8).reshape(
            num_blocks, scale_bytes)
    return (kv.contiguous().to(device), data_ref.to(device),
            scale_ref.to(device), block_stride)


def _run_quant(layout, kv_cache_flat, cache_block_size, head_dim, device):
    num_blocks = layout["num_blocks"]
    block_stride = kv_cache_flat.shape[1]
    # Reshape to 3-D [num_blocks, cache_block_size, ...] so size(1)=cbs and
    # stride(0)=block_stride, matching what the kernel reads. The kernel treats
    # everything as char and computes byte offsets internally, so the 3rd dim
    # just needs to make stride(0)==block_stride and size(1)==cache_block_size.
    assert block_stride % cache_block_size == 0
    kv3 = kv_cache_flat.view(num_blocks, cache_block_size,
                             block_stride // cache_block_size)
    num_tokens = layout["num_tokens"]
    dst_k = torch.zeros((num_tokens, head_dim),
                        dtype=torch.uint8, device=device)
    # dst_scale.size(1)==SCALE_COLS(=4) -> derived qbs==head_dim -> the kernel
    # writes dst_scale flat index == token, i.e. one scale per token, dense.
    dst_scale = torch.zeros((num_tokens, SCALE_COLS),
                            dtype=torch.float32, device=device)
    block_table = layout["block_table"].to(device)
    cu = layout["cu_seq_lens"].to(device)
    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        kv3, dst_k, dst_scale, block_table, cu)
    torch.cuda.synchronize()
    # collapse the [num_tokens, 4] view to the dense per-token scale (flat==token)
    dst_scale_flat = dst_scale.reshape(-1)[:num_tokens]
    return dst_k, dst_scale_flat


def _cpu_reference_quant(data_ref_cpu, scale_ref_cpu, block_table_cpu, cu_cpu,
                         cache_block_size):
    batch_size = block_table_cpu.shape[0]
    head_dim = data_ref_cpu.shape[2]
    num_tokens = int(cu_cpu[-1].item())
    out = torch.zeros((num_tokens, head_dim), dtype=data_ref_cpu.dtype)
    out_s = torch.zeros((num_tokens,), dtype=torch.float32)
    for b in range(batch_size):
        s = int(cu_cpu[b].item())
        e = int(cu_cpu[b + 1].item())
        for t in range(e - s):
            phys = int(block_table_cpu[b, t // cache_block_size].item())
            off = t % cache_block_size
            out[s + t] = data_ref_cpu[phys, off]
            out_s[s + t] = scale_ref_cpu[phys, off]
    return out, out_s


# ---------------------------------------------------------------------------
# Case matrix: cover every BLOCK_Y_SIZE bucket, including the failing >=512 and
# a large >2k case. seq lens deliberately cross cache_block_size boundaries.
# (batch_size, [seq_lens], cache_block_size, head_dim)
# ---------------------------------------------------------------------------
CASES = [
    # small buckets (BLOCK_Y_SIZE 1/2/4/8 -> <=1 physical warp, historically OK)
    (1, [7], 16, 128),
    (2, [15, 17], 16, 128),
    (3, [40, 33, 25], 16, 128),
    (4, [60, 70, 55, 65], 32, 128),
    # boundary + medium buckets
    (4, [130, 90, 140, 100], 64, 128),
    (5, [200, 150, 220, 180, 90], 64, 128),
    # >=512 -> BLOCK_Y_SIZE=32 == 4 physical warps: the FAILING regime
    (4, [200, 180, 260, 160], 64, 128),          # 800 tokens
    (6, [300, 250, 200, 350, 150, 220], 64, 128),  # 1470 tokens
    # >2k GLM-like case, multi-batch, crossing 64-token blocks
    (4, [700, 650, 800, 550], 64, 128),          # 2700 tokens
    (3, [1200, 1000, 900], 128, 128),            # 3100 tokens
    # head_dim variations (must stay multiple of 16 bytes)
    (4, [520, 540, 500, 560], 64, 64),           # 2120 tokens, head_dim=64
    (2, [1100, 1300], 64, 256),                  # 2400 tokens, head_dim=256
]


def _device():
    assert torch.cuda.is_available(), "CUDA not available"
    return "cuda"


def _run_one_nonquant(case, seed=1234, verbose=True, dtype=torch.bfloat16):
    batch_size, seq_lens, cbs, head_dim = case
    esz = torch.tensor([], dtype=dtype).element_size()
    assert (head_dim * esz) % VEC_BYTES == 0, (
        "head_dim*element_size must be a multiple of 16 bytes (float4 copy)")
    device = _device()
    layout = _make_layout(batch_size, seq_lens, cbs, head_dim, seed)
    data = _fill_data_cache(layout["num_blocks"], cbs, head_dim, seed, device,
                            dtype=dtype)

    # warmup + timing
    dst = _run_nonquant(layout, data, head_dim, device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        dst = _run_nonquant(layout, data, head_dim, device)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3

    ref = _cpu_reference(data.cpu(), layout["block_table"],
                         layout["cu_seq_lens"], cbs)
    got = dst.cpu()
    exact = torch.equal(got, ref)
    mism = int((got != ref).any(dim=1).sum().item())
    bytes_moved = layout["num_tokens"] * head_dim * esz  # 1R+1W of head_dim
    gbps = (2 * bytes_moved) / (ms * 1e-3) / 1e9
    if verbose:
        status = "OK  " if exact else "FAIL"
        dt = str(dtype).replace("torch.", "")
        print(f"[nonquant {status}] dtype={dt:8s} bs={batch_size} "
              f"tok={layout['num_tokens']:5d} cbs={cbs} hd={head_dim} | "
              f"mismatched_tokens={mism:5d} | {ms:.3f} ms | {gbps:6.1f} GB/s")
    return exact, mism, layout["num_tokens"]


def _run_one_quant(case, seed=1234, verbose=True):
    batch_size, seq_lens, cbs, head_dim = case
    assert head_dim % VEC_BYTES == 0
    device = _device()
    layout = _make_layout(batch_size, seq_lens, cbs, head_dim, seed)
    kv, data_ref, scale_ref, _ = _build_quant_cache(
        layout, cbs, head_dim, seed, device)

    dst_k, dst_s = _run_quant(layout, kv, cbs, head_dim, device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    iters = 20
    for _ in range(iters):
        dst_k, dst_s = _run_quant(layout, kv, cbs, head_dim, device)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3

    ref_k, ref_s = _cpu_reference_quant(
        data_ref.cpu(), scale_ref.cpu(), layout["block_table"],
        layout["cu_seq_lens"], cbs)
    got_k = dst_k.cpu()
    got_s = dst_s.cpu()
    exact_k = torch.equal(got_k, ref_k)
    exact_s = torch.allclose(got_s, ref_s, rtol=0, atol=0)
    exact = exact_k and exact_s
    mism = int((got_k != ref_k).any(dim=1).sum().item())
    mism_s = int((got_s != ref_s).sum().item())
    bytes_moved = layout["num_tokens"] * (head_dim + 4)  # data + 1 fp32 scale
    gbps = (2 * bytes_moved) / (ms * 1e-3) / 1e9
    if verbose:
        status = "OK  " if exact else "FAIL"
        print(f"[ quant   {status}] bs={batch_size} tok={layout['num_tokens']:5d} "
              f"cbs={cbs} hd={head_dim} | mism_data={mism:5d} mism_scale={mism_s:5d}"
              f" | {ms:.3f} ms | {gbps:6.1f} GB/s")
    return exact, mism + mism_s, layout["num_tokens"]


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------
# The element/byte bug manifests as correctness rate 1/element_size, so multi-byte
# dtypes MUST be tested. uint8 alone (element==byte) cannot catch it.
NONQUANT_DTYPES = [torch.uint8, torch.float16, torch.bfloat16, torch.float32]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not _nonquant_available(),
                    reason="cp_gather_indexer_k_cache not available")
@pytest.mark.parametrize("dtype", NONQUANT_DTYPES)
@pytest.mark.parametrize("case", CASES)
def test_cp_gather_indexer_k_cache(case, dtype):
    exact, mism, _ = _run_one_nonquant(case, dtype=dtype)
    assert exact, (f"non-quant gather mismatch: {mism} wrong tokens for "
                   f"case {case} dtype {dtype}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(not _quant_available(),
                    reason="cp_gather_indexer_k_quant_cache not available")
@pytest.mark.parametrize("case", CASES)
def test_cp_gather_indexer_k_quant_cache(case):
    exact, mism, _ = _run_one_quant(case)
    assert exact, f"quant gather mismatch: {mism} wrong rows for case {case}"


# ---------------------------------------------------------------------------
# Standalone runner (no pytest) -- prints a table and a final PASS/FAIL summary.
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("CUDA not available; abort")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES')}")
    nq_ok = q_ok = True
    print("\n--- non-quant: cp_gather_indexer_k_cache (all dtypes) ---")
    if _nonquant_available():
        for dt in (torch.uint8, torch.float16, torch.bfloat16, torch.float32):
            for c in CASES:
                e, _, _ = _run_one_nonquant(c, dtype=dt)
                nq_ok = nq_ok and e
    else:
        print("SKIP: op not available")
        nq_ok = False
    print("\n--- quant: cp_gather_indexer_k_quant_cache ---")
    if _quant_available():
        for c in CASES:
            e, _, _ = _run_one_quant(c)
            q_ok = q_ok and e
    else:
        print("SKIP: op not available")
        q_ok = False
    print("\n==================== SUMMARY ====================")
    print(f"non-quant : {'ALL PASS' if nq_ok else 'FAILED'}")
    print(f"quant     : {'ALL PASS' if q_ok else 'FAILED'}")
    ok = nq_ok and q_ok
    print("RESULT    :", "ALL PASS (no precision anomaly, no kernel trap)"
          if ok else "FAILURES DETECTED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
