"""Test for `FusedAttentionPrepare` (op/glm_attention_prepare.cu).

The kernel fuses per-head RMSNorm (with weight) + GPT-NeoX partial RoPE:

1. Split the packed ``qkv`` into query heads and kv heads:
   qkv layout = [num_tokens, (num_heads + num_kv_heads) * head_dim].
2. For every head, apply RMSNorm over the head dim then scale by ``weight``.
3. Apply NeoX RoPE to the first ``head_dim // 2`` elements only (pairing
   distance ``head_dim // 4``), frequency dim = int(head_dim * partial_rotary_factor).
   The second half of each head is left un-rotated.

Only bfloat16 is supported.
"""

import math

import torch

import mcoplib.op as ops


def _rms_norm(x, weight, eps):
    # RMSNorm over the last dim (head dim), then weight.
    rms = torch.rsqrt((x.float() ** 2).mean(-1, keepdim=True) + eps)
    return x.float() * rms * weight.float()


def _apply_partial_neox_rope(x, pos, base, rot_dim, head_dim):
    """Apply the kernel's partial NeoX RoPE in place (float view).

    Only the first head_dim//2 elements are rotated, as pairs
    (i, i + head_dim//4); the rest is untouched.
    """
    x = x.clone()
    half = head_dim // 4  # fixed pairing distance inside the kernel
    for i in range(half):
        inv_freq = 1.0 / (base ** (2.0 * i / rot_dim))
        angle = pos * inv_freq
        c = math.cos(angle)
        s = math.sin(angle)
        a = x[i].item()
        b = x[half + i].item()
        x[i] = a * c - b * s
        x[half + i] = b * c + a * s
    return x


def ref_fused_attention_prepare(qkv, weight, positions, num_heads,
                                num_kv_heads, head_dim, base, eps,
                                partial_rotary_factor):
    num_tokens = qkv.shape[0]
    rot_dim = int(head_dim * partial_rotary_factor)
    qkv = qkv.float()
    weight = weight.float()
    out_q = torch.zeros(num_tokens, num_heads * head_dim)
    out_kv = torch.zeros(num_tokens, num_kv_heads * head_dim)

    for t in range(num_tokens):
        pos = float(positions[t])
        for h in range(num_heads):
            q = qkv[t, h * head_dim:(h + 1) * head_dim]
            q = _rms_norm(q, weight, eps)
            q = _apply_partial_neox_rope(q, pos, base, rot_dim, head_dim)
            out_q[t, h * head_dim:(h + 1) * head_dim] = q

            if h < num_kv_heads:
                k_start = num_heads * head_dim + h * head_dim
                k = qkv[t, k_start:k_start + head_dim]
                k = _rms_norm(k, weight, eps)
                k = _apply_partial_neox_rope(k, pos, base, rot_dim, head_dim)
                out_kv[t, h * head_dim:(h + 1) * head_dim] = k

    return out_q.to(torch.bfloat16), out_kv.to(torch.bfloat16)


def run_case(num_tokens, num_heads, num_kv_heads, head_dim, base,
             max_position_embeddings, rms_norm_eps, partial_rotary_factor):
    device = "cuda"
    qkv_stride = (num_heads + num_kv_heads) * head_dim
    qkv = torch.randn(num_tokens, qkv_stride, dtype=torch.bfloat16, device=device)
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device=device)
    positions = torch.randint(0, max_position_embeddings, (num_tokens,),
                              dtype=torch.int64, device=device)

    out_q = torch.zeros(num_tokens, num_heads * head_dim,
                        dtype=torch.bfloat16, device=device)
    out_kv = torch.zeros(num_tokens, num_kv_heads * head_dim,
                         dtype=torch.bfloat16, device=device)

    ops.FusedAttentionPrepare(qkv, weight, positions, out_q, out_kv,
                              num_heads, num_kv_heads, head_dim, base,
                              max_position_embeddings, rms_norm_eps,
                              partial_rotary_factor)
    torch.cuda.synchronize()

    ref_q, ref_kv = ref_fused_attention_prepare(
        qkv, weight, positions, num_heads, num_kv_heads, head_dim, base,
        rms_norm_eps, partial_rotary_factor)

    # bfloat16 kernel vs float reference: allow ~2-3 ulp of rounding.
    torch.testing.assert_close(out_q, ref_q.cuda(), atol=0.1, rtol=0.1)
    torch.testing.assert_close(out_kv, ref_kv.cuda(), atol=0.1, rtol=0.1)


def test_fused_attention_prepare():
    torch.manual_seed(0)
    # head_dim must be 128: the kernel's fixed vectorization rotates exactly
    # head_dim//2 elements in pairs of head_dim//4.
    run_case(num_tokens=4, num_heads=8, num_kv_heads=2, head_dim=128,
             base=10000.0, max_position_embeddings=32768, rms_norm_eps=1e-6,
             partial_rotary_factor=0.5)
    run_case(num_tokens=1, num_heads=32, num_kv_heads=8, head_dim=128,
             base=10000.0, max_position_embeddings=32768, rms_norm_eps=1e-6,
             partial_rotary_factor=0.5)


if __name__ == "__main__":
    test_fused_attention_prepare()
    print("test_glm_attention_prepare PASS")
