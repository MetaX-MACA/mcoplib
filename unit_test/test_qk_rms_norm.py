import itertools

import pytest
import torch
import torch.nn.functional as F

from mcoplib.op import qk_rms_norm_inplace_cuda


Q_KV_HEAD_CASES = [(8, 1), (32, 4), (16, 2)]
NUM_TOKEN_CASES = [16, 1024, 2048, 4096, 8192, 10240]
QK_HEAD_DIM = 128
EPS = 1e-5


def qk_rms_norm_reference(
    token_data: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    q_head_num: int,
    kv_head_num: int,
) -> torch.Tensor:
    expected = token_data.clone()
    q_end = q_head_num * QK_HEAD_DIM
    k_end = q_end + kv_head_num * QK_HEAD_DIM

    q = expected[:, :q_end].view(-1, q_head_num, QK_HEAD_DIM)
    k = expected[:, q_end:k_end].view(-1, kv_head_num, QK_HEAD_DIM)

    # Compute the reference in FP32 and cast back to the in-place output dtype.
    q.copy_(
        F.rms_norm(q.float(), (QK_HEAD_DIM,), q_norm_weight, EPS).to(q.dtype)
    )
    k.copy_(
        F.rms_norm(k.float(), (QK_HEAD_DIM,), k_norm_weight, EPS).to(k.dtype)
    )
    return expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "q_head_num,kv_head_num,num_tokens",
    [
        (*head_case, num_tokens)
        for head_case, num_tokens in itertools.product(
            Q_KV_HEAD_CASES, NUM_TOKEN_CASES
        )
    ],
)
@torch.inference_mode()
def test_qk_rms_norm_inplace_cuda(
    q_head_num: int, kv_head_num: int, num_tokens: int
) -> None:
    torch.manual_seed(42)
    total_dim = (q_head_num + 2 * kv_head_num) * QK_HEAD_DIM
    token_data = torch.randn(
        num_tokens, total_dim, dtype=torch.bfloat16, device="cuda"
    )
    q_norm_weight = torch.randn(QK_HEAD_DIM, dtype=torch.float32, device="cuda")
    k_norm_weight = torch.randn(QK_HEAD_DIM, dtype=torch.float32, device="cuda")
    expected = qk_rms_norm_reference(
        token_data,
        q_norm_weight,
        k_norm_weight,
        q_head_num,
        kv_head_num,
    )

    input_data_ptr = token_data.data_ptr()
    result = qk_rms_norm_inplace_cuda(
        token_data,
        q_norm_weight,
        k_norm_weight,
        q_head_num,
        kv_head_num,
        QK_HEAD_DIM,
        EPS,
    )

    assert result.data_ptr() == input_data_ptr
    torch.testing.assert_close(token_data, expected, rtol=1e-2, atol=1e-2)

    v_start = (q_head_num + kv_head_num) * QK_HEAD_DIM
    assert torch.equal(token_data[:, v_start:], expected[:, v_start:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_qk_rms_norm_rejects_unsupported_shape() -> None:
    token_data = torch.empty(1, 10, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(QK_HEAD_DIM, dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match=r"token_data.size\(1\)"):
        qk_rms_norm_inplace_cuda(
            token_data, weight, weight, 8, 1, QK_HEAD_DIM, EPS
        )
