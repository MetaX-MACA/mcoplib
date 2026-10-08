# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import mcoplib._C
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_update,
)
from vllm.models.kimi_k3.nvidia.kda import (
    is_fused_kda_decode_supported,
)
from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
    fused_recurrent_kda_packed_decode,
)


DEVICE = "cuda"


def assert_close(
    name: str,
    ref: torch.Tensor,
    actual: torch.Tensor,
    atol: float = 3e-2,
    rtol: float = 3e-2,
):
    max_err = (ref.float() - actual.float()).abs().max().item()
    print(f"{name}: max error = {max_err}")

    torch.testing.assert_close(
        actual,
        ref,
        atol=atol,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("num_heads", "num_seqs", "lower_bound", "fuse_output_norm"),
    [
        (12, 1, -5.0, True),
        (12, 4, None, False),
        (24, 4, None, False),
        (48, 1, -5.0, True),
        (96, 1, -5.0, True),
    ],
)
@torch.inference_mode()
def test_fused_kda_decode_correctness(
    num_heads: int,
    num_seqs: int,
    lower_bound: float | None,
    fuse_output_norm: bool,
):
    D = 128
    W = 4

    if not is_fused_kda_decode_supported(
        num_heads,
        D,
        W,
        num_spec=0,
        input_dtype=torch.bfloat16,
        conv_state_dtype=torch.bfloat16,
    ):
        pytest.skip("Fused KDA decode is not supported")

    torch.manual_seed(
        967 + num_heads + num_seqs
    )

    dim = num_heads * D
    slots = num_seqs + 2


    # -----------------------------
    # input x
    # -----------------------------
    packed_x_storage = torch.randn(
        num_seqs,
        3 * dim + 17,
        dtype=torch.bfloat16,
        device=DEVICE,
    )

    packed_x = packed_x_storage[:, :3 * dim]


    # -----------------------------
    # conv weight/state
    # -----------------------------
    weight = (
        0.1
        * torch.randn(
            3 * dim,
            W,
            dtype=torch.float32,
            device=DEVICE,
        )
    )


    conv_seed = torch.randn(
        slots,
        W - 1,
        3 * dim,
        dtype=torch.bfloat16,
        device=DEVICE,
    ).transpose(1, 2)


    # -----------------------------
    # KDA gate
    # -----------------------------
    raw_g = torch.randn(
        1,
        num_seqs,
        num_heads,
        D,
        dtype=torch.bfloat16,
        device=DEVICE,
    )


    raw_beta = torch.randn(
        1,
        num_seqs,
        num_heads,
        dtype=torch.bfloat16,
        device=DEVICE,
    )


    A_log = (
        0.5
        * torch.randn(
            num_heads,
            dtype=torch.float32,
            device=DEVICE,
        )
    )


    dt_bias = (
        0.1
        * torch.randn(
            dim,
            dtype=torch.float32,
            device=DEVICE,
        )
    )


    state_indices = torch.arange(
        num_seqs,
        0,
        -1,
        dtype=torch.int32,
        device=DEVICE,
    )


    state_seed = (
        0.01
        * torch.randn(
            slots,
            num_heads,
            D,
            D,
            dtype=torch.float32,
            device=DEVICE,
        )
    )


    # -----------------------------
    # reference
    # -----------------------------
    conv_ref = conv_seed.clone()
    state_ref = state_seed.clone()


    mixed_qkv = causal_conv1d_update(
        packed_x,
        conv_ref,
        weight,
        activation="silu",
        conv_state_indices=state_indices,
        validate_data=True,
        out=torch.empty_like(packed_x),
    )


    expected, _ = fused_recurrent_kda_packed_decode(
        mixed_qkv=mixed_qkv,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        initial_state=state_ref,
        state_indices=state_indices,
    )


    # -----------------------------
    # allocate fused kernel cache
    # -----------------------------
    conv_slot_elements = 3 * dim * (W - 1)

    state_slot_elements = (
        num_heads * D * D
    )


    conv_slot_bytes = (
        conv_slot_elements
        * torch.bfloat16.itemsize
    )


    page_bytes = (
        conv_slot_bytes
        + state_slot_elements
        * torch.float32.itemsize
    )


    cache_storage = torch.empty(
        slots * page_bytes,
        dtype=torch.uint8,
        device=DEVICE,
    )


    conv_actual = torch.as_strided(
        cache_storage.view(torch.bfloat16),
        size=(
            slots,
            3 * dim,
            W - 1,
        ),
        stride=(
            page_bytes
            // torch.bfloat16.itemsize,
            1,
            3 * dim,
        ),
    )


    state_actual = torch.as_strided(
        cache_storage.view(torch.float32),
        size=(
            slots,
            num_heads,
            D,
            D,
        ),
        stride=(
            page_bytes
            // torch.float32.itemsize,
            D * D,
            D,
            1,
        ),
        storage_offset=(
            conv_slot_bytes
            // torch.float32.itemsize
        ),
    )


    conv_actual.copy_(conv_seed)
    state_actual.copy_(state_seed)


    fused_weight = (
        weight.reshape(
            3,
            dim,
            W,
        )
        .transpose(1, 2)
        .contiguous()
    )

    out = torch.empty(
            1,
            packed_x.shape[0],
            raw_g.shape[2],
            raw_g.shape[3],
            dtype=packed_x.dtype,
            device=packed_x.device,
        )
    # -----------------------------
    # real op
    # -----------------------------
    torch.ops._C.fused_kda_decode(
        x=packed_x,
        weight=fused_weight,
        bias=None,
        conv_state=conv_actual,
        raw_g=raw_g,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        state_indices=state_indices,
        state=state_actual,
        out=out,
        lower_bound=lower_bound,
        output_gate=None,
        norm_weight=None,
        norm_eps=1e-5,
    )


    # -----------------------------
    # verify
    # -----------------------------
    print(f"========########>expected:{expected} out:{out}")
    assert_close(
        "output",
        expected,
        out,
    )


    assert_close(
        "conv_state",
        conv_ref,
        conv_actual,
        atol=0,
        rtol=0,
    )


    assert_close(
        "state",
        state_ref,
        state_actual,
    )