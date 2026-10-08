import pytest
import torch
import mcoplib._C

CUDA_DEVICES = ["cuda"]


def get_default_atol(tensor):
    if tensor.dtype == torch.float16:
        return 2e-3
    elif tensor.dtype == torch.bfloat16:
        return 2e-2
    return 1e-5


def get_default_rtol(tensor):
    if tensor.dtype == torch.float16:
        return 2e-3
    elif tensor.dtype == torch.bfloat16:
        return 2e-2
    return 1e-5



def situ_reference(
    input: torch.Tensor,
    beta: float,
    linear_beta: float,
):
    """
    input:
        [..., 2*d]

    output:
        [..., d]
    """

    gate, up = input.float().chunk(2, dim=-1)

    gate_out = (
        beta
        * torch.tanh(gate / beta)
        * torch.sigmoid(gate)
    )

    if linear_beta > 0:
        up = linear_beta * torch.tanh(up / linear_beta)

    out = gate_out * up

    return out.to(input.dtype)



@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "linear_beta",
    [
        -1.0,
        2.0,
    ],
)
@pytest.mark.parametrize(
    "beta",
    [
        1.0,
        1.5,
    ],
)
@torch.inference_mode()
def test_situ_and_mul(
    dtype,
    linear_beta,
    beta,
):
    """
    Test normal SiTU activation.
    """

    device = CUDA_DEVICES[0]

    tokens = 17
    hidden = 512

    input = torch.randn(
        tokens,
        2 * hidden,
        dtype=dtype,
        device=device,
    )

    output = torch.empty(
        tokens,
        hidden,
        dtype=dtype,
        device=device,
    )


    torch.ops._C.situ_and_mul(
        output,
        input,
        beta,
        linear_beta,
    )


    expected = situ_reference(
        input,
        beta,
        linear_beta,
    )


    torch.testing.assert_close(
        output,
        expected,
        atol=get_default_atol(output),
        rtol=get_default_rtol(output),
    )


    torch.library.opcheck(
        torch.ops._C.situ_and_mul,
        (
            output,
            input,
            beta,
            linear_beta,
        ),
    )



@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "linear_beta",
    [
        -1.0,
        2.0,
    ],
)
@torch.inference_mode()
def test_masked_situ_and_mul(
    dtype,
    linear_beta,
):
    """
    Test masked SiTU.

    expert_num_tokens controls valid rows.
    Invalid rows must stay zero.
    """

    device = CUDA_DEVICES[0]

    num_experts = 4
    max_num_tokens = 7
    hidden = 512

    beta = 1.5


    input = torch.randn(
        num_experts,
        max_num_tokens,
        2 * hidden,
        dtype=dtype,
        device=device,
    )


    expert_num_tokens = torch.tensor(
        [
            0,
            1,
            4,
            7,
        ],
        dtype=torch.int32,
        device=device,
    )


    output = torch.zeros(
        num_experts,
        max_num_tokens,
        hidden,
        dtype=dtype,
        device=device,
    )


    torch.ops._C.masked_situ_and_mul(
        output,
        input,
        expert_num_tokens,
        beta,
        linear_beta,
    )


    expected = situ_reference(
        input,
        beta,
        linear_beta,
    )


    for expert, num_tokens in enumerate(
        expert_num_tokens.cpu().tolist()
    ):

        if num_tokens > 0:
            torch.testing.assert_close(
                output[expert, :num_tokens],
                expected[expert, :num_tokens],
                atol=get_default_atol(output),
                rtol=get_default_rtol(output),
            )


        assert torch.count_nonzero(
            output[expert, num_tokens:]
        ) == 0


    torch.library.opcheck(
        torch.ops._C.masked_situ_and_mul,
        (
            output,
            input,
            expert_num_tokens,
            beta,
            linear_beta,
        ),
    )