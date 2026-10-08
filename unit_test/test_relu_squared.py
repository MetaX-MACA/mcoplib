# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F

import mcoplib._C


def relu_squared_reference(x):
    """
    Reference implementation of relu_squared:

        output = relu(input)^2
    """
    return torch.relu(x) ** 2


def run_test():
    # =========================================================
    # 1. Test configuration
    # =========================================================
    NUM_TOKENS = 4096
    HIDDEN_SIZE = 4096

    DTYPES = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    print("=== 开始测试 relu_squared ===")

    for dtype in DTYPES:
        print(
            f"\n配置: "
            f"Tokens={NUM_TOKENS}, "
            f"HiddenSize={HIDDEN_SIZE}, "
            f"Dtype={dtype}"
        )

        # =====================================================
        # 2. Input
        # =====================================================
        torch.manual_seed(42)

        x = torch.randn(
            (NUM_TOKENS, HIDDEN_SIZE),
            dtype=dtype,
            device="cuda",
        )

        # =====================================================
        # 3. Output buffer
        # =====================================================
        out = torch.empty_like(x)

        # =====================================================
        # 4. Reference
        # =====================================================
        ref_out = relu_squared_reference(x)

        # =====================================================
        # 5. CUDA operator
        # =====================================================
        torch.ops._C.relu_squared(
            out,
            x,
        )

        torch.cuda.synchronize()

        # =====================================================
        # 6. Accuracy
        # =====================================================

        # 最大绝对误差
        max_diff = torch.max(
            torch.abs(
                ref_out.float()
                - out.float()
            )
        ).item()

        # 平均绝对误差
        mean_diff = torch.mean(
            torch.abs(
                ref_out.float()
                - out.float()
            )
        ).item()

        # cosine similarity
        cos_sim = F.cosine_similarity(
            ref_out.float().flatten(),
            out.float().flatten(),
            dim=0,
        ).item()

        print(
            f"📊 Cosine Similarity: "
            f"{cos_sim:.8f}"
        )

        print(
            f"📊 Max Absolute Diff: "
            f"{max_diff:.8e}"
        )

        print(
            f"📊 Mean Absolute Diff: "
            f"{mean_diff:.8e}"
        )

        # =====================================================
        # 7. Different tolerances by dtype
        # =====================================================
        if dtype == torch.float16:
            atol = 1e-3
            rtol = 1e-3
        elif dtype == torch.bfloat16:
            atol = 2e-2
            rtol = 2e-2
        else:
            atol = 1e-6
            rtol = 1e-6

        torch.testing.assert_close(
            out,
            ref_out,
            atol=atol,
            rtol=rtol,
        )

        print(
            f"✅ {dtype} 精度测试通过 "
            f"(atol={atol}, rtol={rtol})"
        )

    print("\n=== relu_squared 精度测试全部通过 ===")


if __name__ == "__main__":
    run_test()