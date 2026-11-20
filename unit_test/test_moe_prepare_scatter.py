
import pytest
import torch
import mcoplib._moe_C


def _has_moe_prepare_scatter():
    return hasattr(torch.ops._moe_C, "moe_prepare_scatter")


def _run_moe_prepare_scatter(topk_ids, n_expert, n_local_expert, expert_map=None, token_expert_indices=None, workspace_extra=0, workspace_dtype=torch.int8):
    device = topk_ids.device
    n_token, topk = topk_ids.shape

    if token_expert_indices is None:
        token_expert_indices = torch.zeros((n_token, topk), dtype=torch.int32, device=device)

    expert_first_token_offset = torch.empty((n_local_expert + 1,), dtype=torch.int64, device=device)
    scatter_idx = torch.empty((n_token, topk), dtype=torch.int32, device=device)
    sort_workspace = torch.empty((n_token * topk * 4 * 4 + workspace_extra,), dtype=workspace_dtype, device=device)
    sorted_experts = torch.empty((n_token, topk), dtype=torch.int32, device=device)
    sorted_rows = torch.empty((n_token, topk), dtype=torch.int32, device=device)
    topk_ids_for_sort = topk_ids.clone()

    torch.ops._moe_C.moe_prepare_scatter(topk_ids, token_expert_indices, expert_map, n_expert, n_local_expert, expert_first_token_offset, scatter_idx, sort_workspace, sorted_experts, sorted_rows, topk_ids_for_sort)
    torch.cuda.synchronize()

    return {
        "offsets": expert_first_token_offset,
        "scatter_idx": scatter_idx,
        "sorted_experts": sorted_experts,
        "sorted_rows": sorted_rows,
        "topk_ids_for_sort": topk_ids_for_sort,
        "token_expert_indices": token_expert_indices,
        "workspace": sort_workspace,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _has_moe_prepare_scatter(), reason="moe_prepare_scatter is unavailable")
@pytest.mark.parametrize(
    "topk_ids,n_expert,n_local_expert,expert_map,expected_offsets",
    [
        # 官方场景：无效路由和部分专家映射
        (
            [[0], [-1], [1], [4], [2]],
            4, 2, [0, 1, -1, -1], [0, 1, 2],
        ),
        # 所有 token 都路由到专家 0
        (
            [[0], [0], [0], [0]],
            4, 4, None, [0, 4, 4, 4, 4],
        ),
        # 每个专家恰好接收一个 token
        (
            [[0], [1], [2], [3]],
            4, 4, None, [0, 1, 2, 3, 4],
        ),
        # 部分专家没有 token
        (
            [[1], [1], [3]],
            4, 4, None, [0, 0, 2, 2, 3],
        ),
        # Top-K=2，专家分布不均匀
        (
            [[0, 1], [1, 2], [0, 2]],
            4, 4, None, [0, 2, 4, 6, 6],
        ),
        # Top-K=4，所有专家均出现
        (
            [[0, 1, 2, 3], [3, 2, 1, 0]],
            4, 4, None, [0, 2, 4, 6, 8],
        ),
        # expert_map 仅保留两个本地专家
        (
            [[0], [1], [2], [3], [0], [1]],
            4, 2, [0, 1, -1, -1], [0, 2, 4],
        ),
        # expert_map 将全局专家重新映射到本地专家
        (
            [[2], [0], [2], [0]],
            3, 2, [1, -1, 0], [0, 2, 4],
        ),
    ],
)
@torch.inference_mode()
def test_moe_prepare_scatter_offsets(topk_ids, n_expert, n_local_expert, expert_map, expected_offsets):
    device = "cuda"
    topk_ids = torch.tensor(topk_ids, dtype=torch.int32, device=device)
    expert_map_tensor = None if expert_map is None else torch.tensor(expert_map, dtype=torch.int32, device=device)

    result = _run_moe_prepare_scatter(topk_ids, n_expert, n_local_expert, expert_map_tensor)

    expected = torch.tensor(expected_offsets, dtype=torch.int64, device=device)
    torch.testing.assert_close(result["offsets"], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _has_moe_prepare_scatter(), reason="moe_prepare_scatter is unavailable")
@pytest.mark.parametrize(
    "topk_ids,n_expert,n_local_expert,expert_map",
    [
        ([[0]], 1, 1, None),
        ([[0], [0]], 1, 1, None),
        ([[0, 0], [0, 0]], 1, 1, None),
        ([[0, 1], [1, 0], [0, 1]], 2, 2, None),
        ([[0, 1, 2], [2, 1, 0]], 3, 3, None),
        ([[0], [-1], [1], [4], [2]], 4, 2, [0, 1, -1, -1]),
    ],
)
@torch.inference_mode()
def test_moe_prepare_scatter_output_invariants(topk_ids, n_expert, n_local_expert, expert_map):
    device = "cuda"
    topk_ids = torch.tensor(topk_ids, dtype=torch.int32, device=device)
    expert_map_tensor = None if expert_map is None else torch.tensor(expert_map, dtype=torch.int32, device=device)

    result = _run_moe_prepare_scatter(topk_ids, n_expert, n_local_expert, expert_map_tensor)

    offsets = result["offsets"]
    scatter_idx = result["scatter_idx"]
    sorted_experts = result["sorted_experts"]
    sorted_rows = result["sorted_rows"]

    # offsets 长度、起点和单调性
    assert offsets.numel() == n_local_expert + 1
    assert offsets[0].item() == 0
    assert torch.all(offsets[1:] >= offsets[:-1]).item()

    # 输出形状、dtype、设备
    assert scatter_idx.shape == topk_ids.shape
    assert scatter_idx.dtype == torch.int32
    assert scatter_idx.device == topk_ids.device
    assert sorted_experts.shape == topk_ids.shape
    assert sorted_experts.dtype == torch.int32
    assert sorted_rows.shape == topk_ids.shape
    assert sorted_rows.dtype == torch.int32

    # 对于当前算子，offsets 最后一项不能超过总扩展路由数
    assert 0 <= offsets[-1].item() <= topk_ids.numel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _has_moe_prepare_scatter(), reason="moe_prepare_scatter is unavailable")
@torch.inference_mode()
def test_moe_prepare_scatter_workspace_extra_capacity():
    device = "cuda"
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.int32, device=device)

    reference = _run_moe_prepare_scatter(topk_ids.clone(), 2, 2, workspace_extra=0)
    result = _run_moe_prepare_scatter(topk_ids.clone(), 2, 2, workspace_extra=128)

    torch.testing.assert_close(result["offsets"], reference["offsets"])
    assert result["workspace"].numel() == reference["workspace"].numel() + 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _has_moe_prepare_scatter(), reason="moe_prepare_scatter is unavailable")
@torch.inference_mode()
def test_moe_prepare_scatter_repeated_calls_with_changed_inputs():
    device = "cuda"
    first_ids = torch.tensor([[0], [0], [1], [1]], dtype=torch.int32, device=device)
    second_ids = torch.tensor([[1], [1], [1], [0]], dtype=torch.int32, device=device)

    first = _run_moe_prepare_scatter(first_ids, 2, 2)
    second = _run_moe_prepare_scatter(second_ids, 2, 2)

    torch.testing.assert_close(first["offsets"], torch.tensor([0, 2, 4], dtype=torch.int64, device=device))
    torch.testing.assert_close(second["offsets"], torch.tensor([0, 1, 4], dtype=torch.int64, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _has_moe_prepare_scatter(), reason="moe_prepare_scatter is unavailable")
@pytest.mark.parametrize("workspace_dtype", [torch.uint8, torch.float32])
@torch.inference_mode()
def test_moe_prepare_scatter_rejects_invalid_workspace_dtype(workspace_dtype):
    device = "cuda"
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32, device=device)

    with pytest.raises(RuntimeError, match="sort_workspace has incorrect dtype"):
        _run_moe_prepare_scatter(topk_ids, 2, 2, workspace_dtype=workspace_dtype)
