import torch

import mcoplib._C


def ngram_compute_n_gram_ids_reference(ne_n, ne_k, ne_weights, ne_mods, exclusive_ne_embedder_size_sums, exclusive_req_len_sums, ne_token_table, row_indices, column_starts):
    """
    Python reference implementation of ComputeNGramIdsKernel.
    完全对齐 CUDA kernel 的索引和计算逻辑。
    """

    batch_size = exclusive_req_len_sums.numel() - 1
    max_context_len = ne_token_table.shape[1]
    num_configs = (ne_n - 1) * ne_k
    token_num = int(exclusive_req_len_sums[-1].item())

    # CUDA 中通过 raw pointer 按一维访问
    ne_weights_flat = ne_weights.reshape(-1)
    ne_mods_flat = ne_mods.reshape(-1)

    n_gram_ids = torch.empty((token_num, num_configs), dtype=torch.int32, device=ne_token_table.device)

    for req_id in range(batch_size):
        req_begin = int(exclusive_req_len_sums[req_id].item())
        req_end = int(exclusive_req_len_sums[req_id + 1].item())

        row_idx = int(row_indices[req_id].item())
        column_start = int(column_starts[req_id].item())

        req_token_table_begin = row_idx * max_context_len

        for i in range(req_begin, req_end):
            current_token_offset = i - req_begin

            current_token_table_index = req_token_table_begin + column_start + current_token_offset

            for n in range(ne_n - 1):
                for k in range(ne_k):
                    ne_weight_base_idx = n * ne_k * ne_n + k * ne_n
                    ne_mod = int(ne_mods_flat[n * ne_k + k].item())

                    n_gram_id = 0

                    for j in range(n + 2):
                        token_index = current_token_table_index - j

                        if token_index < req_token_table_begin:
                            break

                        token_row = token_index // max_context_len
                        token_col = token_index % max_context_len

                        token_value = int(ne_token_table[token_row, token_col].item())

                        if token_value < 0:
                            break

                        weight = int(ne_weights_flat[ne_weight_base_idx + j].item())
                        term = token_value * weight
                        n_gram_id += term % ne_mod

                    n_gram_id %= ne_mod
                    n_gram_id += int(exclusive_ne_embedder_size_sums[n * ne_k + k].item())

                    output_col = n * ne_k + k
                    n_gram_ids[i, output_col] = n_gram_id

    return n_gram_ids


def run_test():
    NE_N = 4
    NE_K = 2
    BATCH_SIZE = 3
    MAX_CONTEXT_LEN = 16

    print("=== 开始测试 ngram_compute_n_gram_ids ===")
    print(f"配置: NE_N={NE_N}, NE_K={NE_K}, Batch={BATCH_SIZE}, MaxContextLen={MAX_CONTEXT_LEN}")

    torch.manual_seed(42)
    device = "cuda"

    # ---------------------------------------------------------
    # 1. Request lengths
    # req0 = 5
    # req1 = 4
    # req2 = 6
    # exclusive = [0, 5, 9, 15]
    # ---------------------------------------------------------
    req_lens = [5, 4, 6]

    exclusive_req_len_sums = torch.tensor([0, 5, 9, 15], dtype=torch.int32, device=device)

    # row_indices 必须是 int64
    row_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)

    # 非零 column offset，覆盖 column_starts 逻辑
    column_starts = torch.tensor([0, 2, 1], dtype=torch.int32, device=device)

    # ---------------------------------------------------------
    # 2. Token table
    # [max_running_reqs, max_context_len]
    # ---------------------------------------------------------
    ne_token_table = torch.tensor([
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36],
        [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
    ], dtype=torch.int32, device=device)

    # ignored token / EOS boundary
    ne_token_table[1, 4] = -1

    # ---------------------------------------------------------
    # 3. N-gram weights
    # shape = [NE_N - 1, NE_K, NE_N] = [3, 2, 4]
    # ---------------------------------------------------------
    ne_weights = torch.tensor([
        [[1, 2, 3, 4], [2, 3, 4, 5]],
        [[3, 1, 2, 4], [4, 2, 1, 3]],
        [[2, 5, 1, 3], [3, 4, 2, 1]],
    ], dtype=torch.int32, device=device)

    # ---------------------------------------------------------
    # 4. Mod
    # shape = [NE_N - 1, NE_K] = [3, 2]
    # ---------------------------------------------------------
    ne_mods = torch.tensor([
        [97, 101],
        [103, 107],
        [109, 113],
    ], dtype=torch.int32, device=device)

    # ---------------------------------------------------------
    # 5. Embedder offsets
    # [(NE_N - 1) * NE_K + 1] = [7]
    # ---------------------------------------------------------
    exclusive_ne_embedder_size_sums = torch.tensor([0, 100, 200, 300, 400, 500, 600], dtype=torch.int32, device=device)

    # ---------------------------------------------------------
    # 6. Output
    # ---------------------------------------------------------
    token_num = sum(req_lens)
    num_configs = (NE_N - 1) * NE_K

    cuda_n_gram_ids = torch.empty((token_num, num_configs), dtype=torch.int32, device=device)

    # ---------------------------------------------------------
    # 7. Reference
    # ---------------------------------------------------------
    print("\n[1/2] 计算 Reference...")

    ref_n_gram_ids = ngram_compute_n_gram_ids_reference(
        NE_N,
        NE_K,
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        exclusive_req_len_sums,
        ne_token_table,
        row_indices,
        column_starts,
    )

    # ---------------------------------------------------------
    # 8. CUDA operator
    # ---------------------------------------------------------
    print("调用 CUDA ngram_compute_n_gram_ids...")

    torch.ops._C.ngram_compute_n_gram_ids(
        NE_N,
        NE_K,
        ne_weights,
        ne_mods,
        exclusive_ne_embedder_size_sums,
        exclusive_req_len_sums,
        ne_token_table,
        row_indices,
        column_starts,
        cuda_n_gram_ids,
    )

    torch.cuda.synchronize()

    # ---------------------------------------------------------
    # 9. Accuracy check
    # ---------------------------------------------------------
    print("\n[1/2] 对比 CUDA 和 Reference...")

    torch.testing.assert_close(cuda_n_gram_ids, ref_n_gram_ids, atol=0, rtol=0)

    print("✅ n_gram_ids 与 Reference 完全一致。")

    print("\nReference 前 10 行:")
    print(ref_n_gram_ids[:10].cpu())

    print("\nCUDA 前 10 行:")
    print(cuda_n_gram_ids[:10].cpu())

    # ---------------------------------------------------------
    # 10. Performance
    # ---------------------------------------------------------
    print("\n[2/2] 开始性能测试...")

    WARMUP = 10
    RUNS = 100

    for _ in range(WARMUP):
        torch.ops._C.ngram_compute_n_gram_ids(
            NE_N,
            NE_K,
            ne_weights,
            ne_mods,
            exclusive_ne_embedder_size_sums,
            exclusive_req_len_sums,
            ne_token_table,
            row_indices,
            column_starts,
            cuda_n_gram_ids,
        )

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    for _ in range(RUNS):
        torch.ops._C.ngram_compute_n_gram_ids(
            NE_N,
            NE_K,
            ne_weights,
            ne_mods,
            exclusive_ne_embedder_size_sums,
            exclusive_req_len_sums,
            ne_token_table,
            row_indices,
            column_starts,
            cuda_n_gram_ids,
        )

    end_event.record()
    torch.cuda.synchronize()

    cuda_avg_time = start_event.elapsed_time(end_event) / RUNS

    print(f"⏱️ CUDA 平均耗时: {cuda_avg_time:.6f} ms")
    print("\n=== ngram_compute_n_gram_ids 测试完成 ===")


if __name__ == "__main__":
    run_test()