# # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for persistent topk."""

import time
import torch
import numpy as np
import mcoplib
import mcoplib._C
import pytest

def top_k_per_row_decode_numpy(logits, seq_lens, topk_tokens):
    if seq_lens.ndim > 1:
        seq_lens = seq_lens.ravel()
    num_rows = logits.shape[0]
    out = np.zeros((num_rows, topk_tokens), dtype=np.int64)
    for i in range(num_rows):
        length = int(seq_lens[i])
        if length <= 0:
            continue
        k = min(topk_tokens, length)
        row = logits[i, :length].astype(np.float32)
        idx = np.argpartition(-row, k - 1)[:k]
        vals = row[idx]
        sort_order = np.argsort(-vals)
        idx = idx[sort_order]
        out[i, :k] = idx.astype(np.int64)
    return out

def run_tests():
    bs_list = [1, 2, 16, 128, 256]
    spec_list = [0, 1, 3]
    maxlen_list = [16387, 65536]
    topk_list = [512, 1024]

    np.random.seed(42)
    for bs in bs_list:
        for spec in spec_list:
            for maxlen in maxlen_list:
                for topk in topk_list:
                    next_n = 1 + spec
                    num_rows = bs * next_n
                    logits_mod = np.random.randn(num_rows, maxlen).astype(np.float32)
                    seq_lens_mod = np.random.randint(maxlen//2, maxlen + 1, size=(bs, next_n)).astype(np.int32)
                    out_mod = top_k_per_row_decode_numpy(logits_mod, seq_lens_mod, topk)

                    logits = torch.tensor(logits_mod, device="cuda")
                    seq_lens = torch.tensor(seq_lens_mod, device="cuda")
                    RADIX_TOPK_WORKSPACE_SIZE = num_rows * 772 * 4
                    workspace = torch.zeros(RADIX_TOPK_WORKSPACE_SIZE, device='cuda').to(torch.uint8)
                    output = torch.zeros(bs * next_n, topk, device='cuda').to(torch.int32)
                    torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

                    torch_sorted = np.sort(output.cpu().numpy(), axis=-1)
                    np_sorted = np.sort(out_mod, axis=-1)
                    assert (np.array_equal(torch_sorted,np_sorted)), f"test persistent_topk failed.{bs=}, {spec=}, {maxlen=}, {topk=}"

def standard_generate(t=512):
    bs = 3
    spec = 0
    maxlen = 16387
    topk = t
    next_n = 1 + spec
    num_rows = bs * next_n

    logits_mod = np.random.randn(num_rows, maxlen).astype(np.float32)
    seq_lens_mod = np.random.randint(maxlen//2, maxlen + 1, size=(bs, next_n)).astype(np.int32)
    logits = torch.tensor(logits_mod, device="cuda")
    seq_lens = torch.tensor(seq_lens_mod, device="cuda")
    RADIX_TOPK_WORKSPACE_SIZE = num_rows * 772 * 4
    workspace = torch.zeros(RADIX_TOPK_WORKSPACE_SIZE, device='cuda').to(torch.uint8)
    output = torch.zeros(bs * next_n, topk, device='cuda').to(torch.int32)
    return bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output

def run_error_tests():
    with pytest.raises(RuntimeError, match="logits must be 2D"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        logits_mod = np.random.randn(num_rows, num_rows, maxlen).astype(np.float32)
        logits = torch.tensor(logits_mod, device="cuda")
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="Only float32 supported"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        logits = logits.to(torch.int32)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="lengths must be int32"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        seq_lens = seq_lens.to(torch.int64)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="output must be int32"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        output = output.to(torch.float32)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="lengths must be 1D or 2D"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        seq_lens_mod = np.random.randn(num_rows, num_rows, maxlen).astype(np.int32)
        seq_lens = torch.tensor(seq_lens_mod, device="cuda")
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="output must be 2D"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        output = torch.zeros(bs * num_rows, device='cuda').to(torch.int32)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="logits strides\\[1\\] must be 1"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        logits = logits.transpose(0, 1)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="output size mismatch"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        output = torch.zeros(topk, topk, device='cuda').to(torch.int32)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="persistent_topk supports k=512, k=1024, or k=2048, got k=128"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate(128)
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

    with pytest.raises(RuntimeError, match="lengths size mismatch"):
        bs,num_rows,maxlen,topk,logits,seq_lens,workspace,output = standard_generate()
        seq_lens_mod = np.random.randn(num_rows, num_rows).astype(np.int32)
        seq_lens = torch.tensor(seq_lens_mod, device="cuda")
        torch.ops._C.persistent_topk(logits, seq_lens, output, workspace, topk, maxlen)

if __name__ == "__main__":
    run_tests()
    run_error_tests()
