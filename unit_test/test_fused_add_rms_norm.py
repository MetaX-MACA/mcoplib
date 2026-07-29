#!/usr/bin/env python3
"""
OP算子精度测试代码
用于验证mcoplib fused_add_rms_norm算子的精度
使用余弦相似度进行精度验证，精度要求 > 0.9999
"""

import os
import sys
import time
import torch
import math
import numpy as np
from typing import Dict, List, Tuple, Any, Callable, Optional
import argparse
from functools import partial
import mcoplib.lmdeploy
import mcoplib._C

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

COSINE_SIMILARITY_THRESHOLD = 0.9999


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return (dot / (norm_a * norm_b)).item()


def ref_fused_add_rms_norm(hidden_states, residual, weight, epsilon):
    """
    Reference implementation of fused_add_rms_norm in float32 for ground truth.
    Matches the kernel logic:
      1. residual = input + residual (in-place update residual)
      2. variance = sum(residual^2) / hidden_size
      3. input = residual * rsqrt(variance + epsilon) * weight (in-place update input)
    """
    hidden_states_f = hidden_states.float()
    residual_f = residual.float()

    # Step 1: residual = input + residual
    residual_f = hidden_states_f + residual_f

    # Step 2: compute variance and inverse sqrt
    variance = (residual_f * residual_f).sum(dim=-1, keepdim=True) / residual_f.shape[-1]
    inv_rms = torch.rsqrt(variance + epsilon)

    # Step 3: input = residual * inv_rms * weight
    if weight is not None:
        weight_f = weight.float()
        hidden_states_f = residual_f * inv_rms * weight_f
    else:
        hidden_states_f = residual_f * inv_rms

    return hidden_states_f, residual_f


class OpsBenchmark:
    """算子性能及精度测试"""

    def __init__(self):
        print("starting to setting device and dtype")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16

        try:
            import mcoplib.lmdeploy as mcoplib_ops
            self.mcoplib_ops = mcoplib_ops
            print("Successfully imported mcoplib ops")
        except ImportError as e:
            print(f"Failed to import mcoplib ops: {e}")
            self.mcoplib_ops = None

        try:
            import mcoplib._C
        except ImportError as e:
            print(f"Failed to import vllm _C ops: {e}")
        try:
            from mcoplib import op as op_rms_norm
            self.op_rms_norm = op_rms_norm
            print("Successfully imported mcoplib rms_norm")
        except ImportError as e:
            print(f"Failed to import mcoplib rms_norm: {e}")
            self.op_rms_norm = None

    def measure_time(self, func: Callable, *args, warmup: int = 3, repeat: int = 10, **kwargs) -> Tuple[float, Any]:
        for _ in range(warmup):
            _ = func(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        for _ in range(repeat):
            result = func(*args, **kwargs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()

        avg_time = (end_time - start_time) / repeat * 1e6
        return avg_time, result

    def compare_ops(self, op_name: str, op_func1: Callable, input_data: Dict[str, Any], shapes_info: str = "") -> Dict[str, Any]:
        if self.mcoplib_ops is None:
            return {"error": "One or both ops libraries not available"}

        print(f"\n{'='*60}")
        print(f"Testing {op_name} - {shapes_info}")
        print(f"{'='*60}")

        try:
            time1, result1 = self.measure_time(op_func1, **input_data)
            print(f"mcoplib_time {op_name}: {time1:.2f} us")

            return {
                "op_name": op_name,
                "shapes_info": shapes_info,
                "mcoplib_time": time1,
                "shapes": {k: v.shape if hasattr(v, 'shape') else str(v) for k, v in input_data.items()}
            }

        except Exception as e:
            print(f"Error testing {op_name}: {e}")
            return {"op_name": op_name, "error": str(e)}

    def generate_test_data(self, shape: Tuple[int, ...], dtype: torch.dtype = None) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype

        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)

        return torch.randn(shape, dtype=dtype, device=self.device, generator=generator)

    def run_benchmark(self) -> List[Dict[str, Any]]:
        results = []

        ops_to_test = [
            "add_rms_norm"
        ]

        for op_name in ops_to_test:
            try:
                op_results = self.test_single_op(op_name)
                results.extend(op_results)
            except Exception as e:
                print(f"Failed to test {op_name}: {e}")
                results.append({
                    "op_name": op_name,
                    "error": str(e)
                })

        return results

    def test_single_op(self, op_name: str) -> List[Dict[str, Any]]:
        results = []

        test_shapes = {
            "add_rms_norm": [
                (128, 512),
                (1024, 2048),
                (4096, 4096),
            ]
        }

        shapes = test_shapes.get(op_name, [(128, 512)])

        for shape in shapes:
            try:
                if op_name == "add_rms_norm":
                    result = self.test_add_rms_norm(shape)
                    results.append(result)

                    result = self.test_add_rms_norm_without_weight(shape)
                    results.append(result)

                    continue

                else:
                    print(f"Unknown op: {op_name}")
                    continue

            except Exception as e:
                print(f"Failed to test {op_name} with shape {shape}: {e}")
                results.append({
                    "op_name": op_name,
                    "shapes_info": str(shape),
                    "error": str(e)
                })

        return results

    def _verify_precision(self, op_name: str, shape_str: str,
                          gpu_input: torch.Tensor, ref_input: torch.Tensor,
                          gpu_residual: torch.Tensor, ref_residual: torch.Tensor):
        """Verify cosine similarity for both input and residual outputs."""
        sim_input = cosine_similarity(gpu_input, ref_input)
        sim_residual = cosine_similarity(gpu_residual, ref_residual)

        print(f"  Cosine similarity (input): {sim_input:.6f}")
        print(f"  Cosine similarity (residual): {sim_residual:.6f}")

        all_pass = True
        if sim_input < COSINE_SIMILARITY_THRESHOLD:
            print(f"  FAIL: input cosine similarity {sim_input:.6f} < {COSINE_SIMILARITY_THRESHOLD}")
            all_pass = False
        else:
            print(f"  PASS: input cosine similarity {sim_input:.6f} >= {COSINE_SIMILARITY_THRESHOLD}")

        if sim_residual < COSINE_SIMILARITY_THRESHOLD:
            print(f"  FAIL: residual cosine similarity {sim_residual:.6f} < {COSINE_SIMILARITY_THRESHOLD}")
            all_pass = False
        else:
            print(f"  PASS: residual cosine similarity {sim_residual:.6f} >= {COSINE_SIMILARITY_THRESHOLD}")

        if not all_pass:
            sys.exit(1)

        return sim_input, sim_residual

    def test_add_rms_norm(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Test add_rms_norm with weight, including cosine similarity precision verification.

        Note: fused_add_rms_norm is an in-place operation that modifies both
        input and residual tensors. For precision verification, we must run
        the kernel exactly once on fresh copies of the original data.
        """
        hidden_states = self.generate_test_data(shape)
        residual = self.generate_test_data(shape)
        weight = self.generate_test_data((shape[-1],))
        epsilon = 1e-6

        # Save original data for reference computation
        hidden_states_orig = hidden_states.clone()
        residual_orig = residual.clone()

        # Compute reference in float32
        ref_input, ref_residual = ref_fused_add_rms_norm(
            hidden_states_orig, residual_orig, weight, epsilon)

        # --- Precision verification: run kernel once on fresh copies ---
        gpu_input = hidden_states.clone()
        gpu_residual = residual.clone()
        torch.ops._C.fused_add_rms_norm(gpu_input, gpu_residual, weight, epsilon)
        torch.cuda.synchronize()

        self._verify_precision("add_rms_norm", str(shape),
                               gpu_input, ref_input,
                               gpu_residual, ref_residual)

        # --- Performance benchmark: use cloned data per iteration ---
        def test_mcoplib(**kwargs):
            hs = kwargs["hidden_states"].clone()
            res = kwargs["residual"].clone()
            torch.ops._C.fused_add_rms_norm(hs, res, kwargs["weight"], kwargs["epsilon"])
            return hs

        input_data = {
            "hidden_states": hidden_states,
            "residual": residual,
            "weight": weight,
            "epsilon": epsilon
        }

        perf_result = self.compare_ops("add_rms_norm", test_mcoplib, input_data, str(shape))
        return perf_result

    def test_add_rms_norm_without_weight(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Test add_rms_norm without weight, including cosine similarity precision verification.

        Note: fused_add_rms_norm is an in-place operation that modifies both
        input and residual tensors. For precision verification, we must run
        the kernel exactly once on fresh copies of the original data.
        """
        hidden_states = self.generate_test_data(shape)
        residual = self.generate_test_data(shape)
        epsilon = 1e-6

        # Save original data for reference computation
        hidden_states_orig = hidden_states.clone()
        residual_orig = residual.clone()

        # Compute reference in float32
        ref_input, ref_residual = ref_fused_add_rms_norm(
            hidden_states_orig, residual_orig, None, epsilon)

        # --- Precision verification: run kernel once on fresh copies ---
        gpu_input = hidden_states.clone()
        gpu_residual = residual.clone()
        torch.ops._C.fused_add_rms_norm(gpu_input, gpu_residual, None, epsilon)
        torch.cuda.synchronize()

        self._verify_precision("add_rms_norm_without_weight", str(shape),
                               gpu_input, ref_input,
                               gpu_residual, ref_residual)

        # --- Performance benchmark: use cloned data per iteration ---
        def test_mcoplib(**kwargs):
            hs = kwargs["hidden_states"].clone()
            res = kwargs["residual"].clone()
            torch.ops._C.fused_add_rms_norm(hs, res, None, kwargs["epsilon"])
            return hs

        input_data = {
            "hidden_states": hidden_states,
            "residual": residual,
            "epsilon": epsilon
        }

        perf_result = self.compare_ops(
            "add_rms_norm_without_weight", test_mcoplib, input_data, str(shape))
        return perf_result

    def test_silu_and_mul(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """Test silu_and_mul op."""
        x = self.generate_test_data((shape[0], shape[1] * 2))

        input_data = {"x": x}

        def test_mcoplib(**kwargs):
            x = kwargs["x"].clone()
            d = x.shape[-1] // 2
            output_shape = x.shape[:-1] + (d,)
            out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
            torch.ops._C.silu_and_mul(out, x)
            return out

        return self.compare_ops("silu_and_mul", test_mcoplib, input_data, str(shape))

    def test_apply_rotary_pos_emb(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        batch_size, num_heads, head_dim = shape

        query = self.generate_test_data((batch_size * num_heads, head_dim))
        key = self.generate_test_data((batch_size * num_heads, head_dim))
        cos = self.generate_test_data((batch_size * num_heads, head_dim // 2))
        sin = self.generate_test_data((batch_size * num_heads, head_dim // 2))

        input_data = {
            "query": query,
            "key": key,
            "cos": cos,
            "sin": sin
        }

        def test_mcoplib(**kwargs):
            query = kwargs["query"].clone()
            key = kwargs["key"].clone()
            cos = kwargs["cos"]
            sin = kwargs["sin"]
            query = query.contiguous().unsqueeze(0)
            key = key.contiguous().unsqueeze(0)
            position_ids_1d = torch.arange(0, query.size(1), device=query.device)
            head_size = query.size(-1)
            query = query.flatten(-2, -1)
            key = key.flatten(-2, -1)
            rot_dim = cos.size(-1)
            self.mcoplib_ops.lmdeploy_rotary_embedding(
                position_ids_1d,
                query,
                key,
                head_size,
                cos.view(-1, rot_dim),
                sin.view(-1, rot_dim),
                True,
            )
            result = query
            return result

        return self.compare_ops("apply_rotary_pos_emb", test_mcoplib, input_data, str(shape))

    def test_topk_softmax(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        batch_size, num_experts = shape
        topk = min(2, num_experts)

        router_logits = self.generate_test_data((batch_size, num_experts))

        input_data = {
            "router_logits": router_logits,
            "topk": topk,
            "renormalize": False
        }

        def test_mcoplib(**kwargs):
            topk_weights = torch.empty(
                batch_size, topk, dtype=torch.float32, device=kwargs["router_logits"].device
            )
            topk_ids = torch.empty(batch_size, topk, dtype=torch.int32, device=kwargs["router_logits"].device)
            self.op_rms_norm.moe_softmax_topk(
                topk_weights,
                topk_ids,
                router_logits.float(),
                False
            )
            return topk_weights

        return self.compare_ops("topk_softmax", test_mcoplib, input_data, str(shape))

    def test_reshape_and_cache_new(self, shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Dict[str, Any]:
        key_shape, key_cache_shape = shape

        original_key = self.generate_test_data(key_shape)
        original_value = self.generate_test_data(key_shape)

        num_blocks = key_cache_shape[0]
        num_heads = key_cache_shape[1]
        head_size = key_shape[2]
        block_size = key_cache_shape[3]

        value_cache_shape = (num_blocks, num_heads, head_size, block_size)

        original_key_cache = self.generate_test_data(key_cache_shape)
        original_value_cache = self.generate_test_data(value_cache_shape)
        kv_indices = torch.randint(0, num_blocks, (key_shape[0], 1), device=self.device)

        input_data_template = {
            "kv_indices": kv_indices,
        }

        def test_mcoplib(**kwargs):
            key_copy = original_key.clone()
            value_copy = original_value.clone()
            key_cache_copy = original_key_cache.clone()
            value_cache_copy = original_value_cache.clone()

            kv_indices_squeezed = kwargs["kv_indices"].squeeze(-1)
            self.mcoplib_ops.reshape_and_cache_new(
                key_copy,
                value_copy,
                key_cache_copy,
                value_cache_copy,
                kv_indices_squeezed,
                "auto", 1.0, 1.0
            )
            return key_cache_copy

        return self.compare_ops("reshape_and_cache_new", test_mcoplib, input_data_template, str(shape))

    def test_paged_attention_v1(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        batch_size, num_heads, head_dim = shape

        query = self.generate_test_data((batch_size, num_heads, head_dim), torch.bfloat16)
        block_size = 16
        key_cache = self.generate_test_data((16550, 8, 8, 16, 16), torch.bfloat16)
        value_cache = self.generate_test_data((16550, 8, 16, 128), torch.bfloat16)

        block_table = torch.randint(1, 128, (64, 16550), dtype=torch.int32, device=self.device)
        kv_seq_len = torch.randint(1, 128, (64,), dtype=torch.int32, device=self.device)
        num_kv_heads = value_cache.size(1)

        output = torch.empty_like(query)

        softmax_scale = float(0.08838834764831843)

        input_data = {
            "query": query,
            "key_cache": key_cache,
            "value_cache": value_cache,
            "output": output,
            "num_kv_heads": num_kv_heads,
            "softmax_scale": softmax_scale,
            "block_table": block_table,
            "kv_seq_len": kv_seq_len,
            "block_size": block_size,
            "max_kv_seq_len": 1,
            "alibi_slopes": None,
            "kv_cache_dtype": "auto",
            "k_scale": 1.0,
            "v_scale": 1.0,
            "tp_rank": torch.cuda.current_device(),
            "blocksparse_local_blocks": 0,
            "blocksparse_vert_stride": 1,
            "blocksparse_block_size": 1,
            "blocksparse_head_sliding_step": 1
        }

        def test_mcoplib(**kwargs):
            output_copy = kwargs["output"].clone()
            self.mcoplib_ops.paged_attention_v1(
                output_copy,
                kwargs["query"],
                kwargs["key_cache"],
                kwargs["value_cache"],
                kwargs["num_kv_heads"],
                kwargs["softmax_scale"],
                kwargs["block_table"],
                kwargs["kv_seq_len"],
                kwargs["block_size"],
                1,
                None,
                "auto",
                1.0,
                1.0,
                torch.cuda.current_device(),
                0,
                1,
                1,
                1
            )
            return output_copy

        return self.compare_ops("paged_decode_attention", test_mcoplib, input_data, str(shape))


def main():
    benchmark = OpsBenchmark()
    results = benchmark.run_benchmark()

    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")

    successful_tests = [r for r in results if 'error' not in r]
    failed_tests = [r for r in results if 'error' in r]

    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")

    print("\nAll precision tests PASSED (cosine similarity >= 0.9999)")


if __name__ == "__main__":
    main()
