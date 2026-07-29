#!/usr/bin/env python3
"""
OP算子性能测试benchmark代码
用于对比mcoplib_ops和maca_extension_ops两个算子库的性能和精度
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
# 添加项目路径到sys.path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置随机种子保证结果可复现
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class OpsBenchmark:
    """算子性能测试基类"""
    
    def __init__(self):

        print("starting to setting device and dtype")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float16  # 默认使用半精度

        try:
            import mcoplib.lmdeploy as mcoplib_ops
            self.mcoplib_ops = mcoplib_ops
            print("✓ Successfully imported mcoplib ops")
        except ImportError as e:
            print(f"✗ Failed to import mcoplib ops: {e}")
            self.mcoplib_ops = None

        try:
            import mcoplib._C 
        except ImportError as e:
            print(f"✗ Failed to import vllm _C ops: {e}")
        try:
            from mcoplib import op as op_rms_norm
            self.op_rms_norm = op_rms_norm
            print("✓ Successfully imported mcoplib rms_norm")
        except ImportError as e:
            print(f"✗ Failed to import mcoplib rms_norm: {e}")
            self.op_rms_norm = None

    
    def measure_time(self, func: Callable, *args, warmup: int = 3, repeat: int = 10, **kwargs) -> Tuple[float, Any]:
        """测量函数执行时间"""
        # 预热
        for _ in range(warmup):
            _ = func(*args, **kwargs)
        
        # 正式测量
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        for _ in range(repeat):
            result = func(*args, **kwargs)
            
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / repeat * 1e6  # 转换为微秒
        return avg_time, result
    
    def compare_ops(self, op_name: str, op_func1: Callable, input_data: Dict[str, Any], shapes_info: str = "") -> Dict[str, Any]:
        """对比两个算子的性能和精度"""
        if self.mcoplib_ops is None:
            return {"error": "One or both ops libraries not available"}
        
        print(f"\n{'='*60}")
        print(f"Testing {op_name} - {shapes_info}")
        print(f"{'='*60}")
        
        try:
            # 测试maca_ext_ops
            time1, result1 = self.measure_time(op_func1, **input_data)
            print(f"mcoplib_time {op_name}: {time1:.2f} μs ")
            

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
        """生成测试数据"""
        if dtype is None:
            dtype = self.dtype
        
        # 使用固定随机种子生成数据
        generator = torch.Generator(device=self.device)
        generator.manual_seed(42)
        
        return torch.randn(shape, dtype=dtype, device=self.device, generator=generator)
    
    def run_benchmark(self) -> List[Dict[str, Any]]:
        """运行所有算子的benchmark测试"""
        results = []
        
        # 需要测试的算子列表
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
        """测试单个算子"""
        results = []
        
        # 定义不同算子的测试shape组合
        test_shapes = {
            "add_rms_norm": [
                (128, 512),
                (1024, 2048),
                (4096, 4096),
            ]
        }
        
        # 获取当前算子的测试shapes
        shapes = test_shapes.get(op_name, [(128, 512)])  # 默认使用小shape
        
        for shape in shapes:
            try:
                if op_name == "add_rms_norm":
                    result = self.test_add_rms_norm(shape)
                else:
                    print(f"Unknown op: {op_name}")
                    continue
                
                results.append(result)
                
            except Exception as e:
                print(f"Failed to test {op_name} with shape {shape}: {e}")
                results.append({
                    "op_name": op_name,
                    "shapes_info": str(shape),
                    "error": str(e)
                })
        
        return results
    
    def test_add_rms_norm(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """测试add_rms_norm算子"""
        # 生成测试数据
        hidden_states = self.generate_test_data(shape)
        residual = self.generate_test_data(shape)
        weight = self.generate_test_data((shape[-1],))
        epsilon = 1e-6
        
        # 测试数据
        input_data = {
            "hidden_states": hidden_states,
            "residual": residual,
            "weight": weight,
            "epsilon": epsilon
        }
    
        
        def test_mcoplib(**kwargs):
            result = torch.ops._C.fused_add_rms_norm(
                kwargs["hidden_states"], 
                kwargs["residual"], 
                kwargs["weight"], 
                kwargs["epsilon"]
            )
            return kwargs["hidden_states"]  # 返回hidden_states用于对比
        
        return self.compare_ops("add_rms_norm", test_mcoplib, input_data, str(shape))
    
    def test_silu_and_mul(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """测试silu_and_mul算子"""
        # 生成测试数据，silu_and_mul需要输入是偶数维度
        x = self.generate_test_data((shape[0], shape[1] * 2))
        
        # 测试数据
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
        """测试apply_rotary_pos_emb算子"""
        batch_size, num_heads, head_dim = shape
        
        # 生成测试数据
        query = self.generate_test_data((batch_size * num_heads, head_dim))
        key = self.generate_test_data((batch_size * num_heads, head_dim))
        cos = self.generate_test_data((batch_size * num_heads, head_dim // 2))
        sin = self.generate_test_data((batch_size * num_heads, head_dim // 2))
        
        # 测试数据
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
            query =query.contiguous().unsqueeze(0)
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
            return result  # 返回query用于对比
        
        return self.compare_ops("apply_rotary_pos_emb", test_mcoplib, input_data, str(shape))
    
    def test_topk_softmax(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        """topk_softmax"""
        batch_size, num_experts = shape
        topk = min(2, num_experts)  # topk不能超过num_experts
        
        # 生成测试数据
        router_logits = self.generate_test_data((batch_size, num_experts))
        
        # 测试数据
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
            #token_expert_indicies = torch.empty_like(topk_ids)
            self.op_rms_norm.moe_softmax_topk(
                topk_weights,
                topk_ids,  
                router_logits.float(),
                False
            )
            return topk_weights  # 返回topk_weights用于对比
        
        return self.compare_ops("topk_softmax", test_mcoplib, input_data, str(shape))
    
    def test_reshape_and_cache_new(self, shape: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> Dict[str, Any]:
        key_shape, key_cache_shape = shape
    
        
        # 生成测试数据 - 保存原始数据用于每次测试
        original_key = self.generate_test_data(key_shape)
        original_value = self.generate_test_data(key_shape)
        

        num_blocks = key_cache_shape[0]
        num_heads = key_cache_shape[1] 
        head_size = key_shape[2]  # 使用原始key的head_size
        block_size = key_cache_shape[3]
        
        value_cache_shape = (num_blocks, num_heads, head_size, block_size)
        
        original_key_cache = self.generate_test_data(key_cache_shape)
        original_value_cache = self.generate_test_data(value_cache_shape)
        kv_indices = torch.randint(0, num_blocks, (key_shape[0], 1), device=self.device)
        
        # 测试数据模板（不包含会被修改的缓存数据）
        input_data_template = {
            "kv_indices": kv_indices,
        }

        
        def test_mcoplib(**kwargs):
            # 每次测试都使用原始数据的副本
            key_copy = original_key.clone()
            value_copy = original_value.clone()
            key_cache_copy = original_key_cache.clone()
            value_cache_copy = original_value_cache.clone()
            
            # 按照fill_kv_cache函数的调用方式修正参数
            kv_indices_squeezed = kwargs["kv_indices"].squeeze(-1)
            self.mcoplib_ops.reshape_and_cache_new(
                key_copy, 
                value_copy, 
                key_cache_copy, 
                value_cache_copy, 
                kv_indices_squeezed, 
                "auto", 1.0, 1.0
            )
            return key_cache_copy  # 返回key_cache用于对比
        
        return self.compare_ops("reshape_and_cache_new", test_mcoplib, input_data_template, str(shape))
    

    def test_paged_attention_v1(self, shape: Tuple[int, ...]) -> Dict[str, Any]:
        batch_size, num_heads, head_dim = shape
#         paged_decode_attention query:torch.Size([64, 32, 128]) key_cache:torch.Size([16550, 8, 8, 16, 16]) value_cache:torch.Size([16550, 8, 16, 128])
# block_table:torch.Size([64, 16550]) block_size:16 kv_seq_len:torch.Size([64]) max_kv_seq_len:1 num_q_heads:32 num_kv_heads:8

#=================>paged_decode_attention query:torch.bfloat16 key_cache:torch.bfloat16 value_cache:torch.bfloat16 block_table:torch.int32  kv_seq_len:torch.int32 max_kv_seq_len:1 softmax_scale:0.08838834764831843

        query = self.generate_test_data((batch_size, num_heads, head_dim), torch.bfloat16)
        block_size = 16  # 使用支持的block_size
        key_cache = self.generate_test_data((16550, 8, 8, 16, 16), torch.bfloat16)  # 16550 blocks, x=8
        value_cache = self.generate_test_data((16550, 8, 16, 128), torch.bfloat16)

        #block_table = torch.randint(0, 256, (batch_size, 16550), device=self.device, dtype=torch.int)  # 32 blocks per sequence
        #kv_seq_len = torch.randint(1, 2048, (64,), device=self.device, dtype=torch.int)
        block_table = torch.randint(1, 128,(64, 16550), dtype= torch.int32, device=self.device)  # 32 blocks per sequence
        kv_seq_len = torch.randint(1, 128, (64,), dtype = torch.int32, device=self.device)
        num_kv_heads = value_cache.size(1)
        # 创建输出张量
        output = torch.empty_like(query)
        
        # 计算softmax_scale
        #softmax_scale = float(1 / math.sqrt(head_dim))
        softmax_scale  = float(0.08838834764831843)
         
        # 测试数据
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
            # 创建output的副本，避免影响后续测试的输入参数
            output_copy = kwargs["output"].clone()
            self.mcoplib_ops.paged_attention_v1(
                output_copy,      # out
                kwargs["query"],       # query
                kwargs["key_cache"],   # key_cache
                kwargs["value_cache"], # value_cache
                kwargs["num_kv_heads"], # num_kv_heads
                kwargs["softmax_scale"], # scale
                kwargs["block_table"], # block_tables
                kwargs["kv_seq_len"],  # seq_lens
                kwargs["block_size"],  # block_size
                1, # max_kv_seq_len
                None, # alibi_slopes
                "auto", # kv_cache_dtype
                1.0,     # k_scale
                1.0,     # v_scale
                torch.cuda.current_device(),     # tp_rank
                0, # blocksparse_local_blocks
                1,   # blocksparse_vert_stride
                1,    # blocksparse_block_size
                1 # blocksparse_head_sliding_step
            )
            return output_copy  # 返回output结果用于对比
        
        return self.compare_ops("paged_decode_attention", test_mcoplib, input_data, str(shape))


def main():
    """主函数"""
    
    # 创建benchmark实例
    benchmark = OpsBenchmark()
    results = benchmark.run_benchmark()
    
    # 打印汇总结果
    print(f"\n{'='*80}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    
    successful_tests = [r for r in results if 'error' not in r]
    failed_tests = [r for r in results if 'error' in r]
    
    print(f"Total tests: {len(results)}")
    print(f"Successful: {len(successful_tests)}")

if __name__ == "__main__":
    main()