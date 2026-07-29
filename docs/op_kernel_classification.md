# mcOpLib 全量算子分类表

> 统计范围: op/ 目录下所有 .cu 文件（含 vLLM、SGLang、LMDeploy、默认算子、CV 算子）
>
> 总计: 175 个 .cu 内核文件，按功能分为 13 大类

---

## 一、算子分类总览

| 序号 | 类别 | 算子数 | 说明 |
|------|------|--------|------|
| 1 | 注意力机制 (Attention) | 14 | Paged Attention、MLA、Flash Attention、注意力状态合并等 |
| 2 | MoE 算子 (Mixture of Experts) | 26 | TopK选择、Gate融合、分组GEMM、MoE求和/归约/对齐/排列等 |
| 3 | 量化算子 (Quantization) | 23 | FP8/INT8/W8A8/W4A16/GPTQ/AWQ/GGUF 量化与反量化 |
| 4 | 量化GEMM (Quantized GEMM) | 19 | FP8/INT8/FP4/W4A8/Marlin 量化矩阵乘 |
| 5 | 归一化算子 (Normalization) | 12 | RMS Norm、LayerNorm、归一化+量化融合 |
| 6 | 位置编码 (Position Encoding) | 9 | RoPE、旋转位置编码、位置编码融合 |
| 7 | 激活函数 (Activation) | 8 | SiLU、GELU、SwiGLU、Softplus、激活+乘法融合 |
| 8 | KV Cache | 10 | KV Cache 读写、传输、MLA KV拼接、Reshape+Cache |
| 9 | 通信算子 (Communication) | 4 | AllReduce、Custom AllReduce、MSCCLPP |
| 10 | 融合算子 (Fused Ops) | 12 | 多算子融合：Norm+RoPE、QKNorm+RoPE、Silu+Mul+Quant等 |
| 11 | 推测解码 (Speculative Decoding) | 5 | 树形推测采样、Eagle、Ngram、PackBit |
| 12 | Mamba / SSM | 2 | Mamba 选择性扫描、因果卷积 |
| 13 | 其他算子 (Utility / CV / Sampling) | 31 | 采样器、词表掩码、内存管理、路由GEMM、CV视觉、KV传输、稀疏GEMM等 |

---

## 二、详细算子分类表

### 1. 注意力机制 (Attention) — 14 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| paged_attention_v1.cu | Paged Attention V1（非分块） | Q/K/V + block_tables → attn_output | Paged KV Cache，block级索引 |
| paged_attention_v2.cu | Paged Attention V2（分块） | Q/K/V + block_tables → attn_output | 分块softmax，减少显存 |
| merge_attn_states.cu (vllm) | 合并多个注意力状态 | attn_output_A/B + scores → merged | 加权合并，split-K后处理 |
| merge_attn_states.cu (sglang) | 合并多个注意力状态（SGLang版） | attn_output_A/B + scores → merged | 加权合并 |
| cascade.cu | 级联注意力 | 多层注意力输出 → 合并 | 层间注意力级联 |
| cutlass_mla_kernel.cu | MLA注意力解码内核 | Q/K/V + 吸收矩阵 → attn_output | CUTLASS/MATLASS GEMM融合 |
| cutlass_mla_entry.cu (vllm) | MLA注意力入口（vLLM版） | Q/K/V → attn_output | CUTLASS MLA dispatch |
| cutlass_mla_kernels.cu | MLA注意力内核（vLLM版） | Q/K/V → attn_output | SM100a MLA |
| fused_mla.cu | MLA融合注意力 | Q/K/V → attn_output | 吸收式MLA |
| lightning_attention_decode_kernel.cu | 闪电注意力解码 | Q/K/V → attn_output | 线性注意力解码 |
| vertical_slash_index.cu | 垂直/斜线注意力索引转换 | attention_pattern → vertical_slash_index | 稀疏注意力模式转换 |
| attention_kernels.cu (lmdeploy) | LMDeploy Paged Attention | Q/K/V + block_tables → attn_output | LMDeploy框架适配 |
| cutlass_mla_entry.cu (sglang) | MLA注意力入口（SGLang版） | Q/K/V → attn_output | CUTLASS MLA dispatch |

### 2. MoE 算子 (Mixture of Experts) — 26 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| **TopK 选择 & Gate** | | | |
| topk_softmax_kernels.cu (vllm) | TopK + Softmax 路由 | gating_logits → topk_weights, topk_indices | 稀疏专家选择 |
| topk_softmax_kernels.cu (sglang) | TopK + Softmax 路由（SGLang版） | gating_logits → topk_weights, topk_indices | 稀疏专家选择 |
| topk_sigmoid.cu | TopK + Sigmoid 路由 | gating_logits → topk_weights, topk_indices | Sigmoid门控 |
| topk_softplus_sqrt.cu | TopK + Softplus+Sqrt 路由 | gating_logits → topk_weights, topk_indices | Softplus√√激活 |
| grouped_topk_kernels.cu | 分组 TopK | gating_logits + expert_map → topk_indices | 专家分组选择 |
| moe_fused_gate.cu | MoE Gate 融合 | gating_logits → topk_weights, topk_indices + expert_permutation | Gate+排列融合 |
| moe_fused_gate_opt.cu (默认) | MoE Gate 优化版 | gating_logits → topk + 排列 + 分组偏移 | 向量化Gate+Softmax |
| moe_fused_gate_opt.cu (sglang) | MoE Gate 优化版（SGLang） | gating_logits → topk + 排列 + 分组偏移 | 向量化Gate+Softmax |
| fused_moe_gate_deepseek.cu | DeepSeek MoE Gate 融合 | gating_logits → topk + 辅助loss | DeepSeek辅助loss计算 |
| kimi_k2_moe_fused_gate.cu | Kimi K2 MoE Gate 融合 | gating_logits → topk + 专家映射 | Kimi K2专家映射 |
| **MoE 求和 & 归约** | | | |
| moe_sum_reduce.cu | MoE Sum Reduce（C500优化） | [T, K, H] bf16 → [T, H] bf16 | 128-bit向量化，warp=64 |
| moe_sum.cu | MoE 加权求和 | expert_outputs * weights → summed_output | 加权专家输出聚合 |
| **MoE 对齐 & 排列** | | | |
| moe_align_sum_kernels.cu | MoE 对齐+求和 | token_expert_ids → aligned_blocks | 专家token计数排序 |
| moe_align_kernel.cu (sglang) | MoE 对齐（SGLang版） | token_expert_ids → aligned_blocks | 专家token计数排序 |
| moe_lora_align_sum_kernels.cu | MoE LoRA 对齐+求和 | token_expert_ids + lora → aligned_blocks | LoRA专家对齐 |
| moe_permute_unpermute_op.cu | MoE 排列/逆排列 | tokens → permuted_tokens / reverse | 专家token重排 |
| prepare_moe_input.cu | MoE 输入预处理 | topk_indices + expert_map → sorted_offsets | 专家偏移计算+排列 |
| **MoE 分组 GEMM** | | | |
| grouped_gemm_cuda.cu | 分组 GEMM | 多组 A×B → 多组 C | 按专家分组矩阵乘 |
| grouped_gemm_mctlass_int8_cuda.cu | INT8 分组 GEMM | 多组 INT8 A×B → 多组 C | mctlass INT8分组GEMM |
| cutlass_moe_helper.cu | MoE GEMM 辅助 | problem_sizes → group_starts | 分组GEMM偏移计算 |
| **MoE 量化 GEMM** | | | |
| fp8_blockwise_moe_kernel.cu | FP8 分块 MoE GEMM | FP8 A×B → BF16 C | 分块量化MoE矩阵乘 |
| nvfp4_blockwise_moe.cu | FP4 分块 MoE GEMM | FP4 A×B → BF16 C | FP4 MoE矩阵乘 |
| moe_fused_w4a16_cuda.cu | W4A16 MoE 融合 GEMM | INT4权重×BF16激活 → BF16 | W4A16量化MoE |
| **MoE W8A8 量化 GEMM** | | | |
| scaled_mm_c2x.cu (moe) | W8A8 MoE GEMM (CUTLASS 2.x) | INT8 A×B → BF16 C | MoE场景W8A8 |
| scaled_mm_entry.cu (w4a8) | W4A8 MoE GEMM 入口 | INT4权重×INT8激活 → C | MoE场景W4A8 |
| w4a8_grouped_mm_c3x.cu | W4A8 MoE GEMM (SM90) | INT4权重×INT8激活 → C | CUTLASS 3.x MoE |
| w4a8_moe_data.cu | W4A8 MoE 数据预处理 | expert_sizes → offsets | W4A8分组偏移计算 |

### 3. 量化算子 (Quantization) — 23 个

| 算子文件 | 主要功能 | 量化类型 | 关键技术 |
|---------|---------|---------|---------|
| **FP8 量化** | | | |
| fp8_quantize_kernel.cu | Per-Token FP8 量化 | BF16/FP16 → FP8 E4M3 | per_token_cast_to_fp8 |
| per_token_quant_fp8.cu | Per-Token FP8 量化（SGLang版） | BF16/FP16 → FP8 E4M3 | per_token量化 |
| per_tensor_quant_fp8.cu | Per-Tensor FP8 量化 | BF16/FP16 → FP8 E4M3 | 整张量统一scale |
| common.cu | FP8 量化（vLLM版） | BF16 → FP8 | 动态/静态FP8量化 |
| per_token_group_quant.cu (vllm fp8) | Per-Token-Group FP8 量化 | BF16 → FP8 (128元素/组) | 分组量化 |
| per_token_group_quant.cu (vllm int8) | Per-Token-Group INT8 量化 | BF16 → INT8 (128元素/组) | 分组量化 |
| per_token_group_quant_8bit.cu | Per-Token-Group 8bit 量化（SGLang版） | BF16 → INT8 | 分组8bit量化 |
| per_token_group_quant_8bit_v2.cu | Per-Token-Group 8bit 量化V2 | BF16 → INT8 | 优化版分组8bit |
| cast.cu | FP8 下转换 | Float → FP8 | downcast_fp8 |
| **INT8 量化** | | | |
| int8_quant_kernels.cu (默认) | 动态 INT8 量化 | BF16 → INT8 | 动态scale+零点 |
| int8_quant_kernels.cu (sglang) | 动态 INT8 量化（SGLang版） | BF16 → INT8 | 动态scale+零点 |
| int8_quant_kernels.cu (vllm) | 动态 INT8 量化（vLLM版） | BF16 → INT8 | 动态scale+零点 |
| scale_dynamic_quant.cu | Scale 动态量化 | BF16 → INT8 | 按scale动态量化 |
| **融合量化** | | | |
| fused_silu_mul_per_group_quant.cu | SiLU×Mul + 分组量化 | BF16 → INT8 (分组) | SiLU+Mul+量化三合一 |
| fused_silu_mul_block_quant.cu | SiLU×Mul + 块量化 | BF16 → FP8/INT8 | SiLU+Mul+块级量化 |
| silu_mul_quant_fp8_nopack.cu | SiLU×Mul + FP8量化(无打包) | BF16 → FP8 | SiLU+Mul+FP8量化 |
| fused_layernorm_dynamic_per_token_quant.cu (vllm) | LayerNorm + Per-Token量化 | BF16 → FP8 | 归一化+量化融合 |
| fused_layernorm_dynamic_per_token_quant_custom.cu | LayerNorm + Per-Token量化(自定义) | BF16 → FP8 | 自定义归一化+量化 |
| fused_layernorm_dynamic_per_group_quant.cu | LayerNorm + 分组量化 | BF16 → FP8 | 归一化+分组量化 |
| rms_norm_dynamic_per_token_quant.cu | RMS Norm + Per-Token量化 | BF16 → FP8 | RMS归一化+量化 |
| **GPTQ / AWQ / GGUF 量化** | | | |
| gptq_marlin.cu (默认) | GPTQ Marlin 反量化GEMM | INT4权重×BF16 → BF16 | Marlin内核 |
| gptq_marlin.cu (sglang) | GPTQ Marlin 反量化GEMM（SGLang版） | INT4权重×BF16 → BF16 | Marlin内核 |
| gguf_kernel.cu (sglang) | GGUF 量化 | BF16 → Q8_1 | GGUF格式量化 |
| gguf_kernel.cu (vllm) | GGUF 量化（vLLM版） | BF16 → Q8_1 | GGUF格式量化 |
| quantize_kernel.cu | AWQ 反量化 | INT4 → BF16 | AWQ权重反量化 |
| activation_kernels.cu (vllm quant) | SiLU+Mul+量化 | BF16 → 量化输出 | 激活+量化融合 |

### 4. 量化 GEMM (Quantized GEMM) — 19 个

| 算子文件 | 主要功能 | 量化类型 | 关键技术 |
|---------|---------|---------|---------|
| **FP8 GEMM** | | | |
| fp8_gemm_kernel.cu | FP8 缩放矩阵乘 | FP8 A×B → BF16 C | CUTLASS FP8 GEMM (SM89/90/100/120) |
| fp8_blockwise_gemm_kernel.cu | FP8 分块缩放矩阵乘 | FP8 A×B → BF16 C | 分块量化GEMM (SM100/120) |
| bmm_fp8.cu | FP8 批量矩阵乘 | FP8 A×B → BF16 C | Batched FP8 GEMM |
| **INT8 GEMM** | | | |
| int8_gemm_kernel.cu | INT8 缩放矩阵乘 | INT8 A×B → BF16 C | CUTLASS INT8 GEMM |
| **CUTLASS W8A8 Scaled MM** | | | |
| scaled_mm_entry.cu (vllm) | W8A8 缩放矩阵乘入口 | INT8 A×B + scale → C | CUTLASS dispatch |
| scaled_mm_c2x.cu (vllm) | W8A8 缩放矩阵乘 (C2X) | INT8 A×B + scale → C | CUTLASS 2.x (SM75/80/89) |
| scaled_mm_entry.cu (sglang) | W8A8 缩放矩阵乘入口（SGLang版） | INT8 A×B + scale → C | CUTLASS dispatch |
| scaled_mm_c2x.cu (sglang) | W8A8 缩放矩阵乘 (C2X, SGLang) | INT8 A×B + scale → C | CUTLASS 2.x (SM75/80/89) |
| scaled_mm_c3x_sm90.cu | W8A8 缩放矩阵乘 (C3X SM90) | INT8 A×B + scale → C | CUTLASS 3.x SM90 |
| scaled_mm_c3x_sm100.cu | W8A8 缩放矩阵乘 (C3X SM100) | INT8 A×B + scale → C | CUTLASS 3.x SM100 |
| scaled_mm_c3x_sm120.cu | W8A8 缩放矩阵乘 (C3X SM120) | INT8 A×B + scale → C | CUTLASS 3.x SM120 |
| **FP4 GEMM** | | | |
| nvfp4_quant_kernels.cu | FP4 量化内核 | BF16 → FP4 | FP4量化 |
| nvfp4_quant_entry.cu (sglang) | FP4 量化入口 | BF16 → FP4 | FP4量化dispatch |
| nvfp4_expert_quant.cu | FP4 专家量化 | BF16 → FP4 | MoE FP4量化 |
| nvfp4_scaled_mm_entry.cu (sglang) | FP4 缩放矩阵乘入口 | FP4 A×B → BF16 C | CUTLASS FP4 GEMM |
| nvfp4_scaled_mm_kernels.cu | FP4 缩放矩阵乘内核 | FP4 A×B → BF16 C | CUTLASS FP4 GEMM SM120 |
| nvfp4_quant_entry.cu (vllm) | FP4 量化入口（vLLM版） | BF16 → FP4 | FP4量化dispatch |
| nvfp4_scaled_mm_entry.cu (vllm) | FP4 缩放矩阵乘入口（vLLM版） | FP4 A×B → BF16 C | CUTLASS FP4 GEMM |
| **GPTQ / AWQ / Marlin / QServe GEMM** | | | |
| q_gemm.cu | GPTQ 量化GEMM | INT4权重×BF16 → BF16 C | GPTQ矩阵乘 |
| gemm_kernels.cu | AWQ 量化GEMM | INT4权重×BF16 → BF16 C | AWQ矩阵乘 |
| awq_kernel.cu | AWQ GEMM（SGLang版） | INT4权重×BF16 → BF16 C | AWQ矩阵乘 |
| gptq_kernel.cu | GPTQ GEMM（SGLang版） | INT4权重×BF16 → BF16 C | GPTQ矩阵乘 |
| gptq_marlin.cu (sglang gemm) | GPTQ Marlin GEMM | INT4权重×BF16 → BF16 C | Marlin内核矩阵乘 |
| gptq_marlin_repack.cu | GPTQ Marlin 重打包 | INT4 → Marlin格式 | 权重重排 |
| awq_marlin_repack.cu | AWQ Marlin 重打包 | INT4 → Marlin格式 | 权重重排 |
| permute_cols.cu | 列排列 | INT4权重 → 重排列权重 | Marlin列重排 |
| ops.cu (marlin_moe_wna16) | Marlin MoE WNA16 GEMM | INT4权重×BF16 → BF16 C | MoE量化矩阵乘 |
| qserve_w4a8_per_chn_gemm.cu | QServe W4A8 Per-Channel GEMM | INT4权重×INT8激活 → C | QServe量化GEMM |
| qserve_w4a8_per_group_gemm.cu | QServe W4A8 Per-Group GEMM | INT4权重×INT8激活 → C | QServe分组量化GEMM |
| **稀疏 GEMM** | | | |
| sparse_scaled_mm_entry.cu | 稀疏缩放矩阵乘 | 稀疏INT8 A×B → C | 2:4结构稀疏GEMM |

### 5. 归一化算子 (Normalization) — 12 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| layernorm_kernels.cu | RMS Norm / Add RMS Norm | input + weight → normalized | RMS归一化 |
| layernorm_quant_kernels.cu | RMS Norm + 静态FP8量化 | input + weight → normalized + FP8 | 归一化+量化融合 |
| fused_add_rms_norm_kernel.cu | Add + RMS Norm 融合 | input + residual → normalized | 残差加+归一化 |
| fused_rms_norm_dq.cu | RMS Norm + 动态量化 | input + weight → normalized + quant | 归一化+动态量化 |
| rms_norm_dynamic_per_token_quant.cu | RMS Norm + Per-Token量化 | input + weight → normalized + FP8 | per-token量化 |
| fused_add_layernorm_per_token_quant_padding_output.cu | Add+LayerNorm+Per-Token量化+Padding | input + residual → normalized + quant + pad | 多功能融合 |
| fused_add_gemma_rmsnorm_per_token_quant_padding_output.cu | Add+Gemma RMSNorm+量化+Padding | input + residual → normalized + quant + pad | Gemma归一化 |
| fused_layernorm_dynamic_per_token_quant.cu (vllm) | LayerNorm+Per-Token动态量化 | input → normalized + FP8 | vLLM版归一化+量化 |
| fused_layernorm_dynamic_per_token_quant_custom.cu | LayerNorm+Per-Token量化(自定义) | input → normalized + FP8 | 自定义分组量化 |
| fused_layernorm_dynamic_per_group_quant.cu | LayerNorm+Per-Group量化 | input → normalized + FP8 | 分组量化归一化 |
| minimax_reduce_rms_kernel.cu | MiniMax Reduce RMS | input → rms_value | RMS值计算 |
| layernorm_quant_kernels.cu (vllm) | RMS Norm + 静态FP8量化 | input + weight → normalized + FP8 | vLLM版归一化+量化 |

### 6. 位置编码 (Position Encoding) — 9 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| pos_encoding_kernels.cu | 旋转位置编码 (RoPE) | Q/K + cos/sin → rotated Q/K | 标准RoPE |
| rotary_embedding.cu | 旋转位置编码（默认版） | Q/K + cos/sin → rotated Q/K | 默认算子RoPE |
| fused_rope.cu | 融合 RoPE | Q/K + positions → rotated Q/K | 融合位置编码 |
| rope.cu | RoPE (pos_ids + cos/sin缓存) | Q/K + pos_ids + cos/sin → rotated Q/K | 按位置索引编码 |
| rope_train.cu | RoPE 训练版（含反向传播） | Q/K + positions → rotated Q/K (前向+反向) | 训练模式RoPE |
| pos_enc.cu (sglang) | 位置编码 | Q/K + cos/sin → rotated Q/K | SGLang版RoPE |
| fused_rotary_emb.cu | 融合旋转位置编码 | Q/K + positions → rotated Q/K | SGLang融合RoPE |
| pos_encoding_kernels.cu (lmdeploy) | 旋转位置编码（LMDeploy版） | Q/K + cos/sin → rotated Q/K | LMDeploy适配 |
| glm_attention_prepare.cu | GLM 注意力准备 (RoPE + token旋转) | Q/K + positions → rotated Q/K | GLM模型专用 |

### 7. 激活函数 (Activation) — 8 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| activation_kernels.cu (vllm) | 激活函数+乘法融合 | input → SiLU/GELU × gate | SiLU_and_mul等 |
| activation.cu | 激活函数+乘法融合（SGLang版） | input → SiLU/GELU × gate | silu_and_mul, gelu_and_mul |
| fused_bias_gelu.cu | GELU + 偏置融合 | input + bias → GELU(input+bias) | 偏置+GELU融合 |
| fused_bias_swiglu.cu | SwiGLU + 偏置融合 | input + bias → SwiGLU(input+bias) | 偏置+SwiGLU融合 |
| fused_softplus_sqrt.cu | Softplus + Sqrt 激活 | input → softplus(sqrt(x)) | 特殊激活函数 |
| fused_bias_dropout.cu | 偏置 + Dropout 融合 | input + bias → dropout(input+bias) | 偏置+Dropout融合 |
| activation_kernels.cu (vllm quant) | SiLU+Mul+量化 | input → SiLU×gate + quant | 激活+量化融合 |
| timestep_embedding.cu | 时间步嵌入（扩散模型） | t → embedding | 正弦/余弦嵌入 |

### 8. KV Cache — 10 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| cache_kernels.cu (vllm) | KV Cache 读写 + MLA拼接 | key/value + slots → k_cache/v_cache | Paged KV Cache |
| cache_kernels_fused.cu | KV Cache + MLA RoPE 融合 | key/value + cos/sin → cache | Cache+RoPE融合写入 |
| store_kv.cu (默认) | KV Cache 存储 | key/value + slot → k_cache/v_cache | KV存储 |
| store.cu (sglang) | KV Cache 存储 | key/value + slot → k_cache/v_cache | SGLang版KV存储 |
| concat_mla.cu | MLA KV 拼接 | k_nope + k_rope → k / q_absorb | MLA Q/K拼接 |
| reshape_and_cache (cache_kernels) | Reshape + Cache 融合 | key/value → cache | 形状变换+缓存 |
| send_to_attention_node_pre_process.cu | 发送到注意力节点预处理 | hidden_states → 发送缓冲区 | 分布式KV传输 |
| recv_from_attention_node_post_process.cu | 从注意力节点接收后处理 | 接收缓冲区 → hidden_states | 分布式KV接收 |
| transfer.cu | KV Cache 跨层/全量传输 | k_cache/v_cache → 目标设备 | KV跨层/跨设备传输 |
| cache_kernels.cu (lmdeploy) | LMDeploy KV Cache | key/value + slots → cache | LMDeploy框架适配 |

### 9. 通信算子 (Communication) — 4 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| all_reduce.cu | AllReduce (Max/Sum) | 多GPU张量 → 归约结果 | 自定义AllReduce |
| custom_all_reduce.cu | 自定义 AllReduce | 多GPU张量 → 归约结果 | 注册缓冲区+IPC |
| mscclpp_allreduce.cu | MSCCLPP AllReduce | 多GPU张量 → 归约结果 | MSCCLPP通信库 |
| quick_all_reduce.cu | Quick AllReduce | 多GPU张量 → 归约结果 | 轻量AllReduce |

### 10. 融合算子 (Fused Ops) — 12 个

| 算子文件 | 主要功能 | 融合操作 | 关键技术 |
|---------|---------|---------|---------|
| dsv4_norm_rope.cu | DeepSeek V4 QKNorm+RoPE+量化 | QK归一化+RoPE+Hadarmard+量化 | DeepSeek V4专用 |
| fused_qknorm_rope_kernel.cu (vllm) | QK Norm + RoPE 融合 | QK归一化+旋转位置编码 | QK归一化+RoPE |
| fused_qknorm_rope_kernel.cu (sglang) | QK Norm + RoPE 融合 | QK归一化+旋转位置编码 | SGLang版 |
| fused_deepseekv4_qkv_rms_norm_rope.cu | DeepSeek V4 QKV RMSNorm+RoPE | QKV归一化+RoPE | DeepSeek V4 QKV融合 |
| fused_split_qkv_gemma_rmsnorm_rope.cu | Split QKV+Gemma RMSNorm+RoPE | QKV拆分+归一化+RoPE | Gemma模型融合 |
| fused_split_qkv_gemma_rmsnorm_rope_no_pack.cu | 同上(无打包版) | QKV拆分+归一化+RoPE | 无打包优化 |
| fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu | DeepSeek V4 QNorm+RoPE+KV插入 | Q归一化+RoPE+KV Cache写入 | 全缓存FP8/BF16 |
| fused_deepseek_v4_qnorm_rope_kv_insert_kernel_bf16.cu | 同上(BF16版) | Q归一化+RoPE+KV Cache写入 | BF16专用 |
| fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu | MiniMax M3 QKNorm+RoPE+KV插入 | QK归一化+RoPE+KV Cache写入 | MiniMax M3专用 |
| dsv3_fused_a_gemm.cu | DeepSeek V3 融合A矩阵GEMM | A矩阵GEMM+后处理 | DeepSeek V3专用 |
| fused_moe_gate_deepseek.cu | DeepSeek MoE Gate融合 | Gate+TopK+Aux Loss | DeepSeek MoE专用 |
| moe_swiglu_dq.cu | MoE SwiGLU+动态量化 | SiLU×Mul+动态量化 | MoE激活+量化融合 |

### 11. 推测解码 (Speculative Decoding) — 5 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| speculative_sampling.cu | 树形推测采样 | 候选token + 概率 → 采样结果 | 树形推测验证 |
| eagle_utils.cu | Eagle 推测工具 | draft_tokens + scores → 验证结果 | Eagle推测框架 |
| ngram_utils.cu | Ngram 推测工具 | tree_mask → 重建索引 | Ngram辅助推测 |
| packbit.cu | PackBit 位压缩 | 树掩码 → 压缩位图 | 稀疏掩码压缩 |
| apply_token_bitmask_inplace_cuda.cu | 词表掩码原位应用 | logits + bitmask → masked_logits | 语法约束采样 |

### 12. Mamba / SSM — 2 个

| 算子文件 | 主要功能 | 输入/输出 | 关键技术 |
|---------|---------|----------|---------|
| selective_scan_fwd.cu | Mamba 选择性扫描前向 | ssm_input → ssm_output | Mamba SSM前向传播 |
| causal_conv1d.cu | 因果1D卷积 | conv_input → conv_output | Mamba因果卷积 |

### 13. 其他算子 (Utility / CV / Sampling / Router GEMM) — 31 个

| 算子文件 | 主要功能 | 类别 |
|---------|---------|------|
| sampler.cu | 采样器 (重复惩罚+TopK) | 采样 |
| topk.cu (vllm) | 持久化TopK (DeepSeek V3稀疏注意力索引) | 采样 |
| topk.cu (sglang) | Fast TopK + TopK Transform | 采样 |
| cuda_utils_kernels.cu | CUDA 工具内核 | 工具 |
| cuda_view.cu | CUDA视图 (CPU张量→GPU视图) | 工具 |
| fused_unpack.cu | 融合解包 (TopK 8bit解包) | 工具 |
| fused_repeat_kv.cu | 重复KV (GQA) | 注意力辅助 |
| copy.cu | 无缓存拷贝到GPU (copy_to_gpu_no_ce) | 内存 |
| greenctx_stream.cu | Green Context 流管理 | 内存 |
| moe_gather.cu | MoE Gather (专家输出收集) | MoE辅助 |
| moe_scatter_dynamic_quant.cu | MoE Scatter + 动态量化 (token分发) | MoE辅助 |
| fp32_router_gemm.cu | FP32 路由器 GEMM (BF16输入→FP32输出) | 路由GEMM |
| fp32_router_gemm_entry.cu | FP32 路由器 GEMM 入口 | 路由GEMM |
| dsv3_router_gemm_bf16_out.cu | DeepSeek V3 路由器 GEMM (BF16输出) | 路由GEMM |
| dsv3_router_gemm_float_out.cu | DeepSeek V3 路由器 GEMM (FP32输出) | 路由GEMM |
| dsv3_router_gemm_entry.cu | DeepSeek V3 路由器 GEMM 入口 | 路由GEMM |
| moe_softmax_topk.cu | MoE Softmax TopK | MoE辅助 |
| moe_fused_gate_opt.cu (默认) | MoE Gate 优化版 (默认算子) | MoE辅助 |
| test_mscclpp_allreduce.cu | MSCCLPP AllReduce 测试 | 测试 |
| **CV 视觉算子 (7个)** | | |
| preprocess.cu | 图像预处理 (NV12/YUV420→RGB) | CV |
| postprocess.cu | 图像后处理 (RGB→NV12/YUV420) | CV |
| split.cu | 图像通道分割 (2/3/4通道) | CV |
| arithm.cu | 算术运算 (带/不带掩码) | CV |
| calsum.cu | 求和运算 (带/不带掩码) | CV |
| countNozero.cu | 非零元素计数 (带/不带掩码) | CV |
| meanstdev.cu | 均值/标准差计算 (带/不带掩码) | CV |

---

## 三、算子功能维度交叉统计

### 按模型系列分

| 模型系列 | 专用算子数 | 涉及算子 |
|---------|----------|---------|
| DeepSeek V3/V4 | 8 | dsv3_fused_a_gemm, dsv3_router_gemm_*, fused_deepseekv4_qkv_rms_norm_rope, fused_qknorm_rope_kernel, dsv4_norm_rope, fused_deepseek_v4_qnorm_rope_kv_insert_* |
| MiniMax M3 | 1 | fused_minimax_m3_qknorm_rope_kv_insert |
| Kimi K2 | 1 | kimi_k2_moe_fused_gate |
| Gemma | 3 | fused_split_qkv_gemma_rmsnorm_rope*, fused_add_gemma_rmsnorm_* |
| GLM | 1 | glm_attention_prepare |

### 按算子融合层级分

| 融合层级 | 算子数 | 示例 |
|---------|--------|------|
| 单算子 | ~80 | paged_attention, rms_norm, topk_softmax, all_reduce |
| 双算子融合 | ~50 | fused_add_rms_norm, silu_and_mul, rms_norm+quant, rope+cache |
| 三算子融合 | ~25 | dsv4_norm_rope (norm+rope+quant), moe_swiglu_dq (silu+mul+quant) |
| 四+算子融合 | ~10 | fused_deepseekv4_qnorm_rope_kv_insert (qnorm+rope+hadamard+kv_insert) |

### 按数据类型支持分

| 数据类型 | 涉及算子数 | 典型算子 |
|---------|----------|---------|
| BF16 | ~120 (绝大多数) | paged_attention, moe_sum_reduce, rms_norm |
| FP16 | ~30 | activation, attention, quantization |
| FP8 (E4M3) | ~25 | per_token_cast_to_fp8, fp8_gemm, scaled_mm |
| INT8 | ~15 | int8_quant, scaled_mm, grouped_gemm_mctlass_int8 |
| INT4 / W4A16 | ~8 | gptq_marlin, moe_fused_w4a16, awq_marlin |
| FP4 | ~5 | nvfp4_quant, nvfp4_scaled_mm |
| FP32 | ~5 | fp32_router_gemm, selective_scan |
