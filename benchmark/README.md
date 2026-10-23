# mxbench 性能测试工具

mxbench 是用于算子性能基准测试（Benchmark）的工具，旨在评估和记录算子的运行效率。

## 性能指标 (Metrics)
mxbench 输出的性能报告包含详细的精度验证与性能统计数据，各列含义说明如下：

### 1. 精度验证 (Accuracy Verification)
* **Acc_Pass**: 精度验证结果（Yes/No）。指示算子输出是否通过了与参考实现的对比测试。
* **Cos_Dist**: 余弦距离（Cosine Distance）。衡量算子输出与标准答案之间的误差，数值越接近 0.00e+00 表示精度越高。
### 2. 基础配置 (Configuration)
* **Op**: 被测试的算子名称（如 `fused_rope_fwd`）。
* **dtype**: 测试使用的数据精度（如 `float16`, `bfloat16`）。
* **Shape**: 输入数据的维度形状（例如 `(4096 4 32 128)`）。
* **Samples**: 性能测试采样的迭代次数。
### 3. 延迟与稳定性 (Latency & Stability)
* **CPU Time**: 算子在 CPU 侧的平均调度/执行时间。
* **GPU Time**: 算子在 GPU 侧的实际平均执行时间（核心性能指标）。
* **Noise**: 性能波动率。表示多次测试中执行时间的抖动比例，数值越小表示性能越稳定。
* **Batch GPU**: 批量处理模式下的 GPU 时间参考值。
### 4. 吞吐与利用率 (Throughput & Efficiency)
* **Elem/s**: 元素吞吐率。每秒处理的数据元素个数（Elements per second），反映计算能力。
* **GlobalMem BW**: 全局内存带宽（Global Memory Bandwidth）。实际达到的显存传输速率（如 `2.528 TB/s`）。
* **BWUtil**: 带宽利用率（Bandwidth Utilization）。实际带宽与硬件理论带宽的比率，用于评估算子是否达到内存瓶颈。


## 环境安装与配置
mxbench提供了两种安装方式：自动安装脚本（推荐）和手动分步安装*

### 方式一：自动一键安装mxbench（推荐）
通过运行脚本自动完成 `mcoplib` 和 `mxbench` 的环境配置与安装。
```shell
#进入项目根目录下的 `benchmark` 目录：
cd mcoplib/benchmark
#运行环境构建脚本：
./build_env_local.sh
注意 目录下 build_env.sh 脚本是给jinkens 进行自动化构建CI/CD dailytest流程用的 ， 本地编译安装执行 ./build_env_local.sh
```

### 方式二：手动安装
如果需要手动控制安装过程，请按照以下顺序执行。
#### 1. 安装 mxbench (C++ Core)
编译 mxbench 的 C++ 后端支持。
```bash
# 进入 mxbench 目录
cd /path/to/source/dir/mcoplib/mxbench
source env.sh
# 创建构建目录
mkdir build && cd build 
# 执行 CMake 配置与编译
cmake_maca -DCMAKE_CXX_STANDARD=17 \
           -DCMAKE_CUDA_STANDARD=17 \
           -DCMAKE_CUDA_ARCHITECTURES=80 \
           -DCMAKE_CUDA_FLAGS="-I/workspace/mcoplib/mxbench/install/include" \
           -DCMAKE_CXX_FLAGS="-Wno-unused-parameter -Wno-error -Wno-implicit-float-conversion -I/workspace/mcoplib/mxbench/install/include" \
           .. && make_maca VERBOSE=1
```

#### 2. 安装 mxbench (Python Interface)
构建并安装 mxbench 的 Python 接口 whl 包。
```bash
# 进入 python 目录
cd /path/to/source/dir/mcoplib/mxbench/python
# 设置编译变量
source env.sh
# 设置 mxbench 安装目录环境变量
export NVBENCH_INSTALL_PATH='/workspace/mcoplib/mxbench/install'
# 安装方式 A：开发者模式 (推荐)
python setup.py develop
# 安装方式 B：打包并安装 whl
python setup.py bdist_wheel 
# 或者直接调用 conda python: /opt/conda/bin/python3 setup.py bdist_wheel -v
pip3 install ./dist/*.whl
```

## 使用方法
安装完成后，在 `benchmark` 目录下使用 `mcoplib_mxbenchmark_ops.py` 脚本进行测试。
### 基础命令
```bash
#查看帮助信息
python mcoplib_mxbenchmark_ops.py --help
#列出所有可用算子
python mcoplib_mxbenchmark_ops.py --list
```
### 性能测试
以下命令中的 `<OP_NAME>` 请替换为实际的算子名称（例如 `fused_bias_dropout`）

```bash
#默认测试，仅运行测试并输出结果，不保存文件。
python mcoplib_mxbenchmark_ops.py --op <OP_NAME>

#生成基准数据 (--generate)
#运行测试并将结果保存到 CSV 文件。如果 CSV 中没有该类记录，则新增记录。
python mcoplib_mxbenchmark_ops.py --op <OP_NAME> --csv statistics/mcoplib_ops_performance.csv --generate

#更新基准数据 (--update)
#选取 CSV 表格中配置条件（Device, Shape, Dtype 等）完全一致的记录进行比较。
#触发条件： 仅当当前 GPU Time 优于历史记录 5% 以上时，才会执行刷新并记录更好的性能指标。
python mcoplib_mxbenchmark_ops.py --op <OP_NAME> --csv statistics/mcoplib_ops_performance.csv --update

#对比模式 (--compare)
#在配置条件一致的情况下，对比当前运行结果与 CSV 中的历史记录
python mcoplib_mxbenchmark_ops.py --op <OP_NAME> --csv statistics/mcoplib_ops_performance.csv --compare
```

# mcoplib_mxbenchmark_ops.py 使用说明

`mcoplib_mxbenchmark_ops.py` 是单算子性能基准测试的入口脚本，用于对单个 kernel 算子执行精度验证 + nvbench 性能采样，并按需写入/对比/更新 CSV 基准库。

## 命令行参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--op <NAME>` | str | 是（除非用 `--list`） | 指定要测试的算子名称，必须是 `--list` 列出的算子之一 |
| `--list` | flag | 否 | 列出所有支持的算子并退出，不执行测试 |
| `--csv <PATH>` | str | 否 | CSV 结果文件路径，未指定时默认 `statistics/mcoplib_ops_performance_C500.csv` |
| `--generate` | flag | 三选一 | 生成模式：仅填充 CSV 中缺失的记录，已存在则跳过 |
| `--update` | flag | 三选一 | 更新模式：仅当当前 GPU Time 优于历史记录 5% 以上时才覆盖该行 |
| `--compare` | flag | 三选一 | 对比模式：与 CSV 历史记录对比，不写文件，性能差异 >5% 报错 |

> `--generate` / `--update` / `--compare` 三个模式互斥，每次只能选一个。若都不选，则只运行测试并打印结果，不操作 CSV。

## 三种工作模式

### 1. 生成模式 `--generate`
- **用途**：首次建立基准库或补全缺失算子。
- **行为**：运行算子 → 精度验证通过后采样性能 → 查询 CSV，若该算子（同 op_name/dtype/shape/device）已存在则跳过，否则追加一行。
- **输出标志**：`[APPEND] <op>` 表示新增，`[SKIP] <op> (Exists)` 表示已存在跳过，末尾 `[SUMMARY] Appended: N, Skipped: M`。

```bash
python mcoplib_mxbenchmark_ops.py --op rms_norm --generate --csv statistics/mcoplib_ops_performance_C500.csv
```

### 2. 更新模式 `--update`
- **用途**：在优化 kernel 后刷新基准，但仅保留更好的数据。
- **行为**：运行算子 → 与 CSV 中同配置的历史行比较 → 仅当当前 GPU Time 比历史值快 5% 以上时才覆盖，否则保留历史。
- **前置条件**：CSV 文件必须存在且包含该算子记录，否则报错退出。
- **输出标志**：`[UPDATE] <op> | Gain: NN.NN%` 表示已更新，末尾 `[SUMMARY] Updated: N, Kept: M`。

```bash
python mcoplib_mxbenchmark_ops.py --op rms_norm --update --csv statistics/mcoplib_ops_performance_C500.csv
```

### 3. 对比模式 `--compare`
- **用途**：回归测试，检测性能是否退化。
- **行为**：运行算子 → 与 CSV 历史记录对比 → 打印 `Performance Comparison` 表格（Current vs Base）→ 输出 `Acc verify:Pass/Fail` 与 `Performance verify:NN.NN%`（base/current×100，<100% 表示当前更慢）→ 性能退化 >5% 时标记为失败。
- **前置条件**：CSV 文件必须存在；若该算子不在 CSV 中，输出 `Acc verify:None / Performance verify:None` 并跳过。
- **不写 CSV**。

```bash
python mcoplib_mxbenchmark_ops.py --op rms_norm --compare --csv statistics/mcoplib_ops_performance_C500.csv
```

## 执行流程
1. 解析参数 → 校验 `--op` 是否在 `SUPPORTED_OPERATORS` 白名单中。
2. 从 `config/<op>.json` 加载配置，从 `runners/mcoplib_mxbenchmark_<op>_runners.py` 加载 runner 类。
3. 调用 `runner.run_verification()` 做精度校验（与参考实现对比余弦相似度）。**校验失败会调用 `os._exit(0)` 终止进程**，此时 stdout 缓冲可能未刷新。
4. 校验通过后，注册到 nvbench 执行性能采样（warmup 10 次 + 采样 N 次，N 由 config 的 `samples` 字段决定）。
5. 采样完成后，按模式做后处理：`--generate` 调 `perform_generate`，`--update` 调 `perform_smart_update`，`--compare` 调 `perform_comparison`。

## 输出说明
- `[VERIFY] <op> -> PASS/FAIL (1-CosSim: <diff>)`：精度验证结果。
- nvbench 表格：含 `Acc_Pass / Cos_Dist / Op / dtype / Shape / Samples / CPU Time / GPU Time / Elem/s / GlobalMem BW / BWUtil` 等列（详见本文档上方"性能指标"章节）。
- `[APPEND]/[SKIP]/[UPDATE]` 行：CSV 写入动作。
- `[SUMMARY]` 行：本轮写入/跳过/更新的计数。

## CSV 文件格式
默认路径 `statistics/mcoplib_ops_performance_C500.csv`，表头：
```
op_name,Device,Device Name,Acc_Pass,Cos_Dist,Op,dtype,Shape,Samples,
CPU Time (sec),Noise,GPU Time (sec),Noise,Elem/s (elem/sec),
GlobalMem BW (bytes/sec),BWUtil,Samples,Batch GPU (sec)
```
每行对应一次基准测试记录，以 `op_name` 为主键。

# testall.py 批量测试使用说明

`testall.py` 是批量驱动脚本，通过调用 `mcoplib_mxbenchmark_ops.py` 对全部支持算子逐一执行基准测试，并汇总每算子的精度/性能结果与成功/失败计数。

## 命令行参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `--generate` | flag | 三选一（默认） | 生成模式：仅填充 CSV 中缺失的算子记录，已存在则跳过 |
| `--compare` | flag | 三选一 | 对比模式：每算子与 CSV 历史基准对比，性能退化 >5% 标记失败 |
| `--update` | flag | 三选一 | 更新模式：覆盖 CSV 中每算子的记录 |
| `--csv <PATH>` | str | 否 | CSV 路径，默认 `statistics/mcoplib_ops_performance_C500.csv` |
| `--ops <a,b,c>` | str | 否 | 仅运行指定的算子子集（逗号分隔） |
| `--dry-run` | flag | 否 | 仅枚举算子列表不执行测试 |

> 不指定任何模式参数时，默认等同于 `--generate`。

## 工作流程
1. **枚举算子**：调用 `python mcoplib_mxbenchmark_ops.py --list`，解析 `  * ` 前缀行获取全部支持算子（当前 116 个）。
2. **逐个执行**：对每个算子调用 `python mcoplib_mxbenchmark_ops.py --op <name> --<mode> --csv <path>`，超时 600 秒。
3. **解析输出**：从子进程 stdout 用正则提取 `[VERIFY]` / `Acc verify:` / `Performance verify:` / `[APPEND]/[SKIP]/[UPDATE]` 等标志。
4. **分类判定**：根据模式与捕获的标志判定每算子为 `SUCCESS` / `FAILED` / `SKIPPED`。
5. **汇总输出**：打印每算子结果表格 + 末尾 `SUMMARY` 行（total/success/failed/skipped）+ 失败算子名单；同时写入 `testall_output.txt` 详细日志。

## 判定规则

### `--generate` 模式
- `[APPEND]` → SUCCESS（新增）
- `[SKIP] (Exists)` → SKIPPED（已存在）
- 校验失败（`os._exit(0)` 导致无 `[VERIFY]` 行）→ 回退检查 CSV 中是否新增该算子行：未新增 → FAILED "no result"；已新增 → SUCCESS "appended(csv)"

### `--compare` 模式
- `Acc verify:Pass` + `Performance verify:NN%`，且 `slowdown_pct = (1 - NN/100) × 100 ≤ 5%` → SUCCESS
- `slowdown_pct > 5%` → FAILED（性能退化超标）
- `Acc verify:Fail/None` → FAILED
- 算子不在 CSV 中（`Acc verify:None / Performance verify:None`）→ SKIPPED "no baseline"

### `--update` 模式
- `[UPDATE]` → SUCCESS（已更新）
- 校验失败且 CSV 中无该算子行 → FAILED "no result"

## 使用示例

```bash
cd mcoplib/benchmark
source ../env.sh
export PATH=$PATH:/opt/conda/bin

# 1. 查看将测试哪些算子（不实际执行）
python testall.py --dry-run

# 2. 生成模式：批量补全缺失算子（默认模式，可省略 --generate）
python testall.py --generate
python testall.py                        # 等同于 --generate

# 3. 对比模式：全量回归测试，找出性能退化的算子
python testall.py --compare

# 4. 更新模式：全量刷新基准数据
python testall.py --update

# 5. 仅测试指定算子子集
python testall.py --generate --ops rms_norm,silu_and_mul,fused_moe_gate_deepseek

# 6. 指定自定义 CSV 路径
python testall.py --compare --csv statistics/my_baseline.csv
```

## 输出说明

### 每算子进度行（控制台实时输出）
```
[1/116] rms_norm ... SUCCESS (acc=Pass perf=180.69%) [48.7s]
[2/116] copy_blocks ... FAILED (no result) [19.0s]
[3/116] silu_and_mul ... SKIPPED (exists) [40.2s]
```

### 每算子结果表格（测试完成后打印）
| 列 | 含义 |
|----|------|
| Op | 算子名称 |
| Acc | 精度验证结果（Pass/Fail/None/-） |
| Perf | 性能比率（`NN.NN%`，仅 `--compare` 模式） |
| Status | SUCCESS / FAILED / SKIPPED |
| Detail | 失败原因或成功详情（如 "appended" / "exists" / "perf=85.20% (slow 14.80%)" / "no baseline"） |
| Time | 该算子耗时 |

### 汇总摘要（末尾）
```
SUMMARY | mode=generate | total=116 success=98 failed=3 skipped=15 | 1820.5s
FAILED OPS (3):
  - copy_blocks
  - paged_attention_v1
  - top_k_per_row

No failures.        # 无失败时显示
```

### 详细日志文件
全部算子的逐个执行详情（含子进程返回码、解析到的标志位、耗时）写入 `testall_output.txt`，便于事后排查。

## 退出码
- `0`：全部成功（含 SKIPPED），无 FAILED
- `2`：存在 FAILED 算子
- `1`：无算子可运行（如 `--list` 失败或 `--ops` 过滤后为空）

## 注意事项
1. 单算子脚本在校验失败时会调用 `os._exit(0)`，导致 stdout 缓冲未刷新、`[VERIFY] FAIL` 行丢失。`testall.py` 通过回退检查 CSV 中算子行是否存在来兜底判定，确保此类静默失败不会被误报为成功。
2. `--compare` / `--update` 模式要求 CSV 已存在；若不存在，`testall.py` 会自动创建带表头的空 CSV，使每个算子走 "no baseline" SKIPPED 路径而非硬退出。
3. 性能退化阈值固定为 5%，与单算子脚本 `--update` 的 "优于 5% 才更新" 规则对称。
4. 全量运行 116 个算子耗时较长（每算子 20-90 秒），建议先用 `--ops` 子集验证流程，再全量执行。
