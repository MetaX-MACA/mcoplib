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

### 方式一：自动安装（推荐）
通过运行脚本自动完成 `mcoplib` 和 `mxbench` 的环境配置与安装。
```shell
#进入项目根目录下的 `benchmark` 目录：
cd mcoplib/benchmark
#运行环境构建脚本：
./build_env.sh
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
