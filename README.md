# mcOpLib
## 编译
note: 请优先在vllm/sglang的发布的镜像中进行编译， 比如:
```shell
docker run  -it  --name=mcoplib-build  --shm-size 16384m --device=/dev/dri --device=/dev/mxcd --group-add=video  --network=host --ulimit memlock=-1 --privileged=true   -v /sw_home/yiyu/:/home/yiyu  -v /pde_ai/models:/models  ai-master/maca/sglang:0.5.1-maca.ai20251013-45-torch2.6-py310-ubuntu22.04-amd64  /bin/bash
```

安装编译依赖：
```shell
#安装cmake, 注意：如果是镜像中编译，又是把代码放在到网络共享盘中的，则先需要切换到root用户，在root用户下安装cmake
pip3 install cmake==3.26.3 -i  https://repo.metax-tech.com/r/pypi/simple
#安装pybind11
pip3 install pybind11 -i  https://repo.metax-tech.com/r/pypi/simple
pip3 install build -i  https://repo.metax-tech.com/r/pypi/simple
pip3 install setuptools-scm==8.0 -i  https://repo.metax-tech.com/r/pypi/simple
pip3 install setuptools==69.5.1 -i  https://repo.metax-tech.com/r/pypi/simple
```
环境变量设置：

```shell

#切换到源码目录, 执行以下脚本设置环境变量：
#线上jenkins编译执行命令：
    source env.sh
#线下本地编译执行命令:
    source env_loacl.sh
```

项目源码编译：

```shell
cd  /path/source/code/dir/mcoplib
#源码编译命令， 该命令不会显示出编译信息，如果需要查看编译信息添加参数："-v" 或者 "-vv" 或者"-vvv"
#编译完成后,生产的动态库及产物在源码目录下的mcoplib下面,不支持增量编译
pip install -e . --no-build-isolation
pip install -e . --no-build-isolation -v 
pip install -e . --no-build-isolation -vv
pip install -e . --no-build-isolation -vvv
#mcoplib 也支持通过python来编译，如下两个命令支持增量编译：
python setup.py develop
#build_ext --inplace 只关注扩展构建策略本身；develop 在构建的基础上还做“安装/注册/依赖处理”
python setup.py build_ext --inplace

#编译打印WCUDA详细信息
export WCUDA_DEBUG=1
```
note: 通过pip install -e . --no-build-isolation -v或者-vv, -vvv命令编译时， 并不会打印出setup.py中的print信息，因为pip 对该子进程使用管道（pipe）捕获 stdout/stderr，以便在失败时回显或在 verbose 模式下合并显示， 也即只有在编译失败时或者编译成功完成后才会打印出setup.py中的print信息

CUTLASS OP API接口编译控制
```shell
#默认开启CUTLASS OP API的编译
#也可以通过环境变量来控制CUTLASS OP 的编译
#开启
export ENABLE_BUILD_CUTLASS_OP=1
#关闭
export ENABLE_BUILD_CUTLASS_OP=0
```

项目打包命令：

```shell
#先设置环境变量
cd  /path/source/code/dir
python  -m build  --no-isolation
#打包命令执行完成后， whl包在源码 dist目录下， 比如：mcoplib-0.1.0+maca3.0.0.8.torch2.6-cp310-cp310-linux_x86_64.whl
```
### 多平台编译(C600/C600U/C588)
```shell
#添加以下环境变量
export CUCC_TARGETS="xcore1000, xcore1089,xcore1500,xcore1501"
```
## 安装

```shell
pip3 install mcoplib-0.1.0+maca3.0.0.8.torch2.6-cp310-cp310-linux_x86_64.whl
```
## mcoplib CV Op Kernel 编译打包
```shell
#切换到源码目录（~/mcOplib/gerrit_mcoplib/mcoplib_dev/mcoplib）, 执行一下命令
source env.sh
cd /path/source/code/dir/mcoplib/op/cv/
#执行命令 配置 + 构建
cmake_maca -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake_maca --build build -j$(nproc)
# 生成 deb
cd build
cpack -G DEB
```

### CV Op Deb包安装

```shell
#cd pkg 目录
dpkg -i mcoplib_cv-0.2.0-Linux.deb
#sudo
sudo dpkg -i mcoplib_cv-0.2.0-Linux.deb
#安装完成后，/opt/maca-ai/mcoplib目录结构如下：
root@lt-srv-10-2-182-63:~/mcoplib# tree
.
|-- include
|   |-- arithm.h
|   |-- calsum.h
|   |-- count_nozero.h
|   |-- meanstdev.h
|   |-- process_interface.h
|   |-- split.h
|   `-- utils.h
`-- lib
    `-- libmcoplib_cv.so
```

### Mcoplib  cv Op kernel 测试
```shell
#mcoplib cv op kernel测试需要先安装mcoplib_cv-0.2.0-Linux.deb包, deb包安装后会/opt/maca-ai/mcoplib目录下存在mcoplib cv库及头文件
dpkg -i mcoplib_cv-0.2.0-Linux.deb
#切换到源码目录（~/mcOplib/gerrit_mcoplib/mcoplib_dev/mcoplib）, 执行一下命令
source env.sh
cd  /path/source/code/dir/mcoplib/unit_test/cpp
mkdir build
cmake_maca .. && make_maca
```

## VLLM自定义算子使能安装

```shell
pip3 install mcoplib-0.1.0+maca3.0.0.8.torch2.6-cp310-cp310-linux_x86_64.whl
```

## 获取版本信息
```shell
#安装mcoplib包后， shell终端执行一下命令获取版本信息
mcoplib_version
````

##  通过环境变量控制编译

```shell

#BUILD_VLLM_SUBMODULE 环境变量控制vllm op 算子是否编译，默认开启
export BUILD_VLLM_SUBMODULE=OFF 
#BUILD_SGLANG_SUBMODULE 环境变量控制sglang op 算子是否编译， 默认开启， sglang 推理框架中一般都依赖vllm op kernel
export  BUILD_SGLANG_SUBMODULE=OFF 
#BUILD_LMDEPLOY_SUBMODULE 环境变量控制lmdeploy op 算子是否编译， 默认开启
export BUILD_LMDEPLOY_SUBMODULE=OFF
#BUILD_DEFAULT_OP_SUBMODULE 环境变量控制默认 op 算子是否编译， 默认开启， 一般情况下默认算子必须开启，存在复用，且import mcoplib时，默认会import mcoplib.op， 如何开启，会导致import错误
export BUILD_DEFAULT_OP_SUBMODULE=OFF 
#多个算子编译模块控制
export BUILD_VLLM_SUBMODULE=OFF  BUILD_SGLANG_SUBMODULE=OFF BUILD_LMDEPLOY_SUBMODULE=OFF
```

## 动态控制算子入参信息终端输出或参数dump到本地磁盘
### 一、功能概述

每个算子入口函数里埋了两行宏，运行期可以在不重新编译的情况下，通过环境变量打开，打印或落盘该算子的入参：

- `DEBUG_TRACE_PARAMS(...)` —— 把入参的 shape / dtype / device / 值打印到终端。
- `DEBUG_DUMP_PARAMS(...)` —— 把入参序列化成 JSON，写到磁盘。
---
### 二、运行流程

#1、环境参数

| 变量 | 作用 | 默认 | 取值 |
|---|---|---|---|
| `MCOP_DEBUG_TRACE` | 终端打印 把算子输入参数打印在终端 | 关 | `1`/`ON`/`on` |
| `MCOP_DEBUG_PARAMS_DUMP` | JSON 落盘 把算子输入参数dump到本地磁盘 | 关 | `1`/`ON`/`on` |
| `MCOP_DEBUG_FILTER` | 指定功能 要调试哪些算子 | 关（未设置=不触发） | `all` / 逗号分隔子串 |
| `MCOP_TENSOR_DUMP_SAMPLE_SIZE` | 每个 tensor 采样多少个元素（前 N + 后 N，共 2N） | `20` | 正整数 |
| `MCOP_TENSOR_DUMP_FULL` | 全量 dump 整个 tensor，不采样 | 关 | 非 `0` 即开 |
| `MCOP_DEBUG_DUMP_MAX_CALLS` | 每个算子最多记录多少次调用（trace/dump 共用） | `20` | 正整数 |
| `MCOP_DEBUG_DUMP_DIR` | dump 输出目录（配合 sitecustomize.py 一个命令一个文件夹） | 无（自动生成带时间戳目录） | 目录路径 |

#2、打印启动流程

```bash
export MCOP_DEBUG_TRACE=1               # 打开终端打印 算子输入参数打印在终端
export MCOP_DEBUG_PARAMS_DUMP=1         # 打开 JSON 落盘 算子输入参数dump到本地磁盘
export MCOP_DEBUG_FILTER=all            # 指定功能 要调试哪些算子

```
- 只设 `MCOP_DEBUG_TRACE=1`、不设 `FILTER` → 什么都不打印（`FILTER` 未设置等于"一个算子都不选"）。
- 只设 `MCOP_DEBUG_FILTER=all`、不开trace/dump → 什么都不打印（没有指定输出格式）。
- trace/dump的两个开关彼此独立，可以只开一个
```
**取值约定**：
trace/dump两个开关用 `1` / `ON` / `on` 表示打开
filter开关 `MCOP_DEBUG_FILTER` 详解

```bash
export MCOP_DEBUG_FILTER=all                  # 所有算子
export MCOP_DEBUG_FILTER=fused_rope           # 函数名包含 "fused_rope" 的算子（子串匹配）
export MCOP_DEBUG_FILTER=fused_rope,rms_norm  # 多个，逗号分隔，前后空格会被去掉
```

### 三、算子入参Dump本地示例
#sample 1 ：打印指定算子 `fused_silu_mul_dq_reorder_quant`
#(1)启动打印

```bash
export MCOP_DEBUG_TRACE=1
export MCOP_DEBUG_PARAMS_DUMP=1
export MCOP_DEBUG_FILTER=fused_silu_mul_dq
python unit_test/test_fused_silu_mul_dq_reorder_quant.py
```
#（2）输出文件 `mcoplib_op_params_dump_时间戳/fused_silu_mul_dq_quant_reordered_topk_interface.json`
```json
{
  "function": "fused_silu_mul_dq_quant_reordered_topk_interface",
  "parameters": [
    {
      "name": "out",
      "type": "at::Tensor",
      "dtype": "Char",
      "shape": "[512, 2048]",
      "value": "[... (unsupported dtype: Char), ..., ... (unsupported dtype: Char)] (showing 40 of 1048576 elements, set MCOP_TENSOR_DUMP_FULL=1 for all) [data_ptr=0x7f56c0200000]",
      "bytes": 1048576
    },
    {
      "name": "scale",
      "type": "at::Tensor",
      "dtype": "Float",
      "shape": "[512, 1]",
      "value": "[5.19539, 6.94539, 5.13291, 5.88289, 5.0704, 5.50795, 5.75788, 3.9727, 5.7579, 5.13291, 5.47667, 3.55082, 5.85164, 6.16416, 7.16413, 4.50789, 4.75789, 5.25792, 8.26583, 6.60168, ..., 4.59375, 4.3125, 5.8125, 7.28125, 6.5, 4.5625, 4.96875, 5.5625, 5.3125, 6.5625, 4.90625, 7.34375, 8.1875, 5.28125, 6.84375, 6.09375, 7.125, 5.125, 6.625, 6.65625] (showing 40 of 512 elements, set MCOP_TENSOR_DUMP_FULL=1 for all) [data_ptr=0x7f56cb601200]",
      "bytes": 2048
    },
    {
      "name": "input",
      "type": "at::Tensor",
      "dtype": "BFloat16",
      "shape": "[512, 4096]",
      "value": "[-0.925781, 0.992188, -0.482422, -0.617188, -0.425781, 1.07031, -0.231445, -2.375, -2.64062, -0.628906, 0.416016, -1.90625, 0.145508, 0.320312, -0.753906, -0.141602, -0.121094, -0.310547, 1.27344, 2.40625, ..., -1.64062, -0.945312, 0.992188, 0.707031, -1.41406, 1.82812, 0.628906, 0.0179443, -0.296875, 0.9375, -1.21094, 1.97656, 0.186523, -0.0610352, 0.209961, 0.3125, 1.51562, 1.14062, -0.816406, -0.170898] (showing 40 of 2097152 elements, set MCOP_TENSOR_DUMP_FULL=1 for all) [data_ptr=0x7f56d2600000]",
      "bytes": 4194304
    },
    {
      "name": "reorder_topk_ids",
      "type": "at::Tensor",
      "dtype": "Long",
      "shape": "[512]",
      "value": "[1, 7, 2, 0, 5, 2, 5, 5, 4, 4, 4, 2, 5, 5, 0, 3, 4, 2, 5, 1, ..., 2, 4, 2, 4, 5, 3, 5, 1, 0, 4, 6, 0, 4, 3, 2, 0, 5, 4, 7, 3] (showing 40 of 512 elements, set MCOP_TENSOR_DUMP_FULL=1 for all) [data_ptr=0x7f56cb600000]",
      "bytes": 4096
    },
    {
      "name": "w2_scale",
      "type": "at::Tensor",
      "dtype": "Float",
      "shape": "[0]",
      "value": "[] [data_ptr=0x0]",
      "bytes": 0
    },
    {
      "name": "start_expert_id",
      "type": "long",
      "dtype": "long",
      "shape": "[]",
      "value": "0",
      "bytes": 8
    },
    {
      "name": "end_expert_id",
      "type": "long",
      "dtype": "long",
      "shape": "[]",
      "value": "7",
      "bytes": 8
    }
  ]
}


```

#sample 2 :打印多个算子 
#（1）启动打印
```bash
export MCOP_DEBUG_TRACE=1
export MCOP_DEBUG_PARAMS_DUMP=1
export MCOP_DEBUG_FILTER=all
python unit_test/run_ops.py fused_bias_gelu.py topk_sigmoid.py #把 fused_bias_gelu.py 和 topk_sigmoid.py 塞进同一进程按序跑输出文件
```
#(2) 输出文件 `mcoplib_op_params_dump_时间戳/fused_gelu_fwd.json` 和 `fused_gelu_bwd.json`，**每个文件正好 20 个 JSON 对象**（截取 1 个示意）：

```json
{
  "function": "fused_gelu_bwd",
  "parameters": [
    {
      "name": "input",
      "type": "at::Tensor",
      "dtype": "Float",
      "shape": "[4096, 1, 14336]",
      "value": "[-0.0394715, 1.0781, -0.45613, -1.68752, 0.63038, -0.217091, -0.0953174, 1.87518, 0.982992, 0.885668, 0.538419, 0.167188, -0.697348, 0.0247884, -0.195, -0.234183, -0.814877, -0.458664, -0.0596179, 1.20023, ..., 0.927596, -0.750395, -0.806976, 0.240616, -0.108705, 0.712145, 0.580186, 0.145358, -1.6756, -0.12453, 0.522908, -0.248826, 1.78173, 1.15039, -0.316377, 0.299093, -0.243433, 0.942879, -0.809986, -0.489415] (showing 40 of 58720256 elements, set MCOP_TENSOR_DUMP_FULL=1 for all) [data_ptr=0x7f71ad400000]",
      "bytes": 234881024
    },
    {
      "name": "input1",
      "type": "at::Tensor",
      "dtype": "Float",
      "shape": "[4096, 1, 14336]",
      "value": "[0.533128, -1.14377, -1.18346, -0.148431, 0.412671, 0.204955, 0.498725, 0.360008, -0.973992, -1.07118, -1.09559, -1.2775, -0.407253, -1.53412, 1.26002, 2.51654, 0.648574, -1.47942, 0.75869, 1.20661, ..., -1.56871, -2.74373, -0.297842, -0.965582, -0.100965, 1.36005, -0.0581853, 1.09586, 1.32505, -0.475635, 0.302815, -0.535586, -1.75935, 0.00314422, -0.26229, -0.484426, 0.560441, 1.48841, 0.414848, 0.968975] (showing 40 of 58720256 elements, set MCOP_TENSOR_DUMP_FULL=1 for all) [data_ptr=0x7f718f200000]",
      "bytes": 234881024
    },
    {
      "name": "bias",
      "type": "at::Tensor",
      "dtype": "Float",
      "shape": "[14336]",
      "value": "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ..., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (showing 40 of 14336 elements, set MCOP_TENSOR_DUMP_FULL=1 for all) [data_ptr=0x7f71aac00000]",
      "bytes": 57344
    }
  ]
}

......

```

## Getting started

### samples

```python
#mcoplib op  以及 vllm  _C中算子调用示例
import contextlib
from typing import TYPE_CHECKING, Optional, Union

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.scalar_type import ScalarType
from mcoplib import op as ops

logger = init_logger(__name__)

if not current_platform.is_tpu() and not current_platform.is_xpu():
    try:
        import mcoplib._C
    except ImportError as e:
        logger.warning("Failed to import from vllm._C with %r", e)

supports_moe_ops = False
with contextlib.suppress(ImportError):
    import mcoplib._moe_C  # noqa: F401
    supports_moe_ops = True

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake

def rms_norm(
    hidden_states: Tensor,
    weight: Tensor,
    epsilon: float,
) -> Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    weight = weight.to(torch.float32)
    output = torch.empty_like(hidden_states)
   
    ops.rms_norm(output, hidden_states, weight, epsilon, None, None,False)#mcoplib op模块中的算子

# page attention ops
def paged_attention_v1(
    out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    #mcoplib vllm 中的op kernel _C模块的paged_attention_v1算子调用
    torch.ops._C.paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank, blocksparse_local_blocks,
        blocksparse_vert_stride, blocksparse_block_size,
        blocksparse_head_sliding_step)

#sglang sgl_kernel调用示例

import torch

try:
    import mcoplib.sgl_kernel as sgl
except ImportError as e:
    print("Failed to import from sgl_kernel with %r", e)


try:
    import mcoplib.sgl_grouped_gemm_cuda
except ImportError as e:
    print("Failed to import from sgl_grouped_gemm_cuda with %r", e)

#功能：mla中，对q做rotary_emb，对latent_cache做rms_normal，更新latent_cache和kv_a，之后对latent_cache做rotary_emb。
#     调用torch的kv_b_proj计算kv，将数据从kv拷贝到k和v，从latent_cache中拷贝数据到k
#输入：
#输出：
#限制：
def fused_mla_normal_rotary_emb(
    kv_a:torch.tensor,
    kv_b_proj,
    q:torch.tensor, # [bs, 128, 192], dtype=bf16
    latent_cache:torch.tensor, # [bs, 576], dtype=bf16
    positions:torch.tensor, # [bs], dtype=int64
    cos_sin_cache:torch.tensor, # [max_position_embeddings, 64], dtype=float
    norm_weight:torch.tensor, # [512], dtype=bf16
    k:torch.tensor, # [bs, 128, 192], dtype=bf16
    v:torch.tensor, # [bs, 128, 192], dtype=bf16
    q_len:int, #bs
    qk_nope_head_dim:int, #128
    qk_rope_head_dim:int, #64
    kv_lora_rank:int, #512
    v_head_dim:int, #128
    num_local_heads:int , #128
):
    out = torch.ops.sgl_kernel.fused_mla_RMS_rotary_emb(q, latent_cache, cos_sin_cache, positions, norm_weight, kv_a, q_len, num_local_heads, kv_lora_rank, qk_rope_head_dim, qk_nope_head_dim)
    if out != 0:
        print("Failed to call mcoplib ops.fused_mla_RMS_rotary_emb")
    kv = kv_b_proj(kv_a)
    kv = kv[0] if isinstance(kv, tuple) else kv
    out = torch.ops.sgl_kernel.fused_mla_normal_kv_element_wise(kv, latent_cache, k, v, q_len, num_local_heads, kv_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim)
    if out != 0:
        print("Failed to call mcoplib ops.fused_mla_normal_kv_element_wise")
    return q, k, v, latent_cache

```


## QA
- 执行python  -m build  --no-isolation 报错：/opt/conda/bin/python: No module named build.__main__; 'build' is a package and cannot be directly executed

    Answer：Python 尝试执行 `python -m build` 时，找不到 `build/_main_.py` 文件，所以无法将 `build` 当作一个 **可执行模块**（即 `__main__` 模块）运行， 你当前环境中的 `build` 不是 PyPA 官方的 `build` 工具包.
需要安装build包： pip install --force-reinstall build
- mcoplib构建打包后， 无法显示版本信息，包文件目录下没有version文件
    Answer: 这是因为构建环境中没有安装git命令导致的，请在构建环境中安装git命令
- 编译时出现错误：FileNotFoundError: [Errno 2] No such file or directory: 'cmake_maca'
    Answer: 请在编译前执行下环境变量env.sh，cd /code/dir/mcoplib/ && source env.sh
- 编译时报错：cmake error while loading shared libraries: libssl.so.1.1: cannot open shared object file: No such file or directory
Traceback (most recent call last):
    Answer: cmake版本太高，请安装低版本，镜像中的open-ssl版本很低与高版本的cmake无法匹配，所有报错，请卸载高版本cmake，安装低版本的cmake，pip3 install cmake==3.26.3 -i  https://repo.metax-tech.com/r/pypi/simple
- 当编译及运行出现错误：ERROR: MACA minimum compatibility version mismatch, aborting， 说明不符合最低MACA release 版本要求， 如果因为排查问题， 需要定位版本差异，可通过以下凡是跳过这个错误(切记： 不保证编译运行没有问题)：
    1：运行时， 修改version文件 中Min_Compatibility_Maca_Version = '3.7.0' 字段，即可 
    2：编译时， 修改/opt/maca/Version.txt 对应的MACA版本
- from mcoplib.op import fused_silu_mul_dq_mask_quant_fp8_nopack，ImportError: /opt/conda/lib/python3.10/site-packages/mcoplib/op.cpython-310-x86_64-linux-gnu.so: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb
    解答：这是因为mcoplib 要求torch2.10版本， 安装的包也是torch2.10编译出来的， 如果在torch 2.8上进行安装时，就会报错，请保证安装的环境与编译的环境一致。
## Release
### Release 0.4.13
- support sglang  0.5.17 op
- support for vllm 0.28.0  op kernels

#### Op New
- add region_topk_ids operator
- add indexer_norm_rope fused operator (indexer RMSNorm + RoPE)
- add fused_deepseek_v4_qnorm_rope_kv_insert operator
- add fused_minimax_m3_qknorm_rope_kv_insert operator
- add jit fused_gemma_qknorm_rope operator on C600U sglang
- add topk_softplus_sqrt moe routing operator
- add moe_step4_weighted_topk_gather operator for step-4 model
- add silu_and_mul_clamp operator for step-4 model
- add merge_attn_states operator
- add fused_layernorm_dynamic_per_token_quant kernels
- add triton_causal_conv1d_fwd operator
- add fused_sigmoid_gating_delta_rule_update operator
- add mhc_pre_big_fuse optimized CUDA specialization
- add rms_norm optional-weight / per-block-quant / static-fp8-quant variants
- add fused_kimi_k3_mla_kv_concat
- fused_kimi_k3_mla_kv_concat_quant_fp8
- fused_gdn_decode_post_conv_mtp
- direct_dcp_a2a_lse_reduce
- direct_dcp_kv_gather
- direct_dcp_q_gather

#### Op Optimization
- optimize single_grouped_topk kernel for GLM-5.1 prefill
- optimize kpool_topk_transform for GLM-5 workloads
- optimize fused Q/K RoPE and add preallocated output API
- optimize act_and_mul_kernel for JoyAI-llm-flash TP4/TP8
- optimize silu_and_mul_with_clamp with flat/2D vectorized kernels and rcp/expf fast path
- optimize per_token_cast_to_fp8 and fix bit-exact accuracy
- optimize moe_sum_reduce kernel on C600U sglang
- optimize weighted_topk_gather kernel
- optimize rms_norm for Gemma4-31B-it model
- optimize minimax_reduce_rms_kernel q_only path
- fix fused_silu_mul_per_group_quant accuracy bug
- fix topk_sigmoid / topk_softmax accuracy error

#### Op Update
- update kimi_k3_attn_res
- update moe_lora_align_block_size
- update cp_gather_and_upconvert_fp8_kv_cache
- update cp_gather_cache
- update dsv3_fused_a_gemm
- update reshape_and_cache_flash
- update persistent_topk

#### Common
- update vLLM kernels to v0.28.0 baseline
- fix custom op torch.compile error under sglang piecewise graph

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
