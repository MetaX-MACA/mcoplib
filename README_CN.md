# mcOpLib
## 编译
note: 请优先在vllm/sglang的发布的镜像中进行编译， 比如:
```shell
docker run  -it  --name=mcoplib-build  --shm-size 16384m --device=/dev/dri --device=/dev/mxcd --group-add=video  --network=host --ulimit memlock=-1 --privileged=true   -v /home/metax/:/home/metax  -v /pde_ai/models:/models  ai-master/maca/sglang:0.5.1-maca.ai20251013-45-torch2.6-py310-ubuntu22.04-amd64  /bin/bash
```

安装编译依赖：
```shell
#安装cmake, 注意：如果是镜像中编译，又是把代码放在到网络共享盘中的，则先需要切换到root用户，在root用户下安装cmake
pip3 install cmake==3.30.4 && pip3 install setuptools-scm==8.0
pip3 install pybind11 
pip3 install build
```
环境变量设置：

```shell

#切换到源码目录, 执行一下命令
source env.sh
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

项目打包命令：

```shell
#先设置环境变量
cd  /path/source/code/dir
python  -m build  --no-isolation
#打包命令执行完成后， whl包在源码 dist目录下， 比如：mcoplib-0.1.0+maca3.0.0.8.torch2.6-cp310-cp310-linux_x86_64.whl
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
cd  ~/mcOplib/gerrit_mcoplib/mcoplib_dev/mcoplib/unit_test/cpp
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

try:
    import mcoplib.sgl_moe_fused_w4a16
except ImportError as e:
    print("Failed to import from sgl_moe_fused_w4a16 with %r", e)



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

## Release
### Release 0.2.0
- add cv op kernel
- support sglang 0.5.7 op 
- optimize mcoplib project build 
- support mxbench for auto test op kernel `s perfromance
- support profiler tools check op kernel `s perfromance
- support for vllm 0.11.2  op kernels
- support Project-customized op kernels
- support k-transformer op kernels
- support verl op kernels
- support all of mcopZoo op kernels

## Acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
