# mcOpLib
## Compilation
note: Please prioritize compiling within the published vllm/sglang Docker images, for example:
```shell
docker run  -it  --name=mcoplib-build  --shm-size 16384m --device=/dev/dri --device=/dev/mxcd --group-add=video  --network=host --ulimit memlock=-1 --privileged=true   -v /sw_home/metax/:/home/metax  -v /pde_ai/models:/models  ai-master/maca/sglang:0.5.1-maca.ai20251013-45-torch2.6-py310-ubuntu22.04-amd64  /bin/bash
```

Install build dependencies:
```shell
# Install cmake. Note: If compiling inside a container and the code is stored on a network shared drive, you must first switch to the root user and install cmake as root.
pip3 install cmake==3.30.4 && pip3 install setuptools-scm==8.0
pip3 install pybind11
pip3 install build
```
Environment variable setup:

```shell
# Switch to the source code directory and execute the following command
source env.sh
```

Project source code compilation:

```shell
cd  /path/source/code/dir/mcoplib
# Source code compilation command. This command will not display compilation logs. If you need to view compilation logs, add parameters: "-v", "-vv", or "-vvv".
# After compilation is complete, the generated dynamic libraries and artifacts are located under `mcoplib` in the source directory. Incremental compilation is not supported.
pip install -e . --no-build-isolation
pip install -e . --no-build-isolation -v 
pip install -e . --no-build-isolation -vv
pip install -e . --no-build-isolation -vvv
# mcoplib also supports compilation via python. The following two commands support incremental compilation:
python setup.py develop
# "build_ext --inplace" focuses only on the extension build strategy itself; "develop" performs "installation/registration/dependency handling" in addition to building.
python setup.py build_ext --inplace

# Print detailed WCUDA information during compilation
export WCUDA_DEBUG=1
```
note: When compiling using the pip install -e . --no-build-isolation -v (or -vv, -vvv) command, print messages within setup.py will not be printed immediately. This is because pip uses a pipe to capture stdout/stderr from the subprocess in order to echo it upon failure or merge the display in verbose mode. Therefore, print messages from setup.py will only be displayed after compilation fails or completes successfully.

Project Packaging Command:

```shell
# First set environment variables
cd  /path/source/code/dir
python  -m build  --no-isolation
# After the packaging command finishes, the whl package will be in the source code's dist directory, for example: mcoplib-0.1.0+maca3.0.0.8.torch2.6-cp310-cp310-linux_x86_64.whl
```

## Installation

```shell
pip3 install mcoplib-0.1.0+maca3.0.0.8.torch2.6-cp310-cp310-linux_x86_64.whl
```
## mcoplib CV Op Kernel Compilation and Packaging
```shell
# Switch to the source directory (~/mcOpLib/gerrit_mcoplib/mcoplib_dev/mcoplib) and execute the following command
source env.sh
cd /path/source/code/dir/mcoplib/op/cv/
# Execute commands: Configure + Build
cmake_maca -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake_maca --build build -j$(nproc)
# Generate deb
cd build
cpack -G DEB
```

### CV Op Deb Package Installation

```shell
# cd pkg directory
dpkg -i mcoplib_cv-0.2.0-Linux.deb
# sudo
sudo dpkg -i mcoplib_cv-0.2.0-Linux.deb
# After installation is complete, the /opt/maca-ai/mcoplib directory structure is as follows:
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

### Mcoplib  cv Op kernel Test
```shell
# Testing the mcoplib cv op kernel requires installing the mcoplib_cv-0.2.0-Linux.deb package first.
# After installing the deb package, the mcoplib cv library and header files will be located in the /opt/maca-ai/mcoplib directory.
dpkg -i mcoplib_cv-0.2.0-Linux.deb
# Switch to the source code directory (~/mcOplib/gerrit_mcoplib/mcoplib_dev/mcoplib) and execute the following commands:
source env.sh
cd  /path/source/code/dir/mcoplib/unit_test/cpp
mkdir build
cmake_maca .. && make_maca
```

## Installation for Enabling VLLM Custom Operators

```shell
pip3 install mcoplib-0.1.0+maca3.0.0.8.torch2.6-cp310-cp310-linux_x86_64.whl
```

## Get Version Information
```shell
# After installing the mcoplib package, execute the following command in the shell terminal to retrieve version information:
mcoplib_version
````

## Control Compilation via Environment Variables

```shell

# The BUILD_VLLM_SUBMODULE environment variable controls whether vllm op operators are compiled; enabled by default.
export BUILD_VLLM_SUBMODULE=OFF 
# The BUILD_SGLANG_SUBMODULE environment variable controls whether sglang op operators are compiled; enabled by default.
# The sglang inference framework generally depends on the vllm op kernel.
export  BUILD_SGLANG_SUBMODULE=OFF 
# The BUILD_LMDEPLOY_SUBMODULE environment variable controls whether lmdeploy op operators are compiled; enabled by default.
export BUILD_LMDEPLOY_SUBMODULE=OFF
# The BUILD_DEFAULT_OP_SUBMODULE environment variable controls whether default op operators are compiled; enabled by default.
# Under normal circumstances, default operators must be enabled as they are reused.
# Additionally, when importing mcoplib, it defaults to importing mcoplib.op; if disabled, it will cause an import error.
export BUILD_DEFAULT_OP_SUBMODULE=OFF 
# Control over multiple operator compilation modules
export BUILD_VLLM_SUBMODULE=OFF  BUILD_SGLANG_SUBMODULE=OFF BUILD_LMDEPLOY_SUBMODULE=OFF
```

## Getting started

### samples

```python
# Example of calling operators in mcoplib op and vllm _C
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
   
    ops.rms_norm(output, hidden_states, weight, epsilon, None, None,False)# Operators in the mcoplib op module

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
    # Invocation of the paged_attention_v1 operator in the mcoplib vllm op kernel _C module
    torch.ops._C.paged_attention_v1(
        out, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank, blocksparse_local_blocks,
        blocksparse_vert_stride, blocksparse_block_size,
        blocksparse_head_sliding_step)

# sglang sgl_kernel invocation example

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



# Function: In MLA, apply rotary_emb to q, apply rms_normal to latent_cache, update latent_cache and kv_a, then apply rotary_emb to latent_cache.
#           Call torch's kv_b_proj to calculate kv, copy data from kv to k and v, and copy data from latent_cache to k.
# Input:
# Output:
# Limitations:
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
- Executing python -m build --no-isolation fails with error: /opt/conda/bin/python: No module named build.__main__; 'build' is a package and cannot be directly executed
    Answer：When Python tries to execute python -m build, it cannot find the build/__main__.py file, so it cannot run build as an executable module (i.e., __main__ module). The build in your current environment is not the official PyPA build toolkit. You need to install the build package: pip install --force-reinstall build
- After building and packaging mcoplib, version information cannot be displayed, and there is no version file in the package directory.
    Answer: This is caused by the lack of the git command in the build environment. Please install the git command in the build environment.
- Error during compilation: FileNotFoundError: [Errno 2] No such file or directory: 'cmake_maca'
    Answer: Please execute the environment variable script env.sh before compiling: cd /code/dir/mcoplib/ && source env.sh

## Release
### Release 0.3.1
- add cv op kernel
- support sglang 0.5.6 op 
- optimize mcoplib project build 
- support mxbench for auto test op kernel `s perfromance
- support profiler tools check op kernel `s perfromance
- support for vllm 0.13.0  op kernels
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
