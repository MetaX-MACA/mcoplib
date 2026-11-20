"""mnnvl_multimem_reduce_scatter 的 host 端契约测试。

算子的 multimem 归约需要一份 NVLS multicast 映射：每张卡一个 local buffer，
外加由 mcMulticastCreate / mcMulticastBindMem 绑定出来的 multicast buffer，
两者的 -sizeof(Signal) 处还要各有一个跨卡可见的 Signal。mcoplib 目前没有把
这套分配能力暴露给 Python，因此本文件不做端到端数值测试，改为覆盖 kernel
launch 之前的全部校验分支：

  1. inp/out dtype 一致、out.numel() * world_size == inp.numel()、非空输入
  2. inp/out 连续性、input_size <= stage_sz_bytes、两个 buffer 非空
  3. world_size 取值、size 16 字节对齐、block_limit 范围

第 3 组检查位于 CustomAllreduce 成员函数开头，同样在任何 kernel 启动之前
抛错，因此不会触达 multicast 硬件，单卡即可验证。

fa 通过 init_custom_ar 构造：CustomAllreduce 的构造函数只保存 rank_、
world_size_ 和 signals[]，既不校验 IPC 指针也不发起跨卡通信，所以单卡就能
拿到一个真实的 fa 对象（此前用 fa=0 会在读 world_size_ 时解引用空指针）。
"""

import pytest
import torch

import mcoplib._C  # noqa: F401  (注册 torch.ops._C_custom_ar)

# CustomAllreduce 只把 ipc_tensors 存进 sg_.signals[]，不会解引用
_FAKE_SIGNAL_PTR = 0x1000

_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
_DTYPE_IDS = ("fp16", "bf16", "fp32")


def _get_op(name):
    if not hasattr(torch.ops, "_C_custom_ar"):
        pytest.skip("_C_custom_ar namespace is not registered")
    op = getattr(torch.ops._C_custom_ar, name, None)
    if op is None:
        pytest.skip(f"{name} is not registered")
    return op


@pytest.fixture
def op():
    return _get_op("mnnvl_multimem_reduce_scatter")


@pytest.fixture
def make_fa():
    """构造真实的 CustomAllreduce 句柄，测试结束后释放。

    rank_data 必须和 fa 同生命周期存活——构造函数只保存它的裸指针。
    """
    created = []

    def _make(world_size, rank=0):
        init = _get_op("init_custom_ar")
        rank_data = torch.zeros(1, dtype=torch.int32, device="cuda")
        fa = init([_FAKE_SIGNAL_PTR] * world_size, rank_data, rank, False)
        created.append((fa, rank_data))
        return fa

    yield _make
    dispose = getattr(torch.ops._C_custom_ar, "dispose", None)
    if dispose is not None:
        for fa, _ in created:
            dispose(fa)


def _buffers(numel, world_size, dtype=torch.float16):
    """inp(broadcast 输入) / out(每卡分片) / local buffer 都是真实显存。"""
    inp = torch.ones(numel, dtype=dtype, device="cuda")
    out = torch.empty(numel // world_size, dtype=dtype, device="cuda")
    local = torch.empty(numel, dtype=dtype, device="cuda")
    return inp, out, local


def test_mnnvl_multimem_reduce_scatter_op_registered(op):
    assert callable(op)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_mnnvl_multimem_reduce_scatter_dtype_mismatch(op, make_fa, dtype):
    fa = make_fa(world_size=2)
    inp = torch.empty(16, dtype=dtype, device="cuda")
    wrong = torch.float32 if dtype != torch.float32 else torch.float16
    out = torch.empty(8, dtype=wrong, device="cuda")
    with pytest.raises(RuntimeError, match="scalar_type"):
        op(fa, inp, out, 0, 0, 1024, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mnnvl_multimem_reduce_scatter_rejects_numel_mismatch(op, make_fa):
    fa = make_fa(world_size=2)
    inp, _, local = _buffers(16, 2)
    out = torch.empty(4, dtype=torch.float16, device="cuda")  # 4 * 2 != 16
    with pytest.raises(RuntimeError, match="world_size_"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(),
           inp.nbytes, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mnnvl_multimem_reduce_scatter_rejects_empty_input(op, make_fa):
    fa = make_fa(world_size=2)
    inp = torch.empty(0, dtype=torch.float16, device="cuda")
    out = torch.empty(0, dtype=torch.float16, device="cuda")
    local = torch.empty(16, dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="> 0"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(), 1024, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mnnvl_multimem_reduce_scatter_rejects_non_contiguous_output(op, make_fa):
    fa = make_fa(world_size=2)
    inp, _, local = _buffers(32, 2)
    out = torch.empty(32, dtype=torch.float16, device="cuda")[::2]  # numel 16
    assert out.numel() * 2 == inp.numel() and not out.is_contiguous()
    with pytest.raises(RuntimeError, match="is_weak_contiguous"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(),
           inp.nbytes, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mnnvl_multimem_reduce_scatter_rejects_small_stage(op, make_fa):
    fa = make_fa(world_size=2)
    inp, out, local = _buffers(16, 2)  # 32 字节输入, stage 只给 31
    with pytest.raises(RuntimeError, match="stage_sz_bytes"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(),
           inp.nbytes - 1, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("null_local", [True, False],
                         ids=("null_local_buffer", "null_multicast_buffer"))
def test_mnnvl_multimem_reduce_scatter_rejects_null_buffers(
        op, make_fa, null_local):
    fa = make_fa(world_size=2)
    inp, out, local = _buffers(32, 2)
    local_ptr = 0 if null_local else local.data_ptr()
    multicast_ptr = local.data_ptr() if null_local else 0
    with pytest.raises(RuntimeError,
                       match="local_buffer" if null_local else "multicast_buffer"):
        op(fa, inp, out, local_ptr, multicast_ptr, inp.nbytes, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mnnvl_multimem_reduce_scatter_rejects_unaligned_size(op, make_fa):
    fa = make_fa(world_size=2)
    inp, out, local = _buffers(8, 2)  # 16 字节, 不是 16 * world_size 的倍数
    with pytest.raises(RuntimeError, match="multiple of 16"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(),
           inp.nbytes, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("block_limit", [0, 9], ids=("zero", "too_large"))
def test_mnnvl_multimem_reduce_scatter_rejects_bad_block_limit(
        op, make_fa, block_limit):
    fa = make_fa(world_size=2)
    inp, out, local = _buffers(32, 2)  # 64 字节, 满足 16 * world_size 对齐
    with pytest.raises(RuntimeError, match="block limit"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(),
           inp.nbytes, block_limit)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mnnvl_multimem_reduce_scatter_rejects_unsupported_dtype(op, make_fa):
    fa = make_fa(world_size=2)
    inp, out, local = _buffers(32, 2, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="only supports float32"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(),
           inp.nbytes, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_mnnvl_multimem_reduce_scatter_supported_dtypes(op, make_fa, dtype):
    """三种 dtype 都要走到成员函数的 world_size 分派。

    用 world_size=6 触发 "(2,4,8)" 分支错误：这块检查在 launch 之前，能证明
    dtype 分支已被接受（若 dtype 不被支持会先抛 "only supports float32"）。
    """
    fa = make_fa(world_size=6)
    inp, out, local = _buffers(96, 6, dtype=dtype)  # 字节数需为 16 * 6 的倍数
    with pytest.raises(RuntimeError, match="num gpus"):
        op(fa, inp, out, local.data_ptr(), local.data_ptr(),
           inp.nbytes, 1)
