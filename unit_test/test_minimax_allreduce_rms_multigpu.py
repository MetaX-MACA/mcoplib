# SPDX-License-Identifier: MIT
"""Multi-GPU test for minimax_allreduce_rms (scalar Lamport kernel).

Port of upstream vLLM tests/kernels/core/test_minimax_reduce_rms.py (official
test covers the _qk variant only): spawn one worker per rank, eager
allreduce+RMSNorm reference vs fused Lamport kernel, assert_close + cosine-sim
floor. Infrastructure lives here (no vllm package):

  - ctypes CUDA-IPC bindings + Lamport workspace (MACA libsymbol_cu /
    libmcruntime), dist is gloo/tcp -- the fused kernel communicates through
    the workspace, dist only carries IPC handle bytes + the reference
    all-reduce.
  - ONE worker group loops over ALL token shapes (process/context init is
    ~10 s). Each shape gets a FRESH workspace; old ones are never closed --
    MACA's cudaIpcCloseMemHandle deadlocks cross-process, process exit
    reclaims them.
  - join timeout so a deadlocked Lamport spin fails instead of hanging.

Run (-s to see per-shape [result] lines):
    CUDA_VISIBLE_DEVICES=4,5 pytest unit_test/test_minimax_allreduce_rms_multigpu.py -s
"""

import array
import ctypes
import glob
import os
import struct
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

JOIN_TIMEOUT_S = 2400  # per rank, whole shape sweep; Lamport spins deadlock

_ALIGN = 1 << 21  # 2 MiB -- CUDA IPC allocation alignment
# CUDA uses 1, but MACA's mcIpcOpenMemHandle takes 0 (vllm_metax cuda_wrapper)
_MC_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 0


class cudaIpcMemHandle_t(ctypes.Structure):
    # passed BY VALUE to cudaIpcOpenMemHandle -- must be a real Structure
    _fields_ = [("internal", ctypes.c_byte * 128)]


# ---------------------------------------------------------------------------
# CUDA runtime bindings (ctypes; MACA cu-bridge cudart shim + libmcruntime)
# ---------------------------------------------------------------------------
def _load_cudart():
    """torch's bundled libcudart first, then the MACA system paths."""
    import ctypes.util

    cands = sorted(glob.glob(os.path.join(
        os.path.dirname(torch.__file__), "lib", "libcudart.so*")), reverse=True)
    cands += [
        ctypes.util.find_library("cudart") or "",
        "/opt/maca/lib/libsymbol_cu.so",
        "/root/cu-bridge/CUDA_DIR/lib64/libcudart.so",
        "libcudart.so",
    ]
    for name in filter(None, cands):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        if hasattr(lib, "cudaIpcGetMemHandle"):
            return lib
    raise RuntimeError(f"no CUDA runtime with cudaIpcGetMemHandle (tried {cands})")


libcudart = _load_cudart()

_p = ctypes.c_void_p
libcudart.cudaMalloc.argtypes = [_p, ctypes.c_size_t]
libcudart.cudaMalloc.restype = ctypes.c_uint
libcudart.cudaFree.argtypes = [_p]
libcudart.cudaFree.restype = ctypes.c_uint
libcudart.cudaMemset.argtypes = [_p, ctypes.c_int, ctypes.c_size_t]
libcudart.cudaMemset.restype = ctypes.c_uint
_memcpy = libcudart.cudaMemcpy
_memcpy.argtypes = [_p, _p, ctypes.c_size_t, ctypes.c_int]
_memcpy.restype = ctypes.c_uint
libcudart.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(cudaIpcMemHandle_t), _p]
libcudart.cudaIpcGetMemHandle.restype = ctypes.c_uint
libcudart.cudaIpcOpenMemHandle.argtypes = [_p, cudaIpcMemHandle_t, ctypes.c_uint]
libcudart.cudaIpcOpenMemHandle.restype = ctypes.c_uint
libcudart.cudaIpcCloseMemHandle.argtypes = [_p]
libcudart.cudaIpcCloseMemHandle.restype = ctypes.c_uint

# fine-grained allocator lives on libmcruntime.so, a separate lib
if not hasattr(libcudart, "mcExtMallocWithFlags"):
    try:
        libcudart.mcExtMallocWithFlags = ctypes.CDLL(
            "/opt/maca/lib/libmcruntime.so").mcExtMallocWithFlags
    except (OSError, AttributeError):
        pass
if hasattr(libcudart, "mcExtMallocWithFlags"):
    libcudart.mcExtMallocWithFlags.argtypes = [_p, ctypes.c_size_t, ctypes.c_uint]
    libcudart.mcExtMallocWithFlags.restype = ctypes.c_uint


def _check(err, what):
    if int(err) != 0:
        raise RuntimeError(f"CUDA runtime error {int(err)} in {what}")


def _mc_alloc(size):
    """2 MiB-aligned allocation; fine-grained shareable on MACA when available."""
    aligned = ((size + _ALIGN - 1) >> 21) << 21
    ptr = ctypes.c_void_p()
    if hasattr(libcudart, "mcExtMallocWithFlags"):
        _check(libcudart.mcExtMallocWithFlags(ctypes.byref(ptr), aligned, 1),
               "mcExtMallocWithFlags")
    else:
        _check(libcudart.cudaMalloc(ctypes.byref(ptr), aligned), "cudaMalloc")
    return ptr.value


# ---------------------------------------------------------------------------
# Lamport workspace (mirrors vllm .../minimax_rms_norm/lamport_workspace.py)
# ---------------------------------------------------------------------------
class IpcBuffer:
    """Device buffer whose IPC handle is exchanged with all ranks.

    Intentionally has no close/free: concurrent cudaIpcCloseMemHandle of each
    other's buffers deadlocks on MACA -- buffers are reclaimed at exit only."""

    def __init__(self, rank, world_size, size, process_group=None):
        self.rank, self.world_size = rank, world_size
        self.peer_ptrs = [0] * world_size
        if size <= 0:
            return
        self.peer_ptrs[rank] = _mc_alloc(size)
        _check(libcudart.cudaMemset(self.peer_ptrs[rank], 0, size), "cudaMemset")

        handle = cudaIpcMemHandle_t()
        _check(libcudart.cudaIpcGetMemHandle(
            ctypes.byref(handle), ctypes.c_void_p(self.peer_ptrs[rank])),
            "cudaIpcGetMemHandle")
        all_handles = [None] * world_size
        dist.all_gather_object(all_handles, bytes(handle), group=process_group)
        for r in range(world_size):
            if r == rank:
                continue
            peer = cudaIpcMemHandle_t.from_buffer_copy(all_handles[r])
            ptr = ctypes.c_void_p()
            _check(libcudart.cudaIpcOpenMemHandle(
                ctypes.byref(ptr), peer, _MC_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
                "cudaIpcOpenMemHandle")
            self.peer_ptrs[r] = ptr.value

    def serialize(self):
        return array.array("Q", b"".join(
            struct.pack("P", p) for p in self.peer_ptrs)).tolist()


def compute_comm_size_for_minimax(max_tokens, world_size):
    """fused_qk=False branch: one float per (rank, token) per slot."""
    return ((world_size * max_tokens * 4 + _ALIGN - 1) >> 21) << 21


class LamportWorkspace:
    def __init__(self, rank, world_size, comm_size, process_group=None):
        assert world_size >= 2 and comm_size > 0
        # -0.0f (0x80000000) is the 'slot empty' Lamport sentinel
        fill = torch.full((3 * comm_size // 4,), -0.0,
                          dtype=torch.float32, device="cuda")
        self._lamport = IpcBuffer(rank, world_size, 3 * comm_size, process_group)
        _check(_memcpy(self._lamport.peer_ptrs[rank], fill.data_ptr(),
                       3 * comm_size, 4),  # kind 4 = Default
               "cudaMemcpy d2d")
        del fill
        # int32[3] = {counter, unused, lamport_flag}; int64[2] = {clear_size, comm_size}
        self._flag_buf = torch.zeros(3, dtype=torch.int32, device="cuda")
        self._layout_buf = torch.tensor([0, comm_size], dtype=torch.int64,
                                        device="cuda")
        n = world_size
        ptrs = [0] * n + [0] * n + self._lamport.serialize()
        ptrs += [self._flag_buf.data_ptr(), self._layout_buf.data_ptr()]
        self.workspace = torch.tensor(ptrs, dtype=torch.int64, device="cuda")


# ---------------------------------------------------------------------------
# Eager reference: RMSNorm variance over the FULL hidden, reduced via dist
# ---------------------------------------------------------------------------
def minimax_rms_tp_eager(x_shard, weight, hidden_full, tp_world, eps):
    orig_dtype = x_shard.dtype
    x = x_shard.to(torch.float32)
    sumsq = x.pow(2).sum(dim=-1, keepdim=True)
    if tp_world > 1:
        cpu = sumsq.cpu()
        dist.all_reduce(cpu)  # in-place on gloo (CPU) group
        sumsq = cpu.to(x_shard.device)
    var = sumsq / hidden_full
    y = x * torch.rsqrt(var + eps) * weight.to(torch.float32)
    return y.to(orig_dtype)


def _cosine_similarity(a, b):
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    denom = (a.norm() * b.norm()).clamp_min(torch.finfo(torch.float32).eps)
    return float(torch.dot(a, b) / denom)


# tighter than the 3e-2 assert_close (catches e.g. a dropped float4 column),
# ~1e-5 headroom for the kernel's different reduction order
_SIM_THRESHOLD = 0.99999

_DTYPE_NAME = {
    torch.bfloat16: "bf16", torch.float16: "fp16", torch.float32: "fp32",
}


def _free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Worker: one process per rank, loops over ALL shapes
# ---------------------------------------------------------------------------
def _worker_forward(local_rank, world_size, port, shapes, seed, eps):
    import mcoplib._C  # noqa: F401  (registers torch.ops._C; without this
    # the hasattr guard below silently returns and the test passes vacuously)

    if not hasattr(torch.ops._C, "minimax_allreduce_rms"):
        return

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=local_rank,
        world_size=world_size,
    )

    for num_tokens, hidden_full, dtype in shapes:
        h = hidden_full // world_size
        # rank-local shards of one seeded global weight / input (the eager
        # reference reduces the variance of the *global* row)
        torch.manual_seed(seed)
        w_full = torch.randn(hidden_full, dtype=dtype, device="cuda")
        weight = w_full[local_rank * h:(local_rank + 1) * h].clone()
        del w_full
        torch.manual_seed(seed + 1000)
        x_full = torch.randn(num_tokens, hidden_full, dtype=dtype, device="cuda")
        x = x_full[:, local_rank * h:(local_rank + 1) * h].contiguous()
        del x_full

        ref = minimax_rms_tp_eager(x.clone(), weight, hidden_full, world_size, eps)

        # fresh workspace per shape; previous ones stay open by design
        ws = LamportWorkspace(
            local_rank, world_size,
            compute_comm_size_for_minimax(num_tokens, world_size),
        )
        dist.barrier()  # all ranks enter the fused op together

        out = torch.ops._C.minimax_allreduce_rms(
            x.clone(), weight, ws.workspace, local_rank, world_size, eps
        )
        torch.cuda.synchronize()

        sim = _cosine_similarity(out, ref)
        max_err = (out.float() - ref.float()).abs().max().item()
        try:
            torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
            assert sim >= _SIM_THRESHOLD, (
                f"fused-vs-eager cosine similarity {sim:.6f} "
                f"< {_SIM_THRESHOLD}"
            )
        except AssertionError as e:
            if local_rank == 0:
                print(f"[result] ws={world_size} T={num_tokens:>5} "
                      f"{_DTYPE_NAME.get(dtype, dtype)} h={hidden_full} | "
                      f"FAIL | cos {sim:.6f} max_err {max_err:.2e}",
                      flush=True)
                print(f"[fail] {e}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)

        # light timing: us/call + effective BW per shape (read+write traffic)
        iters = int(os.environ.get("MINIMAX_QUICK_ITERS", "100"))
        for _ in range(10):  # warmup
            torch.ops._C.minimax_allreduce_rms(
                x, weight, ws.workspace, local_rank, world_size, eps)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            torch.ops._C.minimax_allreduce_rms(
                x, weight, ws.workspace, local_rank, world_size, eps)
        t1.record()
        torch.cuda.synchronize()
        if local_rank == 0:
            t_us = t0.elapsed_time(t1) / iters * 1e3
            act_bw = (2 * num_tokens * h * x.element_size()
                      / (t_us / 1e6) / 1e9)
            print(f"[result] ws={world_size} T={num_tokens:>5} "
                  f"{_DTYPE_NAME.get(dtype, dtype)} h={hidden_full} | PASS | "
                  f"cos {sim:.6f} max_err {max_err:.2e} | "
                  f"{t_us:9.2f} us/call | {act_bw:7.2f} GB/s",
                  flush=True)

        del out, ref, x, weight, ws
        torch.cuda.empty_cache()

    dist.barrier()
    dist.destroy_process_group()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


_GPU_CLEAN_DONE = False


def _clear_gpu_occupancy():
    """Best-effort: SIGKILL our own leftover workers parked on the target
    GPUs (deadlocked Lamport ranks survive os._exit-style teardown only when
    SIGTERM'd -- they hold device memory and distort the next run). A foreign
    process on a target GPU aborts: its occupancy invalidates the numbers."""
    import re
    import signal
    import subprocess
    import time

    global _GPU_CLEAN_DONE
    if _GPU_CLEAN_DONE or os.environ.get("MINIMAX_NO_CLEAN", "0") == "1":
        return
    _GPU_CLEAN_DONE = True
    targets = {d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d}
    if not targets:
        return
    try:
        out = subprocess.run(["mx-smi"], capture_output=True, text=True,
                             timeout=30).stdout
    except Exception as e:
        print(f"  [clean] mx-smi failed ({e}); skipping")
        return
    pat = re.compile(r"bench_minimax|test_minimax|minimax_allreduce"
                     r"|multiprocessing|pytest", re.IGNORECASE)
    row = re.compile(r"^\|\s*(\d+)\s+(\d+)\s+(\S.*?)\s+(\d+)\s+\|\s*$")
    mine, foreign, in_proc = [], [], False
    for line in out.splitlines():
        if "Process:" in line:
            in_proc = True
            continue
        m = row.match(line) if in_proc else None
        if not m or m.group(1) not in targets:
            continue
        pid = int(m.group(2))
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read()
            cmd = cmd.replace(b"\0", b" ").decode(errors="replace")
            if os.stat(f"/proc/{pid}").st_uid != os.getuid():
                foreign.append((pid, cmd))
            elif pat.search(cmd):
                mine.append(pid)
        except OSError:
            pass  # raced with exit

    for pid in mine:
        print(f"  [clean] killing leftover pid {pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    if mine:
        time.sleep(1.5)  # driver releases device memory asynchronously
    if foreign:
        for pid, cmd in foreign:
            print(f"  [clean] FOREIGN process on target GPU: pid {pid} {cmd[:80]}")
        raise SystemExit("[clean] target GPUs held by foreign processes; "
                         "clear them or change CUDA_VISIBLE_DEVICES")


def _spawn_with_timeout(fn, args, nprocs):
    """spawn + join timeout: a deadlocked Lamport spin fails the test
    instead of hanging forever."""
    _clear_gpu_occupancy()
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=fn, args=(i,) + tuple(args), daemon=False)
             for i in range(nprocs)]
    for p in procs:
        p.start()
    failed = None
    for p in procs:
        p.join(JOIN_TIMEOUT_S)
        if p.exitcode != 0 and failed is None:
            failed = p.exitcode
    alive = [i for i, p in enumerate(procs) if p.is_alive()]
    if alive:
        for i in alive:
            print(f"[rank {i}] still running after {JOIN_TIMEOUT_S}s -- "
                  "Lamport spin deadlock? terminating", file=sys.stderr)
            procs[i].terminate()
            procs[i].join(10)
        pytest.fail(f"ranks {alive} timed out (Lamport spin deadlock?)")
    if failed is not None:
        pytest.fail(f"a worker rank failed with exit code {failed}")


# bench sweep list plus the official [1, 7, 128, 333, 2049] (union, sorted)
_TOKENS = [
    1, 2, 7, 20, 32, 64, 128, 185, 256, 333, 512,
    1024, 2048, 2049, 3072, 4096, 8192, 12288, 16384, 32768, 65536,
]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA (MACA) required"
)
@pytest.mark.parametrize("world_size", [2, 4], ids=("ws2", "ws4"))
@pytest.mark.parametrize(
    "dtype",
    [torch.bfloat16, torch.float16, torch.float32],
    ids=("bf16", "fp16", "fp32"),
)
@pytest.mark.parametrize("eps", [1e-6])
@pytest.mark.parametrize("seed", [42])
def test_minimax_reduce_rms(world_size, dtype, eps, seed):
    """Lamport fused kernel vs eager allreduce+RMSNorm reference.

    One spawned worker group sweeps all token shapes; a FAIL [result] line
    names the offending shape."""
    num_gpus = torch.cuda.device_count()
    if num_gpus < world_size:
        pytest.skip(f"Need >= {world_size} GPUs, have {num_gpus}")
    shapes = [(t, 6144, dtype) for t in _TOKENS]
    _spawn_with_timeout(
        _worker_forward,
        (world_size, _free_port(), shapes, seed, eps),
        nprocs=world_size,
    )
