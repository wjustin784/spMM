#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict, List

import numpy as np
import scipy.sparse as sp_cpu

import cupy as cp
from cupyx.scipy import sparse as cpx_sparse

import threading
import time 
import gc

def cleanup_gpu() -> None:
    """Best-effort cleanup after failures to avoid cascading OOMs."""
    try:
        synchronize()
    except Exception:
        pass
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass
    gc.collect()

# -------------------------------
# Data structures / formatting
# -------------------------------

@dataclass
class BenchResult:
    name: str
    time_ms: float
    peak_vram: Optional[int]
    peak_ram: Optional[int]
    out_shape: Optional[tuple]
    out_dtype: Optional[str]


def human_bytes(x: Optional[int]) -> str:
    if x is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    val = float(x)
    while val >= 1024.0 and i < len(units) - 1:
        val /= 1024.0
        i += 1
    return f"{val:.2f} {units[i]}"


# -------------------------------
# GPU profiling
# -------------------------------

def _sample_gpu(
    pool: cp.cuda.MemoryPool,
    stop_evt: threading.Event,
    out_dict: Dict[str, int],
    period_s: float = 1e-4,
) -> None:
    peak_used = 0
    min_free = None

    while not stop_evt.is_set():
        used = pool.used_bytes()
        if used > peak_used:
            peak_used = used
        free, _ = cp.cuda.runtime.memGetInfo()
        if min_free is None or free < min_free:
            min_free = free
        time.sleep(period_s)

    used = pool.used_bytes()
    if used > peak_used:
        peak_used = used
    free, _ = cp.cuda.runtime.memGetInfo()
    if min_free is None or free < min_free:
        min_free = free

    out_dict["peak_used"] = peak_used
    out_dict["min_free"] = min_free


def synchronize() -> None:
    cp.cuda.Stream.null.synchronize()


def profile_op_gpu(name: str, fn: Callable[[], Any]) -> BenchResult:
    """
    Run `fn` once on GPU under a fresh CuPy memory pool, measure:
      - elapsed time (ms)
      - approximate peak VRAM delta (Δ free memory on the device).

    Important:
      - What exactly ΔPeak Mem means depends on when you allocate inputs
        relative to this function (kernel-only vs end-to-end).
    """
    synchronize()

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        free0, _ = cp.cuda.runtime.memGetInfo()

        out_stats: Dict[str, int] = {}
        stop_evt = threading.Event()
        th = threading.Thread(
            target=_sample_gpu,
            args=(pool, stop_evt, out_stats),
            daemon=True,
        )
        th.start()

        t0 = time.perf_counter()
        out = fn()
        synchronize()
        t1 = time.perf_counter()

        stop_evt.set()
        th.join()

        pool.free_all_blocks()

    peak_vram_delta = (
        int(free0 - out_stats["min_free"]) if "min_free" in out_stats else None
    )

    return BenchResult(
        name=name,
        time_ms=(t1 - t0) * 1000.0,
        peak_vram=peak_vram_delta,
        peak_ram=None,
        out_shape=getattr(out, "shape", None),
        out_dtype=str(getattr(out, "dtype", "")) if hasattr(out, "dtype") else None,
    )


from typing import Optional

def repeat_gpu(
    name: str,
    fn: Callable[[], Any],
    runs: int,
    do_warmup: bool = True,
) -> Optional[BenchResult]:
    """
    Run `fn` on GPU multiple times, return median-by-time BenchResult.
    If OOM happens, return None (skipped).
    """
    # warmup
    if do_warmup:
        try:
            _ = profile_op_gpu(name + " [warmup]", fn)
        except (cp.cuda.memory.OutOfMemoryError, MemoryError, RuntimeError) as e:
            print(f"[SKIP] {name}: warmup failed ({type(e).__name__}: {e})")
            cleanup_gpu()
            return None

    times: List[float] = []
    mem_deltas: List[int] = []
    last_out: Optional[BenchResult] = None

    for i in range(runs):
        try:
            r = profile_op_gpu(name, fn)
        except (cp.cuda.memory.OutOfMemoryError, MemoryError, RuntimeError) as e:
            print(f"[SKIP] {name}: run {i+1}/{runs} failed ({type(e).__name__}: {e})")
            cleanup_gpu()
            return None

        times.append(r.time_ms)
        mem_deltas.append(r.peak_vram if r.peak_vram is not None else 0)
        last_out = r

    times_np = np.asarray(times, dtype=np.float64)
    mem_np = np.asarray(mem_deltas, dtype=np.int64)
    median_idx = int(np.argsort(times_np)[runs // 2])
    assert last_out is not None

    return BenchResult(
        name=name,
        time_ms=float(times_np[median_idx]),
        peak_vram=int(mem_np[median_idx]),
        peak_ram=None,
        out_shape=last_out.out_shape,
        out_dtype=last_out.out_dtype,
    )



# -------------------------------
# Matrix generation / transfer
# -------------------------------

def make_sparse_matrix(
    m: int,
    n: int,
    density: float,
    fmt: str = "csr",
    dtype=np.float32,
):
    """Generate a SciPy sparse matrix on CPU."""
    return sp_cpu.random(m, n, density=density, format=fmt, dtype=dtype)


def to_sparse(M) -> cpx_sparse.csr_matrix:
    """
    Transfer a SciPy sparse matrix to GPU as a CuPy CSR matrix.
    """
    return cpx_sparse.csr_matrix(M)


def to_dense(M) -> cp.ndarray:
    """Transfer a SciPy sparse matrix to GPU as a dense CuPy array."""
    return cp.asarray(M.toarray())


# -------------------------------
# Single-case SpGEMM / GEMM benchmark
# -------------------------------

def run_spmm_case(  # name 保留以免你其他地方有 import；語意其實是 SpGEMM vs GEMM
    m: int,
    n: int,
    p: int,
    density: float,
    dtype,
    dtype_str: str,
    runs: int,
    seed: int,
    do_warmup: bool = True,
) -> None:
    # Make CPU-side sparse matrices (SciPy)
    np.random.seed(seed)
    A = make_sparse_matrix(m, n, density, fmt="csr", dtype=dtype)
    B = make_sparse_matrix(n, p, density, fmt="csr", dtype=dtype)

    # -----------------------------------
    # Kernel-only mode:
    #   Inputs are already on GPU before profiling.
    #   ΔPeak Mem ~= extra workspace + output (excluding inputs).
    # -----------------------------------
    A_sparse = to_sparse(A)
    B_sparse = to_sparse(B)
    A_dense = to_dense(A)
    B_dense = to_dense(B)

    # -----------------------------------
    # End-to-end mode:
    #   Start from CPU SciPy matrices inside fn.
    #   ΔPeak Mem ~= inputs + output + workspace (total delta).
    # -----------------------------------
    def spgemm_end_to_end():
        A_sp = to_sparse(A)
        B_sp = to_sparse(B)
        return A_sp @ B_sp

    def gemm_end_to_end():
        A_dn = to_dense(A)
        B_dn = to_dense(B)
        return A_dn @ B_dn

    print("\n" + "*" * 80)
    print("=== CuPy SpGEMM (CSR @ CSR) vs dense GEMM: A @ B ===")
    print(f"A / B shape (CSR) : A=({m}, {n}), B=({n}, {p}), target_density={density}")
    print(f"actual_density    : {density:.6f}  (SciPy random target)")
    print(f"dtype             : {dtype_str}")
    print(f"runs              : {runs}")
    print()

    op_name = "A @ B"

    # -------- kernel-only --------
    result_sparse_kernel = repeat_gpu(
        name=op_name + " [sparse, inputs_on_gpu]",
        fn=lambda: A_sparse @ B_sparse,
        runs=runs,
        do_warmup=do_warmup,
    )
    result_dense_kernel = repeat_gpu(
        name=op_name + " [dense, inputs_on_gpu]",
        fn=lambda: A_dense @ B_dense,
        runs=runs,
        do_warmup=do_warmup,
    )

    # # -------- end-to-end --------
    # result_sparse_end2end = repeat_gpu(
    #     name=op_name + " [sparse, end2end]",
    #     fn=spgemm_end_to_end,
    #     runs=runs,
    #     do_warmup=do_warmup,
    # )
    # result_dense_end2end = repeat_gpu(
    #     name=op_name + " [dense, end2end]",
    #     fn=gemm_end_to_end,
    #     runs=runs,
    #     do_warmup=do_warmup,
    # )

    # -------- pretty print --------
    header = (
        f"{'name':40}  {'time(ms)':>10}  {'ΔPeak Mem':>16}  "
        f"{'out_shape':>16}  {'dtype':>10}"
    )
    print(header)
    print("-" * len(header))

    def print_result(r: Optional[BenchResult]) -> None:
        if r is None:
            print(f"{'SKIPPED (OOM)':40}  {'-':>10}  {'-':>16}  {'-':>16}  {'-':>10}")
            return
        print(
            f"{r.name:40}  {r.time_ms:10.6f}  "
            f"{human_bytes(r.peak_vram):>16}  "
            f"{str(r.out_shape):>16}  {str(r.out_dtype):>10}"
        )

    for r in [
        result_sparse_kernel,
        result_dense_kernel,
        # result_sparse_end2end,
        # result_dense_end2end,
    ]:
        print_result(r)
