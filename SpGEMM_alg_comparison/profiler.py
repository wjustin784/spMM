#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CuPy sparse GPU benchmark:

Objects:
  - A: sparse CSR (m x n), controllable density
  - B: sparse CSR (n x p), controllable density

Ops:
  - A @ B  (SpGEMM: sparse @ sparse -> sparse) using cusparse.spgemm

Reports:
  - Median of N repeats
  - Peak VRAM delta during operation

Notes:
  - Based on a script for SciPy, adapted for CuPy.
"""
import cupy.cuda.profiler as profiler

import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.profiler import benchmark
from cupyx import cusparse

import resource
import sys
import argparse
import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from collections import defaultdict

import os, json, time, gc, sys, resource
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp_cpu # Renamed to avoid conflict

# dependencies for nicer CPU control / memory

import psutil

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

import cupy as cp
import cupyx
from cupyx.scipy import sparse as cpx_sparse
import threading, time
import itertools 


@dataclass
class BenchResult:
    name: str
    time_ms: float
    peak_vram: int
    peak_ram: Optional[int]
    out_shape: Optional[tuple]
    out_dtype: Optional[str]


def human_bytes(x: int) -> str:
    if x is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    val = float(x)
    while val >= 1024.0 and i < len(units) - 1:
        val /= 1024.0
        i += 1
    return f"{val:.2f} {units[i]}"


def _sample_gpu(pool, stop_evt, out_dict, period_s=0.0005):
    peak_used = 0
    min_free = None
    while not stop_evt.is_set():
        u = pool.used_bytes()
        if u > peak_used:
            peak_used = u
        free, _ = cp.cuda.runtime.memGetInfo()
        if min_free is None or free < min_free:
            min_free = free
        time.sleep(period_s)  

    u = pool.used_bytes()
    if u > peak_used:
        peak_used = u
    free, _ = cp.cuda.runtime.memGetInfo()
    if min_free is None or free < min_free:
        min_free = free
    out_dict["peak_used"] = peak_used
    out_dict["min_free"]  = min_free


def synchronize():
    cp.cuda.Stream.null.synchronize()


def profile_op_gpu(name, fn):
    synchronize()
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        free0, _ = cp.cuda.runtime.memGetInfo()

        out_stats = {}
        stop_evt = threading.Event()
        th = threading.Thread(target=_sample_gpu, args=(pool, stop_evt, out_stats), daemon=True)
        th.start()

        t0 = time.perf_counter()
        out = fn()
        synchronize()
        t1 = time.perf_counter()

        stop_evt.set(); th.join()

        pool_used_after  = pool.used_bytes()
        pool_total_after = pool.total_bytes()
        free_before_free_all, _ = cp.cuda.runtime.memGetInfo()

        pool.free_all_blocks()
        free_after_free_all, _ = cp.cuda.runtime.memGetInfo()

    live_peak_from_free = int(free0 - out_stats["min_free"]) if "min_free" in out_stats else None


    return BenchResult(
        name=name,
        time_ms=(t1 - t0) * 1000,
        peak_vram=live_peak_from_free, 
        peak_ram=None,
        out_shape=getattr(out, "shape", None),
        out_dtype=str(getattr(out, "dtype", "")) if hasattr(out, "dtype") else None,
    )


def make_sparse_matrix(m: int, n: int, density: float, fmt, dtype=np.float32, seed: int = 0) -> sp_cpu.csr_matrix:
    rng = np.random.default_rng(seed)
    # SciPy's sparse.random uses COO generation; specify data_rvs for values
    M = sp_cpu.random(m, n, density=density, format=fmt, dtype=dtype,
                      random_state=rng, data_rvs=rng.standard_normal)
    M.eliminate_zeros()
    return M


def to_gpu_sparse(fmt: str, M: sp_cpu.spmatrix):
    if fmt == "csr":
        return cpx_sparse.csr_matrix(M)
    if fmt == "csc":
        return cpx_sparse.csc_matrix(M)
    if fmt == "coo":
        return cpx_sparse.coo_matrix(M)
    raise ValueError(fmt)


def run_all(m, n, p, densityA, densityB, dtype, dtype_str, runs, seed):
    """
    執行一組特定參數 (m, n, p, densities) 的基準測試
    """
    results = []
    A = defaultdict()
    B = defaultdict()

    for fmt in ["csr", "csc", "coo"]:
        # A[fmt] = to_gpu_sparse(fmt, make_sparse_matrix(m, n, densityA, fmt, dtype=dtype, seed=seed))
        # B[fmt] = to_gpu_sparse(fmt, make_sparse_matrix(n, p, densityB, fmt, dtype=dtype, seed=seed))
        A[fmt] = make_sparse_matrix(m, n, densityA, fmt, dtype=dtype, seed=seed)
        B[fmt] = make_sparse_matrix(n, p, densityB, fmt, dtype=dtype, seed=seed)


    print("\n\n","*"*91)
    print("\n\n=== Config (GPU/cupy) ===")
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    print(f"GPU Name     : {props['name'].decode()}")
    print(f"A shape      : ({m}, {n}) CSR density={densityA}")
    print(f"B shape      : ({n}, {p}) CSR density={densityB}")
    print(f"dtype        : {dtype_str}")
    print(f"runs         : {runs}")


    def repeat_gpu(name: str, fn: Callable[[], Any], runs: int):
        times = []
        mem_deltas = []
        last_out = None
        for _ in range(runs):
            r = profile_op_gpu(name, fn)
            times.append(r.time_ms)
            mem_deltas.append(r.peak_vram)
            last_out = r
        median_idx = int(np.argsort(cp.asarray(times).get())[runs // 2])
        rsum = BenchResult(
            name=name,
            time_ms=float(cp.asarray(times).get().tolist()[median_idx]),
            peak_vram=int(cp.asarray(mem_deltas).get().tolist()[median_idx]),
            peak_ram=None,
            out_shape=last_out.out_shape,
            out_dtype=last_out.out_dtype,
        )
        return rsum

    def SpGEMM(A_fmt, B_fmt, alg, A_name, B_name):
        A_fmt_gpu = to_gpu_sparse(A_name, A_fmt)
        B_fmt_gpu = to_gpu_sparse(B_name, B_fmt)
        return cusparse.spgemm(A_fmt_gpu, B_fmt_gpu, alg=alg)

    results = []
    algorithms_to_test = [1, 2, 3]

    for A_name, A_fmt in [("csr", A["csr"])]:
        for B_name, B_fmt in [("csr", B["csr"])]:
            for alg in algorithms_to_test:
                op_name = f"A_{A_name} @ B_{B_name} (alg={alg})"
                results.append(repeat_gpu(op_name, lambda: SpGEMM(A_fmt, B_fmt, alg, A_name, B_name), runs))

    print("\n=== Results (alg comparison) ===")
    header = f"{'name':40}  {'time(ms)':>10}  {'ΔPeak VRAM':>12}  {'out_shape':>16}  {'dtype':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:40}  {r.time_ms:10.6f}  {human_bytes(r.peak_vram):>12}  "
              f"{str(r.out_shape):>16}  {str(r.out_dtype):>10}")


def main():
    parser = argparse.ArgumentParser(description="CuPy sparse GPU benchmark (A,B)")
    
    parser.add_argument("--size", type=int, nargs='+', default=[1024], 
                        help="List of matrix dimensions (sets m, n, p simultaneously)")
    parser.add_argument("--density", type=float, nargs='+', default=[1e-1], 
                        help="List of densities (sets densityA, densityB simultaneously)")
    
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32", "float64"], help="Element dtype")
    parser.add_argument("--runs", type=int, default=1, help="Hot-run repeats")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--threads", type=int, default=0,
                        help="Limit BLAS/OpenMP threads (0 = no limit).")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup call")

    args = parser.parse_args()

    dtype_map = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
    dtype = dtype_map[args.dtype]

    if args.threads > 0 and threadpool_limits is not None:
        tp_ctx = threadpool_limits(limits=args.threads)
    else:
        tp_ctx = None


    param_combinations = itertools.product(
        args.size,
        args.density
    )


    for (size, density) in param_combinations:
        
        run_params = {
            "m": size,
            "n": size,
            "p": size,
            "densityA": density,
            "densityB": density,
            "dtype": dtype,
            "dtype_str": args.dtype,
            "runs": args.runs,
            "seed": args.seed
        }

        if tp_ctx is None:
            run_all(**run_params)
        else:
            with tp_ctx:
                run_all(**run_params)


if __name__ == "__main__":
    main()