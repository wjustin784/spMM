#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CuPy sparse/dense benchmark:
- Build sparse A (m x n), sparse B (n x p), dense C (n x 1)
- Compute: A@B (SpMM), A@C (SpMV dense vec)
- Profile runtime (s) and memory deltas (bytes).
"""

import argparse
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional
from collections import defaultdict

import cupy as cp
import cupyx
from cupyx.scipy import sparse as cpx_sparse


def print_best_combos(results):
    """
    Scan BenchResult entries from this density run and print the best combos:
      - For each mode in {hot, cold}:
        * SpMM: best among all A_{CSR,CSC,COO} @ B_{CSR,CSC,COO}
        * SpMV: best among A_{CSR,CSC,COO} @ C (dense vec)
    """
    import re
    spmm_pat = re.compile(r'^(hot|cold)_A_(CSR|CSC|COO)\s*@\s*B_(CSR|CSC|COO)\s*\(SpMM\)$')
    spmv_pat = re.compile(r'^(hot|cold)_A_(CSR|CSC|COO)\s*@\s*C\s*\(SpMV,\s*dense\s*vec\)$')

    # Collect by op+mode
    best = {
        ("hot","spmm"): None,
        ("cold","spmm"): None,
        ("hot","spmv"): None,
        ("cold","spmv"): None,
    }

    for r in results:
        if not r or r == '':
            continue
        name = r.name.strip()
        m = spmm_pat.match(name)
        if m:
            mode = m.group(1)
            key = (mode, "spmm")
            if best[key] is None or r.time_s < best[key].time_s:
                best[key] = r
            continue
        m = spmv_pat.match(name)
        if m:
            mode = m.group(1)
            key = (mode, "spmv")
            if best[key] is None or r.time_s < best[key].time_s:
                best[key] = r
            continue

    print("\n=== Best combos (by time) ===")
    for (mode, op) in [("hot","spmm"), ("hot","spmv"), ("cold","spmm"), ("cold","spmv")]:
        r = best[(mode, op)]
        if r is not None:
            print(f"{r.name} is the best combo  (time={r.time_s:.6f}s)")
    print()


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


def synchronize():
    cp.cuda.Stream.null.synchronize()


@dataclass
class BenchResult:
    name: str
    time_s: float
    dev_mem_used_delta: int
    pool_total_bytes_delta: int
    out_shape: Optional[tuple]
    out_dtype: Optional[str]


def profile_op(name: str, func: Callable[[], Any]) -> BenchResult:
    """
    Profile a single GPU op:
    - Synchronize before/after
    - time.perf_counter() wall time
    - Device free memory delta (memGetInfo)
    - Memory pool growth delta as a crude peak/footprint proxy
    """
    device = cp.cuda.Device()
    pool = cp.get_default_memory_pool()

    synchronize()
    free0, total0 = cp.cuda.runtime.memGetInfo()
    pool0 = pool.total_bytes()

    t0 = time.perf_counter()
    out = func()
    synchronize()
    t1 = time.perf_counter()

    free1, total1 = cp.cuda.runtime.memGetInfo()
    pool1 = pool.total_bytes()

    # Positive delta means more device memory now in use than before the op.
    dev_mem_used_delta = int(free0 - free1)
    pool_total_bytes_delta = int(pool1 - pool0)  # pool growth (blocks kept)

    out_shape = getattr(out, "shape", None)
    out_dtype = str(getattr(out, "dtype", "")) if hasattr(out, "dtype") else None

    return BenchResult(
        name=name,
        time_s=t1 - t0,
        dev_mem_used_delta=dev_mem_used_delta,
        pool_total_bytes_delta=pool_total_bytes_delta,
        out_shape=out_shape,
        out_dtype=out_dtype,
    )


def make_sparse_matrix(m: int, n: int, density: float, fmt, dtype=cp.float32, seed: int = 0) -> cpx_sparse.csr_matrix:
    rs = cp.random.RandomState(seed)
    # Generate on GPU directly; include build time in profile when called via profile_op
    M = cpx_sparse.random(m, n, density=density, format=fmt, dtype=dtype, random_state=rs)
    # (optional) eliminate zeros (rare for random, but keeps CSR tidy)
    M.eliminate_zeros()
    return M


def make_dense_vector(n: int, dtype=cp.float32, seed: int = 3) -> cp.ndarray:
    rs = cp.random.RandomState(seed)
    return rs.standard_normal((n, 1), dtype=dtype)


def warmup():
    # Run a tiny op to JIT-initialize backend kernels
    x = cp.ones((8, 8), dtype=cp.float32)
    y = x @ x
    del x, y
    synchronize()


def main():
    parser = argparse.ArgumentParser(description="CuPy sparse/dense benchmark (A,B,C,D)")
    parser.add_argument("--m", type=int, default=1000, help="Rows of A")
    parser.add_argument("--n", type=int, default=1000, help="Cols of A / rows of B, C")
    parser.add_argument("--p", type=int, default=1000, help="Cols of B")
    parser.add_argument("--densityA", type=float, default=1e-4, help="Density for sparse A")
    parser.add_argument("--densityB", type=float, default=1e-4, help="Density for sparse B")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    parser.add_argument("--runs", type=int, default=10, help="Number of timed runs (after one cold run)")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup call")
    args = parser.parse_args()

    dtype_map = {"float16": cp.float16, "float32": cp.float32, "float64": cp.float64}
    dtype = dtype_map[args.dtype]

    print("=== Config (GPU/cupy) ===")
    print(f"Device       : {cp.cuda.runtime.getDevice()}")
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    print(f"GPU Name     : {props['name'].decode()}")
    print(f"A shape      : ({args.m}, {args.n}) CSR density={args.densityA}")
    print(f"B shape      : ({args.n}, {args.p}) CSR density={args.densityB}")
    print(f"C shape      : ({args.n}, 1) dense")
    print(f"dtype        : {args.dtype}")
    print(f"runs         : {args.runs}")
    print("================\n")

    if not args.no_warmup:
        warmup()

    # ----- Build data (profiled) -----
    results = []
    A = defaultdict()
    B = defaultdict()
    for fmt in ["csr", "csc", "coo"]:
        # results.append(profile_op(f"build_A_sparse_{fmt}", lambda: make_sparse_matrix(args.m, args.n, args.densityA, dtype=dtype, seed=args.seed, fmt=fmt)))
        # results.append(profile_op(f"build_B_sparse_{fmt}", lambda: make_sparse_matrix(args.n, args.p, args.densityB, dtype=dtype, seed=args.seed, fmt=fmt)))
        A[fmt] = make_sparse_matrix(args.m, args.n, args.densityA, fmt, dtype=dtype, seed=args.seed)
        B[fmt] = make_sparse_matrix(args.p, args.p, args.densityB, fmt, dtype=dtype, seed=args.seed)

    # results.append(profile_op("build_C_dense_vec",
    #                           lambda: make_dense_vector(args.n, dtype=dtype, seed=args.seed + 2)))
    C = make_dense_vector(args.n, dtype=dtype, seed=args.seed + 2)
    results.append("")

    # ----- Cold runs (first-call latency) -----
    for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
        for B_name, B_fmt in [("CSR", B["csr"]), ("CSC", B["csc"]), ("COO", B["coo"])]:
            results.append(profile_op(f"cold_A_{A_name} @ B_{B_name} (SpMM)", lambda: A_fmt @ B_fmt))

    for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
        results.append(profile_op(f"cold_A_{A_name} @ C (SpMV, dense vec)", lambda: A_fmt @ C))
    results.append("")


    # ----- Hot runs (steady-state) -----
    def repeat(name: str, fn: Callable[[], Any], runs: int):
        times = []
        mem_deltas = []
        pool_deltas = []
        last_out = None
        for _ in range(runs):
            r = profile_op(name, fn)
            times.append(r.time_s)
            mem_deltas.append(r.dev_mem_used_delta)
            pool_deltas.append(r.pool_total_bytes_delta)
            last_out = r
        # Report median to reduce noise
        import numpy as np
        median_idx = int(np.argsort(cp.asarray(times).get())[runs // 2])
        # Compose a synthetic result using medians
        rsum = BenchResult(
            name=f"hot_{name}",
            time_s=float(cp.asarray(times).get().tolist()[median_idx]),
            dev_mem_used_delta=int(cp.asarray(mem_deltas).get().tolist()[median_idx]),
            pool_total_bytes_delta=int(cp.asarray(pool_deltas).get().tolist()[median_idx]),
            out_shape=last_out.out_shape,
            out_dtype=last_out.out_dtype,
        )
        return rsum


    for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
        for B_name, B_fmt in [("CSR", B["csr"]), ("CSC", B["csc"]), ("COO", B["coo"])]:
            results.append(repeat(f"A_{A_name} @ B_{B_name} (SpMM)", lambda: A_fmt @ B_fmt, args.runs))

    for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
        results.append(repeat(f"A_{A_name} @ C (SpMV, dense vec)", lambda: A_fmt @ C, args.runs))


    # ----- Print summary -----
    print("\n=== Results ===")
    header = f"{'name':36}  {'time(s)':>10}  {'Δdev_mem':>12}  {'Δpool':>12}  {'out_shape':>16}  {'dtype':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r == '':
            print()
        else:
            print(f"{r.name:36}  {r.time_s:10.6f}  {human_bytes(r.dev_mem_used_delta):>12}  "
                f"{human_bytes(r.pool_total_bytes_delta):>12}  {str(r.out_shape):>16}  {str(r.out_dtype):>10}")
    print_best_combos(results)
    
    # Optional: free pools (not necessary for single-shot runs)
    # cp.get_default_memory_pool().free_all_blocks()
    # cupyx.set_allocator(None)


if __name__ == "__main__":
    main()
