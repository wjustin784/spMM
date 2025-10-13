#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SciPy sparse CPU benchmark (mirrors the CuPy script):

Objects:
  - A: sparse CSR (m x n), controllable density
  - B: sparse CSR (n x p), controllable density
  - C: dense vector (n x 1)

Ops:
  - A @ B  (SpGEMM: sparse @ sparse -> sparse)
  - A @ C  (SpMV:   sparse @ dense  -> dense)

Reports:
  - Build costs (time + RSS delta)
  - Cold run (first call) for each op
  - Hot runs: median of N repeats
  - Memory deltas via process RSS (psutil)

Notes:
  - For fair CPU baselines, you may want to fix thread counts (e.g. --threads 1 or 8).
"""

import argparse
import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from collections import defaultdict

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



import numpy as np
import scipy.sparse as sp

# Optional dependencies for nicer CPU control / memory
try:
    import psutil
except Exception:
    psutil = None

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


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


@dataclass
class BenchResult:
    name: str
    time_s: float
    rss_delta: Optional[int]
    out_shape: Optional[tuple]
    out_dtype: Optional[str]


def _rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss


def profile_op(name: str, func: Callable[[], Any]) -> BenchResult:
    """
    CPU-side profiler:
    - GC + RSS before/after (rough memory delta)
    - time.perf_counter() wall time
    """
    gc.collect()
    rss0 = _rss_bytes()

    t0 = time.perf_counter()
    out = func()
    t1 = time.perf_counter()

    gc.collect()
    rss1 = _rss_bytes()
    rss_delta = (rss1 - rss0) if (rss0 is not None and rss1 is not None) else None

    out_shape = getattr(out, "shape", None)
    out_dtype = str(getattr(out, "dtype", "")) if hasattr(out, "dtype") else None

    return BenchResult(
        name=name,
        time_s=t1 - t0,
        rss_delta=rss_delta,
        out_shape=out_shape,
        out_dtype=out_dtype,
    )


def make_sparse_matrix(m: int, n: int, density: float, fmt, dtype=np.float32, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    # SciPy's sparse.random uses COO generation; specify data_rvs for values
    M = sp.random(m, n, density=density, format=fmt, dtype=dtype,
                  random_state=rng, data_rvs=rng.standard_normal)
    M.eliminate_zeros()
    return M


def make_dense_vector(n: int, dtype=np.float32, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 1), dtype=dtype)


def main():
    parser = argparse.ArgumentParser(description="SciPy sparse CPU benchmark (A,B,C)")
    parser.add_argument("--m", type=int, default=1000, help="Rows of A")
    parser.add_argument("--n", type=int, default=1000, help="Cols of A / rows of B, C")
    parser.add_argument("--p", type=int, default=1000, help="Cols of B")
    parser.add_argument("--densityA", type=float, default=1e-4, help="Density for sparse A")
    parser.add_argument("--densityB", type=float, default=1e-4, help="Density for sparse B")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float16", "float32", "float64"], help="Element dtype")
    parser.add_argument("--runs", type=int, default=10, help="Hot-run repeats")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--threads", type=int, default=0,
                        help="Limit BLAS/OpenMP threads (0 = no limit).")
    args = parser.parse_args()

    dtype_map = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
    dtype = dtype_map[args.dtype]

    # Optional: limit threads for reproducibility/fairness
    # If threadpoolctl is available, we can scope the entire run.
    if args.threads > 0 and threadpool_limits is not None:
        tp_ctx = threadpool_limits(limits=args.threads)
    else:
        tp_ctx = None

    def run_all():
        results = []

        print("=== Config (CPU/SciPy) ===")
        # Basic BLAS info (best-effort)
        try:
            import numpy.__config__ as npconf  # noqa
            info = []
            for key in ["openblas_info", "blas_opt_info", "lapack_opt_info", "mkl_info"]:
                d = npconf.get_info(key)
                if d:
                    info.append(key)
            blas_tag = ",".join(info) if info else "unknown"
        except Exception:
            blas_tag = "unknown"

        print(f"BLAS/LAPACK   : {blas_tag}")
        print(f"A shape       : ({args.m}, {args.n}) CSR density={args.densityA}")
        print(f"B shape       : ({args.n}, {args.p}) CSR density={args.densityB}")
        print(f"C shape       : ({args.n}, 1) dense")
        print(f"dtype         : {args.dtype}")
        print(f"runs          : {args.runs}")
        if args.threads > 0:
            print(f"threads       : {args.threads}")
        else:
            print(f"threads       : (no limit)")
        print("==========================\n")

        # ----- Build data (profiled) -----
        A = defaultdict()
        B = defaultdict()

        for fmt in ["csr", "csc", "coo"]:
            # results.append(profile_op(f"build_A_sparse_{fmt}", lambda: make_sparse_matrix(args.m, args.n, args.densityA, dtype=dtype, seed=args.seed, fmt=fmt)))
            # results.append(profile_op(f"build_B_sparse_{fmt}", lambda: make_sparse_matrix(args.n, args.p, args.densityB, dtype=dtype, seed=args.seed, fmt=fmt)))
            A[fmt] = make_sparse_matrix(args.m, args.n, args.densityA, fmt, dtype=dtype, seed=args.seed)
            B[fmt] = make_sparse_matrix(args.p, args.p, args.densityB, fmt, dtype=dtype, seed=args.seed)

        # results.append(profile_op("build_C_dense_vec",
        #                         lambda: make_dense_vector(args.n, dtype=dtype, seed=args.seed + 2)))
        C = make_dense_vector(args.n, dtype=dtype, seed=args.seed + 2)
        results.append("")

        # ----- Cold runs -----
        for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
            for B_name, B_fmt in [("CSR", B["csr"]), ("CSC", B["csc"]), ("COO", B["coo"])]:
                results.append(profile_op(f"cold_A_{A_name} @ B_{B_name} (SpMM)", lambda: A_fmt @ B_fmt))

        for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
            results.append(profile_op(f"cold_A_{A_name} @ C (SpMV, dense vec)", lambda: A_fmt @ C))
        results.append("")

        # ----- Hot runs (median of N) -----
        def repeat(name: str, fn: Callable[[], Any], runs: int):
            times = []
            rss = []
            last = None
            for _ in range(runs):
                r = profile_op(name, fn)
                times.append(r.time_s)
                rss.append(0 if r.rss_delta is None else r.rss_delta)
                last = r
            # median (robust to outliers)
            order = np.argsort(times)
            med = order[runs // 2]
            return BenchResult(
                name=f"hot_{name}",
                time_s=times[med],
                rss_delta=rss[med] if rss else None,
                out_shape=last.out_shape if last else None,
                out_dtype=last.out_dtype if last else None,
            )

        for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
            for B_name, B_fmt in [("CSR", B["csr"]), ("CSC", B["csc"]), ("COO", B["coo"])]:
                results.append(repeat(f"A_{A_name} @ B_{B_name} (SpMM)", lambda: A_fmt @ B_fmt, args.runs))

        for A_name, A_fmt in [("CSR", A["csr"]), ("CSC", A["csc"]), ("COO", A["coo"])]:
            results.append(repeat(f"A_{A_name} @ C (SpMV, dense vec)", lambda: A_fmt @ C, args.runs))

        # ----- Print summary -----
        print("\n=== Results (CPU/SciPy) ===")
        header = f"{'name':36}  {'time(s)':>10}  {'ΔRSS':>12}  {'out_shape':>16}  {'dtype':>10}"
        print(header)
        print("-" * len(header))
        for r in results:
            if r == '':
                print()
            else:
                print(f"{r.name:36}  {r.time_s:10.6f}  {human_bytes(r.rss_delta or 0):>12}  "
                    f"{str(r.out_shape):>16}  {str(r.out_dtype):>10}")
        print_best_combos(results)


    if tp_ctx is None:
        # No thread limit or threadpoolctl not installed
        run_all()
    else:
        # Scope the entire benchmark with a fixed number of threads
        with tp_ctx:
            run_all()


if __name__ == "__main__":
    main()
