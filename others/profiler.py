#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CuPy profiler-based benchmark (hot timings via cupyx.profiler.benchmark)

Objects:
  - A: sparse CSR (m x n), controllable density
  - B: sparse CSR (n x p), controllable density
  - C: dense vector (n x 1)

Ops:
  - A @ B  (SpGEMM, sparse @ sparse -> sparse)
  - A @ C  (SpMV,   sparse @ dense  -> dense)

Notes:
  - Uses CuPy's built-in benchmark (auto GPU sync, warmup, repeats).
  - Includes timing for data-build steps (generation) using the same profiler.
  - Default sparse generator is OOM-safe (per-row sampling) to scale to large shapes.
"""

import argparse
from typing import Callable, Tuple

import cupy as cp
import cupyx
import cupyx.scipy.sparse as cpx_sparse
from cupyx import profiler as cxprof


# ------------------------------
# OOM-safe sparse generator
# ------------------------------
def make_sparse_csr_perrow(m: int, n: int, density: float, dtype=cp.float32, seed: int = 0) -> cpx_sparse.csr_matrix:
    """Build a CSR matrix by per-row sampling: memory ~ O(nnz)."""
    if density <= 0.0:
        return cpx_sparse.csr_matrix((m, n), dtype=dtype)
    rs = cp.random.RandomState(seed)

    # Expected nnz per row: Binomial(n, p)
    k_per_row = rs.binomial(n=n, p=density, size=m).astype(cp.int32)
    nnz = int(k_per_row.sum().get())
    if nnz == 0:
        return cpx_sparse.csr_matrix((m, n), dtype=dtype)

    rows = cp.empty(nnz, dtype=cp.int32)
    cols = cp.empty(nnz, dtype=cp.int32)
    data = rs.standard_normal(nnz, dtype=dtype)

    write = 0
    # Chunk rows to reduce Python↔GPU overhead
    chunk = 2048
    for r0 in range(0, m, chunk):
        r1 = min(r0 + chunk, m)
        k_host = k_per_row[r0:r1].get()
        for off, k in enumerate(k_host):
            if k == 0:
                continue
            # Choose k distinct columns in [0, n)
            # For very high densities, consider complement strategy (not needed typically)
            col_idx = rs.choice(n, size=int(k), replace=False)
            rows[write:write + k] = r0 + off
            cols[write:write + k] = col_idx
            write += int(k)

    M = cpx_sparse.coo_matrix((data, (rows, cols)), shape=(m, n))
    M.sum_duplicates()
    M.eliminate_zeros()
    return M.tocsr()


def make_sparse_csr_cupy(m: int, n: int, density: float, dtype=cp.float32, seed: int = 0) -> cpx_sparse.csr_matrix:
    """CuPy's built-in random sparse generator (may allocate large temporaries on huge shapes)."""
    rs = cp.random.RandomState(seed)
    M = cpx_sparse.random(m, n, density=density, format="csr", dtype=dtype, random_state=rs)
    M.eliminate_zeros()
    return M


def make_dense_vec(n: int, dtype=cp.float32, seed: int = 123) -> cp.ndarray:
    rs = cp.random.RandomState(seed)
    return rs.standard_normal((n, 1), dtype=dtype)


# ------------------------------
# Benchmark helpers (CuPy profiler)
# ------------------------------
def bench(label: str, fn: Callable[[], object], n_warmup: int, n_repeat: int) -> Tuple[str, float, float, int]:
    """
    Run cupyx.profiler.benchmark on a 0-arg callable.
    Returns (label, mean_seconds, std_seconds, repeats).
    """
    res = cxprof.benchmark(fn, n_warmup=n_warmup, n_repeat=n_repeat)
    # res.gpu_times is in milliseconds
    mean_s = float(res.gpu_times.mean() / 1e3)
    std_s = float(res.gpu_times.std() / 1e3)
    return (label, mean_s, std_s, n_repeat)


def fmt(label: str, mean_s: float, std_s: float, n: int) -> str:
    return f"{label:30}  {mean_s:10.6f} s  ± {std_s:8.6f} s  (n={n})"


# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="CuPy profiler-based sparse benchmarks")
    ap.add_argument("--m", type=int, default=2000, help="Rows of A")
    ap.add_argument("--n", type=int, default=2000, help="Cols of A / rows of B, C")
    ap.add_argument("--p", type=int, default=2000, help="Cols of B")
    ap.add_argument("--densityA", type=float, default=1e-1, help="Density for sparse A")
    ap.add_argument("--densityB", type=float, default=1e-1, help="Density for sparse B")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "float64"])
    ap.add_argument("--warmup", type=int, default=1, help="Warmup runs for each benchmark")
    ap.add_argument("--repeat", type=int, default=10, help="Repeat runs for each benchmark")
    ap.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    ap.add_argument("--sparse_gen", type=str, default="perrow", choices=["perrow", "cupy"],
                    help="Sparse generator: OOM-safe perrow (default) or CuPy's random()")
    ap.add_argument("--annotate", action="store_true",
                    help="Annotate ops with profiler.time_range for Nsight")
    args = ap.parse_args()

    dtype_map = {"float16": cp.float16, "float32": cp.float32, "float64": cp.float64}
    dtype = dtype_map[args.dtype]

    # Device info
    dev_id = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(dev_id)
    print("=== Config ===")
    print(f"Device       : {dev_id}")
    print(f"GPU Name     : {props['name'].decode()}")
    print(f"A (m x n)    : ({args.m}, {args.n}) CSR, density={args.densityA}")
    print(f"B (n x p)    : ({args.n}, {args.p}) CSR, density={args.densityB}")
    print(f"C (n x 1)    : ({args.n}, 1) dense")
    print(f"dtype        : {args.dtype}")
    print(f"warmup       : {args.warmup}")
    print(f"repeat       : {args.repeat}")
    print(f"sparse_gen   : {args.sparse_gen}")
    print("================\n")

    # Choose generator
    gen_sparse = make_sparse_csr_perrow if args.sparse_gen == "perrow" else make_sparse_csr_cupy

    # --------------------------
    # Benchmark data builds
    # --------------------------
    # A
    label, mean_s, std_s, n = bench(
        "build_A_sparse",
        lambda: gen_sparse(args.m, args.n, args.densityA, dtype=dtype, seed=args.seed),
        args.warmup, args.repeat,
    )
    print(fmt(label, mean_s, std_s, n))
    A = gen_sparse(args.m, args.n, args.densityA, dtype=dtype, seed=args.seed)

    # B
    label, mean_s, std_s, n = bench(
        "build_B_sparse",
        lambda: gen_sparse(args.n, args.p, args.densityB, dtype=dtype, seed=args.seed + 1),
        args.warmup, args.repeat,
    )
    print(fmt(label, mean_s, std_s, n))
    B = gen_sparse(args.n, args.p, args.densityB, dtype=dtype, seed=args.seed + 1)

    # C
    label, mean_s, std_s, n = bench(
        "build_C_dense_vec",
        lambda: make_dense_vec(args.n, dtype=dtype, seed=args.seed + 2),
        args.warmup, args.repeat,
    )
    print(fmt(label, mean_s, std_s, n))
    C = make_dense_vec(args.n, dtype=dtype, seed=args.seed + 2)

    # Sanity
    assert A.shape == (args.m, args.n)
    assert B.shape == (args.n, args.p)
    assert C.shape == (args.n, 1)

    # Optional single-run annotations (for Nsight timeline clarity)
    if args.annotate:
        with cxprof.time_range("A@B (single)", color_id=0):
            _ = A @ B
        with cxprof.time_range("A@C (single)", color_id=1):
            _ = A @ C
        cp.cuda.Stream.null.synchronize()

    # --------------------------
    # Benchmark ops (hot timing)
    # --------------------------
    print("\n=== Benchmarks (CuPy profiler) ===")
    print(fmt(*bench("A@B (SpGEMM)", lambda: A @ B, args.warmup, args.repeat)))
    print(fmt(*bench("A@C (SpMV)",   lambda: A @ C, args.warmup, args.repeat)))


if __name__ == "__main__":
    main()
