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
import scipy.sparse as sp

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


def _posix_available() -> bool:
    return os.name == "posix"


def synchronize():
    cp.cuda.Stream.null.synchronize()

@dataclass
class BenchResult:
    name: str
    time_ms: float
    peak_vram: int
    peak_ram: Optional[int]
    out_shape: Optional[tuple]
    out_dtype: Optional[str]


def _rss_bytes() -> Optional[int]:
    if psutil is None:
        return None
    return psutil.Process(os.getpid()).memory_info().rss

def _ru_maxrss_bytes():
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss * (1024 if sys.platform != "darwin" else 1)


def profile_op_cpu(name: str, func: Callable[[], Any], *, use_subproc: bool = True) -> BenchResult:
    """
    CPU-side profiler.
    If use_subproc and POSIX: run in a fresh child process to get clean RSS/peak.
    Else: fall back to in-process measurement.
    """
    data = _profile_in_child(func)
    time_ms = data.get("time_ms", None) or 0.0
    # Prefer peak for reporting (more informative); you can switch to peak_ram if you want steady-state.
    peak_ram = data.get("rss_peak_delta", None)
    out_shape = data.get("out_shape", None)
    out_dtype = data.get("out_dtype", None)
    return BenchResult(
        name=name,
        time_ms=time_ms,
        peak_ram=peak_ram,  # reporting peak as ΔRSS; rename field if you want both
        peak_vram=None,
        out_shape=out_shape,
        out_dtype=out_dtype,
    )


def _profile_in_child(func: Callable[[], Any]):
    """
    Run `func()` in a fresh POSIX child process and return a dict with:
      time_ms, peak_ram, rss_peak_delta, out_shape, out_dtype
    """
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        # --- Child ---
        try:
            os.close(r_fd)
            gc.collect()
            rss0 = _rss_bytes()
            # peak0 = _ru_maxrss_bytes()

            t0 = time.perf_counter()
            out = func()
            t1 = time.perf_counter()

            # Measure after (steady-state) and peak
            # NOTE: do NOT gc.collect() here; we want the child's "after" as-is.
            # rss1 = _rss_bytes()
            peak1 = _ru_maxrss_bytes()

            res = {
                "time_ms": (t1 - t0) * 1000,
                "rss_peak_delta": (peak1 - rss0) if None not in (peak1, rss0) else None,
                "out_shape": getattr(out, "shape", None),
                "out_dtype": str(getattr(out, "dtype", "")) if hasattr(out, "dtype") else None,
            }
            payload = json.dumps(res).encode("utf-8")
            os.write(w_fd, payload)
        except Exception as e:
            try:
                os.write(w_fd, json.dumps({"error": repr(e)}).encode("utf-8"))
            except Exception:
                pass
        finally:
            try:
                os.close(w_fd)
            except Exception:
                pass
            os._exit(0)
    else:
        # --- Parent ---
        os.close(w_fd)
        chunks = []
        while True:
            b = os.read(r_fd, 65536)
            if not b:
                break
            chunks.append(b)
        os.close(r_fd)
        _, status = os.waitpid(pid, 0)
        data = {}
        if chunks:
            try:
                data = json.loads(b"".join(chunks).decode("utf-8"))
            except Exception:
                data = {}
        if "error" in data:
            raise RuntimeError(f"Child error: {data['error']}")
        return data


def _sample_gpu(pool, stop_evt, out_dict, period_s=0.000005):
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


def print_best_combos(results, topk: int = 1):
    """
    從結果中挑出時間最短的 SpGEMM / SpMV 組合（支援有/無 hot_/cold_ 前綴）。
    - 會忽略 build_*, pass_* 等非乘法項目
    - 預設各類別只顯示前1名，可用 topk>1 看更多
    """
    import re

    spgemm_pat = re.compile(
        r'^(?:hot_|cold_)?A_(CSR|CSC|COO)\s*@\s*B_(CSR|CSC|COO)\s*\(SpGEMM\)',
        re.IGNORECASE
    )
    spmv_pat = re.compile(
        r'^(?:hot_|cold_)?A_(CSR|CSC|COO)\s*@\s*C\s*\(SpMV,\s*dense\s*vec\)',
        re.IGNORECASE
    )

    spgemm = []
    spmv = []

    for r in results:
        if not r or r == '' or not isinstance(r.name, str):
            continue
        name = r.name.strip()

        if name.startswith("build_") or name.startswith("pass_"):
            continue

        if spgemm_pat.search(name):
            spgemm.append(r)
        elif spmv_pat.search(name):
            spmv.append(r)

    spgemm.sort(key=lambda x: x.time_ms)
    spmv.sort(key=lambda x: x.time_ms)

    print("\n=== Best combos (by time, ms) ===")
    if spgemm:
        print("[SpGEMM] A @ B")
        for i, r in enumerate(spgemm[:topk], 1):
            print(f"  #{i}: {r.name}   time={r.time_ms:.3f} ms   out={r.out_shape}   dtype={r.out_dtype}")
    else:
        print("[SpGEMM] A @ B - no entries")

    if spmv:
        print("[SpMV]   A @ C (dense vec)")
        for i, r in enumerate(spmv[:topk], 1):
            print(f"  #{i}: {r.name}   time={r.time_ms:.3f} ms   out={r.out_shape}   dtype={r.out_dtype}")
    else:
        print("[SpMV]   A @ C - no entries")

    print()



def make_sparse_matrix(m: int, n: int, density: float, fmt, dtype=np.float32, seed: int = 0) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    # SciPy's sparse.random uses COO generation; specify data_rvs for values
    M = sp.random(m, n, density=density, format=fmt, dtype=dtype,
                  random_state=rng, data_rvs=rng.standard_normal)
    M.eliminate_zeros()
    return M


def make_dense_vector_cpu(n: int, dtype=np.float32, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 1), dtype=dtype)


def make_dense_vector_gpu(n: int, dtype=cp.float32, seed: int = 3) -> cp.ndarray:
    rs = cp.random.RandomState(seed)
    return rs.standard_normal((n, 1), dtype=dtype)

def main():
    parser = argparse.ArgumentParser(description="SciPy sparse CPU benchmark (A,B,C)")
    parser.add_argument("--m", type=int, default=1024, help="Rows of A")
    parser.add_argument("--n", type=int, default=1024, help="Cols of A / rows of B, C")
    parser.add_argument("--p", type=int, default=1024, help="Cols of B")
    parser.add_argument("--densityA", type=float, default=1e-4, help="Density for sparse A")
    parser.add_argument("--densityB", type=float, default=1e-4, help="Density for sparse B")
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

        # print(f"BLAS/LAPACK   : {blas_tag}")
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

        def repeat_cpu(name: str, fn: Callable[[], Any], runs: int):
            times = []
            rss = []
            last = None
            for _ in range(runs):
                r = profile_op_cpu(name, fn)
                times.append(r.time_ms)
                rss.append(0 if r.peak_ram is None else r.peak_ram)
                last = r
            # median (robust to outliers)
            order = np.argsort(times)
            med = order[runs // 2]
            return BenchResult(
                name=f"{name}",
                time_ms=times[med],
                peak_ram=rss[med] if rss else None,
                peak_vram=None,
                out_shape=last.out_shape if last else None,
                out_dtype=last.out_dtype if last else None,
            )
        A = defaultdict()
        B = defaultdict()

        for fmt in ["csr", "csc", "coo"]:
            # results.append(repeat_cpu(f"build_A_sparse_{fmt}", lambda: make_sparse_matrix(args.m, args.n, args.densityA, dtype=dtype, seed=args.seed, fmt=fmt), args.runs))
            # results.append(repeat_cpu(f"build_B_sparse_{fmt}", lambda: make_sparse_matrix(args.n, args.p, args.densityB, dtype=dtype, seed=args.seed, fmt=fmt), args.runs))
            A[fmt] = make_sparse_matrix(args.m, args.n, args.densityA, fmt, dtype=dtype, seed=args.seed)
            B[fmt] = make_sparse_matrix(args.n, args.p, args.densityB, fmt, dtype=dtype, seed=args.seed)

        # results.append(repeat_cpu("build_C_dense_vec", lambda: make_dense_vector_cpu(args.n, dtype=dtype, seed=args.seed + 2), args.runs))
        C = make_dense_vector_cpu(args.n, dtype=dtype, seed=args.seed + 2)
        results.append("")



        for A_name, A_fmt in [("csr", A["csr"]), ("csc", A["csc"]), ("coo", A["coo"])]:
            for B_name, B_fmt in [("csr", B["csr"]), ("csc", B["csc"]), ("coo", B["coo"])]:
                results.append(repeat_cpu(f"A_{A_name} @ B_{B_name} (SpGEMM)", lambda: A_fmt @ B_fmt, args.runs))

        for A_name, A_fmt in [("csr", A["csr"]), ("csc", A["csc"]), ("coo", A["coo"])]:
            results.append(repeat_cpu(f"A_{A_name} @ C (SpMV, dense vec)", lambda: A_fmt @ C, args.runs))

        # ----- Print summary -----
        print("\n=== Results (CPU/SciPy) ===")
        header = f"{'name':36}  {'time(ms)':>10}  {'ΔPeak RAM':>12}  {'out_shape':>16}  {'dtype':>10}"
        print(header)
        print("-" * len(header))
        for r in results:
            if r == '':
                print()
            else:
                print(f"{r.name:36}  {r.time_ms:10.6f}  {human_bytes(r.peak_ram or 0):>12}  "
                    f"{str(r.out_shape):>16}  {str(r.out_dtype):>10}")
        print_best_combos(results)

        print("\n\n","*"*91)
        print("\n\n=== Config (GPU/cupy) ===")
        # print(f"Device       : {cp.cuda.runtime.getDevice()}")
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        print(f"GPU Name     : {props['name'].decode()}")
        print(f"A shape      : ({args.m}, {args.n}) CSR density={args.densityA}")
        print(f"B shape      : ({args.n}, {args.p}) CSR density={args.densityB}")
        print(f"C shape      : ({args.n}, 1) dense")
        print(f"dtype        : {args.dtype}")
        print(f"runs         : {args.runs}")
        print("================\n")

        # ----- Build data (profiled) -----

        def repeat_gpu(name: str, fn: Callable[[], Any], runs: int):
            times = []
            mem_deltas = []
            pool_deltas = []
            last_out = None
            for _ in range(runs):
                r = profile_op_gpu(name, fn)
                times.append(r.time_ms)
                mem_deltas.append(r.peak_vram)
                last_out = r
            # Report median to reduce noise
            import numpy as np
            median_idx = int(np.argsort(cp.asarray(times).get())[runs // 2])
            # Compose a synthetic result using medians
            rsum = BenchResult(
                name=f"{name}",
                time_ms=float(cp.asarray(times).get().tolist()[median_idx]),
                peak_vram=int(cp.asarray(mem_deltas).get().tolist()[median_idx]),
                peak_ram=None,
                out_shape=last_out.out_shape,
                out_dtype=last_out.out_dtype,
            )
            return rsum

        def to_gpu_sparse(fmt: str, M: sp.spmatrix):
            if fmt == "csr":
                return cpx_sparse.csr_matrix(M)
            if fmt == "csc":
                return cpx_sparse.csc_matrix(M)
            if fmt == "coo":
                return cpx_sparse.coo_matrix(M)
            raise ValueError(fmt)

        results = []
        # for fmt in ["csr", "csc", "coo"]:
        #     results.append(repeat_gpu(f"pass_A({fmt}) to GPU", lambda: to_gpu_sparse(fmt, A[fmt]), args.runs))
        #     results.append(repeat_gpu(f"pass_B({fmt}) to GPU", lambda: to_gpu_sparse(fmt, B[fmt]), args.runs))
            # A[fmt] = to_gpu_sparse(fmt, A[fmt])  
            # B[fmt] = to_gpu_sparse(fmt, B[fmt])  

        # results.append(repeat_gpu("pass_C_to_GPU", lambda: cp.asarray(C, dtype=dtype), args.runs))
        # C = cp.asarray(C, dtype=dtype)
        # results.append("")


        def SpGEMM(A_fmt, B_fmt, A_name, B_name):
            A_gpu = to_gpu_sparse(A_name, A_fmt)  
            B_gpu = to_gpu_sparse(B_name, B_fmt)  
            return A_gpu @ B_gpu 

        def SpMV(A_fmt, C, A_name):
            A_gpu = to_gpu_sparse(A_name, A_fmt)  
            C_gpu = cp.asarray(C, dtype=dtype)
            return A_gpu @ C_gpu


        for A_name, A_fmt in [("csr", A["csr"]), ("csc", A["csc"]), ("coo", A["coo"])]:
            for B_name, B_fmt in [("csr", B["csr"]), ("csc", B["csc"]), ("coo", B["coo"])]:
                results.append(repeat_gpu(f"A_{A_name} @ B_{B_name} (SpGEMM)", lambda: SpGEMM(A_fmt, B_fmt, A_name, B_name), args.runs))

        for A_name, A_fmt in [("csr", A["csr"]), ("csc", A["csc"]), ("coo", A["coo"])]:
            results.append(repeat_gpu(f"A_{A_name} @ C (SpMV, dense vec)", lambda: SpMV(A_fmt, C, A_name), args.runs))


        # ----- Print summary -----
        print("\n=== Results ===")
        header = f"{'name':36}  {'time(ms)':>10}  {'ΔPeak VRAM':>12}  {'out_shape':>15}  {'dtype':>10}"
        print(header)
        print("-" * len(header))
        for r in results:
            if r == '':
                print()
            else:
                print(f"{r.name:36}  {r.time_ms:10.6f}  {human_bytes(r.peak_vram):>12}  "
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