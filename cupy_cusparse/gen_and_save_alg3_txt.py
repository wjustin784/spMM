#!/usr/bin/env python3
import os
import argparse
import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx import cusparse
import numpy as np

def save_csr_txt(prefix: str, csr):
    indptr  = cp.asnumpy(csr.indptr.astype(cp.int32, copy=False))
    indices = cp.asnumpy(csr.indices.astype(cp.int32, copy=False))
    data    = cp.asnumpy(csr.data.astype(cp.float32, copy=False))
    np.savetxt(prefix + "_indptr.txt",  indptr,  fmt="%d")
    np.savetxt(prefix + "_indices.txt", indices, fmt="%d")
    np.savetxt(prefix + "_data.txt",    data,    fmt="%.9g")

def gen_rand_csr(n, d, seed):
    rs = cp.random.RandomState(seed)
    return sp.random(n, n, density=d, format="csr",
                     dtype=cp.float32, random_state=rs)

def run_once(n, d, outdir, chunk_fraction=0.2, seed=0, include_cf_in_tag=False):
    os.makedirs(outdir, exist_ok=True)
    A = gen_rand_csr(n, d, seed);   A.sort_indices()
    B = gen_rand_csr(n, d, seed+1); B.sort_indices()

    # ALG3
    C = cusparse.spgemm(A, B, alg=3, chunk_fraction=chunk_fraction)

    cf_tag = f"_cf{str(chunk_fraction).replace('.','p')}" if include_cf_in_tag else ""
    tag = f"n{n}_dens{str(d).replace('.','p')}_alg3{cf_tag}"

    save_csr_txt(os.path.join(outdir, f"A_{tag}"), A)
    save_csr_txt(os.path.join(outdir, f"B_{tag}"), B)
    save_csr_txt(os.path.join(outdir, f"C_py_{tag}"), C)
    print(f"[PY] saved A/B/C txt to {outdir} ({tag}, chunk_fraction={chunk_fraction})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[32,64,128,256,512,1024])
    ap.add_argument("--densities", nargs="+", type=float, default=[0.01,0.1,0.3,0.5])
    ap.add_argument("--outdir", default="dump_alg3_txt")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--chunk-fraction", type=float, default=0.2,
                    help="ALG3 chunk_fraction in (0,1], default=0.2")
    ap.add_argument("--include-cf-in-tag", action="store_true",
                    help="write chunk_fraction to report（例如 _cf0p2）")
    args = ap.parse_args()

    if not (0.0 < args.chunk_fraction <= 1.0):
        raise SystemExit(f"chunk_fraction must be in (0,1], got {args.chunk_fraction}")

    for n in args.sizes:
        for d in args.densities:
            run_once(n, d, args.outdir,
                     chunk_fraction=args.chunk_fraction,
                     seed=args.seed,
                     include_cf_in_tag=args.include_cf_in_tag)
