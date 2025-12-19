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

def run_once(n, d, outdir, seed=0):
    os.makedirs(outdir, exist_ok=True)
    A = gen_rand_csr(n, d, seed);   A.sort_indices()
    B = gen_rand_csr(n, d, seed+1); B.sort_indices()

    # ALG1
    C = cusparse.spgemm(A, B, alg=1)

    tag = f"n{n}_dens{str(d).replace('.','p')}_alg1"
    save_csr_txt(os.path.join(outdir, f"A_{tag}"), A)
    save_csr_txt(os.path.join(outdir, f"B_{tag}"), B)
    save_csr_txt(os.path.join(outdir, f"C_py_{tag}"), C)
    print(f"[PY] saved A/B/C txt to {outdir} ({tag})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[32,64,128,256,512,1024])
    ap.add_argument("--densities", nargs="+", type=float, default=[0.01,0.1,0.3,0.5])
    ap.add_argument("--outdir", default="dump_alg1_txt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    for n in args.sizes:
        for d in args.densities:
            run_once(n, d, args.outdir, seed=args.seed)
