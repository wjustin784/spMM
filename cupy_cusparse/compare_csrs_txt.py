#!/usr/bin/env python3
import numpy as np
import argparse

def load(prefix):
    indptr  = np.loadtxt(prefix + "_indptr.txt",  dtype=np.int32)
    indices = np.loadtxt(prefix + "_indices.txt", dtype=np.int32)
    data    = np.loadtxt(prefix + "_data.txt",    dtype=np.float32)

    
    if indptr.ndim == 0:  indptr  = indptr.reshape(1)
    if indices.ndim == 0: indices = indices.reshape(1)
    if data.ndim == 0:    data    = data.reshape(1)

    rows = indptr.size - 1
    nnz  = indices.size
    return rows, nnz, indptr, indices, data


def main(py_prefix, cu_prefix):
    r1,n1,p1,i1,d1 = load(py_prefix)
    r2,n2,p2,i2,d2 = load(cu_prefix)

    ok = True

    # rows, nnz
    if r1 != r2 or n1 != n2:
        print(f"rows/nnz mismatch: py=({r1},{n1}) cu=({r2},{n2})")
        ok = False

    # indptr 
    if not np.array_equal(p1, p2):
        print("indptr mismatch")
        ok = False

    # indices 
    if not np.array_equal(i1, i2):
        print("indices mismatch")
        ok = False

    # calculate bitwise identical. 
    if not np.array_equal(d1, d2):
        print("data mismatch")
        ok = False

    print("EQUAL" if ok else "NOT EQUAL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("py_prefix")
    ap.add_argument("cu_prefix")
    args = ap.parse_args()
    raise SystemExit(main(args.py_prefix, args.cu_prefix))
