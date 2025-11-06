import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx import cusparse
import numpy as np
import argparse



# matrix parameter
matrix_size = [32, 64, 128, 256, 512, 1024]
density = [0.01, 0.1, 0.3, 0.5]
end = cp.cuda.Event()
n = 1024
d = 0.2
cp.random.seed(2008)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output file name")
    parser.add_argument("--seed", type=int, default=2008, help="random seed to make deteministic")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="matrix data type")
    args = parser.parse_args()
    d_type = cp.float32 if args.dtype == "float32" else cp.float64
    with open(args.out, "w") as f:
        for mat_size in matrix_size:
            for den in density:
                A = sp.random(mat_size , mat_size, density=den, format="csr", dtype=d_type)
                B = sp.random(mat_size , mat_size, density=den, format="csr", dtype=d_type)
                C = A @ B
                end.record()
                end.synchronize()
                f.write(str(C.nnz) + "\n")
                f.write(str(C.indices) + "\n")
                f.write(str(C.indptr) + "\n")
                f.write(str(C.data) + "\n\n")

if __name__ == "__main__":
    main()

