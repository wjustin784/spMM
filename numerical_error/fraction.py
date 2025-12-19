import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx import cusparse
import matplotlib.pyplot as plt
import numpy as np

matrix_size = 1024
density = 0.
chunk_fraction = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

cp.random.seed(10)
low, high = 0, 1

def uniform_in_range(n):
    return cp.random.uniform(low, high, size=n).astype(cp.float32)


def main():
    cf_error = []
    for cf in chunk_fraction:
        A = sp.random(matrix_size, matrix_size, density=density, format="csr", data_rvs=uniform_in_range, dtype=cp.float32)
        B = sp.random(matrix_size, matrix_size, density=density, format="csr", data_rvs=uniform_in_range, dtype=cp.float32)
        C = cusparse.spgemm(A, B, alg=1)
        D = cusparse.spgemm(A, B, alg=3, chunk_fraction=cf)
        diff = cp.abs(C.toarray() - D.toarray())
        error = float(diff.max())
        cf_error.append(error)

    plt.figure(figsize=(8, 5))
    x = np.arange(len(chunk_fraction))
    plt.bar(x, cf_error) 
    plt.xticks(x, [str(cf) for cf in chunk_fraction])
    plt.xlabel("chunk_fraction")
    plt.ylabel("Max Error")
    plt.title(f"SpGEMM max error (alg1 vs alg3) | dim={matrix_size}, density={density}")
    outname = f"spgemm_error_bars_dim{matrix_size}_dens{str(density).replace('.','p')}.png"
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()