import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx import cusparse
import matplotlib.pyplot as plt
import numpy as np

matrix_size = [256, 512, 1024]
density = [0.01, 0.1, 0.5]

cp.random.seed(10)
low, high = 0, 1

def uniform_in_range(n):
    return cp.random.uniform(low, high, size=n).astype(cp.float32)

def main():
    error_matrix = np.zeros((len(matrix_size), len(density)))

    for i, dim in enumerate(matrix_size):
        for j, D in enumerate(density):
            A = sp.random(dim, dim, density=D, format="csr",
                          data_rvs=uniform_in_range, dtype=cp.float32)
            B = sp.random(dim, dim, density=D, format="csr",
                          data_rvs=uniform_in_range, dtype=cp.float32)
            C = cusparse.spgemm(A, B, alg=1)
            D = cusparse.spgemm(A, B, alg=3, chunk_fraction=0.3)
            diff = cp.abs(C.toarray() - D.toarray())
            error = float(diff.max())
            error_matrix[i, j] = error

    # plot heapmap
    plt.figure(figsize=(8, 6))
    plt.imshow(error_matrix, origin='lower', aspect='auto')
    plt.colorbar(label="Max error")
    plt.xticks(range(len(density)), density)
    plt.yticks(range(len(matrix_size)), matrix_size)
    plt.xlabel("Density")
    plt.ylabel("Matrix size")
    plt.title("SpGEMM max error heatmap (alg1 vs alg3) chunk_fraction: 0.3")

    #Add text values to each cell
    for i in range(len(matrix_size)):
        for j in range(len(density)):
            value = error_matrix[i, j]
            plt.text(j, i, f"{value:.2e}", ha='center', va='center', color='black')

    plt.savefig("spgemm_error_heapmap.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()