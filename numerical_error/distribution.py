import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx import cusparse
import matplotlib.pyplot as plt
import numpy as np

matrix_size = 1024
density = 0.1
chunk_fraction = 0.3

cp.random.seed(10)
low, high = 0, 1

def uniform_in_range(n):
    return cp.random.uniform(low, high, size=n).astype(cp.float32)

def main():
    A = sp.random(matrix_size, matrix_size, density=density, format="csr",
                  data_rvs=uniform_in_range, dtype=cp.float32)
    B = sp.random(matrix_size, matrix_size, density=density, format="csr",
                  data_rvs=uniform_in_range, dtype=cp.float32)

    C = cusparse.spgemm(A, B, alg=1).toarray()
    D = cusparse.spgemm(A, B, alg=3, chunk_fraction=chunk_fraction).toarray()
    abs_diff = cp.abs(C - D)
    max_diff = abs_diff.max() 
    print(max_diff)
    diff_np = abs_diff.get().flatten()
    diff_np = diff_np[np.isfinite(diff_np)]
    weights = np.full_like(diff_np, fill_value=100.0 / diff_np.size, dtype=np.float64)

    plt.figure(figsize=(8,5))
    plt.hist(diff_np, bins=100,
             weights=weights,
             edgecolor='black',
             linewidth=0.3)
    plt.xlabel("Absolute Error |C - D|")
    plt.ylabel("Percentage (%)")
    plt.title(f"Error Distribution (% count)\n"
              f"alg1 vs alg3 | dimgs={matrix_size}, density={density}")
    plt.savefig("error_distribution_with_zero.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
