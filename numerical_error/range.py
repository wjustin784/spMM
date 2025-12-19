import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx import cusparse
import matplotlib.pyplot as plt
import numpy as np

matrix_size = 1024
density = 0.1
chunk_fraction = 0.3
num_repeat = 300  


high_values = [1, 10, 100, 500, 1000, 5000, 10000]

def uniform_generator(low, high):
    return lambda n: cp.random.uniform(low, high, size=n).astype(cp.float32)

def max_error_for_range(high):

    low = 0
    errors = []

    for _ in range(num_repeat):
        cp.random.seed()  

        data_rvs = uniform_generator(low, high)

        A = sp.random(matrix_size, matrix_size, density=density,
                      format="csr", data_rvs=data_rvs, dtype=cp.float32)
        B = sp.random(matrix_size, matrix_size, density=density,
                      format="csr", data_rvs=data_rvs, dtype=cp.float32)

        C = cusparse.spgemm(A, B, alg=1)
        D = cusparse.spgemm(A, B, alg=3, chunk_fraction=chunk_fraction)

        diff = cp.abs(C.toarray() - D.toarray())
        errors.append(float(diff.max()))

    return max(errors)

def main():
    results = []
    for high in high_values:
        print(f"Testing high={high} ...")
        max_err = max_error_for_range(high)
        results.append(max_err)
        print(f"high={high}, max_err={max_err:.4e}")


    plt.figure(figsize=(8,5))
    x = np.arange(len(high_values))
    plt.bar(x, results, width=0.3)
    plt.xticks(x, [str(h) for h in high_values])
    plt.xlabel(f"High Value (0 ~ {high_values[-1]})")
    plt.ylabel("Max Error")
    plt.title(f"Max SpGEMM Error vs Data Range (dim={matrix_size}, density={density})")

    outname = "spgemm_error_vs_range.png"
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {outname}")

if __name__ == "__main__":
    main()