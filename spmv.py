import cupy as cp
import cupyx
from cupyx.profiler import benchmark
import time
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt

# sparse matrix multiply dense vector
def spmv(A, x):
    return A @ x

DENSITY = [0.1, 0.2, 0.3, 0.4]
SIZE = [8, 16, 32, 64, 128, 256, 512, 1024]

def cpu_spmvbenchmark(A, x, warm_up=10, repeat=10):
    for _ in range(warm_up):
        _ = A @ x
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = A @ x
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e3) # ms
    return np.mean(ts)

def spmv_benchmark(mat_size, density):
    A = sp.random(mat_size, mat_size, density=density, format="csr", dtype=np.float32)
    x = np.random.rand(mat_size).astype(np.float32)
    cpu_time = cpu_spmvbenchmark(A, x)
    A_gpu = cupyx.scipy.sparse.csr_matrix(A)
    x_gpu = cp.asarray(x)
    gpu_time = np.mean(benchmark(spmv, (A_gpu, x_gpu), n_warmup=10, n_repeat=10).gpu_times)*1e3
    return cpu_time, gpu_time

def main():
    cpu_mul_time = []
    gpu_mul_time = []
    for size in SIZE:
        cpu, gpu = spmv_benchmark(size, 0.3)
        cpu_mul_time.append(cpu)
        gpu_mul_time.append(gpu)
        # plot result

    x = np.arange(len(SIZE))
    width = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - width/2, cpu_mul_time, width, label="CPU time")
    plt.bar(x + width/2, gpu_mul_time, width, label="GPU time")
    plt.xticks(x, SIZE)
    plt.xlabel("Matrix size (n x n)")
    plt.ylabel("Time (ms)")
    plt.title(f"SPMV Compute time (density={DENSITY[0]})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("imgs/spmv_compute_time.png")
        

if __name__ == "__main__":
    main()