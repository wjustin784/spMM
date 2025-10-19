import cupy as cp
import cupyx
from cupyx.profiler import benchmark
import time
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import os

# sparse matrix multiply dense vector
def spmv(A, x):
    return A @ x

DENSITY = [0.1, 0.2, 0.3]
SIZE = [8, 16, 32, 64, 128, 256, 512, 1024]

def cpu_spmvbenchmark(A, x, warm_up=10, repeat=1000):
    for _ in range(warm_up):
        _ = A @ x
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = A @ x
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e3) # ms
    return np.median(ts)

def spmv_benchmark(mat_size, density):
    A = sp.random(mat_size, mat_size, density=density, format="csr", dtype=np.float32)
    x = np.random.rand(mat_size).astype(np.float32)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    cpu_time = cpu_spmvbenchmark(A, x)
    A_gpu = cupyx.scipy.sparse.csr_matrix(A)
    x_gpu = cp.asarray(x)
    end.record()
    end.synchronize()
    elapsed_ms = cp.cuda.get_elapsed_time(start, end)
    gpu_ctime = np.mean(benchmark(spmv, (A_gpu, x_gpu), n_warmup=10, n_repeat=10).gpu_times)*1e3
    gpu_bctime = gpu_ctime + elapsed_ms
    return cpu_time, gpu_ctime, gpu_bctime


def main():
    os.makedirs("imgs", exist_ok=True)
    os.makedirs("imgs/spmv", exist_ok=True)
    for density in DENSITY:
        cpu_mul_time = []
        gpu_mul_time = []
        gpu_end_to_end = []
        for size in SIZE:
            cpu, gpu, gpu_e2e= spmv_benchmark(size, density)
            cpu_mul_time.append(cpu)
            gpu_mul_time.append(gpu)
            gpu_end_to_end.append(gpu_e2e)
            # plot result

        x = np.arange(len(SIZE))
        width = 0.28
        plt.figure(figsize=(7, 5))
        plt.bar(x - width/2, cpu_mul_time, width, label="CPU time")
        plt.bar(x + width/2, gpu_mul_time, width, label="GPU time")
        plt.xticks(x, SIZE)
        plt.xlabel("Matrix size (n x n)")
        plt.ylabel("Time (ms)")
        plt.title(f"SPMV Compute time (density={density})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"imgs/spmv_compute_time_density_{density}.png")
            
        # 圖2：含建置/傳輸（end-to-end, build+copy+compute）
        plt.figure(figsize=(7,5))
        plt.bar(x - width/2, cpu_mul_time, width, label="CPU (compute)")
        plt.bar(x + width/2, gpu_end_to_end, width, label="GPU (end-to-end: copy+compute)")
        plt.xticks(x, SIZE)
        plt.xlabel("Matrix size (n x n)")
        plt.ylabel("Time (ms)")
        plt.title(f"SPMV End-to-end (density={density})")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"imgs/spmv_end_to_end_density_{density}.png")

if __name__ == "__main__":
    main()