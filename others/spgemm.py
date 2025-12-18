import cupy as cp
import cupyx
from cupyx.profiler import benchmark
import time
import scipy.sparse as sp
import numpy as np
import matplotlib.pyplot as plt
import os

def spmv(A, B):
    return A @ B

DENSITY = [0.3]
SIZE = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

def cpu_spmmbenchmark(A, B, warm_up=10, repeat=10):
    for _ in range(warm_up):
        _ = A @ B
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = A @ B
        t1 = time.perf_counter()
        ts.append((t1 - t0) * 1e3) # ms
    return np.median(ts)

def spgemm_benchmark(mat_size, density):
    A = sp.random(mat_size, mat_size, density=density, format="csr", dtype=np.float32)
    B = sp.random(mat_size, mat_size, density=density, format="csr", dtype=np.float32)
    _ = cupyx.scipy.sparse.csr_matrix(A)
    _ = cupyx.scipy.sparse.csr_matrix(B)
    cpu_time = cpu_spmmbenchmark(A, B)
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    A_gpu = cupyx.scipy.sparse.csr_matrix(A)
    B_gpu = cupyx.scipy.sparse.csr_matrix(B)
    end.record()
    end.synchronize()
    copy_ms = cp.cuda.get_elapsed_time(start, end)
    gpu_ctime = np.mean(benchmark(spmv, (A_gpu, B_gpu), n_warmup=10, n_repeat=10).gpu_times)*1e3
    gpu_e2e = copy_ms + gpu_ctime
    return cpu_time, gpu_ctime, gpu_e2e

def main():
    os.makedirs("imgs", exist_ok=True)
    os.makedirs("imgs/spgemm", exist_ok=True)

    for density in DENSITY:
        cpu_times = []
        gpu_compute = []
        gpu_e2e = []

        for n in SIZE:
            cpu_t, gpu_c_t, gpu_e2e_t = spgemm_benchmark(n, density)
            cpu_times.append(cpu_t)
            gpu_compute.append(gpu_c_t)
            gpu_e2e.append(gpu_e2e_t)

        x = np.arange(len(SIZE))
        width = 0.28

        # 圖1：純計算（compute-only）
        plt.figure(figsize=(7, 5))
        plt.bar(x - width/2, cpu_times, width, label="CPU (compute)")
        plt.bar(x + width/2, gpu_compute, width, label="GPU (compute)")
        plt.xticks(x, SIZE)
        plt.xlabel("Matrix size (n x n)")
        plt.ylabel("Time (ms)")
        plt.title(f"SPGEMM Compute Time (density={density})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"imgs/spgemm/spgemm_compute_time_density_{density}.png")

        # 圖2：end-to-end（build+copy+compute）
        plt.figure(figsize=(7, 5))
        plt.bar(x - width/2, cpu_times, width, label="CPU (compute)")
        plt.bar(x + width/2, gpu_e2e, width, label="GPU (end-to-end: copy+compute)")
        plt.xticks(x, SIZE)
        plt.xlabel("Matrix size (n x n)")
        plt.ylabel("Time (ms)")
        plt.title(f"SPGEMM End-to-End (density={density})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"imgs/spgemm/spgemm_end_to_end_density_{density}.png")

if __name__ == "__main__":
    main()