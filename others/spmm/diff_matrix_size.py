import cupy as cp
import cupyx.scipy.sparse as sp
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
matrix_size = [2**11, 2**12, 2**13, 2**14]
density_list = [0.001]
run_time = 1000
warmup = 10

# Results: key=(n,d), value=median ms
spgemm_results = {}
spmm_results = {}

# Peak VRAM results: key=(n,d), value=MB
spgemm_vram = {}
spmm_vram = {}

# Use one global pool handle
pool = cp.get_default_memory_pool()

# -----------------------------
# Helper: median
# -----------------------------
def median_ms(records):
    records.sort()
    return records[len(records) // 2]

# -----------------------------
# Helper: measure PEAK VRAM (MB)
# -----------------------------
def measure_peak_vram_mb(func):
    """
    Measure peak GPU VRAM usage (MB) using CUDA runtime free memory difference.

    We approximate peak usage as:
        peak = free_before - free_after
    after forcing synchronization and keeping output alive.

    This captures workspace + output allocations that actually reduce device free memory,
    and is closer to "peak VRAM" than memory-pool used_bytes().
    """
    # Clean allocator state as much as possible
    pool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    free_before, total = cp.cuda.runtime.memGetInfo()

    out = func()  # IMPORTANT: keep output alive
    cp.cuda.Stream.null.synchronize()

    free_after, _ = cp.cuda.runtime.memGetInfo()

    peak_mb = (free_before - free_after) / (1024 ** 2)

    # cleanup to avoid affecting the next measurement
    del out
    cp.cuda.Stream.null.synchronize()
    pool.free_all_blocks()
    cp.cuda.Stream.null.synchronize()

    return peak_mb

# -----------------------------
# Benchmark
# -----------------------------
for n in matrix_size:
    for d in density_list:
        # Fix matrices for fair comparison
        A_sp = sp.random(n, n, density=d, format="csr", dtype=cp.float32)
        B_sp = sp.random(n, n, density=d, format="csr", dtype=cp.float32)
        B_dense = B_sp.toarray()

        # Warm-up
        for _ in range(warmup):
            _ = A_sp @ B_sp
            _ = A_sp @ B_dense
        cp.cuda.Stream.null.synchronize()

        start = cp.cuda.Event()
        end = cp.cuda.Event()

        # ---- SpGEMM timing
        spgemm_record = []
        for _ in range(run_time):
            start.record()
            _ = A_sp @ B_sp
            end.record()
            end.synchronize()
            spgemm_record.append(cp.cuda.get_elapsed_time(start, end))
        spgemm_results[(n, d)] = median_ms(spgemm_record)

        # ---- SpMM timing
        spmm_record = []
        for _ in range(run_time):
            start.record()
            _ = A_sp @ B_dense
            end.record()
            end.synchronize()
            spmm_record.append(cp.cuda.get_elapsed_time(start, end))
        spmm_results[(n, d)] = median_ms(spmm_record)

        # ---- Peak VRAM measurement (MB)
        spgemm_vram[(n, d)] = measure_peak_vram_mb(lambda: (A_sp @ B_sp))
        spmm_vram[(n, d)]   = measure_peak_vram_mb(lambda: (A_sp @ B_dense))

        print(
            f"n={n:4d}, density={d}, "
            f"SpGEMM={spgemm_results[(n,d)]:.4f} ms, "
            f"SpMM={spmm_results[(n,d)]:.4f} ms | "
            f"Peak VRAM SpGEMM={spgemm_vram[(n,d)]} MB, "
            f"SpMM={spmm_vram[(n,d)]} MB"
        )

# -----------------------------
# Plotting helpers
# -----------------------------
def set_log_x_ticks(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(matrix_size)
    ax.set_xticklabels([str(n) for n in matrix_size])

# -----------------------------
# Figure A: Time (SpGEMM vs SpMM) per density
# -----------------------------
fig, axes = plt.subplots(1, len(density_list), figsize=(5 * len(density_list), 4), squeeze=False)

for idx, d in enumerate(density_list):
    ax = axes[0, idx]
    spgemm_times = [spgemm_results[(n, d)] for n in matrix_size]
    spmm_times   = [spmm_results[(n, d)]   for n in matrix_size]

    ax.plot(matrix_size, spgemm_times, marker="o", label="SpGEMM (A@B_sparse)")
    ax.plot(matrix_size, spmm_times,   marker="s", label="SpMM (A@B_dense)")

    ax.set_title(f"Time (density = {d})")
    ax.set_xlabel("Matrix size (n x n)")
    ax.set_ylabel("Median time (ms)")

    set_log_x_ticks(ax)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig("time_spgemm_vs_spmm.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Figure B: Time ratio SpGEMM/SpMM (>1 means SpMM faster)
# -----------------------------
fig = plt.figure(figsize=(6, 4))
ax = fig.gca()

for d in density_list:
    ratio = [spgemm_results[(n, d)] / spmm_results[(n, d)] for n in matrix_size]
    ax.plot(matrix_size, ratio, marker="o", label=f"density={d}")

ax.axhline(1.0, linestyle="--")
ax.set_xlabel("Matrix size (n x n)")
ax.set_ylabel("SpGEMM / SpMM time ratio")
ax.set_title("Relative performance (SpGEMM vs SpMM)")

set_log_x_ticks(ax)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.legend()

plt.tight_layout()
plt.savefig("time_ratio_spgemm_over_spmm.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# -----------------------------
# Figure C: Peak VRAM usage (SpGEMM vs SpMM) per density
# -----------------------------
fig, axes = plt.subplots(1, len(density_list), figsize=(5 * len(density_list), 4), squeeze=False)

for idx, d in enumerate(density_list):
    ax = axes[0, idx]
    spgemm_mem = [spgemm_vram[(n, d)] for n in matrix_size]
    spmm_mem   = [spmm_vram[(n, d)]   for n in matrix_size]

    ax.plot(matrix_size, spgemm_mem, marker="o", label="SpGEMM peak VRAM")
    ax.plot(matrix_size, spmm_mem,   marker="s", label="SpMM peak VRAM")

    ax.set_title(f"Peak VRAM (density = {d})")
    ax.set_xlabel("Matrix size (n x n)")
    ax.set_ylabel("Peak GPU memory usage (MB)")

    set_log_x_ticks(ax)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()

plt.tight_layout()
plt.savefig("vram_peak_spgemm_vs_spmm.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved PNGs:")
print(" - time_spgemm_vs_spmm.png")
print(" - time_ratio_spgemm_over_spmm.png")
print(" - vram_peak_spgemm_vs_spmm.png")
