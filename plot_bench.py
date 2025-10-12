#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse benchmark text (your run_bench.sh output) and draw charts.

Charts supported (toggle via flags):
- Hot times:    SpGEMM, SpMV
- Cold times:   SpGEMM, SpMV
- Build times:  build_A, build_B

Usage examples:
  python3 plot_bench.py --log benchmark_results.txt --out out_charts --plots hot
  python3 plot_bench.py --log benchmark_results.txt --out out_charts --plots hot cold
  python3 plot_bench.py --log benchmark_results.txt --out out_charts --plots hot cold build
"""
import numpy as np
import argparse
import os
import re
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

def parse_log_text(text: str) -> pd.DataFrame:
    density_re = re.compile(r">>>\s*Running density\s*=\s*([0-9.]+)")
    matches = list(density_re.finditer(text))
    if not matches:
        raise ValueError("No '>>> Running density = ...' sections found in log.")

    blocks = []
    for i, m in enumerate(matches):
        dens = float(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks.append((dens, text[start:end]))

    def grab(section: str) -> Dict[str, float]:
        d: Dict[str, float] = {}
        # Hot A@B (accept SpGEMM or SpMM)
        m = re.search(r"hot_A@B\s*\((?:SpGEMM|SpMM)\)\s+([0-9.]+)", section)
        if m: d["hot_spgemm"] = float(m.group(1))
        # Hot A@C
        m = re.search(r"hot_A@C\s*\(SpMV,\s*dense\s*vec\)\s+([0-9.]+)", section)
        if m: d["hot_spmv"] = float(m.group(1))
        # Cold A@B
        m = re.search(r"cold_A@B\s*\((?:SpGEMM|SpMM)\)\s+([0-9.]+)", section)
        if m: d["cold_spgemm"] = float(m.group(1))
        # Cold A@C
        m = re.search(r"cold_A@C\s*\(SpMV,\s*dense\s*vec\)\s+([0-9.]+)", section)
        if m: d["cold_spmv"] = float(m.group(1))
        # Build times
        m = re.search(r"build_A_sparse\s+([0-9.]+)", section)
        if m: d["build_A"] = float(m.group(1))
        m = re.search(r"build_B_sparse\s+([0-9.]+)", section)
        if m: d["build_B"] = float(m.group(1))
        return d

    rows: List[dict] = []
    for dens, block in blocks:
        sci, cupy = {}, {}
        if "--- SciPy ---" in block:
            s = block.split("--- SciPy ---", 1)[1]
            s = s.split("--- CuPy ---", 1)[0] if "--- CuPy ---" in s else s
            sci = grab(s)
        if "--- CuPy ---" in block:
            c = block.split("--- CuPy ---", 1)[1]
            cupy = grab(c)

        rows.append({
            "density": dens,
            # HOT
            "cpu_hot_spgemm_s": sci.get("hot_spgemm", float("nan")),
            "cpu_hot_spmv_s":   sci.get("hot_spmv",   float("nan")),
            "gpu_hot_spgemm_s": cupy.get("hot_spgemm", float("nan")),
            "gpu_hot_spmv_s":   cupy.get("hot_spmv",   float("nan")),
            # COLD
            "cpu_cold_spgemm_s": sci.get("cold_spgemm", float("nan")),
            "cpu_cold_spmv_s":   sci.get("cold_spmv",   float("nan")),
            "gpu_cold_spgemm_s": cupy.get("cold_spgemm", float("nan")),
            "gpu_cold_spmv_s":   cupy.get("cold_spmv",   float("nan")),
            # BUILD
            "cpu_buildA_s": sci.get("build_A", float("nan")),
            "cpu_buildB_s": sci.get("build_B", float("nan")),
            "gpu_buildA_s": cupy.get("build_A", float("nan")),
            "gpu_buildB_s": cupy.get("build_B", float("nan")),
        })

    df = pd.DataFrame(rows).sort_values("density").reset_index(drop=True)
    # Speedups
    df["hot_speedup_spgemm"]  = df["cpu_hot_spgemm_s"]  / df["gpu_hot_spgemm_s"]
    df["hot_speedup_spmv"]    = df["cpu_hot_spmv_s"]    / df["gpu_hot_spmv_s"]
    df["cold_speedup_spgemm"] = df["cpu_cold_spgemm_s"] / df["gpu_cold_spgemm_s"]
    df["cold_speedup_spmv"]   = df["cpu_cold_spmv_s"]   / df["gpu_cold_spmv_s"]
    return df

def plot_time_vs_density(df, cpu_col, gpu_col, title, fname, out_dir):
    plt.figure()
    plt.xscale("log"); plt.yscale("log")
    plt.plot(df["density"], df[cpu_col], marker="o", label="CPU (SciPy)")
    plt.plot(df["density"], df[gpu_col], marker="o", label="GPU (CuPy)")
    plt.xlabel("Density"); plt.ylabel("Time (s)"); plt.title(title)
    plt.legend(); plt.tight_layout()
    path = os.path.join(out_dir, fname); plt.savefig(path, dpi=150); plt.close()
    return path


# def plot_time_vs_density(df, cpu_col, gpu_col, title, fname, out_dir):
#     import matplotlib.ticker as ticker
#     import numpy as np
#     import os
#     import matplotlib.pyplot as plt

#     fig, ax = plt.subplots()

#     # X on LINEAR scale, Y on LOG scale
#     ax.set_xscale("linear")
#     ax.set_yscale("log")

#     x = df["density"].astype(float).values
#     ax.plot(x, df[cpu_col], marker="o", label="CPU (SciPy)")
#     ax.plot(x, df[gpu_col], marker="o", label="GPU (CuPy)")

#     # Put ticks exactly at your densities so spacing reflects absolute differences
#     ax.set_xticks(x)
#     ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter(useMathText=True))
#     ax.ticklabel_format(axis="x", style="plain")  # avoid scientific notation if you prefer

#     ax.set_xlabel("Density (linear scale)")
#     ax.set_ylabel("Time (s, log scale)")
#     ax.set_title(title)
#     ax.legend()
#     fig.tight_layout()

#     path = os.path.join(out_dir, fname)
#     fig.savefig(path, dpi=150)
#     plt.close(fig)
#     return path


def plot_speedup(df, speedup_col1, label1, speedup_col2, label2, title, fname, out_dir):
    plt.figure()
    plt.xscale("log")
    plt.plot(df["density"], df[speedup_col1], marker="o", label=label1)
    plt.plot(df["density"], df[speedup_col2], marker="o", label=label2)
    plt.xlabel("Density"); plt.ylabel("CPU/GPU speedup (×)"); plt.title(title)
    plt.legend(); plt.tight_layout()
    path = os.path.join(out_dir, fname); plt.savefig(path, dpi=150); plt.close()
    return path

def main():
    ap = argparse.ArgumentParser(description="Plot charts from benchmark_results.txt")
    ap.add_argument("--log", required=True, help="Path to run_bench output")
    ap.add_argument("--out", default="charts_out", help="Output directory")
    ap.add_argument("--plots", nargs="+", default=["hot", "cold", "build"],
                    choices=["hot","cold","build"],
                    help="Which chart sets to generate")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.log, "r", encoding="utf-8") as f:
        text = f.read()

    df = parse_log_text(text)
    df.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    generated = []

    if "hot" in args.plots:
        generated.append(plot_time_vs_density(
            df, "cpu_hot_spgemm_s","gpu_hot_spgemm_s",
            "Hot SpGEMM (A@B) time vs density","hot_spgemm_time_vs_density.png", args.out))
        generated.append(plot_time_vs_density(
            df, "cpu_hot_spmv_s","gpu_hot_spmv_s",
            "Hot SpMV (A@C) time vs density","hot_spmv_time_vs_density.png", args.out))
        generated.append(plot_speedup(
            df, "hot_speedup_spgemm","SpGEMM speedup",
            "hot_speedup_spmv","SpMV speedup",
            "GPU speedup over CPU (hot)","hot_gpu_speedup_over_cpu.png", args.out))

    if "cold" in args.plots:
        generated.append(plot_time_vs_density(
            df, "cpu_cold_spgemm_s","gpu_cold_spgemm_s",
            "Cold SpGEMM (A@B) time vs density","cold_spgemm_time_vs_density.png", args.out))
        generated.append(plot_time_vs_density(
            df, "cpu_cold_spmv_s","gpu_cold_spmv_s",
            "Cold SpMV (A@C) time vs density","cold_spmv_time_vs_density.png", args.out))
        generated.append(plot_speedup(
            df, "cold_speedup_spgemm","SpGEMM speedup",
            "cold_speedup_spmv","SpMV speedup",
            "GPU speedup over CPU (cold)","cold_gpu_speedup_over_cpu.png", args.out))

    if "build" in args.plots:
        # Build times for A and B (CPU vs GPU), log-log by density
        generated.append(plot_time_vs_density(
            df, "cpu_buildA_s","gpu_buildA_s",
            "Build time vs density (A CSR)","buildA_time_vs_density.png", args.out))
        generated.append(plot_time_vs_density(
            df, "cpu_buildB_s","gpu_buildB_s",
            "Build time vs density (B CSR)","buildB_time_vs_density.png", args.out))

    print("Saved files:")
    for p in generated:
        print(" -", p)
    print(" -", os.path.join(args.out, "summary.csv"))

if __name__ == "__main__":
    main()
