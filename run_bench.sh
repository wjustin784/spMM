#!/bin/bash


# Sweeps density from 0.0001 to 0.1 (log scale)

OUTFILE="benchmark_results.txt"
echo "Benchmark results - $(date)" > "$OUTFILE"
echo -e "==========================================\n" >> "$OUTFILE"

# Define density list (log spaced)
for d in 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
do
    echo ">>> Running density = $d" | tee -a "$OUTFILE"
    echo "--- SciPy ---" | tee -a "$OUTFILE"
    python3 scipy_bench.py --densityA $d --densityB $d >> "$OUTFILE" 2>&1

    echo "--- CuPy ---" | tee -a "$OUTFILE"
    python3 cupy_bench.py --densityA $d --densityB $d >> "$OUTFILE" 2>&1

    echo "" >> "$OUTFILE"
done

echo -e "All runs completed. Results saved to $OUTFILE\n"


python plot_bench.py --log benchmark_results.txt