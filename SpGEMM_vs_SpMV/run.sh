#!/bin/bash


# Sweeps density from 0.0001 to 0.1 (log scale)

OUTFILE="benchmark_results.txt"
echo "Benchmark results - $(date)" > "$OUTFILE"
echo -e "==========================================\n" >> "$OUTFILE"

for s in 128 256 512 1024
do
    for d in 0.01 0.05 0.1 0.5
    do
        echo ">>> Running size = $s    Running density = $d" | tee -a "$OUTFILE"
        echo "--- computing ---" | tee -a "$OUTFILE"
        python3 profiler.py --densityA $d --densityB $d --m $s --n $s --p $s >> "$OUTFILE" 2>&1
        echo "complete!"

        echo "" >> "$OUTFILE"
    done
done

echo -e "All runs completed. Results saved to $OUTFILE\n"
