#!/bin/bash


# Sweeps density from 0.0001 to 0.1 (log scale)

OUTFILE="benchmark_results.txt"
RESULT_JSON="results.json"
echo "Benchmark results - $(date)" > "$OUTFILE"
echo -e "==========================================\n" >> "$OUTFILE"

# Define density list (log spaced)
for s in 512 1024
do
    for d in 0.1 0.5
    do
        echo -e "size = $s, density = $d" | tee -a  "$OUTFILE"
        echo "--- computing ---" | tee -a "$OUTFILE"
        python3 profiler.py --density $d  --size $s --runs 100 >> "$OUTFILE" 2>&1
        echo "complete!"

        echo "" >> "$OUTFILE"
    done
done

echo -e "All runs completed. Results saved to $OUTFILE\n"