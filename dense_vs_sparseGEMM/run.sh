#!/bin/bash


# Sweeps density from 0.0001 to 0.1 (log scale)

OUTFILE="benchmark_results.txt"
RESULT_JSON="results.json"
echo "Benchmark results - $(date)" > "$OUTFILE"
echo -e "==========================================\n" >> "$OUTFILE"

# Define density list (log spaced)
for s in 1024 2048 4096 8192
do
    for d in 0.001 0.01 0.05 0.1
    do
        echo -e "size = $s, density = $d" | tee -a  "$OUTFILE"
        echo "--- computing ---" | tee -a "$OUTFILE"
        python3 main.py --density $d  --size $s >> "$OUTFILE" 2>&1
        echo "complete!"

        echo "" >> "$OUTFILE"
    done
done

echo -e "All runs completed. Results saved to $OUTFILE\n"