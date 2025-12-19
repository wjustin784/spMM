#!/bin/bash


python3 cupy_alg1.py --out file1.txt --dtype float32
python3 cupy_alg1.py --out file2.txt --dtype float32

if diff -q file1.txt file2.txt >/dev/null 2>&1; then
    echo "alg1 is deterministic"
else
    echo "not deterministic"
fi