#!/usr/bin/env bash
set -e


nvcc spgemm_from_txt_alg1.cu -o spgemm_from_txt_alg1 -lcusparse
nvcc spgemm_from_txt_alg2.cu -o spgemm_from_txt_alg2 -lcusparse
nvcc spgemm_from_txt_alg3.cu -o spgemm_from_txt_alg3 -lcusparse

echo "Compile Finish"
