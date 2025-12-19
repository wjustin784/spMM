# Cupy vs Cusparse
This directory contains tests for verifying whether the numerical results produced by CuPy and cuSPARSE are identical.
The test workflow is as follows:
1. CuPy is used to generate two sparse matrices.
2. The matrices are serialized and stored in text files.
3. cuSPARSE reads the matrices from the text files and performs sparse matrix–matrix multiplication.
4. The results computed by CuPy and cuSPARSE are compared to detect any numerical differences.

### first compile C code by execute.
> ./build.sh

### for test Cusparse ALG1
> ./run_all_alg1.sh

### for test Cusparse ALG2
> ./run_all_alg2.sh

### for test Cusparse ALG3
> ./run_all_alg3.sh