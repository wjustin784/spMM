# cuSPARSE Algorithm Consistency Test

### Overview
This folder contains tests for verifying whether different cuSPARSE algorithms produce identical results.

### Method
We use a fixed random seed in CuPy to generate the same input sparse matrices **A** and **B**.  
Each cuSPARSE algorithm is executed in a separate process, and the resulting matrix multiplication outputs are written to disk.

### Verification
The outputs from different algorithms are compared using `diff` to check whether they are identical.

### Purpose
This procedure ensures that any observed differences are due to algorithmic behavior rather than input randomness or process-level interference.

## How to run
> ./test_deterministic.sh