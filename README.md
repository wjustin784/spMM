# Sparse Matrix Multiplication Benchmarks (Reproducibility Guide)

This repository contains benchmark scripts used in our project report.  
Please follow the instructions below to set up the environment and reproduce the benchmark result `.txt` files.

---

## 0. Environment Setup (Required)

### 0.1 Install and use the modified CuPy source

```bash
cd modify_src/cupy-src

conda install -c conda-forge -y libstdcxx-ng>=13.2.0 libgcc-ng>=13.2.0
conda install -c conda-forge -y _libgcc_mutex sysroot_linux-64

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export LD=/usr/bin/ld

export CUPY_NVCC_GENERATE_CODE=current

export CUDA_PATH=/usr/local/cuda
export PATH=$PATH:$CUDA_PATH/bin
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

python -m pip install -e . -v
```

### 0.2 Install Python dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

---

## 1. Running the Benchmarks

Each benchmark directory contains a `run.sh` script.

Steps:
1. `cd` into the directory
2. run `bash run.sh`
3. after completion, benchmark result `.txt` file(s) will be generated and can be used for inspection and verification

---

## 2. Benchmark Directories

### A) `SpGEMM_vs_SpMV`

**Description:**  
This directory reproduces the comparison between SpGEMM (sparse × sparse) and SpMV (sparse × dense vector) behavior. In the report, SpMV is shown to be strongly memory/transfer sensitive, and end-to-end GPU performance can look worse than CPU once H2D is included.  

<img src="figures/SPGEMM-gpu-speedup.png" alt="SpGEMM" width="400">
<img src="figures/spmv-density.png" alt="SpMV" width="400">  


**How to run:**
```bash
cd SpGEMM_vs_SpMV
bash run.sh
```

**Output:**  
Benchmark result `.txt` file(s) generated after execution.

---

### B) `SpGEMM_alg_comparison`

**Description:**  
This directory compares cuSPARSE SpGEMM algorithm variants (ALG1/ALG2/ALG3) and highlights the core trade-off: ALG1 is fastest but uses the most VRAM, while ALG3 reduces peak memory via chunking but is typically slowest, and ALG2 is a middle ground.  


<img src = "figures/alg_comparison.png" alt = "alg_comparison" width = "800"  >  

**How to run:**
```bash
cd SpGEMM_alg_comparison
bash run.sh
```

**Output:**  
Benchmark result `.txt` file(s) generated after execution.

---

### C) `dense_vs_sparseGEMM`

**Description:**  
This directory reproduces the CSR@CSR SpGEMM vs dense GEMM comparison under an inputs-on-GPU setting (i.e., focusing on operator behavior rather than transfer overheads).
The report’s main finding here is that sparse computation is only beneficial in an extremely sparse regime, and the break-even point shifts with matrix size.  

<img src = "figures/runtime_vs_density.png" alt = runtime_vs_density width = "800">  


**How to run:**
```bash
cd dense_vs_sparseGEMM
bash run.sh
```

**Output:**  
Benchmark result `.txt` file(s) generated after execution.
