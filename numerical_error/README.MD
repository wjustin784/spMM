# Numerical Error on SPGEMM ALG1 and ALG3
This directory contains several Python scripts for evaluating numerical differences between cuSPARSE SpGEMM ALG1 and ALG3.
Each script focuses on a different factor that may influence numerical behavior.
All experiments generate visualizations to illustrate the observed numerical differences.

# distribution.py
This script evaluates the distribution of numerical errors between SpGEMM ALG1 and ALG3.

It computes element-wise differences between the two results and visualizes the error distribution.
Users can modify the low and high parameters to control the minimum and maximum values of the nonzero entries in the input matrices.

# fraction.py
This script investigates whether different chunk_fraction values in SpGEMM ALG3 lead to numerical differences when using the same input matrices.

It helps determine how the chunking strategy in ALG3 affects numerical stability.

# error.py
This script generates a 3×3 heatmap showing the maximum absolute error between SpGEMM ALG1 and ALG3 across different combinations of matrix density and matrix size.

The visualization highlights how numerical differences scale with problem configuration.

# range.py
range.py sweeps the input value range [0,high] and reports the worst-case max absolute error between SpGEMM ALG1 and ALG3, visualized as spgemm_error_vs_range.png.