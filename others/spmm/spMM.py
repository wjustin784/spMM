import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np

matrix_size =  [128, 256, 512, 1024]
density_list = [0.01, 0.1, 0.5]


run_time = 1000  

sparse_results = {}
spmm_results = {}

for n in matrix_size:
    for d in density_list:

        sparse_record = []

        for _ in range(run_time):

            A_sp = sp.random(n, n, density=d, format="csr", dtype=cp.float32)
            B_sp = sp.random(n, n, density=d, format="csr", dtype=cp.float32)

            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            C_sp = A_sp @ B_sp 
            end.record()
            end.synchronize()

            sparse_ms = cp.cuda.get_elapsed_time(start, end)
            sparse_record.append(sparse_ms)
            
        spmm_record = []
        B_dense = B_sp.toarray()
        
        for _ in range(run_time):

            start = cp.cuda.Event()
            end = cp.cuda.Event()

            start.record()
            C_dense = A_sp @ B_dense 
            end.record()
            end.synchronize()


            dense_ms = cp.cuda.get_elapsed_time(start, end)
            spmm_record.append(dense_ms)

        spmm_record.sort()
        spmm_median = spmm_record[len(spmm_record) // 2]
        spmm_results[(n, d)] = spmm_median
        

        sparse_record.sort()
        sparse_median = sparse_record[len(sparse_record) // 2]
        sparse_results[(n, d)] = sparse_median

        print(
            f"n={n:4d}, density={d:.2f}, "
            f"Spgemm={sparse_median:.4f} ms, "
            f"spMM median= {spmm_median:.4f}"
        )
