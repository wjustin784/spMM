import cupy as cp
import cupyx.scipy.sparse as sp
import numpy as np
from cupyx import cusparse

n = 2**20          
d = 0.00001

k = int(d * n * n)  

rows = cp.random.randint(0, n, size=k, dtype=cp.int32)
cols = cp.random.randint(0, n, size=k, dtype=cp.int32)
data = cp.random.random(k, dtype=cp.float32)

A = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
A.sum_duplicates()
A.sort_indices()

# reuse different random for B
rows2 = cp.random.randint(0, n, size=k, dtype=cp.int32)
cols2 = cp.random.randint(0, n, size=k, dtype=cp.int32)
data2 = cp.random.random(k, dtype=cp.float32)

B = sp.coo_matrix((data2, (rows2, cols2)), shape=(n, n)).tocsr()
B.sum_duplicates()
B.sort_indices()

print(A, B)

start = cp.cuda.Event()
end = cp.cuda.Event()

start.record()
C = cusparse.spgemm(A, B, alg=1, verbose=True)
end.record()
end.synchronize()

print("spgemm time (ms):", cp.cuda.get_elapsed_time(start, end))
print(C)

