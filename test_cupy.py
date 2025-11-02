import cupy as cp
import cupyx.scipy.sparse as sp
from cupyx.profiler import benchmark
from cupyx import cusparse

cp.random.seed(0)
dim = 1024
D = 0.3
A = sp.random(dim, dim, density=D, format="csr", dtype=cp.float32)
B = sp.random(dim, dim, density=D, format="csr", dtype=cp.float32)

C = A @ B
D = cusparse.spgemm(A, B, alg=3)

print(cp.allclose(C.toarray(), D.toarray()))
print(cp.all(C.toarray() == D.toarray()))

diff = cp.abs(C.toarray() - D.toarray())
print("max error =", diff.max()) 