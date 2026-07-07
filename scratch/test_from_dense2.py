import quimb.tensor as qtn
import numpy as np
raw_data = np.random.rand(4, 4, 4, 4)
m1 = qtn.MatrixProductState.from_dense(raw_data, dims=[4] * 4, method='svd')
m2 = qtn.MatrixProductState.from_dense(raw_data, dims=[4] * 4, method='qr')
print("m1 norm:", m1.norm())
print("m2 norm:", m2.norm())
