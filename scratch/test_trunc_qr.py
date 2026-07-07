import quimb.tensor as qtn
import numpy as np
raw_data = np.random.rand(4, 4, 4, 4)
m2 = qtn.MatrixProductState.from_dense(raw_data, dims=[4] * 4, max_bond=4, cutoff=1e-10, method='qr')
print(m2.max_bond())
