import quimb.tensor as qtn
import numpy as np
raw_data = np.random.rand(4, 4, 4, 4)
try:
    mps = qtn.MatrixProductState.from_dense(raw_data, dims=[4] * 4, max_bond=4, method='qr')
    print("from_dense worked with method=qr")
except Exception as e:
    print(e)
