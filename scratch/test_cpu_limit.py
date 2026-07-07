import torch
from concurrent.futures import ProcessPoolExecutor
print("Threads set to 1")
torch.set_num_threads(1)
import numpy as np

def worker(x):
    A = torch.randn(1000, 1000)
    Q, R = torch.linalg.qr(A)
    return float(Q[0,0])

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        res = list(pool.map(worker, range(4)))
    print("Done:", res)
