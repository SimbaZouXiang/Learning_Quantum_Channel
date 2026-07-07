import time
from TDME_Trott import Learning_TDME_scheduler

def run_benchmark():
    N = 8
    start_time = time.time()
    
    Learning_TDME_scheduler(N=N, MPO_layer=3, model_to_learn_layer=3, mu=0.1, gamma=[0.0] * N, epochs=0, use_scheduler=False, num_threads=1)
    
    elapsed = time.time() - start_time
    print(f"-> Time N={N}: {elapsed:.2f} seconds\n")

if __name__ == "__main__":
    run_benchmark()
