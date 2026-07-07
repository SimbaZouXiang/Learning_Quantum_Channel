import time
import torch
import os
os.environ["OMP_NUM_THREADS"] = "1"
from TDME_Trott import Learning_TDME_scheduler

def run_benchmark():
    N = 6
    MPO_layer = 1
    model_to_learn_layer = 1
    mu = 0.1
    gamma = [0.0] * N
    
    threads_list = [1, 4, 8]
    times = []

    for threads in threads_list:
        print(f"--- Benchmark with num_threads = {threads} ---")
        start_time = time.time()
        
        model, lloss, tloss, tloss_list = Learning_TDME_scheduler(
            N=N, 
            MPO_layer=MPO_layer, 
            model_to_learn_layer=model_to_learn_layer, 
            mu=mu, 
            gamma=gamma, 
            epochs=0,
            use_scheduler=False,
            num_threads=threads
        )
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"-> Time taken for {threads} thread(s): {elapsed:.2f} seconds\n")

    print("\n=== SUMMARY N=6 ===")
    for t, elapsed in zip(threads_list, times):
        print(f"Threads: {t:>2} | Time: {elapsed:.2f}s | Speedup: {times[0]/elapsed:.2f}x")

if __name__ == "__main__":
    run_benchmark()
