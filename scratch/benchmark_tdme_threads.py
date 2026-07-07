import time
from TDME_Trott import Testing_TDME_Trotterization_parallel

def benchmark():
    N = 10
    model_layer = 2
    model_to_learn_layer = 4
    mu = 0.5
    gamma = [0.1] * N
    num_samples = 64
    
    threads_to_test = [1, 4, 8, 16]
    
    print("Starting benchmark for Testing_TDME_Trotterization_parallel...")
    print(f"Parameters: N={N}, model_layer={model_layer}, model_to_learn_layer={model_to_learn_layer}, mu={mu}, gamma={gamma}, num_samples={num_samples}")
    print("-" * 50)
    
    for num_threads in threads_to_test:
        print(f"\nRunning with num_threads = {num_threads}...")
        start_time = time.time()
        
        Testing_TDME_Trotterization_parallel(
            N=N, 
            model_layer=model_layer, 
            model_to_learn_layer=model_to_learn_layer, 
            mu=mu, 
            gamma=gamma, 
            num_samples=num_samples, 
            num_threads=num_threads
        )
        execution_time = time.time() - start_time
        print(f"Time taken for {num_threads} threads: {execution_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
