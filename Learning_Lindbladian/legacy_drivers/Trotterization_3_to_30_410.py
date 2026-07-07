import os
import numpy as np
import sys

# Ensure we can import from TDME_Trott
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TDME_Trott import Testing_TDME_Trotterization_parallel

def _run_single_test(args):
    t, g, N, T, model_to_learn_layer, mu, J, normalized, max_bd, truncation, noise_type, use_scheduler, inner_threads, output_dir = args

    print(f"[{os.getpid()}] Starting Testing_TDME_Trotterization_parallel for t={t}, gamma={g}")
    gamma_array = [g] * N

    loss_file = os.path.join(output_dir, f"Trotterization_Testing_loss_t{t}_gamma{g}_N{N}_T{T}_Modeltolearnlayer{model_to_learn_layer}_mu{float(mu)}_gamma{round(gamma_array[0], 2)}_t{t}.npy")
    list_file = os.path.join(output_dir, f"Trotterization_Testing_loss_list_t{t}_gamma{g}_N{N}_T{T}_Modeltolearnlayer{model_to_learn_layer}_mu{float(mu)}_gamma{round(gamma_array[0], 2)}_t{t}.npy")
    partial_prefix = os.path.join(output_dir, f"PARTIAL_Trotterization_t{t}_gamma{g}_N{N}_T{T}_Modeltolearnlayer{model_to_learn_layer}_mu{float(mu)}_gamma{round(gamma_array[0], 2)}")

    testing_loss, testing_loss_list = Testing_TDME_Trotterization_parallel(
        N=N,
        model_layer=T,
        model_to_learn_layer=model_to_learn_layer,
        mu=mu,
        gamma=gamma_array,
        J=J,
        t=t,
        normalized=normalized,
        max_bd=max_bd,
        truncation=truncation,
        noise_type=noise_type,
        use_scheduler=use_scheduler,
        num_threads=inner_threads,
        incremental_save_prefix=partial_prefix,
    )

    np.save(loss_file, testing_loss)
    np.save(list_file, testing_loss_list)
    for suffix in ("_list.npy", "_loss.npy"):
        p = partial_prefix + suffix
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"[{os.getpid()}] Completed t={t}, gamma={g}. Saved to {loss_file}\n")
    return t, g

def run_tests():
    N                    = 30
    T                    = 3
    model_to_learn_layer = 30
    mu                   = 1
    J                    = 1
    normalized           = False
    truncation           = True
    noise_type           = "dephasing"
    use_scheduler        = False
    max_bd               = 64

    target_times = [1.0]
    gamma_names  = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]

    # Flat parallelism: run the (few) outer tasks one at a time, but give each
    # one a bounded CPU pool for its inner ProcessPoolExecutor.  Nesting an
    # outer pool of 4 around an inner pool of 16 only used 64 of the 192 CPUs
    # on the node and left sibling inner workers fighting for the same cores.
    # Cap at 100 workers so the node doesn't OOM — each sample worker loads
    # its own copy of the pre-computed unitaries + jump matrices, and running
    # all 192 in parallel risked exhausting the 767 GB node memory.
    MAX_WORKERS   = 48
    total_cpus    = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    inner_threads = min(total_cpus, MAX_WORKERS)

    output_dir = "Learning_result"
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for t in target_times:
        for g in gamma_names:
            tasks.append((t, g, N, T, model_to_learn_layer, mu, J, normalized, max_bd, truncation, noise_type, use_scheduler, inner_threads, output_dir))

    print(f"Running {len(tasks)} tasks sequentially with {inner_threads} inner workers each...", flush=True)

    for task in tasks:
        t, g = _run_single_test(task)

if __name__ == "__main__":
    run_tests()

