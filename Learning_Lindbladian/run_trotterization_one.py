"""Run one (t, gamma) Trotterization task, selected by SLURM_ARRAY_TASK_ID.

Grid: 3 times × 10 gammas = 30 tasks (array index 0..29).
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
from TDME_Trott import Testing_TDME_Trotterization_parallel

TARGET_TIMES = [0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
GAMMA_NAMES  = [0, 2, 4, 6, 8, 10, 20, 30, 40, 50]


def main():
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= task_id < len(TARGET_TIMES) * len(GAMMA_NAMES):
        raise ValueError(f"task_id {task_id} out of range")

    t_idx, g_idx = divmod(task_id, len(GAMMA_NAMES))
    t_val = TARGET_TIMES[t_idx]
    g_name = GAMMA_NAMES[g_idx]
    gamma_val = g_name * 0.01

    N = 30
    T_model = 3
    T_target = 30
    mu = 1
    J = 1
    max_bd = 64
    truncation = True
    noise_type = "dephasing"
    use_scheduler = False
    num_samples = 300

    total_cpus = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else (os.cpu_count() or 1)
    )
    # 48 workers proved memory-safe (MaxRSS ~217 GB on Apr 24); below the
    # 767 GB node limit and the 100-worker OOM ceiling we hit earlier.
    num_threads = min(48, total_cpus)

    output_dir = os.path.join(SCRIPT_DIR, "Learning_result")
    os.makedirs(output_dir, exist_ok=True)

    gamma_array = [gamma_val] * N
    # Filename matches the legacy Trotterization_3_to_30_*.py convention
    # exactly, so this overwrites the buggy Apr-29 saved files in place.
    suffix = (
        f"t{t_val}_gamma{g_name}_N{N}_T{T_model}_"
        f"Modeltolearnlayer{T_target}_mu{float(mu)}_"
        f"gamma{g_name}_t{t_val}"
    )
    loss_file = os.path.join(output_dir, f"Trotterization_Testing_loss_{suffix}.npy")
    list_file = os.path.join(output_dir, f"Trotterization_Testing_loss_list_{suffix}.npy")
    partial_prefix = os.path.join(
        output_dir,
        f"PARTIAL_Trotterization_t{t_val}_gamma{g_name}_N{N}_T{T_model}_"
        f"Modeltolearnlayer{T_target}_mu{float(mu)}_"
        f"gamma{g_name}",
    )

    print(
        f"[task {task_id}] t={t_val} gamma_name={g_name} "
        f"(physical gamma={gamma_val})  threads={num_threads}",
        flush=True,
    )

    testing_loss, testing_loss_list = Testing_TDME_Trotterization_parallel(
        N=N,
        model_layer=T_model,
        model_to_learn_layer=T_target,
        mu=mu,
        gamma=gamma_array,
        J=J,
        t=t_val,
        normalized=False,
        max_bd=max_bd,
        truncation=truncation,
        noise_type=noise_type,
        use_scheduler=use_scheduler,
        num_samples=num_samples,
        num_threads=num_threads,
        incremental_save_prefix=partial_prefix,
    )

    np.save(loss_file, testing_loss)
    np.save(list_file, testing_loss_list)
    for s in ("_list.npy", "_loss.npy"):
        p = partial_prefix + s
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"[task {task_id}] Done. Saved {loss_file}", flush=True)


if __name__ == "__main__":
    main()
