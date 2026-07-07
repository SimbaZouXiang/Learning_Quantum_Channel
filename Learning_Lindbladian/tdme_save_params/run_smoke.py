"""Smoke run: 1 cell (t=1.0, gamma=0.1, variant=w1), 10 epochs, parallel data gen.

Estimates per-epoch + total wall-time so we can plan production batching.
Saves all student artifacts + meta.json so we can reconstruct later.
"""
import os, sys, time, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)

# ---------- cell + knobs ----------
N           = 8
T_STUDENT   = 3
L_TARGET    = 10
MU          = 1
J           = 1
T_TIME      = 1.0
GAMMA       = 0.1
VARIANT_TAG = "w2full"
INPUT_PW    = "2_full"
EPOCHS      = 3                  # smoke: short, just to measure
NUM_THREADS = 16                 # parallel data gen + parallel testing
LR          = 0.01
MAX_BD      = 64
MAX_ERR     = 1e-6
TRUNCATION  = False
USE_COMPRESSED = False           # matches existing Learning_TDME_using_data_*.py drivers
USE_SCHED   = True
NOISE_TYPE  = "dephasing"
SEED        = 12345


def main():
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 32))
    # leave headroom for the data-gen child processes spawned by num_threads
    inner_threads = max(1, threads // NUM_THREADS)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ.setdefault(v, str(inner_threads))
    os.environ.setdefault("COTENGRA_PARALLEL", "false")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")

    import torch
    torch.set_num_threads(inner_threads)
    if PARENT_DIR not in sys.path:
        sys.path.insert(0, PARENT_DIR)
    import TDME_Trott as tdme

    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fn_tag = (f"smoke_N{N}_T{T_STUDENT}_L{L_TARGET}_t{T_TIME}"
              f"_g{int(round(GAMMA*100)):03d}_{VARIANT_TAG}")
    prefix = os.path.join(RESULTS_DIR, fn_tag)
    os.environ["LEARNING_TDME_CKPT_DIR"] = RESULTS_DIR

    gamma_vec = [GAMMA] * N
    np.random.seed(SEED); torch.manual_seed(SEED)

    print(f"=== SMOKE RUN ===  cell=(t={T_TIME}, gamma={GAMMA}, variant={VARIANT_TAG})", flush=True)
    print(f"epochs={EPOCHS}  threads={threads}  data-gen workers={NUM_THREADS}  "
          f"inner_threads={inner_threads}", flush=True)

    t0 = time.time()
    model, learning_loss, testing_loss, testing_loss_list = tdme.Learning_TDME_scheduler(
        N=N, MPO_layer=T_STUDENT, model_to_learn_layer=L_TARGET,
        mu=MU, gamma=gamma_vec, J=J, t=T_TIME,
        epochs=EPOCHS, lr=LR, normalized=False,
        max_bd=MAX_BD, max_err=MAX_ERR,
        truncation=TRUNCATION,
        noise_type=NOISE_TYPE,
        use_scheduler=USE_SCHED,
        use_compressed=USE_COMPRESSED,
        num_threads=NUM_THREADS,
        data_dir=None,
        input_pauli_weight=INPUT_PW,
    )
    elapsed = time.time() - t0
    total_epochs_ran = len(learning_loss)
    per_epoch = elapsed / max(total_epochs_ran, 1)

    print(f"\n[SMOKE] elapsed={elapsed:.1f}s  total_epochs={total_epochs_ran}  per_epoch~{per_epoch:.1f}s",
          flush=True)
    print(f"[SMOKE] extrapolated (240 epochs incl. fine-tune): {per_epoch * 240:.0f}s ~ {per_epoch * 240 / 60:.1f}min",
          flush=True)
    print(f"[SMOKE] testing_loss={testing_loss:.6e}", flush=True)

    torch.save(model.state_dict(), f"{prefix}_state_dict.pt")
    np.save(f"{prefix}_params.npy",     model.params.detach().cpu().numpy())
    np.save(f"{prefix}_p_depolar.npy",  model.p_depolar.detach().cpu().numpy())
    np.save(f"{prefix}_p_dephaseX.npy", model.p_dephaseX.detach().cpu().numpy())
    np.save(f"{prefix}_p_dephaseY.npy", model.p_dephaseY.detach().cpu().numpy())
    np.save(f"{prefix}_p_dephaseZ.npy", model.p_dephaseZ.detach().cpu().numpy())
    np.save(f"{prefix}_learning_loss.npy",     np.array(learning_loss))
    np.save(f"{prefix}_testing_loss.npy",      np.array(testing_loss))
    np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))

    meta = dict(
        N=N, T_student=T_STUDENT, L_target=L_TARGET, mu=MU, gamma=GAMMA, gamma_vec=gamma_vec,
        J=J, t=T_TIME, variant=VARIANT_TAG, input_pauli_weight=INPUT_PW,
        lr=LR, epochs=EPOCHS, max_bd=MAX_BD, max_err=MAX_ERR,
        truncation=TRUNCATION, use_compressed=USE_COMPRESSED, use_scheduler=USE_SCHED,
        noise_type=NOISE_TYPE, random_seed=SEED,
        num_threads=NUM_THREADS,
        elapsed_seconds=elapsed, total_epochs_ran=total_epochs_ran,
        per_epoch_seconds=per_epoch,
        extrapolated_full_run_seconds=per_epoch * 240,
        final_testing_loss=float(testing_loss),
    )
    with open(f"{prefix}_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[SMOKE] saved {prefix}_*.{{npy,pt,json}}", flush=True)


if __name__ == "__main__":
    main()
