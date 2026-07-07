"""Save explicit teacher parameter arrays for every (L, p) cell, so the
target channel can be reconstructed from a uniform schema:

  target_params.npy       (L, N, 16) complex SU(4) gate params
  target_p_depolar.npy    (L, N) float64  — set to p uniformly
  target_p_dephaseX.npy   (L, N) float64  — zeros
  target_p_dephaseY.npy   (L, N) float64  — zeros
  target_p_dephaseZ.npy   (L, N) float64  — zeros
"""
import os
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

N = 8
LTARGETS = [3, 4, 5, 6, 7]
DEPOL_LIST = [0.01, 0.05]

for L in LTARGETS:
    for p in DEPOL_LIST:
        # Teacher SU(4) gates — already saved as target_param.pt for any variant;
        # use the w1 copy as the canonical source.
        tp_path = os.path.join(RESULTS_DIR, f"N{N}_T3_L{L}_p{p}_w1_target_param.pt")
        target_param = torch.load(tp_path, map_location="cpu", weights_only=True)
        prefix = os.path.join(RESULTS_DIR, f"N{N}_TEACHER_L{L}_p{p}")
        np.save(f"{prefix}_target_params.npy",
                np.asarray(target_param.cpu().numpy()))
        np.save(f"{prefix}_target_p_depolar.npy",
                np.ones((L, N), dtype=np.float64) * p)
        np.save(f"{prefix}_target_p_dephaseX.npy",
                np.zeros((L, N), dtype=np.float64))
        np.save(f"{prefix}_target_p_dephaseY.npy",
                np.zeros((L, N), dtype=np.float64))
        np.save(f"{prefix}_target_p_dephaseZ.npy",
                np.zeros((L, N), dtype=np.float64))
        print(f"  Saved teacher params for L={L} p={p}  "
              f"(SU(4): {target_param.shape}, depolarizing p={p})")

print("\nDone.")
