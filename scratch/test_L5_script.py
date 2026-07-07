import TDME_Trott as tdme
import os
import torch
import numpy as np

torch.set_num_threads(1)
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

print("Starting single L=5 evaluation with 16 threads...")
(model, learning_loss, target_param, p_depolar,
 testing_loss, testing_loss_list, model_param, model_p_depolar,
 model_p_dephaseX, model_p_dephaseY, model_p_dephaseZ
) = tdme.Learning_MPO_scheduler(
    N=30, T=3, L=5,
    depolarizing_strength=0.02,
    epochs=1, lr=0.05,
    normalized=False,
    truncation=True,
    noise_type="depolarizing",
    num_threads=16,
    use_compressed=True,
    max_bd=64,
)
print("Finished!")
