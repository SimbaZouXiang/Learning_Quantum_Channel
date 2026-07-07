import torch
import sys
import os

# Set a safe Numba cache directory before anything else imports numba/quimb
os.environ["NUMBA_CACHE_DIR"] = "/tmp/numba_cache_debugging"
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)

import numpy as np
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed   

target_time = 0.2
gamma_name  = 0

N                    = 10
T                    = 3
model_to_learn_layer = 30
mu                   = 1
J                    = 1
epochs               = 200
lr                   = 0.05
normalized           = False
truncation           = True
noise_type           = "dephasing"
use_scheduler        = False
use_compressed       = False
max_bd               = 64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# make sure we can import project modules
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
import TDME_Trott as tdme                       # noqa: E402

gamma = [gamma_name * 0.01] * N

data_dir = os.path.join(SCRIPT_DIR, "Learning_data", f"N{N}_T{T}_mu{float(mu)}_gamma{round(gamma[0], 2)}_t{target_time}")

threads = 1  # setting a default value for threads since it's not defined
tag = f"time={target_time}, gamma_name={gamma_name}"
print(f"[PID {os.getpid()}] START  {tag}  (threads={threads})",
        flush=True)

model, learning_loss, testing_loss, testing_loss_list = \
    tdme.Learning_TDME_scheduler(
        N, T,
        model_to_learn_layer=model_to_learn_layer,
        mu=mu, gamma=gamma, J=J, t=target_time,
        epochs=epochs, lr=lr,
        normalized=normalized,
        truncation=truncation,
        noise_type=noise_type,
        use_scheduler=use_scheduler,
        use_compressed=use_compressed,
        max_bd=max_bd,
        data_dir=data_dir,
    )

# -- persist results (one set of files per grid point) --
prefix = f"TDME_N{N}_T{T}_time{target_time}_gamma{gamma_name:03d}"
np.save(f"{prefix}_learning_loss.npy",     np.array(learning_loss))
np.save(f"{prefix}_testing_loss.npy",       np.array(testing_loss))
np.save(f"{prefix}_testing_loss_list.npy", np.array(testing_loss_list))

print(f"[PID {os.getpid()}] DONE   {tag}  testing_loss={testing_loss}",
        flush=True)