"""Generation of (input MPS, target MPS) training/testing pairs."""
import multiprocessing as _mp_module
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from .backend import tensor_network_distance
from .states import (Pauli_MPS_weight_1, Pauli_MPS_weight_2, Pauli_MPS_weight_2_full,
                     Pauli_MPS_combined_1_and_2_full, Pauli_MPS_random,
                     random_pauli_MPS)
from .models import QMLM_output_only
from .tdme import (construct_TDME_unitary, construct_jump_matrices,
                   Pauli_MPS_after_TDME_output_only)


def get_target_tn(N, T, param = None):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    MPS_weight1 = Pauli_MPS_weight_1(N)
    QMLM_MPS_output = QMLM_output_only(N, T, param = param)
    target_tn_list = []
    for input in MPS_weight1:
        MPS_target = QMLM_MPS_output.forward(input.copy())
        #target_tn = MPS_target
        for i, tensor in enumerate(MPS_target):
            tensor = tensor.reindex_({f'input{i}': f'k{i}'})
        target_tn = input.copy() | MPS_target
        target_tn_list.append(target_tn)
    
    return target_tn_list, param



def _process_single_mps(args):
    """Worker function for parallel get_input_and_output_MPS.
    Processes a single Pauli MPS through the quantum channel."""
    # Ensure PyTorch does not spawn extra threads in the worker process
    torch.set_num_threads(1)
    
    idx, mps_input, QMLM_MPS_output, noise_type, truncation, N, T, p_depolar = args
    with torch.no_grad():
        if noise_type == "all":
            MPS_target = QMLM_MPS_output.forward(mps_input.copy())
        elif noise_type == "depolarizing":
            MPS_target = QMLM_MPS_output.forward_depolarizing_only(mps_input.copy(), truncation=truncation)
        else:
            raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
        for i, tensor in enumerate(MPS_target):
            tensor.reindex_({f'input{i}': f'k{i}'})
    return idx, MPS_target



def get_input_and_output_MPS(N, T, param, p_depolar, p_dephaseX=None, p_dephaseY=None, p_dephaseZ=None,
                             truncation=False, max_bd=64, max_err=1E-10, noise_type="all",
                             num_threads=None, input_weight=1):
    """Generate input Pauli MPS and their outputs through a noisy QMLM channel.

    Parameters
    ----------
    input_weight : int, default 1
        Which Pauli-string set to use as inputs.
          1 → all weight-1 Pauli MPS (3N samples).
          2 → all weight-2 same-operator Pauli MPS (3 * N*(N-1)/2 samples).
    num_threads : int or None
        Number of Python threads for the parallel forward passes.
        Defaults to 1 (sequential).  Set > 1 only for standalone use
        outside of ProcessPoolExecutor workers.
    """
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    if p_depolar is None:
        p_depolar = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseX is None:
        p_dephaseX = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseY is None:
        p_dephaseY = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseZ is None:
        p_dephaseZ = torch.zeros((int(T), int(N)), dtype=torch.float64)

    if input_weight == 1:
        MPS_weight1 = Pauli_MPS_weight_1(N)
    elif input_weight == 2:
        MPS_weight1 = Pauli_MPS_weight_2(N)
    elif input_weight in ("2_full", "full_2", "2full"):
        MPS_weight1 = Pauli_MPS_weight_2_full(N)
    elif input_weight in ("combined", "1_and_2full", "1+2full"):
        MPS_weight1 = Pauli_MPS_combined_1_and_2_full(N)
    elif isinstance(input_weight, tuple) and len(input_weight) >= 2 and input_weight[0] == "random":
        n_samples = int(input_weight[1])
        seed = int(input_weight[2]) if len(input_weight) >= 3 else None
        MPS_weight1 = Pauli_MPS_random(N, n_samples, seed=seed)
    else:
        raise ValueError(
            f"input_weight={input_weight!r} not supported; "
            "expected 1, 2, '2_full', 'combined', or ('random', n_samples [, seed])."
        )
    QMLM_MPS_output = QMLM_output_only(
        N, T, param=param, p_depolar=p_depolar,
        p_dephaseX=p_dephaseX, p_dephaseY=p_dephaseY, p_dephaseZ=p_dephaseZ,
        max_bd=max_bd, max_err=max_err,
    )

    total = len(MPS_weight1)
    if num_threads is None:
        num_threads = 1

    if num_threads <= 1:
        # ── Fast sequential path (no thread-pool overhead) ──────────
        target_mps_list = []
        with torch.no_grad():
            for counter, inp in enumerate(MPS_weight1):
                if noise_type == "all":
                    MPS_target = QMLM_MPS_output.forward(inp.copy())
                elif noise_type == "depolarizing":
                    MPS_target = QMLM_MPS_output.forward_depolarizing_only(inp.copy(), truncation=truncation)
                else:
                    raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
                for i, tensor in enumerate(MPS_target):
                    tensor.reindex_({f'input{i}': f'k{i}'})
                target_mps_list.append(MPS_target)
        print(f"N{N}, L{T}, p{p_depolar[0][0]}: processed {total} MPS", flush=True)
    else:
        # ── Process threaded path (for standalone use) ──────────────
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context('spawn')
        task_args = [
            (idx, mps, QMLM_MPS_output, noise_type, truncation, N, T, p_depolar)
            for idx, mps in enumerate(MPS_weight1)
        ]
        target_mps_list = [None] * total
        print(f"N{N}, L{T}, p{p_depolar[0][0]}: processing {total} MPS with {num_threads} processes", flush=True)
        with ProcessPoolExecutor(max_workers=num_threads, mp_context=ctx) as executor:
            futures = {executor.submit(_process_single_mps, arg): arg[0] for arg in task_args}
            for future in as_completed(futures):
                idx, MPS_target = future.result()
                target_mps_list[idx] = MPS_target

    return MPS_weight1, target_mps_list, param



def get_random_input_output_MPS(N, T, param, p_depolar, no_sample=30, p_dephaseX=None, p_dephaseY=None, p_dephaseZ=None,
                                truncation=False, max_bd=64, max_err=1E-10, noise_type="all",
                                num_threads=None):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    if p_depolar is None:
        p_depolar = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseX is None:
        p_dephaseX = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseY is None:
        p_dephaseY = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseZ is None:
        p_dephaseZ = torch.zeros((int(T), int(N)), dtype=torch.float64)
    mps_list = []
    for i in range(no_sample):
        mps_list.append(random_pauli_MPS(N)[0])
    QMLM_MPS_output = QMLM_output_only(
        N, T, param=param, p_depolar=p_depolar,
        p_dephaseX=p_dephaseX, p_dephaseY=p_dephaseY, p_dephaseZ=p_dephaseZ,
        max_bd=max_bd, max_err=max_err,
    )
    
    total = len(mps_list)
    if num_threads is None:
        num_threads = 1
        
    target_mps_list = []
    if num_threads <= 1:
        with torch.no_grad():
            for counter, inp in enumerate(mps_list):
                if noise_type == "all":
                    MPS_target = QMLM_MPS_output.forward(inp.copy())
                elif noise_type == "depolarizing":
                    MPS_target = QMLM_MPS_output.forward_depolarizing_only(inp.copy(), truncation=truncation)
                else:
                    raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
                for i, tensor in enumerate(MPS_target):
                    tensor.reindex_({f'input{i}': f'k{i}'})
                target_mps_list.append(MPS_target)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context('spawn')
        task_args = [
            (idx, mps, QMLM_MPS_output, noise_type, truncation, N, T, p_depolar)
            for idx, mps in enumerate(mps_list)
        ]
        target_mps_list = [None] * total
        with ProcessPoolExecutor(max_workers=num_threads, mp_context=ctx) as executor:
            futures = {executor.submit(_process_single_mps, arg): arg[0] for arg in task_args}
            for future in as_completed(futures):
                idx, MPS_target = future.result()
                target_mps_list[idx] = MPS_target
        
    return mps_list, target_mps_list, param


def get_training_data(N, T, param = None, no_sample = 20):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    mps_list = []
    for i in range(no_sample):
        mps_list.append(random_pauli_MPS(N)[0])
    QMLM_MPS_output = QMLM_output_only(N, T, param = param)
    target_mps_list = []
    for input in mps_list:
        MPS_target = QMLM_MPS_output.forward(input.copy())
        for i, tensor in enumerate(MPS_target):
            tensor = tensor.reindex_({f'input{i}': f'k{i}'})
        target_mps_list.append(MPS_target)
    
    return mps_list, target_mps_list, param


def get_average_loss_weight_tn(tn, tn_target, N, n_times=50, weight = 1):
    total_loss = 0.0
    for _ in range(n_times):
        M_input, weight = random_pauli_MPS(N, weight)
        with torch.no_grad():
            M_realoutput = tn_target | M_input.copy(deep=True)
        M_output = tn | M_input
        M_output.astype_('complex128')
        M_realoutput.astype_('complex128')
        loss = tensor_network_distance(M_output.copy(), M_realoutput.copy())
        total_loss += loss.item()
    average_loss = total_loss / (n_times)
    return average_loss



def get_OSE_output(N, T, param, p_depolar, p_dephaseX=None, p_dephaseY=None, p_dephaseZ=None, truncation = False, max_bd = 64, max_err = 1E-10, noise_type="all"):
    if param is None:
        param = torch.rand(T, N, 16, dtype=torch.float64)+1j*torch.rand(T, N, 16, dtype=torch.float64)
    if p_depolar is None:
        p_depolar = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseX is None:
        p_dephaseX = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseY is None:
        p_dephaseY = torch.zeros((int(T), int(N)), dtype=torch.float64)
    if p_dephaseZ is None:
        p_dephaseZ = torch.zeros((int(T), int(N)), dtype=torch.float64)
    MPS_weight1 = Pauli_MPS_weight_1(N)
    MPS_OSE_in = MPS_weight1[0]
    for i in range(1, len(MPS_weight1)):
        MPS_OSE_in = MPS_OSE_in + MPS_weight1[i]
    MPS_OSE_in = MPS_OSE_in/len(MPS_weight1)
    QMLM_MPS_output = QMLM_output_only(N, T, param = param, p_depolar = p_depolar, p_dephaseX = p_dephaseX, p_dephaseY = p_dephaseY, p_dephaseZ = p_dephaseZ, max_bd = max_bd, max_err = max_err)
    with torch.no_grad():
        if noise_type == "all":
            MPS_target = QMLM_MPS_output.forward(MPS_OSE_in.copy())
        elif noise_type == "depolarizing":
            MPS_target = QMLM_MPS_output.forward_depolarizing_only(MPS_OSE_in.copy(), truncation=truncation)
        else:
            raise ValueError(f"noise_type are all, depolarizing. Not: {noise_type}")
        for i, tensor in enumerate(MPS_target):
            tensor = tensor.reindex_({f'input{i}': f'k{i}'})
    #MPS_target = QMLM_MPS_output.forward_depolarizing_only(MPS_OSE_in.copy(), truncation=truncation)
    return MPS_target, param



def _process_single_mps_tdme(args):
    """Worker for parallel get_input_and_output_MPS_TDME.
    Receives pre-computed unitaries and jump matrices to avoid redundant
    scipy.linalg.expm calls across samples."""
    # Ensure PyTorch does not spawn extra threads in the worker process
    torch.set_num_threads(1)
    
    idx, mps_input, T, r, all_unitary, all_jumping, max_bd, max_err, truncation = args
    with torch.no_grad():
        try:
            MPS_target, truncation_error = Pauli_MPS_after_TDME_output_only(
                mps_input.copy(), T, r=r,
                all_unitary=all_unitary, all_jumping=all_jumping,
                max_bd=max_bd, max_err=max_err, truncation=truncation,
            )
            for i, tensor in enumerate(MPS_target):
                tensor.reindex_({f'input{i}': f'k{i}'})
            return idx, MPS_target, truncation_error, False   # (index, result, truncation_error, skipped)
        except Exception as e:
            return idx, None, None, True



def get_input_and_output_MPS_TDME(N, T, mu, gamma, J=1, t=1,
                                  truncation=False, max_bd=64, max_err=1E-10,
                                  num_threads=None, input_weight=1):
    """Generate input Pauli MPS and their TDME outputs.

    Parameters
    ----------
    input_weight : int, default 1
        Which Pauli-string set to use as inputs.
          1 → all weight-1 Pauli MPS (3N samples).
          2 → all weight-2 same-operator Pauli MPS (3 * N*(N-1)/2 samples).
    num_threads : int or None
        Number of Python threads for parallel forward passes.
        Defaults to 1 (sequential).  Set > 1 only for standalone use
        outside of ProcessPoolExecutor workers.
    """
    if input_weight == 1:
        MPS_weight1 = Pauli_MPS_weight_1(N)
    elif input_weight == 2:
        MPS_weight1 = Pauli_MPS_weight_2(N)
    elif input_weight in ("2_full", "full_2", "2full"):
        MPS_weight1 = Pauli_MPS_weight_2_full(N)
    elif input_weight in ("combined", "1_and_2full", "1+2full"):
        MPS_weight1 = Pauli_MPS_combined_1_and_2_full(N)
    elif isinstance(input_weight, tuple) and len(input_weight) >= 2 and input_weight[0] == "random":
        n_samples = int(input_weight[1])
        seed = int(input_weight[2]) if len(input_weight) >= 3 else None
        MPS_weight1 = Pauli_MPS_random(N, n_samples, seed=seed)
    else:
        raise ValueError(
            f"input_weight={input_weight!r} not supported; "
            "expected 1, 2, '2_full', 'combined', or ('random', n_samples [, seed])."
        )
    total = len(MPS_weight1)

    # Pre-compute unitaries and jump matrices ONCE (they are identical
    # for every input MPS — they depend only on model parameters).
    r = t / T
    all_unitary = construct_TDME_unitary(N, T, r=r, mu=mu, J=J)
    all_jumping = construct_jump_matrices(N, gamma, r=r)

    # Seed before any branch so the final `return ... truncation_error`
    # works even if every sample is skipped.  Previously this was only
    # assigned inside the success path of each branch, so an all-failed
    # run raised UnboundLocalError.
    truncation_error = 0.0

    if num_threads is None:
        num_threads = 1

    if num_threads <= 1:
        # ── Fast sequential path (no thread-pool overhead) ──────────
        target_mps_list = []
        skipped_MPS = []
        with torch.no_grad():
            for counter, inp in enumerate(MPS_weight1):
                print(f"Processing MPS {counter+1}/{total}", flush=True)
                try:
                    MPS_target, truncation_error = Pauli_MPS_after_TDME_output_only(
                        inp.copy(), T, r=r,
                        all_unitary=all_unitary, all_jumping=all_jumping,
                        max_bd=max_bd, max_err=max_err, truncation=truncation,
                    )
                    for i, tensor in enumerate(MPS_target):
                        tensor.reindex_({f'input{i}': f'k{i}'})
                    target_mps_list.append(MPS_target)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"Failed to process MPS {counter+1} due to truncation, skipping...")
                    skipped_MPS.append(inp)
        MPS_weight1 = [mps for mps in MPS_weight1 if mps not in skipped_MPS]
        print(len(MPS_weight1), len(target_mps_list), flush=True)
    else:
        # ── Process or Threaded path ────────────────────────────────
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        
        ctx = mp.get_context('spawn')
        task_args = [
            (idx, mps, T, r, all_unitary, all_jumping, max_bd, max_err, truncation)
            for idx, mps in enumerate(MPS_weight1)
        ]
        results = [None] * total
        skipped_indices = set()
        print(f"Processing {total} MPS with {num_threads} workers", flush=True)

        with ProcessPoolExecutor(max_workers=num_threads, mp_context=ctx) as executor:
            futures = {
                executor.submit(_process_single_mps_tdme, arg): arg[0]
                for arg in task_args
            }
            for future in as_completed(futures):
                idx, MPS_target, trunc_err, skipped = future.result()
                if skipped:
                    print(f"Failed to process MPS {idx+1} due to truncation, skipping...")
                    skipped_indices.add(idx)
                else:
                    results[idx] = MPS_target
                    truncation_error = trunc_err

        # Filter out skipped entries, preserving order
        target_mps_list = [r for i, r in enumerate(results) if i not in skipped_indices]
        MPS_weight1 = [m for i, m in enumerate(MPS_weight1) if i not in skipped_indices]
        print(len(MPS_weight1), len(target_mps_list), flush=True)

    return MPS_weight1, target_mps_list, truncation_error

