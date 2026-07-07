"""Training and evaluation loops for all teacher/student setups."""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .backend import tensor_network_distance
from .states import random_pauli_MPS
from .models import QMLM, QMLM_output_only, apply_mpo_to_mps_compressed
from .evolve import _strip_size1_outer_bonds
from .bath import (get_input_and_output_MPS_with_bath,
                   get_random_input_output_MPS_with_bath)
from .tdme import (TDME, construct_TDME_unitary, construct_jump_matrices,
                   construct_TDME_unitary_first_order,
                   construct_jump_matrices_first_order,
                   Pauli_MPS_after_TDME_output_only,
                   Pauli_MPS_after_TDME_output_only_first_order)
from .datagen import (get_input_and_output_MPS, get_random_input_output_MPS,
                      get_input_and_output_MPS_TDME)


def _finite_grad_step(model, optimizer, where=""):
    """Apply optimizer.step() only if every parameter gradient is finite.

    Backward through the compressed forward path can produce NaN gradients
    (SVD backward has 1/(s_i^2 - s_j^2) terms that blow up on near-degenerate
    singular values). One such step permanently poisons the parameters with
    NaN, after which every subsequent forward — including the final MPO build
    and the whole testing phase — crashes. Skipping the poisoned update keeps
    the model finite: training merely stalls for that epoch and continues.

    Returns True if the step was applied, False if it was skipped.
    """
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"WARNING: non-finite gradients detected{' at ' + where if where else ''}; "
                  f"skipping this optimizer step to keep parameters finite.", flush=True)
            optimizer.zero_grad(set_to_none=True)
            return False
    optimizer.step()
    return True


def Learning_MPO(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, depolarizing_strength=0.2,
    epochs=100, lr=0.01, normalized = False, truncation = False, max_bd=64, max_err=1e-8, noise_type = "all",
    use_compressed=False
):
    times = 3
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )                                               # teacher params [attached_file:1]
    else:
        param = param_list                              # if provided externally [attached_file:1]

    p_depolar_MPO = torch.ones((MPO_layer, N), dtype=torch.float64) * depolarizing_strength*0.5
    p_depolar = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * depolarizing_strength 
    model = QMLM(N, MPO_layer, p_depolar = p_depolar_MPO)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=50, min_lr=0.001)
    learning_loss = []
    clamp_value = min(depolarizing_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)

    MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar, max_bd=max_bd, max_err=max_err, truncation=truncation, noise_type=noise_type
    )
    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)  # freeze input MPS tensors
    for target_tensor in target_mps_list:
        for tensor in target_tensor.tensors:
            tensor.data.requires_grad_(False)

    for epoch in range(epochs):
        losses = []
        if use_compressed:
            # Layer-by-layer application with truncation → output is a proper MPS
            # tensor_network_distance between two MPS is O(N D^3) instead of exponential
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()                      
        _finite_grad_step(model, optimizer)
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value+0.01)

    for epoch in range(epochs//5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()                      
        _finite_grad_step(model, optimizer)
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, min(2*clamp_value+0.01, 1.0))

    # final MPO after training
    final_mpo_fit = model.get_MPO(noise_type=noise_type)                     # rebuild MPO once more if you need it [attached_file:1]
    model_to_learn = QMLM(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar
    )
    mpo_target = model_to_learn.get_MPO()
    testing_loss = 0.0
    for i in range(200):
        print(f"Testing sample {i+1}/200")
        Random_MPS, weight = random_pauli_MPS(N)
        if use_compressed:
            M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type=noise_type)
            for j, tensor in enumerate(M_out):
                tensor.reindex_({f'input{j}': f'k{j}'})
            M_target = mpo_target.astype("complex128") | Random_MPS.astype("complex128")
            loss = tensor_network_distance(M_out.astype("complex128"), M_target.astype("complex128")).item()
        else:
            loss = tensor_network_distance(final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"), mpo_target.astype("complex128") | Random_MPS.astype("complex128")).item()
        testing_loss += loss
    testing_loss /= 200
    print("Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss





def _compute_single_loss_compressed(sample_idx, output_mps_list, target_mps_list, normalized):
    """Compute loss for one sample (compressed path). Thread-safe."""
    return tensor_network_distance(
        output_mps_list[sample_idx].astype("complex128"),
        target_mps_list[sample_idx].astype("complex128"),
        normalized=normalized,
    )


def _compute_single_loss_mpo(sample_idx, mpo_fit, MPS_weight1, target_mps_list, normalized):
    """Compute loss for one sample (MPO path). Thread-safe."""
    M_input = mpo_fit | MPS_weight1[sample_idx]
    return tensor_network_distance(
        M_input.astype("complex128"),
        target_mps_list[sample_idx].astype("complex128"),
        normalized=normalized,
    )


def _compute_losses_parallel(
    model, MPS_weight1, target_mps_list, normalized,
    use_compressed, noise_type, max_bd, max_err,
    num_threads=1,
):
    """Compute per-sample losses, optionally in parallel using threads.

    Uses ThreadPoolExecutor so the autograd graph is preserved (shared
    memory) and `loss.backward()` works after `torch.stack(losses).mean()`.

    Parameters
    ----------
    num_threads : int
        Number of threads.  1 = sequential (no pool overhead).
    """
    n_samples = len(MPS_weight1)

    if use_compressed:
        output_mps_list = apply_mpo_to_mps_compressed(
            model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
        )
        losses = [
            _compute_single_loss_compressed(i, output_mps_list, target_mps_list, normalized)
            for i in range(n_samples)
        ]
    else:
        mpo_fit = model.get_MPO(noise_type=noise_type)
        losses = [
            _compute_single_loss_mpo(i, mpo_fit, MPS_weight1, target_mps_list, normalized)
            for i in range(n_samples)
        ]

    return losses



def Learning_MPO_scheduler(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, depolarizing_strength=0.2,
    epochs=100, lr=0.01, normalized = False, max_bd=32, max_err=1E-6, truncation = False, noise_type = "all",
    use_compressed=False, num_threads=None, weight_1_pauli_strings = True,
    input_pauli_weight=None,
    fine_tune_epochs=None,
):
    # input_pauli_weight selects the Pauli-basis training set when set:
    #   1 → 3N weight-1 Pauli MPS (X_i, Y_i, Z_i for i in 0..N-1)
    #   2 → 3 * N*(N-1)/2 weight-2 same-operator MPS (X_iX_j, Y_iY_j, Z_iZ_j, i<j)
    # When None (default), falls back to the legacy `weight_1_pauli_strings`
    # boolean: True → weight-1 basis (3N samples), False → random Pauli inputs
    # of weight ~Unif[1,N] with `no_sample=3*N`.
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )
    else:
        param = param_list

    p_depolar_MPO = torch.ones((MPO_layer, N), dtype=torch.float64) * depolarizing_strength * 0.5
    p_depolar = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * depolarizing_strength
    if MPO_layer == model_to_learn_layer:
        print("MPO_layer equals model_to_learn_layer")
        model = QMLM(N, MPO_layer, param=param*1.05, p_depolar=p_depolar_MPO)
    else:
        print("MPO_layer does not equal model_to_learn_layer")
        model = QMLM(N, MPO_layer, p_depolar=p_depolar_MPO)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # we want to reduce LR when loss stops decreasing
        factor=0.5,      # shrink LR by a factor of 2
        patience=10,     # epochs to wait with no improvement
        threshold=5e-4,
        min_lr=1e-5,     # do not go below this LR
    )

    learning_loss = []
    clamp_value = min(depolarizing_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)
    _is_random_tuple = (
        isinstance(input_pauli_weight, tuple)
        and len(input_pauli_weight) >= 2
        and input_pauli_weight[0] == "random"
    )
    if input_pauli_weight in (1, 2, "2_full", "full_2", "2full",
                               "combined", "1_and_2full", "1+2full") or _is_random_tuple:
        MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
            N, model_to_learn_layer, param=param, p_depolar=p_depolar,
            max_bd=max_bd, max_err=max_err, truncation=truncation,
            noise_type="depolarizing", num_threads=num_threads,
            input_weight=input_pauli_weight,
        )
    elif weight_1_pauli_strings:
        MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
            N, model_to_learn_layer, param=param, p_depolar=p_depolar,
            max_bd=max_bd, max_err=max_err, truncation=truncation,
            noise_type="depolarizing", num_threads=num_threads,
        )
    else:
        MPS_weight1, target_mps_list, param = get_random_input_output_MPS(
            N, model_to_learn_layer, param=param, p_depolar=p_depolar, no_sample=3*N,
        )
    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)
    for target_tensor in target_mps_list:
        for tensor in target_tensor.tensors:
            tensor.data.requires_grad_(False)
    # Cast frozen tensors to complex128 once; drop redundant per-sample astype calls below.
    MPS_weight1 = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    # main training loop
    for epoch in range(epochs):
        #print(f"Epoch {epoch+1}/{epochs}")
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                loss = tensor_network_distance(
                    output_mps_list[sample_idx],
                    target_mps_list[sample_idx],
                    normalized=normalized
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    target_mps_list[sample_idx],
                    normalized=normalized
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad(set_to_none=True)
        loss_this_epoch.backward()
        _finite_grad_step(model, optimizer)
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value + 0.01)

        # step scheduler with the current epoch loss (as metric)
        scheduler.step(loss_this_epoch.detach())
        if epoch % 1 == 0:
            print(f"N{N}, T{MPO_layer}, L{model_to_learn_layer}, p{depolarizing_strength}, Epoch {epoch}, Loss:", loss_this_epoch.item(), end = ", ")
            print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'])

    # fine-tuning loop. Default = epochs // 5 (gives 40 at epochs=200);
    # caller can override via fine_tune_epochs kwarg.
    _ft_epochs = fine_tune_epochs if fine_tune_epochs is not None else epochs // 5
    for epoch in range(_ft_epochs):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
            )
            for sample_idx in range(len(MPS_weight1)):
                loss = tensor_network_distance(
                    output_mps_list[sample_idx],
                    target_mps_list[sample_idx],
                    normalized=normalized
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    target_mps_list[sample_idx],
                    normalized=normalized
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        #print(f"Loss at fine-tune epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad(set_to_none=True)
        loss_this_epoch.backward()
        _finite_grad_step(model, optimizer)
        model.p_depolar.data = torch.clamp(
            model.p_depolar.data,
            0.0,
            min(2 * clamp_value + 0.01, 1.0)
        )

        # also update scheduler here (still monitoring the same loss)
        scheduler.step(loss_this_epoch.detach())
        if epoch % 1 == 0:
            print(f"Fine-tune, N{N}, T{MPO_layer}, L{model_to_learn_layer}, p{depolarizing_strength}, Epoch {epoch}, Loss:", loss_this_epoch.item(), end = ", ")
            print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'])

    # final MPO after training
    final_mpo_fit = model.get_MPO(noise_type=noise_type)
    testing_loss_list = []
    QMLM_MPS_output = QMLM_output_only(N, model_to_learn_layer, param = param, p_depolar = p_depolar, max_bd = max_bd, max_err = max_err)
    num_samples = 100

    def _test_single_sample(j):
        """Evaluate one random test sample. No gradients needed.
        Returns None if the sample fails (numerical blow-up in either forward)
        so one bad sample cannot crash the run after training succeeded."""
        with torch.no_grad():
            Random_MPS, weight = random_pauli_MPS(N)
            try:
                MPS_target = QMLM_MPS_output.forward_depolarizing_only(Random_MPS.copy(), truncation=truncation)
                for i, tensor in enumerate(MPS_target):
                    tensor = tensor.reindex_({f'input{i}': f'k{i}'})
                if use_compressed:
                    M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type=noise_type)
                    for i, tensor in enumerate(M_out):
                        tensor.reindex_({f'input{i}': f'k{i}'})
                    loss = tensor_network_distance(
                        M_out.astype("complex128"),
                        MPS_target.astype("complex128")
                    ).item()
                else:
                    loss = tensor_network_distance(
                        final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
                        MPS_target.astype("complex128")
                    ).item()
            except Exception as e:
                print(f"Testing sample {j+1}/{num_samples} failed "
                      f"({type(e).__name__}); skipping.", flush=True)
                return None
        return loss

    test_threads = max(1, num_threads if num_threads else 1)
    # We use a sequential loop here. ThreadPoolExecutor is roughly 2x slower
    # due to Python's Global Interpreter Lock (GIL).
    raw_results = [_test_single_sample(j) for j in range(num_samples)]
    testing_loss_list = [r for r in raw_results if r is not None]
    n_skipped = num_samples - len(testing_loss_list)
    if n_skipped > 0:
        print(f"Skipped {n_skipped} failed testing samples.", flush=True)
    if testing_loss_list:
        testing_loss = sum(testing_loss_list) / len(testing_loss_list)
    else:
        testing_loss = float("nan")
        print("WARNING: every testing sample failed — testing_loss set to NaN.",
              flush=True)
    print(f"N{N}, T{MPO_layer}, L{model_to_learn_layer}, p{depolarizing_strength}, Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss, testing_loss_list, model.params.detach().numpy(), model.p_depolar.detach().numpy(), model.p_dephaseX.detach().numpy(), model.p_dephaseY.detach().numpy(), model.p_dephaseZ.detach().numpy()



def Learning_MPO_dephasing_noise_only(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, dephasingX_strength=0.2, dephasingY_strength=0.2, dephasingZ_strength=0.2,
    epochs=100, lr=0.01, normalized = False, max_bd=64, max_err=1e-8,
    use_compressed=False
):
    times = 3
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )                                               # teacher params [attached_file:1]
    else:
        param = param_list                              # if provided externally [attached_file:1]

    p_depolar_MPO = torch.zeros((MPO_layer, N), dtype=torch.float64)
    p_depolar = torch.zeros((model_to_learn_layer, N), dtype=torch.float64) 
    p_dephaseX = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingX_strength
    p_dephaseY = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingY_strength
    p_dephaseZ = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingZ_strength
    model = QMLM(N, MPO_layer)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=50, min_lr=0.001)
    learning_loss = []
    clamp_value_X = min(dephasingX_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
    clamp_value_Y = min(dephasingY_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
    clamp_value_Z = min(dephasingZ_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)

    MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar, p_dephaseX = p_dephaseX, p_dephaseY = p_dephaseY, p_dephaseZ = p_dephaseZ
    )
    for MPS_weight1_tensor in MPS_weight1:
        #MPS_weight1_tensor = MPS_weight1_tensor.astype("complex64")
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)  # freeze input MPS tensors


    for epoch in range(epochs):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            #indices_this_epoch = np.random.permutation(np.arange(len(MPS_weight1)))[:min(int(len(MPS_weight1)*5*epoch/epochs)+1, len(MPS_weight1))]
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()
        _finite_grad_step(model, optimizer)
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, clamp_value_X+1e-6)
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, clamp_value_Y+1e-6)
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, clamp_value_Z+1e-6)

    for epoch in range(epochs//5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(output_mps_list[sample_idx].astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            #indices_this_epoch = np.random.permutation(np.arange(len(MPS_weight1)))[:min(int(len(MPS_weight1)*5*epoch/epochs)+1, len(MPS_weight1))]
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx] 
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(M_input.astype("complex128"), M_check.astype("complex128"), normalized=normalized)
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())
        optimizer.zero_grad()
        loss_this_epoch.backward()                      
        _finite_grad_step(model, optimizer)
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, min(2*clamp_value_X+1e-6, 1.0))
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, min(2*clamp_value_Y+1e-6, 1.0))
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, min(2*clamp_value_Z+1e-6, 1.0))
    #scheduler.step(loss_this_epoch.item())
    #optimizer = optim.Adam(model.parameters(), lr=lr)


    # final MPO after training
    final_mpo_fit = model.get_MPO()                     # rebuild MPO once more if you need it [attached_file:1]
    model_to_learn = QMLM(
        N, model_to_learn_layer, param=param, p_depolar=p_depolar, p_dephaseX = p_dephaseX, p_dephaseY = p_dephaseY, p_dephaseZ = p_dephaseZ
    )
    mpo_target = model_to_learn.get_MPO()
    testing_loss_list = []
    testing_loss = 0.0
    for i in range(200):
        Random_MPS, weight = random_pauli_MPS(N)
        if use_compressed:
            M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type="all")
            for j, tensor in enumerate(M_out):
                tensor.reindex_({f'input{j}': f'k{j}'})
            M_target = mpo_target.astype("complex128") | Random_MPS.astype("complex128")
            loss = tensor_network_distance(M_out.astype("complex128"), M_target.astype("complex128")).item()
        else:
            loss = tensor_network_distance(final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"), mpo_target.astype("complex128") | Random_MPS.astype("complex128")).item()
        testing_loss_list.append(loss)
        testing_loss += loss
    testing_loss /= 200
    print("Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss, testing_loss_list, model.params, model.p_depolar, model.p_dephaseX, model.p_dephaseY, model.p_dephaseZ




def Learning_MPO_dephasing_noise_only_scheduler(
    N, MPO_layer, model_to_learn_layer,
    param_list=None, dephasingX_strength=0.2, dephasingY_strength=0.2, dephasingZ_strength=0.2,
    epochs=100, lr=0.01, normalized=False, max_bd=64, max_err=1e-8,
    use_compressed=False
):
    times = 3
    if param_list is None:
        param = (
            torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
            + 1j * torch.rand(model_to_learn_layer, N, 16, dtype=torch.float64)
        )
    else:
        param = param_list

    p_depolar_MPO = torch.zeros((MPO_layer, N), dtype=torch.float64)
    p_depolar = torch.zeros((model_to_learn_layer, N), dtype=torch.float64)
    p_dephaseX = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingX_strength
    p_dephaseY = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingY_strength
    p_dephaseZ = torch.ones((model_to_learn_layer, N), dtype=torch.float64) * dephasingZ_strength

    model = QMLM(N, MPO_layer)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)

    # Add ReduceLROnPlateau scheduler (monitoring the training loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,        # shrink LR by 2x when plateauing
        patience=10,       # epochs with no improvement before reducing LR
        threshold=5e-4,
        min_lr=1e-7,
    )

    learning_loss = []
    clamp_value_X = min(dephasingX_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)
    clamp_value_Y = min(dephasingY_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)
    clamp_value_Z = min(dephasingZ_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)

    MPS_weight1, target_mps_list, param = get_input_and_output_MPS(
        N, model_to_learn_layer, param=param,
        p_depolar=p_depolar,
        p_dephaseX=p_dephaseX, p_dephaseY=p_dephaseY, p_dephaseZ=p_dephaseZ
    )
    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)

    # ---------- first training phase ----------
    for epoch in range(epochs):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    output_mps_list[sample_idx].astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad()
        loss_this_epoch.backward()
        _finite_grad_step(model, optimizer)

        # clamp noise parameters
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, clamp_value_X + 1e-6)
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, clamp_value_Y + 1e-6)
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, clamp_value_Z + 1e-6)

        # update LR based on epoch loss
        scheduler.step(loss_this_epoch)
        if epoch % 10 == 0:
            print(f"Learning Rate at epoch {epoch}: {optimizer.param_groups[0]['lr']}")

    # ---------- second training phase ----------
    for epoch in range(epochs // 5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type="dephasing"
            )
            for sample_idx in range(len(MPS_weight1)):
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    output_mps_list[sample_idx].astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO_without_depolarizing_noise()
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                M_check = target_mps_list[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    M_check.astype("complex128"),
                    normalized=normalized,
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        print(f"Loss (phase 2) at epoch {epoch}:", loss_this_epoch.item())
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad()
        loss_this_epoch.backward()
        _finite_grad_step(model, optimizer)

        model.p_dephaseX.data = torch.clamp(
            model.p_dephaseX.data, 0.0, min(2 * clamp_value_X + 1e-6, 1.0)
        )
        model.p_dephaseY.data = torch.clamp(
            model.p_dephaseY.data, 0.0, min(2 * clamp_value_Y + 1e-6, 1.0)
        )
        model.p_dephaseZ.data = torch.clamp(
            model.p_dephaseZ.data, 0.0, min(2 * clamp_value_Z + 1e-6, 1.0)
        )

        # keep scheduling in phase 2 as well
        scheduler.step(loss_this_epoch)
        if epoch % 10 == 0:
            print(f"Learning Rate at epoch {epoch}: {optimizer.param_groups[0]['lr']}")

    # final MPO after training
    final_mpo_fit = model.get_MPO()
    model_to_learn = QMLM(
        N, model_to_learn_layer,
        param=param,
        p_depolar=p_depolar,
        p_dephaseX=p_dephaseX, p_dephaseY=p_dephaseY, p_dephaseZ=p_dephaseZ
    )
    mpo_target = model_to_learn.get_MPO()
    testing_loss = 0.0
    for i in range(200):
        Random_MPS, weight = random_pauli_MPS(N)
        if use_compressed:
            M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type="all")
            for j, tensor in enumerate(M_out):
                tensor.reindex_({f'input{j}': f'k{j}'})
            M_target = mpo_target.astype("complex128") | Random_MPS.astype("complex128")
            loss = tensor_network_distance(M_out.astype("complex128"), M_target.astype("complex128")).item()
        else:
            loss = tensor_network_distance(
                final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
                mpo_target.astype("complex128") | Random_MPS.astype("complex128"),
            ).item()
        testing_loss += loss
    testing_loss /= 200
    print("Testing loss:", testing_loss)
    return model, learning_loss, param, p_depolar, testing_loss



_worker_model_for_test = None

_worker_mpo_for_test = None


def _init_worker_learning_test(N_in, MPO_layer_in, model_state_dict_in, noise_type_in, use_compressed_in):
    global _worker_model_for_test
    global _worker_mpo_for_test
    # Pin BLAS/OMP to 1 thread per worker. torch.set_num_threads only affects
    # torch's own intra-op pool; OMP/MKL/OpenBLAS threads are inherited from
    # the parent env (which sets them to e.g. 48) and will oversubscribe each
    # worker unless we reset them here.
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import torch
    torch.set_num_threads(1)
    
    _worker_model_for_test = QMLM(N_in, MPO_layer_in)
    _worker_model_for_test.load_state_dict(model_state_dict_in)
    
    if not use_compressed_in:
        _worker_mpo_for_test = _worker_model_for_test.get_MPO(noise_type=noise_type_in)



def _process_single_sample_learning_test_tdme(args):
    """
    Worker function to process a single test sample during Learning_TDME_scheduler.
    """
    import torch
    import numpy as np
    
    j, N, model_to_learn_layer, r_target, target_all_unitary, target_all_jumping, max_bd, max_err, truncation, use_compressed, noise_type, num_samples = args
    
    Random_MPS, weight = random_pauli_MPS(N)
    
    try:
        with torch.no_grad():
            MPS_target, _ = Pauli_MPS_after_TDME_output_only(
                Random_MPS.copy(), model_to_learn_layer, r=r_target,
                all_unitary=target_all_unitary,
                all_jumping=target_all_jumping,
                max_bd=max_bd, max_err=max_err, truncation=truncation,
            )
            for i, tensor in enumerate(MPS_target):
                tensor.reindex_({f'input{i}': f'k{i}'})
                
            if use_compressed:
                M_out = _worker_model_for_test.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type=noise_type)
                for i, tensor in enumerate(M_out):
                    tensor.reindex_({f'input{i}': f'k{i}'})
                loss = tensor_network_distance(
                    M_out.astype("complex128"),
                    MPS_target.astype("complex128")
                ).item()
            else:
                loss = tensor_network_distance(
                    _worker_mpo_for_test.astype("complex128") | Random_MPS.astype("complex128"),
                    MPS_target.astype("complex128")
                ).item()
                # print(f"Testing sample {j+1}/{num_samples}, loss: {loss}")
        return j, loss
    except Exception as e:
        return j, None


def Learning_TDME_scheduler(
    N, MPO_layer, model_to_learn_layer, mu, gamma, J=1, t = 1,
    epochs=100, lr=0.01, normalized = False, max_bd=128, max_err=1E-8, truncation = False, noise_type = "all", use_scheduler = True,
    use_compressed=True, num_threads=None, data_dir=None,
    input_pauli_weight=1,
):
    # input_pauli_weight selects the training-set Pauli basis:
    #   1 → 3N weight-1 Pauli MPS (legacy default; X_i, Y_i, Z_i for i in 0..N-1)
    #   2 → 3 * N*(N-1)/2 weight-2 same-operator MPS (X_iX_j, Y_iY_j, Z_iZ_j
    #       for i < j). A larger, more informative set whose generalisation
    #       behaviour we want to compare to the weight-1 baseline.
    #   "2_full" → 9 * N*(N-1)/2 full weight-2 Pauli basis (all op1, op2 in
    #       {X,Y,Z}^2 on each pair i < j). 3x more inputs than the same-op set.
    import os
    import json
    # load_mps is only needed when reading a pre-generated dataset from disk;
    # defer the import to the branch that actually uses it.

    model = QMLM(N, MPO_layer)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',      # we want to reduce LR when loss stops decreasing
        factor=0.5,      # shrink LR by a factor of 2
        patience=10,     # epochs to wait with no improvement
        threshold=5e-4,
        min_lr=1e-5,     # do not go below this LR 
    )
    depolarizing_strength = gamma[0] if isinstance(gamma, (list, tuple, np.ndarray)) else gamma
    learning_loss = []
    clamp_value = min(depolarizing_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)

    # Load from disk if provided, otherwise generate.
    # Disk caches were generated for input_pauli_weight=1; only use them in that case.
    if data_dir is not None and os.path.isdir(data_dir) and input_pauli_weight == 1:
        from .io import load_mps
        print(f"Loading cached MPS datasets from {data_dir}...", flush=True)
        with open(os.path.join(data_dir, "params.json"), "r") as f:
            params = json.load(f)
            num_samples = params["num_samples"]

        MPS_weight1 = [load_mps(os.path.join(data_dir, f"input_MPS_{i}.npz")) for i in range(num_samples)]
        target_mps_list = [load_mps(os.path.join(data_dir, f"target_MPS_{i}.npz"), input = False) for i in range(num_samples)]
    else:
        print(f"Generating MPS datasets locally (input_pauli_weight={input_pauli_weight})...",
              flush=True)
        MPS_weight1, target_mps_list, _ = get_input_and_output_MPS_TDME(
            N, model_to_learn_layer, mu=mu, gamma=gamma, J=J, t=t,
            max_bd=max_bd, max_err=max_err, truncation=truncation,
            num_threads=num_threads, input_weight=input_pauli_weight,
        )
        # Fallback: truncation=True currently hits a known IndexError in
        # Pauli_MPS_after_TDME_output_only that can skip every sample.  If
        # that happened, regenerate with truncation=False so the grid point
        # still makes progress (slower, larger bond dim, but correct).
        if truncation and (len(MPS_weight1) == 0 or len(target_mps_list) == 0):
            print("All samples skipped with truncation=True — retrying with "
                  "truncation=False (slower but reliable).", flush=True)
            MPS_weight1, target_mps_list, _ = get_input_and_output_MPS_TDME(
                N, model_to_learn_layer, mu=mu, gamma=gamma, J=J, t=t,
                max_bd=max_bd, max_err=max_err, truncation=False,
                num_threads=num_threads, input_weight=input_pauli_weight,
            )

    # Fail loudly if the dataset is empty — otherwise the per-sample
    # backward loops divide by zero and the worker crashes with an
    # uninformative "float division by zero", which loky then surfaces
    # as the opaque grid-point failure message.
    if len(MPS_weight1) == 0 or len(target_mps_list) == 0:
        raise RuntimeError(
            f"No training samples available for N={N}, T={MPO_layer}, "
            f"model_to_learn_layer={model_to_learn_layer}, t={t}, "
            f"gamma={gamma[0] if hasattr(gamma, '__len__') else gamma}. "
            f"data_dir={data_dir}.  Even the truncation=False fallback "
            f"produced no samples — skipping this grid point."
        )

    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)
    for target_tensor in target_mps_list:
        for tensor in target_tensor.tensors:
            tensor.data.requires_grad_(False)
    # Cast frozen tensors once. The MPO-building path builds a complex MPO via
    # model.get_MPO and has no autograd-through-SVD concern, so complex128 is
    # fine. The compressed path runs real transfer matrices through QR/SVD
    # inside the forward pass — and torch's svd_backward is ill-defined for
    # complex inputs — so it must stay float64. Pauli-basis coefficients are
    # real anyway, so keeping float64 here is also physically correct.
    if use_compressed:
        MPS_weight1 = [m.astype("float64") for m in MPS_weight1]
        target_mps_list = [m.astype("float64") for m in target_mps_list]
    else:
        MPS_weight1 = [m.astype("complex128") for m in MPS_weight1]
        target_mps_list = [m.astype("complex128") for m in target_mps_list]

    # main training loop
    if noise_type == "depolarizing":
        for epoch in range(epochs):
            losses = []
            if use_compressed:
                output_mps_list = apply_mpo_to_mps_compressed(
                    model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
                )
                for sample_idx in range(len(MPS_weight1)):
                    loss = tensor_network_distance(
                        output_mps_list[sample_idx],
                        target_mps_list[sample_idx],
                        normalized=normalized
                    )
                    losses.append(loss)
            else:
                mpo_fit = model.get_MPO(noise_type=noise_type).astype("complex128")
                for sample_idx in range(len(MPS_weight1)):
                    # mpo_fit and MPS_weight1 are both complex128 already;
                    # skip the redundant per-sample .astype copies that were
                    # burning tens of thousands of allocations per epoch.
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    loss = tensor_network_distance(
                        M_input,
                        target_mps_list[sample_idx],
                        normalized=normalized
                    )
                    losses.append(loss)

            loss_this_epoch = torch.stack(losses).mean()
            learning_loss.append(loss_this_epoch.item())

            optimizer.zero_grad(set_to_none=True)
            loss_this_epoch.backward()
            _finite_grad_step(model, optimizer)
            model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value + 0.01)

            # step scheduler with the current epoch loss (as metric)
            if use_scheduler:
                scheduler.step(loss_this_epoch.detach())
            if epoch % 10 == 0:
                print(f"Loss at epoch {epoch}:", loss_this_epoch.item(), flush=True)
                print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'], flush=True)

        # fine-tuning loop
        for epoch in range(epochs // 4):
            losses = []
            if use_compressed:
                output_mps_list = apply_mpo_to_mps_compressed(
                    model, MPS_weight1, max_bond=max_bd, cutoff=max_err, noise_type=noise_type
                )
                for sample_idx in range(len(MPS_weight1)):
                    loss = tensor_network_distance(
                        output_mps_list[sample_idx],
                        target_mps_list[sample_idx],
                        normalized=normalized
                    )
                    losses.append(loss)
            else:
                mpo_fit = model.get_MPO(noise_type=noise_type).astype("complex128")
                for sample_idx in range(len(MPS_weight1)):
                    # mpo_fit and MPS_weight1 are both complex128 already;
                    # skip the redundant per-sample .astype copies that were
                    # burning tens of thousands of allocations per epoch.
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    loss = tensor_network_distance(
                        M_input,
                        target_mps_list[sample_idx],
                        normalized=normalized
                    )
                    losses.append(loss)

            loss_this_epoch = torch.stack(losses).mean()
            print(f"Loss at fine-tune epoch {epoch}:", loss_this_epoch.item(), flush=True)
            learning_loss.append(loss_this_epoch.item())

            optimizer.zero_grad(set_to_none=True)
            loss_this_epoch.backward()
            _finite_grad_step(model, optimizer)
            model.p_depolar.data = torch.clamp(
                model.p_depolar.data,
                0.0,
                min(2 * clamp_value + 0.01, 1.0)
            )
    elif noise_type == "dephasing":
        dephasingX_strength = gamma[0]*2
        dephasingY_strength = gamma[0]*2
        dephasingZ_strength = gamma[0]*2
        clamp_value_X = min(dephasingX_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
        clamp_value_Y = min(dephasingY_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
        clamp_value_Z = min(dephasingZ_strength*1.1*model_to_learn_layer/MPO_layer, 1.0)
        for epoch in range(epochs):
            # Per-sample backward: accumulate gradients, free each sample's
            # autograd graph as soon as we've backed through it.  The previous
            # pattern (stack all sample losses → single .backward()) kept ~180
            # per-sample graphs alive simultaneously and OOM-killed workers.
            n = len(MPS_weight1)
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            if use_compressed:
                # The compressed path rebuilds outputs in-process from model
                # params, so rebuild them per sample to avoid the same leak.
                for sample_idx in range(n):
                    output_mps = apply_mpo_to_mps_compressed(
                        model, [MPS_weight1[sample_idx]],
                        max_bond=max_bd, cutoff=max_err, noise_type=noise_type
                    )[0]
                    loss_i = tensor_network_distance(
                        output_mps,
                        target_mps_list[sample_idx],
                        normalized=normalized,
                    )
                    (loss_i / n).backward()
                    running_loss += loss_i.detach().item()
            else:
                for sample_idx in range(n):
                    mpo_fit = model.get_MPO(noise_type=noise_type).astype("complex128")
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    loss_i = tensor_network_distance(
                        M_input,
                        target_mps_list[sample_idx],
                        normalized=normalized,
                    )
                    (loss_i / n).backward()
                    running_loss += loss_i.detach().item()

            loss_this_epoch_value = running_loss / n
            learning_loss.append(loss_this_epoch_value)

            _finite_grad_step(model, optimizer)
            model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, clamp_value_X+1e-6)
            model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, clamp_value_Y+1e-6)
            model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, clamp_value_Z+1e-6)

            # step scheduler with the current epoch loss (as metric)
            if use_scheduler:
                scheduler.step(torch.tensor(loss_this_epoch_value))
            if epoch % 1 == 0:
                print(f"Loss at epoch {epoch}:", loss_this_epoch_value, flush=True)
                print(f"Learning rate at epoch {epoch}:", optimizer.param_groups[0]['lr'], flush=True)

        # fine-tuning loop
        for epoch in range(epochs // 5):
            n = len(MPS_weight1)
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            if use_compressed:
                for sample_idx in range(n):
                    output_mps = apply_mpo_to_mps_compressed(
                        model, [MPS_weight1[sample_idx]],
                        max_bond=max_bd, cutoff=max_err, noise_type=noise_type,
                    )[0]
                    loss_i = tensor_network_distance(
                        output_mps,
                        target_mps_list[sample_idx],
                        normalized=normalized,
                    )
                    (loss_i / n).backward()
                    running_loss += loss_i.detach().item()
            else:
                for sample_idx in range(n):
                    mpo_fit = model.get_MPO(noise_type=noise_type).astype("complex128")
                    M_input = mpo_fit | MPS_weight1[sample_idx]
                    loss_i = tensor_network_distance(
                        M_input,
                        target_mps_list[sample_idx],
                        normalized=normalized,
                    )
                    (loss_i / n).backward()
                    running_loss += loss_i.detach().item()

            loss_this_epoch_value = running_loss / n
            print(f"Loss at fine-tune epoch {epoch}:", loss_this_epoch_value, flush=True)
            learning_loss.append(loss_this_epoch_value)

            _finite_grad_step(model, optimizer)
            model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, min(2*clamp_value_X+1e-6, 1.0))
            model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, min(2*clamp_value_Y+1e-6, 1.0))
            model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, min(2*clamp_value_Z+1e-6, 1.0))

        # also update scheduler here (still monitoring the same loss)
        if use_scheduler:
            scheduler.step(torch.tensor(loss_this_epoch_value))
        if epoch % 10 == 0:
            print(f"Learning rate at fine-tune epoch {epoch}:", optimizer.param_groups[0]['lr'], flush=True)

    # final MPO after training
    final_mpo_fit = model.get_MPO(noise_type=noise_type)
    # 150 sequential test samples × full depth-30 TDME teacher per sample
    # took >15h on the previous run (most workers got <10/150 done before
    # SLURM cancelled the job).  Drop to 30 — gives ~5x faster testing while
    # still well above the std-error noise floor for the testing-loss mean.
    num_samples = 30

    # ── Persist trained model + learning loss BEFORE the (slow) testing
    #    phase so that if SLURM kills the job mid-test we don't lose the
    #    14h training run.  save_path_prefix is opt-in via env var; falls
    #    back to a deterministic location under Learning_result/.
    try:
        import os
        gamma_scalar = gamma[0] if hasattr(gamma, "__len__") else gamma
        ckpt_dir = os.environ.get("LEARNING_TDME_CKPT_DIR", "Learning_result")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_prefix = os.path.join(
            ckpt_dir,
            f"TDME_N{N}_T{MPO_layer}_Modeltolearnlayer{model_to_learn_layer}"
            f"_time{t}_gamma{int(round(gamma_scalar*100)):03d}",
        )
        np.save(f"{ckpt_prefix}_learning_loss.npy", np.array(learning_loss))
        torch.save(model.state_dict(), f"{ckpt_prefix}_model.pt")
        print(f"Saved post-training checkpoint to {ckpt_prefix}_*.{{npy,pt}}",
              flush=True)
    except Exception as _e:
        print(f"WARNING: post-training checkpoint save failed: {_e}",
              flush=True)

    # Pre-compute target-model unitaries and jump matrices ONCE
    # (identical for every test sample — avoids N×T expm calls per sample).
    r_target = t / model_to_learn_layer
    target_all_unitary = construct_TDME_unitary(N, model_to_learn_layer, r=r_target, mu=mu, J=J)
    target_all_jumping = construct_jump_matrices(N, gamma, r=r_target)

    def _test_single_sample_tdme(j):
        """Evaluate one random test sample. No gradients needed."""
        with torch.no_grad():
            Random_MPS, weight = random_pauli_MPS(N)
            try:
                MPS_target, _ = Pauli_MPS_after_TDME_output_only(
                    Random_MPS.copy(), model_to_learn_layer, r=r_target,
                    all_unitary=target_all_unitary,
                    all_jumping=target_all_jumping,
                    max_bd=max_bd, max_err=max_err, truncation=truncation,
                )
            except Exception:
                return None  # sentinel: this sample failed
            # The student forward must be guarded too — an SVD failure or
            # non-finite value here previously crashed the whole run AFTER
            # training had already succeeded (matching the parallel worker,
            # which wraps everything in try/except).
            try:
                for i, tensor in enumerate(MPS_target):
                    tensor.reindex_({f'input{i}': f'k{i}'})
                if use_compressed:
                    M_out = model.forward_compressed(Random_MPS.copy(), max_bond=max_bd, cutoff=max_err, noise_type=noise_type)
                    for i, tensor in enumerate(M_out):
                        tensor.reindex_({f'input{i}': f'k{i}'})
                    loss = tensor_network_distance(
                        M_out.astype("complex128"),
                        MPS_target.astype("complex128")
                    ).item()
                else:
                    loss = tensor_network_distance(
                        final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
                        MPS_target.astype("complex128")
                    ).item()
                    print(f"Testing sample {j+1}/{num_samples}, loss: {loss}", flush=True)
            except Exception as e:
                print(f"Testing sample {j+1}/{num_samples} failed in the student "
                      f"forward ({type(e).__name__}); skipping.", flush=True)
                return None
        return loss

    # ── Fast sequential path ─────────────────────────────────────────
    # We use a sequential loop here. ThreadPoolExecutor is roughly 2x slower
    # due to Python's Global Interpreter Lock (GIL). 
    # ProcessPoolExecutor cannot be used because quimb tensor networks contain
    # lambda closures that are unpicklable.
    # PyTorch's native C++ multithreading (via OMP) implicitly parallelizes the 
    # tensor operations underneath.
    if num_threads is None or num_threads <= 1:
        raw_results = [_test_single_sample_tdme(j) for j in range(num_samples)]
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context('spawn')
        model_state_dict = model.state_dict()
        task_args = [
            (j, N, model_to_learn_layer, r_target, target_all_unitary, target_all_jumping, max_bd, max_err, truncation, use_compressed, noise_type, num_samples)
            for j in range(num_samples)
        ]
        
        raw_results_dict = {}
        print(f"Starting parallel testing with {num_samples} samples on {num_threads} workers...")
        with ProcessPoolExecutor(
            max_workers=num_threads, 
            mp_context=ctx,
            initializer=_init_worker_learning_test,
            initargs=(N, MPO_layer, model_state_dict, noise_type, use_compressed)
        ) as executor:
            futures = {
                executor.submit(_process_single_sample_learning_test_tdme, arg): arg[0]
                for arg in task_args
            }
            
            for future in as_completed(futures):
                j, res = future.result()
                raw_results_dict[j] = res
                
        raw_results = [raw_results_dict[j] for j in range(num_samples)]

    # Filter out failed samples (None sentinels)
    testing_loss_list = [r for r in raw_results if r is not None]
    n_skipped = num_samples - len(testing_loss_list)
    if n_skipped > 0:
        print(f"Skipped {n_skipped} failed testing samples.", flush=True)
    if testing_loss_list:
        testing_loss = sum(testing_loss_list) / len(testing_loss_list)
    else:
        # NaN, not 0.0: a zero here is indistinguishable from a perfect fit
        # in downstream .npy files and plots, silently masking total failure.
        testing_loss = float("nan")
        print("WARNING: every testing sample failed — testing_loss set to NaN "
              "(was previously reported as 0.0, which looks like a perfect fit).",
              flush=True)
    print("Testing loss:", testing_loss, flush=True)
    return model, learning_loss, testing_loss, testing_loss_list



def Testing_TDME_Trotterization(
    N, model_layer, model_to_learn_layer, mu, gamma, J=1, t = 1, normalized = False, max_bd=64, max_err=1E-8, truncation = False, noise_type = "all", use_scheduler = True
):
    testing_loss = 0.0
    model_with_small_layer = TDME(N, model_layer, mu=mu, gamma=gamma, J=J, max_bd=max_bd, max_err=max_err)
    QMLM_MPS_output = TDME(N, model_to_learn_layer, mu=mu, gamma=gamma, J=J, max_bd=max_bd, max_err=max_err)
    num_samples = 300
    for j in range(num_samples):
        Random_MPS, weight = random_pauli_MPS(N)
        try:
            MPS_target = QMLM_MPS_output.forward(Random_MPS.copy(), t = t, truncation=truncation)
            MPS_model_with_small_layer = model_with_small_layer.forward(Random_MPS.copy(), t = t, truncation=truncation)
        except:
            print(f"Failed to generate target MPS for sample {j}, it is skipped in testing loss calculation.")
            num_samples -= 1
            continue
        loss = tensor_network_distance(
            MPS_model_with_small_layer.astype("complex128"),
            MPS_target.astype("complex128")
        ).item()
        print(f"Testing sample {j+1}/{num_samples}, loss: {loss}")
        testing_loss += loss

        """loss = tensor_network_distance(
            final_mpo_fit.astype("complex128") | Random_MPS.astype("complex128"),
            mpo_target.astype("complex128") | Random_MPS.astype("complex128")
        ).item()
        testing_loss += loss"""

    testing_loss /= num_samples
    print("Testing loss:", testing_loss)
    return testing_loss




# Worker-process-global storage for the constants that are identical for every
# sample in a Testing_TDME_Trotterization_parallel call.  Set once via the
# ProcessPoolExecutor `initializer=` hook so we don't re-pickle them per sample.
_TDME_TEST_GLOBALS = {}



def _init_test_tdme_worker(model_all_unitary, model_all_jumping,
                           target_all_unitary, target_all_jumping,
                           same_target_and_model,
                           model_first_order=False):
    # One thread per worker; the OS scheduler spreads workers across cores.
    torch.set_num_threads(1)
    _TDME_TEST_GLOBALS["model_all_unitary"]  = model_all_unitary
    _TDME_TEST_GLOBALS["model_all_jumping"]  = model_all_jumping
    _TDME_TEST_GLOBALS["target_all_unitary"] = target_all_unitary
    _TDME_TEST_GLOBALS["target_all_jumping"] = target_all_jumping
    _TDME_TEST_GLOBALS["same_target_and_model"] = same_target_and_model
    _TDME_TEST_GLOBALS["model_first_order"] = model_first_order

    # Pin this worker to a unique CPU so sibling workers don't pile onto the
    # same physical core.  Falls back silently on platforms without affinity.
    try:
        avail = sorted(os.sched_getaffinity(0))
        # Use the worker's PID modulo the available CPUs as a stable index.
        cpu = avail[os.getpid() % len(avail)]
        os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass



def _process_single_sample_test_tdme(args):
    """
    Worker function to process a single sample for Testing_TDME_Trotterization.
    Pulls the (large) per-task constants from worker-global state instead of
    receiving them via the pickled args tuple.
    """
    j, N, model_layer, model_to_learn_layer, r_model, r_target, max_bd, max_err, truncation = args

    model_all_unitary  = _TDME_TEST_GLOBALS["model_all_unitary"]
    model_all_jumping  = _TDME_TEST_GLOBALS["model_all_jumping"]
    target_all_unitary = _TDME_TEST_GLOBALS["target_all_unitary"]
    target_all_jumping = _TDME_TEST_GLOBALS["target_all_jumping"]
    same_target_and_model = _TDME_TEST_GLOBALS["same_target_and_model"]
    model_first_order = _TDME_TEST_GLOBALS.get("model_first_order", False)
    model_evolve = (
        Pauli_MPS_after_TDME_output_only_first_order
        if model_first_order
        else Pauli_MPS_after_TDME_output_only
    )

    Random_MPS, weight = random_pauli_MPS(N)

    try:
        with torch.no_grad():
            MPS_target, _ = Pauli_MPS_after_TDME_output_only(
                Random_MPS.copy(), model_to_learn_layer, r=r_target,
                all_unitary=target_all_unitary, all_jumping=target_all_jumping,
                max_bd=max_bd, max_err=max_err, truncation=truncation
            )
            if same_target_and_model and not model_first_order:
                MPS_model_with_small_layer = MPS_target
            else:
                MPS_model_with_small_layer, _ = model_evolve(
                    Random_MPS.copy(), model_layer, r=r_model,
                    all_unitary=model_all_unitary, all_jumping=model_all_jumping,
                    max_bd=max_bd, max_err=max_err, truncation=truncation
                )

            loss = tensor_network_distance(
                MPS_model_with_small_layer.astype("complex128"),
                MPS_target.astype("complex128")
            ).item()

        return j, loss
    except Exception as e:
        return j, None



def Testing_TDME_Trotterization_parallel(
    N, model_layer, model_to_learn_layer, mu, gamma, J=1, t=1, normalized=False,
    max_bd=64, max_err=1E-8, truncation=False, noise_type="all", use_scheduler=True, num_samples=300, num_threads=4,
    incremental_save_prefix=None,
    model_first_order=False,
):
    # If model_first_order=True, the model side uses first-order Trotter
    # (J · U_o · U_e per layer) while the target stays second-order. Per-step
    # gates are constructed with the matching helpers so jump strengths and
    # unitaries are consistent with the chosen splitting order.
    print(f"Starting parallel testing with {num_samples} samples on {num_threads} workers...", flush=True)

    r_model = t / model_layer
    if model_first_order:
        model_all_unitary = construct_TDME_unitary_first_order(N, model_layer, r=r_model, mu=mu, J=J)
        model_all_jumping = construct_jump_matrices_first_order(N, gamma, r=r_model)
    else:
        model_all_unitary = construct_TDME_unitary(N, model_layer, r=r_model, mu=mu, J=J)
        model_all_jumping = construct_jump_matrices(N, gamma, r=r_model)

    r_target = t / model_to_learn_layer
    same_target_and_model = (model_layer == model_to_learn_layer) and not model_first_order
    if same_target_and_model:
        target_all_unitary = model_all_unitary
        target_all_jumping = model_all_jumping
    else:
        target_all_unitary = construct_TDME_unitary(N, model_to_learn_layer, r=r_target, mu=mu, J=J)
        target_all_jumping = construct_jump_matrices(N, gamma, r=r_target)

    # Lightweight per-sample tuple — no big tensors inside.
    task_args = [
        (j, N, model_layer, model_to_learn_layer, r_model, r_target, max_bd, max_err, truncation)
        for j in range(num_samples)
    ]

    testing_loss = 0.0
    successful_samples = 0
    testing_loss_list = []

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(
            max_workers=num_threads,
            mp_context=ctx,
            initializer=_init_test_tdme_worker,
            initargs=(model_all_unitary, model_all_jumping,
                      target_all_unitary, target_all_jumping,
                      same_target_and_model, model_first_order),
    ) as executor:
        futures = {
            executor.submit(_process_single_sample_test_tdme, arg): arg[0]
            for arg in task_args
        }

        for future in as_completed(futures):
            j, loss = future.result()
            if loss is not None:
                successful_samples += 1
                testing_loss += loss
                testing_loss_list.append(loss)
                print(f"Testing sample completed, loss so far: {loss:.5f} (Success: {successful_samples})", flush=True)
            else:
                print(f"Failed to generate target MPS for a sample, it is skipped.", flush=True)

            if incremental_save_prefix is not None and successful_samples > 0:
                try:
                    np.save(f"{incremental_save_prefix}_list.npy",
                            np.asarray(testing_loss_list))
                    np.save(f"{incremental_save_prefix}_loss.npy",
                            np.asarray(testing_loss / successful_samples))
                except Exception as exc:
                    print(f"[warn] incremental save failed: {exc}", flush=True)

    if successful_samples > 0:
        testing_loss /= successful_samples
    else:
        testing_loss = 0.0

    print(f"Testing loss over {successful_samples} samples: {testing_loss}", flush=True)
    return testing_loss, testing_loss_list



# ──────────────────────────────────────────────────────────────────────
# Learning driver for the system-bath teacher (QMLM_with_bath_output_only).
# Mirrors Learning_MPO_scheduler: the student is a plain QMLM on N data qubits
# with trainable noise; the teacher is a 2N-site QMLM_with_bath whose bath is
# partial-traced before comparison.
# ──────────────────────────────────────────────────────────────────────

def Learning_QMLM_with_bath_scheduler(
    N, MPO_layer, model_to_learn_layer,
    coupling_strength=0.05, J_b=1.0,
    haar_list=None, weak_list=None,
    depolarizing_strength=0.2,
    epochs=100, lr=0.01, normalized=False,
    max_bd=32, max_err=1E-6, truncation=True, noise_type="all",
    use_compressed=False, num_threads=None,
    post_train_callback=None,
):
    """Fit a T-layer student QMLM (N qubits, trainable noise) to an L-layer
    system-bath teacher (2N qubits, no noise, Haar data-data + weak data-bath).

    Parameters
    ----------
    post_train_callback : callable or None
        If provided, called once after the training and fine-tune loops complete
        but BEFORE the 100-sample testing block. Receives a dict with keys
        {'model', 'learning_loss', 'haar_list', 'weak_list'}. Useful for
        persisting the trained model to disk so that a slow/hung testing block
        does not destroy the training results.
    """
    # Generate inputs + bath-traced targets. The teacher samples Haar unitaries
    # internally; pass them back through to keep the randomness reproducible.
    MPS_weight1, target_mps_list, haar_list, weak_list = get_input_and_output_MPS_with_bath(
        N, model_to_learn_layer,
        haar_list=haar_list, weak_list=weak_list,
        coupling_strength=coupling_strength, J_b=J_b,
        truncation=truncation, max_bd=max_bd, max_err=max_err,
        num_threads=num_threads,
    )

    for MPS_weight1_tensor in MPS_weight1:
        for tensor in MPS_weight1_tensor.tensors:
            tensor.data.requires_grad_(False)
    for target_tensor in target_mps_list:
        for tensor in target_tensor.tensors:
            tensor.data.requires_grad_(False)
    MPS_weight1 = [m.astype("complex128") for m in MPS_weight1]
    target_mps_list = [m.astype("complex128") for m in target_mps_list]

    # Build the student (data-side only, with noise). Use a random param init
    # rather than the default identity init: the compressed SVD backward gives
    # NaN gradients at exact identity because the output MPS has degenerate
    # singular values. Random init keeps autograd stable.
    p_depolar_MPO = torch.ones((MPO_layer, N), dtype=torch.float64) * depolarizing_strength * 0.5
    rand_param = (
        torch.rand(MPO_layer, N, 16, dtype=torch.float64)
        + 1j * torch.rand(MPO_layer, N, 16, dtype=torch.float64)
    )
    rand_param = torch.nn.Parameter(rand_param, requires_grad=True)
    model = QMLM(N, MPO_layer, param=rand_param, p_depolar=p_depolar_MPO)

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        threshold=5e-4,
        min_lr=1e-5,
    )

    learning_loss = []
    clamp_value = min(depolarizing_strength * 1.1 * model_to_learn_layer / MPO_layer, 1.0)

    # Main training loop
    for epoch in range(epochs):
        losses = []
        if use_compressed:
            # truncation=True is required: apply_mpo_to_mps_compressed defaults
            # to False, which leaves bonds uncapped and defeats the O(N D^3)
            # advantage (and is in fact slower than the MPO-build path).
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err,
                noise_type=noise_type, truncation=True,
            )
            for sample_idx in range(len(MPS_weight1)):
                out_stripped = _strip_size1_outer_bonds(output_mps_list[sample_idx])
                loss = tensor_network_distance(
                    out_stripped.astype("complex128"),
                    target_mps_list[sample_idx],
                    normalized=normalized,
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    target_mps_list[sample_idx],
                    normalized=normalized,
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad(set_to_none=True)
        loss_this_epoch.backward()
        _finite_grad_step(model, optimizer)
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, clamp_value + 0.01)
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, clamp_value + 0.01)
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, clamp_value + 0.01)
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, clamp_value + 0.01)
        scheduler.step(loss_this_epoch.detach())
        print(f"Bath-N{N}, T{MPO_layer}, L{model_to_learn_layer}, g{coupling_strength}, "
              f"Epoch {epoch}, Loss:", loss_this_epoch.item(),
              f", lr: {optimizer.param_groups[0]['lr']}", flush=True)

    # Fine-tune loop with relaxed clamp.
    for epoch in range(epochs // 5):
        losses = []
        if use_compressed:
            output_mps_list = apply_mpo_to_mps_compressed(
                model, MPS_weight1, max_bond=max_bd, cutoff=max_err,
                noise_type=noise_type, truncation=True,
            )
            for sample_idx in range(len(MPS_weight1)):
                out_stripped = _strip_size1_outer_bonds(output_mps_list[sample_idx])
                loss = tensor_network_distance(
                    out_stripped.astype("complex128"),
                    target_mps_list[sample_idx],
                    normalized=normalized,
                )
                losses.append(loss)
        else:
            mpo_fit = model.get_MPO(noise_type=noise_type)
            for sample_idx in range(len(MPS_weight1)):
                M_input = mpo_fit | MPS_weight1[sample_idx]
                loss = tensor_network_distance(
                    M_input.astype("complex128"),
                    target_mps_list[sample_idx],
                    normalized=normalized,
                )
                losses.append(loss)

        loss_this_epoch = torch.stack(losses).mean()
        learning_loss.append(loss_this_epoch.item())

        optimizer.zero_grad(set_to_none=True)
        loss_this_epoch.backward()
        _finite_grad_step(model, optimizer)
        relaxed = min(2 * clamp_value + 0.01, 1.0)
        model.p_depolar.data = torch.clamp(model.p_depolar.data, 0.0, relaxed)
        model.p_dephaseX.data = torch.clamp(model.p_dephaseX.data, 0.0, relaxed)
        model.p_dephaseY.data = torch.clamp(model.p_dephaseY.data, 0.0, relaxed)
        model.p_dephaseZ.data = torch.clamp(model.p_dephaseZ.data, 0.0, relaxed)
        scheduler.step(loss_this_epoch.detach())
        print(f"Fine-tune Bath-N{N}, T{MPO_layer}, L{model_to_learn_layer}, g{coupling_strength}, "
              f"Epoch {epoch}, Loss:", loss_this_epoch.item(),
              f", lr: {optimizer.param_groups[0]['lr']}", flush=True)

    # Hand the trained model to the caller before the (potentially slow)
    # testing block so partial-completion outputs can be persisted.
    if post_train_callback is not None:
        try:
            post_train_callback({
                'model': model,
                'learning_loss': learning_loss,
                'haar_list': haar_list,
                'weak_list': weak_list,
            })
        except Exception as e:
            print(f"  post_train_callback failed: {type(e).__name__}: {e}", flush=True)

    # Testing on random Pauli inputs.
    num_samples = 100
    test_data_list, test_target_list, _, _ = get_random_input_output_MPS_with_bath(
        N, model_to_learn_layer, no_sample=num_samples,
        haar_list=haar_list, weak_list=weak_list,
        coupling_strength=coupling_strength, J_b=J_b,
        truncation=truncation, max_bd=max_bd, max_err=max_err,
    )
    test_data_list = [m.astype("complex128") for m in test_data_list]
    test_target_list = [m.astype("complex128") for m in test_target_list]

    testing_loss_list = []
    n_skipped = 0
    final_mpo_fit = model.get_MPO(noise_type=noise_type) if not use_compressed else None
    with torch.no_grad():
        for j in range(num_samples):
            try:
                if use_compressed:
                    M_out = model.forward_compressed(
                        test_data_list[j].copy(), max_bond=max_bd,
                        cutoff=max_err, noise_type=noise_type,
                    )
                    for i, tensor in enumerate(M_out):
                        tensor.reindex_({f'input{i}': f'k{i}'})
                    M_out = _strip_size1_outer_bonds(M_out)
                    loss = tensor_network_distance(
                        M_out.astype("complex128"),
                        test_target_list[j],
                    ).item()
                else:
                    loss = tensor_network_distance(
                        final_mpo_fit.astype("complex128") | test_data_list[j],
                        test_target_list[j],
                    ).item()
            except Exception as e:
                print(f"Testing sample {j+1}/{num_samples} failed "
                      f"({type(e).__name__}); skipping.", flush=True)
                n_skipped += 1
                continue
            testing_loss_list.append(loss)

    if n_skipped > 0:
        print(f"Skipped {n_skipped} failed testing samples.", flush=True)
    if testing_loss_list:
        testing_loss = sum(testing_loss_list) / len(testing_loss_list)
    else:
        testing_loss = float("nan")
        print("WARNING: every testing sample failed — testing_loss set to NaN.",
              flush=True)
    print(f"Bath-N{N}, T{MPO_layer}, L{model_to_learn_layer}, g{coupling_strength}, "
          f"Testing loss: {testing_loss}")
    return (
        model, learning_loss, haar_list, weak_list,
        testing_loss, testing_loss_list,
        model.params.detach().numpy(),
        model.p_depolar.detach().numpy(),
        model.p_dephaseX.detach().numpy(),
        model.p_dephaseY.detach().numpy(),
        model.p_dephaseZ.detach().numpy(),
    )

