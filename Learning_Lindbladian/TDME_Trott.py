"""Backward-compatibility shim — the real code now lives in the `qcl` package
(one directory up). Kept so that every existing driver, SLURM script, and
`sys.path` hack that does `import TDME_Trott as tdme` continues to work.

Do NOT add new code here; add it to the appropriate qcl module and re-export.
"""
import os as _os
import sys as _sys

# Make the project root (which contains the qcl package) importable no matter
# where this shim is imported from.
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from qcl.backend import (
    ProcessPoolExecutor,
    ReduceLROnPlateau,
    ThreadPoolExecutor,
    _SVD_JITTER_REL,
    _orig_svd,
    _safe_svd,
    _tensor_network_distance,
    as_completed,
    block_diag,
    csc_matrix,
    ctg,
    eye,
    hstack,
    kron,
    mp,
    ncon,
    nn,
    np,
    nx,
    opt,
    optim,
    os,
    plt,
    qtn,
    qu,
    tensor_network_distance,
    time,
    torch,
    warnings,
)

from qcl.pauli import (
    Haar_random_unitary,
    Pauli_operator_basis,
    _DEPHASING_TM_X,
    _DEPHASING_TM_Y,
    _DEPHASING_TM_Z,
    _DIR_X,
    _DIR_Y,
    _DIR_Z,
    _EYE4,
    _PAULI_BASIS,
    _PAULI_BASIS_2SITE,
    _TRACE_VEC_PAULI,
    _build_pauli_basis,
    construct_SU4_from_input,
    construct_all_SU4,
    unitary_to_transfer_matrix_single_site,
    unitary_to_transfer_matrix_single_site_truncated,
    unitary_to_transfer_matrix_two_site,
    unitary_to_transfer_matrix_two_site_reshape,
    unitary_to_transfer_matrix_two_site_truncated,
    unitary_to_transfer_matrix_two_site_truncated_batched,
)

from qcl.noise import (
    dephasing_noise_transfer_matrix,
    dephasing_noise_transfer_matrix_X,
    dephasing_noise_transfer_matrix_Y,
    dephasing_noise_transfer_matrix_Z,
    depolarization_noise_transfer_matrix,
)

from qcl.states import (
    Identity_init,
    Pauli_MPS_combined_1_and_2_full,
    Pauli_MPS_random,
    Pauli_MPS_weight_1,
    Pauli_MPS_weight_2,
    Pauli_MPS_weight_2_full,
    operator_assignment,
    operator_assignment_single_site,
    random_pauli_MPS,
    random_pauli_string,
)

from qcl.mpo import (
    Apply_one_site_layer,
    Apply_two_site_layer,
    Build_2_site_MPO_from_transfer_matrices,
    Build_QMLM_MPO,
    First_identity_layer,
    Pauli_MPS_after_QMLM,
    compress_tn_to_mps,
)

from qcl.evolve import (
    Pauli_MPS_after_QMLM_output_only,
    _qr_compress_cyclic_to_obc,
    _strip_size1_outer_bonds,
    ensure_complex_torch,
    evol_one_site,
    evol_two_site,
    one_site_layer,
    sanitize_mps,
    two_site_layer,
)

from qcl.bath import (
    Pauli_MPS_weight_1_with_bath,
    QMLM_with_bath_output_only,
    _compress_mps_with_bath,
    evol_three_site,
    get_input_and_output_MPS_with_bath,
    get_random_input_output_MPS_with_bath,
    partial_trace_bath,
    three_site_haar_layer,
    three_site_haar_transfer_matrix,
    weak_data_bath_unitary,
    weak_db_layer,
)

from qcl.models import (
    QMLM,
    QMLM_output_only,
    apply_mpo_to_mps_compressed,
)

from qcl.tdme import (
    Pauli_MPS_after_TDME_output_only,
    Pauli_MPS_after_TDME_output_only_first_order,
    TDME,
    TDME_two_site_hamiltonian,
    construct_TDME_unitary,
    construct_TDME_unitary_first_order,
    construct_jump_matrices,
    construct_jump_matrices_first_order,
)

from qcl.datagen import (
    _process_single_mps,
    _process_single_mps_tdme,
    get_OSE_output,
    get_average_loss_weight_tn,
    get_input_and_output_MPS,
    get_input_and_output_MPS_TDME,
    get_random_input_output_MPS,
    get_target_tn,
    get_training_data,
)

from qcl.training import (
    Learning_MPO,
    Learning_MPO_dephasing_noise_only,
    Learning_MPO_dephasing_noise_only_scheduler,
    Learning_MPO_scheduler,
    Learning_QMLM_with_bath_scheduler,
    Learning_TDME_scheduler,
    Testing_TDME_Trotterization,
    Testing_TDME_Trotterization_parallel,
    _TDME_TEST_GLOBALS,
    _compute_losses_parallel,
    _compute_single_loss_compressed,
    _compute_single_loss_mpo,
    _init_test_tdme_worker,
    _init_worker_learning_test,
    _process_single_sample_learning_test_tdme,
    _process_single_sample_test_tdme,
    _worker_model_for_test,
    _worker_mpo_for_test,
)
