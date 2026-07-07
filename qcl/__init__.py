"""qcl — Quantum Channel Learning.

Tensor-network variational learning of noisy quantum channels (Lindbladian
dynamics) in the vectorised Pauli basis. This package is the modularised form
of the former monolithic ``Learning_Lindbladian/TDME_Trott.py``; that file is
now a thin re-export shim kept for backward compatibility
(``import TDME_Trott as tdme`` in the existing drivers keeps working).

Module map
----------
backend   cotengra path-cache optimiser, tensor_network_distance wrapper,
          global torch.linalg.svd safety monkey-patch (applied on import)
pauli     Pauli-basis constants, unitary→transfer-matrix conversion, SU(4)
noise     depolarizing / dephasing single-site transfer matrices
states    identity/product MPS and Pauli-string input MPS builders
mpo       MPO-building forward path (exact, exponential in depth)
evolve    layer-by-layer compressed MPS forward path (O(N·D³))
bath      system-bath (2N-site) teacher, partial trace, bath data generation
models    QMLM (trainable student), QMLM_output_only (teacher)
tdme      Trotterised time-dependent master equation physics + TDME model
io        save_mps / load_mps (.npz serialisation of quimb MPS)
datagen   (input, target) MPS pair generation, incl. process-pool workers
training  Learning_* / Testing_* driver-level loops
"""

# backend must load first: it installs the torch.linalg.svd fallback patch and
# the shared cotengra ReusableHyperOptimizer that everything else relies on.
from .backend import opt, tensor_network_distance

from .pauli import (
    Haar_random_unitary,
    Pauli_operator_basis,
    construct_SU4_from_input,
    construct_all_SU4,
    unitary_to_transfer_matrix_single_site,
    unitary_to_transfer_matrix_two_site,
    unitary_to_transfer_matrix_two_site_reshape,
    unitary_to_transfer_matrix_single_site_truncated,
    unitary_to_transfer_matrix_two_site_truncated,
    unitary_to_transfer_matrix_two_site_truncated_batched,
)
from .noise import (
    dephasing_noise_transfer_matrix,
    dephasing_noise_transfer_matrix_X,
    dephasing_noise_transfer_matrix_Y,
    dephasing_noise_transfer_matrix_Z,
    depolarization_noise_transfer_matrix,
)
from .states import (
    Identity_init,
    operator_assignment,
    operator_assignment_single_site,
    random_pauli_string,
    random_pauli_MPS,
    Pauli_MPS_weight_1,
    Pauli_MPS_weight_2,
    Pauli_MPS_weight_2_full,
    Pauli_MPS_combined_1_and_2_full,
    Pauli_MPS_random,
)
from .mpo import (
    Build_2_site_MPO_from_transfer_matrices,
    First_identity_layer,
    Apply_one_site_layer,
    Apply_two_site_layer,
    Build_QMLM_MPO,
    Pauli_MPS_after_QMLM,
    compress_tn_to_mps,
)
from .evolve import (
    evol_one_site,
    evol_two_site,
    one_site_layer,
    two_site_layer,
    Pauli_MPS_after_QMLM_output_only,
    sanitize_mps,
    ensure_complex_torch,
)
from .bath import (
    weak_data_bath_unitary,
    three_site_haar_transfer_matrix,
    evol_three_site,
    three_site_haar_layer,
    weak_db_layer,
    partial_trace_bath,
    Pauli_MPS_weight_1_with_bath,
    QMLM_with_bath_output_only,
    get_input_and_output_MPS_with_bath,
    get_random_input_output_MPS_with_bath,
)
from .models import (
    QMLM,
    QMLM_output_only,
    apply_mpo_to_mps_compressed,
)
from .tdme import (
    TDME,
    TDME_two_site_hamiltonian,
    construct_TDME_unitary,
    construct_jump_matrices,
    construct_TDME_unitary_first_order,
    construct_jump_matrices_first_order,
    Pauli_MPS_after_TDME_output_only,
    Pauli_MPS_after_TDME_output_only_first_order,
)
from .io import save_mps, load_mps
from .datagen import (
    get_target_tn,
    get_input_and_output_MPS,
    get_input_and_output_MPS_TDME,
    get_random_input_output_MPS,
    get_training_data,
    get_OSE_output,
    get_average_loss_weight_tn,
)
from .training import (
    Learning_MPO,
    Learning_MPO_scheduler,
    Learning_MPO_dephasing_noise_only,
    Learning_MPO_dephasing_noise_only_scheduler,
    Learning_TDME_scheduler,
    Learning_QMLM_with_bath_scheduler,
    Testing_TDME_Trotterization,
    Testing_TDME_Trotterization_parallel,
)
