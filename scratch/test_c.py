from quimb.tensor.fitting import tensor_network_distance as _quimb_tnd
import cotengra as ctg
import sys
sys.path.append("/scratch/simba/")

opt = ctg.ReusableHyperOptimizer(
    methods=['kahypar', 'greedy'], 
    max_repeats=2, 
    max_time=0.1,
    parallel=False,
    progbar=False
)

def tensor_network_distance(*args, optimize=opt, **kwargs):
    print("CALLED WITH COTENGRA OPTIMIZER!")
    return _quimb_tnd(*args, optimize=optimize, **kwargs)

from Learning_Lindbladian import TDME_Trott
TDME_Trott.tensor_network_distance = tensor_network_distance

if __name__ == '__main__':
    TDME_Trott.Learning_TDME_scheduler(
        N=8, MPO_layer=1, model_to_learn_layer=1, mu=0.5, gamma=[0.1]*8, J=1.0, t=0.6,
        epochs=1, lr=0.01, normalized=False, max_bd=16, max_err=1E-6, 
        truncation=True, noise_type="all", use_scheduler=False,
        use_compressed=True, num_threads=4  # fewer threads for isolation testing
    )
