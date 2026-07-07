with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'r') as f:
    text = f.read()

new_import = """from quimb.tensor.fitting import tensor_network_distance as _tensor_network_distance
import cotengra as ctg

# Global optimizer for tensor network distances
# ReusableHyperOptimizer caches the contraction paths and accelerates subsequent distance evaluations
opt = ctg.ReusableHyperOptimizer(
    methods=['kahypar', 'greedy'], 
    max_repeats=16, 
    max_time=1.0,
    parallel=True,
    progbar=False
)

def tensor_network_distance(*args, optimize=opt, **kwargs):
    return _tensor_network_distance(*args, optimize=optimize, **kwargs)
"""
import_stmt = "from quimb.tensor.fitting import tensor_network_distance"

if new_import in text:
    text = text.replace(new_import, import_stmt)
    with open('/scratch/simba/Learning_Lindbladian/TDME_Trott.py', 'w') as f:
        f.write(text)
    print("Unpatched successfully.")
else:
    print("New import not found!")

