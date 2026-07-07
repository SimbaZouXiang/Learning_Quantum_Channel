with open("TDME_Trott.py", "r") as f:
    content = f.read()

import_str = """from quimb.tensor.fitting import tensor_network_distance
from ncon import ncon
import cotengra as ctg
opt_ctg = ctg.ReusableHyperOptimizer(progbar=False, reconf_opts={}, max_repeats=16, max_time='eq:2', parallel=True)"""

content = content.replace("from quimb.tensor.fitting import tensor_network_distance\nfrom ncon import ncon", import_str)

# Replace target calls in Learning_TDME_scheduler and related functions
import re
# We'll use regex to inject `optimize=opt_ctg` into `tensor_network_distance(` calls 
# where it's missing, mostly when dealing with the ML loops.
replacements = [
    (r"loss = tensor_network_distance\([^)]+normalized=normalized\)", "loss = tensor_network_distance(\\g<0>[:-1], optimize=opt_ctg)"),
    (r"loss = tensor_network_distance\([^)]+M_check.astype\(\"complex128\"\),[^)]+\)", "loss = tensor_network_distance(\\g<0>[:-1], optimize=opt_ctg)"),
    # Replace anything of form: loss = tensor_network_distance( ... ) with optimize=opt_ctg if applicable
]

# A better way is to simply explicitly replace lines in the file.
