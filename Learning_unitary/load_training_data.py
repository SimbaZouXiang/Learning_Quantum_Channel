"""Backward-compatibility shim — save_mps / load_mps now live in qcl.io.

Kept so existing imports (`from load_training_data import save_mps, load_mps`
and `from Learning_unitary.load_training_data import load_mps`) keep working.
"""
import os as _os
import sys as _sys

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from qcl.io import save_mps, load_mps
