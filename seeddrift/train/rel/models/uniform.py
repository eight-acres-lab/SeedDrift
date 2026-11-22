"""The stub arm of the relational-model line: every target cell gets the uniform distribution over the 65 target
symbols (1-64 the codons, 65 "other"), so log2(65) = 6.02 bits per cell. No parameters, numpy only. It lets the
registry, the trainer and the evaluator be exercised end to end before any real arm exists, and it is the floor every
arm must beat.

logp(rendered, target, truth=None) -> array (number of target cells, 65) of natural-log probabilities, column j for
symbol j + 1, in the order of numpy's flattening of the boolean array `target`; `rendered` and `truth` are not read.
The interface follows the contract (seeddrift/train/rel/api.py); the names of `info` follow its version 1 once merged.
"""
from __future__ import annotations

import numpy as np

SYMBOLS = 65

info = dict(name="uniform", parameters=0, flops_per_cell=0, factorization="independent", column_layout="any", memory_mb=0, common_version=None)


def logp(rendered, target, truth=None):
    n = int(np.count_nonzero(np.asarray(target, dtype=bool)))
    return np.full((n, SYMBOLS), -np.log(SYMBOLS))
