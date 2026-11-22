"""The "within tolerance" function of `python -m ci compare A B --expect within_tolerance` (ci/__main__.py), at a fixed
path. Package W4 implements it; the programme lead writes the thresholds into the `tolerance` block of a round's
protocol (configs/rel/protocols/<round>.yaml) after round 1. It runs in the CI environment: standard library and numpy
only, never torch.

within_tolerance(run_a, run_b, stage, outputs_a, outputs_b, tolerance) is called once for every stage that
configs/rel/kinds.yaml marks `tolerance` and whose outputs differ byte for byte between the two runs. outputs_a and
outputs_b map each output name of the stage record to its path relative to the run directory. It returns
dict(verdict="within" | "outside" | "undecided", diffs=[...]); compare passes only when every such stage is "within".
"""
from __future__ import annotations


def within_tolerance(run_a, run_b, stage, outputs_a, outputs_b, tolerance):
    return dict(verdict="undecided", diffs=[dict(stage=stage, outputs=sorted(set(outputs_a) | set(outputs_b)),
                                                 error="the tolerance function is not written yet (package W4); thresholds come from the round's protocol",
                                                 thresholds_given=bool(tolerance))])
