"""Stage rel_w12_v1x (package W12; pipeline rel_tables): the tables of version 1 of the held-out genomes that a
reference table set did not have (rel_w12_v1x.new_since: {run, sha256}, a passed set of stage rel_w12_v1; the
programme lead, 2026-11-24: the new held-out genomes of round 2 form an evaluation set of their own, read and reported
apart, never merged into the 3,000 held-out tables of round 1). `python -m ci.stages.rel.w12_v1x RUN_DIR`. The same
code as stage rel_w12_v1 (ci/stages/rel/w12_v1.py), its own table set under data/rel_w12/v1x/, held-out part only,
marked purpose heldout and not_for_training. Profile block rel_w12_v1x (search_from, expect, new_since,
heldout_per_group, rows, seed, name, ...). Gate G4_rel_w12_v1x: as w12_v1's, and no examined genome is one the
reference set had.
"""
from __future__ import annotations

from ci.stages.common import main_args
from ci.stages.rel import common
from ci.stages.rel import w12_v1


def gate(st, cfg, run_dir):
    return w12_v1.gate(st, cfg, run_dir, "x")


def summary(run_dir):
    return w12_v1.summary(run_dir, "x")


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w12_v1x") as st:
        w12_v1.build(st, st.config, run_dir, "x")


if __name__ == "__main__":
    main()
