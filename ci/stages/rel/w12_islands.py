"""Stage rel_w12_islands (package W12; pipeline rel_tables): the tables of version 1 with the island set's development
chromosomes as row 0 (task breakdown 8.8), the other rows training genomes, for the diagnosis of package W18 only.
`python -m ci.stages.rel.w12_islands RUN_DIR`. The same code as stage rel_w12_v1 (ci/stages/rel/w12_v1.py), its own
table set under data/rel_w12/islands/, marked purpose heldout and not_for_training: no training reads it. Profile block
rel_w12_islands (search_from, per_chromosome, ...); the search must have been run with its `islands`.
"""
from __future__ import annotations

from ci.stages.common import main_args
from ci.stages.rel import common
from ci.stages.rel import w12_v1


def gate(st, cfg, run_dir):
    return w12_v1.gate(st, cfg, run_dir, "islands")


def summary(run_dir):
    return w12_v1.summary(run_dir, "islands")


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w12_islands") as st:
        w12_v1.build(st, st.config, run_dir, "islands")


if __name__ == "__main__":
    main()
