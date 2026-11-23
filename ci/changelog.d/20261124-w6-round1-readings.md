# W6: round 1's readings in the report

- `rel_w6_report.threshold_floor`: the seeds' threshold is at least this (round 1: 0.008, protocol readings.threshold).
- What relatives are worth carries the evaluator's paired bootstrap (split units inside each group, 2,000 replicates,
  seed 20261123, 95%; the blocks apart) at the end (the cells of `heldout_final.npz`, the arm without relatives at its
  best from `heldout_best.npz`) and at the last common node (`node_cells.npz`), on the tables with a relative on the
  target block; a worth counts only beyond its threshold with its interval excluding 0 (protocol evaluation.intervals).
- New sections: the mask schemes' realised shares of the target cells and of the loss per arm, and the whole gene's
  frame relatives by distance in training and on the held-out tables (the trainer's summary).
- `evaluate_checkpoints` (stage `w6_eval`) keeps the best weights' cells too (`heldout_best.npz`).
