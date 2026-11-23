# W3: the strict rendering for training, and its checks (aligner version 2)

- `align.strict` (version 2) also aligns a row again in a block where only that row hides residues (training hides
  cells of every row: scattered 15% of each, a span in any row); every hidden residue is unknown to the aligner. Where
  only row 0 is hidden (every evaluation condition) the result is version 1's.
- `seeddrift/train/rel/tables/strict.py`: the one strict rendering for training and evaluation (place, render:
  strict or frame relative, then leak-free) and W3's fourth test of one rendering (fourth_test: every hidden residue of
  any row randomised by the evaluator's hash, every stored position a hidden residue could have shaped erased, the same
  grid, cells laid by extension and cells to score). The trainer's sampler calls fourth_test on one sample in 1,000.
  Stage w3_offline renders and tests through it.
- Stage `w3_traincheck` (kind rel_tables, numpy): the fourth test on the training's own renderings of 1,000 training
  tables, every mask scheme, the rows drawn as the sampler draws them; profile rel-dev-w3-traincheck-v1-i1; the gate
  of round 1's third batch.
