# W6: w6_eval reads the weights an entry names

- `rel_w6_eval`: an arm entry may name the weights it reads, `{name, weights: final | best | node@<tables>}` (folder
  `<arm>.<weights>`, rows `ema@best` / `ema@tables=<n>`, a readout's `info.weights` as given; one arm listed several
  times); `best: all` reads the best weights of every arm beside its final ones; `tables.heldout_offline` and
  `end_controls` name the strict offline rendering and the controls in place of the training's own; a readout never
  reads an offline rendering.
- `rel_w6_eval.readout.offline`: the run of stage `rel_w18_offline` on the ruler set (protocol w18 version 3), passed
  to the module's `run_ruler`, which renders leak-free itself.
- Round 1's rulings of 2026-11-23: the result rows' `budget` names the micro-batch trained with (`;micro=<n>`); the
  summary counts the whole gene's frame relatives by distance (`frame_tiers`: self_copy, tier_<t>, none) in training
  (the rows drawn) and on every held-out table; the evaluator's three relatives strata are read where the evaluator
  has them (`strata_of(..., kind)`).
- `rel_w6_fit`: a preflight's memory allowance is the variant's, or what the free GPU memory leaves for the training
  of the arm alone (free less 500 MB, over 1.1), so that an arm preflighted at a micro-batch is never refused its
  training for memory.
