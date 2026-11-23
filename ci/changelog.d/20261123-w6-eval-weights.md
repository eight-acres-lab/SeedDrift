# W6: w6_eval reads the weights an entry names

- `rel_w6_eval`: an arm entry may name the weights it reads, `{name, weights: final | best | node@<tables>}` (folder
  `<arm>.<weights>`, rows `ema@best` / `ema@tables=<n>`, a readout's `info.weights` as given; one arm listed several
  times); `best: all` reads the best weights of every arm beside its final ones; `tables.heldout_offline` and
  `end_controls` name the strict offline rendering and the controls in place of the training's own; a readout never
  reads an offline rendering.
