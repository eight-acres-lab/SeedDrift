Package W5 (with the copying chain of W10), first slice: classical baselines on the held-out tables.

- `seeddrift/train/rel/evalh/`: `conditions.py`, the evaluation conditions every arm and every baseline share (which
  cells of the examined row are hidden and scored, which rows are given; a 64-bit hash instead of a numpy generator, so
  two interpreters plan the same cells; `ItemsDigest` proves it); `layout.py`, the stored layout in numpy; `score.py`,
  from per-cell bits to rows of the result table (per group, then the mean of the groups; strata by the identity of
  the nearest row); `shards.py`, reading the tables of a table-set manifest.
- `seeddrift/train/rel/baselines/`: own codon frequencies, the average of the rows, copy the nearest row (three
  per-position error rates; a variant fitted per identity bin), each a normalised distribution over tokens 1..65;
  `fit.py` fits them on training tables only; `switch.py` runs the chain as a baseline.
- `seeddrift/train/rel/chain/switching.py`: the copying chain with switching (forward, backward, three visibility
  modes, the probability of a stretch as a whole), tested against enumeration.
- Stage `w5_baselines` (kind `rel_tables`, the select lock; gate `G4_rel_w5_baselines`) and profile
  `rel-dev-w5-baselines-i1`.

Today's limits, written into the results: stored layout only (whole-gene numbers are an upper bound); the chain runs
inside the target locus only; scored cells are the whole target block, not yet the 128-codon items of the evaluator.
