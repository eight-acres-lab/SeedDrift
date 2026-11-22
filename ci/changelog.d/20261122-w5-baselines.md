Package W5 (with the copying chain of W10), first slice: the evaluation conditions and classical baselines on the
held-out tables.

- `seeddrift/train/rel/evalh/` (the one implementation; the trainer's held-out evaluation calls it):
  `conditions.py`, which cells of the examined row are hidden and scored and which rows are given, under the five
  conditions scattered, whole_gene, both without relatives, and whole_gene with the rows re-paired. Every random
  choice is a 64-bit hash of (seed, stream, table, index), never a numpy generator, and the plan of a fixed batch is
  pinned. Scattered: 15% of the target block's residue cells, at least one per table. Re-paired: one random cycle
  through the other rows that exist (by their rank, so leaving out absent rows changes nothing), the same in every
  non-target block; a table with fewer than two other rows has no such condition. `table_condition` plans one table,
  `subset` gives the tables of the evaluation during training (the first 50 of each group), `ItemsDigest` the sha256
  of what was scored. `layout.py`, the stored layout in numpy; `score.py`, per-cell bits to rows of the result table
  (per group, then the mean of the groups; strata by the identity of the nearest row in ten bins, and by the number
  of other rows); `shards.py`, reading the tables of a table-set manifest, every file checked against it.
- `seeddrift/train/rel/baselines/`: own codon frequencies, the average of the rows, copy the nearest row (three
  per-position error rates; a variant fitted per identity bin), each a normalised distribution over tokens 1..65;
  `fit.py` fits them on training tables only; `switch.py` runs the copying chain as a baseline (both sides, the left
  only, four columns ahead, and the stretch read as a whole).
- `seeddrift/train/rel/chain/switching.py`: the copying chain with switching (forward, backward, three visibility
  modes, the chain-rule terms of a stretch), tested against enumeration.
- Stage `w5_baselines` (kind `rel_tables`, the select lock; gate `G4_rel_w5_baselines`) and profile
  `rel-dev-w5-baselines-i1`.
- The evaluation items: `evalh/items.py` and stage `w5_items` (gate `G4_rel_w5_items`, profile
  `rel-dev-w5-items-i1`): per group 100 stretches of at most 128 codons inside the stored part of held-out genes, the
  genomes of a group taking turns, halves (screening, confirmation) by genus, every choice a 64-bit hash; read from the
  training-data arrays of the source run, the held-out split only, after the sealed check.

- Conditions only an evaluation uses (`conditions.EXTRA_CONDITIONS`, outside today's five): suffix (the second half
  of the target gene hidden), the other rows' correspondence moved 1, 3 or 12 codons, and the whole gene hidden with
  the other rows' relation blocks recoded by synonymous codons; the stage takes a list of conditions (profile
  `rel-dev-w5-extra-i1`). Bits are also reported in two parts, the amino acid and the codon given the amino acid.

Today's limits, written into the results: stored layout only (whole-gene numbers are an upper bound); the chain runs
inside the target locus only; scored cells are the whole target block, not yet the 128-codon items of the evaluator.
