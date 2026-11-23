# W6: training on the strict rendering (round 1's third batch on)

- Variant `layout: strict` (with `whole_gene_layout: frame`): the scattered and span samples are laid by package W3's
  strict aligner on the rows drawn and the cells hidden (`tables/strict.py` render, aligner version 2), then rendered
  leak-free; the whole gene stays on its frame relative. The nodes read those conditions the same way online (rows
  `leakfree-strict`); the end reads W3's offline rendering (`leakfree`).
- W3's fourth test (`strict.fourth_test`) on the sample numbered 0 and every 1,000th (by its number over the training,
  so the same samples with or without data workers): `steps.csv` `traincheck_checked` and `traincheck_failures`, the
  summary's `traincheck`; `G4_rel_w6_fit` fails an arm with a failure. The preflight reports the strict rendering's cost
  per table.
