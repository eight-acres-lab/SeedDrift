# W5: the three relatives strata of round 1 (evaluator version 4)

- `evalh/score.strata_of(identity, other_rows, kind)`: given the kinds (N, R, B) of the tables, the nearest identity
  (strata nearest_identity=..., relatives_on_target=...) is that of the nearest other-genome row, self-copy rows
  (KIND_SELF_COPY on some block) left out, and the strata `relatives=other_genomes`, `relatives=self_copies_only`,
  `relatives=none` are added (round 1's protocol). Without kinds, as before.
- The classical baselines pass the kinds of the stored tables.
