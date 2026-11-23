# W5: the classical baselines on tables of version 1, read on the strict offline rendering

- Stage w5_baselines takes `offline` {run, path, sha256}: a passed w3_offline run of the held-out set. Every condition
  with relatives whose mask has a part there (scattered, suffix, whole_gene, whole_gene+repaired) is scored on that
  part, layout leakfree (the cells checked to be those of the stored tables); the conditions without relatives on the
  stored tables. The result rows carry each condition's layout; the gate checks it. The fit stays on the stored
  training tables (a limit written into the report).
- evalh/layout.masked_view reads a block laid on its frame relative (the whole gene in the strict rendering): row 0 is
  laid from the frame's first column, so its columns stay its own positions.
- Profile rel-dev-w5-baselines-v1-i1: the 3,000 held-out tables of version 1, round 1's final conditions.
