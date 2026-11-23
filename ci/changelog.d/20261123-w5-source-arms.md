# W5: arms S1 and S2 of decision D7 (source structure)

- New model module `seeddrift/train/rel/models/source.py` on W6's common layers: the reference trunk's blocks, the first
  `self_layers` run with every row alone (an own-row branch that never sees another row), the rest across the table.
  S2 is the learned copy model (p = lambda * sum_r w_r sub(codon of row r) + (1 - lambda) p_self); S1 adds a chain of
  sources along each block's columns with a stay bonus from the trunk and an evidence term (forward and backward in
  log space). Both give every target cell its own normalised distribution from the grid alone (factorisation
  independent), so the trainer, the objective `mlm` and the evaluator read them unchanged; `readout()` gives the source
  weights, S1's source path, its posterior given the true codons and the joint bits (read-outs only).
- Variants `w5-src30m-s1` and `w5-src30m-s2` (the leak-free reference arm at 1.5e-4 but for the model); profiles
  `rel-dev-w5-source-selftest-i1` (self-tests, CPU) and `rel-srcsmoke-i1` (smoke test beside the reference arm, GPU).
