# W12: round 2's tables (the held-out part of round 1 copied; the new held-out genomes in a set of their own)

- Stage `w12_v1`: `heldout_from` ({run, sha256}: a passed table set of version 1) copies that set's held-out part file
  by file (each copy hashed again) instead of building it, when the same groups and rows and every genome its tables
  name at the same place of this search's genome list; the part records `copied_from`. With `heldout_equals` naming
  the same set the gate checks it again (the programme lead, 2026-11-24: round 2's 3,000 held-out tables are run
  20261123T122205's).
- Stage `w12_v1x` (`ci/stages/rel/w12_v1x.py`, the code of w12_v1, `WHICH["x"]`): the held-out genomes that the table
  set `new_since` ({run, sha256}) did not have are the examined genomes of a set of their own under data/rel_w12/v1x/,
  marked purpose heldout and not_for_training, its manifest naming the reference (`new_since`). Gate G4_rel_w12_v1x:
  as w12_v1's, and no examined genome is one the reference set had. Request 20261124-w12-v1x asks C0 to register it.
