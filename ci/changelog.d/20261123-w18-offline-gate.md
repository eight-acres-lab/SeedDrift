# W18: w18_offline's gate per table and mask family (since CI 0.6)

- After rel-w5's review: gate G4_rel_w18_offline recomputes the plan table by table and requires the placement of
  every mask family the passes use (and of no other); offline.json and the readout page state what the fourth test
  does not cover (which rows a block holds, their kinds and lengths come from the search; only the presence passes
  measure them). Profile rel-dev-w18-offline-v3-i1 (the ruler set of rel-dev-w18-rulers-v3-i1, once it has passed).
