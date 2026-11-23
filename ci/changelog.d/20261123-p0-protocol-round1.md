# P0: protocol round1, and w3-offline-v0 version 2

- `configs/rel/protocols/round1.yaml`: the first selection round on tables of version 1 under the leak-free layout,
  revised after an independent review: the gates that must pass before it starts; leak-free training for every scheme
  (the whole gene on the frame relative with the strict aligner); the reference trunk unchanged (no genome_of_row
  bias before its own arm); batch 1 = a 30 M long arm on i-2 and two seeds plus an arm without relatives on i-1;
  batch 2 = a 100 M long arm alone on i-1; batch 3 = the two D7 arms, a whole-gene-heavy mask mix and the same-genome
  bias, each against a fresh reference seed in the same window; readings for data, seed noise, relatives, scale,
  controls and over-fitting, with a paired bootstrap; the HGT reading under a version 3 of w18.
- `configs/rel/protocols/w3-offline-v0.yaml` version 2: round 1 trains every scheme leak-free with the strict aligner.
