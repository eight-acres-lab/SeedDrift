Package W15: the event registry, rules version 0.2 (draft) and a sample of 20 records for review. No scores.

- `bench/rel/registry/RULES.md`: the fields of a record (one claimed event, not one gene), the scenario tags and the
  position evidence of each scenario, grades A-D for "a transfer happened" and a separate donor grade, statuses
  (standing, replicated, disputed, overturned, retracted, narrowed), probability intervals that never reach 0 or 1,
  the exclusion radius (at least the genus) and the groups (overlapping radii, a shared paper, a shared assembly).
  Every role stays `unassigned` until a seeded hash runs.
- `bench/rel/registry/registry.py` (standard library only): checks a draft against the rules (intervals taken from
  the rules' table, radius rank, roles), computes the groups and writes the reports: the association graph by
  exclusion lineage, counts per grade and per cell of direction x recipient group, scenario tags, the share of
  records with recheckable evidence, and the agreement of a double registration.
- `bench/rel/registry/events_v0.2-sample.tsv` (20 records: 16 claimed transfers across all seven directions, 4
  overturned claims as hard negatives), `double_v0.1-sample.tsv` (8 records registered twice), and the generated
  `report_v0.2-sample.md`.
- Nothing is written to `bench/catalogue/events.tsv`; a test keeps that path empty. Tests:
  `tests/ci/test_rel_w15_registry.py`.
- Rules 0.2 (the same day, after review of the sample): grades A and B need evidence of foreign origin, and in
  replacement scenarios a named recombination test stands in for the phylogeny of grade B (both provisional, pending
  a change to the charter's grade definitions); FISH and genetic mapping count as position evidence; a virus chimera's
  position evidence is a molecule spanning the junction within a stated window; discovery routes S, P, N; columns
  `ev_origin` and `full_text`; disputed, overturned and grade-D records and claim sets stay ungrouped. The sample is
  regraded; the double registration of rules 0.1 is kept as recorded.
- Stage `w15_registry` of kind `rel_events` (`ci/stages/rel/w15_registry.py`, rule `rel_w15_registry`, gate
  `G4_rel_w15_registry`; standard library only), replacing the stub rule: the draft's checks; every exclusion taxon
  looked up in NCBI taxonomy (taxid, rank, lineage, names NCBI lacks); genomes inside each radius in the train and
  held-out splits of a training-data index (the sealed test split is never looked at); overlap with the sealed island
  genera of `bench/hard/EXCLUDE.tsv` and their families; with `fetch`, Europe PMC metadata, open full text and named
  supplements (size-capped) of every DOI in the draft, each stored with URL, size and sha256 in the run's data folder.
  Profile `rel-dev-w15-registry-i1`; request `configs/rel/requests/20261122-w15-registry-unstub.yaml`; tests
  `tests/ci/test_rel_w15_stage.py`.
