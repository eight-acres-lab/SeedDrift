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
- Batch 2 (`bench/rel/registry/events_v0.2-batch2.tsv`, 10 records) for the cells the sample left empty: fungal
  recipients, a fish, a giant virus, a plasmid, recombination between close bacterial relatives, and two more hard
  negatives (host contamination; an overturned genome-wide claim). Reports for the batch and for both drafts
  together (`report_v0.2-batch2.md`, `report_v0.2-all.md`). `registry.py` and the stage read several drafts as one
  registry.
- Fix after the first server run (20261122T151028, passed with nothing fetched): the stage read responses with
  `read(-1)`, which returns the chunk framing of a chunked body, so every metadata file was stored unparsable; it now
  reads the whole body and parses with `strict=False`. The gate now fails when fewer than half the lookups return a
  parsed record. A name found at another rank keeps its taxid when one candidate fits the lineage. The run's
  taxonomy and corpus counts are written into the sample (taxids, NCBI ranks, the train and held-out counts); the
  profile reads both drafts and five more supplements.
- After the second server run (20261122T153204: 110 of 111 DOIs looked up, 17 full texts, 1 of 8 supplements): Europe
  PMC's back end failed intermittently, answering open-access articles with HTTP 500 and supplements with an XML
  errorBean under status 200. Lookups, full texts and supplements are now retried (after 10 s and 30 s); a supplement
  must be a zip archive and a full text an article, else nothing is stored. DOIs keep balanced parentheses. A stated
  taxid settles homonyms such as "environmental samples". Batch 2's taxonomy and corpus counts from the run are
  written into its records.
- After the third server run (20261122T175251, stopped after 2,553 s): a supplement's chunked body was cut short
  (`http.client.IncompleteRead`), which the retries did not cover, and the stage stopped. Retries now cover
  `http.client.HTTPException` (IncompleteRead among them) and every `OSError` (connection resets); a paper still
  failing after the retries is recorded as not fetched, and any other failure of one paper is recorded too
  (`fetch_all`), so one paper never stops the stage.
- Every part that could not be fetched records its error type and the last HTTP status (`error_type`, `status`);
  the report counts them per part (`fetch_failures`). The gate's rule (with fetch on, at least half the lookups must
  return a parsed record) still applies after the guard: papers the guard records count as failed lookups.
- From the fourth server run (20261122T185535: 111 of 111 lookups, 28 full texts, 3 of 8 supplements; Europe PMC
  still answered 34 full-text requests with 500 after the retries): the psbA claim set is regraded A on the supported
  trees of its full text, and the crucivirus record keeps C (the junction window is in a figure and the GenBank
  features, not in the text). A supplement over the size cap now records its error type (`TooLarge`).
- Full texts and supplements of open-access articles now come first from PMC's open-access dataset on AWS Open Data
  (`pmc-oa-opendata`: the latest version's article XML, text and metadata, and its supplementary files without the
  figures), the sanctioned route for automated retrieval; Europe PMC is the fallback. Articles outside the open
  subset are not retrieved automatically (PMC's terms bar automated bulk download from its own site).
