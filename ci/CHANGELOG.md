# CI changelog

Every change to `ci/`, `configs/` or `deploy/envs/` is a new CI version (`ci/VERSION`, tag `ci-vMAJOR.MINOR` on main).
MAJOR: a gate changes meaning, a rule is added or removed, or the run record format changes. MINOR: thresholds,
configurations and fixes.

## 0.1 (bootstrap, in progress)

- Run record (`ci/record.py`), hashing and stage keys (`ci/hashing.py`), content-addressed manifests (`ci/manifest.py`).
- NCBI taxonomy lineages (`ci/taxonomy.py`) and genome groups (`ci/kingdom.py`, `configs/kingdom_map.yml`).
- Gates: G1 multi-kingdom floors (`configs/corpus/floors.yaml`), G2 exclusions by full lineage (`bench/hard/EXCLUDE.tsv`),
  G6 clean committed code, G7 locked environment and resources, G8 CI version. Each with a test that must fail.
- Locked environment `ci/envs/ci.lock.yml`.
- G2_holdout: no evaluation sequence occurs in the training corpus.
- After an independent review: a resume continues the same run only (same commit, clean tree, pipeline, lock and
  profile) and re-decides every gate first; a run passes only with every required gate present and passed; gates
  that cannot run fail; waivers last at most 90 days and only G5 may use scope "*"; G1 compares spec and profile
  tiers and applies the floors to full corpora; G2 requires its rules, follows merged taxids and catches relative and
  dynamic generator imports; G6 checks the requested commit, ignored files and write bits; G8 fails on tags that are
  not ancestors; `--set` is limited to training and thread settings; launchers validate their arguments, require
  passing T0 and T1, take the GPU lock in the foreground and report success only once the record exists.
- Smoke tier (T2, server): stage `fetch` downloads remote sources (ENA assemblies and sequences, release-pinned
  Ensembl and Ensembl Genomes coding sequences) into a content-addressed store and lists them in a fetch manifest;
  the corpus stage reads Ensembl CDS files (protein-coding genes, longest CDS, no organelles), turns ENA names into
  sequence accessions and cuts held-out evaluation regions out of corpus genomes; the reference stage can keep its
  tables in the store (`reference.cache`), keyed by the FASTA, the taxonomy and the stage code, and records an
  in-place DIAMOND database in its manifest. Profile `smoke` with corpus spec `configs/corpus/smoke.yaml`.
- G7_prefix: the paths compiled into a conda environment (the OpenSSL certificate file) must lie inside the prefix
  that runs; the launcher builds environments at their final path with a completion marker (an environment built
  aside and moved could not verify TLS certificates).
- G3 (static in the pre gate, built in the eval gate): no evaluation genome under a sealed genus and no sealed
  accession in the configuration, the evaluation tables, any manifest or the training ids; tier freeze fails until a
  CI version adds the freeze rules.
- Stage `released`: the released screen recomputed in the run from tag v0.1.0 (`git archive`), in its own locked
  environment (`ci/envs/released-v0.1.0.lock.yml`, the deployed tools environment of v0.1.0), with a run-local data
  folder (shared reference read-only, own cache, no services). G4 (new gate phase `report`): every scenario has the
  six rows model, alien_index, best_hit, composition, counts, released_v0.1.0 on one gene set (hash in every row);
  the released row must be recomputed from the tag's commit except in tier tiny, which declares the skip.
- Launcher: `ci/launch.sh reproduce RUN_ID` (same commit and profile, RUN_ID as parent), `compare A B`, `publish
  RUN_ID` (`ci/publish.py`: the three summary files, at most 200 KB, home paths replaced, denylist and sealed
  accessions refused, committed on branch runs/RUN_ID from the run's commit with a row in runs-index/INDEX.tsv);
  release tags travel with each launch. Tests that both launchers refuse unsafe names, unpushed commits, failing
  checks, missing profiles, wrong trees, existing run ids and a held GPU lock.
- After a second independent review: G4 checks per method which genes it scored itself (the others only under the
  method's stated rule), not a hash stamped on every row; G3 matches sealed accessions in any case and inside
  composite ids, and fails when a sealed genus is not in the taxonomy; the released screen gets no SEEDDRIFT_*
  setting of the calling shell; fetch stores only complete downloads (every requested sequence, whole gzip streams),
  with private temporary names and a locked index merge; held-out regions must lie inside their replicon; the
  reference table cache key covers the taxonomy and array-writing code; the launcher refuses unsafe branch names
  (they reach the server command line), counts check runs over all result pages, and the server launcher removes
  the worktree on every early exit (including a failed environment build, now reported); publish accepts only plain
  profile, status and version values for the public index.
