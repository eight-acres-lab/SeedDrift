# CI changelog

Every change to `ci/`, `configs/` or `deploy/envs/` is a new CI version (`ci/VERSION`, tag `ci-vMAJOR.MINOR` on branch `model`, the line of the model work; `main` holds the released screen).
MAJOR: an existing gate changes what it decides or is removed, a rule of an existing pipeline is added or removed, or the
run record format changes incompatibly. MINOR: new pipeline kinds with their own gates, thresholds, configurations and
fixes.

## 0.3 (branch exp/truth-labels, in progress)

The truth-labels line: a self-supervised model whose training targets are observed facts of real genomes, run as its
own pipeline kind with its own report and gates; nothing of pipeline train changes.
- Pipeline selfsup (`pipeline: selfsup`): corpus, search (usually reused whole), training data, the model
  (`train.model: selfsup_v0`, seeddrift/train/selfsup.py: A genome membership by contrast, B1 synonymous codons from the
  protein, B2 aligned hit residues from the query, each with a genome-agnostic twin and read as a likelihood ratio),
  then stage selfsup_report and gate G4_selfsup (every variant diagnosed on the held-out genomes; with `islands:` every
  island gene scored). No released screen and no scenario report (G4 is not required in this pipeline).
- Stage islands (data, score, report): the 24 development chromosomes of the genomic-island set, their own search with
  the V1 view hidden at search time, the model's arrays, scoring by every variant and by this line's comparators,
  bench/hard/evaluate.py and contrasts paired over units. Gate G3_islands: development chromosomes only, no sealed
  accession anywhere.
- Train stage: `train.model: selfsup_v0` (variants with objectives, width, fraction, trim); an unknown model name now
  fails instead of training the pipeline-check model.
- GPU lease (ci/gpu_lease.py, per machine in <CI root>/host/gpu_lease.json, absent = no-op): another service holding
  the GPU is stopped for the GPU part of a selfsup run and started again by a watchdog once no live holder is left for
  a grace period. G7_resources counts the memory the lease can free (only where a lease is configured).
- Reference: `reference.tables_from` takes a previous run's reference dataset as it is (manifest checked against its
  hash, tables against that run's stage record), for a machine that holds the database but not the FASTA.
- Run record schema: pipeline values search and selfsup.
- `train.python_lock` names the lock of the model's Python (default ci/envs/model-torch.pip.txt, unchanged for the
  source model). New ci/envs/model-torch-venv.pip.txt: the same packages and versions without the 24 Ubuntu system
  packages i-1's interpreter also lists, for a virtual environment built from i-1's packages (machine i-2).
- tests/ci/test_select.py: the pipeline kinds now include selfsup.
- Objective C and stage pairs: the corpus searched against itself (the nearest homolog in each other genome, with
  its alignment); the island stage searches its chromosomes against the same database. C predicts the partner's
  residues and synonymous codons given both genomes' representations (aware) or neither (agnostic).
- `search.reuse_required: true`: a run whose search key matches no reuse candidate fails instead of searching.
- After an independent review (four reviewers, each finding checked by a skeptic): padded positions are masked in
  the amino-acid states; resume restores AdamW's state and the counts, dropout is drawn per step; the model loads the
  training and held-out genomes only; the GPU lease records its state and starts its watchdog before stopping, retries
  a failed start, takes up a restore interrupted by a new run, registers the training children (a killed stage does
  not free the GPU under them), treats BACKOFF and STOPPING as running, and an unreadable configuration counts no
  memory as leasable; the island search keeps 4,000 targets per gene before the view is applied (as the benchmark's
  V1 views) and counts the lists it empties; contrasts use bench/hard/metrics.py; score tables and gate G4_selfsup
  follow each variant's objectives; G3_islands uses the scanner of gate G3 and refuses a corpus genome of an island
  chromosome's genus; reference.tables_from checks the taxonomy; ci/gpu_lease.py is part of the pipeline hash.
- `train.models_from` (pipeline selfsup): the variants trained by another run of the pipeline, imported instead of
  trained, for a machine that does not train them. Imported only when that run's training passed on the same corpus,
  training data, split and pairs with the same variant configurations, its model environment's lock is contained in
  this run's, the files its stage record lists have their hashes, and every variant reproduces its held-out scores
  here (same genes and counts; relative deviation at most `train.recheck_max_rel`, correlation at least
  `train.recheck_min_corr`; the published held-out summary matches the recomputed one). `train.models_wait_s` waits
  for a copy still arriving. model.json records the source.
- `train.device: cpu` (default auto) hides the GPUs from the model's processes (training, the held-out recheck, the
  island scoring) and gives torch `train.cpu_threads` threads; selfsup.py score writes to `score_out` when given.
- Profiles selfsup-pilot-i1 (scoring on the CPU, `needs_gpu: false`: the select lock) and selfsup-pilot-i1-gpu
  (scoring on the GPU, the GPU lock): the island part of the pilot on machine i-1 with the variants of the i-2 pilot
  imported (i-2's network was throttled to about 15 KB/s, below what the 17 GB reference database needs).
- A second review (objective C, the lease rework, the island streaming): the partner genome is hidden when scored as
  in training and its composition never contains the partner gene; island partners exclude the corpus genomes the
  chromosome's view hides, its own taxon and its (merged) genus, before the nearest genomes are chosen; pair rows must
  start in the stored codons; a C run without eligible pairs fails clearly; pair_track checks the CIGAR against the
  residues; pairs memory is bounded; G3_islands is merge-aware, checks the view's taxa and counts identical proteins
  genome by genome; the lease drains and stops outside its state lock, keeps running children registered on release,
  ignores reused pids, counts the grace period from when the last holder went away, gives up after repeated failed
  starts (status shows it), and its watchdog leaves under the lock.
- Profiles selfsup-tiny and selfsup-pilot (machine i-2).
- Round 1 of the new objectives (train.readouts): panel, description-length, geometry, codon-usage, shuffled-genome
  and hiding-depth readouts of trained models on the island chromosomes; train.train_splits (train only by default;
  [train, heldout] for special runs, whose held-out diagnosis is then in-sample); the shared test protocol
  (ci/stages/protocol.py) in the island report, with the benchmark's own scores as comparators and the matched and
  accessory views of benchmark v2 family A; islands.reuse_search_from (the island search of a run with the same key).
- Pipeline codon (`pipeline: codon`, ci/stages/codon.py, seeddrift/train/codon_study.py): the model-free codon-usage
  family on the island chromosomes from passed runs' outputs (checked by sha256), with gates G3_codon (panels and
  manifolds of other genera, train and held-out splits only, none in the test part of any corpus) and G4_codon
  (coverage, reproductions); paired cluster-bootstrap intervals in the protocol (protocol.paired_delta).
- Stage train_data through ci/stages/train_data_or_reuse.py: `train_data.reuse_from` takes the output of a run with an
  equal key whole (datasets, taxdump, settings, code and the helpers that shape the bytes), `reuse_required` fails
  without one; `search.reuse_manifest_only` (pipeline selfsup, with a required training-data reuse only) takes the
  hits manifest without the hit files.
- Pipeline register (`pipeline: register`, ci/stages/register.py, rule bench/hard/V2_REGISTER.md): the register of the
  island reference set for benchmark v2 (parts, declared exposure levels, sealing candidates), metadata only, gate
  G3_register. Run record schema: pipeline values codon and register.
- Pipeline audit (`pipeline: audit`, ci/stages/audit.py): benchmark v2 milestone M0, slim: the old island set's bias
  audit on its development chromosomes (trivial and metadata columns from the checked records, numpy fused baselines
  fitted leave one unit out, a metadata-only shortcut detector, a label-shuffle control, the margin-free cell for
  window scores), gates G3_audit (development chromosomes only, no accession of the register's other parts in any
  output) and G4_audit. The accessory view's negatives: published "none" only (the label column let published
  backbone and "both" genes in). Run record schema: pipeline value audit.
- Stage sealed_registry (pipeline register, with `register.sealed`): the registry of everything sealed for tools that
  publish or move data (the sealed island chromosomes and the benchmark-v2 candidates with aliases and taxids; the
  sealed test part of every corpus split three ways, with assembly accessions, the split code's sha256 and the runs
  that trained on those genomes); checked by G3_register; installed unchanged as host/sealed_registry.json.
- Pipeline family_a (`pipeline: family_a`, ci/stages/family_a.py, rules bench/hard/V2_FAMILY_A.md, thresholds
  configs/benchmark_v2/family_a.yaml): benchmark v2 family A re-cut on the development chromosomes; increment 1: GTDB
  R232 (md5-checked, in the CAS), the chromosomes' GTDB placement, the re-cut same- and other-species panels, three
  graded outgroup genera from the GTDB tree with two genomes each and reserves, their genomes from ENA; gates G3_famA
  (development only, no forbidden genus, family or taxon, focal NCBI order, no score or search import in the label code)
  and G4_famA_taxa. Run record schema: pipeline value family_a.

## 0.2 (in progress)

- Pipeline kinds: a profile names its pipeline (`pipeline: train`, the default, or `select`); each kind has its
  Snakemake target, required gates and resume phases (`PIPELINES` in ci/__main__.py). Existing runs are unchanged.
- Pipeline select: stage `catalog` (snapshots of NCBI Datasets genome and virus reports and Ensembl species lists,
  canonical, complete pages only, kept in the content-addressed store; modes refresh and lock, `ci promote` and
  `ci/launch.sh promote` print a refresh run's lock for the repository), stage `select` (the multi-kingdom corpus:
  every group, one genome per species, diversity first by fixed seed, a reason for every rejected candidate, viruses
  placed and excluded by their hosts, protein-share repair by raising short classes), stages `budget` (proteins, disk
  and search time as ranges, rates measured on passed calibration runs, assumptions with their basis) and `fit` (the
  budget against the free disk; environmental, not compared). Gates G1_select_spec (pre), G1_select, G2_select,
  G3_select. Profiles select-tiny (T1, committed fixture snapshots, run twice and compared) and select-refresh (server).
  A select profile takes its own server lock (`needs_gpu: false`), not the GPU's.
- Groups: a virus with a known host is placed by the host (`virus_hosts` in configs/kingdom_map.yml): archaeal and
  unlisted prokaryote viruses are prokaryote viruses, a tailed phage on a eukaryote host is a conflict. Without a host
  the taxon rule applies as before.
- G2: the exclusion rules are one object (`Excluder`) shared by the gates and the selection; entries may carry a host,
  whose lineage is checked too.
- Versioning convention: a new pipeline kind with its own gates is a minor version; a major version changes what an
  existing gate decides or removes one.
- Launcher: `reproduce` re-invokes the launcher through bash, so it works when the script is not executable (it
  failed with "Permission denied"); the launch scripts are executable in git. Tests for `reproduce`.

## 0.1 (2026-11-17, tag ci-v0.1)

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
