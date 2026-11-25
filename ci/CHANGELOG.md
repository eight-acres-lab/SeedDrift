# CI changelog

Every change to `ci/`, `configs/` or `deploy/envs/` is a new CI version (`ci/VERSION`, tag `ci-vMAJOR.MINOR` on branch `model`, the line of the model work; `main` holds the released screen).
MAJOR: an existing gate changes what it decides or is removed, a rule of an existing pipeline is added or removed, or the
run record format changes incompatibly. MINOR: new pipeline kinds with their own gates, thresholds, configurations and
fixes.

## 0.8 (branch exp/rel-corpus-ready-20261125, 2026-11-25: preserve held-out table indices)

The uncapped W12 search accepts `output_order_from`, a passed version-1 table set pinned by its manifest digest.
Its final arrays and lookup index retain that set's complete genome-index prefix, with new genomes appended.
The DIAMOND database, query order, call boundaries and resumable part files keep the source order. Consolidation
renumbers and sorts target pairs after the main and self searches. The gate verifies the reference and final order.
This permits byte-for-byte reuse of the original held-out tables when expanded source IDs sort before old IDs.
Regression coverage checks identical search inputs, alignments, self copies, island indices, lookups and resumed
output arrays, and rejects changed, missing, duplicate or unverified reference metadata.

## 0.7 (branch exp/rel-corpus-resume-20261125, 2026-11-25: damaged archive retries)

NCBI Datasets ZIP validation now normalizes DEFLATE errors and truncated-member EOFs into the existing damaged
archive retry path. A valid ZIP directory with a damaged compressed member is retried up to three times. A file
is stored only after validation succeeds. After three failures, the configured missing-source policy determines
whether the download fails or records an unavailable source. Regression tests cover recovery and both policies.

## 0.6 (branch exp/rel, 2026-11-23: round 1's foundation)

Round 1 on tables of version 1 under the leak-free layout (protocol round1, version 3), the HGT capability check's
version 3, and the packages' work since 0.5, folded here from their change notes. Nothing of the existing pipelines,
gates or stages changes what it decides.
- Protocols (P0; `configs/rel/protocols/`): round1 version 3 (the gates before batch 1; leak-free training for every
  scheme with the whole gene on its frame relative by the strict aligner; the reference trunk unchanged; batches 1 to 3;
  the readings with a paired bootstrap; 201,600 training tables of version 1), w18-v3 (the counting reading of the HGT
  capability check on round 1's arms), w3-offline-v0 version 3 (the controls spare row 0 only; what the fourth validity
  test does not cover). The contract's note on self-copy rows follows the version-1 builder; the API stays 1.
- The reference trunk on version 1 (W6): `whole_gene_layout: frame` (W3's `align.frame_layout` after the rows are drawn,
  no pair index), the evaluation online at the nodes (`leakfree-online`) and on the strict offline rendering at the end,
  `only_conditions`, `nodes_every` and `nodes_every_from`, node weights (`keep_node_weights`, `ema_tables<n>.pt`),
  `data_workers`, the mask schemes' shares of cells and loss, `preflight_only` probes, the same-genome bias as an option
  (`genome_bias`, off in batches 1 and 2), the genome of each row passed only to a model that reads it; variants
  `w6-ref30m-v1`, `w6-ref30m-norel-v1`, `w6-ref100m-v1`.
- The renderer's controls (W3, renderer version 2): shuffle and presence act on every row but row 0, also on a block
  laid on its frame relative. The strict offline rendering reads tables of version 1 and ruler sets, in several
  processes with the same output; profile `rel-dev-w3-offline-v1-i1` (round 1's gate).
- Tables of version 1 (W12): every genome's target genes in a shuffled order before any repeats; stages w12_search and
  w12_v1 read a source without stage pairs; an optional subset of the search's pair index carried with a table set
  (`frame_index`, off); profiles `rel-dev-w12-full-k3-i1` (the uncapped search on corpus medium-k3) and
  `rel-dev-w12-v1-k3-i1` (201,600 training tables, 16,800 per group).
- The HGT capability check, version 3 (W18): stage `w18_offline` (the strict placements of every ruler table and mask
  family, the fourth test on each), the leak-free reading on them, self-copy rows, relatives layers, the two verdicts,
  the 2 x 2 table; the donor search of the implant's gene against the examined genome's own proteins; the adapted
  reading of version-1 rulers by arms of version 0 (`adapt_v0_arms`, a diagnosis); exploratory readouts.
- D7's source-structure arms (W5): `seeddrift/train/rel/models/source.py` (S1: a chain of sources with a stay bonus and
  an evidence term; S2: a learned copy model), variants `w5-src30m-s1`, `w5-src30m-s2`, their self-test and smoke
  profiles.
- The event registry (W15): batches 3 and 4, full texts and supplements from PMC's open-access data on AWS first, every
  network error retried, unverified relayed statements removed.
- Close relatives (W13): the pick and build with the programme lead's rule for records the archive cannot give; corpus
  spec `medium-k3` (526 relatives) and its parts.
- The coordinator (C0): stages w13_pick, w13_build, w13_scaffold, w18_offline registered; w15_registry no longer a stub;
  the lock calendar's version 6 with the measured speed of a search beside trainings.
- Round profiles of round 1: `rel-r1probe-i1` and `rel-r1probe-i2` (the start gate's 50-step probes), `rel-r1short-i1`
  (batch 1 on i-1: ref30m-s2, ref30m-s3, norel30m), `rel-r1long-i2` (batch 1's ref30m-long), `rel-r1long100m-i1`
  (batch 2's ref100m-long). The tables' and the offline rendering's runs are filled in once they are built.

## 0.5 (branch exp/rel, 2026-11-23: round 1b's foundation)

The packages' work since the foundation, merged into exp/rel and folded here from their change notes; the round
profiles of round 1b. Nothing of the existing pipelines, gates or stages changes what it decides; stage fetch gains two
switches that are off by default.
- Contract version 1 (package P0; `seeddrift/train/rel/api.py`, `API` = 1): the vocabulary of 73 tokens (a target is a
  residue, tokens 1 to 65), the 19 blocks of a row of 1,024 slots, the kinds of a (row, block), the arrays of a shard
  and those no model reads, layouts, conditions and result-row columns; `col` 0-based inside a block of the frame row.
  Additions for tables of version 1 (API stays 1): `SHARD_FIELDS_ADDED` (`genome_of_row`), `META_FIELDS_ADDED`
  (`row_gene`, `row_first`), `KIND_ABSENT`, `KIND_SELF_COPY`, at most 2 rows per genome, provisional distance tiers.
- Protocols (P0; `configs/rel/protocols/`): w18-v0, the pre-registered HGT capability check on implant tables, and
  w3-offline-v0, the strict offline rendering of evaluation items under the leak-free layout.
- Tables of version 0 and their census (W2; `seeddrift/train/rel/tables/v0_*`, stages w2_v0, w2_census, w2_dist):
  genomes x loci from a passed selfsup run's training arrays and homolog rows, plain `.npy` shards in the contract's
  fields, bit scores dropped before anything is read; the census of fill, nearest-row identity and codons; MinHash
  distances between genomes for sampling and reporting only.
- Evaluation and baselines (W5 with W10; `seeddrift/train/rel/evalh/`, `baselines/`, `chain/`; stages w5_items,
  w5_baselines): the five conditions (scattered, whole_gene, both without relatives, whole_gene with the rows re-paired)
  and the conditions only an evaluation uses (suffix, shifted correspondence, synonymous recoding), every random choice
  a 64-bit hash; the stored layout in numpy; bits per group and stratum; the classical baselines (own codon frequencies,
  the rows' average, copy the nearest row) and the copying chain with switching; the evaluation items.
- The reference trunk and its trainer (W6 with W4; `seeddrift/train/rel/models/axial.py`, `data/`, `objectives/`,
  `trainer/`; stages w6_selftest, w6_fit, w6_eval, w6_report): the axial trunk in sizes 10m, 30m, 100m; bidirectional
  fill-in with warm-up then a constant rate to the wall limit; evaluation by the weights' moving average at doubling
  tables seen and at the end; a preflight as the divergence guard; the leak-free layout through package W3's renderer,
  a leak-free arm scored on W3's strict offline rendering (`tables.heldout_offline`) and its whole-gene conditions also
  in the stored layout (`<condition>@stored`); `extra_conditions`; w6_eval's ruler mode for W18; the report's curves at
  equal tables seen, the ladder over two seeds, the learning-rate check and sweep, curves by pass, `notes`.
- The uncapped search and tables of version 1 (W12; `seeddrift/train/rel/tables/v1_hits.py`, `v1_build.py`; stages
  w12_search, w12_v1, w12_islands): every pair once with all its alignments, no score, e-value or rank stored; the 1 %
  trial's linear extrapolation; the island set's 24 development chromosomes as extra queries (the 17 sealed never) and
  their tables in a set marked `not_for_training`; a memory-mapped lookup index of the rows.
- Close relatives (W13; stages w13_catalog, w13_scaffold, w13_census, w13_pick, w13_build): a wider ENA catalogue
  snapshot (one record per accession, the highest version), NCBI's GenBank assembly summaries for the scaffold level
  judged by quality thresholds, the census counted on three snapshots (twins counted once, virus hosts from the
  selection's host records), the first batch of relatives by name (the user's k = 3; each pick's ENA record and FASTA
  tried, candidates in a seeded random order, replacements listed, at most 5 % not given), and the corpus build in
  parts (`rel_w13_build.upto`) with its checks of the pinned splits, the relatives' units and the sealed check.
  `configs/corpus/medium-k3.yaml` and its two part specs.
- Stage fetch: `fetch.retry_s` (waits between tries) and `fetch.server_errors_unavailable` (a server error or no
  connection through every try is listed as unavailable under `fetch.allow_missing`); both off by default, so every
  other profile behaves as before (the programme lead's rule of 2026-11-23).
- The event registry, first version, no scores (W15; `bench/rel/registry/`; stage w15_registry): rules 0.2, a sample of
  20 records and a batch of 10; the server checks (NCBI taxonomy of every exclusion taxon, genomes inside each radius
  in the train and held-out splits, overlap with the sealed island genera) and the papers' metadata, open full texts and
  named supplements from Europe PMC, every network error retried and a paper still missing listed with its error type.
  Nothing is written to `bench/catalogue/events.tsv`.
- The HGT capability check's ruler set (W18; `bench/rel/implant.py`, `seeddrift/train/rel/readouts/`; stages
  w18_rulers, w18_readout): implant triples of held-out genomes, kind ruler, never trained on; the readouts' passes and
  statistics, the readout over several evaluation runs.
- The coordinator (C0): every stage above registered in `configs/rel/kinds.yaml`; the field `reads` (the table kinds
  a stage may read; a train-side stage never reads ruler sets; `common.read_tableset`); island rows only in sets marked
  `not_for_training`, checked by `common.development_chromosomes`, refused as input of every training stage; the lock
  calendar and state in `configs/rel/coordinator/`.
- Round profiles of round 1b: `rel-r1bshort-i1` (the 10 M and 30 M arms on the leak-free layout and the hierarchical
  output head on the same layout, three at once, 4 h each) and `rel-r1blong-i2` (the 30 M arm on the leak-free layout at 1.5e-4, 11 h).

## 0.4 (branch exp/rel, 2026-11-22: the foundation)

The relational-model line: a general model of how genomes inherit from each other (pretraining, then post-training),
built by several packages in parallel on the integration branch exp/rel. This version is its foundation; nothing of the
existing pipelines, gates or stages changes what it decides.
- Pipeline kinds rel_tables, rel_traineval, rel_readout, rel_corpus and rel_events, registered in configs/rel/kinds.yaml
  (lock, gates, every stage with its package, rule, output, module, gates and side; ci/stages/rel/kinds.py). A profile
  names its kind and the stages it runs (`rel.stages`); its target, required gates and resume phases follow from them
  (ci/__main__.py `pipeline`). The run record schema takes pipeline names rel_<name>.
- ci/Snakefile includes every ci/rules/rel_*.smk (one file per package) and adds the target rule rel_all, the gate
  rule of every stage (gate phase rel_<stage>, decided by the stage module's `gate` function) and of the kind (phase
  <kind>: G3_rel, the sealed check on every genome of every table set the run wrote, and G4_rel_handoff, every
  hand-off file complete and matching its files). ci/stages/gates.py hands these phases to ci/stages/rel/gates.py.
  Stub rules for the stages of packages W2, W3, W4, W5, W11, W12, W13 and W15 (ci/stages/rel/stub.py fails).
- Hand-off files between packages (seeddrift/train/rel/handoff.py, standard library): the table-set manifest (kind
  real, synthetic or ruler; coverage; files with their roles; every genome of a row) and the model index.
- The sealed check (seeddrift/train/rel/sealed.py, both interpreters; configs/rel/sealed.json): the 257 genomes of the
  medium corpus's sealed test split (the pinned training-data index, sha256 and count checked, cross-checked against
  the machine's sealed registry where it exists), their assemblies, the sealed chromosome accessions; and the lineage
  half in the CI environment (ci/stages/rel/common.py: gate G2's rules plus the provisional recipients list).
- "This configuration belongs to this machine" (ci/stages/rel/common.py check_machine, configs/rel/machines.json): every
  rel stage starts by checking the GPU's name and memory, the CI root and the trainings at once (RelStage).
- The run-summary hook (ci/stages/rel/summary.py) adds each stage's own summary part; `ci compare --expect
  within_tolerance --protocol FILE` judges the outputs of stages marked `tolerance` with the function at the fixed path
  seeddrift/train/rel/trainer/tolerance.py (a stub that answers "undecided" until package W4 writes it).
- seeddrift/train/rel/: the package folders, registry.py (models, objectives, masks, baselines and readouts by module
  name), the stub arm models/uniform.py (uniform over the 65 target symbols), api.py as a placeholder for the contract.
- The generator of synthetic tables (ci/stages/rel/synth.py, numpy): random symbols, 16 to 32 rows copied from each
  other at 0.1% to 30% substitution, the two capability tasks (whole locus masked; source switch inside the locus) and
  the correct sources for tests only; fixtures in tests/ci/rel_fixtures.py. Train-side code never imports it.
- Profiles: one dry-run profile per kind (rel-dev-c0-dry-*, static check only).
- Tests (tests/ci/test_rel_c0_*.py): the registry and its rules, every rel-dev-* profile with `needs_gpu: false` and
  the CPU, every rel profile without the GPU lease and on a known machine, no rel module importing ci/gpu_lease, the
  train side free of generator imports, the guards, the gate dispatch, the summary hook and the tolerance comparison.
- configs/rel/ (variants, protocols, requests, coordinator state) and ci/changelog.d/ for the packages' change notes.

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
- Pipeline family_a, increment 2 (V2_FAMILY_A.md part 2, fixed before its first run): presence calls of every panel
  and outgroup genome (DIAMOND blastx of genome DNA in 200 kb windows, frameshift-aware, every HSP, against the unit's
  development proteins; loci merged per protein, places owned by the best score; ortho, hom, og and pos calls) and the
  labels from the genomes' own evidence only (S1/S2 runs, outgroup polarity, the empty-site test, the edge margin N
  from the literature islands' edges; classes insertion and core, the rest excluded; the published labels only as an
  agreement column); gate G4_famA_labels re-checks every insertion and core gene against its rule from the label
  table. Part 1 amendment of 2026-11-21 (the first run's G4_famA_taxa failed on GTDB R232's split of E. coli): a
  chromosome whose GTDB genus offers fewer than 3 other species is walked up the tree from an anchor genus (the user's
  rule); outgroup genera of reduced genomes are not eligible; at least 2 outgroup genera are required. The taxa stage
  takes the GTDB files from the CAS when the profile pins their sha256 (famA.gtdb_known).
- Pipeline family_a, part 2 amendment after the first labels run: the edge margin N is the median of the matched
  literature edge distances (was the 0.9 quantile, which reached the 10 kb cap and left positives only in segments of
  20 kb or more); a site with flanks more than 100 kb apart is broken whatever the strands. Pipeline selfsup: rule
  gates_islands takes the corpus manifest as input (G3_islands reads it; a reused island search could run the gate
  before the corpus existed).
- Pipeline family_a_eval (`pipeline: family_a_eval`, ci/stages/family_a_eval.py, rules bench/hard/V2_FAMILY_A.md
  part 3, written before its first run): on the labels of a passed family_a run, element clusters of the insertion
  segments (reciprocal blastp 80/80, IS-like genes linking nothing) and the merge-rule units, computed from the labels
  alone; then the arms (main: insertion against 1:1 matched core on length, homolog rows and best identity; core;
  noise; v1, the old island set reproduced against the audit report), the matching ladder, AUROCs per chromosome,
  unit and overall with the protocol's cluster-bootstrap intervals, contrasts of every model column against four
  comparators, published strata, and lk_og (the outgroup genera visible in the hits). Gates G3_famA_eval and
  G4_famA_eval (balance in every unit, own-covariate intervals, shuffle at 0.5 over 200 draws, exact reproduction).
  Run record schema: pipeline value family_a_eval.
- Pipeline family_a_eval, part 3 amendment after its first run (the user's decision): the leak columns lk_og and
  lk_og_id are negated (fewer outgroup genera with hits, and a lower best identity, rank as more island-like), so the
  flag above 0.6 can mark a leak; the threshold and the column it reads are unchanged.

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
