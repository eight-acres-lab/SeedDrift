# CI: one reproducible pipeline for data, training and evaluation

Every data step, training run, evaluation, test and demo of the model line runs through this pipeline. Each change
to it is a new CI version (`ci/VERSION`, tag `ci-vMAJOR.MINOR`, reasons in `ci/CHANGELOG.md`), and each run leaves
a record from which it can be reproduced.

## Tiers

| Tier | Where | When | What |
|---|---|---|---|
| T0 | GitHub | every push and pull request | lint, repository hygiene, gates G1, G2 and G8 on the specs, Snakemake dry runs |
| T1 | GitHub | after T0 | all tests; the `ci-tiny` profile on the committed fixture, twice, compared byte for byte; a killed and resumed training compared with an uninterrupted one |
| T2 | project server | by hand, `ci/launch.sh <commit> smoke` | the same pipeline on a small real multi-kingdom set (minutes to an hour) |
| T3 | project server | by hand, after a passing T2 of the same pipeline | the full run (days) |
| T4 | project server | by the project lead only | the one-time scoring of the sealed benchmark set |

Real data is only ever processed by T2 and above, from a clean commit that is on GitHub and passed T0. There is no
self-hosted runner: nothing pushed to the repository runs on the server by itself.

## Running

```
python -m ci check                                  # T0
python -m ci run --profile ci-tiny --run-dir runs/a # T1 (any machine with the environment of ci/envs/ci.lock.yml)
python -m ci compare runs/a runs/b                  # reproduction verdict, written into b's record
ci/launch.sh <commit> smoke                         # T2 on the server (needs CI_SSH)
```

`python -m ci run` writes `runs/<run_id>/`: `record.json` (commit, CI version, pipeline hash, configuration hash,
environment lock hash, taxonomy hash, seeds, hardware, each stage with its outputs and their hashes, gate results,
waivers used, metrics, status), `config.resolved.yaml`, `env/`, `gates/`, `logs/`, `data/` and the small
`summary.json`, `summary.md` and `curve.csv`. Records are never deleted.

## Gates

A gate that fails stops the run; there is no force option. The only override is a waiver in `ci/waivers/` (gate,
scope, reason, who approved it, date, expiry); an expired waiver fails, and every waiver used is printed on the report.

| Gate | Checks |
|---|---|
| G1 | the training corpus covers every group of `configs/kingdom_map.yml` (bacteria, archaea, mammals, other vertebrates, invertebrates, plants and algae, fungi, protists, phages, eukaryotic viruses, giant viruses, plasmids); groups come from NCBI lineages, never from file names; a full corpus meets `configs/corpus/floors.yaml`; checked on the spec, on the built corpus and on the ids the training loader reads |
| G2 | nothing in `bench/hard/EXCLUDE.tsv` enters training, by full lineage: sealed genera with their families, the development order, benchmark panel assemblies and their species; unknown taxids fail; the list may only grow; training code may not import the benchmark's generators |
| G6 | the run uses a clean, committed worktree, and every module of this repository comes from it |
| G7 | the environment is the locked one; temporary files and caches stay on the data disk; disk and GPU memory suffice |
| G8 | a change to the CI comes with a new CI version |

Coming in later CI versions: G3 (the sealed set is not on disk before the freeze), G4 (every evaluation reports the
comparators next to the model on the same genes, the released v0.1.0 screen recomputed in the same run) and G5 (the
catalogue of public, independently confirmed transfer events as the final judge; until it exists every report says
"no final judge yet").

## Reproduction

Data stages must reproduce byte for byte (archives are written with fixed timestamps); search results are compared
after sorting; training on a shared GPU must reproduce within the tolerances measured at the first full run, and the
tiny CPU training exactly. Allowed to differ: timestamps, durations, host names, process ids, GPU co-tenant readings.
Never allowed to differ: commit, configuration hash, environment lock, input manifests, gate results.
