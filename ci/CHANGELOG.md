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
