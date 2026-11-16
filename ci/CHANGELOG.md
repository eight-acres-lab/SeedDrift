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
