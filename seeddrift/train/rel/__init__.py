"""The relational-model line: a general model of how genomes inherit from each other (pretraining, then
post-training); transfer detection is one of its applications. Layout (one owner per file; see each package):

  api.py        the contract between the packages (the programme lead)
  registry.py   finds models, objectives, masks, baselines and readouts by module name (the coordinator)
  sealed.py     the sealed check every data-reading stage calls; handoff.py the table-set manifest and model index
  tables/       building tables (v0_*: W2; render*: W3; v1_*: W12)       data/        reading shards (W4); data/masks/
  trainer/      the trainer (W4)          models/      one architecture per module; models/common/ shared layers (W6)
  objectives/   one objective per module  chain/       the copying-chain library and classical copy models (W10)
  baselines/, evalh/  classical baselines and the evaluator (W5)          readouts/    readouts (W11)
Files that both interpreters run (the CI environment has no torch) never import torch; model modules may. Nothing
here imports the generators of synthetic tables (ci/stages/rel/synth.py) or of rulers (bench/rel/)."""
