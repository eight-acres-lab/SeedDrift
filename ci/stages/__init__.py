"""Stage scripts of the pipeline. Each is run by a Snakemake rule as `python -m ci.stages.<name> <run_dir> [args]`,
checks at start that every module of this repository comes from the run's worktree (G6), and ends with a stage report
(ci/record.py stage_report) that lists its outputs and their hashes."""
