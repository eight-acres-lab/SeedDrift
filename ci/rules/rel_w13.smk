# Package W13 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W13. rel_w13_census runs its stage (ci/stages/rel/w13_census.py). Rule names rel_<stage>; outputs
# under data/rel_w13/.

# w13_census: the close relatives each group can get from the catalogue snapshots of the selection (no download)
rule rel_w13_census:
    input: "gates/pre.ok"
    output: "data/rel_w13/census/report.json"
    shell: STAGE.format(name="rel.w13_census")
