# Package W13 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W13. These are stubs: each runs ci/stages/rel/stub.py and fails, until the package replaces the
# rule with its stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w13/.

# w13_census: the census of the close relatives each group can get (no download)
rule rel_w13_census:
    input: "gates/pre.ok"
    output: "data/rel_w13/census/report.json"
    shell: STAGE.format(name="rel.stub") + " w13_census"
