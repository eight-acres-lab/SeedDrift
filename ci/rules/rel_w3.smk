# Package W3 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W3. These are stubs: each runs ci/stages/rel/stub.py and fails, until the package replaces the
# rule with its stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w3/.

# w3_offline: the strict offline rendering of the evaluation items
rule rel_w3_offline:
    input: "gates/pre.ok"
    output: "data/rel_w3/offline/tableset.json"
    shell: STAGE.format(name="rel.stub") + " w3_offline"
