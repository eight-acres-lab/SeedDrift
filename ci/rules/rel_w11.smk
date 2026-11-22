# Package W11 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W11. These are stubs: each runs ci/stages/rel/stub.py and fails, until the package replaces the
# rule with its stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w11/.

# w11_rulers: the ruler sets (generators in bench/rel/, called by this stage only)
rule rel_w11_rulers:
    input: "gates/pre.ok"
    output: "data/rel_w11/rulers/tableset.json"
    shell: STAGE.format(name="rel.stub") + " w11_rulers"

# w11_readout: the readouts on the models of a passed training run
rule rel_w11_readout:
    input: "gates/pre.ok"
    output: "data/rel_w11/readout/report.json"
    shell: STAGE.format(name="rel.stub") + " w11_readout"
