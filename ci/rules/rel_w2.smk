# Package W2 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W2. These are stubs: each runs ci/stages/rel/stub.py and fails, until the package replaces the
# rule with its stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w2/.

# w2_dist: distances between genomes and the provisional distance bins (first delivery)
rule rel_w2_dist:
    input: "gates/pre.ok"
    output: "data/rel_w2/dist/report.json"
    shell: STAGE.format(name="rel.stub") + " w2_dist"

# w2_v0: tables v0 from the arrays of a passed run (training and held-out tables)
rule rel_w2_v0:
    input: "gates/pre.ok"
    output: "data/rel_w2/v0/tableset.json"
    shell: STAGE.format(name="rel.stub") + " w2_v0"

# w2_items: the evaluation items of W5 with their context rows and relation sites
rule rel_w2_items:
    input: "gates/pre.ok"
    output: "data/rel_w2/items/tableset.json"
    shell: STAGE.format(name="rel.stub") + " w2_items"

# w2_census: the data census (a run of its own)
rule rel_w2_census:
    input: "gates/pre.ok"
    output: "data/rel_w2/census/report.json"
    shell: STAGE.format(name="rel.stub") + " w2_census"
