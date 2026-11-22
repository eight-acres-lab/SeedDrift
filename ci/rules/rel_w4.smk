# Package W4 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W4. These are stubs: each runs ci/stages/rel/stub.py and fails, until the package replaces the
# rule with its stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w4/.

# w4_train: the trainer of every arm (and its probe mode on synthetic table sets); writes the model index
rule rel_w4_train:
    input: "gates/pre.ok"
    output: "data/rel_w4/train/models.json"
    shell: STAGE.format(name="rel.stub") + " w4_train"
