# Package W15 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W15. These are stubs: each runs ci/stages/rel/stub.py and fails, until the package replaces the
# rule with its stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w15/.

# w15_registry: the event registry, first version, without any score
rule rel_w15_registry:
    input: "gates/pre.ok"
    output: "data/rel_w15/registry/report.json"
    shell: STAGE.format(name="rel.stub") + " w15_registry"
