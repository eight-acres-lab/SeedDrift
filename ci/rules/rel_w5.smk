# Package W5 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W5. These are stubs: each runs ci/stages/rel/stub.py and fails, until the package replaces the
# rule with its stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w5/.

# w5_items: the evaluation items: genome, gene, base range, screening or confirmation half
rule rel_w5_items:
    input: "gates/pre.ok"
    output: "data/rel_w5/items/items.json"
    shell: STAGE.format(name="rel.stub") + " w5_items"

# w5_eval: the evaluator on the models of the model index
rule rel_w5_eval:
    input: "gates/pre.ok"
    output: "data/rel_w5/eval/results.json"
    shell: STAGE.format(name="rel.stub") + " w5_eval"

# w5_report: the report of the round
rule rel_w5_report:
    input: "gates/pre.ok"
    output: "data/rel_w5/report/report.json"
    shell: STAGE.format(name="rel.stub") + " w5_report"
