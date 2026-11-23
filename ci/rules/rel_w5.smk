# Package W5 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W5. A stub runs ci/stages/rel/stub.py and fails, until the package replaces the rule with its
# stage (same rule name and output; inputs as the stage needs, always including gates/pre.ok) and asks the
# coordinator to drop the stub flag. Rule names rel_<stage>; outputs under data/rel_w5/.

# w5_items: the evaluation items: genome, gene, codons and bases of the stretch, screening or confirmation half
# (ci/stages/rel/w5_items.py; numpy only)
rule rel_w5_items:
    input: "gates/pre.ok"
    output: "data/rel_w5/items/items.json"
    shell: STAGE.format(name="rel.w5_items")

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

# w5_baselines: the classical baselines on the held-out tables of a passed tables run (ci/stages/rel/w5_baselines.py;
# numpy only, no GPU): what the first models have to get under
rule rel_w5_baselines:
    input: "gates/pre.ok"
    output: "data/rel_w5/baselines/results.json"
    shell: STAGE.format(name="rel.w5_baselines")

# w5_sources: what arms S1 and S2 of decision D7 take from their sources, read with the final weights of a passed
# training (ci/stages/rel/w5_sources.py; the model interpreter on the CPU, the GPUs hidden; a diagnosis, nothing trained)
rule rel_w5_sources:
    input: "gates/pre.ok"
    output: "data/rel_w5/sources/report.json"
    shell: STAGE.format(name="rel.w5_sources")
