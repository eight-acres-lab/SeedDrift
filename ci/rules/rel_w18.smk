# Package W18 of the relational-model line: the implant tables, a ruler set (generator bench/rel/implant.py), and the
# readouts read on them (configs/rel/kinds.yaml lists each stage, its output and its gates). Rule names rel_<stage>;
# outputs under data/rel_w18/. Evaluation only: nothing here trains, and the ruler set is never a training table.

# w18_rulers: the implant tables (kind ruler, purpose ruler)
rule rel_w18_rulers:
    input: "gates/pre.ok"
    output: "data/rel_w18/rulers/tableset.json"
    shell: STAGE.format(name="rel.w18_rulers")
