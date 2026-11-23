# Package W18 of the relational-model line: the implant tables, a ruler set (generator bench/rel/implant.py), and the
# readouts read on them (configs/rel/kinds.yaml lists each stage, its output and its gates). Rule names rel_<stage>;
# outputs under data/rel_w18/. Evaluation only: nothing here trains, and the ruler set is never a training table.

# w18_rulers: the implant tables (kind ruler, purpose ruler)
rule rel_w18_rulers:
    input: "gates/pre.ok"
    output: "data/rel_w18/rulers/tableset.json"
    shell: STAGE.format(name="rel.w18_rulers")

# w18_readout: the readouts and their AUCs from the per-cell files the evaluation of a ruler set wrote (CPU only; no
# model is run); after rel_w6_eval when the profile runs it
rule rel_w18_readout:
    input: ["gates/pre.ok"] + (["gates/rel_w6_eval.ok"] if "w6_eval" in REL_ON else [])
    output: "data/rel_w18/readout/report.json"
    shell: STAGE.format(name="rel.w18_readout")

# w18_offline: the strict offline rendering of a ruler set (protocol w18-v3; CPU only): W3's aligner on every table and
# mask family, W3's fourth validity test on each; run_ruler reads it for a leak-free reading
rule rel_w18_offline:
    input: "gates/pre.ok"
    output: "data/rel_w18/offline/offline.json"
    shell: STAGE.format(name="rel.w18_offline")
