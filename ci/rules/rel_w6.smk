# Package W6 of the relational-model line (with the minimal trainer of W4): the self-tests of the torch code, the
# training of the arms with their held-out evaluation, and the report of the curves (configs/rel/kinds.yaml lists each
# stage, its output and its gates). Rule names rel_<stage>; outputs under data/rel_w6/.

# w6_selftest: every test_ function of the torch modules, in the model interpreter with the GPUs hidden
rule rel_w6_selftest:
    input: "gates/pre.ok"
    output: "data/rel_w6/selftest/report.json"
    shell: STAGE.format(name="rel.w6_selftest")


# w6_fit: every arm preflighted, then trained (at most rel.parallel at once); a failed arm does not stop the others.
# When the profile also runs the self-tests, no training starts before they passed.
rule rel_w6_fit:
    input: ["gates/pre.ok"] + (["gates/rel_w6_selftest.ok"] if "w6_selftest" in REL_ON else [])
    output: "data/rel_w6/fit/models.json"
    resources: gpu=1
    shell: STAGE.format(name="rel.w6_fit")


# w6_report: the curves aligned by cells seen, beside the classical baselines
rule rel_w6_report:
    input: ["gates/pre.ok"] + (["gates/rel_w6_fit.ok"] if "w6_fit" in REL_ON else [])
    output: "data/rel_w6/report/report.json"
    shell: STAGE.format(name="rel.w6_report")


# w6_eval: the saved weights of a training run's arms evaluated again (an arm whose final evaluation the wall limit cut
# short); nothing is trained
rule rel_w6_eval:
    input: "gates/pre.ok"
    output: "data/rel_w6/eval/models.json"
    resources: gpu=1
    shell: STAGE.format(name="rel.w6_eval")
