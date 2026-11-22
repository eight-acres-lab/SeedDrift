# Package C0 (the coordinator) of the relational-model line: the development stage that writes synthetic table sets as
# ordinary shards (ci/stages/rel/c0_synth.py; generator ci/stages/rel/synth.py, which no train-side code imports).

rule rel_c0_synth:
    input: "gates/pre.ok"
    output: "data/rel_c0/synth/tableset.json"
    shell: STAGE.format(name="rel.c0_synth")
