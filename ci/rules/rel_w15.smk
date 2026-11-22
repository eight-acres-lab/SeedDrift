# Package W15 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W15. Rule names rel_<stage>; outputs under data/rel_w15/.

# w15_registry: the event registry, first version, without any score: the draft's checks, NCBI taxonomy of every
# exclusion taxon, genomes inside each radius in the train and held-out splits of the medium corpus (never the sealed
# test split), overlap with the sealed island genera, and with `fetch` the papers' metadata, open full text and named
# supplements from Europe PMC (ci/stages/rel/w15_registry.py; standard library only)
rule rel_w15_registry:
    input: "gates/pre.ok"
    output: "data/rel_w15/registry/report.json"
    shell: STAGE.format(name="rel.w15_registry")
