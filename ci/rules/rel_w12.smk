# Package W12 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W12. Rule names rel_<stage>; outputs under data/rel_w12/.

# w12_search: the search without the caps of stage pairs (ci/stages/rel/w12_search.py): training genomes as the
# database, training and held-out genomes as queries, held-out genomes also against their own proteins; a sample run
# (rel_w12_search.sample) extrapolates the full search; a full run stops at wall_s and is resumed
rule rel_w12_search:
    input: "gates/pre.ok"
    output: "data/rel_w12/search/manifest.json"
    resources: diamond=1
    shell: STAGE.format(name="rel.w12_search")

# w12_v1: tables v1 from a passed full run of rel_w12_search (ci/stages/rel/w12_v1.py), training and held-out parts
rule rel_w12_v1:
    input: "gates/pre.ok"
    output: "data/rel_w12/v1/tableset.json"
    shell: STAGE.format(name="rel.w12_v1")

# w12_islands: tables v1 with the island set's development chromosomes as row 0, never for training
# (ci/stages/rel/w12_islands.py; task breakdown 8.8)
rule rel_w12_islands:
    input: "gates/pre.ok"
    output: "data/rel_w12/islands/tableset.json"
    shell: STAGE.format(name="rel.w12_islands")

# w12_tstats: how often the target genes of a built set of tables are used, per group (ci/stages/rel/w12_tstats.py;
# reads the tables, writes none)
rule rel_w12_tstats:
    input: "gates/pre.ok"
    output: "data/rel_w12/tstats/report.json"
    shell: STAGE.format(name="rel.w12_tstats")

# w12_compo: who the training and held-out genomes of a few groups are, with each held-out genome's nearest training
# genome (ci/stages/rel/w12_compo.py; reads table sets, the taxonomy and codon arrays; writes no table)
rule rel_w12_compo:
    input: "gates/pre.ok"
    output: "data/rel_w12/compo/report.json"
    shell: STAGE.format(name="rel.w12_compo")

# w12_v1x: tables v1 of the held-out genomes a reference table set did not have (round 2's new held-out genomes), an
# evaluation set of their own, never for training (ci/stages/rel/w12_v1x.py)
rule rel_w12_v1x:
    input: "gates/pre.ok"
    output: "data/rel_w12/v1x/tableset.json"
    shell: STAGE.format(name="rel.w12_v1x")
