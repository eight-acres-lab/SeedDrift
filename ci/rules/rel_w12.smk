# Package W12 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W12. rel_w12_search runs its stage; rel_w12_v1 is still a stub: it runs ci/stages/rel/stub.py and
# fails, until the package replaces the rule with its stage (same rule name and output; inputs as the stage needs,
# always including gates/pre.ok) and asks the coordinator to drop the stub flag. Rule names rel_<stage>; outputs under
# data/rel_w12/.

# w12_search: the search without the caps of stage pairs (ci/stages/rel/w12_search.py): training genomes as the
# database, training and held-out genomes as queries, held-out genomes also against their own proteins; a sample run
# (rel_w12_search.sample) extrapolates the full search; a full run stops at wall_s and is resumed
rule rel_w12_search:
    input: "gates/pre.ok"
    output: "data/rel_w12/search/manifest.json"
    resources: diamond=1
    shell: STAGE.format(name="rel.w12_search")

# w12_v1: tables v1 after the re-search
rule rel_w12_v1:
    input: "gates/pre.ok"
    output: "data/rel_w12/v1/tableset.json"
    shell: STAGE.format(name="rel.stub") + " w12_v1"
