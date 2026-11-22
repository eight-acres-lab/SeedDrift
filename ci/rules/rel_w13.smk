# Package W13 of the relational-model line: its stage rules (configs/rel/kinds.yaml lists each stage, its output and
# its gates). Owned by W13. Rule names rel_<stage>; outputs under data/rel_w13/.

# w13_catalog: a wider catalogue snapshot of the public assemblies, metadata only (ci/stages/rel/w13_catalog.py: the
# ENA portal's assembly records of bacteria, archaea and viruses, locked with date and sha256); no sequence
rule rel_w13_catalog:
    input: "gates/pre.ok"
    output: "data/rel_w13/catalog/manifest.json"
    shell: STAGE.format(name="rel.w13_catalog")

# w13_census: the close relatives each group can get from the catalogue snapshots of the selection (no download), and
# a second time from the snapshot of w13_catalog (rel_w13_census.catalog_from; "this run" when the profile runs both)
rule rel_w13_census:
    input: ["gates/pre.ok"] + (["data/rel_w13/catalog/manifest.json"] if "w13_catalog" in REL_ON else [])
    output: "data/rel_w13/census/report.json"
    shell: STAGE.format(name="rel.w13_census")
