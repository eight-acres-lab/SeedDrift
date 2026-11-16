"""SeedDrift CI: one versioned, reproducible pipeline for every data step, training run, test and demo.

Layout (see ci/README.md):
  hashing.py   canonical JSON, streaming sha256 with sidecars, stage keys, pipeline hash
  manifest.py  content-addressed dataset manifests
  record.py    the run record (atomic writes, heartbeat)
  taxonomy.py  NCBI taxdump lineages (fail closed on unknown taxids)
  kingdom.py   genome groups for the multi-kingdom quota gate
  gates/       G1 quotas, G2 exclusions, G6 clean code, G7 environment, G8 CI version
  Snakefile    the pipeline; rules/ and profiles/ hold its parts
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def version():
    with open(os.path.join(ROOT, "ci", "VERSION")) as f:
        return f.read().strip()
