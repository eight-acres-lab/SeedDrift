"""Stage selfsup_report (pipeline selfsup): data/selfsup/report.json, the run's results in one place.

  variants     per variant of the model: objectives, seed, width, fraction, and the diagnosis on the held-out corpus
               genomes (heldout_summary.json: the mean log-likelihood ratios of B1 and B2 with the genome against
               without it, and A's mean -log p(own genome) against chance); diagnosis only
  islands      with `islands:` configured: chromosomes, genes, and this line's contrasts on the development island
               chromosomes (data/islands/contrasts.json); without it, absent
Gate G4_selfsup checks it.
"""
from __future__ import annotations

import json
import os

from ci.stages.common import Stage, main_args, write_json


def main():
    run_dir = main_args()
    with Stage(run_dir, "selfsup_report", requires=("pre",)) as st:
        cfg = st.config
        with open(os.path.join(run_dir, "data", "model", "model.json")) as f:
            mj = json.load(f)
        variants = {}
        for name, v in mj["variants"].items():
            with open(os.path.join(run_dir, v["out"], "heldout_summary.json")) as f:
                variants[name] = dict(v, heldout=json.load(f))
        rep = dict(model_id=mj["model_id"], kind=mj["kind"], variants=variants, islands=None, note=mj.get("note", ""))
        if cfg.get("islands"):
            base = os.path.join(run_dir, "data", "islands")
            with open(os.path.join(base, "scores.json")) as f:
                sj = json.load(f)
            with open(os.path.join(base, "contrasts.json")) as f:
                cj = json.load(f)
            with open(os.path.join(base, "index.json")) as f:
                ij = json.load(f)
            rep["islands"] = dict(chromosomes=[g["id"] for g in ij["genomes"]], genes=sj["genes"], index_genes=sum(g["genes"] for g in ij["genomes"]),
                                  dataset_id=ij["dataset_id"], units=cj["units"], evaluated=cj["chromosomes"], contrasts=cj["contrasts"])
        out = st.path("selfsup", "report.json")
        write_json(out, rep)
        st.output("selfsup_report", out)


if __name__ == "__main__":
    main()
