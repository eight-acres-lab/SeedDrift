#!/usr/bin/env python
"""Run the test chromosomes of the genomic-island reference set (protocol: README.md).

  proteins  the proteins of every chromosome exactly as the pipeline names them, one FASTA per chromosome and
            one combined file (ids "<accession>__<gene>") for the deep search behind the placement model
  jobs      every chromosome through the SeedDrift API, twice: default settings, and home_rank=family.
            The genome's own taxon is given as the genus of the record's organism, as in the other benchmarks.
  scores    per-gene scores of the finished jobs, one folder per configuration
            (<dir>/<set>_pipeline/<configuration>/scores_<accession>.tsv, read by evaluate.py):
              tier      3 / 2 / 1 for a high / medium / low confidence call, 0 for a gene that is not called
              category  the call (mammal, prokaryote, eukaryote, virus), or native / orphan / repeat
              rules     the tier, then the Alien Index within a tier (called genes only)
              ai        the Alien Index of the called genes
              ai_all    the Alien Index of every gene that has one
              comp_abs  the pipeline's own composition column, |z|
            A gene without a value in a column is left empty and ranks last for that score.

--set dev runs the development chromosomes (results/dev_genomes.tsv, <dir>/dev_genomes) instead of the test ones.

usage (repo root, on the machine with the data store and the API; needs the seeddrift package):
  python bench/islands/run.py proteins
  python bench/islands/run.py jobs --api http://127.0.0.1:8600
  python bench/islands/run.py scores
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))

CONFIGS = {"default": {}, "family": {"home_rank": "family"}}
SETS = {"test": ("genomes.tsv", "genomes", "jobs.json"), "dev": ("dev_genomes.tsv", "dev_genomes", "jobs_dev.json")}
TIER = {"high": 3, "medium": 2, "low": 1}
FOREIGN = ("mammal", "prokaryote", "eukaryote", "virus")


def genomes(out, which="test", accessions=()):
    rows = list(csv.DictReader(open(os.path.join(out, SETS[which][0])), delimiter="\t"))
    return [r for r in rows if not accessions or r["accession"] in accessions]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def write_scores(genes_tsv, dest):
    """genes.tsv of one job -> the score columns of this benchmark."""
    f = lambda v: "" if v != v else "%.4f" % v
    with open(dest, "w") as fo:
        fo.write("gene_id\ttier\tcategory\trules\tai\tai_all\tcomp_abs\n")
        for r in csv.DictReader(open(genes_tsv), delimiter="\t"):
            ai, called = num(r.get("AI")), r.get("category") in FOREIGN
            tier = TIER.get(r.get("confidence") or "", 0) if called else 0
            ok = called and ai == ai
            fo.write("\t".join([r["gene_id"], str(tier), r.get("category") or "", f(tier * 1e4 + min(max(ai, 0.0), 9999.0)) if ok else "",
                                f(ai) if ok else "", f(ai), f(abs(num(r.get("comp_z"))))]) + "\n")


def genus_taxid(tax, path):
    m = re.search(r'/db_xref="taxon:(\d+)"', open(path, errors="replace").read(200000))
    t = tax.at_rank(int(m.group(1)), "genus") if m else None
    if not t:
        sys.exit("no genus for the organism of " + path)
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["proteins", "jobs", "scores"])
    ap.add_argument("--set", default="test", choices=list(SETS), help="test chromosomes, or the development chromosomes")
    ap.add_argument("--accessions", nargs="*", default=[], help="only these chromosomes of the set")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS), help="only these configurations")
    ap.add_argument("--dir", default=os.path.join(os.environ.get("SEEDDRIFT_DATA", "."), "evalsets", "islandpick"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--api", default="http://127.0.0.1:8600")
    a = ap.parse_args()
    from seeddrift.pipeline import clean_proteins, get_taxonomy
    _, gb_dir, jobs_file = SETS[a.set]
    pre = "" if a.set == "test" else a.set + "_"
    todo = genomes(a.out, a.set, a.accessions)
    jpath = os.path.join(a.out, jobs_file)
    jobs = json.load(open(jpath)) if os.path.exists(jpath) else {}
    if a.mode == "scores":
        from seeddrift import config
        for cfg in a.configs:
            os.makedirs(os.path.join(a.dir, a.set + "_pipeline", cfg), exist_ok=True)
            for g in todo:
                j = jobs.get("%s@%s" % (g["accession"], cfg), {})
                if j.get("status") != "done":
                    print("%s@%s: no finished job" % (g["accession"], cfg), flush=True)
                    continue
                write_scores(os.path.join(config.JOBS, j["job"], "genes.tsv"),
                             os.path.join(a.dir, a.set + "_pipeline", cfg, "scores_%s.tsv" % g["accession"]))
        return
    if a.mode == "proteins":
        from seeddrift.inputs import from_genbank
        os.makedirs(os.path.join(a.dir, pre + "proteins"), exist_ok=True)
        with open(os.path.join(a.dir, pre + "proteins", "all.faa"), "w") as fall:
            for g in todo:
                acc = g["accession"]
                prots, _ = clean_proteins(from_genbank(open(os.path.join(a.dir, gb_dir, acc + ".gb")).read()).proteins)
                with open(os.path.join(a.dir, pre + "proteins", acc + ".faa"), "w") as fo:
                    for k, v in prots.items():
                        fo.write(">%s\n%s\n" % (k, v))
                        fall.write(">%s__%s\n%s\n" % (acc, k, v))
                print("%s %d proteins" % (acc, len(prots)), flush=True)
        return
    from run_benchmark import post_job, wait
    tax = get_taxonomy()
    for cfg in a.configs:                                    # all default runs first: they also give the proteins' evidence for the feature model
        extra = CONFIGS[cfg]
        for g in todo:
            acc = g["accession"]
            key = "%s@%s" % (acc, cfg)
            if jobs.get(key, {}).get("status") == "done":
                continue
            gb = os.path.join(a.dir, gb_dir, acc + ".gb")
            self_t = genus_taxid(tax, gb)
            fields = dict(name="islands %s %s" % (acc, cfg), self_taxon=str(self_t), domain="prokaryote", structure="false", **extra)
            jid = post_job(a.api, {"file": gb}, fields)
            st = wait(a.api, jid)
            s = st.get("summary", {})
            jobs[key] = dict(job=jid, status=st["status"], error=st.get("error"), self_taxid=self_t, self_name=s.get("self_name"),
                             home_group=s.get("home_group"), n_genes=s.get("n_genes"), seconds=s.get("seconds"))
            json.dump(jobs, open(jpath, "w"), indent=1)
            print("%s %s %s %s s" % (key, st["status"], s.get("home_group"), s.get("seconds")), flush=True)


if __name__ == "__main__":
    main()
