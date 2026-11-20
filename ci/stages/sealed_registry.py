"""Stage sealed_registry (pipeline register, with `register.sealed`): the registry of everything that is sealed, for
tools that publish or move data (the cross-project data lake: nothing in this registry is ever uploaded). Metadata
only; it opens no sequence, label or score.

  chromosomes        the island-set chromosomes that are sealed: bench/hard/results/sealed.tsv (17 sealed_v1 and
                     NC_010167.1) and the benchmark-v2 sealing candidates (bench/hard/results/v2_sealed_candidates.tsv,
                     registered under bench/hard/V2_REGISTER.md), each with its accession with and without version, its
                     taxid (the store's island_set_taxa.json), organism and why it is sealed
  corpus_test_splits the sealed test part of every corpus split three ways (train_data.test_split: true;
                     ci/stages/train_data.py three_way: the held-out part by split(salt "split"), the test part from the
                     rest by split(salt "test", every 4)), from the passed runs of register.sealed.test_split_runs: the
                     corpus dataset id, the genomes with their assembly accessions (corpus manifest accession and
                     paired_accession) and taxids, the sha256 of the run's train_data index and of the train_data.py it
                     ran, and the runs of this machine that trained or drew panels on those genomes (opened_by)
Output data/register/sealed_registry.json; gate G3_register checks it when present. The file is installed for the
publishing tools by copying it, unchanged (sha256 as reported), to the CI root's host/sealed_registry.json, mode 600.
"""
from __future__ import annotations

import csv
import datetime
import json
import os

from ci.hashing import sha256_file
from ci.stages.common import Stage, main_args, write_json

SOURCES = ("bench/hard/results/sealed.tsv", "bench/hard/results/v2_sealed_candidates.tsv", "bench/hard/results/v2_reference_set.tsv",
           "bench/hard/V2_REGISTER.md")


def _tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def chromosomes(repo, store):
    """The sealed island-set chromosomes with aliases and taxids."""
    with open(os.path.join(store, "panel", "island_set_taxa.json")) as f:
        taxid = {k: int(v) for k, v in json.load(f).items()}
    ref = {r["accession"]: r for r in _tsv(repo("bench/hard/results/v2_reference_set.tsv"))}
    out = []
    for r in _tsv(repo("bench/hard/results/sealed.tsv")):
        a = r["accession"]
        out.append(dict(accession=a, why="sealed_secondary" if ref.get(a, {}).get("part") == "sealed_secondary" else "sealed_v1"))
    for r in _tsv(repo("bench/hard/results/v2_sealed_candidates.tsv")):
        out.append(dict(accession=r["accession"], why="v2_candidate"))
    rows = []
    for x in out:
        a = x["accession"]
        if a not in taxid or a not in ref:
            raise SystemExit("sealed_registry: %s has no taxid or no register row" % a)
        rows.append(dict(x, aliases=sorted({a, a.split(".")[0]}), taxid=taxid[a], organism=ref[a]["organism"], genus=ref[a]["genus"]))
    if len({r["accession"] for r in rows}) != len(rows):
        raise SystemExit("sealed_registry: a chromosome is listed twice")
    return sorted(rows, key=lambda r: r["accession"])


def test_splits(runs, ci_runs):
    """The sealed test parts of the runs named, and the runs under ci_runs whose training data share the corpus
    dataset and put those genomes in their train or held-out split (opened_by)."""
    from ci.stages.codon import stage_output
    out = []
    for run in runs:
        ip = os.path.join(run, "data", "train_data", "index.json")
        sha = stage_output(run, "train_data", "train_data_index", ip)
        with open(ip) as f:
            idx = json.load(f)
        with open(os.path.join(run, "data", "corpus", "manifest.json")) as f:
            man = json.load(f)
        acc = {e["id"]: e for e in man["entries"]}
        test = [g for g in idx["genomes"] if g["split"] == "test"]
        if not test:
            raise SystemExit("sealed_registry: %s has no test split" % run)
        code = os.path.join(os.path.dirname(os.path.dirname(run)), "wt", os.path.basename(run), "ci", "stages", "train_data.py")
        ids = {g["id"] for g in test}
        opened = []
        for other in sorted(os.listdir(ci_runs)):
            op = os.path.join(ci_runs, other, "data", "train_data", "index.json")
            if other == os.path.basename(run) or not os.path.exists(op):
                continue
            with open(os.path.join(ci_runs, other, "data", "corpus", "manifest.json")) as f:
                if json.load(f).get("dataset_id") != man.get("dataset_id"):
                    continue
            with open(op) as f:
                sp = {g["id"]: g["split"] for g in json.load(f)["genomes"] if g["id"] in ids}
            n = sum(1 for s in sp.values() if s in ("train", "heldout"))
            if n:
                opened.append(dict(run=other, genomes_in_train_or_heldout=n))
        out.append(dict(run=os.path.basename(run), dataset_id=man.get("dataset_id"), train_data_index_sha256=sha,
                        split_code_sha256=sha256_file(code, use_sidecar=False) if os.path.exists(code) else None,
                        settings=dict(test_split=True, split="three_way", heldout=dict(salt="split", every=5, every_cluster=2), test=dict(salt="test", every=4)),
                        genomes=[dict(id=g["id"], aliases=sorted({x for x in (acc.get(g["id"], {}).get("accession"), acc.get(g["id"], {}).get("paired_accession")) if x}),
                                      taxid=int(g["taxid"]), group=g["group"]) for g in sorted(test, key=lambda g: g["id"])],
                        opened_by=opened))
    return out


def main():
    run_dir = main_args()
    with Stage(run_dir, "sealed_registry", requires=("pre",)) as st:
        cfg = st.config
        rc = cfg["register"]
        sc = rc.get("sealed") or {}
        rows = chromosomes(st.repo, rc["store"])
        splits = test_splits(list(sc.get("test_split_runs") or []), os.path.dirname(run_dir))
        reg = dict(version=1, created=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), commit=cfg["git"]["sha"],
                   note=__doc__.strip().splitlines()[0], sources={p: sha256_file(st.repo(p), use_sidecar=False) for p in SOURCES},
                   taxid_cache_sha256=sha256_file(os.path.join(rc["store"], "panel", "island_set_taxa.json"), use_sidecar=False),
                   chromosomes=rows, corpus_test_splits=splits,
                   counts=dict(chromosomes=len(rows), sealed_v1=sum(1 for r in rows if r["why"] == "sealed_v1"),
                               sealed_secondary=sum(1 for r in rows if r["why"] == "sealed_secondary"),
                               v2_candidates=sum(1 for r in rows if r["why"] == "v2_candidate"),
                               corpus_test_genomes=sum(len(s["genomes"]) for s in splits)),
                   gaps=["the assembly accessions (GCA/GCF) of the island-set chromosomes are not resolved (no network here); match them by "
                         "accession and taxid", "a corpus split three ways on another machine is covered only when its run is listed"])
        p = st.path("register", "sealed_registry.json")
        write_json(p, reg)
        st.output("sealed_registry", p)
        st.extra.update(sealed_counts=reg["counts"])


if __name__ == "__main__":
    main()
