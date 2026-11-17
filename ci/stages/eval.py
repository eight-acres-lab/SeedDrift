"""Stage eval: score the evaluation genomes with the model and, on the same genes, with the comparators.

Comparators (always reported next to the model; never a bar, never an input of the model):
  alien_index   ln2 x (best bit score outside the host's phylum - best inside), hits of the host's own species ignored
  best_hit      minus the identity of the closest hit (no hit = most foreign)
  counts        minus the number of reference genera with a hit
  composition   synonymous codon usage against the genome's own (bench/islands/composition.py)
  released_v0.1.0  the released screen, recomputed in the same run from its tag (stage released, gate G4); it sees
                the reference as shipped (the donor's genus is not hidden from it)
Hits to the host's own species and to the donor's genus are hidden for every gene (the donor lineage is taken as
unsampled, as the benchmark hides the own genus). Per scenario: positives = genes of the truth table, negatives = all
other genes; AUROC with ties at expectation and
recall at the lowest false-positive rate the negatives allow. Every row carries the hash of the gene set it was scored
on (gene ids and proteins in order; gate G4 requires the same set in every row of a scenario). The report ends with
the public-event catalogue section; until the catalogue exists it says so (waiver G5).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

from ci import ROOT
from ci.gates import find_waiver
from ci.hashing import sha256_json
from ci.stages.common import Stage, main_args, read_fasta, write_json
from ci.stages.train import features, softmax
from ci.taxonomy import Taxonomy

sys.path.insert(0, os.path.join(ROOT, "bench", "islands"))


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg) or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    s = allv[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_lowest_fpr(pos, neg):
    if not len(pos) or not len(neg) or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        return None
    return float(np.mean(np.asarray(pos) > np.max(neg)))


def released_row(run_dir, eid, gene_ids, truth):
    with open(os.path.join(run_dir, "data", "released", "manifest.json")) as f:
        man = json.load(f)
    if man.get("status") == "skipped":
        return dict(method="released_v0.1.0", auroc=None, recall_at_lowest_fpr=None, status="skipped: %s" % man.get("reason", ""))
    with open(os.path.join(run_dir, "data", "released", eid, "scores.tsv")) as f:
        sc = {r["gene_id"]: float(r["score"]) for r in csv.DictReader(f, delimiter="\t")}
    floor = min(sc.values(), default=0.0) - 1.0                  # genes the screen does not score (under 30 aa) rank last
    x = np.array([sc.get(g, floor) for g in gene_ids])
    return dict(method="released_v0.1.0", auroc=auroc(x[truth], x[~truth]), recall_at_lowest_fpr=recall_at_lowest_fpr(x[truth], x[~truth]), status="computed",
                unscored=sum(1 for g in gene_ids if g not in sc), provenance=dict(tag=man["provenance"]["tag"], commit=man["provenance"]["commit"]))


def main():
    run_dir = main_args()
    with Stage(run_dir, "eval", requires=("pre", "corpus", "train", "eval")) as st:
        cfg = st.config
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        lib = np.load(os.path.join(run_dir, "data", "reference", "library.npz"))
        ranks = lib["ranks"]
        model = np.load(os.path.join(run_dir, "data", "model", "model.npz"))
        W, classes = model["W"], [str(c) for c in model["classes"]]
        with open(os.path.join(run_dir, "data", "eval_inputs", "manifest.json")) as f:
            entries = json.load(f)["entries"]
        import composition                                     # bench/islands/composition.py
        with open(st.repo(cfg["scenario"]["truth"])) as f:
            truth_rows = list(csv.DictReader(f, delimiter="\t"))
        scenarios = []
        for e in entries:
            folder = os.path.join(run_dir, "data", "eval_inputs", e["id"])
            with open(os.path.join(folder, "genes.tsv")) as f:
                genes = list(csv.DictReader(f, delimiter="\t"))
            cds = read_fasta(os.path.join(folder, "cds.fna"))
            seqs = [cds["%s__%s" % (e["id"], g["gene_id"])] for g in genes]
            truth = np.array([g["truth"] != "native" for g in genes])
            hits = np.load(os.path.join(run_dir, "data", "hits", e["id"] + ".npz"))
            off, prot, pid, bits = hits["off"], hits["prot"], hits["pid"], hits["bits"]
            host_phylum = tax.at_rank(e["host_taxid"], "phylum")
            host_species = tax.at_rank(e["host_taxid"], "species")
            mine = [t for t in truth_rows if t["scenario"] == e["id"]]
            hidden_genera = {tax.at_rank(int(t["donor_taxid"]), "genus") for t in mine if t.get("donor_taxid")} - {0}
            ai, best, counts = [], [], []
            for i in range(len(genes)):
                rows = range(off[i], off[i + 1])
                rows = [r for r in rows if ranks[prot[r], 5] != host_species and ranks[prot[r], 4] not in hidden_genera]
                b_in = max([float(bits[r]) for r in rows if ranks[prot[r], 0] == host_phylum] or [0.0])
                b_out = max([float(bits[r]) for r in rows if ranks[prot[r], 0] != host_phylum] or [0.0])
                ai.append(math.log(2) * (b_out - b_in))
                best.append(-max([pid[r] / 10 for r in rows] or [0.0]))
                counts.append(-len({int(ranks[prot[r], 4]) or -int(prot[r]) for r in rows}))
            comp = composition.genome_scores(seqs)["codon"]
            comp = np.where(np.isfinite(comp), comp, np.nanmedian(comp))
            X = np.array([features(s) for s in seqs])
            P = softmax(X @ W)
            host = classes.index(e["host"]) if e["host"] in classes else None
            model_score = 1 - P[:, host] if host is not None else np.full(len(seqs), np.nan)
            rows = []
            for name, sc in (("model", model_score), ("alien_index", np.array(ai)), ("best_hit", np.array(best)), ("counts", np.array(counts, float)),
                             ("composition", comp)):
                pos, neg = sc[truth], sc[~truth]
                ok = np.isfinite(sc).all()
                rows.append(dict(method=name, auroc=auroc(pos, neg) if ok else None, recall_at_lowest_fpr=recall_at_lowest_fpr(pos, neg) if ok else None,
                                 status="computed" if ok else "not_applicable (no score for every gene)"))
            rows.append(released_row(run_dir, e["id"], [g["gene_id"] for g in genes], truth))
            prots = read_fasta(os.path.join(folder, "proteins.faa"))
            gs = sha256_json([[g["gene_id"], prots["%s__%s" % (e["id"], g["gene_id"])]] for g in genes])
            for r in rows:
                r["genes_sha256"] = gs
            donors = sorted({int(t["donor_taxid"]) for t in mine if t.get("donor_taxid")})
            covers = "into %s (%s); donors %s" % (tax.name(e["host_taxid"]), tax.name(host_phylum) if host_phylum else "?",
                                                 ", ".join("%s (%s)" % (tax.name(d), tax.name(tax.at_rank(d, "phylum")) if tax.at_rank(d, "phylum") else "?") for d in donors))
            scenarios.append(dict(name=e["id"], kind="constructed (fixture)", covers=covers, n_pos=int(truth.sum()), n_neg=int((~truth).sum()), genes_sha256=gs, rows=rows))
        g5 = find_waiver("G5", "*")
        report = dict(scenarios=scenarios, catalogue_judge=dict(status="absent", note="no public-event catalogue yet: no final judge (waiver G5 until %s)" % (g5["expires"] if g5 else "none")),
                      model=dict(kind="toy_source_id", note="pipeline check, not a scientific model"))
        p = st.path("eval", "eval_report.json")
        write_json(p, report)
        st.output("eval_report", p)


if __name__ == "__main__":
    main()
