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
recall at the lowest false-positive rate the negatives allow. Every row carries the hash of the genes it ranks (ids
and proteins in order) and of the genes the method scored itself, with the count of the others its documented rule
covers (composition: genes under 30 codons get the median; released: proteins under 30 aa rank last); gate G4 checks
both against the scenario. The report ends with
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
from ci.stages.train import KMERS as KMER_INDEX, features, softmax
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


def gene_set(pairs):
    """Hash of a list of (gene id, protein) pairs, in order."""
    return sha256_json([list(p) for p in pairs])


def coverage(pairs, scored, documented, rule):
    """What a method scored itself: the scored genes' hash and count, and how many of the others the method's
    documented rule covers (gate G4 accepts unscored genes only under such a rule)."""
    scored = np.asarray(scored, bool)
    documented = np.asarray(documented, bool)
    return dict(genes_sha256=gene_set(pairs), n_genes=len(pairs), n_scored=int(scored.sum()),
                scored_sha256=gene_set([p for p, m in zip(pairs, scored) if m]),
                unscored_documented=int((~scored & documented).sum()), unscored_rule=rule)


def released_row(run_dir, eid, pairs, truth):
    with open(os.path.join(run_dir, "data", "released", "manifest.json")) as f:
        man = json.load(f)
    if man.get("status") == "skipped":
        return dict(method="released_v0.1.0", auroc=None, recall_at_lowest_fpr=None, status="skipped: %s" % man.get("reason", ""), genes_sha256=gene_set(pairs))
    with open(os.path.join(run_dir, "data", "released", eid, "scores.tsv")) as f:
        sc = {r["gene_id"]: float(r["score"]) for r in csv.DictReader(f, delimiter="\t")}
    unknown = sorted(set(sc) - {g for g, _ in pairs})
    if unknown:
        raise SystemExit("released scores for genes that are not evaluation genes: %s" % unknown[:5])
    floor = min(sc.values(), default=0.0) - 1.0                  # genes the screen does not score (under 30 aa) rank last
    x = np.array([sc.get(g, floor) for g, _ in pairs])
    return dict(method="released_v0.1.0", auroc=auroc(x[truth], x[~truth]), recall_at_lowest_fpr=recall_at_lowest_fpr(x[truth], x[~truth]), status="computed",
                provenance=dict(tag=man["provenance"]["tag"], commit=man["provenance"]["commit"]),
                **coverage(pairs, [g in sc for g, _ in pairs], [len(p) < 30 for _, p in pairs], "proteins under 30 aa are not scored by the released screen; they rank last"))


STRATA = ("same genus", "same family", "same order", "same phylum", "other phylum", "other kingdom", "mobile element")


def _codon_profile(cds):
    v = np.zeros(64)
    s = cds.upper()
    for i in range(0, len(s) - 2, 3):
        j = KMER_INDEX.get(s[i:i + 3])
        if j is not None:
            v[j] += 1
    t = v.sum()
    return v / t if t else v


def _recall_at(pos, neg, fpr):
    if not len(pos) or not len(neg):
        return None
    thr = np.quantile(neg, 1 - fpr, method="higher")
    return float(np.mean(pos > thr))


def dev_constructed(run_dir, path, ranks):
    """Development diagnosis on constructed insertions (held-out recipients and donors, genus hidden on both sides): the
    model and the comparators on the same examples, overall, by donor stratum and by recipient group. Not a result."""
    z = np.load(path)
    with open(os.path.join(run_dir, "data", "train_data", "index.json")) as f:
        meta = {m["id"]: m for m in json.load(f)["genomes"]}
    cache = {}

    def genome(gid):
        if gid not in cache:
            folder = os.path.join(run_dir, "data", "corpus", gid)
            with open(os.path.join(folder, "genes.tsv")) as f:
                genes = [r["gene_id"] for r in csv.DictReader(f, delimiter="\t")]
            cds = read_fasta(os.path.join(folder, "cds.fna"))
            seqs = [cds["%s__%s" % (gid, g)] for g in genes]
            prof = np.array([_codon_profile(x) for x in seqs])
            h = np.load(os.path.join(run_dir, "data", "hits", gid + ".npz"))
            cache[gid] = dict(seqs=seqs, prof=prof, mean=prof.mean(0), off=h["off"], prot=h["prot"], pid=h["pid"], bits=h["bits"])
        return cache[gid]
    rec, gene, don, dgene, strat, score = [x.decode() for x in z["recipient"]], z["gene"], [x.decode() for x in z["donor"]], z["donor_gene"], z["stratum"], z["score"]
    cols = {m: [] for m in ("model", "alien_index", "best_hit", "counts", "composition")}
    for r, i, d, j, sc in zip(rec, gene, don, dgene, score):
        R = genome(r)
        src, k = (genome(d), int(j)) if d else (R, int(i))
        hidden = {meta[r]["ranks"][4]} | ({meta[d]["ranks"][4]} if d else set())
        rows = [x for x in range(src["off"][k], src["off"][k + 1]) if ranks[src["prot"][x], 4] not in hidden]
        ph = meta[r]["ranks"][0]
        b_in = max([float(src["bits"][x]) for x in rows if ranks[src["prot"][x], 0] == ph] or [0.0])
        b_out = max([float(src["bits"][x]) for x in rows if ranks[src["prot"][x], 0] != ph] or [0.0])
        cols["model"].append(float(sc))
        cols["alien_index"].append(math.log(2) * (b_out - b_in))
        cols["best_hit"].append(-max([src["pid"][x] / 10 for x in rows] or [0.0]))
        cols["counts"].append(-len({int(ranks[src["prot"][x], 4]) or -int(src["prot"][x]) for x in rows}))
        cols["composition"].append(float(np.abs(src["prof"][k] - R["mean"]).sum()))
    y = np.array([1 if d else 0 for d in don])
    grp = np.array([meta[r]["group"] for r in rec])
    strat = np.asarray(strat)
    out = dict(note="constructed insertions into held-out genomes from held-out donors, genus hidden on both sides; development diagnosis, "
                    "not a result (charter section 8)", examples=int(len(y)), insertions=int(y.sum()), recipients=len(set(rec)), methods={})
    for m, v in cols.items():
        v = np.array(v)
        pos, neg = v[y == 1], v[y == 0]
        row = dict(auroc=auroc(pos, neg), recall_at_fpr_0_01=_recall_at(pos, neg, 0.01), recall_at_fpr_0_05=_recall_at(pos, neg, 0.05), by_stratum={}, by_recipient_group={})
        for k, name in enumerate(STRATA):
            sel = strat == k
            if sel.any():
                row["by_stratum"][name] = dict(n=int(sel.sum()), auroc=auroc(v[sel], neg))
        for g in sorted(set(grp)):
            sel = grp == g
            if (y[sel] == 1).any() and (y[sel] == 0).any():
                row["by_recipient_group"][g] = dict(n=int(sel.sum()), auroc=auroc(v[sel & (y == 1)], v[sel & (y == 0)]))
        out["methods"][m] = row
    return out


def main():
    run_dir = main_args()
    with Stage(run_dir, "eval", requires=("pre", "corpus", "train", "eval")) as st:
        cfg = st.config
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        lib = np.load(os.path.join(run_dir, "data", "reference", "library.npz"))
        ranks = lib["ranks"]
        kind = cfg["train"].get("model", "toy_source_id")
        if kind == "source_v0":                                # the source model scored the evaluation genomes itself
            es = np.load(os.path.join(run_dir, "data", "model", "eval_scores.npz"))
            src_scores = {}
            for g, i, v in zip(es["genome"], es["gene"], es["score"]):
                src_scores.setdefault(g.decode(), {})[int(i)] = float(v)
        else:
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
            comp = composition.genome_scores(seqs)["codon"]            # genes under 30 codons get the genome's median
            comp = np.where(np.isfinite(comp), comp, np.nanmedian(comp))
            prots = read_fasta(os.path.join(folder, "proteins.faa"))
            pairs = [(g["gene_id"], prots["%s__%s" % (e["id"], g["gene_id"])]) for g in genes]
            short_cds = np.array([len(s) // 3 < 30 for s in seqs])
            if kind == "source_v0":
                ms = src_scores.get(e["id"], {})
                model_score = np.array([ms.get(i, np.nan) for i in range(len(genes))])
            else:
                X = np.array([features(s) for s in seqs])
                P = softmax(X @ W)
                host = classes.index(e["host"]) if e["host"] in classes else None
                model_score = 1 - P[:, host] if host is not None else np.full(len(seqs), np.nan)
            rows = []
            every = np.ones(len(genes), bool)
            for name, sc, scored, rule in (("model", model_score, np.isfinite(model_score), ""),
                                           ("alien_index", np.array(ai), every, "a gene without hits scores 0"),
                                           ("best_hit", np.array(best), every, "a gene without hits scores 0 (most foreign)"),
                                           ("counts", np.array(counts, float), every, "a gene without hits scores 0"),
                                           ("composition", comp, ~short_cds, "genes under 30 codons get the genome's median")):
                pos, neg = sc[truth], sc[~truth]
                ok = np.isfinite(sc).all()
                rows.append(dict(method=name, auroc=auroc(pos, neg) if ok else None, recall_at_lowest_fpr=recall_at_lowest_fpr(pos, neg) if ok else None,
                                 status="computed" if ok else "not_applicable (no score for every gene)",
                                 **coverage(pairs, scored, short_cds if name == "composition" else np.zeros(len(genes), bool), rule)))
            rows.append(released_row(run_dir, e["id"], pairs, truth))
            gs = gene_set(pairs)
            donors = sorted({int(t["donor_taxid"]) for t in mine if t.get("donor_taxid")})
            covers = "into %s (%s); donors %s" % (tax.name(e["host_taxid"]), tax.name(host_phylum) if host_phylum else "?",
                                                 ", ".join("%s (%s)" % (tax.name(d), tax.name(tax.at_rank(d, "phylum")) if tax.at_rank(d, "phylum") else "?") for d in donors))
            scenarios.append(dict(name=e["id"], kind="constructed (fixture)", covers=covers, n_pos=int(truth.sum()), n_neg=int((~truth).sum()), genes_sha256=gs, rows=rows))
        g5 = find_waiver("G5", "*")
        note = "pipeline check, not a scientific model" if kind == "toy_source_id" else "pilot (development diagnosis only; charter section 8, 2026-11-17)"
        report = dict(scenarios=scenarios, catalogue_judge=dict(status="absent", note="no public-event catalogue yet: no final judge (waiver G5 until %s)" % (g5["expires"] if g5 else "none")),
                      model=dict(kind=kind, note=note))
        p = st.path("eval", "eval_report.json")
        write_json(p, report)
        st.output("eval_report", p)
        dp = os.path.join(run_dir, "data", "model", "dev_eval.npz")
        if os.path.exists(dp):
            q = st.path("eval", "dev_constructed.json")
            write_json(q, dev_constructed(run_dir, dp, ranks))
            st.output("dev_constructed", q)


if __name__ == "__main__":
    main()
