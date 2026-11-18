"""Stage eval: score the evaluation genomes with the model and, on the same genes, with the comparators.

Comparators (always reported next to the model; never a bar, never an input of the model):
  alien_index   ln2 x (best bit score outside the host's phylum - best inside), hits of the host's own species ignored
  best_hit      minus the identity of the closest hit (no hit = most foreign)
  counts        minus the number of reference genera with a hit
  composition   synonymous codon usage against the genome's own (bench/islands/composition.py)
  (development set only: gc_difference, |GC of the gene - GC of the genome|; alien_index_fair, inside = the recipient's
  highest defined rank among phylum, class, order, family; two logistic regressions on summary features, all and
  composition only, and a one-hidden-layer network on all of them, trained on the same construction in training
  genomes. Per method also: average precision at 2% and 5% insertions, AUROC within bins of the GC difference and
  within quintiles of the composition distance; and a balance table of quantities that should not separate the classes)
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


def auroc_fast(pos, neg):
    """AUROC with ties at their mean rank, vectorised (the bootstrap calls it many times)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    u, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    r = avg[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def bootstrap_ci(v, y, rec, reps=500, seed=0):
    """95% interval of the AUROC over recipient genomes resampled with replacement (examples stay with their genome)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(rec)
    if not len(ids):
        return None
    rows = {r: np.flatnonzero(rec == r) for r in ids}
    out = []
    for _ in range(reps):
        take = np.concatenate([rows[r] for r in rng.choice(ids, size=len(ids), replace=True)])
        a = auroc_fast(v[take][y[take] == 1], v[take][y[take] == 0])
        if a is not None:
            out.append(a)
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))] if out else None


FEATURES = ("composition_l1", "gc_diff", "abs_gc_diff", "log_codons", "log1p_rows", "no_hit", "best_identity", "signed_log_ai", "log1p_genera",
            "frac_rows_same_phylum", "frac_rows_same_class", "frac_rows_same_order")


def logistic(X, y, l2=1e-2, iters=50):
    """L2-regularised logistic regression by Newton steps on standardised features: (mean, std, weights, bias)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], l2 * len(X))
    reg[-1] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + reg * w
        H = (Z * (p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return mu, sd, w


def ap_at_prevalence(pos, neg, pi):
    """Average precision with the natives weighted so that insertions make up the fraction `pi` of all genes (the
    constructed set is half insertions; real genomes have a few percent). Tied scores share one precision."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    w = len(pos) / len(neg) * (1 - pi) / pi
    v = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    u, inv = np.unique(-v, return_inverse=True)                   # descending score groups
    tp = np.bincount(inv, weights=lab, minlength=len(u))
    fp = np.bincount(inv, weights=1 - lab, minlength=len(u))
    ctp, cfp = np.cumsum(tp), np.cumsum(fp) * w
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    return float((tp * prec).sum() / len(pos))


def mlp(X, y, hidden=32, epochs=400, lr=1e-2, l2=1e-4, seed=0):
    """A one-hidden-layer network on standardised features (full batch, Adam, fixed epochs, fixed seed): a predict
    function. The nonlinear counterpart of logistic() for the summary-feature baseline."""
    rng = np.random.default_rng(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    P = [rng.normal(0, 1 / np.sqrt(Z.shape[1]), (Z.shape[1], hidden)), np.zeros(hidden), rng.normal(0, 1 / np.sqrt(hidden), hidden), np.zeros(1)]
    m = [np.zeros_like(p) for p in P]
    v = [np.zeros_like(p) for p in P]
    for t in range(1, epochs + 1):
        H = np.tanh(Z @ P[0] + P[1])
        p = 1 / (1 + np.exp(-np.clip(H @ P[2] + P[3], -30, 30)))
        e = (p - y) / len(y)
        gH = np.outer(e, P[2]) * (1 - H ** 2)
        grads = [Z.T @ gH + l2 * P[0], gH.sum(0), H.T @ e + l2 * P[2], np.array([e.sum()])]
        for k in range(4):
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
            P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
    return lambda Xn: np.tanh(((Xn - mu) / sd) @ P[0] + P[1]) @ P[2] + P[3][0]


GC_BINS = (0.0, 0.02, 0.05, 0.10, 1.01)                          # |GC of the gene - GC of the genome|


def within_bins(v, y, b):
    """AUROC of insertions against natives of the same bin, averaged over bins weighted by their insertions: the
    separation that is left once the binned quantity is held fixed."""
    tot, acc, rows = 0, 0.0, []
    for k in np.unique(b):
        sp, sn = (b == k) & (y == 1), (b == k) & (y == 0)
        a = auroc_fast(v[sp], v[sn]) if sp.any() and sn.any() else None
        rows.append(dict(bin=int(k), n_pos=int(sp.sum()), n_neg=int(sn.sum()), auroc=a))
        if a is not None:
            tot += int(sp.sum())
            acc += a * int(sp.sum())
    return (acc / tot if tot else None), rows


def model_hits_centre(run_dir, default=16):
    """How many visible hits the model read for the centre gene (the run's source model configuration)."""
    for name in ("source_sets.json", "source_config.json", "source_config-seed-1.json"):
        p = os.path.join(run_dir, "data", "train", name)
        if os.path.exists(p):
            with open(p) as f:
                return int(json.load(f).get("hits_centre", default))
    return default


def dev_constructed(run_dir, path, ranks, lib_taxid=None, k_hits=None):
    """Development diagnosis on constructed insertions (held-out recipients, held-out partners; the genome's own proteome
    and genus hidden for the recipient and for the partner, the donor of an insertion or the decoy of a native): the
    model (each seed and their mean) and the comparators on the same examples, with a logistic regression on summary
    features trained on the same construction in training genomes. Overall, by partner stratum (insertions against
    natives whose decoy is of the same stratum) and by recipient group; 95% intervals by resampling recipient genomes.
    Not a result (charter section 8)."""
    z = np.load(path)
    k_hits = int(k_hits or model_hits_centre(run_dir))
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
            gc = np.array([(x.upper().count("G") + x.upper().count("C")) / max(1, len(x)) for x in seqs])
            ln = np.array([len(x) for x in seqs], dtype=float)
            h = np.load(os.path.join(run_dir, "data", "hits", gid + ".npz"))
            so = h["sseq_off"] if "sseq_off" in h.files else None
            cache[gid] = dict(seqs=seqs, prof=prof, total=prof.sum(0), n=len(prof), gc=gc, len=ln, gc_sum=float((gc * ln).sum()), len_sum=float(ln.sum()),
                              off=h["off"], prot=h["prot"], pid=h["pid"], bits=h["bits"], sseq_off=so)
        return cache[gid]

    def hide(gid):
        """The model's rule (seeddrift/train/source_model.py hide_for at genus depth): the genome's own proteome, and its
        genus, or its species when it has no genus."""
        r = meta[gid]["ranks"]
        rules = [("taxid", int(meta[gid]["taxid"]))]
        if r[4]:
            rules.append((4, r[4]))
        elif r[5]:
            rules.append((5, r[5]))
        return rules

    def hidden_row(p, rules):
        return any((lib_taxid[p] == v) if c == "taxid" else (ranks[p, c] == v) for c, v in rules)

    def features(r, i, d, j, partner):
        """Comparator scores and baseline features of one example, from the rows the model reads."""
        R = genome(r)
        src, k = (genome(d), int(j)) if d else (R, int(i))
        rules = hide(r) + (hide(partner) if partner else [])
        # the rows the model reads: those that carry aligned residues (the nearest ones), not the extra best-per-genus rows
        cand = [x for x in range(src["off"][k], src["off"][k + 1]) if src["sseq_off"] is None or src["sseq_off"][x + 1] > src["sseq_off"][x]]
        # the first k_hits of them that are not hidden: exactly the hits the model reads for the centre gene
        rows = [x for x in cand if not hidden_row(src["prot"][x], rules)][:k_hits]
        rk = meta[r]["ranks"]
        b_in = max([float(src["bits"][x]) for x in rows if ranks[src["prot"][x], 0] == rk[0]] or [0.0])
        b_out = max([float(src["bits"][x]) for x in rows if ranks[src["prot"][x], 0] != rk[0]] or [0.0])
        ai = math.log(2) * (b_out - b_in)
        # the fair version: inside = the recipient's highest defined rank among phylum, class, order, family (a
        # recipient without a phylum, as many viruses, otherwise has every hit outside)
        cf = next((c for c in (0, 1, 2, 3) if rk[c]), None)
        if cf is None:
            ai_fair = 0.0
        else:
            f_in = max([float(src["bits"][x]) for x in rows if ranks[src["prot"][x], cf] == rk[cf]] or [0.0])
            f_out = max([float(src["bits"][x]) for x in rows if ranks[src["prot"][x], cf] != rk[cf]] or [0.0])
            ai_fair = math.log(2) * (f_out - f_in)
        best = max([src["pid"][x] / 1000.0 for x in rows] or [0.0])
        genera = len({int(ranks[src["prot"][x], 4]) or -int(src["prot"][x]) for x in rows})
        # the genome without gene i, for both classes (a native is scored against the rest; an insertion replaces gene i)
        bg = (R["total"] - R["prof"][int(i)]) / max(1, R["n"] - 1)
        comp = float(np.abs(src["prof"][k] - bg).sum())
        rest = R["len_sum"] - R["len"][int(i)]
        gcd = float(src["gc"][k] - ((R["gc_sum"] - R["gc"][int(i)] * R["len"][int(i)]) / rest if rest > 0 else R["gc"][int(i)]))
        frac = [float(np.mean([ranks[src["prot"][x], c] == rk[c] for x in rows])) if rows and rk[c] else 0.0 for c in (0, 1, 2)]
        feat = [comp, gcd, abs(gcd), math.log1p(src["len"][k] / 3), math.log1p(len(rows)), float(not rows), best,
                math.copysign(math.log1p(abs(ai)), ai), math.log1p(genera)] + frac
        bal = dict(codons=src["len"][k] / 3, rows_before_hiding=len(cand), rows_visible=len(rows), rows_hidden=len(cand) - len(rows),
                   hidden_fraction=(len(cand) - len(rows)) / len(cand) if cand else 0.0)
        return dict(alien_index=ai, alien_index_fair=ai_fair, best_hit=-best, counts=-genera, composition=comp, gc_difference=abs(gcd)), feat, bal

    def examples(zz):
        return ([x.decode() for x in zz["recipient"]], zz["gene"], [x.decode() for x in zz["donor"]], zz["donor_gene"], np.asarray(zz["stratum"]),
                [x.decode() for x in zz["partner"]] if "partner" in zz.files else [""] * len(zz["gene"]))
    rec, gene, don, dgene, strat, part = examples(z)
    cols = {m: [] for m in ("alien_index", "alien_index_fair", "best_hit", "counts", "composition", "gc_difference")}
    X, B = [], []
    for r, i, d, j, pa in zip(rec, gene, don, dgene, part):
        c, f, bq = features(r, i, d, j, pa)
        for m in cols:
            cols[m].append(c[m])
        X.append(f)
        B.append(bq)
    y = np.array([1 if d else 0 for d in don])
    methods = {}
    # the model: the run's (first) seed, each seed, and the mean of the seeds' scores on the same examples
    mj = os.path.join(run_dir, "data", "model", "model.json")
    seed_dirs = []
    if os.path.exists(mj):
        with open(mj) as f:
            md = json.load(f)
        seed_dirs = list(zip(md.get("seeds") or [], md.get("seed_dirs") or []))
    methods["model"] = np.asarray(z["score"], float)
    if len(seed_dirs) > 1:
        per = []
        for sd, dd in seed_dirs:
            zs = np.load(os.path.join(run_dir, dd, "dev_eval.npz"))
            if [x.decode() for x in zs["recipient"]] != rec or not np.array_equal(zs["gene"], gene):
                raise SystemExit("seed %s scored other development examples" % sd)
            methods["model_seed_%d" % sd] = np.asarray(zs["score"], float)
            per.append(methods["model_seed_%d" % sd])
        methods["model_mean_of_seeds"] = np.mean(per, 0)
    # the simple trained baseline: logistic regression on the summary features, fitted on training-genome constructions
    bt = os.path.join(run_dir, "data", "model", "baseline_train.npz")
    baseline = None
    if os.path.exists(bt):
        zb = np.load(bt)
        br, bg_, bd, bj, _, bp = examples(zb)
        XB = np.array([features(r, i, d, j, pa)[1] for r, i, d, j, pa in zip(br, bg_, bd, bj, bp)], float).reshape(len(bd), len(FEATURES))
        yb = np.array([1.0 if d else 0.0 for d in bd])
        if len(X) and 0 < yb.sum() < len(yb):                    # both classes to learn from, and examples to score
            XD = np.array(X, float)
            baseline = dict(features=list(FEATURES), train_examples=int(len(yb)), train_insertions=int(yb.sum()), train_recipients=len(set(br)))
            # all features, and the composition features only (no hit feature): what the hits add to composition
            for name, cols_ in (("baseline_logistic", list(range(len(FEATURES)))), ("baseline_logistic_composition_only", [0, 1, 2, 3])):
                mu, sd_, w = logistic(XB[:, cols_], yb)
                methods[name] = np.concatenate([(XD[:, cols_] - mu) / sd_, np.ones((len(XD), 1))], 1) @ w
                baseline["weights" if name == "baseline_logistic" else "weights_composition_only"] = dict(
                    zip([FEATURES[c] for c in cols_] + ["bias"], [round(float(x), 4) for x in w]))
            methods["baseline_mlp"] = mlp(XB, yb)(XD)                 # the same features, one hidden layer of 32
    for m, v in cols.items():
        methods[m] = np.array(v, float)
    grp = np.array([meta[r]["group"] for r in rec])
    reca = np.array(rec)
    ph0 = sorted({r for r in rec if not meta[r]["ranks"][0]})
    out = dict(note="constructed insertions into held-out genomes from held-out donors; natives carry a decoy partner drawn like a donor; the "
                    "genome's own proteome and genus hidden for the recipient and the partner; model and comparators read the same nearest hits; "
                    "development diagnosis, not a result (charter section 8)",
               examples=int(len(y)), insertions=int(y.sum()), recipients=len(set(rec)),
               recipients_by_group={g: len({r for r in rec if meta[r]["group"] == g}) for g in sorted(set(grp))},
               near_relative_recipients=sorted({r for r in rec if meta[r].get("cluster")}),
               alien_index_phylum_undefined=dict(recipients=ph0, examples=int(sum(1 for r in rec if r in set(ph0))),
                                                 note="no phylum for the recipient: every visible hit counts as outside, the index is not meaningful"),
               baseline=baseline, gc_bins=list(GC_BINS), hits_read=k_hits, methods={})
    # how far each quantity alone separates insertions from natives (AUROC, larger = higher for insertions); those the
    # construction sets (rows hidden, their fraction) should sit near 0.5, a gene's length and visible rows may carry
    # real signal or a construction artifact (donor genes drawn at random from another genome)
    out["balance"] = {}
    for q in ("codons", "rows_before_hiding", "rows_visible", "rows_hidden", "hidden_fraction"):
        qv = np.array([b[q] for b in B], float)
        out["balance"][q] = dict(auroc=auroc_fast(qv[y == 1], qv[y == 0]), mean_insertions=float(qv[y == 1].mean()) if y.any() else None,
                                 mean_natives=float(qv[y == 0].mean()) if (y == 0).any() else None)
    gcb = np.digitize(np.array(cols["gc_difference"], float), GC_BINS[1:-1]) if len(y) else np.zeros(0, int)
    comp_v = np.array(cols["composition"], float)
    compb = np.digitize(comp_v, np.quantile(comp_v, [0.2, 0.4, 0.6, 0.8])) if len(y) else np.zeros(0, int)
    for m, v in methods.items():
        pos, neg = v[y == 1], v[y == 0]
        row = dict(auroc=auroc_fast(pos, neg), auroc_ci95=bootstrap_ci(v, y, reca), recall_at_fpr_0_01=_recall_at(pos, neg, 0.01),
                   recall_at_fpr_0_05=_recall_at(pos, neg, 0.05), by_stratum={}, by_recipient_group={})
        for k, name in enumerate(STRATA):
            sp, sn = (strat == k) & (y == 1), (strat == k) & (y == 0)
            if sp.any() and sn.any():                       # insertions against natives whose decoy has the same stratum
                row["by_stratum"][name] = dict(n_pos=int(sp.sum()), n_neg=int(sn.sum()), recipients=len(set(reca[sp | sn])), auroc=auroc_fast(v[sp], v[sn]))
        per = []
        for g in sorted(set(grp)):
            sel = grp == g
            if (y[sel] == 1).any() and (y[sel] == 0).any():
                a = auroc_fast(v[sel & (y == 1)], v[sel & (y == 0)])
                per.append(a)
                row["by_recipient_group"][g] = dict(n=int(sel.sum()), recipients=len(set(reca[sel])), auroc=a,
                                                    recall_at_fpr_0_05=_recall_at(v[sel & (y == 1)], v[sel & (y == 0)], 0.05))
        row["macro_auroc_over_groups"] = float(np.mean(per)) if per else None
        row["average_precision_at_prevalence"] = {"0.02": ap_at_prevalence(pos, neg, 0.02), "0.05": ap_at_prevalence(pos, neg, 0.05)}
        row["auroc_within_gc_bins"], row["by_gc_bin"] = within_bins(v, y, gcb)
        row["auroc_within_composition_quintiles"], _ = within_bins(v, y, compb)
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
            write_json(q, dev_constructed(run_dir, dp, ranks, lib["taxid"]))
            st.output("dev_constructed", q)


if __name__ == "__main__":
    main()
