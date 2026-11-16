#!/usr/bin/env python
"""Fixed statistics on the saved trials: what scores that learn nothing reach, and which trials no visible row can decide.

Every saved trial of fit.py is rebuilt (same seeds, checked against the saved table) and the query is scored by:

  top_all       how far from the genome the query's five best rows are, over every visible panel row
                (an Alien Index with the genome's own measured distances instead of a taxonomy)
  own_all       misfit of the query to the genome's profile over every visible panel row. The profile comes from
                about 100 genes of the genome; the model is "similarity = rate of the gene x distance of the genome".
  own_shown     the same over the 32 rows the model is shown
  ctx_shown     misfit to a profile estimated from the 12 shown genes only, over the 32 shown rows: a fixed
                statistic on exactly the model's input
  oracle_all    misfit to the genome's profile minus misfit to the donor's profile, every visible panel row.
                It is told who the donor is, so it is not a method. It is this one misfit statistic with the
                right alternative supplied, not an upper bound: it ignores which cells are missing, and the
                trained models beat it on most trials.
  oracle_shown  the same over the 32 shown rows

It also records, per trial, whether any visible row could tell the donor from the genome at all (on the panel
rows of the trial, and again with every reference proteome as a possible row, the *_all pair):

  donor_side    the donor's distance to the genome's own proteome minus its distance to its nearest visible row.
                Positive: some visible row is closer to the donor than the genome is, so a gene of the donor can
                match that row better than the genome's genes do.
  genome_side   the genome's distance to the donor's proteome minus its distance to its nearest visible row.
                Positive: some visible row is closer to the genome than the donor is.

When both are at most 0.05 the genome and the donor are each other's nearest neighbours among what is
visible: every panel row is about equally far from both, and none of the scores here separates the two. A side
is undefined when the two genomes have no measurable distance to each other (they are far apart); such trials
count as separable.

usage:
  python research/array/ceiling.py --data DIR --names a b --profiles DIR/profiles.npz --library LIB \
      --which unseen --runs msa_pairs=DIR/run2/msa_pairs num_anchor=DIR/run2/num_anchor --out DIR/ceiling_unseen.md
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as M  # noqa: E402
from fit import splits  # noqa: E402
from report import auroc  # noqa: E402

CAP = 2.5            # distance used where a genome has no measurable distance to a proteome
STATS = ("top_all", "own_all", "own_shown", "ctx_shown", "oracle_all", "oracle_shown", "donor_side", "genome_side",
         "donor_side_all", "genome_side_all")
SCORES = STATS[:6]
CHECK = ("k_g", "k_t", "n_c", "n_present", "rule_ai")      # recomputed with the trial and compared with the saved table
REC, ST, DS = {}, None, None


def delta(score_per_len):
    """The transform of build.py (delta_of) applied to an alignment score per gene position."""
    return -np.log(np.clip(0.1925 * score_per_len, 0.005, 1.0)) + 0.02


def misfit(dx, D):
    """Mean absolute deviation of ln(dx / D) from its median: how badly one rate times the profile explains the query."""
    if len(dx) < 3:
        return np.nan
    lr = np.log(dx) - np.log(D)
    return float(np.abs(lr - np.median(lr)).mean())


def capped(D):
    return np.where(np.isfinite(D), D, CAP)


def rank_one(X):
    r = np.ones(len(X))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(4):
            D = np.nanmedian(X / r[:, None], axis=0)
            r = np.nanmedian(X / D[None, :], axis=1)
            r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
            r = r / np.median(r)
    return D


def sides_all(st, g, t, k_g, k_t):
    """donor_side and genome_side when every reference proteome can be a row and hiding is by radius (wide.py hide)."""
    h = np.zeros(st.n_prot, dtype=bool)
    for genome, k in ((g, k_g), (t, k_t)):
        if st.own[genome] >= 0:
            h[st.own[genome]] = True
        order = st.near(genome)
        if k and len(order):
            with np.errstate(invalid="ignore"):
                h |= st.D[genome] <= st.D[genome][order[min(k, len(order)) - 1]]
    out = []
    for p, q in ((t, g), (g, t)):
        D = st.D[p]
        vis = np.isfinite(D) & ~h
        ok = vis.any() and st.own[q] >= 0 and np.isfinite(D[st.own[q]])
        out.append(float(D[st.own[q]] - D[vis].min()) if ok else np.nan)
    return out


def _example(st, cfg, rng, g, x, ctx, hidden, truth):
    REC.update(x=int(x), ctx=np.array(ctx), hidden=hidden.copy())
    return _example.orig(st, cfg, rng, g, x, ctx, hidden, truth)


def _choose(st, cfg, rng, g, x, hidden):
    r = _choose.orig(st, cfg, rng, g, x, hidden)
    REC.update(rows=r[0])
    return r


def members(st, gene, rows):
    """delta of the gene to each of the rows (NaN where it has no member there)."""
    rp, sc = st.rows(gene), st.scores(gene)
    out = np.full(len(rows), np.nan)
    if len(rp):
        k = np.searchsorted(rp, rows)
        k[k == len(rp)] = 0
        hit = rp[k] == rows
        out[hit] = delta(sc[k[hit]])
    return out


def work(span):
    st, out = ST, []
    for i in range(*span):
        ex = DS[i]
        g, t, x, hidden, rows = int(ex["g"]), int(ex["donor"]), REC["x"], REC["hidden"], REC["rows"]
        rp, sc = st.rows(x), st.scores(x)
        vis = ~hidden[rp]
        ra, dx = rp[vis], delta(sc[vis])
        Dg, Dt = capped(st.D[g][ra]), capped(st.D[t][ra])
        s = dict.fromkeys(STATS, np.nan)
        if len(ra):
            near = st.near(g)
            near = near[~hidden[near]][:5]
            best = np.argsort(dx, kind="stable")[:5]
            s["top_all"] = float(Dg[best].mean() - (st.D[g][near].mean() if len(near) else CAP))
        for key, p, q in (("donor_side", t, g), ("genome_side", g, t)):       # needs no query: only the two profiles
            order = st.near(p)
            order = order[~hidden[order]]
            if len(order) and st.own[q] >= 0 and np.isfinite(st.D[p][st.own[q]]):
                s[key] = float(st.D[p][st.own[q]] - st.D[p][order[0]])
        s["donor_side_all"], s["genome_side_all"] = sides_all(st, g, t, int(ex["k_g"]), int(ex["k_t"]))
        s["own_all"] = misfit(dx, Dg)
        s["oracle_all"] = s["own_all"] - misfit(dx, Dt)
        ds = members(st, x, rows)
        ok = np.isfinite(ds)
        s["own_shown"] = misfit(ds[ok], capped(st.D[g][rows[ok]]))
        s["oracle_shown"] = s["own_shown"] - misfit(ds[ok], capped(st.D[t][rows[ok]]))
        X = np.stack([members(st, int(c), rows) for c in REC["ctx"]])
        s["ctx_shown"] = misfit(ds[ok], capped(rank_one(X))[ok])
        out.append([int(ex["x"]), g, t, float(ex["label"])] + [s[k] for k in STATS] + [float(ex[k]) for k in CHECK])
    return np.array(out, dtype=np.float64)


def main():
    global ST, DS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--profiles", required=True); ap.add_argument("--library", required=True)
    ap.add_argument("--which", default="unseen", choices=("unseen", "validation"))
    ap.add_argument("--runs", nargs="+", required=True, help="name=run directory (the first one supplies the trial table)")
    ap.add_argument("--per-genome", type=int, default=54); ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import multiprocessing as mp
    st = ST = M.Store(a.data, a.names, a.profiles)
    cfg = M.Cfg()
    sp = splits(st, a.library)
    enough = {g for g, genes in st.genes_of.items() if len(genes) > cfg.M + 1}
    k, seed = (2, 202) if a.which == "unseen" else (1, 101)
    focal = sorted(g for g, s in sp.items() if s == k and g in enough)
    _example.orig, _choose.orig = M.example, M.choose_rows
    M.example, M.choose_rows = _example, _choose
    DS = M.Trials(st, cfg, focal, M.Donors(st, focal), a.per_genome, seed)
    runs = [x.split("=", 1) for x in a.runs]
    t = dict(np.load(os.path.join(runs[0][1], "trials_%s.npz" % a.which)))
    n = len(DS)
    if n != len(t["label"]):
        sys.exit("trial count differs from the saved table")
    spans = [(i, min(n, i + 200)) for i in range(0, n, 200)]
    with mp.get_context("fork").Pool(a.procs) as pool:
        R = np.concatenate(pool.map(work, spans, chunksize=1))
    for j, key in enumerate(("x", "g", "donor", "label")):
        if not np.array_equal(R[:, j], t[key].astype(np.float64)):
            sys.exit("rebuilt trials differ from the saved table (%s)" % key)
    for j, key in enumerate(CHECK):                            # the later draws too: hiding, other genes, rows
        if not np.allclose(R[:, 4 + len(STATS) + j], t[key].astype(np.float64), atol=1e-5):
            sys.exit("rebuilt trials differ from the saved table (%s)" % key)
    y = t["label"] > 0.5
    scores = {}
    for j, key in enumerate(SCORES):
        v = R[:, 4 + j]
        scores[key] = (np.where(np.isfinite(v), v, np.nanmedian(v)), float(np.isfinite(v).mean()))
    for name, d in runs:
        z = np.load(os.path.join(d, "trials_%s.npz" % a.which))
        if not (np.array_equal(z["x"], t["x"]) and np.array_equal(z["label"], t["label"])):
            sys.exit("%s scores other trials" % name)
        scores["model: " + name] = (z["logit"].astype(np.float64), 1.0)
    scores["rule_ai (shown rows)"] = (t["rule_ai"].astype(np.float64), 1.0)
    near = t["d_gt"] < 0.8
    cells = [("all", np.ones(n, bool)), ("near", near), ("mid", (t["d_gt"] >= 0.8) & (t["d_gt"] < 1.3)), ("far", t["d_gt"] >= 1.3),
             ("near, nothing hidden", near & (t["k_g"] == 0) & (t["k_t"] == 0)),
             ("near, genome's rows kept", near & (t["k_g"] == 0)),
             ("near, 16 hidden around the genome", near & (t["k_g"] == 16)),
             ("near, 16 hidden around the donor", near & (t["k_t"] == 16))]
    L = ["# Fixed statistics on the same trials (%s)" % a.which, "",
         "Generated by `research/array/ceiling.py`. AUROC, native against foreign queries; %d trials." % n, "",
         "| score | defined | " + " | ".join(c for c, _ in cells) + " |", "|---|---|" + "---|" * len(cells)]
    for key, (v, frac) in scores.items():
        L.append("| %s | %.0f%% | %s |" % (key, 100 * frac, " | ".join("%.3f" % auroc(v[m], y[m]) for _, m in cells)))
    L += ["", "- Trials per cell: " + ", ".join("%s %d" % (c, int(m.sum())) for c, m in cells) + ".",
          "- A score that is not defined for a trial (fewer than three rows) is set to the median, so such trials tie.",
          "- `oracle_*` is told who the donor is. It is one misfit statistic with the right alternative supplied: not a method and not an upper bound."]
    ds_, gs_ = R[:, 4 + STATS.index("donor_side")], R[:, 4 + STATS.index("genome_side")]
    strata = [("no visible row tells them apart (both sides <= 0.05)", (ds_ <= 0.05) & (gs_ <= 0.05)),
              ("a row closer to the genome only", (ds_ <= 0.05) & ~(gs_ <= 0.05)),
              ("a row closer to the donor, by 0.05 to 0.3", (ds_ > 0.05) & (ds_ <= 0.3)),
              ("a row closer to the donor, by 0.3 to 0.6", (ds_ > 0.3) & (ds_ <= 0.6)),
              ("a row closer to the donor, by more than 0.6", ds_ > 0.6),
              ("no measurable distance between the two (far apart)", ~np.isfinite(ds_))]
    keys = [k for k in scores if k.startswith("model") or k.startswith("rule") or k in ("top_all", "oracle_shown")]
    L += ["", "## By whether a visible row can tell the donor from the genome", "",
          "Split by the two profiles only (no query is looked at). This split was made after the first results were seen.", "",
          "| trials | n | share of near / mid / far | " + " | ".join(keys) + " |", "|---|---|---|" + "---|" * len(keys)]
    for name, m in strata:
        if m.sum() < 20:
            continue
        L.append("| %s | %d | %.0f%% / %.0f%% / %.0f%% | %s |" % (
            name, int(m.sum()), 100 * (m & cells[1][1]).sum() / max(1, cells[1][1].sum()), 100 * (m & cells[2][1]).sum() / max(1, cells[2][1].sum()),
            100 * (m & cells[3][1]).sum() / max(1, cells[3][1].sum()), " | ".join("%.3f" % auroc(scores[k][0][m], y[m]) for k in keys)))
    has = t["n_present"] >= 8
    L += ["", "The same, only trials where the query has members in at least 8 of the shown rows:", "",
          "| trials | n | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for name, m in strata:
        m = m & has
        if m.sum() < 20:
            continue
        L.append("| %s | %d | %s |" % (name, int(m.sum()), " | ".join("%.3f" % auroc(scores[k][0][m], y[m]) for k in keys)))
    da, ga = R[:, 4 + STATS.index("donor_side_all")], R[:, 4 + STATS.index("genome_side_all")]
    L += ["", "- With every reference proteome as a possible row and hiding by radius (the wide table), %.0f%% of the trials have no visible row that "
          "tells the donor from the genome (%.0f%% with panel rows only)." % (100 * ((da <= 0.05) & (ga <= 0.05)).mean(), 100 * ((ds_ <= 0.05) & (gs_ <= 0.05)).mean())]
    np.savez(os.path.splitext(a.out)[0] + ".npz", stats=R[:, :4 + len(STATS)], names=np.array(STATS))
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
