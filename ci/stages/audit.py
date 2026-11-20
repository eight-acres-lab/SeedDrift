"""Stage audit (pipeline audit): benchmark v2, milestone M0, slim (the user, 2026-11-20: "Slim, then new units"): the
bias audit of the old island set on its 24 development chromosomes, as a baseline before any new label family.
Inputs are a passed island run's outputs (checked by sha256, as pipeline codon checks them) and the store's records
and tables of the development chromosomes only.

  columns   label-free per gene (data/audit/columns/scores_<acc>.tsv): this line's comparators recomputed from the
            checked arrays (cmp_codon, cmp_gc3, cmp_hits, cmp_short, cmp_kmer), and the trivial scores the benchmark's
            own tables (e5_*) lack:
              tv_gc        GC of the coding sequence            tv_gc_dev   |GC - the chromosome's median|
              tv_cai       minus the codon adaptation index against the chromosome's ribosomal-protein genes
                           (by /product; NaN with fewer than 20 of them)
              tv_win_gc    the mean quantile of GC over the gene and 5 genes on each side (bench/hard/metrics.py)
              tv_win_dev   the same window over the mean quantile of tv_gc_dev, cmp_gc3, cmp_codon and cmp_kmer
              tv_mobile    1 when /product names a mobile-element function (bench/islands/evaluate.py PHAGE words)
              tv_random    a fixed pseudo-random number per gene (control)
            and metadata for the shortcut detector (meta_<acc>.tsv): md_pos (place on the replicon, 0..1),
            md_origin_dist (circular distance to position 1), md_strand, md_trna_dist (genes to the nearest tRNA),
            md_cat_* (/product category: hypothetical, mobile, ribosomal, transport, regulator, other)
  report    the shared protocol (ci/stages/protocol.py) on every column and comparator, with the margin-free cell
            for window scores (natives and edge genes as negatives: the gated natives keep 5 genes from any island
            gene, so a window of +-5 genes never meets an island from a native); two fused baselines over every
            trivial column (fz_l1: L1 logistic, fz_gbm: gradient boosting; leave one genus unit out, fixed
            hyperparameters); the shortcut detector (metadata only, leave one unit out, AUROC within chromosomes;
            core = place, strand, tRNA distance; annotation = + /product categories); the negative control (labels
            shuffled within each chromosome: every column must come back to 0.5); intervals of every score
            (protocol.paired_delta, cluster bootstrap)
With 3 genus units no verdict is possible: the report is the old set's baseline, not a judgement.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re

import numpy as np

from ci.hashing import sha256_file
from ci.stages.common import Stage, main_args, write_json

TRIVIAL = ("tv_gc", "tv_gc_dev", "tv_cai", "tv_win_gc", "tv_win_dev", "tv_mobile", "tv_random")
WINDOW = ("tv_win_gc", "tv_win_dev")
CATS = (("mobile", None), ("hypothetical", r"hypothetical|uncharacteri[sz]ed|unknown function|duf\d"),
        ("ribosomal", r"ribosom"), ("transport", r"transport|permease|abc |porin|efflux"),
        ("regulator", r"regulat|repressor|activator|sigma factor|two-component"))
RIBO = re.compile(r"\b(30s|50s)?\s*ribosomal protein\b", re.I)
RIBO_NOT = re.compile(r"methyltransferase|acetyltransferase|modification|synthase|kinase|binding|maturation|assembly|pseudouridine", re.I)
KEYS = ("islands_from", "fused", "detector", "controls", "boot", "block")


def _bench(name, rel, repo):
    """A bench module loaded under a unique name (bench/hard/evaluate.py and bench/islands/evaluate.py share theirs)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, repo(rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- record features ---------------------------------------------------------------------------------------------

def record_features(gb_path, names):
    """Per gene (in the order of the island run's genes.tsv, `names`): /product; per record: topology, replicon
    lengths, GC; the tRNA genes' places. The genes and their names follow ci/stages/islands.build_inputs
    (seeddrift.inputs.from_genbank, islands.clean_names), which must give exactly `names`."""
    from Bio import SeqIO
    from ci.stages.islands import clean_names
    from seeddrift.inputs import from_genbank
    with open(gb_path) as f:
        text = f.read()
    g = from_genbank(text)
    _, ren = clean_names(g.proteins)
    if list(ren.values()) != list(names):
        raise SystemExit("audit: the record's genes differ from the island run's genes.tsv (%s)" % os.path.basename(gb_path))
    from seeddrift.inputs import translate
    product, trna, closed, gc, lengths = {}, [], False, [], {}
    import io
    kept = 0                                  # from_genbank's rule: the ids, the skipped CDS and the fallback count
    for rec in SeqIO.parse(io.StringIO(text), "genbank"):
        lengths[rec.id] = len(rec.seq)
        closed = closed or (rec.annotations.get("topology") == "circular")
        s = str(rec.seq).upper()
        gc.append((s.count("G") + s.count("C"), sum(s.count(b) for b in "ACGT")))
        dna = str(rec.seq) if rec.seq is not None else ""
        for f in rec.features:
            if f.type == "tRNA":
                trna.append((rec.id, int(f.location.start) + 1))
            if f.type != "CDS" or "pseudo" in f.qualifiers:
                continue
            q = f.qualifiers
            gid = (q.get("locus_tag") or q.get("protein_id") or q.get("gene") or ["%s_%d" % (rec.id, kept + 1)])[0]
            try:
                nt = str(f.location.extract(rec.seq)) if dna else ""
            except Exception:                   # noqa: BLE001 - as from_genbank
                nt = ""
            if not (q.get("translation", [""])[0] or (translate(nt, int(q.get("transl_table", ["11"])[0])) if nt else "")):
                continue
            if gid not in product:
                kept += 1
            product[gid] = q.get("product", [""])[0]
    by_name = {new: product.get(gid, "") for gid, new in ren.items()}
    coords = {new: g.coords[gid] for gid, new in ren.items()}
    return dict(product=[by_name[n] for n in names], coords=[coords[n] for n in names], trna=trna, closed=closed, lengths=lengths,
                genome_gc=sum(a for a, _ in gc) / max(sum(b for _, b in gc), 1))


def cai(counts, ref_rows):
    """Codon adaptation index per gene from codon counts [genes, 64] (ACGT order) against the reference genes' usage:
    w_c = RSCU-normalised frequency (relative to the most used synonym), geometric mean over the gene's codons of
    amino acids with more than one codon (stop codons, Met and Trp left out)."""
    from seeddrift.train.codon_study import TCAG, amino_acids
    aa = amino_acids()
    names = sorted(set(TCAG))
    ref = counts[ref_rows].sum(0) + 0.5
    w = np.zeros(64)
    for a in range(21):
        idx = np.where(aa == a)[0]
        w[idx] = ref[idx] / ref[idx].max()
    keep = np.array([names[aa[c]] not in ("*", "M", "W") for c in range(64)])
    lw = np.log(w) * keep
    n = (counts * keep).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.exp((counts * lw).sum(1) / n), np.nan)


def gene_codon_counts(cds_seqs):
    """[genes, 64] codon counts (ACGT order, frame from the first base, incomplete or ambiguous codons left out)."""
    idx = {a: i for i, a in enumerate("ACGT")}
    out = np.zeros((len(cds_seqs), 64))
    for k, s in enumerate(cds_seqs):
        s = s.upper()
        for j in range(0, len(s) - 2, 3):
            a, b, c = s[j:j + 3]
            if a in idx and b in idx and c in idx:
                out[k, 16 * idx[a] + 4 * idx[b] + idx[c]] += 1
    return out


def columns_of(feat, cds_seqs, cmp_cols, bm, phage):
    """The label-free columns of one chromosome (see the module docstring)."""
    n = len(cds_seqs)
    gcn = np.array([(s.upper().count("G") + s.upper().count("C")) for s in cds_seqs], float)
    ln = np.array([sum(s.upper().count(b) for b in "ACGT") for s in cds_seqs], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        gc = np.where(ln > 0, gcn / ln, np.nan)
    cols = dict(tv_gc=gc, tv_gc_dev=np.abs(gc - np.nanmedian(gc)))
    counts = gene_codon_counts(cds_seqs)
    ribo = np.array([bool(RIBO.search(p)) and not RIBO_NOT.search(p) for p in feat["product"]])
    cols["tv_cai"] = -cai(counts, ribo) if ribo.sum() >= 20 else np.full(n, np.nan)
    contig = np.array([c[0] for c in feat["coords"]])
    cols["tv_win_gc"] = bm.window_scored(bm.quantile(gc), contig, 5, feat["closed"])
    devq = np.nanmean(np.stack([bm.quantile(v) for v in (cols["tv_gc_dev"], cmp_cols["cmp_gc3"], cmp_cols["cmp_codon"], cmp_cols["cmp_kmer"])]), 0)
    cols["tv_win_dev"] = bm.window_scored(devq, contig, 5, feat["closed"])
    low = [p.lower() for p in feat["product"]]
    cols["tv_mobile"] = np.array([1.0 if any(w in p for w in phage) else 0.0 for p in low])
    cols["tv_random"] = np.array([int(hashlib.sha256(("%d|random-v1" % i).encode()).hexdigest()[:8], 16) / 2.0 ** 32 for i in range(n)])
    meta = {}
    start = np.array([c[1] for c in feat["coords"]], float)
    rlen = np.array([feat["lengths"].get(c[0], 1) for c in feat["coords"]], float)
    meta["md_pos"] = start / rlen
    meta["md_origin_dist"] = np.minimum(start, rlen - start) / rlen
    meta["md_strand"] = np.array([1.0 if c[3] == "+" else 0.0 for c in feat["coords"]])
    order = np.lexsort((start, contig))
    rank = np.empty(n)
    rank[order] = np.arange(n)
    tr = {}
    for c, s in feat["trna"]:
        tr.setdefault(c, []).append(s)
    td = np.full(n, np.nan)
    for c, starts in tr.items():
        ix = np.where(contig == c)[0]
        if not len(ix):
            continue
        ss = np.sort(start[ix])
        pos_of_trna = np.searchsorted(ss, np.array(starts))              # tRNA places in gene-order units
        g_rank = np.searchsorted(ss, start[ix])
        td[ix] = np.abs(g_rank[:, None] - pos_of_trna[None, :]).min(1)
    meta["md_trna_dist"] = td
    for name, rx in CATS:
        if name == "mobile":
            meta["md_cat_mobile"] = cols["tv_mobile"]
        else:
            meta["md_cat_" + name] = np.array([1.0 if re.search(rx, p) else 0.0 for p in low])
    meta["md_cat_other"] = 1.0 - np.clip(sum(meta["md_cat_" + c] for c, _ in CATS), 0, 1)
    return cols, meta


# --- fused baselines and the shortcut detector (numpy only, fixed hyperparameters) ----------------------------------

def l1_logistic_fit(X, y, lam=0.01, iters=500):
    """L1-penalised logistic regression on standardised X by FISTA (intercept unpenalised). Returns (mu, sd, w)."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.c_[np.ones(len(X)), (X - mu) / sd]
    w = np.zeros(Z.shape[1])
    v, t = w.copy(), 1.0
    L = 0.25 * (Z ** 2).sum(1).max() + 1e-9
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ v, -30, 30)))
        g = Z.T @ (p - y) / len(y)
        w_new = v - g / L
        w_new[1:] = np.sign(w_new[1:]) * np.maximum(np.abs(w_new[1:]) - lam / L, 0.0)
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        v = w_new + (t - 1) / t_new * (w_new - w)
        w, t = w_new, t_new
    return mu, sd, w


def l1_logistic_score(fit, X):
    mu, sd, w = fit
    return np.c_[np.ones(len(X)), (X - mu) / sd] @ w


def bin_edges(x, bins):
    """Edges of one feature: midpoints between its distinct values when it has at most `bins` of them (a rare value of
    a near-binary feature stays separable), else quantile edges plus the midpoint below its largest value."""
    v = np.unique(x[np.isfinite(x)])
    if len(v) <= 1:
        return np.array([])
    if len(v) <= bins:
        return (v[:-1] + v[1:]) / 2
    return np.unique(np.r_[np.quantile(x[np.isfinite(x)], np.linspace(0, 1, bins + 1)[1:-1]), (v[-2] + v[-1]) / 2])


def gbm_fit(X, y, rounds=200, depth=2, lr=0.05, bins=32, l2=1.0, min_leaf=50):
    """Histogram gradient boosting for the logistic loss with Newton leaf values; bins from the training data
    (bin_edges; NaN in its own lowest bin); deterministic. Returns the model."""
    edges = [bin_edges(X[:, j], bins) for j in range(X.shape[1])]
    B = np.stack([_bin(X[:, j], edges[j]) for j in range(X.shape[1])], 1)
    p0 = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
    f = np.full(len(y), np.log(p0 / (1 - p0)))
    trees = []
    for _ in range(rounds):
        p = 1.0 / (1.0 + np.exp(-f))
        g, h = p - y, p * (1 - p)
        tree = _grow(B, g, h, np.arange(len(y)), depth, l2, min_leaf, bins + 2)
        f += lr * _apply(tree, B)
        trees.append(tree)
    return dict(edges=edges, base=float(np.log(p0 / (1 - p0))), lr=lr, trees=trees)


def _bin(x, e):
    b = np.searchsorted(e, x, side="right") + 1
    return np.where(np.isfinite(x), b, 0)


def _grow(B, g, h, rows, depth, l2, min_leaf, nb):
    G, H = g[rows].sum(), h[rows].sum()
    leaf = dict(value=-G / (H + l2))
    if depth == 0 or len(rows) < 2 * min_leaf:
        return leaf
    best = None
    for j in range(B.shape[1]):
        gb = np.bincount(B[rows, j], weights=g[rows], minlength=nb)
        hb = np.bincount(B[rows, j], weights=h[rows], minlength=nb)
        cb = np.bincount(B[rows, j], minlength=nb)
        gl, hl, cl = np.cumsum(gb)[:-1], np.cumsum(hb)[:-1], np.cumsum(cb)[:-1]
        ok = (cl >= min_leaf) & (len(rows) - cl >= min_leaf)
        if not ok.any():
            continue
        gain = gl ** 2 / (hl + l2) + (G - gl) ** 2 / (H - hl + l2) - G ** 2 / (H + l2)
        gain = np.where(ok, gain, -np.inf)
        k = int(np.argmax(gain))
        if best is None or gain[k] > best[0]:
            best = (gain[k], j, k)
    if best is None or best[0] <= 0:
        return leaf
    _, j, k = best
    left = rows[B[rows, j] <= k]
    right = rows[B[rows, j] > k]
    return dict(j=j, k=k, left=_grow(B, g, h, left, depth - 1, l2, min_leaf, nb), right=_grow(B, g, h, right, depth - 1, l2, min_leaf, nb))


def _apply(tree, B):
    out = np.zeros(len(B))
    stack = [(tree, np.arange(len(B)))]
    while stack:
        t, rows = stack.pop()
        if "value" in t:
            out[rows] = t["value"]
            continue
        m = B[rows, t["j"]] <= t["k"]
        stack.append((t["left"], rows[m]))
        stack.append((t["right"], rows[~m]))
    return out


def gbm_score(model, X):
    B = np.stack([_bin(X[:, j], model["edges"][j]) for j in range(X.shape[1])], 1)
    return model["base"] + model["lr"] * sum(_apply(t, B) for t in model["trees"])


def quantile_features(t, names):
    """Within-chromosome quantiles of the named columns, 0.5 with a missing flag where a value is missing."""
    from ci.stages import protocol as PR
    F = []
    for n in names:
        q = PR.within_quantile(np.asarray(t["cols"][n], float))
        miss = ~np.isfinite(q)
        F += [np.where(miss, 0.5, q), miss.astype(float)]
    return np.stack(F, 1)


def leave_unit_out(table, units, feats, kind, params):
    """{chromosome: score of every gene} from models fitted on the other units' labelled genes."""
    out = {}
    for u in units:
        tr = [t for t in table.values() if t["unit"] != u]
        X = np.concatenate([feats(t)[t["y"] >= 0] for t in tr])
        y = np.concatenate([t["y"][t["y"] >= 0] for t in tr]).astype(float)
        if not (0 < y.sum() < len(y)):
            continue
        if kind == "l1":
            fit = l1_logistic_fit(X, y, **params)
            score = lambda F: l1_logistic_score(fit, F)
        else:
            fit = gbm_fit(X, y, **params)
            score = lambda F: gbm_score(fit, F)
        for acc, t in table.items():
            if t["unit"] == u:
                out[acc] = score(feats(t))
    return out


def shuffle_control(table, names, draws, seed):
    """Mean AUROC of every column over `draws` label shuffles within each chromosome (the labelled genes permuted)."""
    from ci.stages import protocol as PR
    rng = np.random.default_rng([int(seed), 17])
    out = {n: [] for n in names}
    for _ in range(draws):
        for t in table.values():
            y = t["y"].copy()
            lab = np.where(y >= 0)[0]
            y[lab] = rng.permutation(y[lab])
            for n in names:
                v = np.asarray(t["cols"][n], float)
                a = PR.auroc(v[y == 1], v[y == 0])
                if a is not None:
                    out[n].append(a)
    return {n: float(np.mean(v)) for n, v in out.items() if v}


def within_auroc(table, scores):
    """Mean over units of the mean per-chromosome AUROC of {chromosome: score}."""
    from ci.stages import protocol as PR
    by = {}
    for acc, s in scores.items():
        t = table[acc]
        a = PR.auroc(np.asarray(s)[t["y"] == 1], np.asarray(s)[t["y"] == 0])
        if a is not None:
            by.setdefault(t["unit"], []).append(a)
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


# --- stages ------------------------------------------------------------------------------------------------------

def sources(cfg):
    """The island run's checked inputs (as codon.sources) and the records of the development chromosomes."""
    from ci.stages.codon import check_files, gate_passed, stage_output
    ac = cfg["audit"]
    isl = ac["islands_from"]
    prov = dict(islands_from=isl)
    ip = os.path.join(isl, "data", "islands", "index.json")
    prov["islands_index"] = stage_output(isl, "islands_data", "islands_index", ip)
    gate_passed(isl, "G3_islands")
    prov["islands_eval"] = stage_output(isl, "islands_report", "islands_eval", os.path.join(isl, "data", "islands", "eval", "dev.json"))
    with open(ip) as f:
        idx = json.load(f)
    with open(os.path.join(isl, "data", "islands", "manifest.json")) as f:
        msha = json.load(f)["sha256"]
    files = []
    for g in idx["genomes"]:
        files += [g["path"]] + [os.path.join(g["folder"], n) for n in ("genes.tsv", "cds.fna")]
    check_files(isl, os.path.join(isl, "manifests", msha + ".json"), files)
    return prov, idx


def do_columns(run_dir):
    with Stage(run_dir, "audit_columns", requires=("pre",)) as st:
        cfg = st.config
        unknown = set(cfg["audit"]) - set(KEYS)
        if unknown:
            raise SystemExit("audit: unknown keys %s" % sorted(unknown))
        prov, idx = sources(cfg)
        isl = cfg["audit"]["islands_from"]
        with open(os.path.join(isl, "config.resolved.yaml")) as f:
            import yaml
            store = (cfg.get("islands") or {}).get("store") or yaml.safe_load(f)["islands"]["store"]
        with open(st.repo("bench/islands/results/dev_genomes.tsv")) as f:
            dev = {r["accession"]: r for r in csv.DictReader(f, delimiter="\t")}
        bm = _bench("bench_hard_metrics", "bench/hard/metrics.py", st.repo)
        phage = _bench("bench_islands_evaluate", "bench/islands/evaluate.py", st.repo).PHAGE
        from ci.stages.common import read_fasta
        from ci.stages.islands import comparators, write_scores
        from ci.stages.search import gene_ids
        from seeddrift.train import source_model as S
        out = os.path.join(run_dir, "data", "audit", "columns")
        os.makedirs(out, exist_ok=True)
        cover, records = {}, {}
        for g in idx["genomes"]:
            acc = g["id"]
            if acc not in dev:
                raise SystemExit("audit: %s is not a development chromosome" % acc)
            gb = os.path.join(store, "dev_genomes", acc + ".gb")
            sha = sha256_file(gb, use_sidecar=False)
            if sha != dev[acc]["sha256"]:
                raise SystemExit("audit: the record of %s is not the one of dev_genomes.tsv" % acc)
            records[acc] = sha
            names = gene_ids(os.path.join(isl, g["folder"]))
            cds = read_fasta(os.path.join(isl, g["folder"], "cds.fna"))
            seqs = [cds.get("%s__%s" % (acc, n), "") for n in names]
            gen = S.Genome(dict(id=acc, taxid=g["taxid"], ranks=g["ranks"], group=g["group"]), os.path.join(isl, g["path"]))
            cmp_cols = comparators(gen)
            feat = record_features(gb, names)
            cols, meta = columns_of(feat, seqs, cmp_cols, bm, phage)
            write_scores(os.path.join(out, "scores_%s.tsv" % acc), names, dict({k: v for k, v in cmp_cols.items() if k in ("cmp_codon", "cmp_gc3", "cmp_hits", "cmp_short", "cmp_kmer")}, **cols))
            write_scores(os.path.join(out, "meta_%s.tsv" % acc), names, meta)
            cover[acc] = {k: round(float(np.isfinite(v).mean()), 4) for k, v in cols.items()}
            cover[acc]["ribosomal_reference_genes"] = int(sum(1 for p in feat["product"] if RIBO.search(p) and not RIBO_NOT.search(p)))
        cj = os.path.join(run_dir, "data", "audit", "columns.json")
        write_json(cj, dict(sources=prov, records_sha256=records, coverage=cover, columns=list(TRIVIAL), store=store))
        st.output("audit_columns", cj)


def do_report(run_dir):
    with Stage(run_dir, "audit_report", requires=("pre",)) as st:
        cfg = st.config
        ac = cfg["audit"]
        from ci.stages import protocol as PR
        from ci.stages.islands import admitted, island_table
        isl = ac["islands_from"]
        with open(os.path.join(run_dir, "data", "audit", "columns.json")) as f:
            cj = json.load(f)
        dev_json = os.path.join(isl, "data", "islands", "eval", "dev.json")
        if sha256_file(dev_json, use_sidecar=False) != cj["sources"]["islands_eval"]:
            raise SystemExit("audit: %s changed since the columns stage checked it" % dev_json)
        with open(dev_json) as f:
            res = json.load(f)
        per = admitted(res)
        store = cj["store"]
        col_dir = os.path.join(run_dir, "data", "audit", "columns")
        table, units, e5 = island_table([col_dir], store, per, res["units"])
        for acc, t in table.items():                                     # the metadata, gene order as the columns
            with open(os.path.join(col_dir, "meta_%s.tsv" % acc)) as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            if [r["gene_id"] for r in rows] != t["names"]:
                raise SystemExit("audit: the metadata of %s is not in the columns' gene order" % acc)
            t["meta"] = {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0] if k != "gene_id"}
        cmp_names = ["cmp_codon", "cmp_gc3", "cmp_hits", "cmp_short", "cmp_kmer"]
        trivial = [c for c in cmp_names + list(TRIVIAL) + e5 if all(c in t["cols"] for t in table.values())]
        fz = ac.get("fused") or {}
        feats = lambda t: quantile_features(t, [c for c in trivial if c not in ("tv_random", "e5_fuse3")])
        fused = {"fz_l1": leave_unit_out(table, units, feats, "l1", dict(lam=float(fz.get("lam", 0.01)))),
                 "fz_gbm": leave_unit_out(table, units, feats, "gbm", dict(rounds=int(fz.get("rounds", 200)), depth=int(fz.get("depth", 2)),
                                                                          lr=float(fz.get("lr", 0.05)), min_leaf=int(fz.get("min_leaf", 50))))}
        for name, sc in fused.items():
            for acc, t in table.items():
                t["cols"][name] = np.asarray(sc.get(acc, np.full(len(t["y"]), np.nan)), float)
        scores = trivial + list(fused)
        comps = [c for c in trivial]
        ev = PR.evaluate(table, units, scores, comps)
        # the margin-free cell: natives and edge genes as negatives (window scores are inflated in the gated cell)
        edge_table = {}
        for acc, t in table.items():
            with open(os.path.join(store, "labels_panel", acc + ".tsv")) as f:
                cls = {r["gene_id"]: r["class"] for r in csv.DictReader(f, delimiter="\t")}
            y = np.array([1 if cls.get(g) == "island" else 0 if cls.get(g) in ("native", "edge") else -1 for g in t["names"]])
            edge_table[acc] = dict(t, y=y)
        ev_edge = PR.evaluate(edge_table, units, scores, [])
        edge = {s: ev_edge[s]["auroc"] for s in scores if s in ev_edge}
        # the shortcut detector: metadata only
        dc = ac.get("detector") or {}
        core = ["md_pos", "md_origin_dist", "md_strand", "md_trna_dist"]
        annot = core + sorted(k for k in next(iter(table.values()))["meta"] if k.startswith("md_cat_"))
        det = {}
        fill = lambda v: np.where(np.isfinite(v), v, np.nanmedian(v) if np.isfinite(v).any() else 0.0)
        for name, fs in (("core", core), ("annotation", annot)):
            mf = lambda t, fs=fs: np.stack([fill(t["meta"][k]) for k in fs], 1)
            sc = leave_unit_out(table, units, mf, "l1", dict(lam=float(dc.get("lam", 0.01))))
            a = within_auroc(table, sc)
            det[name] = dict(features=fs, auroc_within=a, flag=bool(a is not None and a >= float(dc.get("flag_auroc", 0.6))))
        cc = ac.get("controls") or {}
        shuffled = shuffle_control(table, scores, int(cc.get("draws", 20)), int(cfg.get("seed", 1)))
        intervals = PR.paired_delta(table, units, [(s, None) for s in scores], n_boot=int(ac.get("boot", 1000)), seed=int(cfg.get("seed", 1)),
                                    block=int(ac.get("block", 10)))
        rows = []
        for s in scores:
            e = ev.get(s) or {}
            iv = next((r for r in intervals if r["a"] == s), {})
            rows.append(dict(score=s, auroc=e.get("auroc"), auroc_edge_cell=edge.get(s), lo=iv.get("lo"), hi=iv.get("hi"), residual=e.get("residual_auroc"),
                             matched=e.get("matched_auroc"), accessory=e.get("accessory_auroc"), shuffled=shuffled.get(s),
                             per_unit=e.get("auroc_per_unit"), window=s in WINDOW, in_sample=s == "e5_fuse3"))
        out = st.path("audit", "report.json")
        write_json(out, dict(note=__doc__.strip().splitlines()[0], units=units, chromosomes=sorted(table), scores=ev, table=rows, detector=det,
                             negative_control=shuffled, edge_cell=edge, fused=list(fused), trivial=trivial, columns_coverage=cj["coverage"],
                             not_implemented=list(PR.NOT_IMPLEMENTED) + ["event-connected units (genus units only; 3 units)", "tAI", "homology-based mobile genes",
                                                                         "the oracle positive controls", "the specificity audit", "co-transfer consistency",
                                                                         "the circularity register file"],
                             caveat="3 genus units: no verdict; e5_fuse3 was chosen on these chromosomes (in-sample); window scores are inflated "
                                    "in the gated cell by the 5-gene label margin (see auroc_edge_cell)"))
        tsv = st.path("audit", "trivial_table.tsv")
        with open(tsv, "w") as f:
            keys = ("score", "auroc", "lo", "hi", "auroc_edge_cell", "residual", "matched", "accessory", "shuffled", "window", "in_sample")
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join("" if r[k] is None else (("%.4f" % r[k]) if isinstance(r[k], float) else str(r[k])) for k in keys) + "\n")
        st.output("audit_report", out)
        st.output("audit_table", tsv)


def main():
    run_dir = main_args()
    import sys
    {"columns": do_columns, "report": do_report}[sys.argv[2]](run_dir)


if __name__ == "__main__":
    main()
