"""Codon-usage study of the truth-labels line (2026-11-20): why the model-free codon-usage score works on the island
chromosomes, and how far its family goes (the user: "我赞同你的去探究一下为啥密码子这条路有信号"). No training and no
label file: every column is a function of the codon counts of a scored chromosome and of panel genomes of the corpus.
Every score is "higher = more foreign", NaN where it is not defined.

The algebra. With q_k the smoothed codon frequencies of panel genome k and q_H those of the scored chromosome without
the gene, Lambda_k = sum_c N_c log(q_k,c / q_H,c) = log p(x | k) - log p(x | H) for independent codons: the
log-likelihood ratio of "the gene was drawn from genome k" against "from its own chromosome", and CU = max_k Lambda_k / n
a generalised likelihood-ratio test against the panel. Exactly Lambda_k = Lambda_k^aa + Lambda_k^syn with
q(c) = q(a(c)) q(c | a(c)): amino-acid usage and synonymous choice (genetic code 11, stop codons one class).

Columns (p = a panel of the configuration; F = the gene's smoothed frequencies (N + 0.5) / (n + 32), the weights of the
readout R_CU_<panel> of selfsup.readouts, which cd_cu_<p> reproduces; "raw" = the counts N, divided by n):
  cd_cu_<p>     max_k sum_c F_c log(q_k,c / q_H,c)
  cd_aa_<p>     the amino-acid part alone, its own max over k: sum_a F_a log(q_k(a) / q_H(a))
  cd_syn_<p>    the synonymous part alone: sum_c F_c log(q_k(c|a) / q_H(c|a))
  cd_cun_<p>    raw: max_k Lambda_k / n
  cd_tot_<p>    max_k Lambda_k in nats (the evidence of the whole gene, not per codon)
  cd_lse_<p>    raw: (log mean_k exp Lambda_k) / n, the panel as a uniform mixture instead of a maximum
  cd_std_<p>    max_k of Lambda_k / n standardised per panel genome over the chromosome's genes (median, 1.4826 MAD,
                no label): a panel genome no longer wins by its noise
  cd_m1_<p>     Markov-1: the codon given the third base of the codon before it, raw per transition
  cd_relax_<p>  two relaxation rates: the gene drawn from (1 - l_a) q_k(a) + l_a q_H(a) and (1 - l_s) q_k(c|a) + l_s q_H(c|a),
                l_a and l_s on a grid in [0, 1) (relax_grid); the best over k and the grid, raw per n (>= cd_cun: l = 0)
  host variants of cd_cun_<p> (the host density changed, the panel not):
  cd_hin_<p>    the host with the gene in it
  cd_trim_<p>   the host without the genes in the top trim share (default 15 %) of cd_cun: the chromosome's own
                transfers out of its null
  cd_block_<p>  the host without the genes within block_bp (default 25 kb) of the gene's middle (block cross-fitting: an
                island no longer teaches the host its own composition)
  cd_mix3_<p>   the host as a mixture of 3 codon-usage classes fitted on the chromosome's genes (EM, started from GC3
                tertiles); the gene's host likelihood is the mixture's
  the manifold m (an exponential-family PCA of the genome profiles of a corpus: log q_G = log softmax(b + z_G W), z_G in
  R^d, fitted by maximum likelihood; a Gaussian prior N(mu, Sigma) fitted to the genomes' positions):
  cd_mf<m><d>   raw per n: log p(x | z*) - log p(x | z_H), z* the gene's MAP position, z_H the chromosome's without the
                gene: the panel replaced by the whole continuous manifold
  cd_mfbf<m><d> the Bayes factor in nats: the Laplace marginal likelihood of the gene under the prior against p(x | z_H)
  cd_mfh<m><d>  raw per n: log p(x | z*) minus the log-likelihood under the empirical host frequencies (the null of CU)
  cd_mfd<m><d>  the Mahalanobis distance between z* and z_H in the metric of the genomes' spread
The manifold's fit is reported as deviance explained against dimension, on the genomes it is fitted on and on held-out
genomes (one in five by hash, positions fitted with W fixed): how many dimensions the codon usage of genomes has.

  python seeddrift/train/codon_study.py CONF.json      (run directory = CONF["run_dir"]; the model interpreter, GPU)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

SPLITS = ("train", "heldout")      # panels never open the sealed test part or the evaluation inputs
TCAG = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"   # NCBI table 11, codons in TCAG order
_TO_TCAG = (2, 1, 3, 0)            # A, C, G, T -> index in TCAG
RANK_COL = dict(phylum=0, **{"class": 1}, order=2, family=3, genus=4)   # train_data.RANKS = phylum, class, order, family, genus, species
SCORES = ("cu", "aa", "syn", "cun", "tot", "lse", "std", "m1", "relax", "host")
GC3 = np.array([1.0 if c % 4 in (1, 2) else 0.0 for c in range(64)])


def amino_acids():
    """The class (0..20, the stop codons one class) of each codon 0..63 (ACGT order: c = 16 b1 + 4 b2 + b3)."""
    names = sorted(set(TCAG))
    idx = np.zeros(64, np.int64)
    for c in range(64):
        b1, b2, b3 = c // 16, (c // 4) % 4, c % 4
        idx[c] = names.index(TCAG[16 * _TO_TCAG[b1] + 4 * _TO_TCAG[b2] + _TO_TCAG[b3]])
    return idx


class Light:
    """The arrays of one genome the study needs (train_data.one_genome's npz, read lazily): codon tokens 1..64 (65 =
    other), the gene offsets and, for scored chromosomes, the places of the genes."""

    def __init__(self, meta, path, places=False):
        z = np.load(path)
        self.meta = meta
        self.codons, self.coff = z["codons"], z["codon_off"]
        self.n = len(self.coff) - 1
        if places:
            self.rep, self.start, self.end = z["replicon"], z["start"], z["end"]

    def gene_counts(self):
        """[genes, 64] codon counts of every gene."""
        t = self.codons.astype(np.int64)
        gid = np.repeat(np.arange(self.n), np.diff(self.coff))
        ok = (t >= 1) & (t <= 64)
        return np.bincount(gid[ok] * 64 + t[ok] - 1, minlength=self.n * 64).reshape(self.n, 64).astype(np.float64)

    def m1_counts(self):
        """[genes, 256] Markov-1 transitions within genes: the third base of a codon (4) by the next codon (64), for
        consecutive codon tokens 1..64 of one gene."""
        t = self.codons.astype(np.int64)
        gid = np.repeat(np.arange(self.n), np.diff(self.coff))
        if len(t) < 2:
            return np.zeros((self.n, 256))
        a, b = t[:-1], t[1:]
        ok = (gid[:-1] == gid[1:]) & (a >= 1) & (a <= 64) & (b >= 1) & (b <= 64)
        key = gid[:-1][ok] * 256 + ((a[ok] - 1) % 4) * 64 + (b[ok] - 1)
        return np.bincount(key, minlength=self.n * 256).reshape(self.n, 256).astype(np.float64)

    def profile(self):
        """The genome's codon counts [64] (all its codon tokens 1..64: selfsup.readouts' ccount[1:65])."""
        t = self.codons.astype(np.int64)
        return np.bincount(t, minlength=66)[1:65].astype(np.float64)


def panel_of(ids, groups, n):
    """selfsup.panel_of on ids with their groups: up to n genomes, group by group in a fixed hash order."""
    by = {}
    for g in sorted(ids):
        by.setdefault(groups[g], []).append(g)
    for grp in by:
        by[grp].sort(key=lambda g: hashlib.sha256(("panel:" + g).encode()).hexdigest())
    out, r = [], 0
    while len(out) < n and any(r < len(v) for v in by.values()):
        for grp in sorted(by):
            if r < len(by[grp]) and len(out) < n:
                out.append(by[grp][r])
        r += 1
    return out


def base_members(spec, corpora, bases):
    """The panel's genomes before any hiding: spec base (another panel's), or corpus, splits (default train; only
    train and heldout), groups (default all), size (default all) and pick (hash: selfsup.panel_of; random: seeded)."""
    if spec.get("base"):
        return list(bases[spec["base"]])
    meta = corpora[spec["corpus"]]
    splits = list(spec.get("splits") or ["train"])
    if set(splits) - set(SPLITS):
        raise SystemExit("panel %s: splits among %s only, not %r" % (spec["name"], ", ".join(SPLITS), splits))
    groups = spec.get("groups")
    ids = sorted(g for g, m in meta.items() if m["split"] in splits and int(m.get("genes", 1)) > 0 and (groups is None or m["group"] in groups))
    size, pick = spec.get("size"), spec.get("pick", "hash")
    if size is None:
        return ids
    if pick == "hash":
        return panel_of(ids, {g: meta[g]["group"] for g in ids}, int(size))
    if pick != "random":
        raise SystemExit("panel %s: pick is hash or random, not %r" % (spec["name"], pick))
    if int(size) >= len(ids):
        return ids
    rng = np.random.default_rng([int(spec.get("seed", 0)), 11])
    return sorted(str(x) for x in rng.choice(ids, size=int(size), replace=False))


def members_for(spec, base, chrom, meta_of):
    """The panel of one scored chromosome: hide (phylum, class, order, family, genus) drops the base genomes of the chromosome's
    clade at that rank; with control random the same number of base genomes is dropped at random instead (seeded by
    the chromosome and the panel), the control of "fewer genomes" against "not the near genomes"."""
    hide = spec.get("hide")
    if not hide:
        return list(base)
    if hide not in RANK_COL:
        raise SystemExit("panel %s: hide is one of %s, not %r" % (spec["name"], ", ".join(RANK_COL), hide))
    col = RANK_COL[hide]
    own = int(chrom["ranks"][col])
    drop = {g for g in base if own and int(meta_of(g)["ranks"][col]) == own}
    if spec.get("control") == "random":
        h = int(hashlib.sha256(("%s:%s:%s" % (chrom["id"], spec["name"], spec.get("seed", 0))).encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(h)
        drop = {str(x) for x in rng.choice(sorted(base), size=len(drop), replace=False)} if drop else set()
    elif spec.get("control"):
        raise SystemExit("panel %s: control is random or absent, not %r" % (spec["name"], spec["control"]))
    return [g for g in base if g not in drop]


def expand_curve(curves):
    """The panel-size curves: per curve (name, corpus, splits, groups, sizes, seeds) random panels of each size and
    seed, named rc_<name>_<size>s<seed>, and the whole base set rc_<name>_all; score cu only."""
    out = []
    for cv in curves or []:
        for s in cv["sizes"]:
            for seed in cv.get("seeds", [1]):
                out.append(dict(name="rc_%s_%ds%d" % (cv["name"], int(s), int(seed)), corpus=cv["corpus"], splits=cv.get("splits"), groups=cv.get("groups"),
                                size=int(s), pick="random", seed=int(seed), scores=["cu"]))
        out.append(dict(name="rc_%s_all" % cv["name"], corpus=cv["corpus"], splits=cv.get("splits"), groups=cv.get("groups"), scores=["cu"]))
    return out


# --- the manifold --------------------------------------------------------------------------------------------------

def fit_manifold(torch, dev, C, d, steps, lr, seed, W=None, b=None):
    """Exponential-family PCA of profiles C [G, 64] (counts): log q_G = log_softmax(b + z_G W) by maximum likelihood with
    every genome weighted equally (its smoothed frequencies). With W and b given, only the positions z are fitted.
    Returns (b, W, Z, mean log-likelihood per codon)."""
    f = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    ft = torch.tensor(f, dtype=torch.float64, device=dev)
    clr = np.log(f) - np.log(f).mean(1, keepdims=True)
    torch.manual_seed(seed)
    if W is None:
        b0 = clr.mean(0)
        u, s, vt = np.linalg.svd(clr - b0, full_matrices=False)
        z0, w0 = u[:, :d] * s[:d], vt[:d]
        bt = torch.tensor(b0, dtype=torch.float64, device=dev, requires_grad=True)
        Wt = torch.tensor(w0, dtype=torch.float64, device=dev, requires_grad=True)
        params = [bt, Wt]
    else:
        bt = torch.tensor(b, dtype=torch.float64, device=dev)
        Wt = torch.tensor(W, dtype=torch.float64, device=dev)
        wn = np.asarray(W)
        z0 = (clr - np.asarray(b)) @ np.linalg.pinv(wn)
        params = []
    Zt = torch.tensor(z0, dtype=torch.float64, device=dev, requires_grad=True)
    opt = torch.optim.Adam(params + [Zt], lr=lr)
    for _ in range(int(steps)):
        opt.zero_grad()
        loss = -(ft * torch.log_softmax(bt + Zt @ Wt, 1)).sum(1).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        ll = float((ft * torch.log_softmax(bt + Zt @ Wt, 1)).sum(1).mean())
    return bt.detach().cpu().numpy(), Wt.detach().cpu().numpy(), Zt.detach().cpu().numpy(), ll


def manifold_curve(torch, dev, C, ids, dims, steps, lr, seed):
    """Deviance explained against dimension: (ll_d - ll_0) / (ll_sat - ll_0) per genome, ll_0 the one shared profile
    (the mean frequencies), on the fitted genomes and on held-out ones (sha256 of the id: one in five)."""
    held = np.array([int(hashlib.sha256(("manifold:" + g).encode()).hexdigest()[:8], 16) % 5 == 0 for g in ids])
    f = (C + 0.5) / (C + 0.5).sum(1, keepdims=True)
    sat = lambda m: float((f[m] * np.log(f[m])).sum(1).mean())
    zero = lambda m, ref: float((f[m] * np.log(f[ref].mean(0))).sum(1).mean())
    tr, he = ~held, held
    out = []
    for d in dims:
        b, W, _, ll_tr = fit_manifold(torch, dev, C[tr], int(d), steps, lr, seed)
        _, _, _, ll_he = fit_manifold(torch, dev, C[he], int(d), steps, lr, seed, W=W, b=b) if he.any() else (None, None, None, float("nan"))
        de = lambda ll, m: (ll - zero(m, tr)) / max(sat(m) - zero(m, tr), 1e-12)
        out.append(dict(d=int(d), ll_train=ll_tr, ll_heldout=ll_he, deviance_explained_train=de(ll_tr, tr),
                        deviance_explained_heldout=de(ll_he, he) if he.any() else None))
    return dict(genomes=int(len(ids)), heldout_genomes=int(held.sum()), curve=out)


def map_positions(torch, N, b, W, mu, P, z0, iters=30):
    """MAP positions z [B, d] of count vectors N [B, 64] on the manifold under the prior N(mu, P^-1): damped Newton on
    the concave phi(z) = sum_c N_c log softmax(b + z W)_c - (z - mu)' P (z - mu) / 2. Returns z, phi(z), the
    log-likelihood term and log det(-Hessian of phi) at z."""
    n = N.sum(1)
    z = z0.clone()

    def phi_of(zz):
        lp = torch.log_softmax(b + zz @ W, 1)
        ll = (N * lp).sum(1)
        dz = zz - mu
        return ll - 0.5 * ((dz @ P) * dz).sum(1), ll, lp
    ph, ll, lp = phi_of(z)
    eye = torch.eye(W.shape[0], dtype=W.dtype, device=W.device)
    for _ in range(iters):
        p = lp.exp()
        g = (N - n[:, None] * p) @ W.T - (z - mu) @ P
        wp = p @ W.T
        wdw = torch.einsum("bc,dc,ec->bde", p, W, W)
        neg_h = n[:, None, None] * (wdw - wp[:, :, None] * wp[:, None, :]) + P + 1e-9 * eye
        step = torch.linalg.solve(neg_h, g.unsqueeze(-1)).squeeze(-1)
        t = torch.ones(len(z), dtype=z.dtype, device=z.device)
        best_z, best = z, ph
        for _h in range(6):                                           # halve the step where phi does not rise
            zn = z + t[:, None] * step
            pn, _, _ = phi_of(zn)
            ok = pn >= best
            best_z = torch.where(ok[:, None], zn, best_z)
            best = torch.where(ok, pn, best)
            t = torch.where(ok, torch.zeros_like(t), t * 0.5)
            if not bool((t > 0).any()):
                break
        z = best_z
        ph, ll, lp = phi_of(z)
    p = lp.exp()
    wp = p @ W.T
    wdw = torch.einsum("bc,dc,ec->bde", p, W, W)
    neg_h = n[:, None, None] * (wdw - wp[:, :, None] * wp[:, None, :]) + P + 1e-9 * eye
    return z, ph, ll, torch.linalg.slogdet(neg_h)[1]


# --- one chromosome ------------------------------------------------------------------------------------------------

def logprof(torch, C):
    """Smoothed log frequencies of count rows C [K, 64] (+0.5 per codon, as selfsup.readouts)."""
    C = C + 0.5
    return torch.log(C / C.sum(-1, keepdim=True))


def block_hosts(gen, N, bp):
    """[genes, 64] the chromosome's codon counts without the genes whose middles lie within bp of the gene's middle
    on its replicon (the gene included)."""
    T = N.sum(0)
    mid = (np.asarray(gen.start, np.float64) + np.asarray(gen.end, np.float64)) / 2.0
    out = np.empty_like(N)
    for r in np.unique(gen.rep):
        idx = np.where(gen.rep == r)[0]
        o = idx[np.argsort(mid[idx], kind="mergesort")]
        m = mid[o]
        pre = np.vstack([np.zeros((1, 64)), np.cumsum(N[o], 0)])
        lo = np.searchsorted(m, m - bp, side="left")
        hi = np.searchsorted(m, m + bp, side="right")
        out[o] = T[None, :] - (pre[hi] - pre[lo])
    return out


def mixture_host(torch, Nt, valid, m=3, iters=100):
    """Log-likelihood of every gene under a mixture of m codon-usage classes fitted by EM on the chromosome's genes
    with codons (started from GC3 tertiles); returns [genes] and the class weights."""
    dev = Nt.device
    n = Nt.sum(1)
    gc3 = (Nt @ torch.tensor(GC3, dtype=Nt.dtype, device=dev)) / n.clamp(min=1)
    v = torch.tensor(valid, device=dev)
    qs = torch.quantile(gc3[v], torch.linspace(0, 1, m + 1, dtype=Nt.dtype, device=dev)[1:-1])
    cls = torch.bucketize(gc3, qs)
    r = torch.nn.functional.one_hot(cls, m).to(Nt.dtype) * v[:, None]
    for _ in range(iters):
        q = r.T @ Nt + 0.5
        lq = torch.log(q / q.sum(1, keepdim=True))
        pi = (r.sum(0) / r.sum()).clamp(min=1e-12)
        lp = Nt @ lq.T + torch.log(pi)[None, :]
        r = torch.softmax(lp, 1) * v[:, None]
    return torch.logsumexp(lp, 1), pi.cpu().numpy()


def panel_columns(want, p):
    """The columns a panel with scores `want` writes (score_chromosome writes exactly these)."""
    out = ["cd_cu_%s" % p]
    out += ["cd_%s_%s" % (s, p) for s in ("aa", "syn") if s in want]
    if want & {"cun", "host", "relax"}:
        out.append("cd_cun_%s" % p)
    out += ["cd_%s_%s" % (s, p) for s in ("tot", "lse", "std", "m1", "relax") if s in want]
    if "host" in want:
        out += ["cd_%s_%s" % (s, p) for s in ("hin", "trim", "block", "mix3")]
    return out


def score_chromosome(torch, dev, gen, panels, prof, m1prof, manifolds, cfg):
    """Every column of one chromosome. panels: [(spec, members)]; prof / m1prof: id -> counts; manifolds: fitted."""
    f64 = dict(dtype=torch.float64, device=dev)
    N = gen.gene_counts()
    Nt = torch.tensor(N, **f64)
    n = Nt.sum(1)
    valid = (n > 0).cpu().numpy()
    nan = torch.full_like(n, float("nan"))
    per_n = lambda x: torch.where(n > 0, x / n.clamp(min=1), nan)
    T = Nt.sum(0)
    F = (Nt + 0.5) / (Nt + 0.5).sum(1, keepdim=True)
    rest = T[None, :] - Nt
    LH = logprof(torch, rest)
    aa = torch.tensor(amino_acids(), device=dev)
    A = torch.nn.functional.one_hot(aa, 21).to(torch.float64)
    cols, diag = {}, {}
    need_m1 = any("m1" in (s.get("scores") or ["cu"]) for s, _ in panels)
    if need_m1:
        M1 = torch.tensor(gen.m1_counts(), **f64)
        m_i = M1.sum(1)
        rest1 = (M1.sum(0)[None, :] - M1 + 0.5).view(-1, 4, 64)
        LH1 = torch.log(rest1 / rest1.sum(-1, keepdim=True)).view(-1, 256)
    grid = [float(x) for x in cfg.get("relax_grid", [0.0, 0.25, 0.5, 0.75, 0.9])]
    for spec, members in panels:
        p = spec["name"]
        want = set(spec.get("scores") or ["cu"])
        if not members:                                             # every column of the panel, NaN (the clade hidden was all of it)
            for c in panel_columns(want, p):
                cols[c] = np.full(gen.n, np.nan)
            diag["win_%s" % p] = np.full(gen.n, -1)
            continue
        Q = torch.tensor(np.stack([prof[g] for g in members]), **f64)
        LQ = logprof(torch, Q)
        sF = F @ LQ.T                                                  # [genes, K]
        cu, win = sF.max(1)
        cols["cd_cu_%s" % p] = (cu - (F * LH).sum(1)).cpu().numpy()
        diag["win_%s" % p] = np.where(valid, win.cpu().numpy(), -1)
        if want & {"aa", "syn", "relax"}:
            q = torch.exp(LQ)
            qa = q @ A
            LQa = torch.log(qa)
            LQs = LQ - LQa[:, aa]
            qh = torch.exp(LH)
            LHa = torch.log(qh @ A)
            LHs = LH - LHa[:, aa]
        if "aa" in want:
            Fa = F @ A
            cols["cd_aa_%s" % p] = ((Fa @ LQa.T).max(1).values - (Fa * LHa).sum(1)).cpu().numpy()
        if "syn" in want:
            cols["cd_syn_%s" % p] = ((F @ LQs.T).max(1).values - (F * LHs).sum(1)).cpu().numpy()
        lam = Nt @ LQ.T - (Nt * LH).sum(1, keepdim=True)                  # Lambda_k, raw [genes, K]
        cun = per_n(lam.max(1).values)
        if "cun" in want or "host" in want or "relax" in want:
            cols["cd_cun_%s" % p] = cun.cpu().numpy()
        if "tot" in want:
            cols["cd_tot_%s" % p] = torch.where(n > 0, lam.max(1).values, nan).cpu().numpy()
        if "lse" in want:
            cols["cd_lse_%s" % p] = per_n(torch.logsumexp(lam, 1) - math.log(len(members))).cpu().numpy()
        if "std" in want:
            r = lam / n.clamp(min=1)[:, None]
            rv = r[torch.tensor(valid, device=dev)]
            med = rv.median(0).values
            mad = (1.4826 * (rv - med).abs().median(0).values).clamp(min=1e-6)
            cols["cd_std_%s" % p] = torch.where(n > 0, ((r - med) / mad).max(1).values, nan).cpu().numpy()
        if "m1" in want:
            Q1 = torch.tensor(np.stack([m1prof[g] for g in members]), **f64).view(-1, 4, 64) + 0.5
            LQ1 = torch.log(Q1 / Q1.sum(-1, keepdim=True)).view(-1, 256)
            lam1 = M1 @ LQ1.T - (M1 * LH1).sum(1, keepdim=True)
            cols["cd_m1_%s" % p] = torch.where(m_i > 0, lam1.max(1).values / m_i.clamp(min=1), nan).cpu().numpy()
        if "relax" in want:
            Na = Nt @ A
            qa_k, qs_k = torch.exp(LQa), torch.exp(LQs)
            best, la_out, ls_out = [], [], []
            for b0 in range(0, gen.n, int(cfg.get("relax_batch", 128))):
                sl = slice(b0, b0 + int(cfg.get("relax_batch", 128)))
                qaH, qsH = torch.exp(LHa[sl]), torch.exp(LHs[sl])
                ba = bs = None
                for li, lv in enumerate(grid):
                    mix_a = torch.log((1 - lv) * qa_k[None] + lv * qaH[:, None]) - LHa[sl][:, None]          # [B, K, 21]
                    va = (Na[sl][:, None, :] * mix_a).sum(-1)
                    mix_s = torch.log((1 - lv) * qs_k[None] + lv * qsH[:, None]) - LHs[sl][:, None]          # [B, K, 64]
                    vs = (Nt[sl][:, None, :] * mix_s).sum(-1)
                    if ba is None:
                        ba, bs = va, vs
                        ia = torch.zeros_like(va, dtype=torch.long)
                        is_ = torch.zeros_like(vs, dtype=torch.long)
                    else:
                        ia = torch.where(va > ba, torch.full_like(ia, li), ia)
                        is_ = torch.where(vs > bs, torch.full_like(is_, li), is_)
                        ba, bs = torch.maximum(ba, va), torch.maximum(bs, vs)
                tot, k = (ba + bs).max(1)
                best.append(tot)
                g = torch.tensor(grid, **f64)
                la_out.append(g[ia.gather(1, k[:, None]).squeeze(1)])
                ls_out.append(g[is_.gather(1, k[:, None]).squeeze(1)])
            cols["cd_relax_%s" % p] = per_n(torch.cat(best)).cpu().numpy()
            diag["relax_la_%s" % p] = torch.cat(la_out).cpu().numpy()
            diag["relax_ls_%s" % p] = torch.cat(ls_out).cpu().numpy()
        if "host" in want:
            top_q = (Nt @ LQ.T).max(1).values                            # the panel side; the host side is changed

            def with_host(H):
                return per_n(top_q - (Nt * logprof(torch, H)).sum(1)).cpu().numpy()
            cols["cd_hin_%s" % p] = with_host(T[None, :].expand_as(Nt))
            c0 = cun.cpu().numpy()
            thr = np.nanquantile(c0[valid], 1.0 - float(cfg.get("trim", 0.15))) if valid.any() else np.inf
            top = torch.tensor(np.where(np.isfinite(c0), c0 > thr, False), device=dev)
            Tt = T - Nt[top].sum(0)
            cols["cd_trim_%s" % p] = with_host(Tt[None, :] - Nt * (~top)[:, None].to(Nt.dtype))
            cols["cd_block_%s" % p] = with_host(torch.tensor(block_hosts(gen, N, float(cfg.get("block_bp", 25000))), **f64))
            hl, pi = mixture_host(torch, Nt, valid)
            cols["cd_mix3_%s" % p] = per_n(top_q - hl).cpu().numpy()
            diag["mix3_weights_%s" % p] = pi
    for m in manifolds:
        b, W = torch.tensor(m["b"], **f64), torch.tensor(m["W"], **f64)
        mu, P = torch.tensor(m["mu"], **f64), torch.tensor(m["P"], **f64)
        d = W.shape[0]
        zT, _, _, _ = map_positions(torch, T[None, :], b, W, mu, P, mu[None, :])
        bs = int(cfg.get("manifold_batch", 512))
        outs = {k: [] for k in ("mf", "mfbf", "mfh", "mfd", "z", "zh")}
        for b0 in range(0, gen.n, bs):
            sl = slice(b0, b0 + bs)
            Ns, Rs = Nt[sl], rest[sl]
            z, ph, ll, ld = map_positions(torch, Ns, b, W, mu, P, zT.expand(len(Ns), -1).clone())
            zh, _, _, _ = map_positions(torch, Rs, b, W, mu, P, zT.expand(len(Ns), -1).clone(), iters=8)
            llh = (Ns * torch.log_softmax(b + zh @ W, 1)).sum(1)
            logm = ph - 0.5 * m["logdet_sigma"] - 0.5 * ld                  # Laplace: log of the marginal under the prior
            outs["mf"].append(ll - llh)
            outs["mfbf"].append(logm - llh)
            outs["mfh"].append(ll - (Ns * LH[sl]).sum(1))
            dz = z - zh
            outs["mfd"].append(((dz @ P) * dz).sum(1).clamp(min=0).sqrt())
            outs["z"].append(z)
            outs["zh"].append(zh)
        tag = "%s%d" % (m["name"], d)
        nn = n
        cols["cd_mf%s" % tag] = torch.where(nn > 0, torch.cat(outs["mf"]) / nn.clamp(min=1), nan).cpu().numpy()
        cols["cd_mfbf%s" % tag] = torch.where(nn > 0, torch.cat(outs["mfbf"]), nan).cpu().numpy()
        cols["cd_mfh%s" % tag] = torch.where(nn > 0, torch.cat(outs["mfh"]) / nn.clamp(min=1), nan).cpu().numpy()
        cols["cd_mfd%s" % tag] = torch.where(nn > 0, torch.cat(outs["mfd"]), nan).cpu().numpy()
        diag["z_%s" % tag] = torch.cat(outs["z"]).cpu().numpy()
        diag["zh_%s" % tag] = torch.cat(outs["zh"]).cpu().numpy()
    return cols, diag


def main():
    import torch
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    run_dir = cfg["run_dir"]
    dev = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    t0 = time.time()
    corpora, paths = {}, {}
    for name, c in cfg["corpora"].items():
        with open(c["index"]) as f:
            idx = json.load(f)
        corpora[name] = {m["id"]: m for m in idx["genomes"] if m["split"] in SPLITS}
        paths[name] = os.path.dirname(c["index"])
    meta_of = lambda g: next(corpora[c][g] for c in corpora if g in corpora[c])
    with open(cfg["islands_index"]) as f:
        chroms = json.load(f)["genomes"]
    specs = [dict(s) for s in cfg.get("panels") or []] + expand_curve(cfg.get("curves"))
    names = [s["name"] for s in specs]
    if len(set(names)) != len(names):
        raise SystemExit("panel names repeat: %r" % names)
    bases, corpus_of = {}, {}
    for s in specs:                                  # in order: a panel's base is defined before it
        if set(s.get("scores") or ["cu"]) - set(SCORES):
            raise SystemExit("panel %s: scores among %s" % (s["name"], ", ".join(SCORES)))
        if s.get("base"):
            if s["base"] not in bases:
                raise SystemExit("panel %s: its base %s is not a panel defined before it" % (s["name"], s["base"]))
            corpus_of[s["name"]] = corpus_of[s["base"]]          # a chain of bases resolves to the first panel's corpus
        elif s.get("corpus") in corpora:
            corpus_of[s["name"]] = s["corpus"]
        else:
            raise SystemExit("panel %s: corpus %r is not one of %s" % (s["name"], s.get("corpus"), sorted(corpora)))
        bases[s["name"]] = base_members(s, corpora, bases)
    # every genome a panel or a manifold needs, read once
    need = {}
    for s in specs:
        for g in bases[s["name"]]:
            need[g] = corpus_of[s["name"]]
    need_m1 = {g for s in specs if "m1" in (s.get("scores") or []) for g in bases[s["name"]]}
    mspecs = cfg.get("manifolds") or []
    m_ids = {}
    for m in mspecs:
        meta = corpora[m["corpus"]]
        splits = list(m.get("splits") or ["train"])
        if set(splits) - set(SPLITS):
            raise SystemExit("manifold %s: splits among %s only" % (m["name"], ", ".join(SPLITS)))
        m_ids[m["name"]] = sorted(g for g, x in meta.items() if x["split"] in splits and int(x.get("genes", 1)) > 0
                                  and (m.get("groups") is None or x["group"] in m["groups"]))
        for g in m_ids[m["name"]]:
            need.setdefault(g, m["corpus"])
    prof, m1prof = {}, {}
    for g, c in sorted(need.items()):
        lg = Light(corpora[c][g], os.path.join(paths[c], g + ".npz"))
        prof[g] = lg.profile()
        if g in need_m1:
            m1prof[g] = lg.m1_counts().sum(0)
    t_load = time.time() - t0
    report = dict(device=str(dev), panels={}, manifolds={}, chromosomes={}, corpora={k: len(v) for k, v in corpora.items()})
    fitted = []
    for m in mspecs:
        ids = m_ids[m["name"]]
        C = np.stack([prof[g] for g in ids])
        steps, lr, seed = int(m.get("steps", 3000)), float(m.get("lr", 0.05)), int(m.get("seed", 1))
        rep = manifold_curve(torch, dev, C, ids, m.get("dims_curve", [1, 2, 4, 8]), steps, lr, seed)
        rep["members"] = ids
        for d in m.get("dims_score", [4]):
            b, W, Z, ll = fit_manifold(torch, dev, C, int(d), steps, lr, seed)
            mu = Z.mean(0)
            Sg = np.cov(Z.T).reshape(int(d), int(d)) + 1e-6 * np.eye(int(d))
            fitted.append(dict(name=m["name"], b=b, W=W, mu=mu, P=np.linalg.inv(Sg), logdet_sigma=float(np.linalg.slogdet(Sg)[1])))
            rep.setdefault("fits", []).append(dict(d=int(d), ll=ll))
        report["manifolds"][m["name"]] = rep
    out_dir = os.path.join(run_dir, cfg["out"])
    os.makedirs(out_dir, exist_ok=True)
    for s in specs:
        report["panels"][s["name"]] = dict(spec={k: v for k, v in s.items()}, corpus=corpus_of[s["name"]], base=bases[s["name"]], per_chromosome={})
    t1 = time.time()
    for ch in chroms:
        gen = Light(ch, os.path.join(cfg["islands_run"], ch["path"]), places=True)
        panels = []
        for s in specs:
            mem = members_for(s, bases[s["name"]], ch, meta_of)
            panels.append((s, mem))
            if s.get("hide"):
                report["panels"][s["name"]]["per_chromosome"][ch["id"]] = mem
        cols, diag = score_chromosome(torch, dev, gen, panels, prof, m1prof, fitted, cfg)
        np.savez_compressed(os.path.join(out_dir, ch["id"] + ".part.npz"), gene=np.arange(gen.n), **{"col_" + k: v for k, v in cols.items()},
                            **{"diag_" + k: v for k, v in diag.items()})
        os.replace(os.path.join(out_dir, ch["id"] + ".part.npz"), os.path.join(out_dir, ch["id"] + ".npz"))
        report["chromosomes"][ch["id"]] = dict(genes=int(gen.n), columns=len(cols))
    report["columns"] = sorted(cols) if chroms else []
    report["seconds"] = dict(load=round(t_load, 1), score=round(time.time() - t1, 1), total=round(time.time() - t0, 1))
    with open(os.path.join(out_dir, "study.json.part"), "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    os.replace(os.path.join(out_dir, "study.json.part"), os.path.join(out_dir, "study.json"))


if __name__ == "__main__":
    main()
