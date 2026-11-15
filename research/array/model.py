"""The array model: genes of one genome (columns) against anonymous reference proteomes (rows).

Input of one example: a query gene and a few other genes of a genome, each with its members in the same
reference proteomes, as raw aligned residues in the gene's coordinates. No taxonomy, no proteome identity.
Output: the probability that the query gene does not belong with the other genes, and a pointer to the row
nearest to where it does belong.

Arms through one trunk:
  mode   "none": only which cells exist (the floor)
         "num":  relations between two aligned sequences are fixed summary numbers (identity, coverage, score)
         "msa":  the same numbers plus relations learned from the aligned residues (starts out equal to "num")
         "plm":  as "msa", and every position of a gene also carries the state of a pretrained protein language
                 model at that site (sites.py), so members are judged against what the language model expects there
  pairs  False: only gene-versus-member relations
         True:  also member-versus-member within every gene, their consensus over the genome's genes (a state
                for every pair of rows), and the column profile across members as site context
  comp   True: each gene's amino-acid composition is given too (a control for what "msa" could read off)
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OTHER, GAP, NOALN, PAD = 20, 21, 22, 23
AA = "ACDEFGHIKLMNPQRSTVWY"
BANDS = ((0.0, 0.8), (0.8, 1.3), (1.3, float("inf")))     # donor distance classes: near, mid, far (or no defined distance)
_B62 = """
 4 0 -2 -1 -2 0 -2 -1 -1 -1 -1 -2 -1 -1 -1 1 0 0 -3 -2
 0 9 -3 -4 -2 -3 -3 -1 -3 -1 -1 -3 -3 -3 -3 -1 -1 -1 -2 -2
 -2 -3 6 2 -3 -1 -1 -3 -1 -4 -3 1 -1 0 -2 0 -1 -3 -4 -3
 -1 -4 2 5 -3 -2 0 -3 1 -3 -2 0 -1 2 0 0 -1 -2 -3 -2
 -2 -2 -3 -3 6 -3 -1 0 -3 0 0 -3 -4 -3 -3 -2 -2 -1 1 3
 0 -3 -1 -2 -3 6 -2 -4 -2 -4 -3 0 -2 -2 -2 0 -2 -3 -2 -3
 -2 -3 -1 0 -1 -2 8 -3 -1 -3 -2 1 -2 0 0 -1 -2 -3 -2 2
 -1 -1 -3 -3 0 -4 -3 4 -3 2 1 -3 -3 -3 -3 -2 -1 3 -3 -1
 -1 -3 -1 1 -3 -2 -1 -3 5 -2 -1 0 -1 1 2 0 -1 -2 -3 -2
 -1 -1 -4 -3 0 -4 -3 2 -2 4 2 -3 -3 -2 -2 -2 -1 1 -2 -1
 -1 -1 -3 -2 0 -3 -2 1 -1 2 5 -2 -2 0 -1 -1 -1 1 -1 -1
 -2 -3 1 0 -3 0 1 -3 0 -3 -2 6 -2 0 0 1 0 -3 -4 -2
 -1 -3 -1 -1 -4 -2 -2 -3 -1 -3 -2 -2 7 -1 -2 -1 -1 -2 -4 -3
 -1 -3 0 2 -3 -2 0 -3 1 -2 0 0 -1 5 1 0 -1 -2 -2 -1
 -1 -3 -2 0 -3 -2 0 -3 2 -2 -1 0 -2 1 5 -1 -1 -3 -3 -2
 1 -1 0 0 -2 0 -1 -2 0 -2 -1 1 -1 0 -1 4 1 -2 -3 -2
 0 -1 -1 -1 -2 -2 -2 -1 -1 -1 -1 0 -1 -1 -1 1 5 0 -2 -2
 0 -1 -3 -2 -1 -3 -3 3 -2 1 1 -3 -2 -2 -3 -2 0 4 -3 -1
 -3 -2 -4 -3 1 -2 -2 -3 -3 -2 -1 -4 -4 -2 -3 -3 -2 -3 11 2
 -2 -2 -3 -2 3 -3 2 -1 -2 -1 -1 -2 -3 -1 -2 -2 -2 -1 2 7
"""                                                           # BLOSUM62 in the order ACDEFGHIKLMNPQRSTVWY


def blosum21():
    b = np.zeros((21, 21), dtype=np.float32)
    b[:20, :20] = np.array(_B62.split(), dtype=np.float32).reshape(20, 20)
    return torch.tensor(b)


# ---------------------------------------------------------------- data

class Store:
    """Aligned rows of every gene (memory-mapped) and the genome profiles."""

    def __init__(self, data_dir, names, profiles, sites=False):
        prof = np.load(profiles)
        self.D, self.own = prof["D"], prof["own"]
        self.genomes = [str(x) for x in prof["genome"]]
        gix = {n: i for i, n in enumerate(self.genomes)}
        self.n_prot = self.D.shape[1]
        self.in_panel = np.zeros(self.n_prot, dtype=bool)
        self.in_panel[prof["panel"]] = True
        self.sites = [np.load(os.path.join(data_dir, name + ".site.npy"), mmap_mode="r") for name in names] if sites else None
        self.msa, parts = [], {k: [] for k in ("L", "genome", "cs", "cl", "codes", "roff", "rp", "rs", "base", "part", "gid")}
        c0 = r0 = 0
        for k, name in enumerate(names):
            z = np.load(os.path.join(data_dir, name + ".index.npz"))
            self.msa.append(np.load(os.path.join(data_dir, name + ".msa.npy"), mmap_mode="r"))
            n = len(z["gene_L"])
            parts["L"].append(z["gene_L"].astype(np.int64))
            parts["genome"].append(np.array([gix[str(x)] for x in z["gene_genome"]], dtype=np.int64))
            parts["cs"].append(z["code_start"][:-1] + c0)
            parts["cl"].append(z["code_start"][:-1])
            parts["codes"].append(z["codes"])
            parts["roff"].append(z["row_off"][:-1] + r0)
            parts["rp"].append(z["row_p"])
            parts["rs"].append(z["row_score"])
            parts["base"].append(z["gene_base"][:-1])
            parts["part"].append(np.full(n, k, dtype=np.int64))
            parts["gid"].append(z["gene_id"])
            c0 += len(z["codes"]); r0 += len(z["row_p"])
        cat = {k: np.concatenate(v) for k, v in parts.items()}
        self.L, self.genome, self.cs, self.all_codes, self.cl = cat["L"], cat["genome"], cat["cs"], cat["codes"], cat["cl"]
        self.row_off = np.r_[cat["roff"], r0]
        self.row_p, self.base, self.part, self.gene_id = cat["rp"], cat["base"], cat["part"], cat["gid"]
        self.row_score = cat["rs"] / np.repeat(self.L, np.diff(self.row_off)).astype(np.float32)
        order = np.argsort(self.genome, kind="stable")
        b = np.flatnonzero(np.r_[True, self.genome[order][1:] != self.genome[order][:-1], True])
        self.genes_of = {int(self.genome[order[s]]): order[s:e] for s, e in zip(b[:-1], b[1:])}
        self._near = {}

    def codes(self, g):
        return self.all_codes[self.cs[g]:self.cs[g] + self.L[g]]

    def site(self, g):
        return self.sites[self.part[g]][self.cl[g]:self.cl[g] + self.L[g]]

    def rows(self, g):
        return self.row_p[self.row_off[g]:self.row_off[g + 1]]

    def scores(self, g):
        return self.row_score[self.row_off[g]:self.row_off[g + 1]]

    def row(self, g, k):
        L = self.L[g]
        s = self.base[g] + k * L
        return self.msa[self.part[g]][s:s + L]

    def near(self, genome):
        """Panel proteomes sorted by the genome's distance to them (nearest first; own proteome left out).
        Only proteomes that enough of the genome's genes reach have a distance (build.py profile)."""
        if genome not in self._near:
            D = self.D[genome]
            c = np.flatnonzero(self.in_panel & np.isfinite(D))
            c = c[c != self.own[genome]]
            self._near[genome] = c[np.argsort(D[c], kind="stable")]
        return self._near[genome]


@dataclass
class Cfg:
    M: int = 12                  # other genes of the genome shown with the query
    R_near: int = 12             # rows nearest to the genome
    R_strat: int = 12            # rows spread over the rest of the genome's distance range
    R_q: int = 8                 # rows where the query itself has its best members
    P: int = 384                 # gene positions read per gene
    hide: tuple = (0, 0, 0, 1, 2, 4, 8, 16, 32)     # how many nearest rows are hidden, around the genome and around the donor alike
    contam: tuple = (0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 4)   # how many of the shown genes come from another genome
    blind: bool = False          # control: the query's own cells are blanked

    @property
    def R(self):
        return self.R_near + self.R_strat + self.R_q


def hidden_mask(st, cfg, rng, g, t, k_g=None, k_t=None, extra=None):
    """Proteomes that may not serve as rows: the genome itself and its k nearest, the donor itself and its k nearest."""
    h = np.zeros(st.n_prot, dtype=bool) if extra is None else extra.copy()
    k_g = int(rng.choice(cfg.hide)) if k_g is None else k_g
    k_t = int(rng.choice(cfg.hide)) if k_t is None else k_t
    for genome, k in ((g, k_g), (t, k_t)):
        if genome is None:
            continue
        if st.own[genome] >= 0:
            h[st.own[genome]] = True
        if k:
            h[st.near(genome)[:k]] = True
    return h, k_g, k_t


def choose_rows(st, cfg, rng, g, x, hidden):
    """The R rows of an example and, for each, whether it is one of the genome's nearest."""
    avail = st.in_panel & ~hidden
    order = st.near(g)
    order = order[avail[order]]
    chosen = list(order[:cfg.R_near])
    n_near = len(chosen)
    rest = order[cfg.R_near:]
    if len(rest):
        edges = np.linspace(0, len(rest), cfg.R_strat + 1).astype(int)
        chosen += [rest[rng.integers(a, b)] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    rp, sc = st.rows(x), st.scores(x)
    ok = avail[rp] & ~np.isin(rp, chosen)
    chosen += list(rp[ok][np.argsort(-sc[ok], kind="stable")[:cfg.R_q]])
    if len(chosen) < cfg.R:
        pool = np.flatnonzero(avail)
        pool = pool[~np.isin(pool, chosen)]
        chosen += list(rng.choice(pool, cfg.R - len(chosen), replace=False))
    rows = np.array(chosen[:cfg.R], dtype=np.int64)
    near = np.arange(cfg.R) < n_near
    perm = rng.permutation(cfg.R)
    return rows[perm], near[perm]


def example(st, cfg, rng, g, x, ctx, hidden, truth):
    """Arrays of one example. Row 0 of every gene's block is the gene itself; then the R shared rows."""
    rows, near = choose_rows(st, cfg, rng, g, x, hidden)
    genes = [int(x)] + [int(c) for c in ctx]
    G, R, P = len(genes), cfg.R, cfg.P
    msa = np.full((G, R + 1, P), PAD, dtype=np.int8)
    present = np.zeros((G, R), dtype=bool)
    site = np.zeros((G, P, st.sites[0].shape[1]), dtype=np.float16) if st.sites is not None else None
    for i, gene in enumerate(genes):
        L = int(st.L[gene])
        pos = np.sort(rng.choice(L, P, replace=False)) if L > P else np.arange(L)
        n = len(pos)
        msa[i, 0, :n] = st.codes(gene)[pos]
        msa[i, 1:, :n] = NOALN
        if site is not None:
            site[i, :n] = st.site(gene)[pos]
        rp = st.rows(gene)
        if len(rp):
            k = np.searchsorted(rp, rows)
            k[k == len(rp)] = 0
            hit = rp[k] == rows
            for j in np.flatnonzero(hit):
                msa[i, 1 + j, :n] = st.row(gene, int(k[j]))[pos]
            present[i] = hit
    out = rules(msa, present, near)
    ptr = -1
    if truth is not None and present[0].any():
        d = np.where(present[0], st.D[truth][rows], np.nan)
        if np.isfinite(d).any():
            ptr = int(np.nanargmin(d))
    q_sim = out.pop("q_sim")
    out.update(ptr=np.int64(ptr), ptr_rule=np.int64(int(np.argmax(q_sim)) if present[0].any() else -2),
               n_present=np.int64(present[0].sum()))
    if cfg.blind:                                     # control: nothing of the query is shown
        msa[0] = PAD
        present[0] = False
        if site is not None:
            site[0] = 0
    out.update(msa=msa, present=present)
    if site is not None:
        out.update(site=site)
    return out


def rules(msa, present, near):
    """Fixed rules on exactly the rows and positions the model reads (higher = more foreign).
    Similarity of a gene to a member = identical positions per gene position read (identity x coverage).
      rule_ai   best similarity of the query outside the genome's nearest rows minus its best inside them
      rule_ctx  largest excess, over the rows, of the query's similarity over the median of the genome's genes"""
    g0, m = msa[:, :1], msa[:, 1:]
    npos = np.maximum(1, (g0 != PAD).sum(-1))
    sim = ((m == g0) & (m < 20)).sum(-1) / npos                 # [G, R]
    q = np.where(present[0], sim[0], 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(np.where(present[1:], sim[1:], np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inside = q[near].max() if near.any() else 0.0
    outside = q[~near].max() if (~near).any() else 0.0
    return {"rule_ai": np.float32(outside - inside), "rule_ctx": np.float32((q - med).max()), "q_sim": q}


class Donors:
    """For every focal genome, the candidate donor genomes with the focal genome's distance to them."""

    def __init__(self, st, donors):
        self.st, self.donors, self._tab = st, np.array(sorted(donors)), {}

    def pick(self, rng, g, kind):
        if g not in self._tab:
            d = self.st.D[g][self.st.own[self.donors]]
            d = np.where(np.isfinite(d), d, np.inf).astype(np.float64)
            keep = self.donors != g
            o = np.argsort(d[keep], kind="stable")
            self._tab[g] = (self.donors[keep][o], d[keep][o])
        t, d = self._tab[g]
        lo, hi = BANDS[kind]
        idx = np.flatnonzero((d >= lo) & ((d < hi) | (kind == 2)))
        if len(idx) == 0:                              # no donor in this class: the nearest ones (near) or any
            idx = np.arange(min(5, len(t))) if kind == 0 else np.arange(len(t))
        j = int(idx[rng.integers(len(idx))])
        return int(t[j]), float(d[j])


def draw(st, cfg, rng, g, donors, label=None, kind=None, k_g=None, k_t=None, n_c=None):
    """One labelled example for focal genome g. label 1: the query is a gene of another genome (the donor).
    A donor is drawn for label 0 too, its surroundings are hidden the same way, and genes of it may be among
    the shown genes, so that only the query itself differs between the labels."""
    label = int(rng.random() < 0.5) if label is None else label
    kind = int(rng.integers(3)) if kind is None else kind
    t, d_gt = donors.pick(rng, g, kind)
    hidden, k_g, k_t = hidden_mask(st, cfg, rng, g, t, k_g, k_t)
    mine = st.genes_of[g]
    pick = rng.permutation(len(mine))[:cfg.M + 1]
    ctx = mine[pick[1:]].copy()
    xf = int(rng.choice(st.genes_of[t]))               # drawn for both labels, so a native and a foreign trial share every other draw
    x = xf if label else int(mine[pick[0]])
    n_c = int(rng.choice(cfg.contam)) if n_c is None else n_c
    for j in rng.choice(len(ctx), n_c, replace=False):
        src = t if rng.random() < 0.5 else donors.pick(rng, g, int(rng.integers(3)))[0]
        pool = st.genes_of[src]
        c = int(rng.choice(pool))
        if c == x:
            c = int(pool[(int(np.flatnonzero(pool == c)[0]) + 1) % len(pool)])
        ctx[j] = c
    out = example(st, cfg, rng, g, x, ctx, hidden, t if label else g)
    out.update(label=np.float32(label), kind=np.int64(kind), k_g=np.int64(k_g), k_t=np.int64(k_t), g=np.int64(g), x=np.int64(x),
               donor=np.int64(t), d_gt=np.float32(d_gt if np.isfinite(d_gt) else 9.0), n_c=np.int64(n_c))
    return out


class Train(torch.utils.data.Dataset):
    def __init__(self, st, cfg, focal, donors, n, seed):
        self.st, self.cfg, self.focal, self.donors, self.n, self.seed = st, cfg, np.array(focal), donors, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        return draw(self.st, self.cfg, rng, int(rng.choice(self.focal)), self.donors)


class Trials(torch.utils.data.Dataset):
    """A fixed set of examples on held-out genomes. For every genome and repeat, a native query and a foreign
    one under identical settings: donor class, rows hidden around the genome and around the donor (a full
    3 x 3 x 3 grid), number of genes from elsewhere among the shown ones. Every arm sees the same trials."""

    def __init__(self, st, cfg, focal, donors, per_genome, seed):
        self.st, self.cfg, self.donors, self.seed = st, cfg, donors, seed
        lv = (0, 4, 16)
        self.spec = [(int(g), lab, r % 3, lv[(r // 3) % 3], lv[(r // 9) % 3], (0, 0, 1, 3)[r % 4])
                     for g in focal for r in range(per_genome) for lab in (0, 1)]

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, i):
        g, lab, kind, k_g, k_t, n_c = self.spec[i]
        rng = np.random.default_rng([self.seed, i // 2])      # the native and the foreign trial of a pair share the draw of the donor
        return draw(self.st, self.cfg, rng, g, self.donors, lab, kind, k_g, k_t, n_c)


class Genome(torch.utils.data.Dataset):
    """Every gene of one genome as the query, with random other genes of the same genome, a few times over."""

    def __init__(self, st, cfg, g, extra_hidden, repeats, seed):
        self.st, self.cfg, self.g, self.extra, self.repeats, self.seed = st, cfg, g, extra_hidden, repeats, seed
        self.genes = st.genes_of[g]

    def __len__(self):
        return len(self.genes) * self.repeats

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        x = int(self.genes[i % len(self.genes)])
        others = self.genes[self.genes != x]
        ctx = rng.choice(others, self.cfg.M, replace=False)
        hidden, _, _ = hidden_mask(self.st, self.cfg, rng, self.g, None, 0, 0, self.extra)
        out = example(self.st, self.cfg, rng, self.g, x, ctx, hidden, None)
        out.update(x=np.int64(x))
        return out


# ---------------------------------------------------------------- model

def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class Relate(nn.Module):
    """Relation between every two rows of a gene's alignment (row 0 is the gene itself): [B, G, R1, R1, k]."""

    def __init__(self, mode, k, profile, K=12, db=4, site_dim=48):
        super().__init__()
        self.mode, self.k, self.K, self.db, self.profile = mode, k, K, db, profile
        self.register_buffer("blosum", blosum21())
        if mode == "num":
            self.out = mlp(6, 64, k)
        elif mode in ("msa", "plm"):
            self.emb = nn.Embedding(24, 24)
            self.site = mlp(22 + 24 + (site_dim if mode == "plm" else 0), 64, 24)
            self.tok = mlp(48, 64, 64)
            self.left, self.right = nn.Linear(64, K * db), nn.Linear(64, K * db)
            self.out = mlp(6 + 2 * K, 64, k)
            with torch.no_grad():
                self.out[0].weight[:, 6:].zero_()            # starts as the "num" arm; the learned channels grow in

    def numbers(self, msa, res, rf, co, npos):
        B, G, R1, P = msa.shape
        oh = F.one_hot(msa.clamp(max=OTHER), 21).float() * res.unsqueeze(-1)
        a = oh[..., :20].reshape(B * G, R1, P * 20)
        match = torch.matmul(a, a.transpose(1, 2)).view(B, G, R1, R1)
        bl = torch.matmul(torch.matmul(oh, self.blosum).view(B * G, R1, P * 21),
                          oh.view(B * G, R1, P * 21).transpose(1, 2)).view(B, G, R1, R1)
        gf = (msa == GAP).float().view(B * G, R1, P)
        eg = torch.matmul(rf, gf.transpose(1, 2)).view(B, G, R1, R1)
        eg = eg + eg.transpose(-1, -2)                               # one has a residue where the other has a gap
        c1 = co.clamp(min=1)
        return torch.stack([match / c1, co / npos, bl / c1 / 4, eg / (co + eg).clamp(min=1), (co > 0).float(),
                            torch.log1p(co) / 6], -1)

    def forward(self, msa, site=None):
        msa = msa.long()
        B, G, R1, P = msa.shape
        if self.mode == "none":
            return torch.zeros(B, G, R1, R1, self.k, device=msa.device)
        res = msa <= OTHER                                           # an aligned residue
        npos = (msa[:, :, 0] != PAD).sum(-1).clamp(min=1).float()[:, :, None, None]
        rf = res.float().view(B * G, R1, P)
        co = torch.matmul(rf, rf.transpose(1, 2)).view(B, G, R1, R1)    # positions where both are aligned
        f = self.numbers(msa, res, rf, co, npos)
        if self.mode == "num":
            return self.out(f)
        seen = msa <= GAP                                            # residue or gap inside the alignment
        sf = seen.float().view(B * G, R1, P)
        coseen = torch.matmul(sf, sf.transpose(1, 2)).view(B, G, R1, R1).clamp(min=1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=msa.is_cuda):
            e = self.emb(msa)
            if self.profile:                                         # what the column looks like across the rows
                oh = F.one_hot(msa.clamp(max=GAP), 22).float() * seen.unsqueeze(-1)
                prof = oh.sum(2) / seen.sum(2).clamp(min=1).unsqueeze(-1)
            else:                                                    # gene-versus-member only: no view across members
                prof = torch.zeros(B, G, P, 22, device=msa.device)
            parts = [prof, e[:, :, 0]] + ([site.float()] if self.mode == "plm" else [])
            ctx = self.site(torch.cat(parts, -1))                    # [B, G, P, 24]
            u = self.tok(torch.cat([e, ctx.unsqueeze(2).expand(-1, -1, R1, -1, -1)], -1))
            m = seen.unsqueeze(-1)
            l = (self.left(u) * m).view(B * G, R1, P, self.K, self.db).permute(0, 3, 1, 2, 4).reshape(B * G, self.K, R1, P * self.db)
            r = (self.right(u) * m).view(B * G, R1, P, self.K, self.db).permute(0, 3, 1, 2, 4).reshape(B * G, self.K, R1, P * self.db)
        pair = torch.matmul(l.float(), r.float().transpose(-1, -2))  # [BG, K, R1, R1], summed in full precision
        pair = (0.5 * (pair + pair.transpose(-1, -2))).permute(0, 2, 3, 1).reshape(B, G, R1, R1, self.K)
        return self.out(torch.cat([f, pair / npos.unsqueeze(-1), pair / coseen.unsqueeze(-1)], -1))


class Block(nn.Module):
    def __init__(self, d, heads, k, dropout, pairs):
        super().__init__()
        self.h, self.pairs = heads, pairs
        self.n1, self.qkv, self.o = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        if pairs:
            self.pb, self.zb, self.flag = nn.Linear(k, heads), nn.Linear(k, heads), nn.Embedding(4, heads)
        self.n2 = nn.LayerNorm(d)
        self.row = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pair, both, zbar, zok):
        B, G, R, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, G, R, 3, self.h, d // self.h).permute(3, 0, 1, 4, 2, 5)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d // self.h)          # along the rows of one gene
        if self.pairs:
            bias = self.pb(pair) * both.unsqueeze(-1) + (self.zb(zbar) * zok.unsqueeze(-1)).unsqueeze(1)
            bias = bias + self.flag(both.long() + 2 * zok.unsqueeze(1).expand(-1, G, -1, -1).long())
            att = att + bias.permute(0, 1, 4, 2, 3)
        x = x + self.drop(self.o(torch.matmul(att.softmax(-1), v).transpose(2, 3).reshape(B, G, R, d)))
        y = self.n2(x).transpose(1, 2).reshape(B * R, G, d)                           # along the genes at one row
        x = x + self.drop(self.row(y, y, y, need_weights=False)[0].view(B, R, G, d).transpose(1, 2))
        return x + self.drop(self.ff(self.n3(x)))


class ArrayNet(nn.Module):
    def __init__(self, mode="num", pairs=True, comp=False, d=128, layers=4, heads=4, k=16, dropout=0.1):
        super().__init__()
        self.cfg = dict(mode=mode, pairs=pairs, comp=comp, d=d, layers=layers, heads=heads, k=k, dropout=dropout)
        self.pairs, self.comp = pairs, comp
        self.rel = Relate(mode, k, profile=pairs)
        self.cell, self.absent, self.role = nn.Linear(k, d), nn.Parameter(torch.zeros(d)), nn.Embedding(2, d)
        if comp:
            self.comp_in = nn.Linear(20, d)
        self.blocks = nn.ModuleList(Block(d, heads, k, dropout, pairs) for _ in range(layers))
        self.read = mlp(4 * d, 2 * d, d)
        self.out, self.ptr = nn.Linear(2 * d, 1), nn.Linear(d, 1)

    def forward(self, msa, present, site=None):
        B, G, R1, P = msa.shape
        rel = self.rel(msa, site)
        role = torch.zeros(G, dtype=torch.long, device=msa.device)
        role[1:] = 1
        x = torch.where(present.unsqueeze(-1), self.cell(rel[:, :, 0, 1:]), self.absent.expand(B, G, R1 - 1, -1))
        x = x + self.role(role)[None, :, None, :]
        if self.comp:                                                # each gene's own amino-acid composition
            own = msa[:, :, 0].long()
            oh = F.one_hot(own.clamp(max=OTHER), 21).float()[..., :20] * (own < OTHER).unsqueeze(-1)
            x = x + self.comp_in(oh.sum(2) / oh.sum((2, 3)).clamp(min=1).unsqueeze(-1)).unsqueeze(2)
        pair = both = zbar = zok = None
        if self.pairs:
            both = present[:, :, :, None] & present[:, :, None, :]
            pair = rel[:, :, 1:, 1:] * both.unsqueeze(-1)
            n = both[:, 1:].float().sum(1)                            # over the genome's other genes
            zbar, zok = pair[:, 1:].sum(1) / n.clamp(min=1).unsqueeze(-1), n > 0
        for blk in self.blocks:
            x = blk(x, pair, both, zbar, zok)
        q, c = x[:, 0], x[:, 1:].mean(1)
        h = self.read(torch.cat([q, c, q * c, (q - c).abs()], -1))
        return self.out(torch.cat([h.mean(1), h.amax(1)], -1)).squeeze(-1), self.ptr(h).squeeze(-1)
