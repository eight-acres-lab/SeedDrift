#!/usr/bin/env python
"""The wide table (third version, experimental): many genes of one genome against many reference proteomes,
and every gene is judged.

What changes against model.py / fit.py (the second version):
  - an example is N genes of one genome (64), some of them replaced by genes of other genomes, and the model
    says for every gene whether it belongs to the genome. Every gene carries a label, not one query per example.
  - the genome is shown through many of its genes at once, so what is normal for this genome at a given row is
    read off the table itself rather than guessed from 12 genes.
  - a cell holds the alignment summary of the gene against its best member in that proteome, as the search
    reports it. Nothing is aligned again, so every reference proteome can be a row (--universe all) and not
    only the panel of the second version (--universe panel).
Unchanged: no taxonomy in the input, rows are anonymous, the genome itself is hidden, rows near the genome and
near the donors are hidden at random, and the outputs are a probability per gene and a pointer to the nearest
row of the gene's origin.

Scoring uses the saved trials of the second version (same query, genome, donor and hiding), with the query as
one of the N genes, so both versions are compared on the same queries. The other genes and the rows are drawn
anew (the saved table does not hold them).

Controls: --show 13 builds the same 64-gene example, rows included, and shows the network only 13 of the gene
columns; --blank 1 leaves the cells without numbers (only whether they exist).

usage (GPU box, repo root):
  python research/array/wide.py --data DIR --names a b --profiles DIR/profiles.npz --library LIB \
      --trials DIR/run2/num_anchor --out DIR/run3/wide_all --universe all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit import auroc, splits, trial_stats  # noqa: E402
from model import Donors  # noqa: E402

NF = 5               # numbers per cell


class Cols:
    """Best member of every gene in every reference proteome (build.py columns) and the genome profiles."""

    def __init__(self, data_dir, names, profiles):
        prof = np.load(profiles)
        self.D, self.own = prof["D"], prof["own"]
        self.genomes = [str(x) for x in prof["genome"]]
        gix = {n: i for i, n in enumerate(self.genomes)}
        self.n_prot = self.D.shape[1]
        self.panel = np.zeros(self.n_prot, dtype=bool)
        self.panel[prof["panel"]] = True
        L, genome, gid, off, P, best, feat, base = [], [], [], [], [], [], [], 0
        for name in names:
            z = np.load(os.path.join(data_dir, "cols_%s.npz" % name))
            o, p, glen = z["off"].astype(np.int64), z["p"].astype(np.int64), z["gene_len"].astype(np.float32)
            gene_of = np.repeat(np.arange(len(glen)), np.diff(o))
            ln = glen[gene_of]
            s = np.clip(z["bits"] / (2.0 * ln), 0.0, 1.25)                       # alignment score per gene position, 1 for the gene against itself
            f = np.stack([s, -np.log(np.clip(s, 0.005, 1.0)) / 5.3, z["pid"] / 100.0, np.clip((z["qe"] - z["qs"] + 1) / ln, 0, 1),
                          np.clip(np.log(np.maximum(z["tlen"], 1) / ln), -1.5, 1.5) / 1.5], 1).astype(np.float16)
            order = np.argsort(gene_of * self.n_prot + p, kind="stable")          # within a gene, by proteome
            P.append(p[order].astype(np.int32)); feat.append(f[order]); best.append(p.astype(np.int32))
            L.append(glen); gid.append(z["gene_id"]); off.append(o[:-1] + base)
            genome.append(np.array([gix[str(x)] for x in z["gene_genome"]], dtype=np.int64))
            base += len(p)
        self.P, self.F, self.best = np.concatenate(P), np.concatenate(feat), np.concatenate(best)
        self.off = np.r_[np.concatenate(off), base]
        self.L, self.genome, self.gene_id = np.concatenate(L), np.concatenate(genome), np.concatenate(gid)
        order = np.argsort(self.genome, kind="stable")
        b = np.flatnonzero(np.r_[True, self.genome[order][1:] != self.genome[order][:-1], True])
        self.genes_of = {int(self.genome[order[s]]): order[s:e] for s, e in zip(b[:-1], b[1:])}
        self._near = {}

    def near(self, genome, panel):
        """Proteomes the genome has a measured distance to, nearest first (its own proteome left out)."""
        key = (genome, panel)
        if key not in self._near:
            D = self.D[genome]
            ok = np.isfinite(D)
            if panel:
                ok &= self.panel
            c = np.flatnonzero(ok)
            c = c[c != self.own[genome]]
            self._near[key] = c[np.argsort(D[c], kind="stable")]
        return self._near[key]


@dataclass
class Cfg:
    N: int = 64                   # genes per example
    R_near: int = 32              # rows nearest to the genome
    R_strat: int = 32             # rows spread over the rest of the genome's distance range
    T: int = 2                    # the T best visible rows of every gene are always in the table
    show: int = 0                 # control: gene columns shown to the network (0 = all N); the rows still come from all N genes
    blank: bool = False           # control: cells carry no numbers, only whether they exist
    panel: bool = False           # rows only from the panel of the second version
    hide: tuple = (0, 0, 0, 1, 2, 4, 8, 16, 32)       # radius of what is hidden, in panel proteomes, around the genome and around a donor
    repl: tuple = (0, 0, 1, 1, 2, 3, 4, 6, 8, 12, 16, 20)  # how many of 64 genes are replaced (scaled to the number of genes)

    @property
    def R(self):
        return self.R_near + self.R_strat + self.T * self.N

    @property
    def G(self):
        return self.show or self.N


def hide(st, h, genome, k):
    """Hide the genome's own proteome and everything as near to it as its k nearest panel proteomes."""
    if st.own[genome] >= 0:
        h[st.own[genome]] = True
    if k:
        order = st.near(genome, True)
        if len(order):
            with np.errstate(invalid="ignore"):
                h |= st.D[genome] <= st.D[genome][order[min(k, len(order)) - 1]]


def choose_rows(st, cfg, rng, g, genes, hidden):
    avail = ~hidden
    if cfg.panel:
        avail &= st.panel
    order = st.near(g, cfg.panel)
    order = order[avail[order]]
    chosen = list(order[:cfg.R_near])
    n_near = len(chosen)
    rest = order[cfg.R_near:]
    if len(rest):
        edges = np.linspace(0, len(rest), cfg.R_strat + 1).astype(int)
        chosen += [rest[rng.integers(a, b)] for a, b in zip(edges[:-1], edges[1:]) if b > a]
    taken = np.zeros(st.n_prot, dtype=bool)
    taken[chosen] = True
    extra = []
    for gene in genes:                                        # the T best visible rows of every gene (many are in the table already)
        t = st.best[st.off[gene]:st.off[gene + 1]]
        t = t[avail[t]][:cfg.T]
        t = t[~taken[t]]
        taken[t] = True
        extra += list(t)
    rows = np.array(chosen + extra, dtype=np.int64)
    near = np.arange(len(rows)) < n_near
    perm = rng.permutation(len(rows))
    return rows[perm], near[perm]


def table(st, cfg, rng, g, genes, hidden, origin, keep=None):
    """Arrays of one example. origin[i] is the genome gene i really comes from. The rows are chosen by all the
    genes; with keep, only those genes become columns (the control that shows fewer genes)."""
    rows, near = choose_rows(st, cfg, rng, g, genes, hidden)
    if keep is not None:
        genes, origin = genes[keep], origin[keep]
    N, R, n, r = cfg.G, cfg.R, len(genes), len(rows)
    x = np.zeros((N, R, NF), dtype=np.float16)
    present = np.zeros((N, R), dtype=bool)
    ptr = np.full(N, -1, dtype=np.int64)
    for i, gene in enumerate(genes):
        a, b = st.off[gene], st.off[gene + 1]
        if a == b:
            continue
        ps = st.P[a:b]
        k = np.searchsorted(ps, rows)
        k[k == len(ps)] = 0
        hit = ps[k] == rows
        x[i, :r][hit] = st.F[a + k[hit]]
        present[i, :r] = hit
        if hit.any():
            d = np.where(hit, st.D[origin[i]][rows], np.nan)
            if np.isfinite(d).any():
                ptr[i] = int(np.nanargmin(d))
    row_ok, gene_ok, nr = np.zeros(R, dtype=bool), np.zeros(N, dtype=bool), np.zeros(R, dtype=bool)
    row_ok[:r], gene_ok[:n], nr[:r] = True, True, near
    s = x[..., 0].astype(np.float32)
    rule = np.where(present & ~nr, s, 0).max(1) - np.where(present & nr, s, 0).max(1)     # best outside the genome's nearest rows minus best inside
    ptr_rule = np.where(present.any(1), np.where(present, s, -1.0).argmax(1), -2).astype(np.int64)      # the most similar row
    if cfg.blank:
        x[:] = 0
    return dict(x=x, present=present, row_ok=row_ok, gene_ok=gene_ok, ptr=ptr, ptr_rule=ptr_rule,
                rule_ai=rule.astype(np.float32), n_present=present.sum(1).astype(np.int64))


def other_gene(st, rng, src, used):
    """A gene of genome src that is not in the example yet."""
    pool = st.genes_of[src]
    c = int(rng.choice(pool))
    for _ in range(8):
        if c not in used:
            break
        c = int(rng.choice(pool))
    used.add(c)
    return c


def draw(st, cfg, rng, g, donors):
    """A training example: genes of genome g, some replaced. Replaced genes come from one to three donors whose
    surroundings are hidden to a random radius, and, in half of the examples, also singly from other genomes
    of which only the own proteome is hidden (the scheme of the saved trials). The donors' surroundings are
    hidden whether or not any gene is replaced, so what is hidden says nothing about the labels."""
    mine = st.genes_of[g]
    n = min(cfg.N, len(mine))
    genes = mine[rng.permutation(len(mine))[:n]].copy()
    origin = np.full(n, g, dtype=np.int64)
    label, dist = np.zeros(n, dtype=np.float32), np.full(n, -1.0, dtype=np.float32)
    hidden = np.zeros(st.n_prot, dtype=bool)
    hide(st, hidden, g, int(rng.choice(cfg.hide)))
    main = [donors.pick(rng, g, int(rng.integers(3))) for _ in range(int(rng.choice((1, 1, 2, 3))))]
    for t, _ in main:
        hide(st, hidden, t, int(rng.choice(cfg.hide)))
    r = int(rng.choice(cfg.repl)) * n / 64.0
    n_f = min(int(r) + int(rng.random() < r - int(r)), n // 3)          # the same share of the genes whatever their number
    p_single, used = float(rng.choice((0.0, 0.0, 0.5, 0.5))), set()
    for i in rng.choice(n, n_f, replace=False):
        if rng.random() < p_single:
            t, d = donors.pick(rng, g, int(rng.integers(3)))
            hide(st, hidden, t, 0)
        else:
            t, d = main[int(rng.integers(len(main)))]
        genes[i], origin[i], label[i], dist[i] = other_gene(st, rng, t, used), t, 1.0, (d if np.isfinite(d) else 9.0)
    keep = np.sort(rng.choice(n, min(cfg.show, n), replace=False)) if cfg.show else None
    out = table(st, cfg, rng, g, genes, hidden, origin, keep)
    lab, dg = np.zeros(cfg.G, dtype=np.float32), np.full(cfg.G, -1.0, dtype=np.float32)
    k = len(label) if keep is None else len(keep)
    lab[:k], dg[:k] = (label, dist) if keep is None else (label[keep], dist[keep])
    out.update(label=lab, d_gt=dg)
    return out


class Train(torch.utils.data.Dataset):
    def __init__(self, st, cfg, focal, donors, n, seed):
        self.st, self.cfg, self.focal, self.donors, self.n, self.seed = st, cfg, np.array(focal), donors, n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        return draw(self.st, self.cfg, rng, int(rng.choice(self.focal)), self.donors)


class Compat(torch.utils.data.Dataset):
    """The trials of the second version (its saved table): same query, genome, donor and hiding. The query is
    gene 0; the other genes are the genome's, with the same share of them from elsewhere as in the trial."""

    def __init__(self, st, cfg, tab, donors, seed):
        self.st, self.cfg, self.t, self.donors, self.seed = st, cfg, tab, donors, seed

    def __len__(self):
        return len(self.t["label"])

    def __getitem__(self, i):
        st, cfg, t = self.st, self.cfg, self.t
        g, x, d, lab = int(t["g"][i]), int(t["x"][i]), int(t["donor"][i]), int(t["label"][i] > 0.5)
        x0 = int(t["x"][i - i % 2])                            # the native partner's query: left out of the other genes in both trials
        rng = np.random.default_rng([self.seed, i // 2])       # so a native trial and its foreign partner share the other genes
        mine = st.genes_of[g]
        perm = mine[rng.permutation(len(mine))]
        others = perm[perm != x0][:cfg.N - 1]
        genes = np.r_[x, others].astype(np.int64)
        origin = np.r_[d if lab else g, np.full(len(others), g)].astype(np.int64)
        hidden = np.zeros(st.n_prot, dtype=bool)
        hide(st, hidden, g, int(t["k_g"][i]))
        hide(st, hidden, d, int(t["k_t"][i]))
        m, used = int(round(int(t["n_c"][i]) * len(others) / 12.0)), {x, x0}
        for j in 1 + rng.choice(len(others), m, replace=False):
            if rng.random() < 0.5:
                src = d
            else:                                              # a gene of some other genome; its own proteome is hidden, as in training
                src = self.donors.pick(rng, g, int(rng.integers(3)))[0]
                hide(st, hidden, src, 0)
            genes[j], origin[j] = other_gene(st, rng, src, used), src
        keep = None
        if cfg.show:
            keep = np.r_[0, 1 + np.sort(rng.choice(len(others), min(cfg.show - 1, len(others)), replace=False))]
        out = table(st, cfg, rng, g, genes, hidden, origin, keep)
        out.update(i=np.int64(i))
        return out


class Whole(torch.utils.data.Dataset):
    """Every gene of one genome, in random groups of N, a few times over."""

    def __init__(self, st, cfg, g, extra_hidden, repeats, seed):
        self.st, self.cfg, self.g, self.extra, self.seed = st, cfg, g, extra_hidden, seed
        genes, rng, self.groups = st.genes_of[g], np.random.default_rng(seed), []
        for _ in range(repeats):
            perm = genes[rng.permutation(len(genes))]
            for a in range(0, len(perm), cfg.N):
                c = perm[a:a + cfg.N]
                if len(c) < cfg.N and len(genes) >= cfg.N:
                    c = np.r_[c, rng.choice(np.setdiff1d(genes, c), cfg.N - len(c), replace=False)]
                self.groups.append(c.astype(np.int64))

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        genes = self.groups[i]
        hidden = self.extra.copy()
        hide(self.st, hidden, self.g, 0)
        out = table(self.st, self.cfg, rng, self.g, genes, hidden, np.full(len(genes), self.g, dtype=np.int64))
        gene = np.full(self.cfg.N, -1, dtype=np.int64)
        gene[:len(genes)] = genes
        out.update(gene=gene)
        return out


class Attention(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.qkv, self.o, self.p = heads, nn.Linear(d, 3 * d), nn.Linear(d, d), dropout

    def forward(self, x, key_ok):                              # x [S, T, d], key_ok [S, T]
        S, T, d = x.shape
        q, k, v = self.qkv(x).view(S, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=key_ok[:, None, None, :], dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(S, T, d))


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.rows, self.genes = Attention(d, heads, dropout), Attention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, row_ok, gene_ok):                     # x [B, N, R, d]
        B, N, R, d = x.shape
        y = self.rows(self.n1(x).reshape(B * N, R, d), row_ok[:, None, :].expand(B, N, R).reshape(B * N, R))     # along the rows of one gene
        x = x + self.drop(y.view(B, N, R, d))
        y = self.genes(self.n2(x).transpose(1, 2).reshape(B * R, N, d), gene_ok[:, None, :].expand(B, R, N).reshape(B * R, N))   # along the genes at one row
        x = x + self.drop(y.view(B, R, N, d).transpose(1, 2))
        return x + self.drop(self.ff(self.n3(x)))


class WideNet(nn.Module):
    def __init__(self, d=128, layers=6, heads=4, dropout=0.1):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, dropout=dropout)
        self.cell = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
        self.absent, self.summary = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.ptr = nn.Linear(d, 1)

    def forward(self, x, present, row_ok, gene_ok):
        B, N, R, _ = x.shape
        c = torch.where(present.unsqueeze(-1), self.cell(x), self.absent.expand(B, N, R, -1))
        s = self.summary.expand(B, N, 1, -1)                                  # one summary position per gene, ahead of its rows
        h = torch.cat([s, c.to(s.dtype) if c.dtype != s.dtype else c], 2)
        ok = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=x.device), row_ok], 1)
        for blk in self.blocks:
            h = blk(h, ok, gene_ok)
        with torch.autocast("cuda", enabled=False):                          # the two outputs in full precision
            h = self.norm(h.float())
            s = h[:, :, 0]
            w = gene_ok.unsqueeze(-1).float()
            mean = (s * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)
            return self.out(torch.cat([s, mean.expand(-1, N, -1)], -1)).squeeze(-1), self.ptr(h[:, :, 1:]).squeeze(-1)


def trim(b):
    """Drop the padded rows that no example of the batch uses (used rows come first in every example)."""
    r = int(b["row_ok"].any(0).nonzero().max()) + 1
    b["x"], b["present"], b["row_ok"] = b["x"][:, :, :r], b["present"][:, :, :r], b["row_ok"][:, :r]
    return b


def forward(model, b, dev):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        logit, ptr = model(b["x"].to(dev).float(), b["present"].to(dev), b["row_ok"].to(dev), b["gene_ok"].to(dev))
    return logit.float(), ptr.float()


def run_compat(model, ds, kw, dev):
    """Score of the query (gene 0) for every trial, in the order of the saved table."""
    model.eval()
    n = len(ds)
    out = {k: np.zeros(n, dtype=np.float32) for k in ("logit", "rule_ai")}
    for k in ("n_present", "ptr", "ptr_rule"):
        out[k] = np.zeros(n, dtype=np.int64)
    ok = np.zeros(n, dtype=bool)
    with torch.no_grad():
        for b in torch.utils.data.DataLoader(ds, **kw):
            logit, ptr = forward(model, trim(b), dev)
            i = b["i"].numpy()
            out["logit"][i] = logit[:, 0].cpu().numpy()
            for k in ("rule_ai", "n_present", "ptr", "ptr_rule"):
                out[k][i] = b[k][:, 0].numpy()
            p = ptr[:, 0].masked_fill(~b["present"][:, 0].to(dev), -1e9).argmax(1).cpu()
            ok[i] = (p == b["ptr"][:, 0]).numpy()
    out["ptr_ok"] = ok
    return out


def score_genomes(model, st, cfg, bench, lib, out, repeats, kw, dev, hidden=None):
    """Scores of every gene of whole genomes (name -> job file that gives the genome's genus): scores_<name>.tsv in out.
    hidden: name -> table with a taxid column; these reference proteomes are hidden instead of the own genus alone."""
    for name, job in bench.items():
        g = st.genomes.index(name)
        genus = int(np.load(job)["genome_ranks"][5])
        if not genus:
            sys.exit("%s: the job gives no genus, so nothing of the genome's own lineage would be hidden" % name)
        extra = lib["ranks"][:, 5] == genus
        if hidden and name in hidden:
            taxa = {int(line.split("\t")[0]) for line in open(hidden[name]).read().splitlines()[1:] if line.strip()}
            extra = extra | np.isin(lib["taxid"], list(taxa))
        score, seen = {}, {}
        model.eval()
        with torch.no_grad():
            for b in torch.utils.data.DataLoader(Whole(st, cfg, g, extra, repeats, 7), **kw):
                logit, _ = forward(model, trim(b), dev)
                for gene, lg, ru, n in zip(b["gene"].numpy().ravel(), logit.cpu().numpy().ravel(), b["rule_ai"].numpy().ravel(), b["n_present"].numpy().ravel()):
                    if gene >= 0:
                        score.setdefault(int(gene), []).append((float(lg), float(ru)))
                        seen[int(gene)] = seen.get(int(gene), 0) + int(n)
        n_out = 0
        with open(os.path.join(out, "scores_%s.tsv" % name), "w") as fh:
            fh.write("gene_id\twide\twide_rule_ai\n")
            for x, s in score.items():
                if seen[x] == 0:                              # no member of the gene in any row shown: no evidence, ranks last like everywhere else
                    n_out += 1
                    continue
                fh.write("%s\t%.4f\t%.4f\n" % ((st.gene_id[x],) + tuple(np.mean(s, axis=0))))
        print("scored %s: %d genes, %d more without any visible member (rank last); own genus hidden: %d proteomes" % (
            name, len(score) - n_out, n_out, int(extra.sum())), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--names", nargs="+", default=[])
    ap.add_argument("--profiles", required=True); ap.add_argument("--library", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--trials", default="", help="a run directory of the second version (trials_validation.npz, trials_unseen.npz)")
    ap.add_argument("--universe", default="all", choices=("all", "panel")); ap.add_argument("--genes", type=int, default=64)
    ap.add_argument("--show", type=int, default=0, help="control: show the network only this many of the genes (rows still come from all)")
    ap.add_argument("--blank", type=int, default=0, help="control: cells without numbers")
    ap.add_argument("--top", type=int, default=2, help="how many of its best visible rows every gene brings into the table")
    ap.add_argument("--train-frac", type=float, default=1.0, help="train on this share of the training genomes (learning curve)")
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--bench", nargs="*", default=[], help="name=job.npz (the job file gives the genome's own genus)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rescore", default="", help="a second-version run directory holding trials_fresh.npz: score the saved model of --out "
                    "on those trials, write trials_fresh.npz, and stop")
    ap.add_argument("--hidden", nargs="*", default=[], help="with --score: name=table with a taxid column (bench/hard/views.py --hidden-out); "
                    "these reference proteomes are hidden for that genome, on top of its own genus")
    ap.add_argument("--score", default="", help="a run directory of this script: score the genomes of --bench with its saved model, "
                    "write scores_<name>.tsv to --out, and stop (needs only the columns and profiles of those genomes)")
    a = ap.parse_args()
    if a.score:                                               # whole genomes with a saved model: no training data, no trials
        old = json.load(open(os.path.join(a.score, "result.json")))["args"]
        if old["show"]:
            sys.exit("the control that shows fewer genes is not scored on whole genomes")
        os.makedirs(a.out, exist_ok=True)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        bench = dict(x.split("=", 1) for x in a.bench)
        st = Cols(a.data, list(bench), a.profiles)
        cfg = Cfg(N=old["genes"], T=old.get("top", 2), blank=bool(old["blank"]), panel=old["universe"] == "panel")
        ck = torch.load(os.path.join(a.score, "model.pt"), weights_only=False, map_location=dev)
        model = WideNet(**ck["cfg"]).to(dev)
        model.load_state_dict(ck["state"])
        score_genomes(model, st, cfg, bench, np.load(a.library), a.out, a.repeats,
                      dict(batch_size=a.batch, num_workers=a.workers, persistent_workers=False), dev)
        return
    if not a.names or not a.trials:
        sys.exit("--names and --trials are needed for training and for --rescore")
    if a.rescore:                                             # the settings of the saved run, not of this command line
        old = json.load(open(os.path.join(a.out, "result.json")))["args"]
        a.universe, a.genes, a.show, a.blank, a.bench = old["universe"], old["genes"], old["show"], old["blank"], []
        a.top = old.get("top", 2)
    if a.show and a.bench:
        sys.exit("the control that shows fewer genes is not scored on whole genomes")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bench = dict(x.split("=", 1) for x in a.bench)
    st = Cols(a.data, a.names + list(bench), a.profiles)
    cfg = Cfg(N=a.genes, T=a.top, show=a.show, blank=bool(a.blank), panel=a.universe == "panel")
    sp = splits(st, a.library)
    enough = {g for g, genes in st.genes_of.items() if len(genes) > 13}        # the same genomes as in the second version
    grp = {k: sorted(g for g, s in sp.items() if s == k and g in enough) for k in range(4)}
    print("genomes: train %d, validation %d, unseen classes %d, benchmark lineages %d (kept out)" % tuple(len(grp[k]) for k in range(4)), flush=True)
    tabs = {w: dict(np.load(os.path.join(a.trials, "trials_%s.npz" % w))) for w in ("validation", "unseen")}
    if a.rescore:
        tabs = {"fresh": dict(np.load(os.path.join(a.rescore, "trials_fresh.npz")))}
    for w, tab in tabs.items():                               # the saved gene indices must mean the same genes here
        if not np.array_equal(st.genome[tab["x"]], np.where(tab["label"] > 0.5, tab["donor"], tab["g"])):
            sys.exit("%s trials: gene indices do not match this store (other --names or order than the run that saved them)" % w)
        for k in ("ptr", "ptr_ok", "ptr_rule", "logit"):      # fields of the run that saved the table; this run writes its own
            tab.pop(k, None)
    kw = dict(batch_size=a.batch, num_workers=a.workers, persistent_workers=False)
    if a.rescore:
        ck = torch.load(os.path.join(a.out, "model.pt"), weights_only=False)
        model = WideNet(**ck["cfg"]).to(dev)
        model.load_state_dict(ck["state"])
        tab = tabs["fresh"]
        r = run_compat(model, Compat(st, cfg, tab, Donors(st, grp[2]), 303), kw, dev)
        full = dict(tab); full.update(r)
        res = json.load(open(os.path.join(a.out, "result.json")))
        res["fresh"], res["fresh_rule_ai"] = trial_stats(full), trial_stats(full, "rule_ai")
        json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
        full = dict(tab)
        full.update(logit=r["logit"], wide_rule_ai=r["rule_ai"], wide_n_present=r["n_present"], wide_ptr_ok=r["ptr_ok"], wide_ptr=r["ptr"],
                    wide_ptr_rule=r["ptr_rule"])
        np.savez(os.path.join(a.out, "trials_fresh.npz"), **full)
        v = res["fresh"]
        print("fresh trials: auroc %.4f; near %.4f, mid %.4f, far %.4f" % (
            v["auroc"], v["donor_near"]["auroc"], v["donor_mid"]["auroc"], v["donor_far"]["auroc"]), flush=True)
        return
    val = Compat(st, cfg, tabs["validation"], Donors(st, grp[1]), 101)
    test = Compat(st, cfg, tabs["unseen"], Donors(st, grp[2]), 202)
    model = WideNet(a.d, a.layers).to(dev)
    print("parameters %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    train = grp[0]
    if a.train_frac < 1:                                      # the same subset whatever the seed
        train = sorted(np.random.default_rng(12345).permutation(train)[:int(round(a.train_frac * len(train)))].tolist())
        print("training on %d of %d training genomes" % (len(train), len(grp[0])), flush=True)
    data = Train(st, cfg, train, Donors(st, train), a.steps * a.batch, a.seed)
    best, t0, step, run_loss, log = -1.0, time.time(), 0, [], []

    def stats(ds, tab):
        r = run_compat(model, ds, kw, dev)
        full = dict(tab); full.update(r)                     # strata from the saved table, scores from this model
        return r, trial_stats(full), trial_stats(full, "rule_ai")

    model.train()
    for b in torch.utils.data.DataLoader(data, **kw):
        logit, ptr = forward(model, trim(b), dev)
        ok, y, tgt = b["gene_ok"].to(dev), b["label"].to(dev), b["ptr"].to(dev)
        loss = F.binary_cross_entropy_with_logits(logit[ok], y[ok])
        has = (tgt >= 0) & ok
        if has.any():
            pl = ptr.masked_fill(~b["present"].to(dev), -1e9)
            loss = loss + 0.3 * F.cross_entropy(pl[has], tgt[has])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss.append(float(loss)); step += 1
        if step % 200 == 0:
            print("step %d loss %.4f  %.1f steps/s" % (step, np.mean(run_loss[-200:]), step / (time.time() - t0)), flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            _, r, _ = stats(val, tabs["validation"])
            log.append({"step": step, "loss": float(np.mean(run_loss[-200:])), "val": r})
            print("  validation: auroc %.4f, near donors %.4f, mid %.4f, far %.4f" % (
                r["auroc"], r["donor_near"]["auroc"], r["donor_mid"]["auroc"], r["donor_far"]["auroc"]), flush=True)
            if r["auroc"] > best:                             # checkpoint by the validation AUROC over all trials
                best = r["auroc"]
                torch.save({"cfg": model.cfg, "state": model.state_dict(), "step": step}, os.path.join(a.out, "model.pt"))
            model.train()
    model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), weights_only=False)["state"])
    res = {"args": vars(a), "log": log}
    for name, ds in (("validation", val), ("unseen", test)):
        r, res[name], res[name + "_rule_ai"] = stats(ds, tabs[name])
        full = dict(tabs[name])
        full.update(logit=r["logit"], wide_rule_ai=r["rule_ai"], wide_n_present=r["n_present"], wide_ptr_ok=r["ptr_ok"], wide_ptr=r["ptr"],
                    wide_ptr_rule=r["ptr_rule"])
        np.savez(os.path.join(a.out, "trials_%s.npz" % name), **full)      # the saved table with this model's score in place of the old one
    r, v = res["unseen"], res["unseen_rule_ai"]
    print("unseen classes: auroc %.4f, recall at 5%% %.3f; near %.4f, mid %.4f, far %.4f; pointer %.3f" % (
        r["auroc"], r["recall_at_5"], r["donor_near"]["auroc"], r["donor_mid"]["auroc"], r["donor_far"]["auroc"],
        r.get("pointer", {}).get("model", float("nan"))), flush=True)
    print("  rule_ai on this model's rows: auroc %.4f; near %.4f, mid %.4f, far %.4f" % (
        v["auroc"], v["donor_near"]["auroc"], v["donor_mid"]["auroc"], v["donor_far"]["auroc"]), flush=True)
    score_genomes(model, st, cfg, bench, np.load(a.library), a.out, a.repeats, kw, dev)
    if dev == "cuda":
        res["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
        print("peak GPU memory %.1f GB" % res["gpu_gb"], flush=True)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
