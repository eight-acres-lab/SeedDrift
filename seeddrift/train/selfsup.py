"""Self-supervised source model v0 (truth-labels line): every training target is an observed fact of real genomes.

  A   membership. Which genome of the batch is this gene in? Contrastive over the genomes of a batch, half of them drawn
      as the nearest training relative of another batch genome (lineage only chooses these hard negatives). Target:
      the genome the gene is in. Foreign score: -log p(own genome) against a fixed panel of training genomes.
  B1  codons from protein. The synonymous codon at every position, given the gene's amino-acid sequence, once with the
      genome's representation (aware) and once without it (agnostic twin, same inputs otherwise). Foreign score: minus
      the mean log-likelihood ratio aware / agnostic per codon that has a synonym.
  B2  hit residues from the query. The aligned target residue at every query position of every visible hit, given the
      query's amino acids and the hit's proteome, aware and agnostic. Foreign score: minus the mean log-likelihood
      ratio per aligned position (no score for a gene without visible hits).
  (C, homolog pairs inside the corpus, comes with the corpus-against-corpus search.)

Run by the train stage of the CI (pipeline selfsup) with the CUDA Python of the machine:

  python selfsup.py fit     RUN_DIR CONFIG_JSON   train, log, checkpoint (resumable), then score the held-out genomes
  python selfsup.py score   RUN_DIR CONFIG_JSON   score the held-out genomes with a saved model
  python selfsup.py genomes RUN_DIR CONFIG_JSON   score the genomes listed in CONFIG_JSON["genomes_index"] (islands)

Reads data/train_data and data/reference/library.npz through seeddrift.train.source_model (Data, Genome; its encoders
are reused through source_model.build, whose behaviour is not changed). Inputs are raw data only: codons, amino acids,
hit alignments residue by residue, the identity of a hit's proteome. No identity, coverage, score or lineage path
enters the network.

No neighbours: the genome representation is a robust summary (the genes farthest from the mean dropped) of many other
genes of the genome, never the window around the gene (a context that explains islands away).
Hiding: every batch removes, from every gene's hits and from every genome representation, each batch genome's own
proteome and its clade at one depth drawn per step (species, genus, family, order). The holes are the same for all
genomes of a batch, so they cannot tell the genomes apart. Scoring hides the scored genome's own proteome, its genus
(or the hidden proteomes given for it) and the panel genomes' proteomes and genera.
Loss: per objective the (1 - trim) lowest per-example losses of a batch: training genomes hold unlabelled transfers.
Training genomes only (split "train", train_fraction of them); held-out genomes only for diagnosis; genomes of the
sealed test split are never read here. Never imports an evaluation generator (gate G2 static).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from seeddrift.train import source_model as S  # noqa: E402
from seeddrift.train.logger import Diverged, TrainLogger, save_npz  # noqa: E402

RANKS, DEPTHS, CODE = S.RANKS, S.DEPTHS, S.CODE
OBJECTIVES = ("A", "B1", "B2")
SCORES = ("A", "B1", "B2", "comb")


def _synonyms():
    t = np.zeros((22, 66), dtype=bool)          # t[a, c]: codon token c (1..64) codes amino-acid token a (21 = stop)
    for c in range(1, 65):
        t[CODE[c], c] = True
    return t


SYN = _synonyms()
N_SYN = SYN.sum(1)


def objectives_of(cfg):
    obj = cfg.get("objectives")
    obj = list(OBJECTIVES) if obj is None else [obj] if isinstance(obj, str) else list(obj)
    bad = [o for o in obj if o not in OBJECTIVES]
    if bad or not obj or len(set(obj)) != len(obj):
        raise SystemExit("objectives: a non-empty set of %s, not %r" % (", ".join(OBJECTIVES), obj))
    return [o for o in OBJECTIVES if o in obj]


def clade(ranks, depth):
    """(column, value) of a genome's clade at `depth`, or the nearest defined rank below it; None without one."""
    for col in range(RANKS.index(depth), len(RANKS)):
        if ranks[col]:
            return col, int(ranks[col])
    return None


def hidden_rows(data, genomes, depth, taxids=()):
    """Library rows hidden for a set of genomes: their own proteomes (taxid), their clades at `depth`, and `taxids`."""
    m = np.isin(data.lib_taxid, sorted({int(g.meta["taxid"]) for g in genomes} | {int(t) for t in taxids}))
    by_col = {}
    for g in genomes:
        cv = clade(g.ranks, depth)
        if cv:
            by_col.setdefault(cv[0], set()).add(cv[1])
    for col, vals in sorted(by_col.items()):
        m |= np.isin(data.lib_ranks[:, col], sorted(vals))
    return m


def visible_prots(gen, j, hidden, n):
    rows = gen.hprot[gen.hoff[j]:gen.hoff[j + 1]]
    return rows[~hidden[rows]][:n].astype(np.int64)


def example(data, gen, i, hidden, rng, cfg):
    """Gene i of `gen` with the hidden library rows removed: its codons, its visible hits (pair tokens for the gene
    encoder, query start and target tokens for B2) and a source of other genes of its genome (codons and the
    proteomes their visible hits fall in) with the genome's codon frequencies without gene i."""
    c = gen.codons[gen.coff[i]:gen.coff[i + 1]][:data.maxc].astype(np.int64)
    q = CODE[c]
    tracks, prots, qss, tgts = [], [], [], []
    for h in range(gen.hoff[i], gen.hoff[i + 1]):
        if len(prots) >= data.kc:
            break
        pr = int(gen.hprot[h])
        if hidden[pr]:
            continue
        t = gen.track[gen.toff[h]:gen.toff[h + 1]].astype(np.int64)
        qs = int(gen.hqs[h])
        qq = q[qs:qs + len(t)] if qs < len(q) else np.zeros(0, np.int64)
        m = min(len(t), len(qq), data.maxt)
        if m == 0:
            continue
        tracks.append(qq[:m] * 23 + t[:m])
        prots.append(pr)
        qss.append(qs)
        tgts.append(t[:m])
    k, ls, hs = int(cfg.get("source_genes", 64)), int(cfg.get("source_codons", 512)), int(cfg.get("source_hits", 8))
    idx = [int(j) for j in rng.choice(gen.n, size=min(k + 1, gen.n), replace=False) if j != i][:k]
    src = [gen.codons[gen.coff[j]:gen.coff[j + 1]][:ls].astype(np.int64) for j in idx]
    src_prots = [visible_prots(gen, j, hidden, hs) for j in idx]
    return dict(c=c, tracks=tracks, prots=prots, qs=qss, tgts=tgts, src=src, src_prots=src_prots,
                gcomp=S.frequencies(gen.ccount - gen.gene_count(i)))


def collate(items, torch, n2=8):
    """Padded tensors of a list of examples. 0 is padding everywhere: codon, amino-acid, pair and proteome tokens are
    shifted where 0 is a real value. B2 reads the first n2 visible hits of each gene (the nearest ones)."""
    B = len(items)
    Lc = max([len(e["c"]) for e in items] + [1])
    H = max([len(e["tracks"]) for e in items] + [1])
    T = max([len(t) for e in items for t in e["tracks"]] + [1])
    N2 = max([min(len(e["tracks"]), n2) for e in items] + [1])
    K = max([len(e["src"]) for e in items] + [1])
    Ls = max([len(s) for e in items for s in e["src"]] + [1])
    Hs = max([len(p) for e in items for p in e["src_prots"]] + [1])
    codons = np.zeros((B, Lc), np.int64)
    tracks = np.zeros((B, H, T), np.int64)
    prots = np.zeros((B, H), np.int64)
    hmask = np.zeros((B, H), bool)
    b2_prot = np.zeros((B, N2), np.int64)
    b2_qs = np.zeros((B, N2), np.int64)
    b2_t = np.zeros((B, N2, T), np.int64)
    src = np.zeros((B, K, Ls), np.int64)
    src_prots = np.zeros((B, K, Hs), np.int64)
    gcomp = np.zeros((B, 66), np.float32)
    for b, e in enumerate(items):
        codons[b, :len(e["c"])] = e["c"]
        for h, (t, p) in enumerate(zip(e["tracks"], e["prots"])):
            tracks[b, h, :len(t)] = t + 1
            prots[b, h] = p + 1
            hmask[b, h] = True
            if h < N2:
                b2_prot[b, h] = p + 1
                b2_qs[b, h] = e["qs"][h]
                b2_t[b, h, :len(e["tgts"][h])] = e["tgts"][h]          # 1..22; 0 = padding
        for k, (s, p) in enumerate(zip(e["src"], e["src_prots"])):
            src[b, k, :len(s)] = s
            src_prots[b, k, :len(p)] = p + 1
        gcomp[b] = e["gcomp"]
    aa = np.where(codons > 0, CODE[codons], 0)
    out = dict(codons=codons, aa=aa, tracks=tracks, prots=prots, hmask=hmask, b2_prot=b2_prot, b2_qs=b2_qs, b2_t=b2_t,
               src=src, src_prots=src_prots, gcomp=gcomp)
    return {k: torch.from_numpy(v) for k, v in out.items()}


def build(torch, n_lib, d=128, trim_genes=0.2):
    """The network. The codon encoder, the hit-track encoder, the proteome embedding, the hit-set transformer and the
    composition projection are the source model's modules (source_model.build); the rest is new."""
    nn, F = torch.nn, torch.nn.functional
    base = S.build(torch, n_lib, d=d, comp=True)
    Conv = type(base.codon)

    def trimmed(v, m, trim):
        """The genes of each row whose encodings lie nearest the row's mean, (1 - trim) of the present ones."""
        with torch.no_grad():
            n = m.sum(1, keepdim=True)
            mean = (v * m.unsqueeze(-1)).sum(1) / n.clamp(min=1)
            dist = ((v - mean.unsqueeze(1)) ** 2).sum(-1).masked_fill(~m, float("inf"))
            rank = dist.argsort(1).argsort(1)
            keep = torch.ceil(n.float() * (1 - trim)).long().clamp(min=1)
            return m & (rank < keep)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.codon, self.hit, self.prot = base.codon, base.hit, base.prot
            self.hit_proj, self.hit_set, self.hit_pool, self.no_hit = base.hit_proj, base.hit_set, base.hit_pool, base.no_hit
            self.comp_proj = base.comp_proj
            self.aa = Conv(22, d)                                        # read per position; its pooling is not used
            self.genome = nn.Sequential(nn.Linear(2 * d + 32, d), nn.GELU(), nn.Linear(d, d))
            self.gene = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
            self.qa, self.ka = nn.Linear(d, d), nn.Linear(d, d)
            self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.b1_aware = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 66))
            self.b1_agn = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 66))
            self.b2_aware = nn.Sequential(nn.Linear(2 * d + 32, d), nn.GELU(), nn.Linear(d, 22))
            self.b2_agn = nn.Sequential(nn.Linear(d + 32, d), nn.GELU(), nn.Linear(d, 22))
            self.register_buffer("syn", torch.from_numpy(SYN), persistent=False)
            self.register_buffer("nsyn", torch.from_numpy(N_SYN.astype(np.int64)), persistent=False)

        def genome_rep(self, x):
            B, K, L = x["src"].shape
            flat = x["src"].view(B * K, L)
            pres = (flat > 0).any(-1)
            cv = torch.zeros(B * K, d, device=flat.device)
            if bool(pres.any()):
                cv[pres] = self.codon(flat[pres])
            cv = cv.view(B, K, d)
            keep = trimmed(cv, pres.view(B, K), trim_genes)
            cmean = (cv * keep.unsqueeze(-1)).sum(1) / keep.sum(1, keepdim=True).clamp(min=1)
            pm = (x["src_prots"] > 0) & keep.unsqueeze(-1)
            prof = (self.prot(x["src_prots"]) * pm.unsqueeze(-1)).sum((1, 2)) / pm.sum((1, 2)).clamp(min=1).unsqueeze(-1)
            comp = self.comp_proj(x["gcomp"] @ self.codon.emb.weight)
            return self.genome(torch.cat([cmean, comp, prof], -1))

        def gene_vec(self, x):
            cvec = self.codon(x["codons"])
            B, H, T = x["tracks"].shape
            hm = x["hmask"]
            flat = hm.view(-1)
            if not bool(flat.any()):
                flat = flat.clone()
                flat[0] = True
            enc = self.hit(x["tracks"].view(B * H, T)[flat])
            tv = torch.zeros(B * H, enc.shape[-1], device=enc.device, dtype=enc.dtype)
            tv[flat] = enc
            tv = self.hit_proj(torch.cat([tv.view(B, H, -1), self.prot(x["prots"])], -1))
            any_hit = hm.any(-1)
            tv = self.hit_set(tv, src_key_padding_mask=~hm & any_hit.unsqueeze(-1))
            hv = self.hit_pool(tv, hm)
            hv = torch.where(any_hit.unsqueeze(-1), hv, self.no_hit.expand_as(hv))
            return self.gene(torch.cat([cvec, hv], -1))

        def aa_states(self, tok):
            c = self.aa
            h = F.gelu(c.c1(c.emb(tok).transpose(1, 2)))
            return F.gelu(c.c2(h) + h).transpose(1, 2)

        def sim(self, z, g):
            """[Bz, Bg] scaled cosine similarity of gene and genome embeddings."""
            return self.scale.clamp(max=math.log(100.0)).exp() * F.normalize(self.qa(z), dim=-1) @ F.normalize(self.ka(g), dim=-1).T

        def b1(self, x, h, G):
            """Per gene: (nll aware, nll agnostic, positions) over codons that have a synonym."""
            aa, cod = x["aa"], x["codons"]
            syn = self.syn[aa]
            ok = (cod > 0) & (cod < 65) & (self.nsyn[aa] > 1)
            tgt = cod.clamp(max=65).unsqueeze(-1)
            la = self.b1_aware(torch.cat([h, G.unsqueeze(1).expand(-1, h.shape[1], -1)], -1)).masked_fill(~syn, -1e4)
            lg = self.b1_agn(h).masked_fill(~syn, -1e4)
            na = -F.log_softmax(la, -1).gather(-1, tgt).squeeze(-1)
            ng = -F.log_softmax(lg, -1).gather(-1, tgt).squeeze(-1)
            n = ok.sum(1)
            return (na * ok).sum(1) / n.clamp(min=1), (ng * ok).sum(1) / n.clamp(min=1), n

        def b2(self, x, h, G):
            """Per gene: (nll aware, nll agnostic, positions) over the aligned positions of its first visible hits."""
            t = x["b2_t"]
            B, N2, T = t.shape
            ok = t > 0
            pos = (x["b2_qs"].unsqueeze(-1) + torch.arange(T, device=t.device)).clamp(0, h.shape[1] - 1)
            hq = h.gather(1, pos.view(B, N2 * T, 1).expand(-1, -1, h.shape[-1])).view(B, N2, T, -1)
            e = self.prot(x["b2_prot"]).unsqueeze(2).expand(-1, -1, T, -1)
            Ge = G.view(B, 1, 1, -1).expand(-1, N2, T, -1)
            la = self.b2_aware(torch.cat([hq, e, Ge], -1))
            lg = self.b2_agn(torch.cat([hq, e], -1))
            tgt = (t - 1).clamp(min=0).unsqueeze(-1)
            na = -F.log_softmax(la, -1).gather(-1, tgt).squeeze(-1)
            ng = -F.log_softmax(lg, -1).gather(-1, tgt).squeeze(-1)
            n = ok.sum((1, 2))
            return (na * ok).sum((1, 2)) / n.clamp(min=1), (ng * ok).sum((1, 2)) / n.clamp(min=1), n

        def forward(self, x, objectives):
            """Per-example outputs of the objectives: A gives gene and genome embeddings, B1 and B2 the likelihoods."""
            G = self.genome_rep(x)
            out = dict(G=G)
            if "A" in objectives:
                out["z"] = self.gene_vec(x)
            if "B1" in objectives or "B2" in objectives:
                h = self.aa_states(x["aa"])
                if "B1" in objectives:
                    out["b1"] = self.b1(x, h, G)
                if "B2" in objectives:
                    out["b2"] = self.b2(x, h, G)
            return out

    return Model()


def trimmed_mean(torch, v, trim, valid=None):
    """Mean of the (1 - trim) smallest values (of the valid ones)."""
    if valid is not None:
        v = v[valid]
    if v.numel() == 0:
        return v.sum() * 0.0
    keep = max(1, int(math.ceil(v.numel() * (1 - trim))))
    return torch.sort(v)[0][:keep].mean()


def losses(torch, model, out, objectives, trim):
    F = torch.nn.functional
    parts = {}
    if "A" in objectives:
        logits = model.sim(out["z"], out["G"])
        ce = F.cross_entropy(logits, torch.arange(len(logits), device=logits.device), reduction="none")
        parts["A"] = trimmed_mean(torch, ce, trim)
        parts["A_acc"] = (logits.argmax(1) == torch.arange(len(logits), device=logits.device)).float().mean().detach()
    for key in ("B1", "B2"):
        if key in objectives:
            na, ng, n = out[key.lower()]
            ok = n > 0
            parts[key] = trimmed_mean(torch, na, trim, ok) + trimmed_mean(torch, ng, trim, ok)
            parts[key + "_lr"] = ((ng - na)[ok].mean() if bool(ok.any()) else na.sum() * 0).detach()
    total = sum(parts[k] for k in objectives)
    return total, parts


class Sampler:
    """Batches of genes from distinct training genomes: half the genomes drawn group by group, each joined by its
    nearest training relative (same genus, else family, order, class, phylum; else a genome of its group), the rest
    drawn at random; one random gene per genome; one hiding depth per batch."""

    def __init__(self, data, ids):
        self.data, self.ids = data, sorted(ids)
        self.groups = sorted({data.g[g].meta["group"] for g in self.ids})
        self.pool = {grp: [g for g in self.ids if data.g[g].meta["group"] == grp] for grp in self.groups}
        R = np.array([data.g[g].ranks for g in self.ids], dtype=np.int64).reshape(len(self.ids), len(RANKS))
        self.near = {}
        for a, gid in enumerate(self.ids):
            cand = []
            for col in (4, 3, 2, 1, 0):
                if R[a, col]:
                    cand = [self.ids[b] for b in np.flatnonzero(R[:, col] == R[a, col]) if b != a]
                    if cand:
                        break
            if not cand:
                cand = [g for g in self.pool[data.g[gid].meta["group"]] if g != gid]
            self.near[gid] = cand

    def _draw(self, rng):
        grp = self.groups[rng.integers(len(self.groups))]
        return self.pool[grp][rng.integers(len(self.pool[grp]))]

    def genomes(self, rng, n):
        """n distinct genomes (all of them when there are fewer): n // 4 pairs of a genome and a near relative, the
        rest drawn group by group."""
        n = min(n, len(self.ids))
        chosen, seen = [], set()
        pairs, tries = 0, 0
        while pairs < n // 4 and len(chosen) + 2 <= n and tries < 50 * n:
            tries += 1
            a = self._draw(rng)
            near = [g for g in self.near[a] if g not in seen and g != a]
            if a in seen or not near:
                continue
            b = near[rng.integers(len(near))]
            chosen += [a, b]
            seen |= {a, b}
            pairs += 1
        while len(chosen) < n and tries < 100 * n:
            tries += 1
            g = self._draw(rng)
            if g not in seen:
                chosen.append(g)
                seen.add(g)
        for g in self.ids:                                   # small corpora: fill up deterministically
            if len(chosen) >= n:
                break
            if g not in seen:
                chosen.append(g)
                seen.add(g)
        return chosen

    def batch(self, rng, n, cfg):
        gids = self.genomes(rng, n)
        gens = [self.data.g[g] for g in gids]
        depth = DEPTHS[rng.integers(len(DEPTHS))]
        hidden = hidden_rows(self.data, gens, depth)
        items = [example(self.data, gen, int(rng.integers(gen.n)), hidden, rng, cfg) for gen in gens]
        return items, [gen.meta["group"] for gen in gens], depth


def panel_of(data, ids, n):
    """Up to n training genomes, group by group in a fixed hash order: the genomes a gene's own genome competes with
    when scoring (the size of a training batch minus one)."""
    by = {}
    for g in sorted(ids):
        by.setdefault(data.g[g].meta["group"], []).append(g)
    for grp in by:
        by[grp].sort(key=lambda g: hashlib.sha256(("panel:" + g).encode()).hexdigest())
    out, r = [], 0
    while len(out) < n and any(r < len(v) for v in by.values()):
        for grp in sorted(by):
            if r < len(by[grp]) and len(out) < n:
                out.append(by[grp][r])
        r += 1
    return out


def model_cfg(cfg):
    return dict(d=int(cfg.get("width", 128)), trim_genes=float(cfg.get("trim_genes", 0.2)))


def load_model(torch, run_dir, cfg, data, dev):
    z = np.load(os.path.join(run_dir, cfg.get("out") or os.path.join("data", "model"), "model.npz"))
    model = build(torch, data.n_lib, **model_cfg(cfg)).to(dev)
    model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("m:")})
    return model


def score_genome(model, data, torch, dev, gen, hidden_taxids, panel, cfg, objectives, seed, batch=64, genes=None, depth="genus"):
    """Per-gene scores of one genome (a training-data Genome) against the panel genomes; columns of SCORES plus the
    likelihood parts. hidden_taxids: proteomes hidden for this genome besides its own and its clade at `depth` (the
    hidden proteomes of an evaluation view); the panel genomes' proteomes and clades are hidden too."""
    rng = np.random.default_rng(seed)
    pg = [data.g[p] for p in panel if p != gen.meta["id"]]
    hidden = hidden_rows(data, [gen] + pg, depth, hidden_taxids)
    genes = list(range(gen.n)) if genes is None else list(genes)
    model.eval()
    res = {k: np.full(len(genes), np.nan) for k in ("A", "A_own", "b1_aware", "b1_agn", "b1_n", "b2_aware", "b2_agn", "b2_n")}
    with torch.no_grad():
        Gp = None
        if "A" in objectives and pg:
            prow = [example(data, p, int(rng.integers(p.n)), hidden, rng, cfg) for p in pg]
            Gp = model.genome_rep({k: v.to(dev) for k, v in collate(prow, torch).items()})
        for b0 in range(0, len(genes), batch):
            idx = genes[b0:b0 + batch]
            items = [example(data, gen, i, hidden, rng, cfg) for i in idx]
            x = {k: v.to(dev) for k, v in collate(items, torch, int(cfg.get("b2_hits", 8))).items()}
            out = model(x, objectives)
            sl = slice(b0, b0 + len(idx))
            if "A" in objectives:
                own = (model.sim(out["z"], out["G"]).diagonal())
                allv = torch.cat([own.unsqueeze(1), model.sim(out["z"], Gp)], 1) if Gp is not None else own.unsqueeze(1)
                res["A"][sl] = (torch.logsumexp(allv, 1) - own).cpu().numpy()
                res["A_own"][sl] = own.cpu().numpy()
            for key in ("b1", "b2"):
                if key in out:
                    na, ng, n = out[key]
                    res[key + "_aware"][sl], res[key + "_agn"][sl], res[key + "_n"][sl] = na.cpu().numpy(), ng.cpu().numpy(), n.cpu().numpy()
    model.train()
    for key, b in (("B1", "b1"), ("B2", "b2")):
        n = res[b + "_n"]
        res[key] = np.where(n > 0, res[b + "_aware"] - res[b + "_agn"], np.nan)    # minus the log-likelihood ratio
    res["comb"] = combine([res[k] for k in ("A", "B1", "B2")])
    res["gene"] = np.array(genes, dtype=np.int64)
    return res


def combine(cols):
    """The registered combination: the mean of the genome-wise z-scores of the scores a gene has (higher = foreign)."""
    zs = []
    for c in cols:
        c = np.asarray(c, dtype=np.float64)
        ok = np.isfinite(c)
        if ok.sum() >= 2 and np.std(c[ok]) > 0:
            zs.append(np.where(ok, (c - np.mean(c[ok])) / np.std(c[ok]), np.nan))
    if not zs:
        return np.full(len(cols[0]) if cols else 0, np.nan)
    z = np.vstack(zs)
    n = np.isfinite(z).sum(0)
    return np.where(n > 0, np.nansum(z, 0) / np.maximum(n, 1), np.nan)


def held_batches(data, ids, cfg, n_batches, batch):
    """Fixed diagnosis batches of held-out genomes (seed dev_seed), drawn like training batches."""
    if len(ids) < 2:
        return []
    sm = Sampler(data, ids)
    rng = np.random.default_rng([int(cfg.get("dev_seed", 0)), 7])
    return [sm.batch(rng, min(batch, len(ids)), cfg)[0] for _ in range(n_batches)]


def diagnose(model, torch, dev, batches, objectives, cfg):
    model.eval()
    acc, lr1, lr2 = [], [], []
    with torch.no_grad():
        for items in batches:
            x = {k: v.to(dev) for k, v in collate(items, torch, int(cfg.get("b2_hits", 8))).items()}
            out = model(x, objectives)
            _, parts = losses(torch, model, out, objectives, 0.0)
            for key, dst in (("A_acc", acc), ("B1_lr", lr1), ("B2_lr", lr2)):
                if key in parts:
                    dst.append(float(parts[key]))
    model.train()
    m = {}
    for key, v in (("held_A_acc", acc), ("held_B1_lr", lr1), ("held_B2_lr", lr2)):
        if v:
            m[key] = float(np.mean(v))
    return m


def fit(run_dir, cfg):
    import torch
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objectives = objectives_of(cfg)
    data = S.Data(run_dir, cfg)
    metas = {g: data.g[g].meta for g in data.g}
    train_ids = S.fraction_subset(metas, data.by_split.get("train", []), float(cfg.get("train_fraction", 1.0)))
    if len(train_ids) < 2:
        raise SystemExit("selfsup needs at least two training genomes (has %d)" % len(train_ids))
    model = build(torch, data.n_lib, **model_cfg(cfg)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=0.01)
    steps, batch, trim = int(cfg["steps"]), int(cfg["batch"]), float(cfg.get("trim", 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    folder = os.path.join(run_dir, cfg.get("logs") or os.path.join("data", "train", "logs"))
    resume = bool(cfg.get("resume")) and os.path.exists(os.path.join(folder, "last.npz"))
    log = TrainLogger(folder, commit=cfg.get("commit", ""), dataset_ids=[data.index["dataset_id"]], resume=resume)
    start = 0
    if resume:
        last = log.last()
        model.load_state_dict({k[2:]: torch.from_numpy(v) for k, v in last.items() if k.startswith("m:")})
        start = int(last["step"])
    for _ in range(start):
        sched.step()
    sampler = Sampler(data, train_ids)
    held = held_batches(data, data.by_split.get("heldout", []), cfg, int(cfg.get("held_batches", 8)), batch)
    n2 = int(cfg.get("b2_hits", 8))
    t0 = time.time()
    for step in range(start + 1, steps + 1):
        rng = np.random.default_rng([seed, step])                 # the data stream does not depend on a resume
        items, groups, depth = sampler.batch(rng, batch, cfg)
        x = {k: v.to(dev) for k, v in collate(items, torch, n2).items()}
        out = model(x, objectives)
        loss, parts = losses(torch, model, out, objectives, trim)
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        sched.step()
        log.count(groups)
        try:
            log.step(step, float(loss), lr=float(sched.get_last_lr()[0]), grad_norm=gn, examples_per_s=len(items) * (step - start) / max(time.time() - t0, 1e-9),
                     depth=depth, **{k: float(v) for k, v in parts.items()})
        except Diverged:
            print("diverged at step %d" % step, file=sys.stderr)
            sys.exit(3)
        if step % int(cfg.get("eval_every", 500)) == 0 or step == steps:
            m = diagnose(model, torch, dev, held, objectives, cfg)
            if m:
                log.evaluate(step, **m)
            print("step %d loss %.4f %s" % (step, float(loss), " ".join("%s %.4f" % kv for kv in sorted(m.items()))), file=sys.stderr, flush=True)
        if step % int(cfg.get("ckpt_every", 1000)) == 0 or step == steps:
            log.checkpoint(step, {"m:" + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, final=step == steps)
    log.save_seen(hashlib.sha256("\n".join(train_ids).encode()).hexdigest())
    model_dir = os.path.join(run_dir, cfg.get("out") or os.path.join("data", "model"))
    os.makedirs(model_dir, exist_ok=True)
    save_npz(os.path.join(model_dir, "model.npz"), **{"m:" + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    score(run_dir, cfg, model=model, data=data)


def score(run_dir, cfg, model=None, data=None):
    """Every gene (up to held_genes per genome) of the held-out genomes, own genus hidden, against the panel:
    data/model/heldout_scores.npz and heldout_summary.json (diagnosis only)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data or S.Data(run_dir, cfg)
    model = model or load_model(torch, run_dir, cfg, data, dev)
    objectives = objectives_of(cfg)
    metas = {g: data.g[g].meta for g in data.g}
    train_ids = S.fraction_subset(metas, data.by_split.get("train", []), float(cfg.get("train_fraction", 1.0)))
    panel = panel_of(data, train_ids, int(cfg.get("batch", 64)) - 1)
    cap = int(cfg.get("held_genes", 200))
    cols = {k: [] for k in SCORES + ("A_own", "b1_aware", "b1_agn", "b1_n", "b2_aware", "b2_agn", "b2_n")}
    keys, genes = [], []
    for gid in sorted(data.by_split.get("heldout", [])):
        gen = data.g[gid]
        sel = np.sort(np.random.default_rng([int(cfg.get("dev_seed", 0)), int(hashlib.sha256(gid.encode()).hexdigest()[:8], 16)]).permutation(gen.n)[:cap])
        r = score_genome(model, data, torch, dev, gen, (), panel, cfg, objectives, seed=int(cfg.get("dev_seed", 0)), genes=sel)
        for k in cols:
            cols[k].append(r[k])
        keys += [gid] * len(sel)
        genes.append(r["gene"])
    model_dir = os.path.join(run_dir, cfg.get("out") or os.path.join("data", "model"))
    os.makedirs(model_dir, exist_ok=True)
    arrays = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}
    save_npz(os.path.join(model_dir, "heldout_scores.npz"), genome=np.array(keys, dtype="S"), gene=np.concatenate(genes) if genes else np.zeros(0, np.int64), **arrays)
    summ = dict(genomes=len(set(keys)), genes=len(keys), panel=panel, objectives=objectives)
    for key, (a, g, n) in (("B1", ("b1_aware", "b1_agn", "b1_n")), ("B2", ("b2_aware", "b2_agn", "b2_n"))):
        ok = arrays[n] > 0 if len(arrays[n]) else np.zeros(0, bool)
        if ok.any():
            summ[key + "_lr_mean"] = float(np.mean(arrays[g][ok] - arrays[a][ok]))
            summ[key + "_aware_nll"] = float(np.mean(arrays[a][ok]))
            summ[key + "_agnostic_nll"] = float(np.mean(arrays[g][ok]))
            summ[key + "_scored"] = int(ok.sum())
    if np.isfinite(arrays["A"]).any():
        a = arrays["A"][np.isfinite(arrays["A"])]
        summ["A_mean_neglogp_own"] = float(np.mean(a))
        summ["A_chance_neglogp"] = float(math.log(len(panel) + 1))
    with open(os.path.join(model_dir, "heldout_summary.json"), "w") as f:
        json.dump(summ, f, indent=1, sort_keys=True)


def score_genomes(run_dir, cfg):
    """Score the genomes of cfg["genomes_index"] (JSON: genomes = [{id, path, taxid, ranks, group, hidden_taxids}],
    paths relative to the run) into cfg["genomes_out"]/<id>.npz: one row per gene in the genome's gene order."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = S.Data(run_dir, cfg)
    model = load_model(torch, run_dir, cfg, data, dev)
    objectives = objectives_of(cfg)
    metas = {g: data.g[g].meta for g in data.g}
    train_ids = S.fraction_subset(metas, data.by_split.get("train", []), float(cfg.get("train_fraction", 1.0)))
    panel = panel_of(data, train_ids, int(cfg.get("batch", 64)) - 1)
    with open(os.path.join(run_dir, cfg["genomes_index"])) as f:
        idx = json.load(f)
    out_dir = os.path.join(run_dir, cfg["genomes_out"])
    os.makedirs(out_dir, exist_ok=True)
    for e in idx["genomes"]:
        meta = dict(id=e["id"], taxid=int(e["taxid"]), ranks=[int(r) for r in e["ranks"]], group=e.get("group", ""), split="external")
        gen = S.Genome(meta, os.path.join(run_dir, e["path"]))
        r = score_genome(model, data, torch, dev, gen, e.get("hidden_taxids") or (), panel, cfg, objectives,
                         seed=int(cfg.get("dev_seed", 0)), depth=e.get("depth", "genus"))
        save_npz(os.path.join(out_dir, e["id"] + ".npz"), **r)


def main():
    mode, run_dir, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(cfg_path) as f:
        cfg = json.load(f)
    if mode == "fit":
        fit(run_dir, cfg)
    elif mode == "score":
        score(run_dir, cfg)
    elif mode == "genomes":
        score_genomes(run_dir, cfg)
    else:
        raise SystemExit("mode is fit, score or genomes")


if __name__ == "__main__":
    main()
