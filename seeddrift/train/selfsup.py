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
  C   homolog pairs. For a gene and its nearest homolog in another corpus genome (stage pairs, a search of the corpus
      against itself), the homolog's residue at every aligned position and, where the residue has synonyms, its codon,
      given the gene's amino acids and codons, once with both genomes' representations (aware) and once without them.
      Evolution has already noised every pair; both ends are real sequences of today. Foreign score: minus the lowest
      log-likelihood ratio over the gene's partners (the pair least like the two genomes' vertical expectation; a pair
      far closer than the genomes predict is a recent sharing), with that partner recorded.

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
sealed test split and the evaluation inputs are not loaded (load_data). Resume restores the weights, AdamW's state and
the counts; every step draws its data and its dropout from (seed, step). Never imports an evaluation generator (gate
G2 static).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from seeddrift.train import source_model as S  # noqa: E402
from seeddrift.train.logger import Diverged, TrainLogger, save_npz  # noqa: E402

RANKS, DEPTHS, CODE = S.RANKS, S.DEPTHS, S.CODE
OBJECTIVES = ("A", "B1", "B2", "C")
DEFAULT_OBJECTIVES = ("A", "B1", "B2")                  # C needs the pairs stage; profiles name it
SCORES = ("A", "B1", "B2", "C", "comb")


def _synonyms():
    t = np.zeros((22, 66), dtype=bool)          # t[a, c]: codon token c (1..64) codes amino-acid token a (21 = stop)
    for c in range(1, 65):
        t[CODE[c], c] = True
    return t


SYN = _synonyms()
N_SYN = SYN.sum(1)


def objectives_of(cfg):
    obj = cfg.get("objectives")
    obj = list(DEFAULT_OBJECTIVES) if obj is None else [obj] if isinstance(obj, str) else list(obj)
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
    k, ls, hs = int(cfg.get("source_genes", 64)), int(cfg.get("source_codons", 256)), int(cfg.get("source_hits", 8))
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


_CIG = re.compile(rb"(\d+)([MID])")
_AA_LUT = np.full(256, 21, dtype=np.int64)
for _i, _a in enumerate("ACDEFGHIKLMNPQRSTVWY"):
    _AA_LUT[ord(_a)] = _i + 1


def pair_track(cigar, sseq, ss):
    """(target residue token per query position of the alignment: 1..20, 21 other, 22 none; target residue index
    (0-based) per query position, -1 for none), from the CIGAR and the aligned target residues (stage pairs)."""
    sb = np.frombuffer(bytes(sseq), dtype=np.uint8)
    tok, pos, j, tp = [], [], 0, int(ss) - 1
    for n, op in _CIG.findall(bytes(cigar)):
        n = int(n)
        if op == b"M":
            tok.append(_AA_LUT[sb[j:j + n]])
            pos.append(np.arange(tp, tp + n))
            j, tp = j + n, tp + n
        elif op == b"I":
            tok.append(np.full(n, 22, dtype=np.int64))
            pos.append(np.full(n, -1, dtype=np.int64))
        else:
            j, tp = j + n, tp + n
    if j != len(sb):
        raise SystemExit("pair row: CIGAR %s does not consume the aligned target residues (%d of %d)" % (bytes(cigar).decode(), j, len(sb)))
    if not tok:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(tok), np.concatenate(pos)


class Pairs:
    """Rows of stage pairs: for a genome, per gene the nearest homologs in other corpus genomes (corpus genomes from
    data/pairs/<id>.npz; other genomes from the path given)."""

    def __init__(self, run_dir, folder="data/pairs"):
        self.run_dir, self.folder = run_dir, folder
        with open(os.path.join(run_dir, folder, "genomes.json")) as f:
            self.order = json.load(f)["genomes"]
        self.z = {}

    def of(self, gid, path=None):
        if gid not in self.z:
            z = np.load(path or os.path.join(self.run_dir, self.folder, gid + ".npz"))
            self.z[gid] = {k: z[k] for k in z.files}
        return self.z[gid]

    def rows(self, gid, i, allowed, path=None, n_codons=None):
        """Row indices of gene i whose target genome is in `allowed` (genome ids), nearest first; with n_codons, only
        rows whose alignment starts within the gene's first n_codons codons (the stored part)."""
        z = self.of(gid, path)
        return [r for r in range(int(z["off"][i]), int(z["off"][i + 1])) if self.order[int(z["tgenome"][r])] in allowed
                and (n_codons is None or int(z["qs"][r]) - 1 < n_codons)]


def source_of(data, gen, exclude, hidden, rng, cfg):
    """A genome's source for its representation: other genes' codons and the proteomes of their visible hits, and its
    codon frequencies without gene `exclude` (-1: none)."""
    k, ls, hs = int(cfg.get("source_genes", 64)), int(cfg.get("source_codons", 256)), int(cfg.get("source_hits", 8))
    idx = [int(j) for j in rng.choice(gen.n, size=min(k + 1, gen.n), replace=False) if j != exclude][:k]
    gcomp = S.frequencies(gen.ccount - (gen.gene_count(exclude) if exclude >= 0 else 0))
    return dict(src=[gen.codons[gen.coff[j]:gen.coff[j + 1]][:ls].astype(np.int64) for j in idx],
                src_prots=[visible_prots(gen, j, hidden, hs) for j in idx], gcomp=gcomp, idx=set(idx))


def c_example(data, pairs, gen, i, r, hidden, rng, cfg, partner, ppath=None, srcB=None, sources=True):
    """Gene i of `gen` and its pair row r (partner genome `partner`, a Genome): the gene's codons, the target residue and
    the target's codon at every query position of the alignment, and the sources of both genomes (the partner's
    without the partner gene)."""
    z = pairs.of(gen.meta["id"], ppath)
    c = gen.codons[gen.coff[i]:gen.coff[i + 1]][:data.maxc].astype(np.int64)
    t, tp = pair_track(z["cigar"][z["cigar_off"][r]:z["cigar_off"][r + 1]], z["sseq"][z["sseq_off"][r]:z["sseq_off"][r + 1]], z["ss"][r])
    qs = int(z["qs"][r]) - 1
    j = int(z["tgene"][r])
    cb = partner.codons[partner.coff[j]:partner.coff[j + 1]].astype(np.int64)
    m = max(0, min(len(t), len(c) - qs, data.maxt))
    t, tp = t[:m], tp[:m]
    tc = np.where((tp >= 0) & (tp < len(cb)), cb[np.clip(tp, 0, max(len(cb) - 1, 0))] if len(cb) else 0, 0)
    out = dict(c=c, qs=qs, t=t, tcod=tc.astype(np.int64), partner_gene=j)
    if sources:
        out["A"] = source_of(data, gen, i, hidden, rng, cfg)
        out["B"] = srcB if srcB is not None and j not in srcB["idx"] else source_of(data, partner, j, hidden, rng, cfg)
    return out


def _src_arrays(srcs):
    B = len(srcs)
    K = max([len(s["src"]) for s in srcs] + [1])
    Ls = max([len(x) for s in srcs for x in s["src"]] + [1])
    Hs = max([len(p) for s in srcs for p in s["src_prots"]] + [1])
    src = np.zeros((B, K, Ls), np.int64)
    sp = np.zeros((B, K, Hs), np.int64)
    gc = np.zeros((B, 66), np.float32)
    for b, s in enumerate(srcs):
        for k, (x, p) in enumerate(zip(s["src"], s["src_prots"])):
            src[b, k, :len(x)] = x
            sp[b, k, :len(p)] = p + 1
        gc[b] = s["gcomp"]
    return src, sp, gc


def collate_c(items, torch):
    B = len(items)
    Lc = max([len(e["c"]) for e in items] + [1])
    T = max([len(e["t"]) for e in items] + [1])
    codons = np.zeros((B, Lc), np.int64)
    qs = np.zeros(B, np.int64)
    t = np.zeros((B, T), np.int64)
    tcod = np.zeros((B, T), np.int64)
    for b, e in enumerate(items):
        codons[b, :len(e["c"])] = e["c"]
        qs[b] = e["qs"]
        t[b, :len(e["t"])] = e["t"]
        tcod[b, :len(e["tcod"])] = e["tcod"]
    aa = np.where(codons > 0, CODE[codons], 0)
    out = dict(codons=codons, aa=aa, qs=qs, t=t, tcod=tcod)
    if all("A" in e for e in items):                                       # scoring passes the representations instead
        out["srcA"], out["src_protsA"], out["gcompA"] = _src_arrays([e["A"] for e in items])
        out["srcB"], out["src_protsB"], out["gcompB"] = _src_arrays([e["B"] for e in items])
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
            self.c_res_aware = nn.Sequential(nn.Linear(3 * d + 64, d), nn.GELU(), nn.Linear(d, 22))
            self.c_res_agn = nn.Sequential(nn.Linear(d + 64, d), nn.GELU(), nn.Linear(d, 22))
            self.c_cod_aware = nn.Sequential(nn.Linear(3 * d + 64, d), nn.GELU(), nn.Linear(d, 66))
            self.c_cod_agn = nn.Sequential(nn.Linear(d + 64, d), nn.GELU(), nn.Linear(d, 66))
            self.register_buffer("code", torch.from_numpy(CODE.copy()), persistent=False)
            self.register_buffer("syn", torch.from_numpy(SYN), persistent=False)
            self.register_buffer("nsyn", torch.from_numpy(N_SYN.astype(np.int64)), persistent=False)

        def genome_rep(self, x):
            cmean, prof = self.genome_parts(x)
            return self.genome_head(cmean, prof, x["gcomp"])

        def genome_head(self, cmean, prof, gcomp):
            comp = self.comp_proj(gcomp @ self.codon.emb.weight)
            return self.genome(torch.cat([cmean, comp, prof], -1))

        def genome_parts(self, x):
            """(robust mean of the source genes' codon encodings, mean proteome embedding of their visible hits)."""
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
            return cmean, prof

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
            """Per-position states; padding is zeroed after each layer, so a gene's states do not depend on the length
            of the longest gene of its batch."""
            c = self.aa
            m = (tok > 0).unsqueeze(1).to(c.emb.weight.dtype)
            h = F.gelu(c.c1(c.emb(tok).transpose(1, 2))) * m
            return (F.gelu(c.c2(h) + h) * m).transpose(1, 2)

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

        def c(self, x, GA=None, GB=None):
            """Per pair: (residue nll aware, agnostic, positions; codon nll aware, agnostic, positions). The target residue
            at every query position (22 classes) and, where it has synonyms and the partner's codon codes it, the
            partner's codon among them; inputs the gene's amino-acid states and codons, aware also both genomes."""
            if GA is None:
                GA = self.genome_rep(dict(src=x["srcA"], src_prots=x["src_protsA"], gcomp=x["gcompA"]))
            if GB is None:
                GB = self.genome_rep(dict(src=x["srcB"], src_prots=x["src_protsB"], gcomp=x["gcompB"]))
            h = self.aa_states(x["aa"])
            t, tc = x["t"], x["tcod"]
            B, T = t.shape
            pos = (x["qs"].unsqueeze(-1) + torch.arange(T, device=t.device)).clamp(0, h.shape[1] - 1)
            hq = h.gather(1, pos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            ce = self.codon.emb(x["codons"].gather(1, pos))
            g = torch.cat([GA, GB], -1).unsqueeze(1).expand(-1, T, -1)
            inp_g, inp = torch.cat([hq, ce, g], -1), torch.cat([hq, ce], -1)
            ok_r = t > 0
            tr = (t - 1).clamp(min=0).unsqueeze(-1)
            ra = -F.log_softmax(self.c_res_aware(inp_g), -1).gather(-1, tr).squeeze(-1)
            rg = -F.log_softmax(self.c_res_agn(inp), -1).gather(-1, tr).squeeze(-1)
            tt = t.clamp(max=21)
            ok_c = (t >= 1) & (t <= 20) & (self.nsyn[tt] > 1) & (tc > 0) & (tc < 65) & (self.code[tc.clamp(max=65)] == t)
            syn = self.syn[tt]
            tcc = tc.clamp(max=65).unsqueeze(-1)
            ca = -F.log_softmax(self.c_cod_aware(inp_g).masked_fill(~syn, -1e4), -1).gather(-1, tcc).squeeze(-1)
            cg = -F.log_softmax(self.c_cod_agn(inp).masked_fill(~syn, -1e4), -1).gather(-1, tcc).squeeze(-1)
            nr, nc = ok_r.sum(1), ok_c.sum(1)
            mean = lambda v, ok, n: (v * ok).sum(1) / n.clamp(min=1)
            return mean(ra, ok_r, nr), mean(rg, ok_r, nr), nr, mean(ca, ok_c, nc), mean(cg, ok_c, nc), nc

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
    if "C" in objectives and "c" in out:
        ra, rg, nr, ca, cg, nc = out["c"]
        okr, okc = nr > 0, nc > 0
        parts["C"] = trimmed_mean(torch, ra, trim, okr) + trimmed_mean(torch, rg, trim, okr) + trimmed_mean(torch, ca, trim, okc) + trimmed_mean(torch, cg, trim, okc)
        lr = (rg - ra) * okr + (cg - ca) * okc
        parts["C_lr"] = (lr[okr].mean() if bool(okr.any()) else ra.sum() * 0).detach()
    total = sum(parts[k] for k in objectives if k in parts)
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


class CSampler:
    """Batches for objective C: genomes drawn group by group (distinct), one random gene each that has a partner in an
    allowed genome (training genomes), and one of its partner rows at random; one hiding depth per batch."""

    def __init__(self, data, pairs, ids, partners):
        self.data, self.pairs, self.ids, self.partners = data, pairs, sorted(ids), set(partners)
        self.groups = sorted({data.g[g].meta["group"] for g in self.ids})
        self.pool = {grp: [g for g in self.ids if data.g[g].meta["group"] == grp] for grp in self.groups}
        self.has = {}
        for g in self.ids:
            z = pairs.of(g)
            gen = data.g[g]
            gene_of_row = np.repeat(np.arange(len(z["off"]) - 1), np.diff(z["off"].astype(np.int64)))
            stored = np.minimum(np.diff(gen.coff), data.maxc)[gene_of_row] if len(gene_of_row) else np.zeros(0, np.int64)
            inside = (z["qs"].astype(np.int64) - 1) < stored
            tg = np.array([pairs.order[int(x)] in self.partners and pairs.order[int(x)] != g for x in z["tgenome"]], dtype=bool) & inside
            tg = tg.astype(np.int64)
            cs = np.concatenate([[0], np.cumsum(tg)])
            off = z["off"].astype(np.int64)
            self.has[g] = np.flatnonzero(cs[off[1:]] - cs[off[:-1]] > 0)
        self.ids = [g for g in self.ids if len(self.has[g])]
        self.groups = sorted({data.g[g].meta["group"] for g in self.ids})
        self.pool = {grp: [g for g in self.ids if data.g[g].meta["group"] == grp] for grp in self.groups}

    def batch(self, rng, n, cfg):
        if not self.ids:
            return [], [], None
        n = min(n, len(self.ids))
        chosen, seen, tries = [], set(), 0
        while len(chosen) < n and tries < 100 * n:
            tries += 1
            grp = self.groups[rng.integers(len(self.groups))]
            g = self.pool[grp][rng.integers(len(self.pool[grp]))]
            if g not in seen:
                seen.add(g)
                chosen.append(g)
        picks = []
        for g in chosen:
            i = int(self.has[g][rng.integers(len(self.has[g]))])
            n_cod = min(int(self.data.g[g].coff[i + 1] - self.data.g[g].coff[i]), self.data.maxc)
            rows = [r for r in self.pairs.rows(g, i, self.partners, n_codons=n_cod) if self.pairs.order[int(self.pairs.of(g)["tgenome"][r])] != g]
            r = rows[rng.integers(len(rows))]
            picks.append((g, i, r, self.pairs.order[int(self.pairs.of(g)["tgenome"][r])]))
        depth = DEPTHS[rng.integers(len(DEPTHS))]
        gens = [self.data.g[g] for g in sorted({p[0] for p in picks} | {p[3] for p in picks})]
        hidden = hidden_rows(self.data, gens, depth)
        items = [c_example(self.data, self.pairs, self.data.g[g], i, r, hidden, rng, cfg, self.data.g[b]) for g, i, r, b in picks]
        return items, [self.data.g[g].meta["group"] for g, _, _, _ in picks], depth


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


def load_data(run_dir, cfg, splits=("train", "heldout")):
    """source_model.Data with the genomes of `splits` only (the sealed test part and the evaluation inputs are not
    opened); the same attributes as Data.__init__ sets."""
    d = object.__new__(S.Data)
    with open(os.path.join(run_dir, "data", "train_data", "index.json")) as f:
        d.index = json.load(f)
    lib = np.load(os.path.join(run_dir, d.index["library"]))
    d.lib_ranks = lib["ranks"].astype(np.int64)
    d.lib_taxid = lib["taxid"].astype(np.int64)
    d.lib_cols = np.concatenate([d.lib_ranks, d.lib_taxid[:, None]], 1)
    d.n_lib = len(d.lib_ranks)
    d.g = {m["id"]: S.Genome(m, os.path.join(run_dir, "data", "train_data", m["id"] + ".npz")) for m in d.index["genomes"] if m["split"] in splits}
    d.by_split = {}
    for m in d.index["genomes"]:
        if m["id"] in d.g and d.g[m["id"]].n:
            d.by_split.setdefault(m["split"], []).append(m["id"])
    d.maxc = int(cfg.get("max_codons", 512))
    d.kc, d.kn, d.maxt = int(cfg.get("hits_centre", 32)), int(cfg.get("hits_neighbour", 4)), int(cfg.get("max_track", 512))
    d.comp = bool(cfg.get("genome_composition", False))
    return d


def model_cfg(cfg):
    return dict(d=int(cfg.get("width", 128)), trim_genes=float(cfg.get("trim_genes", 0.2)))


def load_model(torch, run_dir, cfg, data, dev):
    z = np.load(os.path.join(run_dir, cfg.get("out") or os.path.join("data", "model"), "model.npz"))
    model = build(torch, data.n_lib, **model_cfg(cfg)).to(dev)
    model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("m:")})
    return model


def score_genome(model, data, torch, dev, gen, hidden_taxids, panel, cfg, objectives, seed, batch=64, genes=None, depth="genus", pairs=None,
                 partners=(), ppath=None):
    """Per-gene scores of one genome (a training-data Genome) against the panel genomes; columns of SCORES plus the
    likelihood parts. hidden_taxids: proteomes hidden for this genome besides its own and its clade at `depth` (the
    hidden proteomes of an evaluation view); the panel genomes' proteomes and clades are hidden too."""
    rng = np.random.default_rng(seed)
    pg = [data.g[p] for p in panel if p != gen.meta["id"]]
    hidden = hidden_rows(data, [gen] + pg, depth, hidden_taxids)
    genes = list(range(gen.n)) if genes is None else list(genes)
    model.eval()
    res = {k: np.full(len(genes), np.nan) for k in ("A", "A_own", "b1_aware", "b1_agn", "b1_n", "b2_aware", "b2_agn", "b2_n", "C", "c_rows")}
    res["c_partner"] = np.full(len(genes), -1, dtype=np.int64)
    Gs = {}
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
            if "C" in objectives:
                for k, i in enumerate(idx):
                    Gs[i] = out["G"][k]
            if "A" in objectives:
                own = (model.sim(out["z"], out["G"]).diagonal())
                allv = torch.cat([own.unsqueeze(1), model.sim(out["z"], Gp)], 1) if Gp is not None else own.unsqueeze(1)
                res["A"][sl] = (torch.logsumexp(allv, 1) - own).cpu().numpy()
                res["A_own"][sl] = own.cpu().numpy()
            for key in ("b1", "b2"):
                if key in out:
                    na, ng, n = out[key]
                    res[key + "_aware"][sl], res[key + "_agn"][sl], res[key + "_n"][sl] = na.cpu().numpy(), ng.cpu().numpy(), n.cpu().numpy()
        if "C" in objectives and pairs is not None:
            c_scores(model, data, torch, dev, gen, genes, Gs, hidden, rng, cfg, pairs, partners, ppath, res, batch, depth=depth)
    model.train()
    for key, b in (("B1", "b1"), ("B2", "b2")):
        n = res[b + "_n"]
        res[key] = np.where(n > 0, res[b + "_aware"] - res[b + "_agn"], np.nan)    # minus the log-likelihood ratio
    res["comb"] = combine([res[k] for k in ("A", "B1", "B2", "C")])
    res["gene"] = np.array(genes, dtype=np.int64)
    return res


def c_scores(model, data, torch, dev, gen, genes, Gs, hidden, rng, cfg, pairs, partners, ppath, res, batch, depth="genus"):
    """Objective C for the scored genes: every pair row of a gene to an allowed partner genome (the nearest c_partners
    whose alignment starts in the stored codons), the log-likelihood ratio (residues plus codons, per position) with
    the gene's own representation (from the scoring pass, without the gene) and the partner's. The partner is hidden as
    in training (its own proteome and clade at `depth`, on top of the scored genome's hiding); its source genes are
    sampled once per partner genome (again without the partner gene when it is among them) and its composition is
    taken without the partner gene for every row. res["C"] = minus the lowest ratio, res["c_partner"] = its genome."""
    allowed = set(partners) - {gen.meta["id"]}
    cap = int(cfg.get("c_partners", 16))
    work = []
    for k, i in enumerate(genes):
        n_cod = min(int(gen.coff[i + 1] - gen.coff[i]), data.maxc)
        for r in pairs.rows(gen.meta["id"], i, allowed, ppath, n_codons=n_cod)[:cap]:
            work.append((k, i, r))
    parts = {}                                                         # partner -> (source, hidden, cmean, prof)
    z = pairs.of(gen.meta["id"], ppath)
    best = {}
    for b0 in range(0, len(work), batch * 4):
        part = work[b0:b0 + batch * 4]
        items, GA, cm, pf, gc = [], [], [], [], []
        for k, i, r in part:
            pid = pairs.order[int(z["tgenome"][r])]
            pg = data.g[pid]
            if pid not in parts:
                hp = hidden | hidden_rows(data, [pg], depth)
                src = source_of(data, pg, -1, hp, rng, cfg)
                c0, p0 = model.genome_parts({k2: v.to(dev) for k2, v in _src_tensors([src], torch).items()})
                parts[pid] = (src, hp, c0[0], p0[0])
            src, hp, c0, p0 = parts[pid]
            j = int(z["tgene"][r])
            if j in src["idx"]:                                            # the partner gene must not describe its genome
                alt = source_of(data, pg, j, hp, rng, cfg)
                c1, p1 = model.genome_parts({k2: v.to(dev) for k2, v in _src_tensors([alt], torch).items()})
                c0, p0 = c1[0], p1[0]
            items.append(c_example(data, pairs, gen, i, r, hidden, rng, cfg, pg, ppath=ppath, sources=False))
            GA.append(Gs[i])
            cm.append(c0)
            pf.append(p0)
            gc.append(torch.from_numpy(S.frequencies(pg.ccount - pg.gene_count(j))))
        GB = model.genome_head(torch.stack(cm), torch.stack(pf), torch.stack(gc).to(dev))
        x = {k2: v.to(dev) for k2, v in collate_c(items, torch).items()}
        ra, rg, nr, ca, cg, nc = model.c(x, torch.stack(GA), GB)
        lr = ((rg - ra) * (nr > 0) + (cg - ca) * (nc > 0)).cpu().numpy()
        ok = ((nr > 0) | (nc > 0)).cpu().numpy()
        for (k, i, r), v, o in zip(part, lr, ok):
            if o and (k not in best or v < best[k][0]):
                best[k] = (float(v), int(z["tgenome"][r]))
            res["c_rows"][k] = (0 if np.isnan(res["c_rows"][k]) else res["c_rows"][k]) + 1
    for k, (v, tg) in best.items():
        res["C"][k] = -v
        res["c_partner"][k] = tg


def _src_tensors(srcs, torch):
    src, sp, gc = _src_arrays(srcs)
    return dict(src=torch.from_numpy(src), src_prots=torch.from_numpy(sp), gcomp=torch.from_numpy(gc))


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


def held_c_batches(data, pairs, ids, partners, cfg, n_batches, batch):
    """Fixed diagnosis batches of objective C: held-out genomes paired with training genomes."""
    cs = CSampler(data, pairs, ids, partners)
    rng = np.random.default_rng([int(cfg.get("dev_seed", 0)), 11])
    return [b for b in (cs.batch(rng, batch, cfg)[0] for _ in range(n_batches)) if b]


def diagnose(model, torch, dev, batches, objectives, cfg, c_batches=()):
    model.eval()
    acc, lr1, lr2, lrc = [], [], [], []
    ab = [o for o in objectives if o != "C"]
    with torch.no_grad():
        for items in (batches if ab else []):
            x = {k: v.to(dev) for k, v in collate(items, torch, int(cfg.get("b2_hits", 8))).items()}
            out = model(x, ab)
            _, parts = losses(torch, model, out, ab, 0.0)
            for key, dst in (("A_acc", acc), ("B1_lr", lr1), ("B2_lr", lr2)):
                if key in parts:
                    dst.append(float(parts[key]))
        for items in (c_batches if "C" in objectives else []):
            xc = {k: v.to(dev) for k, v in collate_c(items, torch).items()}
            _, parts = losses(torch, model, dict(c=model.c(xc)), ["C"], 0.0)
            lrc.append(float(parts["C_lr"]))
    model.train()
    m = {}
    for key, v in (("held_A_acc", acc), ("held_B1_lr", lr1), ("held_B2_lr", lr2), ("held_C_lr", lrc)):
        if v:
            m[key] = float(np.mean(v))
    return m


def opt_arrays(opt):
    """AdamW's state as arrays for a checkpoint ("o:<parameter index>:<name>")."""
    out = {}
    for i, st in opt.state_dict()["state"].items():
        for k, v in st.items():
            out["o:%d:%s" % (i, k)] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    return out


def load_opt(torch, opt, arrays):
    sd = opt.state_dict()
    state = {}
    for k, v in arrays.items():
        if k.startswith("o:"):
            _, i, name = k.split(":", 2)
            state.setdefault(int(i), {})[name] = torch.from_numpy(np.array(v))
    if state:
        sd["state"] = state
        opt.load_state_dict(sd)


def fit(run_dir, cfg):
    import torch
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objectives = objectives_of(cfg)
    data = load_data(run_dir, cfg)
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
        load_opt(torch, opt, last)
        if "seen" in last:
            log.seen = json.loads(bytes(last["seen"]).decode())
        start = int(last["step"])
    for _ in range(start):
        sched.step()
    ab = [o for o in objectives if o != "C"]
    sampler = Sampler(data, train_ids)
    held = held_batches(data, data.by_split.get("heldout", []), cfg, int(cfg.get("held_batches", 8)), batch)
    pairs = csampler = None
    held_c = []
    if "C" in objectives:
        pairs = Pairs(run_dir, cfg.get("pairs_dir", os.path.join("data", "pairs")))
        csampler = CSampler(data, pairs, train_ids, train_ids)
        if not csampler.ids:
            raise SystemExit("objective C: no training genome has a pair row to another training genome")
        held_c = held_c_batches(data, pairs, data.by_split.get("heldout", []), train_ids, cfg, int(cfg.get("held_batches", 8)), int(cfg.get("c_batch", batch)))
    n2 = int(cfg.get("b2_hits", 8))
    t0 = time.time()
    for step in range(start + 1, steps + 1):
        rng = np.random.default_rng([seed, step])                 # the data stream does not depend on a resume
        torch.manual_seed(int(rng.integers(1 << 62)))              # nor does dropout
        out, items, groups, depth = {}, [], [], None
        if ab:
            items, groups, depth = sampler.batch(rng, batch, cfg)
            x = {k: v.to(dev) for k, v in collate(items, torch, n2).items()}
            out = model(x, ab)
        if csampler is not None:
            c_items, c_groups, c_depth = csampler.batch(rng, int(cfg.get("c_batch", batch)), cfg)
            if c_items:
                out["c"] = model.c({k: v.to(dev) for k, v in collate_c(c_items, torch).items()})
                groups = groups + c_groups
                depth = depth or c_depth
        loss, parts = losses(torch, model, out, objectives, trim)
        if not parts:
            raise SystemExit("step %d: no objective produced a loss (empty batches)" % step)
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        sched.step()
        log.count(groups)
        try:
            log.step(step, float(loss.detach()), lr=float(sched.get_last_lr()[0]), grad_norm=gn, examples_per_s=len(items) * (step - start) / max(time.time() - t0, 1e-9),
                     depth=depth, **{k: float(v.detach()) for k, v in parts.items()})
        except Diverged:
            print("diverged at step %d" % step, file=sys.stderr)
            sys.exit(3)
        if step % int(cfg.get("eval_every", 500)) == 0 or step == steps:
            m = diagnose(model, torch, dev, held, objectives, cfg, held_c)
            if m:
                log.evaluate(step, **m)
            print("step %d loss %.4f %s" % (step, float(loss.detach()), " ".join("%s %.4f" % kv for kv in sorted(m.items()))), file=sys.stderr, flush=True)
        if step % int(cfg.get("ckpt_every", 1000)) == 0 or step == steps:
            arrays = {"m:" + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            arrays.update(opt_arrays(opt))
            arrays["seen"] = np.frombuffer(json.dumps(log.seen, sort_keys=True).encode(), dtype=np.uint8)
            log.checkpoint(step, arrays, final=step == steps)
    log.save_seen(hashlib.sha256("\n".join(train_ids).encode()).hexdigest())
    model_dir = os.path.join(run_dir, cfg.get("out") or os.path.join("data", "model"))
    os.makedirs(model_dir, exist_ok=True)
    save_npz(os.path.join(model_dir, "model.npz"), **{"m:" + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    score(run_dir, cfg, model=model, data=data)


def score(run_dir, cfg, model=None, data=None):
    """Every gene (up to held_genes per genome) of the held-out genomes, own genus hidden, against the panel:
    heldout_scores.npz and heldout_summary.json (diagnosis only) in the model's folder (cfg out), or in cfg score_out
    when given (a recomputation that must not replace the model's own scores)."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data or load_data(run_dir, cfg)
    model = model or load_model(torch, run_dir, cfg, data, dev)
    objectives = objectives_of(cfg)
    metas = {g: data.g[g].meta for g in data.g}
    train_ids = S.fraction_subset(metas, data.by_split.get("train", []), float(cfg.get("train_fraction", 1.0)))
    panel = panel_of(data, train_ids, int(cfg.get("batch", 64)) - 1)
    pairs = Pairs(run_dir, cfg.get("pairs_dir", os.path.join("data", "pairs"))) if "C" in objectives else None
    cap = int(cfg.get("held_genes", 200))
    cols = {k: [] for k in SCORES + ("A_own", "b1_aware", "b1_agn", "b1_n", "b2_aware", "b2_agn", "b2_n", "c_partner", "c_rows")}
    keys, genes = [], []
    for gid in sorted(data.by_split.get("heldout", [])):
        gen = data.g[gid]
        sel = np.sort(np.random.default_rng([int(cfg.get("dev_seed", 0)), int(hashlib.sha256(gid.encode()).hexdigest()[:8], 16)]).permutation(gen.n)[:cap])
        r = score_genome(model, data, torch, dev, gen, (), panel, cfg, objectives, seed=int(cfg.get("dev_seed", 0)), genes=sel, pairs=pairs, partners=train_ids)
        for k in cols:
            cols[k].append(r[k])
        keys += [gid] * len(sel)
        genes.append(r["gene"])
    model_dir = os.path.join(run_dir, cfg.get("score_out") or cfg.get("out") or os.path.join("data", "model"))
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
    if np.isfinite(arrays["C"]).any():
        summ["C_lr_mean_of_minimum"] = float(-np.mean(arrays["C"][np.isfinite(arrays["C"])]))
        summ["C_scored"] = int(np.isfinite(arrays["C"]).sum())
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
    data = load_data(run_dir, cfg)
    model = load_model(torch, run_dir, cfg, data, dev)
    objectives = objectives_of(cfg)
    metas = {g: data.g[g].meta for g in data.g}
    train_ids = S.fraction_subset(metas, data.by_split.get("train", []), float(cfg.get("train_fraction", 1.0)))
    panel = panel_of(data, train_ids, int(cfg.get("batch", 64)) - 1)
    pairs = Pairs(run_dir, cfg.get("pairs_dir", os.path.join("data", "pairs"))) if "C" in objectives else None
    with open(os.path.join(run_dir, cfg["genomes_index"])) as f:
        idx = json.load(f)
    out_dir = os.path.join(run_dir, cfg["genomes_out"])
    os.makedirs(out_dir, exist_ok=True)
    for e in idx["genomes"]:
        meta = dict(id=e["id"], taxid=int(e["taxid"]), ranks=[int(r) for r in e["ranks"]], group=e.get("group", ""), split="external")
        gen = S.Genome(meta, os.path.join(run_dir, e["path"]))
        if "C" in objectives and not e.get("pairs"):
            raise SystemExit("objective C needs the pair rows of %s (its search against the corpus)" % e["id"])
        r = score_genome(model, data, torch, dev, gen, e.get("hidden_taxids") or (), panel, cfg, objectives,
                         seed=int(cfg.get("dev_seed", 0)), depth=e.get("depth", "genus"), pairs=pairs, partners=train_ids,
                         ppath=os.path.join(run_dir, e["pairs"]) if e.get("pairs") else None)
        save_npz(os.path.join(out_dir, e["id"] + ".npz"), **r)
        if pairs is not None:
            pairs.z.pop(e["id"], None)


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
