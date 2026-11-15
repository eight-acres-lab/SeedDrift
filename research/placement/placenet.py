"""Placement model: where does a protein belong, given the homologs retrieved from the reference library?

One example = one query protein and a selection of its hits after some lineages have been hidden from the
library. Tokens are the hits. The only taxonomic input is relative: for every pair of hits, the deepest rank
they share (0 = different domains ... 6 = same genus, 7 = same proteome), used as an attention bias (an
embedding of each hit's domain exists as an option and is off by default). Outputs: for every hit, the
deepest rank the QUERY shares with that hit (0 ... 6); and one distribution over the taxonomy nodes at which
the query attaches.

Nothing here knows about horizontal transfer. A transfer shows up afterwards, as a gene that the model
places closer to some lineage than the genome it sits in (trials.py, score.py).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

K = 64                                   # hits per query
N_REL = 8                                # relation between two hits: shared depth 0..7
N_OUT = 7                                # relation between query and hit: shared depth 0..6
LEVELS = (6, 5, 4, 3, 2)                 # hide the query's own genus / family / order / class / phylum (and deeper)
CAPS = ((4, 4), (3, 8), (2, 16), (1, 32))  # (rank column, most tokens from one family / order / class / phylum)
N_FEAT = 8
N_SLOT = K * 6                           # candidate attachment nodes of one example: the six ancestors of every token
ROOT = N_SLOT                            # "attaches above every clade found among the tokens"
SINK = N_SLOT + 1                        # where the slots of missing tokens go
N_DOMAIN = 5                             # unknown, Bacteria, Archaea, Eukaryota, Viruses
DOMAIN_ID = {0: 0, 2: 1, 2157: 2, 2759: 3, 10239: 4}
COLS = ("bits", "pid", "qcov", "tcov", "loge", "lr")


def shared_depth(ranks_a, ranks_b):
    """Deepest rank at which two lineages carry the same defined node: 0 (none) ... 6 (genus).
    ranks_*: [..., 6] node ids (domain, phylum, class, order, family, genus), 0 = rank not defined. Checked
    rank by rank, so a missing intermediate rank does not break a match at a deeper one."""
    eq = (ranks_a == ranks_b) & (ranks_a != 0)
    return (eq * np.arange(1, ranks_a.shape[-1] + 1)).max(-1).astype(np.uint8)


def cumcount(g):
    """Running count of each value of g, in order of appearance: [a, b, a, a] -> [0, 0, 1, 2]."""
    order = np.argsort(g, kind="stable")
    gs = g[order]
    start = np.r_[0, np.flatnonzero(gs[1:] != gs[:-1]) + 1]
    run = np.arange(len(g)) - np.repeat(start, np.diff(np.r_[start, len(g)]))
    out = np.empty(len(g), dtype=np.int64)
    out[order] = run
    return out


class Library:
    """The reference proteomes: lineage of each, relation between any two, and clade sizes."""

    def __init__(self, path):
        z = np.load(path)
        self.taxid, self.ranks = z["taxid"], z["ranks"]
        P = self.P = len(self.taxid)
        self.pidx = {int(t): i for i, t in reversed(list(enumerate(self.taxid)))}
        self.S = np.zeros((P, P), dtype=np.uint8)            # shared depth; the diagonal is 7 (same proteome)
        self.node = np.zeros((P, 6), dtype=np.int64)         # dense clade index at each rank
        self.size = []                                       # proteomes per clade, per rank
        for r in range(6):
            col = self.ranks[:, r]
            self.S[(col[:, None] == col[None, :]) & (col[:, None] != 0)] = r + 1
            key = np.where(col != 0, col, -(np.arange(P) + 1))   # a lineage without this rank is its own clade
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            self.node[:, r] = inv
            self.size.append(cnt)
        np.fill_diagonal(self.S, 7)
        self.offset = np.concatenate([[0], np.cumsum([len(c) for c in self.size])[:-1]])   # makes clade indices unique across ranks
        self.domain = np.array([DOMAIN_ID.get(int(x), 0) for x in self.ranks[:, 0]])

    def relation_to(self, ranks):
        """Shared depth of one lineage (6 node ids) with every reference proteome."""
        return shared_depth(np.asarray(ranks)[None, :], self.ranks)


def make_example(lib, hp, d, qlen, hidden, list_limit, strict=True):
    """Tokens of one query when the proteomes flagged in `hidden` are not in the library.

    hp: proteome index of every stored hit, best score first; d: the stored hit columns (COLS).
    Selection (uses no label): best hit per proteome, then at most 4 per family, 8 per order, 16 per class and
    32 per phylum, best first, up to K. A densely sampled clade therefore cannot fill the list.

    Features describe only the alignment of the query with each selected hit. Nothing is computed from the
    hidden set or from counts over the stored list: an earlier version gave each token the size of its clades
    after hold-out and the share of each clade with a hit, and the clade that had shrunk was the query's own
    (the label), as a code review showed. Counts over a stored list that was cut off also differ from what a
    search of the reduced library returns.

    Returns (features [n, N_FEAT], proteome index of each token [n]), or None when there is no hit left or
    when the stored list was cut off before K tokens were found: a search of the reduced library would then
    return hits the stored list does not contain, and the tokens would not be the ones it selects
    (strict=False keeps such a query, for scoring a genome where every gene needs an answer)."""
    keep = np.flatnonzero(~hidden[hp])
    if len(keep) == 0:
        return None
    _, first = np.unique(hp[keep], return_index=True)
    idx = keep[np.sort(first)]                               # best hit per proteome, in score order
    for col, cap in CAPS:
        idx = idx[cumcount(lib.node[hp[idx], col]) < cap]
    idx = idx[:K]
    if strict and len(hp) >= list_limit and len(idx) < K:
        return None
    bits = d["bits"][idx]
    f = np.zeros((len(idx), N_FEAT), dtype=np.float32)
    f[:, 0], f[:, 1], f[:, 2], f[:, 3], f[:, 4] = bits / qlen, bits / bits[0], d["pid"][idx], d["qcov"][idx], d["tcov"][idx]
    f[:, 5], f[:, 6], f[:, 7] = d["loge"][idx] / 200.0, np.clip(d["lr"][idx], -2, 2), np.log(qlen) / 7.0
    return f, hp[idx]


class Block(nn.Module):
    def __init__(self, d, heads, dropout, use_attn, use_pair):
        super().__init__()
        self.heads, self.use_attn = heads, use_attn
        if use_attn:
            self.n1 = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.rel = nn.Embedding(N_REL, heads) if use_pair else None
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, rel, pad):
        if self.use_attn:
            B, n, d = x.shape
            q, k, v = self.qkv(self.n1(x)).view(B, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
            bias = torch.zeros(B, self.heads, n, n, device=x.device, dtype=x.dtype)
            if self.rel is not None:
                bias = bias + self.rel(rel).permute(0, 3, 1, 2)
            bias = bias.masked_fill(pad[:, None, None, :], float("-inf"))
            h = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
            x = x + self.drop(self.proj(h.transpose(1, 2).reshape(B, n, d)))
        return x + self.drop(self.ff(self.n2(x)))


class PlaceNet(nn.Module):
    def __init__(self, d=128, layers=4, heads=4, dropout=0.1, use_attn=True, use_pair=True, use_domain=False):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, dropout=dropout, use_attn=use_attn, use_pair=use_pair, use_domain=use_domain)
        self.inp = nn.Sequential(nn.Linear(N_FEAT, d), nn.GELU(), nn.Linear(d, d))
        self.dom = nn.Embedding(N_DOMAIN, d) if use_domain else None
        self.blocks = nn.ModuleList(Block(d, heads, dropout, use_attn, use_pair) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, N_OUT)
        self.slot = nn.Linear(d, 6)                          # "the query attaches at my ancestor of rank r"
        self.root = nn.Linear(d, 1)

    def forward(self, x, dom, rel, pad, nid):
        """x [B, K, F] hit features; dom [B, K] domain of each hit; rel [B, K, K] shared depth between hits;
        pad [B, K] True where there is no hit; nid [B, K, 6] index of each token's ancestor at each rank among
        the candidate nodes of its example.
        Returns per-hit logits [B, K, 7] (depth the query shares with the hit) and node logits [B, N_SLOT + 1]:
        one distribution over where the query attaches, the last entry being ROOT. A node's logit is the mean
        over the tokens below it, so the hits cannot contradict each other; absent nodes are -inf."""
        h = self.inp(x)
        if self.dom is not None:
            h = h + self.dom(dom)
        for b in self.blocks:
            h = b(h, rel, pad)
        h = self.norm(h)
        B = len(x)
        u = self.slot(h).reshape(B, -1)
        ix = nid.reshape(B, -1)
        tot = torch.zeros(B, SINK + 1, device=x.device, dtype=u.dtype).scatter_add(1, ix, u)
        cnt = torch.zeros(B, SINK + 1, device=x.device, dtype=u.dtype).scatter_add(1, ix, torch.ones_like(u))
        node = torch.where(cnt > 0, tot / cnt.clamp(min=1), torch.full_like(tot, float("-inf")))[:, :ROOT]
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        root = self.root((h * keep).sum(1) / keep.sum(1).clamp(min=1))
        return self.out(h), torch.cat([node, root], 1)


def pack(examples, lib):
    """examples: list of (features [n, F], proteome index [n]) -> padded arrays
    X [E, K, F] float16; P [E, K] int16 (-1 = no hit); NID [E, K, 6] int16: for each token and rank, the index of
    that ancestor among the distinct candidate nodes of the example (SINK where there is no token)."""
    X = np.zeros((len(examples), K, N_FEAT), dtype=np.float16)
    P = np.full((len(examples), K), -1, dtype=np.int16)
    NID = np.full((len(examples), K, 6), SINK, dtype=np.int16)
    for i, (f, tp) in enumerate(examples):
        n = len(tp)
        X[i, :n], P[i, :n] = f, tp
        NID[i, :n] = np.unique(lib.node[tp] + lib.offset[None, :], return_inverse=True)[1].reshape(n, 6)
    return X, P, NID


class Runner:
    """Holds the library tensors on the device and runs the model on packed examples."""

    def __init__(self, lib, device):
        self.dev = device
        self.S = torch.from_numpy(lib.S).to(device)
        self.dom = torch.from_numpy(lib.domain).to(device)

    def inputs(self, X, P, NID):
        p = P.long()
        pad = p < 0
        pc = p.clamp(min=0)
        return X.float(), self.dom[pc], self.S[pc[:, :, None], pc[:, None, :]].long(), pad, NID.long(), pc

    @torch.no_grad()
    def predict(self, model, X, P, NID, bs=2048):
        """Per-hit class probabilities [E, K, 7] and node logits [E, N_SLOT + 1] (numpy) for packed numpy examples."""
        hit, node = [], []
        for i in range(0, len(X), bs):
            x, dom, rel, pad, nid, _ = self.inputs(*(torch.from_numpy(v[i:i + bs]).to(self.dev) for v in (X, P, NID)))
            h, n = model(x, dom, rel, pad, nid)
            hit.append(h.softmax(-1).cpu().numpy()); node.append(n.cpu().numpy())
        if not hit:
            return np.zeros((0, K, N_OUT), dtype=np.float32), np.zeros((0, N_SLOT + 1), dtype=np.float32)
        return np.concatenate(hit), np.concatenate(node)


def discordance(prob, P, s):
    """Foreign-origin scores of packed examples against a genome.
    prob [E, K, 7] model output; P [E, K] token proteomes (-1 = none); s [E, K] depth the genome shares with each token.
      disc   max_i ( E[depth the protein shares with hit i] - s_i )
      p1/p2  max_i P(the protein shares at least 1 / 2 more ranks with hit i than the genome does)
      top    the excess for the best hit;  mean5  mean excess of the five best hits
    Also returns the expected depth per token (ev) and the excess per token (exc, -9 where there is no token)."""
    pad = P < 0
    ev = (prob * np.arange(N_OUT, dtype=np.float32)).sum(-1)
    exc = np.where(pad, -9.0, ev - s)
    cum = np.cumsum(prob[:, :, ::-1], -1)[:, :, ::-1]         # P(depth >= k)
    out = dict(disc=exc.max(1), top=exc[:, 0], ev=ev, exc=exc,
               mean5=np.array([exc[i, :min(5, int((~pad[i]).sum()))].mean() for i in range(len(exc))]))
    for name, add in (("p1", 1), ("p2", 2)):
        k = np.clip(s.astype(int) + add, 0, N_OUT - 1)
        pk = np.take_along_axis(cum, k[:, :, None], 2)[:, :, 0]
        out[name] = np.where(pad | (s + add > N_OUT - 1), 0.0, pk).max(1)
    return out


def _lse(x, mask):
    """log sum exp of x over the entries where mask is True, per row (-inf for an empty selection)."""
    y = np.where(mask, x, -np.inf)
    m = y.max(1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):
        return (m + np.log(np.exp(y - m).sum(1, keepdims=True)))[:, 0]


def offpath(node, NID, P, s):
    """Log odds that the query attaches at a node that is NOT an ancestor of the genome.
    node [E, N_SLOT + 1] logits; NID [E, K, 6]; P [E, K] (-1 = none); s [E, K] depth the genome shares with each token.
    The ancestor of token j at rank r contains the genome exactly when the genome shares at least that rank
    with token j. ROOT counts as on the genome's path: attaching above everything is "unresolved", never foreign."""
    E = len(node)
    on = np.zeros((E, SINK + 1), dtype=bool)
    slot_on = (s[:, :, None] >= np.arange(1, 7)[None, None, :]) & (P >= 0)[:, :, None]
    rows, cols = np.nonzero(slot_on.reshape(E, -1))
    on[rows, NID.reshape(E, -1).astype(np.int64)[rows, cols]] = True
    on = on[:, :SINK]
    on[:, ROOT] = True
    present = np.isfinite(node)
    return np.clip(_lse(node, present & ~on), -60, 60) - np.clip(_lse(node, present & on), -60, 60)


def load_model(path, device):
    """A trained checkpoint, or "untrained" for a randomly initialised model: the floor that the read-outs reach
    from the structure of the tokens alone (which lineages the hits come from), with nothing learned."""
    if path == "untrained":
        torch.manual_seed(0)
        return PlaceNet().to(device).eval()
    ck = torch.load(path, map_location=device, weights_only=False)
    m = PlaceNet(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()
