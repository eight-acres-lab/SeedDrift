"""Undated DTL (duplication-transfer-loss) parsimony reconciliation.

Dynamic programme of Bansal, Alm & Kellis (2012), vectorised over species with
numpy, plus all-rootings evaluation of unrooted gene trees in one pass (each
directed subtree is computed once).  Optimal scenarios are sampled with random
tie-breaking so that repeated sampling over a set of gene trees yields
ALE-like transfer frequencies f(donor -> recipient).
"""
from __future__ import annotations

import copy
import math
import random
import zlib

import numpy as np

from .tree import (contract_weak, has_support, parse_newick, prune_to, resolve_polytomies,
                   resolve_polytomies_guided, unrooted_adjacency)

INF = np.inf
EPS = 1e-7


class DTLModel:
    """DTL costs on a species tree.

    dated=True restricts transfers to pairs of branches that coexist in time
    (branch intervals from species-tree branch lengths, overlap tolerance `slack`
    as a fraction of tree height). Without it any two non-nested branches may
    exchange genes, which lets parsimony pick time-impossible directions.
    """

    def __init__(self, sp, dup=2.0, transfer=3.0, loss=1.0, dated=False, slack=0.02):
        self.sp = sp
        self.D, self.T, self.L = float(dup), float(transfer), float(loss)
        self.dated = dated
        depth = np.asarray(sp.depth, dtype=float)
        desc = sp.desc
        # desc_cost[s, x] = L * (#edges from s down to x) if x is in subtree(s)
        self.desc_cost = np.where(desc, self.L * (depth[None, :] - depth[:, None]), INF)
        self.incomp = ~(desc | desc.T)
        if dated:
            start, end = sp.branch_intervals()
            self.incomp &= (start[:, None] < end[None, :] + slack) & (start[None, :] < end[:, None] + slack)
        self.inc_cost = np.where(self.incomp, 0.0, INF)
        self.t_block = None             # species where no transfer may start (see forbid_clade)
        internal = [s for s in range(sp.n) if sp.children[s]]
        self.internal = np.array(internal, dtype=int)
        self.ch1 = np.array([sp.children[s][0] for s in internal], dtype=int)
        self.ch2 = np.array([sp.children[s][1] for s in internal], dtype=int)

    def forbid_clade(self, R):
        """Copy of the model in which no transfer may start or land on R or any branch below it."""
        m = copy.copy(self)
        m.inc_cost = self.inc_cost.copy()
        m.inc_cost[:, self.sp.desc[R]] = INF
        m.t_block = self.sp.desc[R].copy()
        return m

    def finish(self, c):
        inn = np.min(c[None, :] + self.desc_cost, axis=1)
        out = np.min(c[None, :] + self.inc_cost, axis=1)
        return c, inn, out

    def leaf(self, s):
        c = np.full(self.sp.n, INF)
        c[s] = 0.0
        return self.finish(c)

    def node_cost(self, A, B, allow_transfer=True):
        _, iA, oA = A
        _, iB, oB = B
        c = np.full(self.sp.n, INF)
        if len(self.internal):
            c[self.internal] = np.minimum(iA[self.ch1] + iB[self.ch2], iA[self.ch2] + iB[self.ch1])
        c = np.minimum(c, self.D + iA + iB)
        if allow_transfer:
            t = self.T + np.minimum(iA + oB, iB + oA)
            if self.t_block is not None:
                t = np.where(self.t_block, INF, t)
            c = np.minimum(c, t)
        return c

    def combine(self, A, B, allow_transfer=True):
        return self.finish(self.node_cost(A, B, allow_transfer))


class GeneTreeDTL:
    """All directed subtrees of an unrooted gene tree, reconciled once."""

    def __init__(self, model, adj, leaf_sp, allow_transfer=True):
        self.model = model
        self.adj = adj
        self.leaf_sp = leaf_sp
        self.allow = allow_transfer
        self.vec = {}
        self.kids = {}
        self._leaves = {}
        self._compute_all()
        self.root_cost = {}
        self.root_c = {}
        for a, b in self.edges:
            c = model.node_cost(self.vec[(a, b)], self.vec[(b, a)], allow_transfer)
            self.root_c[(a, b)] = c
            self.root_cost[(a, b)] = float(np.min(c))
        self.best_cost = min(self.root_cost.values()) if self.root_cost else 0.0

    # -- dynamic programme over directed edges ---------------------------
    def _compute_all(self):
        adj = self.adj
        nodes = [i for i in range(len(adj)) if adj[i]]
        self.edges = [(a, b) for a in nodes for b, _ in adj[a] if a < b]
        if not nodes:
            return
        r0 = next((i for i in nodes if len(adj[i]) >= 2), nodes[0])
        parent = {r0: -1}
        order = [r0]
        stack = [r0]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    stack.append(v)
        model, allow = self.model, self.allow
        for v in reversed(order):            # down keys (parent -> v)
            p = parent[v]
            if p < 0:
                continue
            self._set(p, v, model, allow)
        for v in order:                       # up keys (child -> v)
            for w, _ in adj[v]:
                if parent.get(w) == v:
                    self._set(w, v, model, allow)

    def _set(self, frm, v, model, allow):
        others = [w for w, _ in self.adj[v] if w != frm]
        if not others:
            self.vec[(frm, v)] = model.leaf(self.leaf_sp[v])
            self.kids[(frm, v)] = None
        elif len(others) == 2:
            A, B = (v, others[0]), (v, others[1])
            self.vec[(frm, v)] = model.combine(self.vec[A], self.vec[B], allow)
            self.kids[(frm, v)] = (A, B)
        else:
            raise ValueError("gene tree node with degree %d (resolve polytomies first)" % (len(others) + 1))

    def leaves(self, key):
        """Leaf node indices of directed subtree `key`."""
        if key in self._leaves:
            return self._leaves[key]
        stack, out = [key], []
        while stack:
            k = stack.pop()
            kd = self.kids[k]
            if kd is None:
                out.append(k[1])
            else:
                stack.extend(kd)
        res = frozenset(out)
        self._leaves[key] = res
        return res

    # -- sampling one optimal scenario ------------------------------------
    def _descend(self, key, s, rng):
        c, inn, _ = self.vec[key]
        sp, L = self.model.sp, self.model.L
        while True:
            opts = [-1] if abs(c[s] - inn[s]) < EPS else []
            for ch in sp.children[s]:
                if abs(L + inn[ch] - inn[s]) < EPS:
                    opts.append(ch)
            pick = rng.choice(opts)
            if pick < 0:
                return s
            s = pick

    def sample(self, rng, prefer_transfer=False):
        """One optimal reconciliation (random root among optima, random ties).

        prefer_transfer=True resolves ties in favour of transfers, so the scenario lists
        every transfer that is part of *some* optimal reconciliation (e.g. a sister-clade
        transfer that costs exactly the same as an ancient duplication + loss).
        """
        if not self.edges:
            return {"transfers": [], "dups": 0, "specs": []}
        best = self.best_cost
        root_edges = [e for e in self.edges if self.root_cost[e] <= best + EPS]
        a, b = rng.choice(root_edges)
        c_root = self.root_c[(a, b)]
        s0 = int(rng.choice(np.flatnonzero(c_root <= best + EPS)))
        model, sp = self.model, self.model.sp
        D, T = model.D, model.T
        transfers, dups, specs = [], 0, []
        stack = [((a, b), (b, a), s0, float(c_root[s0]))]
        while stack:
            A, B, s, total = stack.pop()
            _, iA, oA = self.vec[A]
            _, iB, oB = self.vec[B]
            choices = []
            if sp.children[s]:
                s1, s2 = sp.children[s]
                if abs(iA[s1] + iB[s2] - total) < EPS:
                    choices.append(("S", (A, s1), (B, s2)))
                if abs(iA[s2] + iB[s1] - total) < EPS:
                    choices.append(("S", (A, s2), (B, s1)))
            if abs(D + iA[s] + iB[s] - total) < EPS:
                choices.append(("D", (A, s), (B, s)))
            if self.allow:
                if abs(T + iA[s] + oB[s] - total) < EPS:
                    choices.append(("T", A, B))
                if abs(T + iB[s] + oA[s] - total) < EPS:
                    choices.append(("T", B, A))
            if not choices:
                raise RuntimeError("backtracking failed (numerical tie issue)")
            if prefer_transfer and any(c[0] == "T" for c in choices):
                choices = [c for c in choices if c[0] == "T"]
            ev = rng.choice(choices)
            if ev[0] == "T":
                stay, moved = ev[1], ev[2]
                x_stay = self._descend(stay, s, rng)
                cm, _, om = self.vec[moved]
                cand = np.flatnonzero(model.incomp[s] & (np.abs(cm - om[s]) < EPS))
                x = int(rng.choice(cand))
                transfers.append({"donor": s, "recipient": x, "moved": moved, "stay": stay})
                targets = [(stay, x_stay), (moved, x)]
            else:
                if ev[0] == "D":
                    dups += 1
                else:   # speciation at s: (subtree going to children[s][0], subtree going to [1])
                    k0, k1 = (ev[1][0], ev[2][0]) if ev[1][1] == sp.children[s][0] else (ev[2][0], ev[1][0])
                    specs.append((s, k0, k1))
                targets = [(ev[1][0], self._descend(ev[1][0], ev[1][1], rng)),
                           (ev[2][0], self._descend(ev[2][0], ev[2][1], rng))]
            for key, x in targets:
                kd = self.kids[key]
                if kd is not None:
                    stack.append((kd[0], kd[1], x, float(self.vec[key][0][x])))
        return {"transfers": transfers, "dups": dups, "specs": specs}


# ----------------------------------------------------------------------------
# Distances (for the recency feature)
# ----------------------------------------------------------------------------

def tree_distances(adj, src):
    dist = {src: 0.0}
    stack = [src]
    while stack:
        u = stack.pop()
        du = dist[u]
        for v, L in adj[u]:
            if v not in dist:
                dist[v] = du + L
                stack.append(v)
    return dist


def prepare_tree(txt, gene_sp, rng, sp=None, collapse=0.0):
    """Parse, prune to known genes, resolve polytomies. Returns (adj, names) or None.

    collapse > 0: contract branches with support < collapse (or ~zero length) and resolve the
    resulting multifurcations along the species tree, so only supported incongruence remains.
    """
    try:
        root = parse_newick(txt)
    except ValueError:
        return None
    root = prune_to(root, gene_sp.keys())
    if root is None or len(root.leaves()) < 2:
        return None
    if collapse > 0 and sp is not None:
        contract_weak(root, collapse)
        resolve_polytomies_guided(root, gene_sp.__getitem__, sp, rng, root_max=3)
    else:
        resolve_polytomies(root, rng, root_max=3)
    return unrooted_adjacency(root)


def split_paralogs(family, txt, gene_sp, model, min_overlap=2, collapse=0.0, seed=0, min_copies=1.5):
    """Split a multi-copy family into orthologous subfamilies at ancient duplications.

    The gene tree is rooted where duplication + loss cost is lowest; a node whose two subtrees
    share >= min_overlap species is an old duplication (species overlap), and its subtrees
    become separate groups. OrthoFinder orthogroups often lump ancient paralogs (transporter or
    regulator superfamilies, integrases); reconciling them as one family turns paralog mixing
    into confident deep "transfers". Overlap 1 (a single misplaced or extra copy, e.g. an
    additive xenolog into one genome) is not cut, nor is an overlap covering less than half of
    the smaller subtree's species. Only paralog-rich families (mean copies per genome >=
    min_copies) are split: cutting also removes the native copy that tells the direction of an
    additive transfer into a clade. Returns a list of gene-name lists (one list = no split).
    """
    names_all = sorted(gene_sp)
    counts = {}
    for g in names_all:
        counts[gene_sp[g]] = counts.get(gene_sp[g], 0) + 1
    if min_overlap <= 0 or max(counts.values()) < 2 or len(names_all) < min_copies * len(counts):
        return [names_all]
    rng = family_rng(family, seed)
    prep = prepare_tree(txt, gene_sp, rng, model.sp, collapse if _tree_has_support(txt) else 0.0)
    if prep is None:
        return [names_all]
    adj, names = prep
    leaf_sp = {i: gene_sp[nm] for i, nm in enumerate(names) if nm is not None}
    rec = GeneTreeDTL(model, adj, leaf_sp, allow_transfer=False)
    if not rec.edges:
        return [names_all]
    best = min(rec.root_cost.values())
    a, b = next(e for e in rec.edges if rec.root_cost[e] <= best + EPS)

    def species(key):
        return set(leaf_sp[i] for i in rec.leaves(key))

    def walk(key):                       # -> (leaves still attached upward, closed groups)
        kd = rec.kids[key]
        if kd is None:
            return [key[1]], []
        (oA, cA), (oB, cB) = walk(kd[0]), walk(kd[1])
        return _join(kd[0], kd[1], oA, cA, oB, cB)

    def _join(A, B, oA, cA, oB, cB):
        sa, sb = species(A), species(B)
        ov = len(sa & sb)
        if ov >= min_overlap and 2 * ov >= min(len(sa), len(sb)):
            return [], cA + cB + [x for x in (oA, oB) if x]
        return oA + oB, cA + cB

    (oA, cA), (oB, cB) = walk((a, b)), walk((b, a))
    top, closed = _join((a, b), (b, a), oA, cA, oB, cB)
    groups = closed + ([top] if top else [])
    out = [sorted(names[i] for i in grp) for grp in groups]
    return sorted(out, key=lambda x: (-len(x), x[0])) if len(out) > 1 else [names_all]


def _tree_has_support(txt):
    try:
        return has_support(parse_newick(txt))
    except ValueError:
        return False


def family_rng(family, seed):
    return random.Random((zlib.crc32(family.encode()) * 2654435761 + seed) & 0xFFFFFFFF)


def reconcile_family(family, trees, gene_sp, model, n_bt=3, seed=0, bg=None, collapse=0.0):
    """Reconcile every gene tree sample of one family; aggregate transfer events.

    Returns a dict with per-recipient aggregates:
      events[R] = {runs (random scenarios with a transfer into R), opt (trees whose
                   transfer-preferring scenario has one), occ, donors{d: count},
                   genes{gid: count}, rec{d: [log ratio]}}
    """
    sp = model.sp
    rng = family_rng(family, seed)
    out = {"family": family, "runs": 0, "n_trees": 0, "cost_dtl": None, "cost_dl": None,
           "rate": 1.0, "events": {}, "clock": [], "alt_cost": {}}
    first = None
    events = out["events"]
    if collapse > 0 and trees and _tree_has_support(trees[0]):
        # The ML tree's branch supports (e.g. UFBoot from 1000 replicates) already summarise the
        # bootstrap sample: reuse the collapsed ML tree, with fresh random tie-breaks each time
        # (>= 11 replicates, so f_T has the same resolution with or without bootstrap trees).
        trees = [trees[0]] * max(len(trees), 11)
    else:
        collapse = 0.0
    for txt in trees:
        prep = prepare_tree(txt, gene_sp, rng, sp, collapse)
        if prep is None:
            continue
        adj, names = prep
        leaf_sp = {i: gene_sp[nm] for i, nm in enumerate(names) if nm is not None}
        rec = GeneTreeDTL(model, adj, leaf_sp)
        has_len = bool(bg) and any(L > 0 for nb in adj for _, L in nb)
        dcache = {}

        def dist_from(i):
            if i not in dcache:
                dcache[i] = tree_distances(adj, i)
            return dcache[i]

        if out["n_trees"] == 0:
            first = (adj, leaf_sp)
            out["cost_dtl"] = rec.best_cost
            out["cost_dl"] = GeneTreeDTL(model, adj, leaf_sp, allow_transfer=False).best_cost
            if has_len:
                out["rate"] = _family_rate(leaf_sp, dist_from, bg, rng)
        out["n_trees"] += 1
        # n_bt random optimal scenarios (-> f_T) + one transfer-preferring scenario (-> f_opt)
        for k in range(n_bt + 1):
            prefer = k == n_bt
            scen = rec.sample(rng, prefer_transfer=prefer)
            if not prefer:
                out["runs"] += 1
            if k == 0 and out["n_trees"] == 1 and has_len:
                out["clock"] = _clock_records(rec, scen["specs"], names, leaf_sp, sp, dist_from,
                                              bg, out["rate"], rng)
            seen = set()
            for ev in scen["transfers"]:
                R, d = ev["recipient"], ev["donor"]
                genes = [names[i] for i in rec.leaves(ev["moved"]) if sp.desc[R, leaf_sp[i]]]
                if not genes:
                    continue
                e = events.get(R)
                if e is None:
                    e = events[R] = {"runs": 0, "opt": 0, "occ": 0, "donors": {}, "genes": {}, "rec": {}}
                if R not in seen:
                    e["opt" if prefer else "runs"] += 1
                    e["occ"] += 1
                    seen.add(R)
                e["donors"][d] = e["donors"].get(d, 0) + 1
                for g in genes:
                    e["genes"][g] = e["genes"].get(g, 0) + 1
                if has_len:
                    lr = _log_ratio(rec, ev, leaf_sp, dist_from, bg, out["rate"])
                    if lr is not None:
                        e["rec"].setdefault(d, []).append(lr)
    # per-event alternative: how much more does the ML tree cost if no transfer may start or land
    # in R's clade (both directions, else the reverse transfer is free)? ~0-1 = duplication +
    # losses explains the incongruence about as well (hidden paralogy)
    if first is not None and events:
        adj0, ls0 = first
        for R in events:
            out["alt_cost"][R] = GeneTreeDTL(model.forbid_clade(R), adj0, ls0).best_cost - out["cost_dtl"]
    return out


def _pair(a, b):
    return (a, b) if a < b else (b, a)


def _clock_records(rec, specs, names, leaf_sp, sp, dist_from, bg, rate, rng, max_pairs=150):
    """Molecular-clock check of every speciation node in one reconciliation.

    For a speciation at species node s, compare the gene distances across the split with
    the genome-wide background for the same species pairs (scaled by the family rate):
    log(median ratio) ~ 0 for vertical descent, < 0 when the two lineages exchanged the
    gene after they split (a transfer that leaves the topology congruent).
    Returns [(s, log_ratio, genes_under_child0, genes_under_child1)].
    """
    raw = []
    present = set(leaf_sp.values())
    for s, k0, k1 in specs:
        c0, c1 = sp.children[s]
        L0 = [i for i in rec.leaves(k0) if sp.desc[c0, leaf_sp[i]]]
        L1 = [i for i in rec.leaves(k1) if sp.desc[c1, leaf_sp[i]]]
        if not L0 or not L1:
            continue
        # locally congruent nodes only: each side must hold the family's genes of every species
        # below it. A gene-tree topology error that moves a subtree next to a distant one makes
        # their path short (a fake young split) but leaves species of that clade elsewhere.
        if (set(leaf_sp[i] for i in L0) != set(x for x in present if sp.desc[c0, x]) or
                set(leaf_sp[i] for i in L1) != set(x for x in present if sp.desc[c1, x])):
            continue
        pairs = [(i, j) for i in L0 for j in L1]
        if len(pairs) > max_pairs:
            pairs = rng.sample(pairs, max_pairs)
        vals = []
        for i, j in pairs:
            ref = bg.get(_pair(leaf_sp[i], leaf_sp[j]))
            if ref:
                vals.append(dist_from(i)[j] / ref)
        if not vals:
            continue
        vals.sort()
        # upper quartile: a real exchange makes (nearly) all cross pairs young, whereas a local
        # topology error in the gene tree shortens only a few of them
        raw.append((s, vals[(3 * len(vals)) // 4], [names[i] for i in L0], [names[j] for j in L1]))
    # Family rate from its own orthologous (speciation) comparisons: random leaf pairs would
    # include old paralog pairs, inflate the rate and make every speciation look young.
    if len(raw) >= 3:
        meds = sorted(r[1] for r in raw)
        rate = meds[len(meds) // 2]
    if rate <= 1e-9:
        # identical sequences across most splits (zero-length gene tree): no clock signal
        return []
    return [(s, math.log(max(m / rate, 1e-6)), g0, g1) for s, m, g0, g1 in raw]


def _family_rate(leaf_sp, dist_from, bg, rng, n_pairs=100):
    leaves = list(leaf_sp)
    ratios = []
    for _ in range(n_pairs * 3):
        if len(ratios) >= n_pairs or len(leaves) < 2:
            break
        i, j = rng.sample(leaves, 2)
        si, sj = leaf_sp[i], leaf_sp[j]
        ref = bg.get(_pair(si, sj)) if si != sj else None
        if ref:
            ratios.append(dist_from(i)[j] / ref)
    if not ratios:
        return 1.0
    ratios.sort()
    return min(20.0, max(0.05, ratios[len(ratios) // 2]))


def _log_ratio(rec, ev, leaf_sp, dist_from, bg, rate):
    moved = sorted(rec.leaves(ev["moved"]))[:20]
    stay = rec.leaves(ev["stay"])
    best = None
    for i in moved:
        di = dist_from(i)
        for j in stay:
            if leaf_sp[i] == leaf_sp[j]:
                continue
            if best is None or di[j] < best[0]:
                best = (di[j], i, j)
    if best is None:
        return None
    d, i, j = best
    ref = bg.get(_pair(leaf_sp[i], leaf_sp[j]))
    if not ref:
        return None
    return math.log(max(d, 1e-6) / (ref * rate))


def background_distances(tasks, sp, max_families=300, seed=0):
    """Median gene-tree distance per species pair, from single-copy families."""
    rng = random.Random(seed)
    pool = []
    for fam, trees, gene_sp in tasks:
        counts = {}
        for s in gene_sp.values():
            counts[s] = counts.get(s, 0) + 1
        if len(counts) >= 4 and max(counts.values()) == 1 and trees:
            pool.append((fam, trees[0], gene_sp))
    rng.shuffle(pool)
    acc = {}
    for fam, txt, gene_sp in pool[:max_families]:
        prep = prepare_tree(txt, gene_sp, rng)
        if prep is None:
            continue
        adj, names = prep
        if not any(L > 0 for nb in adj for _, L in nb):
            continue
        leaves = [i for i, nm in enumerate(names) if nm is not None]
        for x, i in enumerate(leaves):
            di = tree_distances(adj, i)
            for j in leaves[x + 1:]:
                acc.setdefault(_pair(gene_sp[names[i]], gene_sp[names[j]]), []).append(di[j])
    bg = {}
    for k, v in acc.items():
        v.sort()
        med = v[len(v) // 2]
        if med > 0:
            bg[k] = med
    return bg
