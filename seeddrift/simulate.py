"""Toy genome simulator with known HGT events (prokaryote / eukaryote / mixed).

Forward-in-time, time-consistent simulation along a Yule species tree:
gene duplication, loss, transfer (with co-transferred blocks and attached mobile
elements), transfers from unsampled lineages (new families: the donor is not in the
dataset), de novo / spurious ORFs, rearrangements, background mobile-element turnover,
(eukaryotes) contaminant contigs, GC / GC3 composition with amelioration, and orthogroup
splits (a clade's copies get their own family id, as OrthoFinder does with fast-evolving
orthologs). Gene-tree estimation error is mimicked with random NNIs.
This is a test harness, not a substitute for Zombi / SimPhy.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random

from .tree import Node, SpeciesTree, copy_tree, prune_to, random_nni, to_newick, unroot

PARAMS = {
    "prok": dict(dup=0.05, loss=0.15, transfer=0.06, rearr=60.0, rearr_len=5.0, me_ins=10.0,
                 me_loss=1.0, me_with_hgt=0.6, block_p=0.5, block_max=5, n_contigs=1,
                 gene_len=(600, 1800), gap=(20, 250), me_len=(800, 1600),
                 hgt_me=["integrase", "transposase", "IS", "phage"],
                 bg_me=["transposase", "IS", "tRNA", "phage", "recombinase", "integrase"],
                 ghost=12.0, orphan=4.0, ghost_len=(300, 1500), orphan_len=(100, 700), split_p=0.04,
                 del_block=0.5, foreign_loss=2.0, foreign_young=0.3, anc_paralogs=0.12,
                 gc_sd=0.025, gc3_sd=0.05, tau=0.4, tau3=0.25),
    "euk": dict(dup=0.08, loss=0.12, transfer=0.02, rearr=150.0, rearr_len=8.0, me_ins=40.0,
                me_loss=0.5, me_with_hgt=0.4, block_p=0.85, block_max=3, n_contigs=4,
                gene_len=(1500, 12000), gap=(1000, 15000), me_len=(300, 6000),
                hgt_me=["TE", "LTR", "DNA"],
                bg_me=["TE", "LTR", "LINE", "SINE", "DNA", "helitron"],
                ghost=1.5, orphan=6.0, ghost_len=(900, 9000), orphan_len=(300, 3000), split_p=0.04,
                del_block=0.6, foreign_loss=2.0, foreign_young=0.3, anc_paralogs=0.15,
                gc_sd=0.04, gc3_sd=0.08, tau=0.6, tau3=0.4),
}


def _poisson(lam, rng):
    if lam <= 0:
        return 0
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


def yule_tree(n, rng):
    lineages = [Node() for _ in range(n)]
    height = {id(x): 0.0 for x in lineages}
    t = 0.0
    while len(lineages) > 1:
        t += rng.expovariate(len(lineages))
        a, b = rng.sample(lineages, 2)
        lineages.remove(a)
        lineages.remove(b)
        p = Node()
        p.add(a)
        p.add(b)
        height[id(p)] = t
        lineages.append(p)
    root = lineages[0]
    H = height[id(root)]
    for nd in root.preorder():
        if nd.parent is not None:
            nd.length = (height[id(nd.parent)] - height[id(nd)]) / H
    return root


class Genome:
    def __init__(self, contigs, domain):
        self.contigs = contigs          # list of [name, [elements]]; int = gene lineage, str = ME type
        self.domain = domain
        self.n_genes = sum(1 for _, els in contigs for e in els if isinstance(e, int))
        self.n_me = sum(len(els) for _, els in contigs) - self.n_genes

    def pick(self, rng, gene):
        if (gene and self.n_genes == 0) or (not gene and self.n_me == 0):
            return None
        lens = [len(els) for _, els in self.contigs]
        tot = sum(lens)
        for _ in range(500):
            r = rng.randrange(tot)
            ci = 0
            while r >= lens[ci]:
                r -= lens[ci]
                ci += 1
            if isinstance(self.contigs[ci][1][r], int) == gene:
                return ci, r
        return None

    def insert_pos(self, rng):
        lens = [len(els) + 1 for _, els in self.contigs]
        r = rng.randrange(sum(lens))
        ci = 0
        while r >= lens[ci]:
            r -= lens[ci]
            ci += 1
        return ci, r


def _chunk(elements, domain):
    k = PARAMS[domain]["n_contigs"]
    if k == 1:
        return [["chr1", list(elements)]]
    size = int(math.ceil(len(elements) / float(k)))
    return [["chr%d" % (i + 1), list(elements[i * size:(i + 1) * size])] for i in range(k)]


GC3_SCALE = 1.8      # GC3 departs from 0.5 (and from the genome mean) more than total GC


def _clip(x, lo, hi):
    return min(hi, max(lo, x))


class Sim:
    def __init__(self, sp, dom, times, rng, mu=None):
        self.sp, self.dom, self.times, self.rng = sp, dom, times, rng
        self.mu = mu or [0.5] * sp.n         # genome GC per species-tree branch
        self.par, self.birth, self.end, self.kids, self.fam = [], [], [], [], []
        self.comp = []                       # per lineage: (GC deviation from host genome, since t0)
        self.fam_root = {}                   # family -> its root lineage
        self.n_fam = 0
        self.ghost_fam, self.orphan_fam = set(), set()
        self.foreign = {}                    # lineage -> time it was acquired by transfer
        self.transfers = []

    def new(self, parent, t, fam, comp=None):
        i = len(self.par)
        self.par.append(parent)
        self.birth.append(t)
        self.end.append(None)
        self.kids.append([])
        self.fam.append(fam)
        self.comp.append(comp if comp is not None else (self.comp[parent] if parent >= 0 else (0.0, t)))
        if parent >= 0:
            self.kids[parent].append(i)
            self.end[parent] = t
            if parent in self.foreign:
                self.foreign[i] = self.foreign[parent]
        else:
            self.fam_root[fam] = i
            self.n_fam = max(self.n_fam, fam + 1)
        return i

    def deviation(self, lin, t, tau):
        """GC deviation of a lineage at time t: foreign composition ameliorates exponentially."""
        d0, t0 = self.comp[lin]
        return d0 if t0 is None else d0 * math.exp(-(t - t0) / tau)

    def split(self, lin, t):
        return self.new(lin, t, self.fam[lin]), self.new(lin, t, self.fam[lin])

    def speciate(self, node, genome, t, alive):
        c1, c2 = self.sp.children[node]
        A, B = [], []
        for name, els in genome.contigs:
            ea, eb = [], []
            for e in els:
                if isinstance(e, int):
                    a, b = self.split(e, t)
                    ea.append(a)
                    eb.append(b)
                else:
                    ea.append(e)
                    eb.append(e)
            A.append([name, ea])
            B.append([name, eb])
        for c, contigs in ((c1, A), (c2, B)):
            d = self.dom[c]
            if d != genome.domain:
                contigs = _chunk([e for _, els in contigs for e in els], d)
            alive[c] = Genome(contigs, d)

    # -- events -------------------------------------------------------------
    def ev_dup(self, b, g, t, alive):
        pos = g.pick(self.rng, True)
        if pos:
            els = g.contigs[pos[0]][1]
            x, y = self.split(els[pos[1]], t)
            els[pos[1]] = x
            els.insert(pos[1] + 1, y)
            g.n_genes += 1

    def ev_loss(self, b, g, t, alive):
        """Deletion of a run of adjacent genes (operons are lost together, as in real genomes)."""
        rng = self.rng
        pos = g.pick(rng, True)
        if not pos:
            return
        k = 1
        while k < 6 and rng.random() < PARAMS[g.domain]["del_block"]:
            k += 1
        els = g.contigs[pos[0]][1]
        i, n = pos[1], 0
        while i < len(els) and n < k:
            if isinstance(els[i], int):
                self.end[els[i]] = t
                g.n_genes -= 1
                n += 1
            else:
                g.n_me -= 1
            del els[i]

    def ev_loss_foreign(self, b, g, t, alive):
        """Extra turnover of recently acquired genes (most new acquisitions are lost again)."""
        pos = g.pick(self.rng, True)
        if not pos:
            return
        els = g.contigs[pos[0]][1]
        lin = els[pos[1]]
        if self.foreign.get(lin) is not None and t - self.foreign[lin] < PARAMS[g.domain]["foreign_young"]:
            self.end[lin] = t
            del els[pos[1]]
            g.n_genes -= 1

    def ev_transfer(self, b, g, t, alive):
        rng = self.rng
        others = [x for x in alive if x != b]
        pos = g.pick(rng, True)
        if not others or not pos:
            return
        r_node = rng.choice(others)
        rg = alive[r_node]
        P = PARAMS[g.domain]
        k = 1
        while k < P["block_max"] and rng.random() > P["block_p"]:
            k += 1
        els = g.contigs[pos[0]][1]
        idxs, i = [], pos[1]
        while i < len(els) and len(idxs) < k:
            if isinstance(els[i], int):
                idxs.append(i)
            i += 1
        moved = []
        for i in idxs:
            stay, mv = self.split(els[i], t)
            els[i] = stay
            dev = self.deviation(mv, t, P["tau"])
            self.comp[mv] = (self.mu[b] + dev - self.mu[r_node], t)
            self.foreign[mv] = t
            moved.append(mv)
            self.transfers.append({"family": self.fam[mv], "donor": b, "recipient": r_node,
                                   "time": t, "lineage": mv})
        RP = PARAMS[rg.domain]
        block = list(moved)
        if rng.random() < RP["me_with_hgt"]:
            te = rng.choice(RP["hgt_me"])
            u = rng.random()
            block = [te] + block + [te] if u < 0.3 else ([te] + block if u < 0.65 else block + [te])
        cj, p = rg.insert_pos(rng)
        rg.contigs[cj][1][p:p] = block
        rg.n_genes += len(moved)
        rg.n_me += len(block) - len(moved)

    def _insert_block(self, g, genes, with_me):
        rng, P = self.rng, PARAMS[g.domain]
        block = list(genes)
        if with_me and rng.random() < P["me_with_hgt"]:
            te = rng.choice(P["hgt_me"])
            u = rng.random()
            block = [te] + block + [te] if u < 0.3 else ([te] + block if u < 0.65 else block + [te])
        cj, p = g.insert_pos(rng)
        g.contigs[cj][1][p:p] = block
        g.n_genes += len(genes)
        g.n_me += len(block) - len(genes)

    def ev_ghost(self, b, g, t, alive):
        """Transfer from a lineage outside the dataset: a block of genes of new families."""
        rng, P = self.rng, PARAMS[g.domain]
        k = 1
        while k < P["block_max"] and rng.random() > P["block_p"]:
            k += 1
        off = rng.uniform(0.28, 0.72) - self.mu[b]      # donor genome GC - recipient GC
        genes = []
        for _ in range(k):
            fam = self.n_fam
            lin = self.new(-1, t, fam, comp=(off + rng.gauss(0.0, 0.01), t))
            self.foreign[lin] = t
            self.ghost_fam.add(fam)
            self.transfers.append({"family": fam, "donor": None, "recipient": b, "time": t, "lineage": lin})
            genes.append(lin)
        self._insert_block(g, genes, True)

    def ev_orphan(self, b, g, t, alive):
        """De novo / spurious ORF: a new single-gene family, not transferred (short, AT-rich)."""
        fam = self.n_fam
        lin = self.new(-1, t, fam, comp=(-0.03, None))
        self.orphan_fam.add(fam)
        self._insert_block(g, [lin], False)

    def ev_rearr(self, b, g, t, alive):
        rng = self.rng
        P = PARAMS[g.domain]
        lens = [len(els) for _, els in g.contigs]
        if sum(lens) < 4:
            return
        r = rng.randrange(sum(lens))
        ci = 0
        while r >= lens[ci]:
            r -= lens[ci]
            ci += 1
        els = g.contigs[ci][1]
        n = len(els)
        if n < 4:
            return
        L = min(n - 1, 1 + int(rng.expovariate(1.0 / P["rearr_len"])))
        s = rng.randrange(n - L + 1)
        seg = els[s:s + L]
        if rng.random() < 0.5:
            els[s:s + L] = seg[::-1]
        else:
            del els[s:s + L]
            cj = rng.randrange(len(g.contigs)) if (len(g.contigs) > 1 and rng.random() < 0.3) else ci
            tgt = g.contigs[cj][1]
            p = rng.randint(0, len(tgt))
            tgt[p:p] = seg

    def ev_me_ins(self, b, g, t, alive):
        cj, p = g.insert_pos(self.rng)
        g.contigs[cj][1].insert(p, self.rng.choice(PARAMS[g.domain]["bg_me"]))
        g.n_me += 1

    def ev_me_loss(self, b, g, t, alive):
        pos = g.pick(self.rng, False)
        if pos:
            del g.contigs[pos[0]][1][pos[1]]
            g.n_me -= 1

    def run(self, n_families):
        sp, rng = self.sp, self.rng
        order = list(range(n_families))
        rng.shuffle(order)
        # some families carry ancient paralogs from duplications before the root: with differential
        # loss afterwards they give the hidden paralogy that OrthoFinder superfamilies show
        P0 = PARAMS[self.dom[sp.root]]
        copies = []
        for f in order:
            lin = self.new(-1, 0.0 if rng.random() >= P0["anc_paralogs"] else -0.6, f)
            if self.birth[lin] < 0:
                a, b = self.split(lin, -0.35)
                grp = [a, b]
                if rng.random() < 0.4:
                    c, d = self.split(grp.pop(), -0.15)
                    grp += [c, d]
                copies.extend(grp)
            else:
                copies.append(lin)
        rng.shuffle(copies)
        root_genome = Genome(_chunk(copies, self.dom[sp.root]), self.dom[sp.root])
        alive = {}
        self.speciate(sp.root, root_genome, 0.0, alive)
        handlers = [self.ev_dup, self.ev_loss, self.ev_transfer, self.ev_rearr, self.ev_me_ins, self.ev_me_loss,
                    self.ev_ghost, self.ev_orphan, self.ev_loss_foreign]
        t = 0.0
        while True:
            next_t = min(self.times[b] for b in alive)
            rates, total = [], 0.0
            for b, g in alive.items():
                P = PARAMS[g.domain]
                r = [g.n_genes * P["dup"], g.n_genes * P["loss"] * (1.0 - P["del_block"]),
                     g.n_genes * P["transfer"] if len(alive) > 1 else 0.0,
                     P["rearr"], P["me_ins"], g.n_me * P["me_loss"], P["ghost"], P["orphan"],
                     g.n_genes * P["foreign_loss"]]
                rates.append((b, r))
                total += sum(r)
            dt = rng.expovariate(total) if total > 0 else float("inf")
            if t + dt >= next_t:
                t = next_t
                if all(sp.is_leaf[b] for b in alive):
                    break
                for b in [x for x in alive if self.times[x] <= t + 1e-12 and not sp.is_leaf[x]]:
                    self.speciate(b, alive.pop(b), t, alive)
                continue
            t += dt
            u = rng.random() * total
            for b, r in rates:
                if u < sum(r):
                    break
                u -= sum(r)
            k = 0
            while k < len(r) - 1 and u >= r[k]:
                u -= r[k]
                k += 1
            handlers[k](b, alive[b], t, alive)
        return alive


def simulate(out, domain="prok", n_species=16, n_families=600, seed=1, n_samples=10,
             noise=0.04, contaminants=8):
    rng = random.Random(seed)
    if domain not in ("prok", "euk", "mixed"):
        raise ValueError("domain must be prok, euk or mixed")
    root = yule_tree(n_species, rng)
    # name leaves by clade/domain, internal nodes N1..Nk
    clade_dom = {}
    for ci, ch in enumerate(root.children):
        d = ("prok" if ci == 0 else "euk") if domain == "mixed" else domain
        for nd in ch.preorder():
            clade_dom[id(nd)] = d
    clade_dom[id(root)] = "prok" if domain == "mixed" else domain
    counters, k = {}, 0
    for nd in root.preorder():
        if nd.children:
            k += 1
            nd.name = "N%d" % k
        else:
            d = clade_dom[id(nd)]
            prefix = {"prok": "P", "euk": "E"}[d] if domain == "mixed" else "S"
            counters[prefix] = counters.get(prefix, 0) + 1
            nd.name = "%s%02d" % (prefix, counters[prefix])
    leaf_dom = {nd.name: clade_dom[id(nd)] for nd in root.leaves()}
    sp = SpeciesTree(root)
    dom = [None] * sp.n
    times = [0.0] * sp.n
    for i in range(sp.n - 1, -1, -1):
        p = sp.parent[i]
        times[i] = 0.0 if p < 0 else times[p] + (sp.lengths[i] or 0.0)
        if sp.is_leaf[i]:
            times[i] = 1.0
            dom[i] = leaf_dom[sp.names[i]]
    for i in range(sp.n):                       # internal nodes: domain of their leaves
        if not sp.is_leaf[i]:
            ds = set(leaf_dom[nm] for nm in sp.leafsets[i])
            dom[i] = ds.pop() if len(ds) == 1 else "prok"

    # genome GC drifts along the species tree
    mu = [0.5] * sp.n
    for i in range(sp.n - 1, -1, -1):
        p = sp.parent[i]
        mu[i] = rng.uniform(0.42, 0.58) if p < 0 else _clip(
            mu[p] + rng.gauss(0.0, 0.05 * math.sqrt(sp.lengths[i] or 0.0)), 0.25, 0.75)

    sim = Sim(sp, dom, times, rng, mu)
    alive = sim.run(n_families)

    # contaminant contigs in eukaryotic genomes (copies of a distant genome's segment)
    contam_lin = {}
    euk_leaves = [b for b in alive if dom[b] == "euk"]
    if contaminants and euk_leaves:
        targets = rng.sample(euk_leaves, max(1, len(euk_leaves) // 8))
        top = sp.children[sp.root]
        for ti, tgt in enumerate(targets):
            side = top[0] if sp.desc[top[0], tgt] else top[1]
            pool = [b for b in alive if not sp.desc[side, b]] or [b for b in alive if b != tgt]
            src = rng.choice(pool)
            contigs = [c for c in alive[src].contigs if sum(isinstance(e, int) for e in c[1]) >= contaminants]
            if not contigs:
                continue
            els = rng.choice(contigs)[1]
            gene_pos = [i for i, e in enumerate(els) if isinstance(e, int)]
            s = rng.randrange(len(gene_pos) - contaminants + 1)
            new = []
            for i in gene_pos[s:s + contaminants]:
                keep, con = sim.split(els[i], 1.0 - 0.002)
                els[i] = keep
                new.append(con)
                contam_lin[con] = sp.names[src]
            alive[tgt].contigs.append(["contam_%d" % (ti + 1), new])

    # label extant genes, write genome tables
    os.makedirs(os.path.join(out, "gene_trees"), exist_ok=True)
    os.makedirs(os.path.join(out, "gene_trees_true"), exist_ok=True)
    extant, extant_leaf = {}, {}
    gene_rows, me_rows, row_lin, comp_rows = [], [], [], []
    for b in sorted(alive, key=lambda x: sp.names[x]):
        g, name = alive[b], sp.names[b]
        P = PARAMS[g.domain]
        k = 0
        for cname, els in g.contigs:
            pos = 1
            for e in els:
                if isinstance(e, int):
                    k += 1
                    gid = "%s_%05d" % (name, k)
                    extant[e], extant_leaf[e] = gid, b
                    sim.end[e] = 1.0
                    L = rng.randint(*(P["orphan_len"] if sim.fam[e] in sim.orphan_fam else
                                      P["ghost_len"] if sim.fam[e] in sim.ghost_fam else P["gene_len"]))
                    gene_rows.append([gid, name, "F%04d" % sim.fam[e], cname, pos, pos + L - 1,
                                      rng.choice("+-")])
                    row_lin.append(e)
                    host = sp.index[contam_lin[e]] if e in contam_lin else b
                    PH = PARAMS[dom[host]]
                    dev = sim.deviation(e, 1.0, PH["tau"])
                    dev3 = GC3_SCALE * sim.deviation(e, 1.0, PH["tau3"])
                    mu3 = _clip(0.5 + GC3_SCALE * (mu[host] - 0.5), 0.1, 0.95)
                    comp_rows.append([gid, "%.4f" % _clip(mu[host] + dev + rng.gauss(0.0, PH["gc_sd"]), 0.1, 0.9),
                                      "%.4f" % _clip(mu3 + dev3 + rng.gauss(0.0, PH["gc3_sd"]), 0.02, 0.98)])
                else:
                    L = rng.randint(*P["me_len"])
                    me_rows.append([name, cname, pos, pos + L - 1, e])
                pos += L + rng.randint(*P["gap"])

    # orthogroup splits: a clade's copies of a vertically inherited family get their own id
    transferred = set(tr["family"] for tr in sim.transfers)
    fam_rows = {}
    for k, e in enumerate(row_lin):
        fam_rows.setdefault(sim.fam[e], []).append(k)
    label = {}                                   # lineage -> family label after splitting
    n_split = 0
    for f in sorted(fam_rows):
        rows = fam_rows[f]
        if f >= n_families or f in transferred or len(rows) < 4 or rng.random() >= PARAMS[dom[sp.root]]["split_p"]:
            continue
        cand = []
        for s_ in range(sp.n):
            if s_ == sp.root:
                continue
            inside = [k for k in rows if sp.desc[s_, extant_leaf[row_lin[k]]]]
            if 1 <= len(inside) <= len(rows) - 2:
                cand.append(inside)
        if not cand:
            continue
        for k in rng.choice(cand):
            gene_rows[k][2] = "F%04ds" % f
            label[row_lin[k]] = "F%04ds" % f
        n_split += 1
    with open(os.path.join(out, "gene_composition.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "gc", "gc3"])
        w.writerows(comp_rows)
    with open(os.path.join(out, "genes.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "species", "family", "contig", "start", "end", "strand"])
        w.writerows(gene_rows)
    with open(os.path.join(out, "mobile_elements.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "contig", "start", "end", "type"])
        w.writerows(me_rows)
    with open(os.path.join(out, "species.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "domain"])
        for i in range(sp.n):
            if sp.is_leaf[i]:
                w.writerow([sp.names[i], dom[i]])
    with open(os.path.join(out, "species_tree.nwk"), "w") as fh:
        fh.write(sp.to_newick() + "\n")

    # gene trees: true genealogy + estimation noise (NNI) for ML and bootstrap samples
    fam_lin = {}
    for lin, gid in extant.items():
        fam_lin.setdefault(sim.fam[lin], []).append(lin)
    n_trees = 0
    for f in range(sim.n_fam):
        if len(fam_lin.get(f, [])) < 2:
            continue
        rate = 0.8 * rng.lognormvariate(0.0, 0.4)
        nodes, stack, order = {}, [sim.fam_root[f]], []
        while stack:
            u = stack.pop()
            order.append(u)
            stack.extend(sim.kids[u])
        for u in reversed(order):
            nd = Node(extant.get(u))
            end = sim.end[u] if sim.end[u] is not None else sim.birth[u]
            nd.length = max(1e-4, (end - sim.birth[u]) * rate * rng.lognormvariate(0.0, 0.25))
            for c in sim.kids[u]:
                nd.add(nodes.pop(c))
            nodes[u] = nd
        tree = prune_to(nodes[sim.fam_root[f]], set(extant[x] for x in fam_lin[f]))
        base = unroot(tree)
        n_int = sum(1 for nd in base.preorder() if nd.children and nd.parent is not None)
        lam = noise * n_int * (4.0 if rng.random() < 0.2 else 1.0)
        ml = copy_tree(base)
        for _ in range(_poisson(lam, rng)):
            random_nni(ml, rng)
        samples = [ml]
        for _ in range(n_samples):
            bt = copy_tree(ml)
            for _ in range(_poisson(lam, rng)):
                random_nni(bt, rng)
            samples.append(bt)
        groups = {}
        for x in fam_lin[f]:
            groups.setdefault(label.get(x, "F%04d" % f), set()).add(extant[x])
        for name, keep in groups.items():
            if len(keep) < 2:
                continue
            whole = len(groups) == 1
            with open(os.path.join(out, "gene_trees_true", name + ".nwk"), "w") as fh:
                t = base if whole else unroot(prune_to(base, keep))
                fh.write(to_newick(t, internal_names=False) + "\n")
            lines = [to_newick(t if whole else unroot(prune_to(t, keep)), internal_names=False) for t in samples]
            with open(os.path.join(out, "gene_trees", name + ".nwk"), "w") as fh:
                fh.write("\n".join(lines) + "\n")
            n_trees += 1

    # truth
    fam_leaves = {}
    for lin, b in extant_leaf.items():
        fam_leaves.setdefault(sim.fam[lin], []).append(b)
    n_true = 0
    with open(os.path.join(out, "truth_transfers.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["transfer_id", "family", "donor_node", "recipient_node", "time", "detectable",
                    "n_genes", "recipient_genes", "donor_kind"])
        n_ghost = 0
        for ti, tr in enumerate(sim.transfers):
            R = tr["recipient"]
            genes, stack = [], [tr["lineage"]]
            while stack:
                u = stack.pop()
                if u in extant and sp.desc[R, extant_leaf[u]] and u not in contam_lin:
                    genes.append(extant[u])
                stack.extend(sim.kids[u])
            ghost = tr["donor"] is None
            if not genes or (not ghost and len(fam_lin.get(tr["family"], [])) < 2):
                continue
            # sampled donor: needs a family member outside R to be seen by reconciliation;
            # unsampled donor: the family itself is new (a gain at R)
            detectable = ghost or any(not sp.desc[R, b] for b in fam_leaves[tr["family"]])
            w.writerow(["T%05d" % ti, "F%04d" % tr["family"], "unsampled" if ghost else sp.names[tr["donor"]],
                        sp.names[R], "%.4f" % tr["time"], int(detectable), len(genes),
                        ",".join(sorted(genes)), "unsampled" if ghost else "sampled"])
            n_true += 1
            n_ghost += ghost
    with open(os.path.join(out, "truth_contaminants.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "species", "source_species"])
        for lin, src in contam_lin.items():
            w.writerow([extant[lin], sp.names[extant_leaf[lin]], src])

    info = {"domain": domain, "n_species": n_species, "n_families": n_families, "seed": seed,
            "n_samples": n_samples, "noise": noise, "genes": len(gene_rows),
            "mobile_elements": len(me_rows), "gene_trees": n_trees,
            "true_transfers_with_survivors": n_true, "of_which_unsampled_donor": n_ghost,
            "orphan_families": len(sim.orphan_fam), "orthogroup_splits": n_split,
            "contaminant_genes": len(contam_lin)}
    with open(os.path.join(out, "sim_params.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    return info
