"""Tables of the relational line, version 1 (package W12): as version 0 (seeddrift/train/rel/tables/v0_build.py: the same
blocks, the same fields, the same stored placement), with the rows of the uncapped search (v1_hits.py) and the rules
the contract adds for this version (api.SHARD_FIELDS_ADDED, META_FIELDS_ADDED, MAX_ROWS_PER_GENOME, MAX_SELF_COPY_ROWS,
TIER_EDGES). Numpy only.

A table: the examined genome A (row 0) with its loci drawn from A alone (a random target gene with codons, its two
neighbours on the replicon, 16 relation genes), then
  self-copy rows  up to MAX_SELF_COPY_ROWS other copies of the target gene in A itself (A's own pairs on it), drawn at
                  random; genome_of_row 0; their cells with codons are KIND_SELF_COPY
  relative rows   training genomes with a homolog row on the target gene inside its stored block, drawn tier by tier
                  (the distance tier of the genome to A: TIER_EDGES on the k-mer distance), tiers taking turns, inside a
                  tier at random; a genome gives at most MAX_ROWS_PER_GENOME rows (its paralogs on the target, drawn at
                  random); genome_of_row 1, 2, ... in row order
The candidates of a tier are only the genomes that have a homolog on the target gene (the rule of version 1: having a
row now means the uncapped search found a homolog). No score or rank is read: the search stores none.
Cells of a row on a block: a pair of A's gene of that block with a gene of the row's genome (the target block: the
pair that chose the row; another block: a pair drawn at random, distinct between the rows of one genome where it can be).
Its alignments are merged into one placement: in the order of their query span (longest first), an alignment is kept
when it overlaps no kept one on the query or on the target and keeps both in the same order; the others are dropped and
counted. The stretch is the target gene's codons from the first to the last aligned one (cut at the block length), and
`col` the column of each in A's coordinates (-1 for a codon without one), as in version 0.
Kinds: KIND_PRESENT (KIND_SELF_COPY for a self-copy row) with codons; KIND_ABSENT when the row's genome holds no
homolog of the block's gene (it was searched: every row genome is a training genome of the search database, and A's
own genome was searched against itself); KIND_TRUNCATED when a homolog exists but none of its aligned codons lies in
the stored parts; KIND_NONE for a row or a block that does not exist.
Meta (never a model input): genome (the index in the search's genome list), group, split, gene, tier (row 0: -1; a
self-copy row: 0, its distance being 0), identity (as version 0), row_gene, row_first.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.tables import v0_build as V0
from seeddrift.train.rel.tables import v0_dist as D
from seeddrift.train.rel.tables import v0_io as IO
from seeddrift.train.rel.tables import v1_hits as H

VERSION = 1
TABLE_VERSION = "w12-v1-1"
SLICES, B = V0.SLICES, V0.B
TARGET, LEFT, RIGHT = V0.TARGET, V0.LEFT, V0.RIGHT
MAX_ROWS_PER_GENOME = int(getattr(A, "MAX_ROWS_PER_GENOME", 2))
MAX_SELF_COPY_ROWS = int(getattr(A, "MAX_SELF_COPY_ROWS", 2))
TIER_EDGES = tuple(getattr(A, "TIER_EDGES", (0.05, 0.2, 0.5, 1.0)))
N_TIERS = len(TIER_EDGES) + 2                       # tiers 0 .. len(edges) on finite distances, then "no shared k-mer"
SPLIT_CODE = dict(train=0, heldout=1, island=1)      # the development chromosomes' tables are evaluation tables


def fields(rows):
    return IO.fields(rows, VERSION)


def alloc(n, rows):
    return IO.alloc(n, rows, VERSION)


def placement(hits, frame_len, n_codons):
    """(query positions, target positions) of the aligned codons of one pair inside the stored parts (query positions
    below frame_len, target positions below n_codons), strictly increasing on both sides; alignments dropped (overlap or
    order). hits: [(qs, qe, ss, se, cigar)] of the pair."""
    order = sorted(range(len(hits)), key=lambda h: (-(hits[h][1] - hits[h][0]), hits[h][0], hits[h][2]))
    kept, dropped = [], 0
    for h in order:
        qs, qe, ss, se, _ = hits[h]
        ok = True
        for k in kept:
            kq, kqe, ks, kse, _ = hits[k]
            if not (qe < kq or qs > kqe) or not (se < ks or ss > kse) or ((qs < kq) != (ss < ks)):
                ok = False
                break
        if ok:
            kept.append(h)
        else:
            dropped += 1
    qs_all, ts_all = [], []
    for h in sorted(kept, key=lambda h: hits[h][0]):
        q, t = H.columns(hits[h][4], hits[h][0], hits[h][2])
        qs_all.append(q)
        ts_all.append(t)
    if not qs_all:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), dropped
    q, t = np.concatenate(qs_all), np.concatenate(ts_all)
    m = (q < frame_len) & (t < n_codons)
    return q[m], t[m], dropped


class Builder:
    """Tables of examined genomes from the uncapped search. genomes: [dict(id, split, group)] the search's genome
    list (genomes.json); arrays: {id: Genome} (seeddrift/train/source_model.Genome) of every genome a table may show;
    rows_of(id) -> the search's arrays of a query genome (v1_hits.load); distance(a, b) -> the k-mer distance of two
    genome indices; allowed: the genome indices other rows are drawn from (the training genomes)."""

    def __init__(self, genomes, arrays, rows_of, distance, allowed, rows=32):
        self.genomes, self.g, self.rows_of, self.distance = genomes, arrays, rows_of, distance
        self.allowed = np.zeros(len(genomes), dtype=bool)
        self.allowed[list(allowed)] = True
        bad = [genomes[i]["id"] for i in np.flatnonzero(self.allowed) if genomes[i]["split"] != "train"]
        if bad:
            raise SystemExit("tables v1: other rows may only be training genomes: %s" % bad[:5])
        if any(g["split"] not in ("train", "heldout", "island") for g in genomes):
            raise SystemExit("tables v1: a genome outside the train, held-out and island lists")
        self.rows = int(rows)
        self.gix = {g["id"]: i for i, g in enumerate(genomes)}
        self._cur = (None, None)
        self._perm = {}

    def state(self, gid):
        if self._cur[0] != gid:
            a = self.rows_of(gid)
            self._cur = (gid, dict(a=a, gene_of_pair=np.repeat(np.arange(len(a["off"]) - 1), np.diff(a["off"]))))
        return self._cur[1]

    def release(self):
        self._cur = (None, None)

    def hits(self, a, p):
        out = []
        for h in range(int(a["hsp_off"][p]), int(a["hsp_off"][p + 1])):
            out.append((int(a["qs"][h]), int(a["qe"][h]), int(a["ss"][h]), int(a["se"][h]), bytes(a["cigar"][a["cigar_off"][h]:a["cigar_off"][h + 1]])))
        return out

    def cell(self, a, p, partner, frame_len, slot):
        """(codons, col, first, gene, dropped alignments, lost codons) of pair p under the block, or None."""
        j = int(a["tgene"][p])
        cb = partner.codons[partner.coff[j]:partner.coff[j + 1]]
        q, t, dropped = placement(self.hits(a, p), frame_len, len(cb))
        if not len(t):
            return None, dropped
        first = int(t[0])
        keep = t < first + slot
        lost = int(len(t) - keep.sum())
        q, t = q[keep], t[keep] - first
        codons = cb[first:first + int(t[-1]) + 1]
        col = np.full(len(codons), -1, dtype=np.int16)
        col[t] = q
        return (codons, col, first, j, lost), dropped

    def target(self, gid, occ, seed, part):
        """The target gene of the occ-th table of a genome in part `part` (train, heldout, island): its genes with a
        codon in an order shuffled once per genome (seed, the part's stream, genome index), taken in turn and from the
        start again when all have been taken."""
        key = (gid, int(seed), part)
        if self._perm.get("key") != key:
            gen = self.g[gid]
            ok = np.flatnonzero(np.diff(gen.coff) > 0)
            if not len(ok):
                raise SystemExit("tables v1: %s has no gene with a codon" % gid)
            rng = np.random.default_rng([int(seed), SPLIT_STREAM[part], 2, self.gix[gid]])
            self._perm = dict(key=key, perm=ok[rng.permutation(len(ok))])
        perm = self._perm["perm"]
        return int(perm[int(occ) % len(perm)])

    def table(self, out, k, gid, group, split, rng, census=None, occ=None, seed=None, part=None, gene=None):
        gen, st, R = self.g[gid], self.state(gid), self.rows
        a = st["a"]
        lens = np.diff(gen.coff).astype(np.int64)
        me = self.gix[gid]
        out["group"][k], out["split"][k] = group, split
        # the loci, from the examined genome alone (as version 0); the target from the plan when it gives one
        if gene is not None:                                            # drawn over the whole group (plan_by_gene)
            i = int(gene)
            if not 0 <= i < gen.n or lens[i] <= 0:
                raise SystemExit("tables v1: target gene %d of %s has no codon" % (i, gid))
        elif occ is not None:
            i = self.target(gid, occ, seed, part)
        else:
            i = int(rng.integers(gen.n))
            for _ in range(64):
                if lens[i] > 0:
                    break
                i = int(rng.integers(gen.n))
        if lens[i] <= 0:
            raise SystemExit("tables v1: %s has no gene with a codon" % gid)
        p0 = int(gen.pos[i])
        near = [int(gen.order[x]) if 0 <= x < gen.n and gen.rep[gen.order[x]] == gen.rep[i] else -1 for x in (p0 - 1, p0 + 1)]
        draw = rng.choice(gen.n, size=min(gen.n, A.N_RELATION + 3), replace=False)
        rel = [int(j) for j in draw if int(j) != i and int(j) not in near][:A.N_RELATION]
        genes = rel + [-1] * (A.N_RELATION - len(rel)) + near + [i]
        genes = [j if j >= 0 and lens[j] > 0 else -1 for j in genes]
        flen, own = np.zeros(B, dtype=np.int64), [None] * B
        for b, (_, s0, s1) in enumerate(SLICES):
            j = genes[b]
            out["gene"][k, b] = j
            if j < 0:
                continue
            c = gen.codons[gen.coff[j]:gen.coff[j + 1]][:s1 - s0]
            own[b], flen[b] = c, len(c)
            out["tok"][k, 0, s0:s0 + len(c)] = c
            out["col"][k, 0, s0:s0 + len(c)] = np.arange(len(c))
            out["n"][k, 0, b] = len(c)
            out["kind"][k, 0, b] = A.KIND_PRESENT
            out["start"][k, 0, b], out["strand"][k, 0, b], out["replicon"][k, 0, b] = V0.locus(gen, j, 0)
            out["row_gene"][k, 0, b], out["row_first"][k, 0, b] = j, 0
        out["frame_len"][k] = flen
        out["genome"][k, 0], out["genome_of_row"][k, 0], out["identity"][k, 0], out["tier"][k, 0] = me, 0, 1.0, -1
        # the pairs of every block's gene, by target genome
        by_block = []
        for b in range(B):
            j = genes[b]
            pairs = {}
            if j >= 0:
                for p in range(int(a["off"][j]), int(a["off"][j + 1])):
                    tg, tj = int(a["tgenome"][p]), int(a["tgene"][p])
                    if tg == me and tj == j:
                        continue
                    pairs.setdefault(tg, []).append(p)
            by_block.append(pairs)
        c = dict(dropped=0, cut_cells=0, cut_codons=0, candidates=0, self_candidates=0, tier_rows=np.zeros(N_TIERS, dtype=np.int64))
        # which pairs reach the stored target block
        tgt = by_block[TARGET]

        def reaches(p, tg):
            partner = self.g[self.genomes[tg]["id"]]
            jj = int(a["tgene"][p])
            q, t, _ = placement(self.hits(a, p), int(flen[TARGET]), int(partner.coff[jj + 1] - partner.coff[jj]))
            return len(t) > 0
        selfs = [p for p in tgt.get(me, []) if reaches(p, me)]
        rows = []                                                    # (genome index, pair on the target)
        for x in rng.permutation(len(selfs))[:MAX_SELF_COPY_ROWS]:
            rows.append((me, selfs[int(x)]))
        c["self_candidates"] = len(selfs)
        tiers = [[] for _ in range(N_TIERS)]
        for tg in sorted(t for t in tgt if t != me and self.allowed[t]):
            ps = [p for p in tgt[tg] if reaches(p, tg)]
            if ps:
                tiers[int(D.tier_of([self.distance(me, tg)], TIER_EDGES)[0])].append((tg, ps))
        c["candidates"] = sum(len(t) for t in tiers)
        queues = []
        for t in tiers:
            q = []
            for x in rng.permutation(len(t)):
                tg, ps = t[int(x)]
                for y in rng.permutation(len(ps))[:MAX_ROWS_PER_GENOME]:
                    q.append((tg, ps[int(y)]))
            queues.append(q)
        while len(rows) < R - 1 and any(queues):
            for t, q in enumerate(queues):
                if q and len(rows) < R - 1:
                    rows.append(q.pop(0))
                    c["tier_rows"][t] += 1
        # the rows' cells
        ident = np.full((R, B), np.nan)
        same, with_col = np.zeros((R, B), dtype=np.int64), np.zeros((R, B), dtype=np.int64)
        number, used = {me: 0}, {}
        for r, (tg, ptarget) in enumerate(rows, start=1):
            partner = self.g[self.genomes[tg]["id"]]
            if tg not in number:
                number[tg] = len(number)
            out["genome"][k, r], out["genome_of_row"][k, r] = tg, number[tg]
            out["tier"][k, r] = 0 if tg == me else int(D.tier_of([self.distance(me, tg)], TIER_EDGES)[0])
            for b, (_, s0, s1) in enumerate(SLICES):
                if genes[b] < 0:
                    continue
                if b == TARGET:
                    p = ptarget
                else:
                    cand = by_block[b].get(tg, [])
                    if not cand:
                        out["kind"][k, r, b] = A.KIND_ABSENT
                        continue
                    free = [x for x in cand if x not in used.get((tg, b), set())] or cand
                    p = free[int(rng.integers(len(free)))]
                used.setdefault((tg, b), set()).add(p)
                got, dropped = self.cell(a, p, partner, int(flen[b]), s1 - s0)
                c["dropped"] += dropped
                if got is None:
                    out["kind"][k, r, b] = A.KIND_TRUNCATED
                    continue
                codons, col, first, j, lost = got
                n = len(codons)
                out["tok"][k, r, s0:s0 + n] = codons
                out["col"][k, r, s0:s0 + n] = col
                out["n"][k, r, b] = n
                out["kind"][k, r, b] = A.KIND_SELF_COPY if tg == me else A.KIND_PRESENT
                out["start"][k, r, b], out["strand"][k, r, b], out["replicon"][k, r, b] = V0.locus(partner, j, first)
                out["row_gene"][k, r, b], out["row_first"][k, r, b] = j, first
                m = col >= 0
                s, w = int((codons[m] == own[b][col[m]]).sum()), int(m.sum())
                ident[r, b], same[r, b], with_col[r, b] = (s / w if w else np.nan), s, w
                if lost:
                    c["cut_cells"] += 1
                    c["cut_codons"] += lost
            out["identity"][k, r] = ident[r, TARGET]
        if census is not None:
            rnd = np.full(B, np.nan)
            n_allowed = max(int(self.allowed.sum()) - (1 if self.allowed[me] else 0), 1)
            for b in range(B):
                if genes[b] >= 0:
                    rnd[b] = sum(1 for tg in by_block[b] if tg != me and self.allowed[tg]) / n_allowed
            census.add(group, out["kind"][k], ident, out["n"][k], same, with_col, rnd, c, out["genome"][k], me,
                       [self.distance(me, int(g)) for g in out["genome"][k][1:] if g >= 0 and g != me])


DIST_EDGES = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5)      # report bins of the rows' distances to row 0


class Census1:
    """The census of version 0 (v0_census.Census) and what version 1 adds: rows per distance tier, self-copy rows,
    candidates on the target, alignments dropped, the distances of the rows to row 0 (histogram, inf apart)."""

    def __init__(self, n_groups, rows):
        from seeddrift.train.rel.tables import v0_census as C
        self.base = C.Census(n_groups, rows)
        g = int(n_groups)
        self.a = dict(tier_rows=np.zeros((g, N_TIERS), np.int64), self_rows=np.zeros(g, np.int64), tables_with_self=np.zeros(g, np.int64),
                      dropped=np.zeros(g, np.int64), candidates_sum=np.zeros(g, np.int64), self_candidates_sum=np.zeros(g, np.int64),
                      dist_rows=np.zeros((g, len(DIST_EDGES) + 1), np.int64))

    def add(self, group, kinds, ident, n, same, with_col, rnd, c, genomes, me, dists):
        self.base.add(group, V0.ROLE_OF_BLOCK, kinds, ident, n, same, with_col, rnd, c["candidates"],
                      dict(cut_cells=c["cut_cells"], cut_codons=c["cut_codons"]))
        a, g = self.a, int(group)
        a["tier_rows"][g] += c["tier_rows"]
        n_self = int((genomes[1:] == me).sum())
        a["self_rows"][g] += n_self
        a["tables_with_self"][g] += int(n_self > 0)
        a["dropped"][g] += c["dropped"]
        a["candidates_sum"][g] += c["candidates"]
        a["self_candidates_sum"][g] += c["self_candidates"]
        d = np.asarray(dists, dtype=np.float64)
        fin = d[np.isfinite(d)]
        a["dist_rows"][g, :len(DIST_EDGES)] += np.bincount(np.clip(np.searchsorted(DIST_EDGES, fin, side="right") - 1, 0, len(DIST_EDGES) - 1),
                                                          minlength=len(DIST_EDGES))[:len(DIST_EDGES)]
        a["dist_rows"][g, len(DIST_EDGES)] += int((~np.isfinite(d)).sum())

    def merge(self, other):
        self.base.merge(other.base)
        for k in self.a:
            self.a[k] += other.a[k]
        return self

    def to_arrays(self):
        out = {"v0_" + k: v.copy() for k, v in self.base.to_arrays().items()}
        out.update({k: v.copy() for k, v in self.a.items()})
        return out

    @classmethod
    def from_arrays(cls, arrays, rows):
        from seeddrift.train.rel.tables import v0_census as C
        c = cls(len(arrays["tier_rows"]), rows)
        c.base = C.Census.from_arrays({k[3:]: np.array(v) for k, v in arrays.items() if k.startswith("v0_")}, rows)
        c.a = {k: np.array(v) for k, v in arrays.items() if not k.startswith("v0_")}
        return c


SPLIT_STREAM = dict(train=0, heldout=1, island=2)    # the random streams: those of version 0 for train and held-out


def plan_set(genomes, groups, split, per_group, seed):
    """As v0_build.plan_set, over the search's genome list (dicts with id, group, split, genes); the split "island"
    (the development chromosomes) has its own stream. Also, per table, how many tables of the same genome come before it
    (plan[3]): the table takes that place of its genome's shuffled genes (Builder.target), so a gene is the target
    again only when every gene of its genome has been one."""
    by = {g: sorted(x["id"] for x in genomes if x["split"] == split and x["group"] == g and int(x.get("genes", 1)) > 0) for g in groups}
    present = [g for g in groups if by[g]]
    perm = {g: np.random.default_rng([int(seed), SPLIT_STREAM[split], 1, groups.index(g)]).permutation(len(by[g])) for g in present}
    genome_of, group_of, occ, seen = [], [], [], {}
    for t in range(int(per_group) * len(present)):
        g = present[t % len(present)]
        gid = by[g][int(perm[g][(t // len(present)) % len(by[g])])]
        genome_of.append(gid)
        group_of.append(groups.index(g))
        occ.append(seen.get(gid, 0))
        seen[gid] = occ[-1] + 1
    return genome_of, group_of, [g for g in groups if not by[g]], occ


def plan_by_gene(genomes, groups, split, per_group, seed, valid):
    """The tables of a part with their target genes drawn over the whole group (the programme lead, 2026-11-23): the
    groups take turns as in plan_set, and the k-th table of a group takes the k-th of the group's genes that can be a
    target (`valid`: {genome id: gene indices with a codon}) in an order shuffled once per pass over them, so every gene
    of the group is a target once before any is twice, and no gene more than once above the mean. Returns
    (genome_of, group_of, groups without a genome, None, target gene of every table)."""
    pool = {g: [(x["id"], int(j)) for x in sorted((x for x in genomes if x["split"] == split and x["group"] == g), key=lambda x: x["id"])
                for j in valid.get(x["id"], ())] for g in groups}
    present = [g for g in groups if pool[g]]
    genome_of, group_of, gene, perms = [], [], [], {}
    for t in range(int(per_group) * len(present)):
        g = present[t % len(present)]
        k = t // len(present)
        n = len(pool[g])
        key = (g, k // n)
        if key not in perms:
            perms = {key: np.random.default_rng([int(seed), SPLIT_STREAM[split], 3, groups.index(g), k // n]).permutation(n)} | \
                {x: v for x, v in perms.items() if x[0] != g}
        gid, j = pool[g][int(perms[key][k % n])]
        genome_of.append(gid)
        group_of.append(groups.index(g))
        gene.append(j)
    return genome_of, group_of, [g for g in groups if not pool[g]], None, gene


def target_stats(plan, groups, gene_of_table, relation=None, valid=None, genome_group=None):
    """Per group (and all): tables, distinct targets, the genes that could be one (with `valid` {genome id: gene indices
    with a codon} and `genome_group` {genome id: group} of the part's genomes), mean and most uses of a target, the share
    of tables whose target is used 10 times or more and the genomes they come from, and with `relation` ({genome id:
    anchor | same species | same genus | ...}) the share of tables by relation."""
    genome_of, group_of = plan[0], plan[1]
    tgt = np.asarray(gene_of_table, dtype=np.int64)
    out = {}
    for gi, g in [(i, x) for i, x in enumerate(groups)] + [(None, "all")]:
        idx = [t for t in range(len(genome_of)) if gi is None or group_of[t] == gi]
        if not idx:
            continue
        keys = [(genome_of[t], int(tgt[t])) for t in idx]
        uses = {}
        for x in keys:
            uses[x] = uses.get(x, 0) + 1
        heavy = [t for t, x in zip(idx, keys) if uses[x] >= 10]
        d = dict(tables=len(idx), distinct=len(uses), max_uses=max(uses.values()), tables_target_used_10_plus=len(heavy),
                 share_target_used_10_plus=round(len(heavy) / len(idx), 4), genomes_of_those=len({genome_of[t] for t in heavy}))
        if valid is not None and genome_group is not None:
            n_valid = sum(len(v) for gid, v in valid.items() if gid in genome_group and (gi is None or genome_group[gid] == g))
            d.update(targetable_genes=n_valid, mean_uses=round(len(idx) / n_valid, 3) if n_valid else None)
        if relation is not None:
            r = {}
            for t in idx:
                k = relation.get(genome_of[t]) or "anchor"
                r[k] = r.get(k, 0) + 1
            d["share_by_relation"] = {k: round(v / len(idx), 4) for k, v in sorted(r.items())}
        out[g] = d
    return out


def table_rng(seed, split, t):
    return np.random.default_rng([int(seed), SPLIT_STREAM[split], 0, int(t)])


_W = {}


def _unit(args):
    gid, tables = args
    w = _W
    b = w["builder"]
    out = alloc(len(tables), b.rows)
    cen = Census1(len(w["groups"]), b.rows)
    for k, t in enumerate(tables):
        occ = w["occ"][t] if w.get("occ") is not None else None
        gene = w["gene"][t] if w.get("gene") is not None else None
        b.table(out, k, gid, w["group_of"][t], SPLIT_CODE[w["split"]], table_rng(w["seed"], w["split"], t), cen, occ=occ, seed=w["seed"], part=w["split"], gene=gene)
    b.release()
    return tables, out, cen.to_arrays()


def build_set(builder, groups, split, plan, out_dir, shard_tables=2400, seed=1, processes=1, progress=None):
    """Build the tables of a plan into shards under out_dir (the fields of version 1). Returns ([(shard, tables)],
    Census1). The result depends on the seed and the data only."""
    import multiprocessing
    import time
    genome_of, group_of = plan[0], plan[1]
    n, shard_tables = len(genome_of), int(shard_tables)
    shards, mm = IO.create(out_dir, n, builder.rows, shard_tables, VERSION)
    by = {}
    for t, g in enumerate(genome_of):
        by.setdefault(g, []).append(t)
    todo = [(g, ts[x:x + V0.UNIT_TABLES]) for g, ts in sorted(by.items()) for x in range(0, len(ts), V0.UNIT_TABLES)]
    todo.sort(key=lambda u: (-len(u[1]), u[0], u[1][0]))
    _W.clear()
    _W.update(builder=builder, groups=list(groups), group_of=group_of, split=split, seed=int(seed), occ=plan[3] if len(plan) > 3 else None,
              gene=plan[4] if len(plan) > 4 else None)
    census, done, t0 = Census1(len(groups), builder.rows), 0, time.time()
    pool = multiprocessing.get_context("fork").Pool(int(processes)) if processes > 1 and len(todo) > 1 else None
    try:
        for tables, out, arrays in (pool.imap_unordered(_unit, todo, chunksize=1) if pool else map(_unit, todo)):
            at = np.asarray(tables, dtype=np.int64)
            for s in np.unique(at // shard_tables):
                m = at // shard_tables == s
                for name in out:
                    mm[int(s), name][at[m] - int(s) * shard_tables] = out[name][m]
            census.merge(Census1.from_arrays(arrays, builder.rows))
            k = len(tables)
            if progress and (done + k) * 20 // max(n, 1) != done * 20 // max(n, 1):
                progress("%s: %d of %d tables, %.0f s" % (split, done + k, n, time.time() - t0))
            done += k
    finally:
        if pool:
            pool.close()
            pool.join()
        _W.clear()
        for x in mm.values():
            x.flush()
        mm.clear()
    if done != n:
        raise SystemExit("tables v1: %d of %d tables were built" % (done, n))
    return shards, census


KINDS_V1 = (A.KIND_NONE if hasattr(A, "KIND_NONE") else 0, A.KIND_PRESENT, A.KIND_ABSENT, A.KIND_TRUNCATED, A.KIND_SELF_COPY)


def verify_set(out_dir, shards, rows, split, examined, allowed, n_groups):
    """Problems of a written set of version 1 (empty when sound)."""
    bad = []
    for name, m in shards:
        z = {f: np.load(IO.shard_path(out_dir, name, f), mmap_mode="r") for f in fields(rows)}

        def need(cond, what):
            if not bool(cond):
                bad.append("%s: %s" % (name, what))
        for f, (dt, shape) in fields(rows).items():
            need(z[f].dtype == dt and z[f].shape == (m,) + shape, "field %s has dtype %s and shape %s" % (f, z[f].dtype, z[f].shape))
        if bad:
            return bad
        kind, genome, gor = np.asarray(z["kind"]), np.asarray(z["genome"]), np.asarray(z["genome_of_row"])
        n, rg, rf = np.asarray(z["n"]).astype(np.int64), np.asarray(z["row_gene"]), np.asarray(z["row_first"])
        need((kind[:, 0, TARGET] == A.KIND_PRESENT).all(), "a table was not written")
        need(np.isin(kind, KINDS_V1).all(), "a kind outside version 1")
        need(np.isin(genome[:, 0], sorted(examined)).all(), "an examined genome outside the %s list" % split)
        other = genome[:, 1:]
        selfrow = other == genome[:, :1]
        need(np.isin(other[(other >= 0) & ~selfrow], sorted(allowed)).all(), "another row that is not a training genome")
        need(((gor[:, 1:] == 0) == selfrow).all() and (gor[:, 0] == 0).all(), "genome_of_row 0 is not exactly the examined genome's rows")
        need((selfrow.sum(1) <= MAX_SELF_COPY_ROWS).all(), "more self-copy rows than allowed")
        for t in range(m):
            g = genome[t][genome[t] >= 0]
            u, cnt = np.unique(g[g != genome[t, 0]], return_counts=True)
            if len(cnt) and cnt.max() > MAX_ROWS_PER_GENOME:
                bad.append("%s: table %d has more rows of one genome than allowed" % (name, t))
                break
        has = genome >= 0
        need((has[:, :-1] | ~has[:, 1:]).all(), "rows are not packed at the front")
        need(((kind > 0).any(2) <= has).all(), "a row without a genome holds cells")
        content = np.isin(kind, (A.KIND_PRESENT, A.KIND_SELF_COPY))
        need((content == (n > 0)).all(), "codons and kinds disagree")
        need(((rg >= 0) == content).all() and ((rf >= 0) == content).all(), "row_gene or row_first where there is no segment, or missing")
        need((kind[:, 1:][selfrow] != A.KIND_PRESENT).all() and (kind[:, 1:][~selfrow & has[:, 1:]] != A.KIND_SELF_COPY).all(), "self-copy kinds on the wrong rows")
        need((np.asarray(z["split"]) == SPLIT_CODE[split]).all(), "a table of another split")
        tier = np.asarray(z["tier"])
        need((tier[:, 0] == -1).all() and ((tier[:, 1:] >= 0) == has[:, 1:]).all() and (tier < N_TIERS).all(), "tiers")
        for b, (_, s0, s1) in enumerate(SLICES):
            tok, col = np.asarray(z["tok"][:, :, s0:s1]), np.asarray(z["col"][:, :, s0:s1])
            inside = np.arange(s1 - s0)[None, None, :] < n[:, :, b, None]
            need((tok[inside] >= A.TARGET_MIN).all() and (tok[inside] <= A.TARGET_MAX).all() and (tok[~inside] == A.PAD).all(), "block %d: tokens" % b)
            need((col[~inside] == -1).all() and (col < np.asarray(z["frame_len"])[:, None, b, None]).all(), "block %d: columns" % b)
            c = col.astype(np.int64)
            prev = np.maximum.accumulate(np.where(c >= 0, c, -1), axis=2)
            need(((c < 0) | (c > np.concatenate([np.full(c.shape[:2] + (1,), -1), prev[:, :, :-1]], axis=2))).all(), "block %d: columns not strictly increasing" % b)
    return bad


def frame_pairs(index, genome, row_gene, row_first, n, blocks=None):
    """The pair numbers of `index` (v1_hits.Index) that the online frame layout (align.frame_layout_online) looks up for
    these tables: in every block of `blocks` (default: all), from the block's frame relative (the first other row with
    a stretch there) to every other row with one, the pair whose target gene is the row's gene, else the other
    direction. genome [T, R], row_gene / row_first / n [T, R, B]."""
    out = []
    blocks = range(A.N_BLOCKS) if blocks is None else blocks
    for t in range(genome.shape[0]):
        for b in blocks:
            rows = [r for r in range(1, n.shape[1]) if n[t, r, b] > 0]
            if not rows:
                continue
            fr = rows[0]
            qg, qi, qf = int(genome[t, fr]), int(row_gene[t, fr, b]), int(row_first[t, fr, b])
            for r in rows[1:]:
                tg, ti, tf = int(genome[t, r]), int(row_gene[t, r, b]), int(row_first[t, r, b])
                if min(qg, qi, qf, tg, ti, tf) < 0:
                    continue
                got = [p for p in index.pairs(qg, tg, qi) if index.tgene(p) == ti]
                if not got:
                    got = [p for p in index.pairs(tg, qg, ti) if index.tgene(p) == qi]
                out += got
    return np.unique(np.asarray(out, dtype=np.int64))


def _frame_unit(args):
    index_folder, folder, name, blocks = args
    z = {f: np.load(IO.shard_path(folder, name, f), mmap_mode="r") for f in ("genome", "row_gene", "row_first", "n")}
    return frame_pairs(H.Index(index_folder), np.asarray(z["genome"]), np.asarray(z["row_gene"]), np.asarray(z["row_first"]), np.asarray(z["n"]), blocks)


def frame_index(index_folder, sets, out_folder, blocks=None, processes=1):
    """Write the lookup index of the pairs the online frame layout needs for the tables of `sets` ([(folder, [shard
    names])]): a subset of the search's index (v1_hits.write_subset), so a machine that trains on these tables needs
    this one, not the whole index. Returns its sizes."""
    import multiprocessing
    todo = [(index_folder, folder, name, None if blocks is None else list(blocks)) for folder, names in sets for name in names]
    pool = multiprocessing.get_context("fork").Pool(int(processes)) if processes > 1 and len(todo) > 1 else None
    try:
        parts = list(pool.imap_unordered(_frame_unit, todo, chunksize=1) if pool else map(_frame_unit, todo))
    finally:
        if pool:
            pool.close()
            pool.join()
    ids = np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)
    return H.write_subset(H.Index(index_folder), ids, out_folder)
