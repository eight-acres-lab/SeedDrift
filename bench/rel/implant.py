"""Implant tables (package W18): the first version of the ruler sets of the relational line. numpy only.

A ruler, never a training table and never an item of a selection round: the stages that call this module are
evaluation stages, and no train-side module imports it (tests/ci/test_rel_c0_imports.py). It imports the table code of
package W2 (seeddrift/train/rel/tables/), which never imports it. Results on it are marked "construct only".

A triple is three tables of one held-out examined genome A at the locus of one of its genes i, with the same rows:
  implant   the target block holds the first n codons of gene j of another held-out genome D (the donor); the other
            rows' target cells are their homolog rows of D:j, placed on D:j (the stored placement, as tables of version 0)
  original  the target block holds the first n codons of A:i (the position control)
  own       the target block holds the first n codons of another gene k of A (not i, not a neighbour of i, not a
            relation block): it breaks the synteny as the implant does, not the genome of origin (the primary control)
n is the same in the three tables: 256 when i, j and k all have 256 stored codons (phase 1, tried first), else the
shortest of the three, with lengths within length_tol (phase 2). The neighbours and relation blocks are A's, drawn from A
alone as in tables of version 0 (relation genes avoid i, its neighbours and k).

Rows: training genomes that hold, on each of A:i, D:j and A:k, a homolog row of stage pairs starting inside the first n
codons with at least max(10, 0.3 n) aligned codons under them; up to rows - 1 of them drawn at random, the same rows in
the same order in the three tables, so every row has content on the target block of all three. To find such rows the
genes are drawn in turn: i among A's genes with a candidate row, j among D's genes that share a candidate row genome with
i, k among A's genes that share one with both. A, D and every genome of their split units are never rows. Without a
common row after the attempts, the triple is drawn unconditioned and keeps zero rows (phase 3; composition only).

Donors: held-out genomes of the standard code (the caller decides which are eligible), by the tier of their distance to
A: for every group and tier, every A with a candidate donor in the tier gets per_tier triples in a seeded order, at most
group_tier_cap triples per (group, tier), one donor at most donor_cap triples per group.

Scenarios (every triple of scenario 0 may have a copy of scenario 1, both marked construct only):
  0 in_place          as above (the implant replaces the gene at its locus)
  1 donor_unsampled   the same three tables without the rows closer to D than to A (distance(row, D) < distance(row, A)):
                      the donor's lineage is not sampled; made when it removes a row (it may remove all of them)

Kinds and fields are those of tables of version 0 (v0_build.py), with these differences: the position fields (start,
strand, replicon) of the target block are unknown (-1, 0, -1) in every row of all three tables; the meta field `gene` of
the target block is -1 for an implant (w18 gene_j names it); `tier` stays -1 (the contract's tiers are not defined yet).
W18's own meta (FIELDS; never a model input) is written beside the shards in one file.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.tables import v0_build as V
from seeddrift.train.rel.tables import v0_census as C
from seeddrift.train.rel.tables import v0_dist as DI
from seeddrift.train.rel.tables import v0_io as IO

TABLE_VERSION = "w18-implant-2"
VARIANTS = ("implant", "original", "own")
SCENARIOS = ("in_place", "donor_unsampled")
IMPLANT, ORIGINAL, OWN = 0, 1, 2
STANDARD_CODES = (1, 11)                 # the same codon to amino-acid map
SLICES, B = V.SLICES, V.B
TARGET, LEFT, RIGHT = V.TARGET, V.LEFT, V.RIGHT
CONTEXT_BLOCKS = tuple(range(TARGET))
SEED_STREAM = 18
FULL = A.LEN_TARGET
# W18's meta per table: (dtype, shape after the table axis, what it is); never a model input
FIELDS = {
    "triple":          ("int32", (), "the triple of the table"),
    "parent":          ("int32", (), "the triple of scenario 0 this one copies (itself for scenario 0); the masks of a pass are hashed on it"),
    "scenario":        ("int8", (), "0 in_place, 1 donor_unsampled"),
    "variant":         ("int8", (), "0 implant, 1 original, 2 own"),
    "donor":           ("int32", (), "index of D in the genome list of the set's index (the same in the three tables of a triple)"),
    "gene_i":          ("int32", (), "A's gene at the locus"),
    "gene_j":          ("int32", (), "D's implanted gene"),
    "gene_k":          ("int32", (), "A's other gene (the own control)"),
    "phase":           ("int8", (), "1: n = 256 for all three genes; 2: lengths matched; 3: drawn without a common row (zero rows)"),
    "donor_dist":      ("float32", (), "distance from D to A (inf without a shared k-mer)"),
    "donor_tier":      ("int8", (), "tier of donor_dist"),
    "row_dist":        ("float32", ("R",), "distance of every row to A (nan for row 0 and rows that do not exist)"),
    "row_dist_d":      ("float32", ("R",), "distance of every row to D (nan for row 0 and rows that do not exist)"),
    "row_tier":        ("int8", ("R",), "tier of row_dist (-1 for row 0 and rows that do not exist)"),
    "row_gene":        ("int32", ("R",), "the row's gene on the target block (-1 for row 0 and rows that do not exist)"),
    "on_target":       ("int8", (), "other rows (all with content on the target block)"),
    "covered":         ("int32", (), "codons of the other rows placed on the target block"),
    "target_len":      ("int16", (), "n: codons of the target block of row 0"),
    "target_ends":     ("bool", (), "the target block reaches its gene's last codon"),
    "block_ends":      ("bool", ("B",), "the block of row 0 reaches its gene's last codon"),
    "host_counts":     ("int32", (64,), "A's codon counts (tokens 1..64) over its stored genes other than i and k"),
    "genus_a":         ("int64", (), "genus taxid of A (0: none)"),
    "genus_d":         ("int64", (), "genus taxid of D (0: none)"),
    "unit_a":          ("int32", (), "split unit of A (index into the index's units)"),
    "unit_d":          ("int32", (), "split unit of D"),
    "context_homolog": ("int8", (), "bit 1: A's homolog of D:j is a context gene of the table (pairs of D or of A); bit 2: a row's homolog of D:j is its homolog on a context block"),
    "attempts":        ("int16", (), "draws the triple took"),
}


def meta_fields(rows):
    dims = dict(R=int(rows), B=B)
    return {k: (np.dtype(dt), tuple(dims.get(s, s) for s in shape)) for k, (dt, shape, _) in FIELDS.items()}


def alloc_meta(n, rows):
    fill = dict(row_dist=np.nan, row_dist_d=np.nan, row_tier=-1, row_gene=-1, donor=-1, gene_i=-1, gene_j=-1, gene_k=-1, donor_tier=-1,
                triple=-1, parent=-1, variant=-1, scenario=-1)
    return {k: np.full((int(n),) + shape, fill.get(k, 0), dtype=dt) for k, (dt, shape) in meta_fields(rows).items()}


def stored_len(gen, j):
    return int(min(int(gen.coff[j + 1] - gen.coff[j]), A.LEN_TARGET))


def genus_of(meta):
    r = meta.get("ranks") or []
    return int(r[4]) if len(r) > 4 and r[4] else 0


def split_units(metas):
    """{genome id: its split unit} as stage train_data forms them: inside a group, the genomes linked by a shared genus
    or cluster are one unit, named by its first cluster, else its genus, else its species, else the genome. A unit that
    holds a sealed genome lies wholly in the sealed split, so the units of the open genomes come out the same without it."""
    out = {}
    for g in sorted({m["group"] for m in metas}):
        ids = sorted(m["id"] for m in metas if m["group"] == g)
        mo = {m["id"]: m for m in metas if m["group"] == g}
        parent = {e: e for e in ids}

        def find(e):
            while parent[e] != e:
                parent[e] = parent[parent[e]]
                e = parent[e]
            return e
        first = {}
        for e in ids:
            gen, cl = genus_of(mo[e]), mo[e].get("cluster") or ""
            for k in ([("g", gen)] if gen else []) + ([("c", cl)] if cl else []):
                if k in first:
                    parent[find(e)] = find(first[k])
                else:
                    first[k] = e
        comp = {}
        for e in ids:
            comp.setdefault(find(e), []).append(e)
        for members in comp.values():
            cl = sorted({mo[e].get("cluster") for e in members if mo[e].get("cluster")})
            if cl:
                key = "c:%s" % cl[0]
            else:
                m = mo[members[0]]
                r = m.get("ranks") or []
                key = ("g:%d" % genus_of(m)) if genus_of(m) else (("s:%d" % int(r[5])) if len(r) > 5 and r[5] else "id:%s" % members[0])
            for e in members:
                out[e] = "%s|%s" % (g, key)
    return out


class Implanter:
    """Triples of implant tables.

    pairs: selfsup.Pairs of the source run; genomes: {id: Genome} of the open splits; allowed: the training genomes (rows);
    ids: the genome list the meta field `genome` indexes; dist: [G, G] distances between the genomes of dist_ids (W2's
    distance run); edges: the tier edges; eligible: the genomes that may be A or D (held out, standard code); units:
    {id: split unit} (split_units)."""

    def __init__(self, pairs, genomes, allowed, ids, dist, dist_ids, edges, eligible, units, rows=16, min_codons=64, length_tol=0.15,
                 attempts=40, donor_cap=3, min_aligned=10, aligned_share=0.3):
        self.b = V.Builder(pairs, genomes, allowed, ids, rows)
        self.pairs, self.g, self.rows = pairs, genomes, int(rows)
        self.order = self.b.order
        self.oix = self.b.oix
        self.gix = self.b.gix
        self.dist = np.asarray(dist)
        self.dix = {g: k for k, g in enumerate(dist_ids)}
        missing = [g for g in list(genomes) if g not in self.dix]
        if missing:
            raise SystemExit("implant tables: genomes without a distance: %s" % missing[:5])
        self.edges = [float(x) for x in edges]
        self.eligible = set(eligible)
        self.units = dict(units)
        self.allowed_ix = {self.oix[g] for g in allowed}
        self.min_codons, self.tol, self.attempts, self.donor_cap = int(min_codons), float(length_tol), int(attempts), int(donor_cap)
        self.min_aligned, self.aligned_share = int(min_aligned), float(aligned_share)
        self._rows = {}
        self.counts = dict(triples=0, copies=0, zero_rows=0, failed=0, attempts=0, duplicate_rows=0, phase1=0, phase2=0, phase3=0, context_homolog=0,
                           no_donor_left=0)

    # ---- homolog rows ----
    def first_rows(self, gid, gene):
        """{pairs index of a partner genome: its first homolog row of gene `gene` of genome gid} over all partners."""
        key = (gid, int(gene))
        if key not in self._rows:
            z = self.pairs.of(gid)
            a, e = int(z["off"][gene]), int(z["off"][gene + 1])
            out = {}
            for r in range(a, e):
                ti = int(z["tgenome"][r])
                if ti in out:
                    self.counts["duplicate_rows"] += 1
                else:
                    out[ti] = r
            self._rows[key] = out
        return self._rows[key]

    def partner_gene(self, gid, gene, ti):
        """The partner genome's gene of the homolog row of (gid, gene) to partner ti, or -1."""
        r = self.first_rows(gid, gene).get(int(ti))
        return -1 if r is None else int(self.pairs.of(gid)["tgene"][r])

    def starts_inside(self, gid, gene, n_codons):
        """The allowed partners whose homolog row of the gene starts inside its first n_codons codons."""
        z = self.pairs.of(gid)
        return {ti for ti, r in self.first_rows(gid, gene).items() if ti in self.allowed_ix and int(z["qs"][r]) - 1 < n_codons}

    def cell(self, gid, gene, ti, n_codons, slot):
        r = self.first_rows(gid, gene).get(ti)
        if r is None:
            return None, False
        return self.b.cell(self.pairs.of(gid), r, self.g[self.order[ti]], int(n_codons), int(slot)), True

    def need(self, n):
        return max(self.min_aligned, int(np.ceil(self.aligned_share * int(n))))

    def common_rows(self, targets, n):
        """The allowed partners with at least need(n) aligned codons under the first n codons of every (genome, gene)."""
        sets = [self.starts_inside(g, j, n) for g, j in targets]
        common = set.intersection(*sets) if sets else set()
        want = self.need(n)
        out = []
        for ti in sorted(common):
            cells = [self.cell(g, j, ti, n, A.LEN_TARGET)[0] for g, j in targets]
            if all(c is not None and c[5] >= want for c in cells):
                out.append(ti)
        return out

    def release(self, gid):
        self.pairs.z.pop(gid, None)
        self._rows = {k: v for k, v in self._rows.items() if k[0] != gid}

    def partner_sets(self, gid):
        """[the allowed partners (pairs indices, sorted) whose homolog row of gene g starts inside its stored part] for
        every gene g of genome gid (cheap: no alignment is read)."""
        key = (gid, "sets")
        if key not in self._rows:
            z, gen = self.pairs.of(gid), self.g[gid]
            off = z["off"].astype(np.int64)
            gene = np.repeat(np.arange(len(off) - 1), np.diff(off))
            tg = z["tgenome"].astype(np.int64)
            stored = np.minimum(np.diff(gen.coff), A.LEN_TARGET)
            ok = np.isin(tg, np.fromiter(self.allowed_ix, dtype=np.int64)) & (z["qs"].astype(np.int64) - 1 < stored[gene])
            self._rows[key] = [np.unique(tg[off[g]:off[g + 1]][ok[off[g]:off[g + 1]]]) for g in range(len(off) - 1)]
        return self._rows[key]

    # ---- loci ----
    def near(self, gen, i):
        p = int(gen.pos[i])
        return [int(gen.order[x]) if 0 <= x < gen.n and gen.rep[gen.order[x]] == gen.rep[i] else -1 for x in (p - 1, p + 1)]

    def d(self, a, b):
        return float(self.dist[self.dix[a], self.dix[b]])

    def tier(self, a, b):
        return int(DI.tier_of(np.array([self.d(a, b)]), self.edges)[0])

    def draw(self, gid_a, donors, rng, full, condition=True):
        """One attempt: (i, D, j, k, n) or None. full: i, j and k have 256 stored codons (n = 256); else their stored
        lengths are within length_tol of i's (n = the shortest). With condition, i is drawn among A's genes with a
        partner, j among D's genes that share a partner with i, k among A's genes that share one with both."""
        ga = self.g[gid_a]
        la = np.minimum(np.diff(ga.coff), A.LEN_TARGET)
        sa = self.partner_sets(gid_a) if condition else None
        ok_i = (la == FULL) if full else (la >= self.min_codons)
        cand_i = np.array([x for x in np.flatnonzero(ok_i) if not condition or len(sa[x])], dtype=np.int64)
        if not len(cand_i) or not donors:
            return None
        i = int(cand_i[int(rng.integers(len(cand_i)))])
        li = int(la[i])
        fits = (lambda x: x == FULL) if full else (lambda x: (x >= li * (1 - self.tol)) & (x <= li * (1 + self.tol)) & (x >= self.min_codons))
        d = donors[int(rng.integers(len(donors)))]
        gd = self.g[d]
        ld = np.minimum(np.diff(gd.coff), A.LEN_TARGET)
        cj = np.flatnonzero(fits(ld))
        if condition:
            sd = self.partner_sets(d)
            cj = np.array([x for x in cj if len(np.intersect1d(sa[i], sd[x], assume_unique=True))], dtype=np.int64)
        if not len(cj):
            return None
        j = int(cj[int(rng.integers(len(cj)))])
        bad = set(self.near(ga, i)) | {i}
        ck = [int(x) for x in np.flatnonzero(fits(la)) if int(x) not in bad]
        if condition:
            both = np.intersect1d(sa[i], sd[j], assume_unique=True)
            ck = [x for x in ck if len(np.intersect1d(both, sa[x], assume_unique=True))]
        if not ck:
            return None
        k = int(ck[int(rng.integers(len(ck)))])
        return i, d, j, k, int(min(la[i], ld[j], la[k]))

    def triple(self, gid_a, donors, rng):
        """The draws of one triple: dict(i, d, j, k, n, rows, phase, attempts) or None. Phase 1 (n = 256) and phase 2
        (lengths matched) each take up to `attempts` conditioned draws until the genes have a common row; phase 3 draws
        without the conditions and keeps zero rows."""
        n_att = 0
        for phase, full in ((1, True), (2, False)):
            for _ in range(self.attempts):
                n_att += 1
                got = self.draw(gid_a, donors, rng, full)
                if got is None:
                    continue
                i, d, j, k, n = got
                rows = self.common_rows([(gid_a, i), (d, j), (gid_a, k)], n)
                if rows:
                    return dict(i=i, d=d, j=j, k=k, n=n, rows=rows, phase=phase, attempts=n_att)
        for _ in range(self.attempts):
            n_att += 1
            got = self.draw(gid_a, donors, rng, False, condition=False)
            if got is not None:
                i, d, j, k, n = got
                return dict(i=i, d=d, j=j, k=k, n=n, rows=[], phase=3, attempts=n_att)
        return None

    def context_flag(self, a, d, j, genes, rows):
        """Bit 1: A's homolog of D:j (by the pairs of D, or by the pairs of A's context genes) is a context gene of the
        table; bit 2: some row's homolog of D:j is that row's homolog on a context block."""
        ctx = [g for g in genes[:TARGET] if g >= 0]
        flag = 0
        if self.partner_gene(d, j, self.oix[a]) in ctx or any(self.partner_gene(a, g, self.oix[d]) == j for g in ctx):
            flag |= 1
        for ti in rows:
            pj = self.partner_gene(d, j, ti)
            if pj >= 0 and any(self.partner_gene(a, g, ti) == pj for g in ctx):
                flag |= 2
                break
        return flag

    # ---- tables ----
    def fill(self, out, meta, t, gid_a, gid_d, genes, owner, tgene, n, rows, census_misc):
        """Table t of `out` (v0_io.alloc) and `meta` (alloc_meta): row 0 is A with the blocks `genes` (the target block from
        the first n codons of gene tgene of genome owner), the other rows the partners `rows` (pairs indices)."""
        ga, go, R = self.g[gid_a], self.g[owner], self.rows
        flen, own = np.zeros(B, dtype=np.int64), [None] * B
        for b, (_, s0, s1) in enumerate(SLICES):
            src, j = (go, tgene) if b == TARGET else (ga, genes[b])
            out["gene"][t, b] = (tgene if owner == gid_a else -1) if b == TARGET else genes[b]
            if j < 0:
                continue
            c = src.codons[src.coff[j]:src.coff[j + 1]][:(n if b == TARGET else s1 - s0)]
            own[b], flen[b] = c, len(c)
            meta["block_ends"][t, b] = int(src.coff[j + 1] - src.coff[j]) <= len(c)
            out["tok"][t, 0, s0:s0 + len(c)] = c
            out["col"][t, 0, s0:s0 + len(c)] = np.arange(len(c))
            out["n"][t, 0, b] = len(c)
            out["kind"][t, 0, b] = A.KIND_PRESENT
            if b == TARGET:
                out["start"][t, 0, b], out["strand"][t, 0, b], out["replicon"][t, 0, b] = -1, 0, -1
            else:
                out["start"][t, 0, b], out["strand"][t, 0, b], out["replicon"][t, 0, b] = V.locus(ga, j, 0)
        out["frame_len"][t] = flen
        out["genome"][t, 0] = self.gix[gid_a]
        out["identity"][t, 0] = 1.0
        ident, same, with_col = np.full((R, B), np.nan), np.zeros((R, B), dtype=np.int64), np.zeros((R, B), dtype=np.int64)
        covered = 0
        for q, ti in enumerate(rows, 1):
            pid = self.order[ti]
            partner = self.g[pid]
            out["genome"][t, q] = self.gix[pid]
            meta["row_dist"][t, q] = self.d(gid_a, pid)
            meta["row_dist_d"][t, q] = self.d(gid_d, pid)
            for b in range(B):
                src_gid, j = (owner, tgene) if b == TARGET else (gid_a, genes[b])
                if j < 0:
                    continue
                cell, has = self.cell(src_gid, j, ti, int(flen[b]), SLICES[b][2] - SLICES[b][1])
                if b == TARGET and cell is None:
                    raise AssertionError("a row of the triple without its target cell")
                if not has:
                    out["kind"][t, q, b] = A.KIND_NOT_SEARCHED
                elif cell is None:
                    out["kind"][t, q, b] = A.KIND_TRUNCATED
                else:
                    self.b._put(out, t, q, b, cell, own[b], partner, ident, same, with_col, census_misc)
                    if b == TARGET:
                        covered += int((cell[1] >= 0).sum())
                        meta["row_gene"][t, q] = int(cell[3])
                        out["start"][t, q, b], out["strand"][t, q, b], out["replicon"][t, q, b] = -1, 0, -1
            out["identity"][t, q] = ident[q, TARGET]
        rd = meta["row_dist"][t].astype(np.float64)
        meta["row_tier"][t] = np.where(np.isnan(rd), -1, DI.tier_of(np.where(np.isnan(rd), 0.0, rd), self.edges))
        meta["on_target"][t] = len(rows)
        meta["covered"][t] = covered
        meta["target_len"][t] = int(n)
        meta["target_ends"][t] = bool(meta["block_ends"][t, TARGET])

    def allocation(self, metas, groups, seed, per_tier, cap):
        """[(group, tier, A)] of every triple to draw, and {(group, tier): eligible A with a candidate donor in the tier}."""
        by = {g: sorted(m["id"] for m in metas if m["split"] == "heldout" and m["group"] == g and m["id"] in self.eligible) for g in groups}
        donors = sorted(m["id"] for m in metas if m["split"] == "heldout" and m["id"] in self.eligible)
        tiers = range(len(self.edges) + 2)
        plan, capacity = [], {}
        for gi, g in enumerate(groups):
            if not by[g]:
                continue
            perm = [by[g][int(x)] for x in np.random.default_rng([int(seed), SEED_STREAM, 1, gi]).permutation(len(by[g]))]
            have = {a: {self.tier(a, d) for d in donors if d != a} for a in perm}
            for t in tiers:
                cands = [a for a in perm if t in have[a]]
                capacity[(g, t)] = len(cands)
                n = 0
                for a in cands:
                    if n >= cap:
                        break
                    k = min(int(per_tier), cap - n)
                    plan += [(g, t, a)] * k
                    n += k
        return plan, capacity, donors

    def build(self, metas, groups, seed, per_tier=2, group_tier_cap=30, scenarios=True, progress=None):
        """(out, meta, triples, counts, capacity). Triples of scenario 0 first (in the order of the allocation), then
        their copies of scenario 1; table 3 s + v is variant v of triple s."""
        meta_of = {m["id"]: m for m in metas}
        self.unit_names = sorted(set(self.units.values()))
        unit_ix = {u: k for k, u in enumerate(self.unit_names)}
        plan, capacity, donors_all = self.allocation(metas, groups, seed, per_tier, group_tier_cap)
        uses, drawn = {}, []
        misc = dict.fromkeys(C.MISC, 0)
        for s, (g, t, a) in enumerate(plan):
            if progress and s and s % 100 == 0:
                progress("%d of %d triples drawn" % (s, len(plan)))
            rng = np.random.default_rng([int(seed), SEED_STREAM, 2, s])
            donors = [d for d in donors_all if d != a and self.tier(a, d) == t and uses.get((g, d), 0) < self.donor_cap]
            if not donors:
                self.counts["no_donor_left"] += 1
                continue
            got = self.triple(a, donors, rng)
            if got is None:
                self.counts["failed"] += 1
                continue
            uses[(g, got["d"])] = uses.get((g, got["d"]), 0) + 1
            ga = self.g[a]
            near = self.near(ga, got["i"])
            draw = rng.choice(ga.n, size=min(ga.n, A.N_RELATION + 4), replace=False)
            lens = np.diff(ga.coff)
            rel = [int(x) for x in draw if int(x) not in (got["i"], got["k"]) and int(x) not in near][:A.N_RELATION]
            genes = rel + [-1] * (A.N_RELATION - len(rel)) + near + [got["i"]]
            genes = [x if x >= 0 and lens[x] > 0 else -1 for x in genes]
            rows = [got["rows"][int(x)] for x in rng.permutation(len(got["rows"]))[:self.rows - 1]]
            banned = {self.units[a], self.units[got["d"]]}
            if any(self.units[self.order[ti]] in banned for ti in rows):
                raise AssertionError("a row of the split unit of A or D")
            flag = self.context_flag(a, got["d"], got["j"], genes, rows)
            drawn.append(dict(got, group=g, tier=t, a=a, genes=genes, rows=rows, scenario=0, parent=len(drawn), context_homolog=flag,
                              host=(ga.ccount - ga.gene_count(got["i"]) - ga.gene_count(got["k"]))[1:65].astype(np.int64)))
            self.counts["triples"] += 1
            self.counts["phase%d" % got["phase"]] += 1
            self.counts["attempts"] += got["attempts"]
            self.counts["zero_rows"] += int(not rows)
            self.counts["context_homolog"] += int(flag > 0)
        if scenarios:
            for tr in list(drawn):
                keep = [ti for ti in tr["rows"] if not self.d(self.order[ti], tr["d"]) < self.d(self.order[ti], tr["a"])]
                if len(keep) < len(tr["rows"]):
                    drawn.append(dict(tr, rows=keep, scenario=1, context_homolog=self.context_flag(tr["a"], tr["d"], tr["j"], tr["genes"], keep)))
                    self.counts["copies"] += 1
        out, meta = IO.alloc(3 * len(drawn), self.rows), alloc_meta(3 * len(drawn), self.rows)
        triples = []
        for s, tr in enumerate(drawn):
            a, d = tr["a"], tr["d"]
            for v, (owner, tg) in enumerate(((d, tr["j"]), (a, tr["i"]), (a, tr["k"]))):
                t = 3 * s + v
                out["group"][t], out["split"][t] = groups.index(tr["group"]), V.SPLITS["heldout"]
                self.fill(out, meta, t, a, d, tr["genes"], owner, tg, tr["n"], tr["rows"], misc)
                for k, x in (("triple", s), ("parent", tr["parent"]), ("scenario", tr["scenario"]), ("variant", v), ("donor", self.gix[d]),
                             ("gene_i", tr["i"]), ("gene_j", tr["j"]), ("gene_k", tr["k"]), ("phase", tr["phase"]), ("donor_dist", self.d(a, d)),
                             ("donor_tier", self.tier(a, d)), ("host_counts", tr["host"]), ("genus_a", genus_of(meta_of[a])), ("genus_d", genus_of(meta_of[d])),
                             ("unit_a", unit_ix[self.units[a]]), ("unit_d", unit_ix[self.units[d]]), ("context_homolog", tr["context_homolog"]),
                             ("attempts", tr["attempts"])):
                    meta[k][t] = x
            triples.append(dict(triple=s, parent=tr["parent"], scenario=SCENARIOS[tr["scenario"]], group=tr["group"], tier=tr["tier"], a=a, d=d, i=tr["i"],
                                j=tr["j"], k=tr["k"], n=tr["n"], rows=len(tr["rows"]), phase=tr["phase"], attempts=tr["attempts"],
                                context_homolog=tr["context_homolog"]))
        self.counts.update({"misc_" + k: int(v) for k, v in misc.items()})
        return out, meta, triples, dict(self.counts), {"%s|%d" % k: v for k, v in sorted(capacity.items())}


def verify(out, meta, eligible_ix, allowed_ix, unit_of=None):
    """Problems of built implant tables (empty when sound). unit_of: {genome index: split unit}, to check that no row is of
    the split unit of A or D."""
    bad = []
    kind, genome, n = out["kind"], out["genome"], out["n"].astype(np.int64)
    trip, var = meta["triple"], meta["variant"]
    if not len(trip):
        return ["no table"]
    if not (np.isin(genome[:, 0], list(eligible_ix)).all() and np.isin(meta["donor"], list(eligible_ix)).all()):
        bad.append("an examined genome or a donor that is not eligible")
    other = genome[:, 1:]
    if not np.isin(other[other >= 0], list(allowed_ix)).all():
        bad.append("an other row that is not a training genome")
    if (other == meta["donor"][:, None]).any() or (other == genome[:, :1]).any():
        bad.append("the donor or the examined genome among the other rows")
    if unit_of is not None:
        for t in range(len(trip)):
            banned = {unit_of.get(int(genome[t, 0])), unit_of.get(int(meta["donor"][t]))}
            if any(unit_of.get(int(x)) in banned for x in other[t] if x >= 0):
                bad.append("table %d: a row of the split unit of A or D" % t)
                break
    if not (out["split"] == V.SPLITS["heldout"]).all():
        bad.append("a table outside the held-out split")
    if (out["start"][:, :, TARGET] != -1).any() or (out["strand"][:, :, TARGET] != 0).any() or (out["replicon"][:, :, TARGET] != -1).any():
        bad.append("a position field of the target block is not erased")
    for s in np.unique(trip):
        ts = np.flatnonzero(trip == s)
        if sorted(var[ts].tolist()) != [0, 1, 2]:
            bad.append("triple %d does not have its three tables" % s)
            continue
        if not all((genome[x] == genome[ts[0]]).all() for x in ts):
            bad.append("triple %d: the three tables do not have the same rows" % s)
        rows = genome[ts[0], 1:] >= 0
        if not all((kind[x, 1:, TARGET][rows] == A.KIND_PRESENT).all() for x in ts):
            bad.append("triple %d: a row without content on the target block of one of its tables" % s)
        if len({int(n[x, 0, TARGET]) for x in ts}) != 1:
            bad.append("triple %d: the target blocks differ in length" % s)
        ctx = [np.delete(kind[x], TARGET, axis=1) for x in ts]
        if not all((c == ctx[0]).all() for c in ctx):
            bad.append("triple %d: the kinds outside the target block differ" % s)
    if not ((kind == A.KIND_PRESENT) == (n > 0)).all():
        bad.append("codons and kinds disagree")
    return bad
