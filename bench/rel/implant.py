"""Implant tables (package W18): the first version of the ruler sets of the relational line. numpy only.

A ruler, never a training table and never an item of a selection round: the stages that call this module are
evaluation stages, and no train-side module imports it (tests/ci/test_rel_c0_imports.py). It imports the table code of
package W2 (seeddrift/train/rel/tables/), which never imports it.

A triple is three tables of one held-out examined genome A at the locus of one of its genes i, with the same rows:
  implant   the target block holds the first codons of gene j of another held-out genome D (the donor); the other rows'
            target cells are their homolog rows of D:j, placed on D:j (the stored placement, as tables of version 0)
  original  the target block holds A:i (the table as it is, with the triple's rows)
  own       the target block holds another gene k of A (not i, not a neighbour of i, not a relation block): it breaks
            the synteny as the implant does, not the genome of origin
The neighbours and relation blocks are A's, drawn from A alone as in tables of version 0 (relation genes avoid i, its
neighbours and k). The other rows are training genomes that hold, on each of A:i, D:j and A:k, a homolog row of stage
pairs starting inside the stored block with at least one aligned codon under it, up to rows - 1 of them drawn at random,
the same rows in the same order in the three tables: so the three tables have the same rows with content on the target
block. To find such rows the genes are drawn in turn (Implanter.draw): i among A's genes with a partner, j among D's
length-matched genes that share a partner with i, k among A's length-matched genes that share one with both. D is never a row (rows are training genomes; D is held out). A and D have the standard genetic code (the caller
decides which genomes are eligible). The donor is drawn by the tier of its distance to A, the tiers taken in turn.

Kinds and fields are those of tables of version 0 (v0_build.py), with these differences: row 0's position fields
(start, strand, replicon) on the target block are unknown (-1, 0, -1) in all three tables; the meta field `gene` of the
target block is -1 for an implant (the gene is not A's; w18_gene_j names it); `tier` stays -1 (the contract's tiers are
not defined yet). W18's own meta (never a model input) is written beside each shard as <shard>.w18_<name>.npy (FIELDS).
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel.tables import v0_build as V
from seeddrift.train.rel.tables import v0_census as C
from seeddrift.train.rel.tables import v0_dist as DI
from seeddrift.train.rel.tables import v0_io as IO

TABLE_VERSION = "w18-implant-1"
VARIANTS = ("implant", "original", "own")
IMPLANT, ORIGINAL, OWN = 0, 1, 2
STANDARD_CODES = (1, 11)                 # the same codon to amino-acid map
SLICES, B = V.SLICES, V.B
TARGET, LEFT, RIGHT = V.TARGET, V.LEFT, V.RIGHT
SEED_STREAM = 18
# W18's meta per table: (dtype, shape after the table axis, what it is); never a model input
FIELDS = {
    "triple":       ("int32", (), "the triple of the table"),
    "variant":      ("int8", (), "0 implant, 1 original, 2 own"),
    "donor":        ("int32", (), "index of D in the genome list of the set's index (the same in the three tables of a triple)"),
    "gene_i":       ("int32", (), "A's gene at the locus"),
    "gene_j":       ("int32", (), "D's implanted gene"),
    "gene_k":       ("int32", (), "A's other gene (the own control)"),
    "donor_dist":   ("float32", (), "distance from D to A (inf without a shared k-mer)"),
    "donor_tier":   ("int8", (), "tier of donor_dist"),
    "row_dist":     ("float32", ("R",), "distance of every row to A (nan for row 0 and rows that do not exist)"),
    "row_tier":     ("int8", ("R",), "tier of row_dist (-1 for row 0 and rows that do not exist)"),
    "on_target":    ("int8", (), "other rows with content on the target block"),
    "covered":      ("int32", (), "codons of the other rows placed on the target block"),
    "target_len":   ("int16", (), "stored codons of the target block of row 0"),
    "target_ends":  ("bool", (), "the stored target reaches the gene's last codon"),
    "host_counts":  ("int32", (64,), "A's codon counts (tokens 1..64) over its stored genes other than i and k"),
    "genus_a":      ("int64", (), "genus taxid of A (0: none)"),
    "genus_d":      ("int64", (), "genus taxid of D (0: none)"),
    "attempts":     ("int16", (), "draws the triple took"),
}


def meta_fields(rows):
    return {k: (np.dtype(dt), tuple(int(rows) if s == "R" else s for s in shape)) for k, (dt, shape, _) in FIELDS.items()}


def alloc_meta(n, rows):
    fill = dict(row_dist=np.nan, row_tier=-1, donor=-1, gene_i=-1, gene_j=-1, gene_k=-1, donor_tier=-1, triple=-1, variant=-1)
    return {k: np.full((int(n),) + shape, fill.get(k, 0), dtype=dt) for k, (dt, shape) in meta_fields(rows).items()}


def stored_len(gen, j):
    return int(min(int(gen.coff[j + 1] - gen.coff[j]), A.LEN_TARGET))


def genus_of(meta):
    r = meta.get("ranks") or []
    return int(r[4]) if len(r) > 4 and r[4] else 0


class Implanter:
    """Triples of implant tables.

    pairs: selfsup.Pairs of the source run; genomes: {id: Genome} of the open splits; allowed: the training genomes (rows);
    ids: the genome list the meta field `genome` indexes; dist: [G, G] distances between the genomes of dist_ids (W2's
    distance run); edges: the tier edges; eligible: the genomes that may be A or D (held out, standard code)."""

    def __init__(self, pairs, genomes, allowed, ids, dist, dist_ids, edges, eligible, rows=16, min_codons=64, length_tol=0.15,
                 attempts=40, donor_cap=3):
        self.b = V.Builder(pairs, genomes, allowed, ids, rows)
        self.pairs, self.g, self.rows = pairs, genomes, int(rows)
        self.order = self.b.order
        self.gix = self.b.gix
        self.dist = np.asarray(dist)
        self.dix = {g: k for k, g in enumerate(dist_ids)}
        missing = [g for g in list(genomes) if g not in self.dix]
        if missing:
            raise SystemExit("implant tables: genomes without a distance: %s" % missing[:5])
        self.edges = [float(x) for x in edges]
        self.eligible = set(eligible)
        self.allowed_ix = {self.b.oix[g] for g in allowed}
        self.min_codons, self.tol, self.attempts, self.donor_cap = int(min_codons), float(length_tol), int(attempts), int(donor_cap)
        self._rows = {}
        self.counts = dict(triples=0, zero_rows=0, failed=0, attempts=0, duplicate_rows=0)

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

    def starts_inside(self, gid, gene, n_codons):
        """The allowed partners whose homolog row of the gene starts inside its first n_codons codons."""
        z = self.pairs.of(gid)
        return {ti for ti, r in self.first_rows(gid, gene).items() if ti in self.allowed_ix and int(z["qs"][r]) - 1 < n_codons}

    def cell(self, gid, gene, ti, n_codons, slot):
        r = self.first_rows(gid, gene).get(ti)
        if r is None:
            return None, False
        return self.b.cell(self.pairs.of(gid), r, self.g[self.order[ti]], int(n_codons), int(slot)), True

    def common_rows(self, targets):
        """The allowed partners with a cell under every (genome, gene, stored length) of targets."""
        sets = [self.starts_inside(g, j, n) for g, j, n in targets]
        common = set.intersection(*sets) if sets else set()
        return sorted(ti for ti in common if all(self.cell(g, j, ti, n, A.LEN_TARGET)[0] is not None for g, j, n in targets))

    def release(self, gid):
        self.pairs.z.pop(gid, None)
        self._rows = {k: v for k, v in self._rows.items() if k[0] != gid}

    # ---- loci ----
    def near(self, gen, i):
        p = int(gen.pos[i])
        return [int(gen.order[x]) if 0 <= x < gen.n and gen.rep[gen.order[x]] == gen.rep[i] else -1 for x in (p - 1, p + 1)]

    def tier(self, a, b):
        return int(DI.tier_of(np.array([self.dist[self.dix[a], self.dix[b]]]), self.edges)[0])

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

    def draw(self, gid_a, donors, rng, condition=True):
        """One attempt: (i, D, j, k) or None. donors: the candidate donors of this attempt's tier. With condition, i is
        drawn among A's genes with a partner, j among D's genes that share a partner with i, k among A's genes that
        share a partner with both (each uniform among the genes that qualify, lengths matched); without, uniform among the
        length-matched genes."""
        ga = self.g[gid_a]
        lens_a = np.diff(ga.coff)
        sa = self.partner_sets(gid_a) if condition else None
        cand_i = np.array([x for x in np.flatnonzero(lens_a >= self.min_codons) if not condition or len(sa[x])], dtype=np.int64)
        if not len(cand_i) or not donors:
            return None
        i = int(cand_i[int(rng.integers(len(cand_i)))])
        li = stored_len(ga, i)
        lo, hi = li * (1 - self.tol), li * (1 + self.tol)
        d = donors[int(rng.integers(len(donors)))]
        gd = self.g[d]
        sl_d = np.minimum(np.diff(gd.coff), A.LEN_TARGET)
        cj = np.flatnonzero((sl_d >= lo) & (sl_d <= hi))
        if condition:
            sd = self.partner_sets(d)
            cj = np.array([x for x in cj if len(np.intersect1d(sa[i], sd[x], assume_unique=True))], dtype=np.int64)
        if not len(cj):
            return None
        j = int(cj[int(rng.integers(len(cj)))])
        bad = set(self.near(ga, i)) | {i}
        sl_a = np.minimum(lens_a, A.LEN_TARGET)
        ck = [int(x) for x in np.flatnonzero((sl_a >= lo) & (sl_a <= hi)) if int(x) not in bad]
        if condition:
            both = np.intersect1d(sa[i], sd[j], assume_unique=True)
            ck = [x for x in ck if len(np.intersect1d(both, sa[x], assume_unique=True))]
        if not ck:
            return None
        return i, d, j, int(ck[int(rng.integers(len(ck)))])

    def triple(self, gid_a, tiers, donors_by_tier, rng):
        """The draws of one triple: (i, D, j, k, rows, attempts) or None when no attempt found a gene pair. tiers: the tiers
        to try, in order (the first that has a donor left is used). Up to `attempts` conditioned draws (draw) until the
        three genes have a common row with a cell under each; failing that, up to `attempts` unconditioned draws give a
        triple without other rows (it serves the composition readout only)."""
        ga = self.g[gid_a]
        n = 0
        for t in tiers:
            donors = donors_by_tier.get(t) or []
            if not donors:
                continue
            for _ in range(self.attempts):
                n += 1
                got = self.draw(gid_a, donors, rng)
                if got is None:
                    continue
                i, d, j, k = got
                rows = self.common_rows([(gid_a, i, stored_len(ga, i)), (d, j, stored_len(self.g[d], j)), (gid_a, k, stored_len(ga, k))])
                if rows:
                    return i, d, j, k, rows, n
            for _ in range(self.attempts):                                # no common row in this tier: a triple of zero rows
                n += 1
                got = self.draw(gid_a, donors, rng, condition=False)
                if got is not None:
                    return got + ([], n)
            break
        return None

    # ---- tables ----
    def fill(self, out, meta, t, gid_a, genes, owner, tgene, rows, census_misc):
        """Table t of `out` (v0_io.alloc) and `meta` (alloc_meta): row 0 is A with the blocks `genes` (the target block from
        gene tgene of genome owner), the other rows the partners `rows` (pairs indices)."""
        ga, go, R = self.g[gid_a], self.g[owner], self.rows
        flen, own = np.zeros(B, dtype=np.int64), [None] * B
        for b, (_, s0, s1) in enumerate(SLICES):
            src, j = (go, tgene) if b == TARGET else (ga, genes[b])
            out["gene"][t, b] = (tgene if owner == gid_a else -1) if b == TARGET else genes[b]
            if j < 0:
                continue
            c = src.codons[src.coff[j]:src.coff[j + 1]][:s1 - s0]
            own[b], flen[b] = c, len(c)
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
            meta["row_dist"][t, q] = self.dist[self.dix[gid_a], self.dix[pid]]
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
            out["identity"][t, q] = ident[q, TARGET]
        rd = meta["row_dist"][t].astype(np.float64)
        tiers = DI.tier_of(np.where(np.isnan(rd), 0.0, rd), self.edges)
        meta["row_tier"][t] = np.where(np.isnan(rd), -1, tiers)
        meta["on_target"][t] = len(rows)
        meta["covered"][t] = covered
        meta["target_len"][t] = int(flen[TARGET])
        meta["target_ends"][t] = int(go.coff[tgene + 1] - go.coff[tgene]) <= A.LEN_TARGET

    def build(self, metas, groups, per_group, seed, progress=None):
        """(out, meta, triples, counts): 3 x per_group x (groups with an eligible A) tables. Triple s belongs to group
        s mod the groups present, so every prefix holds the groups equally; table 3 s + v is variant v of triple s."""
        by = {g: sorted(m["id"] for m in metas if m["split"] == "heldout" and m["group"] == g and m["id"] in self.eligible) for g in groups}
        present = [g for g in groups if by[g]]
        donors_all = sorted(m["id"] for m in metas if m["split"] == "heldout" and m["id"] in self.eligible and m["id"] in self.g)
        meta_of = {m["id"]: m for m in metas}
        n_tr = int(per_group) * len(present)
        out, meta = IO.alloc(3 * n_tr, self.rows), alloc_meta(3 * n_tr, self.rows)
        misc = dict.fromkeys(C.MISC, 0)
        uses, turn, triples = {}, {}, []
        perm = {g: np.random.default_rng([int(seed), SEED_STREAM, 1, groups.index(g)]).permutation(len(by[g])) for g in present}
        tier_order = {}
        for s in range(n_tr):
            g = present[s % len(present)]
            a = by[g][int(perm[g][(s // len(present)) % len(by[g])])]
            rng = np.random.default_rng([int(seed), SEED_STREAM, 2, s])
            if a not in tier_order:
                tiers = sorted({self.tier(a, d) for d in donors_all if d != a})
                tier_order[a] = [tiers[int(x)] for x in np.random.default_rng([int(seed), SEED_STREAM, 3, self.gix[a]]).permutation(len(tiers))]
            order = tier_order[a]
            if not order:
                self.counts["failed"] += 1
                triples.append(dict(triple=s, group=g, a=a, status="no donor"))
                continue
            n_a = turn.get(a, 0)
            turn[a] = n_a + 1
            start = n_a % len(order)
            tiers = order[start:] + order[:start]
            donors_by_tier = {}
            for d in donors_all:
                if d != a and uses.get((g, d), 0) < self.donor_cap:
                    donors_by_tier.setdefault(self.tier(a, d), []).append(d)
            got = self.triple(a, tiers, donors_by_tier, rng)
            if got is None:
                self.counts["failed"] += 1
                triples.append(dict(triple=s, group=g, a=a, status="no gene pair"))
                continue
            i, d, j, k, cand, n_att = got
            uses[(g, d)] = uses.get((g, d), 0) + 1
            self.counts["attempts"] += n_att
            rows = [cand[int(x)] for x in rng.permutation(len(cand))[:self.rows - 1]]
            if not rows:
                self.counts["zero_rows"] += 1
            ga = self.g[a]
            near = self.near(ga, i)
            draw = rng.choice(ga.n, size=min(ga.n, A.N_RELATION + 4), replace=False)
            lens = np.diff(ga.coff)
            rel = [int(x) for x in draw if int(x) not in (i, k) and int(x) not in near][:A.N_RELATION]
            genes = rel + [-1] * (A.N_RELATION - len(rel)) + near + [i]
            genes = [x if x >= 0 and lens[x] > 0 else -1 for x in genes]
            host = (ga.ccount - ga.gene_count(i) - ga.gene_count(k))[1:65].astype(np.int64)
            for v, (owner, tg) in enumerate(((d, j), (a, i), (a, k))):
                t = 3 * s + v
                out["group"][t], out["split"][t] = groups.index(g), V.SPLITS["heldout"]
                self.fill(out, meta, t, a, genes, owner, tg, rows, misc)
                meta["triple"][t], meta["variant"][t], meta["donor"][t] = s, v, self.gix[d]
                meta["gene_i"][t], meta["gene_j"][t], meta["gene_k"][t] = i, j, k
                meta["donor_dist"][t] = self.dist[self.dix[a], self.dix[d]]
                meta["donor_tier"][t] = self.tier(a, d)
                meta["host_counts"][t] = host
                meta["genus_a"][t], meta["genus_d"][t] = genus_of(meta_of[a]), genus_of(meta_of[d])
                meta["attempts"][t] = n_att
            self.counts["triples"] += 1
            self.release(d)                                               # read again when needed; bounds the memory
            triples.append(dict(triple=s, group=g, a=a, d=d, i=i, j=j, k=k, rows=len(rows), tier=self.tier(a, d), attempts=n_att, status="ok"))
            if progress and (s + 1) % 50 == 0:
                progress("%d of %d triples" % (s + 1, n_tr))
        written = meta["triple"] >= 0
        keep = np.flatnonzero(written)
        out = {k: v[keep] for k, v in out.items()}
        meta = {k: v[keep] for k, v in meta.items()}
        self.counts.update({"misc_" + k: int(v) for k, v in misc.items()})
        return out, meta, triples, dict(self.counts)


def verify(out, meta, eligible_ix, allowed_ix):
    """Problems of built implant tables (empty when sound)."""
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
    if not (out["split"] == V.SPLITS["heldout"]).all():
        bad.append("a table outside the held-out split")
    for s in np.unique(trip):
        ts = np.flatnonzero(trip == s)
        if sorted(var[ts].tolist()) != [0, 1, 2]:
            bad.append("triple %d does not have its three tables" % s)
            continue
        if not all((genome[x] == genome[ts[0]]).all() for x in ts):
            bad.append("triple %d: the three tables do not have the same rows" % s)
        on = [(kind[x, 1:, TARGET] == A.KIND_PRESENT).sum() for x in ts]
        if len(set(on)) != 1 or on[0] != int((genome[ts[0], 1:] >= 0).sum()):
            bad.append("triple %d: not every row has content on the target block in every table" % s)
        ctx = [np.delete(kind[x], TARGET, axis=1) for x in ts]
        if not all((c == ctx[0]).all() for c in ctx):
            bad.append("triple %d: the kinds outside the target block differ" % s)
    if not ((kind == A.KIND_PRESENT) == (n > 0)).all():
        bad.append("codons and kinds disagree")
    return bad
