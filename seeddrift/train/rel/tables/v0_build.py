"""Tables of the relational line, version 0 (package W2): genomes x loci cut from the arrays a passed run of pipeline
selfsup already holds (data/train_data and data/pairs). Numpy only; run by ci/stages/rel/tables_v0.py.

A table has one examined genome (row 0) and up to rows - 1 other genomes. Its columns are the 19 blocks of the contract
(seeddrift/train/rel/api.py), all of them stretches of the examined genome: 16 relation blocks (other genes of the
genome drawn at random, their first 32 codons), the left and the right neighbour of the target gene (the genes before
and after it on its replicon in coordinate order, their first 128 codons) and the target gene (drawn at random, its
first 256 codons). The genome's loci are drawn from the examined genome alone: no draw looks at another genome.

The other rows are allowed genomes (training genomes) that have a homolog row of stage pairs on the target gene
starting inside the target block, up to rows - 1 of them drawn at random, in random order. This is the row rule of the
first read-outs only: before the corpus is searched again without caps, having a homolog row is decided by a
similarity ranking (stage pairs keeps the 64 nearest genomes of a gene). Bit scores are dropped before anything is
read, and no draw depends on the order of the homolog rows (which is by bit score).

Kinds of a (row, block), version 0 (the contract's; the files are written and read by v0_io.py):
  KIND_NONE           nothing there: the row does not exist (fewer candidates than rows) or the block does not exist
                      (no neighbour on the replicon, a genome of fewer than 19 genes)
  KIND_PRESENT        codons: for row 0 the block itself; for another row the homolog's stretch under the block, in the
                      homolog's own coordinates (left-justified, cut at the block length), and apart from it the column
                      of each codon in the examined genome's coordinates (-1: a codon the examined gene has no residue
                      for). The columns come from the protein alignment anchored on the examined gene (the stored
                      placement): they know the examined row's residues, so a rendering that uses them is an upper bound
  KIND_NOT_SEARCHED   stage pairs holds no homolog row of this gene in this genome (no homolog, or outside the 64 kept)
  KIND_TRUNCATED      a homolog row exists, but none of its aligned codons lies in the stored parts: the alignment
                      starts after the block, or in the part of the homolog cut off at the source's codon limit
KIND_ABSENT and KIND_SELF_COPY do not occur before the uncapped search.

start, strand, replicon of a (row, block): the position where the stretch begins in its reading direction (the gene's
start plus 3 per codon on the plus strand, its end minus 3 per codon on the minus strand: exact for unspliced genes),
the gene's strand, and the index of its replicon inside its genome.
"""
from __future__ import annotations

import hashlib
import multiprocessing
import os
import time

import numpy as np

from seeddrift.train import selfsup as SS
from seeddrift.train import source_model as S
from seeddrift.train.rel import api as A
from seeddrift.train.rel.tables import v0_census as C
from seeddrift.train.rel.tables import v0_io as IO

VERSION = 0
TABLE_VERSION = "w2-v0-1"                          # the version string a table-set manifest carries
API_VERSION = getattr(A, "API", getattr(A, "API_VERSION", None))      # the version of the contract the shards follow
KIND_NONE = IO.KIND_NONE
SLICES = A.block_slices()
B = A.N_BLOCKS
TARGET, LEFT, RIGHT = B - 1, A.N_RELATION, A.N_RELATION + 1
ROLE_OF_BLOCK = np.array([s[0] for s in SLICES])
SPLITS = {"train": 0, "heldout": 1}
OPEN_SPLITS = ("train", "heldout")                    # the splits a table may show; the sealed test part is never opened
UNIT_TABLES = 512                                     # tables of one examined genome built in one piece
fields, alloc, shard_path = IO.fields, IO.alloc, IO.shard_path      # the files are written and read by v0_io.py


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def locus(gen, j, first):
    """(position where the stretch starting at codon `first` of gene j begins in its reading direction, strand,
    replicon index or -1 when it does not fit the field)."""
    strand = int(gen.strand[j])
    start = int(gen.start[j]) + 3 * int(first) if strand > 0 else int(gen.end[j]) - 3 * int(first)
    rep = int(gen.rep[j])
    return start, strand, rep if rep < 32768 else -1


class Builder:
    """Tables of examined genomes. genomes: {id: Genome} of every genome a table may show (splits train and heldout
    only); allowed: the genomes other rows are drawn from (training genomes only); ids: the genome list the meta field
    `genome` indexes."""

    def __init__(self, pairs, genomes, allowed, ids, rows=16):
        self.pairs, self.g, self.rows = pairs, genomes, int(rows)
        self.allowed = set(allowed)
        bad = sorted(g for g, gen in genomes.items() if gen.meta.get("split") not in OPEN_SPLITS)
        if bad:
            raise SystemExit("tables v0: genomes outside the train and held-out splits were loaded: %s" % bad[:5])
        bad = sorted(g for g in self.allowed if g not in genomes or genomes[g].meta.get("split") != "train")
        if bad:
            raise SystemExit("tables v0: other rows may only be training genomes: %s" % bad[:5])
        if self.rows < 1 or self.rows > 127:
            raise SystemExit("tables v0: rows must be 1..127")
        self.gix = {g: i for i, g in enumerate(ids)}
        self.order = list(pairs.order)
        self.oix = {g: i for i, g in enumerate(self.order)}
        self.allowed_mask = np.array([g in self.allowed for g in self.order], dtype=bool)
        self._state = (None, None)

    def state(self, gid):
        """The examined genome's homolog rows, without their bit scores."""
        if self._state[0] != gid:
            z = self.pairs.of(gid)
            z.pop("bits", None)                                     # never read: scores and ranks enter no array
            gen = self.g[gid]
            off = z["off"].astype(np.int64)
            if len(off) != gen.n + 1:
                raise SystemExit("tables v0: %s has %d genes in its training arrays and %d in stage pairs" % (gid, gen.n, len(off) - 1))
            tg = z["tgenome"].astype(np.int64)
            ok = self.allowed_mask[tg] & (tg != self.oix.get(gid, -1)) if len(tg) else np.zeros(0, dtype=bool)
            self._state = (gid, dict(z=z, off=off, tg=tg, ok=ok, qs=z["qs"].astype(np.int64) - 1, lens=np.diff(gen.coff).astype(np.int64),
                                     others=max(1, len(self.allowed) - (1 if gid in self.allowed else 0))))
        return self._state[1]

    def release(self, gid):
        self.pairs.z.pop(gid, None)
        self._state = (None, None)

    def cell(self, z, r, partner, frame_len, slot):
        """The homolog's stretch of pair row r under the first frame_len codons of the examined gene: (its codons in the
        homolog's coordinates, the column of each codon in the examined gene or -1, the index of its first codon in the
        homolog gene, the homolog gene, aligned codons lost because the stretch is longer than the slot, aligned codons
        kept, and how many of them code the residue the alignment shows); None when no aligned codon of the stored
        parts lies under the block. The homolog's codons are taken as selfsup.c_example takes them: codon k of the
        gene for residue k of the protein."""
        t, tp = SS.pair_track(z["cigar"][z["cigar_off"][r]:z["cigar_off"][r + 1]], z["sseq"][z["sseq_off"][r]:z["sseq_off"][r + 1]], z["ss"][r])
        j = int(z["tgene"][r])
        cb = partner.codons[partner.coff[j]:partner.coff[j + 1]]
        q = int(z["qs"][r]) - 1 + np.arange(len(tp))
        ok = (tp >= 0) & (q < frame_len) & (tp < len(cb))
        if not ok.any():
            return None
        q, tp, t = q[ok], tp[ok], t[ok]
        first = int(tp[0])
        keep = tp < first + slot
        lost = int(len(tp) - keep.sum())
        q, tp, t = q[keep], tp[keep] - first, t[keep]
        codons = cb[first:first + int(tp[-1]) + 1]
        col = np.full(len(codons), -1, dtype=np.int16)
        col[tp] = q
        agree = int((SS.CODE[codons[tp].astype(np.int64)] == t).sum())
        return codons, col, first, j, lost, len(tp), agree

    def _put(self, out, k, q, b, cell, own, partner, ident, same, with_col, misc):
        codons, col, first, j, lost, checked, agree = cell
        s0 = SLICES[b][1]
        n = len(codons)
        out["tok"][k, q, s0:s0 + n] = codons
        out["col"][k, q, s0:s0 + n] = col
        out["n"][k, q, b] = n
        out["kind"][k, q, b] = A.KIND_PRESENT
        out["start"][k, q, b], out["strand"][k, q, b], out["replicon"][k, q, b] = locus(partner, j, first)
        m = col >= 0
        s, w = int((codons[m] == own[col[m]]).sum()), int(m.sum())
        ident[q, b], same[q, b], with_col[q, b] = s / w, s, w
        if lost:
            misc["cut_cells"] += 1
            misc["cut_codons"] += lost
        misc["checked"] += checked
        misc["agree"] += agree

    def table(self, out, k, gid, group, split, rng, census=None):
        """Fill table k of the arrays `out` (alloc) with a table of genome gid; count it into `census`."""
        gen, st, R = self.g[gid], self.state(gid), self.rows
        z, off, lens = st["z"], st["off"], st["lens"]
        out["group"][k], out["split"][k] = group, split
        # the loci, from the examined genome alone: the target gene, its neighbours on the replicon, the relation genes
        i = int(rng.integers(gen.n))
        for _ in range(64):
            if lens[i] > 0:
                break
            i = int(rng.integers(gen.n))
        if lens[i] <= 0:
            raise SystemExit("tables v0: %s has no gene with a codon" % gid)
        p = int(gen.pos[i])
        near = [int(gen.order[x]) if 0 <= x < gen.n and gen.rep[gen.order[x]] == gen.rep[i] else -1 for x in (p - 1, p + 1)]
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
            out["start"][k, 0, b], out["strand"][k, 0, b], out["replicon"][k, 0, b] = locus(gen, j, 0)
        out["frame_len"][k] = flen
        out["genome"][k, 0] = self.gix[gid]
        out["identity"][k, 0] = 1.0
        # the other rows: allowed genomes with a homolog row that starts inside the target block, drawn at random. The
        # candidates are put in genome order first, so the draw does not depend on the order of the rows (bit score)
        ident, same, with_col = np.full((R, B), np.nan), np.zeros((R, B), dtype=np.int64), np.zeros((R, B), dtype=np.int64)
        misc = dict.fromkeys(C.MISC, 0)
        tg = st["tg"]
        cand = sorted((int(tg[r]), r) for r in self.pairs.rows(gid, i, self.allowed, n_codons=int(flen[TARGET])) if self.order[int(tg[r])] != gid)
        n_cand = len({c[0] for c in cand})
        slot = {}
        for x in rng.permutation(len(cand)):
            if len(slot) == R - 1:
                break
            ti, r = cand[int(x)]
            if ti in slot:
                misc["duplicate_rows"] += 1
                continue
            pid = self.order[ti]
            cell = self.cell(z, r, self.g[pid], int(flen[TARGET]), A.LEN_TARGET)
            if cell is None:
                misc["rows_without_codons"] += 1
                continue
            q = len(slot) + 1
            slot[ti] = q
            out["genome"][k, q] = self.gix[pid]
            self._put(out, k, q, TARGET, cell, own[TARGET], self.g[pid], ident, same, with_col, misc)
            out["identity"][k, q] = ident[q, TARGET]
        # their cells on the other blocks: the homolog row of that gene in that genome, where stage pairs holds one
        sel = np.fromiter(slot, dtype=np.int64, count=len(slot))
        rnd = np.full(B, np.nan)
        for b in range(B):
            j = genes[b]
            if j < 0:
                continue
            a, e = int(off[j]), int(off[j + 1])
            rnd[b] = float((st["ok"][a:e] & (st["qs"][a:e] < flen[b])).sum()) / st["others"]
            if b == TARGET or not slot:
                continue
            found = {}
            for x in np.flatnonzero(np.isin(tg[a:e], sel)):
                ti = int(tg[a + x])
                if ti in found:
                    misc["duplicate_rows"] += 1
                else:
                    found[ti] = a + int(x)
            for ti, q in slot.items():
                r = found.get(ti)
                if r is None:
                    out["kind"][k, q, b] = A.KIND_NOT_SEARCHED
                    continue
                pid = self.order[ti]
                cell = self.cell(z, r, self.g[pid], int(flen[b]), SLICES[b][2] - SLICES[b][1])
                if cell is None:
                    out["kind"][k, q, b] = A.KIND_TRUNCATED
                else:
                    self._put(out, k, q, b, cell, own[b], self.g[pid], ident, same, with_col, misc)
        if census is not None:
            census.add(group, ROLE_OF_BLOCK, out["kind"][k], ident, out["n"][k], same, with_col, rnd, n_cand, misc)


def plan_set(metas, groups, split, per_group, seed):
    """(examined genome of every table, its group index, groups without a genome in the split). Table t belongs to
    group t mod (the groups that have a genome in the split), so every stretch of tables holds the groups equally; inside
    a group the genomes take turns in a seeded order, so they are examined equally often."""
    tag = SPLITS[split]
    by = {g: sorted(m["id"] for m in metas if m["split"] == split and m["group"] == g and int(m.get("genes", 1)) > 0) for g in groups}
    present = [g for g in groups if by[g]]
    perm = {g: np.random.default_rng([int(seed), tag, 1, groups.index(g)]).permutation(len(by[g])) for g in present}
    genome_of, group_of = [], []
    for t in range(int(per_group) * len(present)):
        g = present[t % len(present)]
        genome_of.append(by[g][int(perm[g][(t // len(present)) % len(by[g])])])
        group_of.append(groups.index(g))
    return genome_of, group_of, [g for g in groups if not by[g]]


def table_rng(seed, split, t):
    return np.random.default_rng([int(seed), SPLITS[split], 0, int(t)])


def _load(args):
    meta, path = args
    if meta.get("split") not in OPEN_SPLITS:
        raise SystemExit("tables v0: refusing to open %s (split %s)" % (meta.get("id"), meta.get("split")))
    gen = S.Genome(meta, path)
    for name in ("hoff", "hprot", "hqs", "track", "toff"):          # the reference hits of the old line: not used by a table
        setattr(gen, name, None)
    lens = np.diff(gen.coff)
    return meta["id"], gen, dict(group=meta["group"], split=meta["split"], genes=int(gen.n), codons=int(len(gen.codons)),
                                 long_genes=int((lens > A.LEN_TARGET).sum()))


def _pool(processes):
    return multiprocessing.get_context("fork").Pool(int(processes))


def load_genomes(folder, metas, processes=1):
    """({id: Genome}, {id: counts}) of the genomes of `metas` (train and held-out only) from a train_data folder."""
    todo = [(m, os.path.join(folder, m["id"] + ".npz")) for m in metas]
    if processes > 1 and len(todo) > 1:
        with _pool(processes) as pool:
            res = pool.map(_load, todo, chunksize=4)
    else:
        res = [_load(x) for x in todo]
    return {g: gen for g, gen, _ in res}, {g: s for g, _, s in res}


_W = {}                                                              # what the forked workers of build_set read


def _unit(args):
    gid, tables = args
    w, t0 = _W, time.time()
    b = w["builder"]
    out = alloc(len(tables), b.rows)
    cen = C.Census(len(w["groups"]), b.rows)
    for k, t in enumerate(tables):
        b.table(out, k, gid, w["group_of"][t], SPLITS[w["split"]], table_rng(w["seed"], w["split"], t), cen)
    b.release(gid)
    return tables, out, cen.to_arrays(), time.time() - t0


def build_set(pairs, genomes, allowed, ids, groups, split, plan, out_dir, rows=16, shard_tables=2400, seed=1, processes=1, progress=None):
    """Build the tables of a plan (plan_set) into shards under out_dir. Returns ([(shard name, tables)], Census). The
    result depends on the seed and the data only: every table has its own random stream, whatever the process count.
    Worker processes build the tables of one examined genome at a time; this process alone writes the shard files."""
    genome_of, group_of = plan[0], plan[1]
    n, shard_tables = len(genome_of), int(shard_tables)
    shards, mm = IO.create(out_dir, n, rows, shard_tables)
    by = {}
    for t, g in enumerate(genome_of):
        by.setdefault(g, []).append(t)
    todo = [(g, ts[x:x + UNIT_TABLES]) for g, ts in sorted(by.items()) for x in range(0, len(ts), UNIT_TABLES)]
    todo.sort(key=lambda u: (-len(u[1]), u[0], u[1][0]))
    _W.clear()
    _W.update(builder=Builder(pairs, genomes, allowed, ids, rows), groups=list(groups), group_of=group_of, split=split, seed=int(seed))
    census, done, t0 = C.Census(len(groups), rows), 0, time.time()
    pool = _pool(processes) if processes > 1 and len(todo) > 1 else None
    try:
        for tables, out, arrays, _ in (pool.imap_unordered(_unit, todo, chunksize=1) if pool else map(_unit, todo)):
            at = np.asarray(tables, dtype=np.int64)
            for s in np.unique(at // shard_tables):
                m = at // shard_tables == s
                for name in out:
                    mm[int(s), name][at[m] - int(s) * shard_tables] = out[name][m]
            census.merge(C.Census.from_arrays(arrays, rows))
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
        raise SystemExit("tables v0: %d of %d tables were built" % (done, n))
    return shards, census


def hash_files(paths, processes=1):
    """sha256 of every path, in order."""
    paths = list(paths)
    if processes > 1 and len(paths) > 1:
        with _pool(processes) as pool:
            return pool.map(sha256_of, paths, chunksize=4)
    return [sha256_of(p) for p in paths]


def hash_shards(out_dir, shards, rows, processes=1):
    """{shard: {field: dict(sha256, bytes)}} of the files of a set."""
    paths = [(name, f, os.path.join(out_dir, "%s.%s.npy" % (name, f))) for name, _ in shards for f in fields(rows)]
    out = {}
    for (name, f, p), sha in zip(paths, hash_files([p for _, _, p in paths], processes)):
        out.setdefault(name, {})[f] = dict(sha256=sha, bytes=os.path.getsize(p))
    return out


def genomes_in(out_dir, shards):
    """(genome indices of the examined rows, genome indices of the other rows) that occur in a set."""
    g = [np.load(os.path.join(out_dir, "%s.genome.npy" % name)) for name, _ in shards]
    other = np.unique(np.concatenate([x[:, 1:].ravel() for x in g]))
    return np.unique(np.concatenate([x[:, 0] for x in g])), other[other >= 0]


def _verify(args):
    out_dir, name, tables, rows, split, examined, allowed, n_groups = args
    z = {f: np.load(os.path.join(out_dir, "%s.%s.npy" % (name, f)), mmap_mode="r") for f in fields(rows)}
    bad = []

    def need(cond, what):
        if not bool(cond):
            bad.append("%s: %s" % (name, what))
    for f, (dt, shape) in fields(rows).items():
        need(z[f].dtype == dt and z[f].shape == (tables,) + shape, "field %s has dtype %s and shape %s" % (f, z[f].dtype, z[f].shape))
    if bad:
        return bad
    kind, genome, n = np.asarray(z["kind"]), np.asarray(z["genome"]), np.asarray(z["n"]).astype(np.int64)
    need((kind[:, 0, TARGET] == A.KIND_PRESENT).all(), "a table was not written (row 0 has no target block)")
    need(np.isin(kind, (KIND_NONE, A.KIND_PRESENT, A.KIND_NOT_SEARCHED, A.KIND_TRUNCATED)).all(), "a kind outside version 0")
    need(np.isin(genome[:, 0], examined).all(), "an examined genome outside the %s split" % split)
    other = genome[:, 1:]
    need(np.isin(other[other >= 0], allowed).all(), "another row that is not a training genome")
    need((other != genome[:, :1]).all(), "the examined genome among the other rows")
    srt = np.sort(other, axis=1)
    need(((srt[:, 1:] != srt[:, :-1]) | (srt[:, 1:] < 0)).all(), "a genome twice in a table")
    has = genome >= 0
    need((has[:, :-1] | ~has[:, 1:]).all(), "rows are not packed at the front")
    need(((kind > 0).any(2) <= has).all() and (kind[:, 1:, TARGET][has[:, 1:]] == A.KIND_PRESENT).all(), "a row without a genome or without its target cell")
    need(((kind == A.KIND_PRESENT) == (n > 0)).all(), "codons and kinds disagree")
    need((np.asarray(z["frame_len"]) == n[:, 0]).all() and (np.asarray(z["frame_row"]) == 0).all(), "the frame is not row 0")
    need((np.asarray(z["split"]) == SPLITS[split]).all(), "a table of another split")
    grp = np.asarray(z["group"])
    need(((grp >= 0) & (grp < n_groups)).all(), "a group outside the list")
    idn = np.asarray(z["identity"])
    need((np.isnan(idn) == ~has).all() and (idn[has] >= 0).all() and (idn[has] <= 1).all(), "identity is not a share on exactly the rows that exist")
    need((np.asarray(z["tier"]) == -1).all(), "a distance tier before tiers are defined")
    for b, (_, s0, s1) in enumerate(SLICES):
        tok, col = np.asarray(z["tok"][:, :, s0:s1]), np.asarray(z["col"][:, :, s0:s1])
        inside = np.arange(s1 - s0)[None, None, :] < n[:, :, b, None]
        need((tok[inside] >= A.TARGET_MIN).all() and (tok[inside] <= A.TARGET_MAX).all() and (tok[~inside] == A.PAD).all(), "block %d: tokens and lengths disagree" % b)
        need((col[~inside] == -1).all() and (col >= -1).all() and (col < np.asarray(z["frame_len"])[:, None, b, None]).all(), "block %d: a column outside the frame" % b)
    return bad


def verify_set(out_dir, shards, rows, split, examined, allowed, n_groups, processes=1):
    """Problems of a written set (empty when it is sound): every table written, the fields of the contract, examined
    genomes of the split, other rows among the allowed genomes only, tokens, lengths and columns consistent."""
    todo = [(out_dir, name, m, rows, split, np.asarray(sorted(examined), dtype=np.int64), np.asarray(sorted(allowed), dtype=np.int64), n_groups) for name, m in shards]
    if processes > 1 and len(todo) > 1:
        with _pool(processes) as pool:
            res = pool.map(_verify, todo, chunksize=1)
    else:
        res = [_verify(x) for x in todo]
    return [p for r in res for p in r]
