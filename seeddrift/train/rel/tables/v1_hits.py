"""Homolog rows of the uncapped search (relational line, package W12): every alignment DIAMOND reports for a query gene
against the training genomes (all targets, all alignments of a pair) and against the query's own genome (its other
copies). Numpy and the standard library only; the tables of version 1 read these files.

Per query genome, `<genome>.npz`:
  off        int64 [genes+1]   the pairs of query gene i at off[i]:off[i+1], ordered by (target genome, target gene)
  tgenome    int32 [pairs]     index of the target genome in the run's genome list (genomes.json of stage w12_search)
  tgene      int32 [pairs]     gene index in the target genome (its genes.tsv order, the order of the training arrays)
  hsp_off    int64 [pairs+1]   the alignments of pair p at hsp_off[p]:hsp_off[p+1], ordered by query start
  qs, qe, ss, se  uint32 [hsps]  1-based first and last residue of the alignment on the query and the target protein
  cigar      uint8 [...]       the CIGAR strings (M, I, D) of the alignments, one after the other
  cigar_off  int64 [hsps+1]
  searched   uint8 [genes]     1 for a query gene that was searched (all of them in a full run; a sample in a trial)
A pair is a (query gene, target gene) and is stored once, however many alignments it has. The query gene itself is not
a pair of its own genome. No bit score, e-value or rank is kept: the order of pairs and alignments carries none.

  FIELDS                           the DIAMOND output fields this module parses (outfmt 6)
  Collector(n_genes)               add(query gene, target genome, target gene, qs, qe, ss, se, cigar); arrays()
  stream(lines, genome_of, gene_index, keep, write)
                                   DIAMOND rows of contiguous query genomes, each written once complete
  merge(a, b)                      the pairs of two arrays of the same query genome (the main and the self search)
  save(path, arrays), load(path)
  columns(cigar, qs, ss)           (query position, target position) of every aligned column (0-based)
  check(arrays, n_targets)         problems of the arrays of one query genome (empty when they are sound)
  write_index(folder, parts)       the lookup index of every query genome's rows, in the order of the genome list
  Index(folder)                    the index, memory-mapped: pairs(query genome, target genome, query gene) -> the pair
                                   indices (a range), tgene(p), alignments(p), lookup(qg, tg, qi) -> [(target gene,
                                   [(qs, qe, ss, se, cigar)])]; a lookup is two binary searches in a memory-mapped slice

The index (index/<field>.npy, plain .npy, memory-mappable): gene_base int64 [genomes+1] (the first global gene of every
query genome), off int64 [genes+1] (the pairs of global gene k at off[k]:off[k+1], ordered by target genome, then
target gene), tgenome, tgene int32 [pairs], hsp_off int64 [pairs+1], qs, qe, ss, se uint32 [alignments], cigar uint8,
cigar_off int64 [alignments+1], searched uint8 [genes]. Query genomes, target genomes and genes are numbered as in the
genome list of the search (genomes.json) and each genome's genes.tsv.
"""
from __future__ import annotations

import re

import numpy as np

FIELDS = ["qseqid", "sseqid", "qstart", "qend", "sstart", "send", "cigar"]
KEYS = ("off", "tgenome", "tgene", "hsp_off", "qs", "qe", "ss", "se", "cigar", "cigar_off", "searched")
_CIG = re.compile(rb"(\d+)([MID])")


class Collector:
    """The rows of one query genome, in any order."""

    def __init__(self, n_genes, searched=None):
        self.n = int(n_genes)
        self.cols = {k: [] for k in ("qgene", "tgenome", "tgene", "qs", "qe", "ss", "se")}
        self.cig = []
        self.searched = np.ones(self.n, dtype=np.uint8) if searched is None else np.asarray(searched, dtype=np.uint8)

    def add(self, qgene, tgenome, tgene, qs, qe, ss, se, cigar):
        c = self.cols
        c["qgene"].append(qgene)
        c["tgenome"].append(tgenome)
        c["tgene"].append(tgene)
        c["qs"].append(qs)
        c["qe"].append(qe)
        c["ss"].append(ss)
        c["se"].append(se)
        self.cig.append(cigar if isinstance(cigar, bytes) else cigar.encode())

    def __len__(self):
        return len(self.cig)

    def arrays(self):
        c = {k: np.asarray(v, dtype=np.int64) for k, v in self.cols.items()}
        order = np.lexsort((c["qs"], c["tgene"], c["tgenome"], c["qgene"])) if len(self.cig) else np.zeros(0, dtype=np.int64)
        c = {k: v[order] for k, v in c.items()}
        cig = [self.cig[i] for i in order]
        h = len(cig)
        key = np.stack([c["qgene"], c["tgenome"], c["tgene"]], 1) if h else np.zeros((0, 3), dtype=np.int64)
        new = np.ones(h, dtype=bool)
        if h > 1:
            new[1:] = (key[1:] != key[:-1]).any(1)
        start = np.flatnonzero(new)
        qgene = c["qgene"][start]
        off = np.searchsorted(qgene, np.arange(self.n + 1), side="left").astype(np.int64)
        lens = np.array([len(x) for x in cig], dtype=np.int64)
        return dict(off=off, tgenome=c["tgenome"][start].astype(np.int32), tgene=c["tgene"][start].astype(np.int32),
                    hsp_off=np.append(start, h).astype(np.int64), qs=c["qs"].astype(np.uint32), qe=c["qe"].astype(np.uint32),
                    ss=c["ss"].astype(np.uint32), se=c["se"].astype(np.uint32),
                    cigar=np.frombuffer(b"".join(cig), dtype=np.uint8).copy(), cigar_off=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
                    searched=self.searched.copy())


def empty(n_genes, searched=None):
    return Collector(n_genes, searched).arrays()


def stream(lines, genome_of, gene_index, keep, write, n_genes, searched=None):
    """Parse DIAMOND rows (FIELDS, tab-separated) whose query genomes come one after the other. genome_of: {genome id:
    index}; gene_index: {genome id: {gene name: index}}; keep(qgenome, qgene, tgenome, tgene) -> bool; write(genome id,
    arrays) is called once per query genome with rows, when its rows are complete; n_genes, searched: {genome id: ...}.
    Returns {query genome id: rows kept}. A query genome whose rows are not contiguous is an error."""
    cur, col, done, kept = None, None, set(), {}
    for line in lines:
        f = line.rstrip("\n").split("\t")
        if len(f) != len(FIELDS):
            raise ValueError("a DIAMOND row with %d fields, %d expected: %r" % (len(f), len(FIELDS), line[:200]))
        qg, qn = f[0].split("__", 1)
        if qg != cur:
            if cur is not None:
                write(cur, col.arrays())
                done.add(cur)
            if qg in done:
                raise ValueError("the rows of query genome %s are not contiguous" % qg)
            cur, col = qg, Collector(n_genes[qg], (searched or {}).get(qg))
            kept[qg] = 0
        tg, tn = f[1].split("__", 1)
        qi, ti, tj = gene_index[qg][qn], genome_of[tg], gene_index[tg][tn]
        if not keep(genome_of[qg], qi, ti, tj):
            continue
        col.add(qi, ti, tj, int(f[2]), int(f[3]), int(f[4]), int(f[5]), f[6])
        kept[qg] += 1
    if cur is not None:
        write(cur, col.arrays())
    return kept


def merge(a, b):
    """The pairs of a and b (arrays of the same query genome) in one set of arrays; a pair in both keeps all its
    alignments (duplicates removed)."""
    n = len(a["off"]) - 1
    if len(b["off"]) - 1 != n:
        raise ValueError("merging arrays of genomes with %d and %d genes" % (n, len(b["off"]) - 1))
    col = Collector(n, np.maximum(a["searched"], b["searched"]))
    seen = set()
    for x in (a, b):
        gene_of_pair = np.repeat(np.arange(n), np.diff(x["off"]))
        for p in range(len(x["tgenome"])):
            for h in range(int(x["hsp_off"][p]), int(x["hsp_off"][p + 1])):
                cig = bytes(x["cigar"][x["cigar_off"][h]:x["cigar_off"][h + 1]])
                key = (int(gene_of_pair[p]), int(x["tgenome"][p]), int(x["tgene"][p]), int(x["qs"][h]), int(x["ss"][h]), cig)
                if key in seen:
                    continue
                seen.add(key)
                col.add(key[0], key[1], key[2], key[3], int(x["qe"][h]), key[4], int(x["se"][h]), cig)
    return col.arrays()


def save(path, arrays):
    from seeddrift.train.logger import save_npz
    save_npz(path, compress=True, **{k: arrays[k] for k in KEYS})


def load(path):
    with np.load(path) as z:
        return {k: z[k] for k in KEYS}


def columns(cigar, qs, ss):
    """(query positions, target positions), 0-based, of the aligned (M) columns of one alignment."""
    q, t, out_q, out_t = int(qs) - 1, int(ss) - 1, [], []
    for n, op in _CIG.findall(bytes(cigar)):
        n = int(n)
        if op == b"M":
            out_q.append(np.arange(q, q + n))
            out_t.append(np.arange(t, t + n))
            q, t = q + n, t + n
        elif op == b"I":
            q += n
        else:
            t += n
    if not out_q:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(out_q), np.concatenate(out_t)


def spans(cigar):
    """(query residues, target residues) an alignment's CIGAR consumes."""
    q = t = 0
    for n, op in _CIG.findall(bytes(cigar)):
        n = int(n)
        q += n if op in (b"M", b"I") else 0
        t += n if op in (b"M", b"D") else 0
    return q, t


def check(arrays, n_targets):
    """Problems of the arrays of one query genome."""
    a, out = arrays, []
    if set(a) != set(KEYS):
        return ["fields %s" % sorted(set(a) ^ set(KEYS))]
    off, hoff, co = a["off"], a["hsp_off"], a["cigar_off"]
    pairs, hsps = len(a["tgenome"]), len(a["qs"])
    if off[0] != 0 or off[-1] != pairs or (np.diff(off) < 0).any() or len(a["searched"]) != len(off) - 1:
        out.append("pair offsets")
    if hoff[0] != 0 or hoff[-1] != hsps or (np.diff(hoff) <= 0).any() or len(hoff) != pairs + 1:
        out.append("alignment offsets (every pair has an alignment)")
    if co[0] != 0 or co[-1] != len(a["cigar"]) or (np.diff(co) <= 0).any() or len(co) != hsps + 1:
        out.append("CIGAR offsets")
    if pairs and ((a["tgenome"] < 0).any() or (a["tgenome"] >= n_targets).any() or (a["tgene"] < 0).any()):
        out.append("target genome or gene out of range")
    if hsps and ((a["qs"] < 1).any() or (a["qe"] < a["qs"]).any() or (a["ss"] < 1).any() or (a["se"] < a["ss"]).any()):
        out.append("alignment coordinates")
    unsearched = np.flatnonzero(a["searched"] == 0)
    if len(unsearched) and (np.diff(off)[unsearched] > 0).any():
        out.append("pairs of a gene that was not searched")
    if not out and hsps:
        for h in range(0, hsps, max(1, hsps // 200)):                    # a sample of the alignments: CIGAR against coordinates
            q, t = spans(a["cigar"][co[h]:co[h + 1]])
            if q != int(a["qe"][h]) - int(a["qs"][h]) + 1 or t != int(a["se"][h]) - int(a["ss"][h]) + 1:
                out.append("a CIGAR that does not span its coordinates (alignment %d)" % h)
                break
    return out


INDEX_FIELDS = dict(gene_base=np.int64, off=np.int64, tgenome=np.int32, tgene=np.int32, hsp_off=np.int64, qs=np.uint32, qe=np.uint32, ss=np.uint32,
                    se=np.uint32, cigar=np.uint8, cigar_off=np.int64, searched=np.uint8)


def write_index(folder, parts):
    """Write the lookup index from `parts`, the arrays of every query genome in the order of the genome list (an
    iterable of callables or dicts, read twice: once for the sizes, once for the content). Returns the sizes."""
    import os
    get = lambda x: x() if callable(x) else x                           # noqa: E731
    sizes = []
    for x in parts:
        a = get(x)
        sizes.append((len(a["off"]) - 1, len(a["tgenome"]), len(a["qs"]), len(a["cigar"])))
    n_genes, n_pairs, n_hsps, n_cig = (sum(z[i] for z in sizes) for i in range(4))
    os.makedirs(folder, exist_ok=True)
    shape = dict(gene_base=len(sizes) + 1, off=n_genes + 1, tgenome=n_pairs, tgene=n_pairs, hsp_off=n_pairs + 1, qs=n_hsps, qe=n_hsps, ss=n_hsps, se=n_hsps,
                 cigar=n_cig, cigar_off=n_hsps + 1, searched=n_genes)
    mm = {k: np.lib.format.open_memmap(os.path.join(folder, k + ".npy"), mode="w+", dtype=dt, shape=(shape[k],)) for k, dt in INDEX_FIELDS.items()}
    g0 = p0 = h0 = c0 = 0
    mm["gene_base"][0] = 0
    for k, x in enumerate(parts):
        a = get(x)
        g, p, h, c = sizes[k]
        if (len(a["off"]) - 1, len(a["tgenome"]), len(a["qs"]), len(a["cigar"])) != (g, p, h, c):
            raise ValueError("the rows of query genome %d changed between the two reads" % k)
        mm["gene_base"][k + 1] = g0 + g
        mm["off"][g0:g0 + g + 1] = a["off"] + p0
        mm["searched"][g0:g0 + g] = a["searched"]
        mm["tgenome"][p0:p0 + p], mm["tgene"][p0:p0 + p] = a["tgenome"], a["tgene"]
        mm["hsp_off"][p0:p0 + p + 1] = a["hsp_off"] + h0
        for f in ("qs", "qe", "ss", "se"):
            mm[f][h0:h0 + h] = a[f]
        mm["cigar"][c0:c0 + c] = a["cigar"]
        mm["cigar_off"][h0:h0 + h + 1] = a["cigar_off"] + c0
        g0, p0, h0, c0 = g0 + g, p0 + p, h0 + h, c0 + c
    for x in mm.values():
        x.flush()
    return dict(genomes=len(sizes), genes=n_genes, pairs=n_pairs, hsps=n_hsps, cigar_bytes=n_cig)


class Index:
    """The lookup index of a search, memory-mapped (see the module doc)."""

    def __init__(self, folder):
        import os
        self.a = {k: np.load(os.path.join(folder, k + ".npy"), mmap_mode="r") for k in INDEX_FIELDS}

    def pairs(self, qg, tg, qi):
        """range of the pair indices of query gene qi of query genome qg with target genome tg."""
        a = self.a
        k = int(a["gene_base"][qg]) + int(qi)
        lo, hi = int(a["off"][k]), int(a["off"][k + 1])
        t = a["tgenome"][lo:hi]
        return range(lo + int(np.searchsorted(t, tg, side="left")), lo + int(np.searchsorted(t, tg, side="right")))

    def tgene(self, p):
        return int(self.a["tgene"][p])

    def alignments(self, p):
        a = self.a
        out = []
        for h in range(int(a["hsp_off"][p]), int(a["hsp_off"][p + 1])):
            out.append((int(a["qs"][h]), int(a["qe"][h]), int(a["ss"][h]), int(a["se"][h]), bytes(a["cigar"][a["cigar_off"][h]:a["cigar_off"][h + 1]])))
        return out

    def lookup(self, qg, tg, qi):
        return [(self.tgene(p), self.alignments(p)) for p in self.pairs(qg, tg, qi)]

