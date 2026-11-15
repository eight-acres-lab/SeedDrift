#!/usr/bin/env python
"""Data for the array model: genes of a genome (columns) against reference proteomes (rows).

No taxonomy is read here. A reference proteome is only an index into the library table.

  columns   one cell per gene and proteome: the gene's best hit in that proteome (from a deep DIAMOND search)
  profile   per genome, its distance to every proteome: a robust rank-one fit over the genome's genes
            (similarity of gene y to proteome h ~ rate of y x distance of the genome to h), defined only where
            at least a tenth of the genome's genes have a hit, so that a block of foreign genes cannot pull a
            proteome next to the genome; also draws the panel
  align     for every cell in a panel proteome, the local alignment of the member to the gene, written in the
            gene's coordinates (one byte per gene position). This is the raw material the model reads.

usage (repo root, system python with numpy and parasail):
  python research/array/build.py columns --library DIR/library.npz --hits hits_a.tsv --out DIR/cols_a.npz
  python research/array/build.py columns --library DIR/library.npz --hits hits_bench_x.tsv --genome x --out DIR/cols_x.npz
  python research/array/build.py profile --columns DIR/cols_*.npz --panel 2048 --out DIR/profiles.npz
  python research/array/build.py align --profiles DIR/profiles.npz --columns DIR/cols_a.npz --fasta queries_a.faa \
      --reference reference.faa --out DIR/a --procs 14
"""
from __future__ import annotations

import argparse
import array
import os
import sys
import time
import warnings

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
OTHER, GAP, NOALN, PAD = 20, 21, 22, 23      # codes after the 20 amino acids
LMAX = 2500                                   # longer proteins are cut for the alignment
LUT = np.full(256, OTHER, dtype=np.int8)
for _i, _c in enumerate(AA):
    LUT[ord(_c)] = _i
    LUT[ord(_c.lower())] = _i
LUT[ord("-")] = GAP


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=sys.stderr, flush=True)


def read_fasta(paths):
    out, name, seq = {}, None, []
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if line[:1] == ">":
                    if name is not None:
                        out[name] = "".join(seq)
                    name, seq = line[1:].split(None, 1)[0], []
                else:
                    seq.append(line.strip())
        if name is not None:
            out[name] = "".join(seq)
            name, seq = None, []
    return out


# ---------------------------------------------------------------- columns

def cmd_columns(a):
    lib = np.load(a.library)
    pidx = {int(t): i for i, t in reversed(list(enumerate(lib["taxid"])))}
    g_id, g_len, g_genome, off = [], [], [], [0]
    P, QS, QE, TLEN = array.array("i"), array.array("i"), array.array("i"), array.array("i")
    BITS, PID = array.array("f"), array.array("f")
    ACC, cur, seen, unknown = [], None, set(), 0
    for path in a.hits:
        with open(path) as fh:
            for line in fh:                                   # a query's hits are contiguous, best score first
                f = line.rstrip("\n").split("\t")
                q = f[0]
                if q != cur:
                    if cur is not None:
                        off.append(len(P))
                    cur, seen = q, set()
                    g_id.append(q)
                    g_len.append(int(f[10]))
                    g_genome.append(a.genome or q.rpartition("|")[2])
                if f[1] == q:                                 # the query itself (reference proteins are in the library)
                    continue
                acc, _, tt = f[1].rpartition("|")
                p = pidx.get(int(tt)) if tt.isdigit() else None
                if p is None:
                    unknown += 1
                    continue
                if p in seen:                                 # only the best hit per proteome
                    continue
                seen.add(p)
                P.append(p); ACC.append(acc.encode()); BITS.append(float(f[5])); PID.append(float(f[2]))
                QS.append(int(f[6])); QE.append(int(f[7])); TLEN.append(int(f[11]))
    off.append(len(P))
    np.savez(a.out, gene_id=np.array(g_id), gene_len=np.array(g_len, dtype=np.int32), gene_genome=np.array(g_genome),
             off=np.array(off, dtype=np.int64), p=np.frombuffer(P, dtype=np.int32), acc=np.array(ACC, dtype="S16"),
             bits=np.frombuffer(BITS, dtype=np.float32), pid=np.frombuffer(PID, dtype=np.float32),
             qs=np.frombuffer(QS, dtype=np.int32), qe=np.frombuffer(QE, dtype=np.int32), tlen=np.frombuffer(TLEN, dtype=np.int32))
    log("genes %d, cells %d (%.0f per gene), hits to unknown taxa %d" % (len(g_id), len(P), len(P) / max(1, len(g_id)), unknown))


# ---------------------------------------------------------------- profile

def delta_of(bits, qlen):
    """Distance-like transform of a normalised bit score (0 for identical, grows as similarity decays)."""
    return -np.log(np.clip(bits / (2.0 * qlen), 0.005, 1.0)) + 0.02


def rank_one(M, min_genes):
    """M[gene, proteome] = delta or NaN. Robust fit M ~ r[gene] * D[proteome] by alternating medians."""
    cnt = np.isfinite(M).sum(0)
    ok = cnt >= min_genes
    S = M[:, ok]
    r = np.ones(len(S))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(6):
            D = np.nanmedian(S / r[:, None], axis=0)
            r = np.nanmedian(S / D[None, :], axis=1)
            r = np.where(np.isfinite(r) & (r > 0), r, 1.0)
            r = r / np.median(r)
    full = np.full(M.shape[1], np.nan, dtype=np.float32)
    full[ok] = D
    return full, (cnt / max(1, len(M))).astype(np.float32), r


def cmd_profile(a):
    names, Ds, covs, owns, ns = [], [], [], [], []
    lib = np.load(a.library)
    n_prot = len(lib["taxid"])
    pidx = {str(int(t)): i for i, t in reversed(list(enumerate(lib["taxid"])))}
    for path in a.columns:
        z = np.load(path)
        off, p, bits, glen, genome = z["off"], z["p"], z["bits"], z["gene_len"], z["gene_genome"]
        order = np.argsort(genome, kind="stable")
        bounds = np.flatnonzero(np.r_[True, genome[order][1:] != genome[order][:-1], True])
        for s, e in zip(bounds[:-1], bounds[1:]):
            genes = order[s:e]
            M = np.full((len(genes), n_prot), np.nan, dtype=np.float32)
            for i, g in enumerate(genes):
                sl = slice(off[g], off[g + 1])
                M[i, p[sl]] = delta_of(bits[sl], glen[g])
            D, cov, _ = rank_one(M, max(5, int(round(a.min_cov * len(genes)))))
            names.append(str(genome[genes[0]])); Ds.append(D); covs.append(cov.astype(np.float16)); ns.append(len(genes))
            owns.append(pidx.get(names[-1], -1))
        log("%s: %d genomes so far" % (os.path.basename(path), len(names)))
    rng = np.random.default_rng(a.seed)
    panel = np.sort(rng.choice(n_prot, min(a.panel, n_prot), replace=False)).astype(np.int32)
    np.savez(a.out, genome=np.array(names), D=np.stack(Ds), cov=np.stack(covs), own=np.array(owns, dtype=np.int32),
             n_genes=np.array(ns, dtype=np.int32), panel=panel)
    D = np.stack(Ds)
    log("genomes %d; proteomes with a defined distance per genome: median %d; panel %d" % (
        len(names), int(np.median(np.isfinite(D).sum(1))), len(panel)))


# ---------------------------------------------------------------- align

SEQ = {}


def fetch(reference, want):
    """Sequences of the wanted accessions (bytes) from the indexed reference FASTA, read in file order."""
    z = np.load(reference + ".sdx.npz")
    acc, offs = z["acc"], z["off"]
    w = np.array(sorted(want), dtype=acc.dtype)
    pos = np.searchsorted(acc, w)
    pos[pos == len(acc)] = 0
    hit = acc[pos] == w
    o = offs[pos[hit]]
    names = w[hit]
    order = np.argsort(o)
    out = {}
    with open(reference, "rb") as fh:
        for i in order:
            fh.seek(int(o[i]))
            fh.readline()
            seq = []
            for line in fh:
                if line[:1] == b">":
                    break
                seq.append(line.strip())
            out[bytes(names[i])] = b"".join(seq).decode()
    return out


def work(task):
    import parasail
    res = []
    for gseq, accs in task:
        q = gseq[:LMAX]
        L = len(q)
        rows = np.full((len(accs), L), NOALN, dtype=np.int8)
        st = np.zeros((len(accs), 3), dtype=np.float32)
        for i, acc in enumerate(accs):
            t = SEQ.get(acc)
            if not t:
                continue
            r = parasail.sw_trace_striped_16(q, t[:LMAX], 11, 1, parasail.blosum62)
            tb = r.traceback
            qa = np.frombuffer(tb.query.encode(), dtype=np.uint8)
            ta = np.frombuffer(tb.ref.encode(), dtype=np.uint8)
            if len(qa) == 0:
                continue
            mq = qa != 45
            qpos = r.cigar.beg_query + np.cumsum(mq) - 1
            rows[i, qpos[mq]] = LUT[ta[mq]]
            st[i] = (r.score, float((qa == ta).sum()) / len(qa), float((~mq).sum()))
        res.append((rows.tobytes(), st))
    return res


def cmd_align(a):
    import multiprocessing as mp
    prof = np.load(a.profiles)
    n_prot = prof["D"].shape[1]
    in_panel = np.zeros(n_prot, dtype=bool)
    in_panel[prof["panel"]] = True
    z = np.load(a.columns)
    off, p, acc = z["off"], z["p"], z["acc"]
    gseq = read_fasta(a.fasta)
    ids = [str(x) for x in z["gene_id"]]
    missing = sum(1 for g in ids if g not in gseq)
    if missing:
        sys.exit("%d genes have no sequence in the FASTA files" % missing)
    keep = in_panel[p]
    log("cells in panel proteomes: %d of %d" % (keep.sum(), len(p)))
    SEQ.update(fetch(a.reference, set(acc[keep].tolist())))
    log("member sequences: %d" % len(SEQ))
    L = np.array([min(len(gseq[g]), LMAX) for g in ids], dtype=np.int64)
    row_off, row_p, tasks, cur, cur_n = [0], [], [], [], 0
    for g in range(len(ids)):
        sl = np.flatnonzero(keep[off[g]:off[g + 1]]) + off[g]
        sl = sl[np.argsort(p[sl], kind="stable")]              # rows of a gene sorted by proteome
        row_p.append(p[sl])
        row_off.append(row_off[-1] + len(sl))
        cur.append((gseq[ids[g]], [bytes(x) for x in acc[sl]]))
        cur_n += len(sl)
        if cur_n >= 400:
            tasks.append(cur); cur, cur_n = [], 0
    if cur:
        tasks.append(cur)
    row_off = np.array(row_off, dtype=np.int64)
    n_rows = np.diff(row_off)
    gene_base = np.r_[0, np.cumsum(n_rows * L)]
    msa = np.lib.format.open_memmap(a.out + ".msa.npy", mode="w+", dtype=np.int8, shape=(int(gene_base[-1]),))
    stats = np.zeros((int(row_off[-1]), 3), dtype=np.float32)
    g = 0
    with mp.get_context("fork").Pool(a.procs) as pool:
        for k, res in enumerate(pool.imap(work, tasks, chunksize=1)):
            for rows, st in res:
                msa[gene_base[g]:gene_base[g + 1]] = np.frombuffer(rows, dtype=np.int8)
                stats[row_off[g]:row_off[g + 1]] = st
                g += 1
            if k % 2000 == 0:
                log("aligned genes %d / %d" % (g, len(ids)))
    msa.flush()
    codes = np.concatenate([LUT[np.frombuffer(gseq[x][:LMAX].encode(), dtype=np.uint8)] for x in ids]) if ids else np.zeros(0, np.int8)
    np.savez(a.out + ".index.npz", gene_id=z["gene_id"], gene_genome=z["gene_genome"], gene_L=L.astype(np.int32),
             code_start=np.r_[0, np.cumsum(L)].astype(np.int64), codes=codes, row_off=row_off,
             row_p=np.concatenate(row_p).astype(np.int32) if row_p else np.zeros(0, np.int32),
             gene_base=gene_base.astype(np.int64), row_score=stats[:, 0], row_ident=stats[:, 1], row_ins=stats[:, 2])
    log("rows %d, bytes %d, empty alignments %d" % (row_off[-1], gene_base[-1], int((stats[:, 0] == 0).sum())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("columns"); s.add_argument("--library", required=True); s.add_argument("--hits", nargs="+", required=True)
    s.add_argument("--genome", default="", help="name of the genome when all queries belong to one (default: the taxid in the query name)")
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_columns)
    s = sub.add_parser("profile"); s.add_argument("--library", required=True); s.add_argument("--columns", nargs="+", required=True)
    s.add_argument("--panel", type=int, default=2048); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--min-cov", type=float, default=0.10, help="a genome has a distance to a proteome only if this share of its genes hit it")
    s.add_argument("--out", required=True); s.set_defaults(fn=cmd_profile)
    s = sub.add_parser("align"); s.add_argument("--profiles", required=True); s.add_argument("--columns", required=True)
    s.add_argument("--fasta", nargs="+", required=True); s.add_argument("--reference", required=True)
    s.add_argument("--out", required=True); s.add_argument("--procs", type=int, default=14); s.set_defaults(fn=cmd_align)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
