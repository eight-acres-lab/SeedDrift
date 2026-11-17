"""Stage search: every protein of the corpus (and of the evaluation genomes) against the reference, with the settings
of the benchmark (DIAMOND --sensitive, up to 1,000 targets, e-value 1e-5) plus --max-hsps 1; the text output is read
as a stream and stored per genome in data/hits/<id>.npz, format 2 (read it with load_hits, which also reads format 1):

  format    int32 [1]            2
  off       int32 [genes + 1]    rows off[i]:off[i+1] belong to gene i (order of the genome's genes.tsv); within a gene
                                 the rows are ordered by target (the kept set is chosen by score, see below)
  nraw      uint16 [genes]       targets reported for the gene (1,000 = the list was full)
  qlen      uint16|32 [genes]    length of the gene's protein (0 when it has no hit)
  tgt       target protein (row of reference/index.npz): the first of a gene absolute, then gaps to the previous one
  pid       identity x 10;  qs qe ss se  alignment coordinates;  slen  target length;  bits  bit score (float32)
  cigar     the CIGARs in binary (varint of length << 2 | op, op M=0 I=1 D=2), cigar_len bytes per row
Every per-row column is stored byte-shuffled (shape [item size, rows], unsigned 16 or 32 bits where it fits), which
the archive's zlib compresses about a third better than plain columns with ASCII CIGARs (format 1, CI 0.1; it also
stored each row's proteome and query length). The proteome of a target, and so its lineage, is
library row searchsorted(library.taxid, index.taxid[tgt]) (hit_proteomes).

Kept per gene: the NEAREST closest targets plus the best target of every reference genus (so that hiding the own
species, genus, family or order at training time still leaves visible hits). Rows are ordered by bit score, then
target index, so the result does not depend on DIAMOND's thread scheduling.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess

import numpy as np

from ci import manifest as M
from ci.stages.common import Stage, main_args, write_json
from seeddrift.train.logger import save_npz

FIELDS = ["qseqid", "sseqid", "pident", "qstart", "qend", "sstart", "send", "qlen", "slen", "bitscore", "cigar"]


def genomes(run_dir, cfg):
    out = []
    with open(os.path.join(run_dir, "data", "corpus", "manifest.json")) as f:
        for e in json.load(f)["entries"]:
            out.append((e["id"], os.path.join(run_dir, "data", "corpus", e["id"])))
    ev = os.path.join(run_dir, "data", "eval_inputs", "manifest.json")
    if os.path.exists(ev):
        with open(ev) as f:
            for e in json.load(f)["entries"]:
                out.append((e["id"], os.path.join(run_dir, "data", "eval_inputs", e["id"])))
    return out


def gene_ids(folder):
    with open(os.path.join(folder, "genes.tsv")) as f:
        return [r["gene_id"] for r in csv.DictReader(f, delimiter="\t")]


def select(rows, genus_of_prot, nearest):
    rows.sort(key=lambda r: (-r[9], r[0]))
    keep, seen_genus = rows[:nearest], set()
    for r in keep:
        seen_genus.add(genus_of_prot[r[1]])
    for r in rows[nearest:]:
        g = genus_of_prot[r[1]]
        if g not in seen_genus:
            seen_genus.add(g)
            keep.append(r)
    return keep


def narrow(vals):
    """uint16 when every value fits, else uint32 (coordinates and lengths of proteins over 65,535 residues)."""
    a = np.array(vals, dtype=np.int64)
    if a.size and (a.min() < 0 or a.max() >= 2 ** 32):
        raise SystemExit("value out of range for the hit format")
    return a.astype(np.uint16) if (a.size == 0 or a.max() < 65536) else a.astype(np.uint32)


def shuffled(a):
    """[item size, n] bytes of a little-endian array (byte k of every value together)."""
    a = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<")))
    return a.view(np.uint8).reshape(-1, a.dtype.itemsize).T.copy()


def unshuffled(b, kind="u"):
    b = np.asarray(b)
    return np.ascontiguousarray(b.T).view("<%s%d" % (kind, b.shape[0])).reshape(-1)


CIGAR_OPS = {"M": 0, "I": 1, "D": 2}
CIGAR_CHARS = "MID"


def cigar_encode(c: str) -> bytes:
    out, num = bytearray(), 0
    for ch in c:
        if ch.isdigit():
            num = num * 10 + ord(ch) - 48
            continue
        if ch not in CIGAR_OPS:
            raise SystemExit("unexpected CIGAR operation %r in %r" % (ch, c))
        x = num << 2 | CIGAR_OPS[ch]
        while True:
            byte, x = x & 0x7F, x >> 7
            out.append(byte | (0x80 if x else 0))
            if not x:
                break
        num = 0
    return bytes(out)


def cigar_decode(b: bytes) -> str:
    out, x, shift = [], 0, 0
    for byte in b:
        x |= (byte & 0x7F) << shift
        if byte & 0x80:
            shift += 7
            continue
        out.append("%d%s" % (x >> 2, CIGAR_CHARS[x & 3]))
        x, shift = 0, 0
    return "".join(out)


def flush(path, genes, by_gene, nraw, genus_of_prot, nearest):
    off, qlen = [0], []
    cols = {k: [] for k in ("tgt", "pid", "qs", "qe", "ss", "se", "slen", "bits")}
    cig, clen = bytearray(), []
    for g in genes:
        rows = sorted(select(by_gene.get(g, []), genus_of_prot, nearest), key=lambda r: (r[0], -r[9]))   # kept by score, stored by target
        qlen.append(rows[0][7] if rows else 0)
        prev = None
        for r in rows:
            tgt, _prot, pid, qs, qe, ss, se, ql, sl, bits, c = r
            cols["tgt"].append(tgt if prev is None else tgt - prev)
            prev = tgt
            for k, v in zip(("pid", "qs", "qe", "ss", "se", "slen", "bits"), (pid, qs, qe, ss, se, sl, bits)):
                cols[k].append(v)
            e = cigar_encode(c)
            cig += e
            clen.append(len(e))
        off.append(off[-1] + len(rows))
    save_npz(path, compress=True, format=np.array([2], dtype=np.int32), off=np.array(off, dtype=np.int32),
             nraw=np.minimum(np.array([nraw.get(g, 0) for g in genes], dtype=np.int64), 65535).astype(np.uint16), qlen=narrow(qlen),
             tgt=shuffled(narrow(cols["tgt"])), pid=shuffled(np.array(cols["pid"], dtype=np.uint16)),
             **{k: shuffled(narrow(cols[k])) for k in ("qs", "qe", "ss", "se", "slen")},
             bits=shuffled(np.array(cols["bits"], dtype=np.float32)), cigar=np.frombuffer(bytes(cig), dtype=np.uint8), cigar_len=shuffled(narrow(clen)))
    return off[-1]


def load_hits(path):
    """The hits of one genome as plain arrays, in format 1 or 2: off, nraw, qlen (per gene), tgt, pid, qs, qe, ss, se,
    slen, bits (per row), cigar(r) -> the CIGAR string of row r, and prot (format 1 only; else see hit_proteomes)."""
    z = np.load(path, allow_pickle=False)
    if "format" not in z.files:                                    # format 1 (CI 0.1)
        off = z["off"]
        out = {k: z[k] for k in ("off", "nraw", "tgt", "prot", "pid", "qs", "qe", "ss", "se", "slen", "bits")}
        out["qlen"] = np.array([z["qlen"][off[i]] if off[i + 1] > off[i] else 0 for i in range(len(off) - 1)])
        cb, co = bytes(z["cigar"]), z["cigar_off"]
        out["cigar"] = lambda r: cb[co[r]:co[r + 1]].decode()
        return out
    if int(z["format"][0]) != 2:
        raise SystemExit("%s: unknown hit format %s" % (path, z["format"][0]))
    off = z["off"]
    out = dict(off=off, nraw=z["nraw"], qlen=z["qlen"])
    cs = np.cumsum(unshuffled(z["tgt"]).astype(np.int64))
    starts = np.asarray(off[:-1], dtype=np.int64)
    gene_of_row = np.repeat(np.arange(len(starts)), np.diff(np.asarray(off, dtype=np.int64)))
    base = np.where(starts > 0, cs[np.maximum(starts - 1, 0)] if len(cs) else 0, 0)   # gaps restart at every gene
    out["tgt"] = (cs - base[gene_of_row]).astype(np.int32) if len(cs) else np.zeros(0, dtype=np.int32)
    out["pid"] = unshuffled(z["pid"])
    for k in ("qs", "qe", "ss", "se", "slen"):
        out[k] = unshuffled(z[k])
    out["bits"] = unshuffled(z["bits"], "f")
    cb = bytes(z["cigar"])
    co = np.concatenate([[0], np.cumsum(unshuffled(z["cigar_len"]).astype(np.int64))])
    out["cigar"] = lambda r: cigar_decode(cb[co[r]:co[r + 1]])
    return out


def hit_proteomes(hits, index_taxid, library_taxid):
    """The library row (proteome, lineage) of every hit row, in both formats."""
    if "prot" in hits:
        return np.asarray(hits["prot"])
    return np.searchsorted(library_taxid, np.asarray(index_taxid)[np.asarray(hits["tgt"])])


def main():
    run_dir = main_args()
    with Stage(run_dir, "search", requires=("pre", "corpus")) as st:
        cfg = st.config
        sc = cfg["search"]
        with open(os.path.join(run_dir, "data", "reference", "manifest.json")) as f:
            db = json.load(f)["database"]
        db = db if os.path.isabs(db) else os.path.join(run_dir, db)
        idx = np.load(os.path.join(run_dir, "data", "reference", "index.npz"))
        acc_s, acc_tax = idx["acc"], idx["taxid"]
        lib = np.load(os.path.join(run_dir, "data", "reference", "library.npz"))
        lib_tax = lib["taxid"]
        genus_of_prot = {i: (int(g) if g else -int(t)) for i, (t, g) in enumerate(zip(lib_tax, lib["ranks"][:, 4]))}
        todo = genomes(run_dir, cfg)
        q = st.path("search", "queries.faa")
        with open(q + ".part", "w") as fo:
            for gid, folder in todo:
                with open(os.path.join(folder, "proteins.faa")) as f:
                    fo.write(f.read())
        os.replace(q + ".part", q)
        cmd = [cfg["tools"]["diamond"], "blastp", "-q", q, "-d", db, "-o", "/dev/stdout", "--" + sc.get("sensitivity", "sensitive"),
               "--max-target-seqs", str(sc.get("max_target_seqs", 1000)), "--max-hsps", "1", "-e", str(sc.get("evalue", 1e-5)),
               "--threads", str(sc.get("threads", 2)), "-b", str(sc.get("block_size", 4)), "-c", "1", "--tmpdir", os.environ.get("TMPDIR", "/tmp"),
               "--quiet", "--outfmt", "6"] + FIELDS
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
        folders = dict(todo)
        hits_dir = st.path("hits", "x")[:-2]
        cur, by_gene, nraw, done, rows_total = None, {}, {}, set(), 0
        unknown = 0
        try:
            for line in p.stdout:
                f = line.rstrip("\n").split("\t")
                gid, gene = f[0].split("__", 1)
                if gid != cur:
                    if cur is not None:
                        rows_total += flush(os.path.join(hits_dir, cur + ".npz"), gene_ids(folders[cur]), by_gene, nraw, genus_of_prot, sc.get("nearest", 32))
                        done.add(cur)
                    if gid in done:
                        raise SystemExit("search output of %s is not contiguous" % gid)
                    cur, by_gene, nraw = gid, {}, {}
                acc, _, t = f[1].partition("|")
                i = int(np.searchsorted(acc_s, acc.encode()))
                if i >= len(acc_s) or acc_s[i] != acc.encode():
                    unknown += 1
                    raise SystemExit("target %s is not in the reference index" % f[1])
                pr = int(np.searchsorted(lib_tax, acc_tax[i]))
                nraw[gene] = nraw.get(gene, 0) + 1
                by_gene.setdefault(gene, []).append((i, pr, int(round(float(f[2]) * 10)), int(f[3]), int(f[4]), int(f[5]), int(f[6]), int(f[7]), int(f[8]), float(f[9]), f[10]))
            if p.wait() != 0:
                raise SystemExit("diamond exited with %d" % p.returncode)
            if cur is not None:
                rows_total += flush(os.path.join(hits_dir, cur + ".npz"), gene_ids(folders[cur]), by_gene, nraw, genus_of_prot, sc.get("nearest", 32))
                done.add(cur)
        finally:
            if p.poll() is None:
                p.kill()
        for gid, folder in todo:
            if gid not in done:
                flush(os.path.join(hits_dir, gid + ".npz"), gene_ids(folder), {}, {}, genus_of_prot, sc.get("nearest", 32))
        entries = [dict(id=gid, files=[M.file_entry(os.path.join(hits_dir, gid + ".npz"), run_dir)]) for gid, _ in todo]
        man = M.build("hits", "hits", entries, meta=dict(nearest=sc.get("nearest", 32), max_target_seqs=sc.get("max_target_seqs", 1000)))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = st.path("hits", "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("hits_manifest", mp)
        os.remove(q)
        st.extra.update(genomes=len(todo), rows=rows_total)


if __name__ == "__main__":
    main()
