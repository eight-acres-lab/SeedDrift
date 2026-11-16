"""Stage search: every protein of the corpus (and of the evaluation genomes) against the reference, with the settings
of the benchmark (DIAMOND --sensitive, up to 1,000 targets, e-value 1e-5) plus --max-hsps 1; the text output is read
as a stream and stored per genome in data/hits/<id>.npz:

  off       int32 [genes + 1]    rows off[i]:off[i+1] belong to gene i (order of the genome's genes.tsv)
  nraw      uint16 [genes]       targets reported for the gene (1,000 = the list was full)
  tgt       int32                target protein: row of reference/index.npz (the sequence can be fetched back)
  prot      int32                row of reference/library.npz of the target's proteome (its lineage)
  pid       uint16               identity x 10
  qs qe ss se  uint32            alignment coordinates (query start, end; target start, end)
  qlen slen    uint32            lengths
  bits      float32              bit score
  cigar     uint8 + cigar_off    the CIGAR of each alignment (concatenated bytes; rows cigar_off[r]:cigar_off[r+1])

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


def flush(path, genes, by_gene, nraw, genus_of_prot, nearest):
    off = [0]
    cols = {k: [] for k in ("tgt", "prot", "pid", "qs", "qe", "ss", "se", "qlen", "slen", "bits")}
    cig, cig_off = bytearray(), [0]
    for g in genes:
        rows = select(by_gene.get(g, []), genus_of_prot, nearest)
        for r in rows:
            tgt, prot, pid, qs, qe, ss, se, ql, sl, bits, c = r
            for k, v in zip(("tgt", "prot", "pid", "qs", "qe", "ss", "se", "qlen", "slen", "bits"), (tgt, prot, pid, qs, qe, ss, se, ql, sl, bits)):
                cols[k].append(v)
            cig += c.encode()
            cig_off.append(len(cig))
        off.append(off[-1] + len(rows))
    save_npz(path, compress=True, off=np.array(off, dtype=np.int32), nraw=np.minimum(np.array([nraw.get(g, 0) for g in genes], dtype=np.int64), 65535).astype(np.uint16),
                        tgt=np.array(cols["tgt"], dtype=np.int32), prot=np.array(cols["prot"], dtype=np.int32), pid=np.array(cols["pid"], dtype=np.uint16),
                        qs=np.array(cols["qs"], dtype=np.uint32), qe=np.array(cols["qe"], dtype=np.uint32), ss=np.array(cols["ss"], dtype=np.uint32),
                        se=np.array(cols["se"], dtype=np.uint32), qlen=np.array(cols["qlen"], dtype=np.uint32), slen=np.array(cols["slen"], dtype=np.uint32),
                        bits=np.array(cols["bits"], dtype=np.float32), cigar=np.frombuffer(bytes(cig), dtype=np.uint8), cigar_off=np.array(cig_off, dtype=np.int64))
    return off[-1]


def main():
    run_dir = main_args()
    with Stage(run_dir, "search") as st:
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
