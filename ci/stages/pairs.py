"""Stage pairs (pipeline selfsup, objective C): homologs inside the corpus, the data of objective C (the divergence of a
pair of homologs given the two genomes). One DIAMOND search of every corpus protein against a database of the same
proteins; for each query gene the best row in every other genome (its own genome left out), the `genomes` nearest
genomes kept, each with its alignment residue by residue.

  data/pairs/db.dmnd           the corpus proteins (also searched by the island stage for objective C)
  data/pairs/genomes.json      the corpus genome ids in database order
  data/pairs/<id>.npz          per query genome, rows of gene i at off[i]:off[i+1], nearest first:
                               tgenome int32 (index into genomes.json), tgene int32 (gene index in the target's genes.tsv),
                               qs, ss uint32 (1-based starts of query and target), bits float32,
                               cigar uint8 + cigar_off, sseq uint8 + sseq_off (the aligned target residues)
  data/pairs/manifest.json
Settings (`pairs:`): sensitivity (sensitive), max_target_seqs (500), evalue (1e-5), genomes (64), threads, block_size.
Rows are ordered by bit score, then target genome and gene, so the result does not depend on thread scheduling.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from ci import manifest as M
from ci.stages.common import Stage, main_args, write_json
from ci.stages.search import gene_ids
from seeddrift.train.logger import save_npz

FIELDS = ["qseqid", "sseqid", "qstart", "sstart", "bitscore", "cigar", "sseq"]


def corpus(run_dir):
    with open(os.path.join(run_dir, "data", "corpus", "manifest.json")) as f:
        return [(e["id"], os.path.join(run_dir, "data", "corpus", e["id"])) for e in json.load(f)["entries"]]


def make_db(diamond, todo, out_faa, db):
    with open(out_faa + ".part", "w") as fo:
        for gid, folder in todo:
            with open(os.path.join(folder, "proteins.faa")) as f:
                fo.write(f.read())
    os.replace(out_faa + ".part", out_faa)
    subprocess.run([diamond, "makedb", "--in", out_faa, "-d", db[:-5], "--quiet"], check=True)


def write_genome(path, n_genes, rows_by_gene, keep):
    """rows_by_gene: {gene index: {target genome: (bits, tgenome, tgene, qs, ss, cigar, sseq)}} (or lists of rows); the
    best row per target genome, the `keep` nearest genomes."""
    off, cols = [0], {k: [] for k in ("tgenome", "tgene", "qs", "ss", "bits")}
    cig, cig_off, sq, sq_off = bytearray(), [0], bytearray(), [0]
    for i in range(n_genes):
        best = {}
        cand = rows_by_gene.get(i, [])
        for r in sorted(cand.values() if isinstance(cand, dict) else cand, key=lambda r: (-r[0], r[1], r[2])):
            if r[1] not in best:
                best[r[1]] = r
        rows = list(best.values())[:keep]
        for bits, tg, tj, qs, ss, c, s in rows:
            cols["tgenome"].append(tg)
            cols["tgene"].append(tj)
            cols["qs"].append(qs)
            cols["ss"].append(ss)
            cols["bits"].append(bits)
            cig += c.encode()
            cig_off.append(len(cig))
            sq += s.encode()
            sq_off.append(len(sq))
        off.append(off[-1] + len(rows))
    save_npz(path, compress=True, off=np.array(off, dtype=np.int32), tgenome=np.array(cols["tgenome"], dtype=np.int32),
             tgene=np.array(cols["tgene"], dtype=np.int32), qs=np.array(cols["qs"], dtype=np.uint32), ss=np.array(cols["ss"], dtype=np.uint32),
             bits=np.array(cols["bits"], dtype=np.float32), cigar=np.frombuffer(bytes(cig), dtype=np.uint8), cigar_off=np.array(cig_off, dtype=np.int64),
             sseq=np.frombuffer(bytes(sq), dtype=np.uint8), sseq_off=np.array(sq_off, dtype=np.int64))
    return off[-1]


def search(diamond, queries, db, pc, genome_index, gene_index, query_genes, out_dir, own=True):
    """Stream DIAMOND's rows of `queries` against `db` into out_dir/<query genome>.npz. genome_index: {genome id: index};
    gene_index: {genome id: {gene name: index}} of the database; query_genes: {query genome id: [gene names]}.
    own: the query genomes are database genomes, whose own rows are left out."""
    keep = int(pc.get("genomes", 64))
    cmd = [diamond, "blastp", "-q", queries, "-d", db, "-o", "/dev/stdout", "--" + pc.get("sensitivity", "sensitive"),
           "--max-target-seqs", str(pc.get("max_target_seqs", 500)), "--max-hsps", "1", "-e", str(pc.get("evalue", 1e-5)),
           "--threads", str(pc.get("threads", 2)), "-b", str(pc.get("block_size", 4)), "-c", "1", "--tmpdir", os.environ.get("TMPDIR", "/tmp"),
           "--quiet", "--outfmt", "6"] + FIELDS
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    cur, rows, done, total = None, {}, set(), 0
    qidx = {g: {n: i for i, n in enumerate(names)} for g, names in query_genes.items()}
    try:
        for line in p.stdout:
            f = line.rstrip("\n").split("\t")
            qg, qn = f[0].split("__", 1)
            if qg != cur:
                if cur is not None:
                    total += write_genome(os.path.join(out_dir, cur + ".npz"), len(query_genes[cur]), rows, keep)
                    done.add(cur)
                if qg in done:
                    raise SystemExit("pairs: search output of %s is not contiguous" % qg)
                cur, rows = qg, {}
            tg, tn = f[1].split("__", 1)
            if own and tg == qg:
                continue
            r = (float(f[4]), genome_index[tg], gene_index[tg][tn], int(f[2]), int(f[3]), f[5], f[6])
            best = rows.setdefault(qidx[qg][qn], {})                      # the best row per target genome only
            old = best.get(r[1])
            if old is None or (-r[0], r[2]) < (-old[0], old[2]):
                best[r[1]] = r
        if p.wait() != 0:
            raise SystemExit("diamond exited with %d" % p.returncode)
        if cur is not None:
            total += write_genome(os.path.join(out_dir, cur + ".npz"), len(query_genes[cur]), rows, keep)
            done.add(cur)
    finally:
        if p.poll() is None:
            p.kill()
    for g, names in query_genes.items():
        if g not in done:
            write_genome(os.path.join(out_dir, g + ".npz"), len(names), {}, keep)
    return total


def main():
    run_dir = main_args()
    with Stage(run_dir, "pairs", requires=("pre", "corpus")) as st:
        cfg = st.config
        pc = cfg.get("pairs") or {}
        todo = corpus(run_dir)
        base = st.path("pairs", "x")[:-2]
        faa, db = os.path.join(base, "db.faa"), os.path.join(base, "db.dmnd")
        make_db(cfg["tools"]["diamond"], todo, faa, db)
        names = {g: gene_ids(folder) for g, folder in todo}
        order = [g for g, _ in todo]
        write_json(os.path.join(base, "genomes.json"), dict(genomes=order))
        total = search(cfg["tools"]["diamond"], faa, db, pc, {g: i for i, g in enumerate(order)}, {g: {n: i for i, n in enumerate(v)} for g, v in names.items()},
                       names, base)
        os.remove(faa)
        entries = [dict(id=g, files=[M.file_entry(os.path.join(base, g + ".npz"), run_dir)]) for g in order]
        man = M.build("pairs", "pairs", entries, meta=dict(settings={k: pc.get(k) for k in ("sensitivity", "max_target_seqs", "evalue", "genomes")},
                                                           genomes="data/pairs/genomes.json"))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        mp = os.path.join(base, "manifest.json")
        write_json(mp, dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
        st.output("pairs_db", db)
        st.output("pairs_genomes", os.path.join(base, "genomes.json"))
        st.output("pairs_manifest", mp)
        st.extra.update(genomes=len(order), rows=total)


if __name__ == "__main__":
    main()
