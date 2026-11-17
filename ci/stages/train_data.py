"""Stage train_data: the corpus and evaluation genomes as tensors for the source model (strategy: model-pilot).

Per genome data/train_data/<id>.npz (genes in the order of the genome's genes.tsv):
  codons uint8 + codon_off int64 [genes+1]  codon tokens of each gene: 1..64 = codon (ACGT order) + 1, 65 = other;
                                            at most MAXC codons (the rest of a longer gene is cut)
  replicon int32, start int64, end int64, strand int8   position of each gene; replicon_len int64 [replicons]
  hit_off int64 [genes+1]                   the K nearest hits of each gene (search rows that carry sseq)
  hit_prot int32, hit_qs int32              per hit: proteome (row of reference/library.npz), 0-based query start
  track uint8 + track_off int64 [hits+1]    per hit, one token per query position of the alignment: the aligned target
                                            residue (1..20 = ACDEFGHIKLMNPQRSTVWY, 21 = other) or 22 = no target residue
                                            (an insertion in the query); target residues without a query residue are left
                                            out. Read residue by residue from the CIGAR and the aligned target sequence.
data/train_data/index.json: every genome with its group, taxid, lineage (phylum, class, order, family, genus, species),
gene count and split: corpus genomes are split by genus within their group (about one in five genera held out, at
least one training and, where a group has two genomes or more, one held-out genome); evaluation genomes are "eval".
No summary number of a hit (identity, coverage, score) is written: the model reads alignments, not their summaries.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re

import numpy as np

from ci import manifest as M
from ci.kingdom import KingdomMap
from ci.stages.common import Stage, main_args, read_fasta, write_json
from ci.taxonomy import Taxonomy
from seeddrift.train.logger import save_npz

RANKS = ("phylum", "class", "order", "family", "genus", "species")
AA = {a: i + 1 for i, a in enumerate("ACDEFGHIKLMNPQRSTVWY")}
GAP = 22
NUC = {"A": 0, "C": 1, "G": 2, "T": 3}


def codon_tokens(cds: str, maxc: int) -> np.ndarray:
    s = cds.upper()
    n = min(len(s) // 3, maxc)
    out = np.full(n, 65, dtype=np.uint8)
    for i in range(n):
        a, b, c = NUC.get(s[3 * i]), NUC.get(s[3 * i + 1]), NUC.get(s[3 * i + 2])
        if a is not None and b is not None and c is not None:
            out[i] = 16 * a + 4 * b + c + 1
    return out


def track(cigar: str, sseq: str) -> np.ndarray:
    """One token per query position of the alignment (M and I consume the query, M and D the target)."""
    out, j = [], 0
    for n, op in re.findall(r"(\d+)([MID])", cigar):
        n = int(n)
        if op == "M":
            out += [AA.get(c, 21) for c in sseq[j:j + n]]
            j += n
        elif op == "I":
            out += [GAP] * n
        else:
            j += n
    if j != len(sseq):
        raise SystemExit("CIGAR %s does not consume the aligned target sequence (%d of %d residues)" % (cigar, j, len(sseq)))
    return np.array(out, dtype=np.uint8)


def one_genome(folder, hits_path, gid, maxc, k):
    with open(os.path.join(folder, "genes.tsv")) as f:
        genes = list(csv.DictReader(f, delimiter="\t"))
    cds = read_fasta(os.path.join(folder, "cds.fna"))
    reps = sorted({g["replicon"] for g in genes})
    rep_ix = {r: i for i, r in enumerate(reps)}
    rep_len = {}
    for g in genes:
        rep_len[g["replicon"]] = max(rep_len.get(g["replicon"], 0), int(g["end"]))
    cod, coff = [], [0]
    for g in genes:
        t = codon_tokens(cds["%s__%s" % (gid, g["gene_id"])], maxc)
        cod.append(t)
        coff.append(coff[-1] + len(t))
    h = np.load(hits_path)
    off = h["off"]
    if len(off) != len(genes) + 1:
        raise SystemExit("%s: hits and genes.tsv disagree on the number of genes" % gid)
    has_sseq = "sseq" in h.files
    cb, co = bytes(h["cigar"]), h["cigar_off"]
    sb, so = (bytes(h["sseq"]), h["sseq_off"]) if has_sseq else (b"", None)
    hoff, hprot, hqs, tr, toff = [0], [], [], [], [0]
    for i in range(len(genes)):
        n = 0
        for r in range(off[i], off[i + 1]):
            if n >= k or not has_sseq or so[r + 1] == so[r]:
                break                                      # only rows that carry sseq: the nearest ones, which come first
            t = track(cb[co[r]:co[r + 1]].decode(), sb[so[r]:so[r + 1]].decode())
            hprot.append(int(h["prot"][r]))
            hqs.append(int(h["qs"][r]) - 1)
            tr.append(t)
            toff.append(toff[-1] + len(t))
            n += 1
        hoff.append(hoff[-1] + n)
    arrays = dict(codons=np.concatenate(cod) if cod else np.zeros(0, np.uint8), codon_off=np.array(coff, dtype=np.int64),
                  replicon=np.array([rep_ix[g["replicon"]] for g in genes], dtype=np.int32), start=np.array([int(g["start"]) for g in genes], dtype=np.int64),
                  end=np.array([int(g["end"]) for g in genes], dtype=np.int64), strand=np.array([1 if g["strand"] in ("+", "1") else -1 for g in genes], dtype=np.int8),
                  replicon_len=np.array([rep_len[r] for r in reps], dtype=np.int64), hit_off=np.array(hoff, dtype=np.int64),
                  hit_prot=np.array(hprot, dtype=np.int32), hit_qs=np.array(hqs, dtype=np.int32),
                  track=np.concatenate(tr) if tr else np.zeros(0, np.uint8), track_off=np.array(toff, dtype=np.int64))
    return arrays, len(genes), hoff[-1]


def split(entries, groups, lin):
    """{id: train|heldout} by genus within each group (about one genus in five held out). Whole genera move in the
    fix-ups (at least one training and, where a group has two genera or more, one held-out genus), so no genus is on
    both sides; a genome without a genus is its own unit (species, else its id)."""
    out = {}
    for g in sorted(set(groups.values())):
        ids = sorted(e for e in entries if groups[e] == g)
        unit = {e: ("g", lin[e]["genus"]) if lin[e]["genus"] else (("s", lin[e]["species"]) if lin[e]["species"] else ("id", e)) for e in ids}
        units = sorted(set(unit.values()), key=str)
        held = {u for u in units if int(hashlib.sha256(("split:%s" % (u,)).encode()).hexdigest(), 16) % 5 == 0}
        if held == set(units) and units:
            held.discard(units[0])
        elif not held and len(units) >= 2:
            held.add(units[-1])
        for e in ids:
            out[e] = "heldout" if unit[e] in held else "train"
        assert not ({unit[e] for e in ids if out[e] == "train"} & {unit[e] for e in ids if out[e] == "heldout"})
    return out


def main():
    run_dir = main_args()
    with Stage(run_dir, "train_data", requires=("pre", "corpus")) as st:
        cfg = st.config
        dc = cfg.get("train_data") or {}
        maxc, k = int(dc.get("max_codons", 1000)), int(dc.get("hits", 32))
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        km = KingdomMap()
        ents = []
        for name in ("corpus", "eval_inputs"):
            p = os.path.join(run_dir, "data", name, "manifest.json")
            if os.path.exists(p):
                with open(p) as f:
                    ents += [(name, e) for e in json.load(f)["entries"]]
        lin, groups = {}, {}
        for name, e in ents:
            t = int(e.get("taxid") or e.get("host_taxid"))
            lin[e["id"]] = {r: tax.at_rank(t, r) for r in RANKS}
            groups[e["id"]] = km.group(tax, t, e.get("source_kind", "")) if name == "corpus" else "eval"
        sp = split([e["id"] for n, e in ents if n == "corpus"], {e["id"]: groups[e["id"]] for n, e in ents if n == "corpus"}, lin)
        out_entries, index = [], []
        for name, e in ents:
            gid = e["id"]
            arrays, n_genes, n_hits = one_genome(os.path.join(run_dir, "data", name, gid), os.path.join(run_dir, "data", "hits", gid + ".npz"), gid, maxc, k)
            p = st.path("train_data", gid + ".npz")
            save_npz(p, compress=True, **arrays)
            out_entries.append(dict(id=gid, files=[M.file_entry(p, run_dir)]))
            index.append(dict(id=gid, source=name, group=groups[gid], taxid=int(e.get("taxid") or e.get("host_taxid")), ranks=[lin[gid][r] for r in RANKS],
                              genes=n_genes, hits=n_hits, split=sp.get(gid, "eval")))
        man = M.build("train-data", "train_data", out_entries, meta=dict(max_codons=maxc, hits=k))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        ip = st.path("train_data", "index.json")
        write_json(ip, dict(sha256=sha, dataset_id=M.dataset_id(man), ranks=list(RANKS), max_codons=maxc, hits=k, genomes=index,
                            library="data/reference/library.npz"))
        st.output("train_data_index", ip)
        st.extra.update(genomes=len(index), with_hits=sum(1 for g in index if g["hits"]))


if __name__ == "__main__":
    main()
