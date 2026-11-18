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
gene count, split and near-relative cluster: corpus genomes are split within their group by units of genomes that share
a genus or a cluster (split() below); evaluation genomes are "eval".
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


NUC_LUT = np.full(256, 4, dtype=np.uint8)
for _c, _v in NUC.items():
    NUC_LUT[ord(_c)] = _v
    NUC_LUT[ord(_c.lower())] = _v
AA_LUT = np.full(256, 21, dtype=np.uint8)
for _c, _v in AA.items():
    AA_LUT[ord(_c)] = _v


def codon_tokens(cds: str, maxc: int) -> np.ndarray:
    n = min(len(cds) // 3, maxc)
    x = NUC_LUT[np.frombuffer(cds[:3 * n].encode(), dtype=np.uint8)].reshape(n, 3).astype(np.int64)
    ok = (x < 4).all(1)
    return np.where(ok, 16 * x[:, 0] + 4 * x[:, 1] + x[:, 2] + 1, 65).astype(np.uint8)


CIGAR_RE = re.compile(r"(\d+)([MID])")


def track(cigar: str, sseq) -> np.ndarray:
    """One token per query position of the alignment (M and I consume the query, M and D the target)."""
    sb = np.frombuffer(sseq.encode() if isinstance(sseq, str) else sseq, dtype=np.uint8)
    parts, j = [], 0
    for n, op in CIGAR_RE.findall(cigar):
        n = int(n)
        if op == "M":
            parts.append(AA_LUT[sb[j:j + n]])
            j += n
        elif op == "I":
            parts.append(np.full(n, GAP, dtype=np.uint8))
        else:
            j += n
    if j != len(sb):
        raise SystemExit("CIGAR %s does not consume the aligned target sequence (%d of %d residues)" % (cigar, j, len(sb)))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)


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
    prot, qs = h["prot"], h["qs"]                      # each column read once (an npz decompresses on every access)
    hoff, hprot, hqs, tr, toff = [0], [], [], [], [0]
    for i in range(len(genes)):
        n = 0
        for r in range(off[i], off[i + 1]):
            if n >= k or not has_sseq or so[r + 1] == so[r]:
                break                                      # only rows that carry sseq: the nearest ones, which come first
            t = track(cb[co[r]:co[r + 1]].decode(), sb[so[r]:so[r + 1]])
            hprot.append(int(prot[r]))
            hqs.append(int(qs[r]) - 1)
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


def split(entries, groups, lin, cluster=None):
    """{id: train|heldout} within each group. Units: genomes that share a genus, or a near-relative cluster (corpus field
    `cluster`: an anchor and its relatives), are one unit (connected components); a genome without a genus and without a
    cluster is its own unit (species, else its id). About one plain unit in five is held out and one cluster unit in
    two (the clusters exist for the near-distance strata of the development set). Whole units move in the fix-ups (at
    least one training and, where a group has two units or more, one held-out unit), so no genus and no cluster is on
    both sides."""
    cluster = cluster or {}
    out = {}
    for g in sorted(set(groups.values())):
        ids = sorted(e for e in entries if groups[e] == g)
        parent = {e: e for e in ids}

        def find(e):
            while parent[e] != e:
                parent[e] = parent[parent[e]]
                e = parent[e]
            return e
        first = {}
        for e in ids:
            for k in ([("g", lin[e]["genus"])] if lin[e]["genus"] else []) + ([("c", cluster[e])] if cluster.get(e) else []):
                if k in first:
                    parent[find(e)] = find(first[k])
                else:
                    first[k] = e
        comp = {}
        for e in ids:
            comp.setdefault(find(e), []).append(e)
        unit = {}
        for members in comp.values():
            cl = sorted({cluster[e] for e in members if cluster.get(e)})
            if cl:
                key = ("c", cl[0])
            else:
                e = members[0]
                key = ("g", lin[e]["genus"]) if lin[e]["genus"] else (("s", lin[e]["species"]) if lin[e]["species"] else ("id", e))
            for e in members:
                unit[e] = key
        units = sorted(set(unit.values()), key=str)
        held = {u for u in units if int(hashlib.sha256(("split:%s" % (u,)).encode()).hexdigest(), 16) % (2 if u[0] == "c" else 5) == 0}
        if held == set(units) and units:
            held.discard(units[0])
        elif not held and len(units) >= 2:
            held.add(units[-1])
        for e in ids:
            out[e] = "heldout" if unit[e] in held else "train"
        for col in ("genus",):
            tr = {lin[e][col] for e in ids if out[e] == "train" and lin[e][col]}
            ho = {lin[e][col] for e in ids if out[e] == "heldout" and lin[e][col]}
            assert not (tr & ho), "a %s on both sides of the split" % col
        assert not ({cluster[e] for e in ids if out[e] == "train" and cluster.get(e)} & {cluster[e] for e in ids if out[e] == "heldout" and cluster.get(e)})
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
        sp = split([e["id"] for n, e in ents if n == "corpus"], {e["id"]: groups[e["id"]] for n, e in ents if n == "corpus"}, lin,
                   {e["id"]: e["cluster"] for n, e in ents if n == "corpus" and e.get("cluster")})
        out_entries, index = [], []
        for name, e in ents:
            gid = e["id"]
            arrays, n_genes, n_hits = one_genome(os.path.join(run_dir, "data", name, gid), os.path.join(run_dir, "data", "hits", gid + ".npz"), gid, maxc, k)
            p = st.path("train_data", gid + ".npz")
            save_npz(p, compress=True, **arrays)
            out_entries.append(dict(id=gid, files=[M.file_entry(p, run_dir)]))
            index.append(dict(id=gid, source=name, group=groups[gid], taxid=int(e.get("taxid") or e.get("host_taxid")), ranks=[lin[gid][r] for r in RANKS],
                              genes=n_genes, hits=n_hits, split=sp.get(gid, "eval"), cluster=e.get("cluster", ""), relation=e.get("relation", "")))
        man = M.build("train-data", "train_data", out_entries, meta=dict(max_codons=maxc, hits=k))
        sha = M.write(man, os.path.join(run_dir, "manifests"))
        ip = st.path("train_data", "index.json")
        write_json(ip, dict(sha256=sha, dataset_id=M.dataset_id(man), ranks=list(RANKS), max_codons=maxc, hits=k, genomes=index,
                            library="data/reference/library.npz"))
        st.output("train_data_index", ip)
        st.extra.update(genomes=len(index), with_hits=sum(1 for g in index if g["hits"]))


if __name__ == "__main__":
    main()
