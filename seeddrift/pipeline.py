"""One genome in, a table of candidate foreign genes out.

  input      protein FASTA, or a nucleotide genome (prokaryote: genes called with pyrodigal); gz ok
  search     DIAMOND --sensitive against the reference library (tools service)
  self       the genome's own taxon: given, or inferred from near-identical best hits
  repeats    repeat-like regions of the genes and of the hit targets (Pfam, coiled coils: tools
             service; tandem repeats: here)
  classify   per gene: category (mammal / prokaryote / eukaryote / virus / native / orphan / repeat),
             Alien Index, donor group, support (single.py)
  structure  optional, for orphans: ESMFold (GPU service) + Foldseek vs AlphaFold Swiss-Prot

Outputs in the job folder: proteins.faa, hits.tsv, genes.tsv, summary.json, structures/*.pdb
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import re
import time
from collections import defaultdict

from .donors import taxonomy
from .repeats import _blosum, tandem_regions
from .single import (CATEGORY_LABELS, MIN_QCOV, classify_gene, classify_structure, home_group, infer_self,
                     summarize)
from . import config

GENE_COLUMNS = ["gene_id", "contig", "start", "end", "strand", "length", "category", "strength", "AI",
                "donor_group", "donor_share", "support", "n_outside", "repeat_hits", "best_hit", "best_taxon",
                "best_group", "best_pident", "best_evalue", "best_home_taxon", "best_home_pident", "mammal_hit",
                "repeats", "plddt", "struct_group", "struct_share", "struct_tm", "struct_hit", "struct_name",
                "struct_taxon", "struct_mammal"]

_TAX = {}


def get_taxonomy():
    if "t" not in _TAX:
        _TAX["t"] = taxonomy(config.REFERENCE)
    return _TAX["t"]


def resolve_taxon(tax, text):
    """'561', 'Escherichia' or 'Escherichia coli' -> taxid (None if empty or unknown)."""
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if "names" not in _TAX:
        _TAX["names"] = {v.lower(): k for k, v in tax.name.items()}
    return _TAX["names"].get(text.lower())


def read_fasta(data):
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    seqs, name, buf = {}, None, []
    for line in io.StringIO(data.decode(errors="replace")):
        line = line.strip()
        if line.startswith(">"):
            if name:
                seqs[name] = "".join(buf)
            name, buf = line[1:].split()[0] if line[1:].split() else "seq%d" % (len(seqs) + 1), []
        elif name:
            buf.append(line)
    if name:
        seqs[name] = "".join(buf)
    return seqs


def is_nucleotide(seqs):
    s = "".join(list(seqs.values())[:50])[:200000].upper()
    return bool(s) and sum(s.count(c) for c in "ACGTUN") >= 0.9 * len(s)


def call_genes(contigs):
    """Prokaryote gene calling with pyrodigal: single mode trained on the genome (metagenome mode for
    < 100 kb). Returns ({gene: protein}, {gene: (contig, start, end, strand)})."""
    import pyrodigal
    total = sum(len(s) for s in contigs.values())
    meta = total < 100_000
    gf = pyrodigal.GeneFinder(meta=meta)
    if not meta:
        gf.train(*[s.upper().encode() for s in contigs.values()])
    prots, coords = {}, {}
    for cid, seq in contigs.items():
        for k, g in enumerate(gf.find_genes(seq.upper().encode()), 1):
            gid = "%s_%d" % (cid, k)
            prots[gid] = g.translate().rstrip("*")
            coords[gid] = (cid, g.begin, g.end, "+" if g.strand == 1 else "-")
    return prots, coords


def clean_proteins(seqs, min_len=30):
    out, seen = {}, set()
    for k, v in seqs.items():
        v = re.sub(r"[^A-Za-z]", "", v).upper()
        k = re.sub(r"[^\w.:-]", "_", k)[:60]
        while k in seen:
            k += "_"
        seen.add(k)
        if len(v) >= min_len:
            out[k] = v
    return out


def _fasta(seqs):
    return "".join(">%s\n%s\n" % kv for kv in seqs.items())


def run(job_dir, raw, params, tools, folder=None, progress=lambda *a: None):
    """raw: bytes of the input file. params: self_taxon (str), structure (bool), max_structures (int)."""
    t0 = time.time()
    tax = get_taxonomy()
    progress("input", "读取输入", 0.02)
    seqs = read_fasta(raw)
    if not seqs:
        raise ValueError("输入里没有 FASTA 序列")
    coords = {}
    if is_nucleotide(seqs):
        progress("input", "核酸序列：用 pyrodigal 预测基因（%d 条 contig）" % len(seqs), 0.04)
        seqs, coords = call_genes(seqs)
    prots = clean_proteins(seqs)
    if not prots:
        raise ValueError("没有 ≥ 30 aa 的蛋白")
    with open(os.path.join(job_dir, "proteins.faa"), "w") as fh:
        fh.write(_fasta(prots))
    progress("search", "DIAMOND 比对参考库：%d 个蛋白" % len(prots), 0.08)
    hits = tools.diamond(_fasta(prots))
    with open(os.path.join(job_dir, "hits.tsv"), "w") as fh:
        keys = ["query", "target", "taxid", "pident", "alnlen", "evalue", "bits", "qstart", "qend", "tstart",
                "tend", "qlen", "tlen"]
        fh.write("\t".join(keys) + "\n")
        for h in hits:
            fh.write("\t".join(str(h[k]) for k in keys) + "\n")
    by_q = defaultdict(list)
    for h in hits:
        by_q[h["query"]].append(h)

    self_taxid = resolve_taxon(tax, params.get("self_taxon"))
    note = ""
    if params.get("self_taxon") and not self_taxid:
        raise ValueError("找不到类群：%s" % params["self_taxon"])
    if not self_taxid:
        self_taxid, rank, n = infer_self(hits, tax)
        note = ("自身类群由 %d 个近乎相同的最佳命中推断为 %s（%s）" % (n, tax.name.get(self_taxid, self_taxid), rank)
                if self_taxid else "无法推断自身类群：没有排除任何类群，建议手动指定")
    home = home_group(tax, self_taxid) if self_taxid else (1, "root")
    progress("repeats", "标注重复区段（Pfam、卷曲螺旋、串联重复）", 0.55)
    self_lin = lambda t: self_taxid and self_taxid in tax.lineage(t)  # noqa: E731
    targets = set()
    for q, hs in by_q.items():
        hs.sort(key=lambda h: -h["bits"])
        k = 0
        for h in hs:
            if h["qcov"] >= MIN_QCOV and not self_lin(h["taxid"]):
                targets.add(h["target"])
                k += 1
                if k >= 20:
                    break
    tseqs = tools.sequences(sorted(targets)) if targets else {}
    reg = tools.repeats(dict(prots, **{"t|" + k: v for k, v in tseqs.items()}))
    B = _blosum()
    qreg, treg = defaultdict(list), defaultdict(list)
    for k, v in reg.items():
        (treg[k[2:]] if k.startswith("t|") else qreg[k]).extend((a, b) for a, b, _ in v)
    kinds = defaultdict(set)
    for k, v in reg.items():
        if not k.startswith("t|"):
            kinds[k].update(x for _, _, x in v)
    for k, v in prots.items():
        t = tandem_regions(v, B)
        qreg[k].extend(t)
        if t:
            kinds[k].add("tandem_repeat")
    for k, v in tseqs.items():
        treg[k].extend(tandem_regions(v, B))

    progress("classify", "逐个基因判定来源", 0.7)
    rows = []
    for g, v in prots.items():
        r = classify_gene(g, len(v), by_q.get(g, []), tax, self_taxid, home, qreg.get(g), treg)
        c = coords.get(g, ("", "", "", ""))
        r.update(contig=c[0], start=c[1], end=c[2], strand=c[3], repeats=",".join(sorted(kinds.get(g, ()))))
        rows.append(r)

    if params.get("structure") and folder is not None:
        orphans = [r for r in rows if r["category"] == "orphan" and 50 <= r["length"] <= 1000]
        orphans = orphans[: int(params.get("max_structures", 400))]
        if orphans:
            progress("structure", "ESMFold 预测 %d 个孤儿蛋白的结构" % len(orphans), 0.75)
            folded = folder.fold({r["gene_id"]: prots[r["gene_id"]] for r in orphans})
            sdir = os.path.join(job_dir, "structures")
            os.makedirs(sdir, exist_ok=True)
            pdbs = {}
            for g, s in folded.items():
                if "pdb" in s:
                    pdbs[g] = s["pdb"]
                    with open(os.path.join(sdir, g + ".pdb"), "w") as fh:
                        fh.write(s["pdb"])
            progress("structure", "Foldseek 比对 AlphaFold Swiss-Prot", 0.9)
            shits = tools.foldseek(pdbs) if pdbs else []
            sby = defaultdict(list)
            for h in shits:
                sby[h["query"]].append(h)
            st = sorted(set(h["target"] for h in shits))
            sseq = tools.sequences(st, source="sprot") if st else {}
            sreg = tools.repeats({"t|" + k: v for k, v in sseq.items()}) if sseq else {}
            streg = defaultdict(list)
            for k, v in sreg.items():
                streg[k[2:]].extend((a, b) for a, b, _ in v)
            for k, v in sseq.items():
                streg[k].extend(tandem_regions(v, B))
            byg = {r["gene_id"]: r for r in rows}
            for g in pdbs:
                byg[g].update(plddt=folded[g]["plddt"],
                              **classify_structure(sby.get(g, []), tax, self_taxid, qreg.get(g), streg))
            for g, s in folded.items():
                if "skipped" in s:
                    byg[g]["struct_group"] = "skipped"

    with open(os.path.join(job_dir, "genes.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GENE_COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    summary.update(self_taxid=self_taxid, self_name=tax.name.get(self_taxid, "") if self_taxid else "",
                   home_group=home[1], note=note, labels=CATEGORY_LABELS, seconds=round(time.time() - t0),
                   genome_input=bool(coords), structure=bool(params.get("structure")),
                   structural=dict(_count(r.get("struct_group") for r in rows)))
    with open(os.path.join(job_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    progress("done", "完成", 1.0)
    return summary


def _count(xs):
    c = defaultdict(int)
    for x in xs:
        if x:
            c[x] += 1
    return c
