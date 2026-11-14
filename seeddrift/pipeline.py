"""One genome in, a table of candidate foreign genes out.

  input       proteins, CDS, GenBank / EMBL, GFF3 (+ FASTA), or raw contigs: prokaryote genes with
              pyrodigal, eukaryote genes with MetaEuk; for raw contigs the domain can be detected by
              a quick search of a sample of genes (inputs.py)
  search      DIAMOND --sensitive against the reference library
  deep        genes without an outside hit: the most sensitive search without repeat masking (tantan
              masking hides repeat-rich proteins completely): MMseqs2-GPU, else DIAMOND --ultra-sensitive
  self        the genome's own taxon: given, or inferred from near-identical best hits
  repeats     repeat-like regions of the genes and of the hit targets (Pfam, coiled coils: tools
              service; tandem repeats: here)
  classify    category, Alien Index, donor group, support (single.py)
  phylogeny   MAFFT + FastTree for each candidate: are its nearest relatives outside the home group?
  context     composition (GC, GC3), clusters of candidates (islands), contaminant contigs
  confidence  high / medium / low for candidates (evidence.py)
  structure   optional, for orphans: ESMFold (GPU service) + Foldseek vs AlphaFold Swiss-Prot

Outputs in the job folder: proteins.faa, hits.tsv, genes.tsv, summary.json, trees/, structures/
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import Counter, defaultdict

from . import config
from .donors import taxonomy
from .evidence import FOREIGN, composition, confidence, contaminant_contigs, islands, tree_jobs, tree_verdict
from .inputs import load
from .repeats import _blosum, is_repeat_hit, tandem_regions
from .single import (CATEGORY_LABELS, EUKARYOTA, MIN_QCOV, classify_gene, classify_structure, home_group,
                     infer_self, summarize)

GENE_COLUMNS = ["gene_id", "contig", "start", "end", "strand", "exons", "length", "category", "confidence",
                "strength", "AI", "donor_group", "donor_share", "support", "tree", "tree_outside", "tree_group",
                "tree_taxa", "gc", "gc3", "comp_z", "island", "contig_flag", "search", "n_outside", "repeat_hits",
                "best_hit", "best_taxon", "best_group", "best_pident", "best_evalue", "best_home_taxon",
                "best_home_pident", "mammal_hit", "repeats", "plddt", "struct_group", "struct_share", "struct_tm",
                "struct_hit", "struct_name", "struct_taxon", "struct_mammal"]
HIT_KEYS = ["query", "target", "taxid", "pident", "alnlen", "evalue", "bits", "qstart", "qend", "tstart", "tend",
            "qlen", "tlen"]

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


def clean_proteins(seqs, min_len=30):
    """Letters only, safe unique names; returns (proteins, {original name: new name})."""
    out, ren = {}, {}
    for k, v in seqs.items():
        v = re.sub(r"[^A-Za-z]", "", v).upper()
        if len(v) < min_len:
            continue
        k2 = re.sub(r"[^\w.:-]", "_", k)[:60] or "gene"
        while k2 in out:
            k2 += "_"
        out[k2], ren[k] = v, k2
    return out, ren


def _db_id(path):
    try:
        st = os.stat(path)
        return "%s:%d:%d" % (path, st.st_size, int(st.st_mtime))
    except OSError:
        return path


def cached(kind, key, fn):
    """Memoise an expensive tool call on disk ($SEEDDRIFT_DATA/cache/<kind>/<sha1>.json.gz): the same
    inputs against the same reference give the same answer, so re-running a genome with other
    settings (or an ablation) skips the searches. SEEDDRIFT_CACHE=0 turns it off."""
    import gzip
    import hashlib
    if config.env("SEEDDRIFT_CACHE", "1") == "0":
        return fn()
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()
    d = config.path("cache", kind)
    p = os.path.join(d, h + ".json.gz")
    if os.path.exists(p):
        try:
            with gzip.open(p, "rt") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    out = fn()
    os.makedirs(d, exist_ok=True)
    with gzip.open(p + ".part", "wt") as fh:
        json.dump(out, fh)
    os.replace(p + ".part", p)
    return out


def _fasta(seqs):
    return "".join(">%s\n%s\n" % kv for kv in seqs.items())


def domain_probe(tools, tax):
    """'eukaryote' or 'prokaryote' from a quick search of the longest predicted genes."""
    def probe(prots):
        sample = dict(sorted(prots.items(), key=lambda kv: -len(kv[1]))[:200])
        best = {}
        for h in tools.diamond(_fasta(sample), max_target_seqs=5, sensitivity="fast"):
            if h["query"] not in best or h["bits"] > best[h["query"]]["bits"]:
                best[h["query"]] = h
        c = Counter("eukaryote" if EUKARYOTA in tax.lineage(h["taxid"]) else "prokaryote" for h in best.values())
        return "eukaryote" if c["eukaryote"] > c["prokaryote"] else "prokaryote"
    return probe


def run(job_dir, files, params, tools, folder=None, progress=lambda *a: None):
    """files: list of raw bytes (main input first, optional second file e.g. FASTA for a GFF3).
    params: self_taxon, domain (auto/prokaryote/eukaryote), home_rank, deep, trees, structure,
    max_structures."""
    t0 = time.time()
    timings, last = {}, [None, t0]
    outer = progress

    def progress(stage, msg, frac):                  # record time per stage
        now = time.time()
        if last[0] and last[0] != stage:
            timings[last[0]] = round(timings.get(last[0], 0) + now - last[1], 1)
        if last[0] != stage:
            last[0], last[1] = stage, now
        outer(stage, msg, frac)
    if isinstance(files, (bytes, bytearray)):
        files = [files]
    tax = get_taxonomy()
    progress("input", "读取输入，识别格式", 0.02)
    g = load(files, params.get("domain", "auto"), euk_caller=lambda contigs: tools.metaeuk(_fasta(contigs)),
             domain_probe=domain_probe(tools, tax))
    prots, ren = clean_proteins(g.proteins)
    if not prots:
        raise ValueError("没有 ≥ 30 aa 的蛋白")
    coords = {ren[k]: v for k, v in g.coords.items() if k in ren}
    cds = {ren[k]: v for k, v in g.cds.items() if k in ren}
    with open(os.path.join(job_dir, "proteins.faa"), "w") as fh:
        fh.write(_fasta(prots))

    progress("search", "DIAMOND 比对参考库：%d 个蛋白" % len(prots), 0.08)
    ref = _db_id(config.DIAMOND_DB)
    hits = cached("diamond", [ref, "sensitive", _fasta(prots)], lambda: tools.diamond(_fasta(prots)))
    by_q = defaultdict(list)
    for h in hits:
        by_q[h["query"]].append(h)

    self_taxid = resolve_taxon(tax, params.get("self_taxon"))
    notes = list(g.notes)
    if params.get("self_taxon") and not self_taxid:
        raise ValueError("找不到类群：%s" % params["self_taxon"])
    if not self_taxid:
        self_taxid, rank, n = infer_self(hits, tax)
        notes.append("自身类群由 %d 个近乎相同的最佳命中推断为 %s（%s）" % (n, tax.name.get(self_taxid, self_taxid), rank)
                     if self_taxid else "无法推断自身类群：没有排除任何类群，建议手动指定")
    self_lin = (lambda t: self_taxid in tax.lineage(t)) if self_taxid else (lambda t: False)

    deep = {}
    if params.get("deep", True):
        lonely = [q for q in prots if not any(h["qcov"] >= MIN_QCOV and not self_lin(h["taxid"]) for h in by_q.get(q, []))]
        if lonely:
            progress("deep", "深度检索 %d 个暂无外部同源的基因（关闭重复屏蔽）" % len(lonely), 0.4)
            new = defaultdict(list)
            lf = _fasta({q: prots[q] for q in lonely})
            for h in cached("deep", [ref, lf], lambda: tools.deep_search(lf)):
                new[h["query"]].append(h)
            for q in lonely:
                if new.get(q):
                    by_q[q] = new[q]
                found = any(h["qcov"] >= MIN_QCOV and not self_lin(h["taxid"]) for h in new.get(q, []))
                deep[q] = "deep" if found else "deep: none"
    with open(os.path.join(job_dir, "hits.tsv"), "w") as fh:
        fh.write("\t".join(HIT_KEYS) + "\n")
        for hs in by_q.values():
            for h in hs:
                fh.write("\t".join(str(h[k]) for k in HIT_KEYS) + "\n")

    home = home_group(tax, self_taxid, params.get("home_rank", "auto")) if self_taxid else (1, "root")
    progress("repeats", "取回命中序列，标注重复区段（Pfam、卷曲螺旋、串联重复）", 0.5)
    targets = set()
    for q, hs in by_q.items():
        hs.sort(key=lambda h: -h["bits"])
        n_out = n_home = 0
        for h in hs:
            if h["qcov"] < MIN_QCOV or self_lin(h["taxid"]):
                continue
            if home[0] in tax.lineage(h["taxid"]):
                if n_home < 12:
                    targets.add(h["target"])
                    n_home += 1
            elif n_out < 25:
                targets.add(h["target"])
                n_out += 1
    tseqs = cached("sequences", [ref, sorted(targets)], lambda: tools.sequences(sorted(targets))) if targets else {}
    rin = dict(prots, **{"t|" + k: v for k, v in tseqs.items()})
    reg = cached("repeats", [_db_id(os.path.join(config.PFAM, "Pfam-A.hmm.dat.gz")), sorted(rin.items())],
                 lambda: tools.repeats(rin))
    B = _blosum()
    qreg, treg, kinds = defaultdict(list), defaultdict(list), defaultdict(set)
    for k, v in reg.items():
        if k.startswith("t|"):
            treg[k[2:]].extend((a, b) for a, b, _ in v)
        else:
            qreg[k].extend((a, b) for a, b, _ in v)
            kinds[k].update(x for _, _, x in v)
    for k, v in prots.items():
        t = tandem_regions(v, B)
        qreg[k].extend(t)
        if t:
            kinds[k].add("tandem_repeat")
    for k, v in tseqs.items():
        treg[k].extend(tandem_regions(v, B))

    progress("classify", "逐个基因判定来源", 0.62)
    rows = []
    if not params.get("repeat_filter", True):        # ablation: no repeat filter
        qreg, treg = defaultdict(list), defaultdict(list)
    for gname, v in prots.items():
        r = classify_gene(gname, len(v), by_q.get(gname, []), tax, self_taxid, home, qreg.get(gname), treg)
        c = coords.get(gname, ("", "", "", "", ""))
        r.update(contig=c[0], start=c[1], end=c[2], strand=c[3], exons=c[4] if len(c) > 4 else "",
                 repeats=",".join(sorted(kinds.get(gname, ()))),
                 search=deep.get(gname, ""))
        rows.append(r)
    byg = {r["gene_id"]: r for r in rows}

    if params.get("trees", True):
        rep_pairs = set()
        for q, hs in by_q.items():
            for h in hs:
                if is_repeat_hit(qreg.get(q, []), treg.get(h["target"], []), h["qstart"], h["qend"], h["tstart"],
                                 h["tend"], h["pident"] / 100.0):
                    rep_pairs.add((q, h["target"]))
        jobs = tree_jobs(rows, by_q, tseqs, prots, tax, self_taxid, home[0], rep_pairs)
        if jobs:
            progress("phylogeny", "为 %d 个候选建树（MAFFT + FastTree）" % len(jobs), 0.68)
            tdir = os.path.join(job_dir, "trees")
            os.makedirs(tdir, exist_ok=True)
            names = list(jobs)
            for i in range(0, len(names), 200):
                chunk = {n: jobs[n] for n in names[i:i + 200]}
                part = cached("trees", sorted((n, sorted(v.items())) for n, v in chunk.items()),
                              lambda: tools.trees(chunk))
                for n, nwk in part.items():
                    if nwk:
                        with open(os.path.join(tdir, n + ".nwk"), "w") as fh:
                            fh.write(nwk + "\n")
                    byg[n].update(tree_verdict(nwk, tax, home[0]))
                progress("phylogeny", "建树 %d / %d" % (min(i + 200, len(names)), len(names)),
                         0.68 + 0.07 * min(i + 200, len(names)) / len(names))

    comp = composition(cds)
    for gname, (gc, gc3, z) in comp.items():
        byg[gname].update(gc="%.3f" % gc, gc3="%.3f" % gc3, comp_z="%.1f" % z)
    if coords:
        for gname, n in islands(rows, coords).items():
            byg[gname]["island"] = n
    euk_genome = EUKARYOTA in tax.lineage(self_taxid) if self_taxid else False
    bad = contaminant_contigs(rows, coords) if (coords and euk_genome) else set()
    for r in rows:
        if r["contig"] in bad:
            r["contig_flag"] = "prokaryote-like contig"
        r["confidence"] = confidence(r)

    if params.get("structure") and folder is not None:
        _structure(job_dir, rows, byg, prots, params, tools, folder, tax, self_taxid, qreg, B, progress)

    with open(os.path.join(job_dir, "genes.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=GENE_COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    summary.update(
        self_taxid=self_taxid, self_name=tax.name.get(self_taxid, "") if self_taxid else "", home_group=home[1],
        input_kind=g.kind, n_contigs=len(g.contigs), genome_size=sum(g.contigs.values()), note="；".join(notes),
        labels=CATEGORY_LABELS, seconds=round(time.time() - t0), has_coords=bool(coords),
        structure=bool(params.get("structure")), structural=dict(Counter(r.get("struct_group") for r in rows
                                                                          if r.get("struct_group"))),
        confidence=dict(Counter(r["confidence"] for r in rows if r["confidence"])),
        confidence_by_category={c: dict(Counter(r["confidence"] for r in rows if r["category"] == c))
                                for c in FOREIGN},
        trees=dict(Counter(r.get("tree") for r in rows if r.get("tree"))),
        contaminant_contigs=sorted(bad), deep_rescued=sum(1 for r in rows if r.get("search") == "deep"
                                                          and r["category"] != "orphan"),
        islands=sum(1 for r in rows if int(r.get("island") or 0) >= 3), timings=timings,
        params={k: v for k, v in params.items()})
    with open(os.path.join(job_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    progress("done", "完成", 1.0)
    with open(os.path.join(job_dir, "summary.json"), "w") as fh:     # with the last stage timed
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary


def _structure(job_dir, rows, byg, prots, params, tools, folder, tax, self_taxid, qreg, B, progress):
    orphans = [r for r in rows if r["category"] == "orphan" and 50 <= r["length"] <= 1000]
    orphans = orphans[: int(params.get("max_structures", 400))]
    if not orphans:
        return
    folded, step = {}, 20
    for i in range(0, len(orphans), step):
        progress("structure", "ESMFold 预测孤儿蛋白的结构：%d / %d" % (i, len(orphans)), 0.78 + 0.14 * i / len(orphans))
        folded.update(folder.fold({r["gene_id"]: prots[r["gene_id"]] for r in orphans[i:i + step]}))
    sdir = os.path.join(job_dir, "structures")
    os.makedirs(sdir, exist_ok=True)
    pdbs = {}
    for gname, s in folded.items():
        if "pdb" in s:
            pdbs[gname] = s["pdb"]
            with open(os.path.join(sdir, gname + ".pdb"), "w") as fh:
                fh.write(s["pdb"])
    progress("structure", "Foldseek 比对 AlphaFold Swiss-Prot", 0.93)
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
    for gname in pdbs:
        byg[gname].update(plddt=folded[gname]["plddt"],
                          **classify_structure(sby.get(gname, []), tax, self_taxid, qreg.get(gname), streg))
    for gname, s in folded.items():
        if "skipped" in s:
            byg[gname]["struct_group"] = "skipped"
