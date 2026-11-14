"""Structural donor search for candidates without any outside sequence homolog.

Folds are far more conserved than sequences: a protein with no BLAST-detectable homolog outside
its own genus may still match a known structure. This module takes the `donors` output, predicts
a structure for one representative per event with ESMFold (GPU), searches the predictions with
Foldseek against AlphaFold DB structures (e.g. the Swiss-Prot set) and assigns the donor group of
the closest structural matches outside the recipient's own genus.

Steps (each skips work that is already done):
  queries   events whose genes have no outside homolog ("no outside homolog" in
            donors_by_gene.tsv); representative = longest protein of the event
  profile   iterative MMseqs2 profile search (GPU) of the representatives against the donors
            reference sequences: more sensitive than DIAMOND for remote homologs; classified
            like `donors` (profile/donors_by_gene.tsv)
  predict   ESMFold (HuggingFace `facebook/esmfold_v1`, local copy) -> pdb/<gene>.pdb, mean pLDDT
  pfam      repeat-like regions of the representatives:
            - Pfam-A domains (pyhmmer hmmsearch, gathering cut-offs) of type Repeat / Coiled-coil /
              Disordered, or in a clan made mostly of Repeat families (LRR, TPR, ankyrin, ...)
            - tandem repeats: BLOSUM62 self-comparison, off-diagonal windows TANDEM_Z SDs above the
              expectation for the sequence's own composition (tandem_regions)
            - coiled coils: COILS / ncoils, window 28, a/d unweighted (-w), P >= COIL_MIN_P
  search    foldseek createdb (queries; target from a PDB tar/folder) + foldseek search
  classify  target accession -> NCBI taxid (UniProt FASTA headers, OX=), donor groups as in
            `donors`; hits need E <= 1e-3, homology probability >= 0.5, query coverage >= 0.4,
            TM >= 0.5. Hits whose aligned segment lies mostly (>= 50%) on repeat-like regions, of the
            query or of the target, and that are less than 40% identical, are left out: such folds recur in unrelated proteins of
            every kingdom, so a match says nothing about the donor. Divergent orphan queries often
            escape the Pfam cut-offs, so the targets (Swiss-Prot / reference proteins hit) are
            scanned with the repeat-like HMMs too. The same filter is applied to the profile hits
            (profile_nr/donors_by_gene.tsv)
"""
from __future__ import annotations

import csv
import gzip
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict

from .donors import _fasta_gz, taxonomy
from .repeats import (AA, COIL_MIN_P, KEEP_IDENTITY, MAX_REPEAT_OVERLAP, REPEAT_DOMINANCE,  # noqa: F401
                      REPEAT_TYPES, TANDEM_Z, _blosum, _hmm_path, _overlap, _pfam_meta, _s,
                      coil_regions, tandem_regions)

MIN_PROB, MAX_EVALUE, MIN_QCOV, MIN_TM = 0.5, 1e-3, 0.4, 0.5      # TM >= 0.5: same fold


def log(msg):
    print("[structure] " + msg, file=sys.stderr, flush=True)


def queries(donors_dir, out):
    os.makedirs(out, exist_ok=True)
    seqs = dict(_fasta_gz(os.path.join(donors_dir, "queries.faa")))
    by_ev = defaultdict(list)
    with open(os.path.join(donors_dir, "donors_by_gene.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["top_group"] == "no outside homolog" and r["gene_id"] in seqs:
                by_ev[r["event_id"]].append(r)
    with open(os.path.join(out, "queries.faa"), "w") as fa, open(os.path.join(out, "queries.tsv"), "w") as ft:
        ft.write("gene_id\tevent_id\tn_genes\tlength\trecipient\tself_taxid\tself_name\tP_HGT\n")
        for ev, rs in sorted(by_ev.items()):
            rep = max(rs, key=lambda r: len(seqs[r["gene_id"]]))
            s = seqs[rep["gene_id"]]
            fa.write(">%s\n%s\n" % (rep["gene_id"], s))
            ft.write("\t".join([rep["gene_id"], ev, str(len(rs)), str(len(s)), rep["recipient"],
                                rep["self_taxid"], rep["self_name"], rep["P_HGT"]]) + "\n")
    log("queries: %d events (%d genes) without outside sequence homolog"
        % (len(by_ev), sum(len(v) for v in by_ev.values())))


def profile(out, donors_dir, ref, mmseqs, threads=16, iterations=3, gpu=True):
    """Iterative profile search of the representatives against the donors reference."""
    from .donors import classify as classify_seq
    pdir = os.path.join(out, "profile")
    os.makedirs(pdir, exist_ok=True)
    keep = set(n for n, _ in _fasta_gz(os.path.join(out, "queries.faa")))
    with open(os.path.join(donors_dir, "queries.tsv")) as fi, open(os.path.join(pdir, "queries.tsv"), "w") as fo:
        for k, line in enumerate(fi):
            if k == 0 or line.split("\t", 1)[0] in keep:
                fo.write(line)
    shutil.copy(os.path.join(out, "queries.faa"), os.path.join(pdir, "queries.faa"))
    hits = os.path.join(pdir, "hits.m8")
    if not os.path.exists(hits):
        target = os.path.join(ref, "refDB_pad" if gpu else "refDB")
        tmp = os.path.join(pdir, "tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        q, res = os.path.join(tmp, "q"), os.path.join(tmp, "r")
        subprocess.run([mmseqs, "createdb", os.path.join(pdir, "queries.faa"), q], check=True)
        cmd = [mmseqs, "search", q, target, res, os.path.join(tmp, "t"), "--num-iterations", str(iterations),
               "-e", "1e-5", "--max-seqs", "2000", "--threads", str(threads)]
        cmd += ["--gpu", "1"] if gpu else ["-s", "7.5"]
        t0 = time.time()
        subprocess.run(cmd, check=True)
        subprocess.run([mmseqs, "convertalis", q, target, res, hits + ".part", "--format-output",
                        "query,target,pident,alnlen,evalue,bits,qcov,tcov,qstart,qend,tstart,tend", "--threads", str(threads)],
                       check=True)
        os.replace(hits + ".part", hits)
        shutil.rmtree(tmp, ignore_errors=True)
        log("profile search (%d iterations, %s): %.0f s" % (iterations, "GPU" if gpu else "CPU", time.time() - t0))
    classify_seq(pdir, taxonomy(ref))


def predict(out, model_dir, max_len=1000, chunk=64):
    """ESMFold on the GPU; one PDB per representative, mean pLDDT in predict.tsv."""
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    pdir = os.path.join(out, "pdb")
    os.makedirs(pdir, exist_ok=True)
    done = set()
    stats = os.path.join(out, "predict.tsv")
    if os.path.exists(stats):
        with open(stats) as fh:
            done = set(line.split("\t")[0] for line in fh if not line.startswith("gene_id"))
    todo = [(n, s) for n, s in _fasta_gz(os.path.join(out, "queries.faa")) if n not in done]
    todo.sort(key=lambda x: len(x[1]))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = EsmForProteinFolding.from_pretrained(model_dir).cuda().eval()
    model.esm = model.esm.half()                 # the 3B language model in fp16; trunk stays fp32
    model.trunk.set_chunk_size(chunk)
    new = not os.path.exists(stats)
    t0 = time.time()
    with open(stats, "a") as fo:
        if new:
            fo.write("gene_id\tlength\tmean_plddt\tseconds\n")
        for k, (name, seq) in enumerate(todo, 1):
            if len(seq) > max_len:
                fo.write("%s\t%d\tNA\tskipped (> %d aa)\n" % (name, len(seq), max_len))
                continue
            seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq.upper())
            t1 = time.time()
            ids = tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
            with torch.no_grad():
                o = model(ids)
            pdb = model.output_to_pdb(o)[0]
            pl = o["plddt"][0, :, 1].float()          # CA atom
            mean = float(pl.mean()) * (100.0 if float(pl.max()) <= 1.0 else 1.0)
            with open(os.path.join(pdir, name + ".pdb"), "w") as fh:
                fh.write(pdb)
            fo.write("%s\t%d\t%.1f\t%.1f\n" % (name, len(seq), mean, time.time() - t1))
            fo.flush()
            if k % 100 == 0:
                log("predicted %d/%d (%.0f s)" % (k, len(todo), time.time() - t0))
    log("prediction done: %d structures in %.0f s" % (len(todo), time.time() - t0))


def search(out, target, foldseek, threads=16, gpu=False):
    tdb = os.path.join(os.path.dirname(os.path.abspath(target)), "targetDB")
    if not os.path.exists(tdb + ".dbtype"):
        log("building the target structure database from %s" % target)
        subprocess.run([foldseek, "createdb", target, tdb, "--threads", str(threads)], check=True)
    q, res, tmp = os.path.join(out, "queryDB"), os.path.join(out, "aln"), os.path.join(out, "tmp")
    subprocess.run([foldseek, "createdb", os.path.join(out, "pdb"), q, "--threads", str(threads)], check=True)
    cmd = [foldseek, "search", q, tdb, res, tmp, "-a", "-e", "1e-2", "--max-seqs", "2000", "--threads", str(threads)]
    t0 = time.time()
    try:
        if not gpu:
            raise RuntimeError("CPU requested")
        if not os.path.exists(tdb + "_pad.dbtype"):  # Foldseek-GPU: ungapped prefilter on a padded DB
            subprocess.run([foldseek, "makepaddedseqdb", tdb, tdb + "_pad"], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run([cmd[0], cmd[1], cmd[2], tdb + "_pad"] + cmd[4:] + ["--gpu", "1"], check=True)
        tdb += "_pad"
    except (RuntimeError, subprocess.CalledProcessError) as e:
        if gpu:                                        # e.g. the avx2 release: has --gpu but no CUDA
            log("Foldseek-GPU not usable (%s); searching on the CPU" % e)
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(cmd, check=True)
        gpu = False
    log("Foldseek search (%s): %.0f s" % ("GPU" if gpu else "CPU", time.time() - t0))
    subprocess.run([foldseek, "convertalis", q, tdb, res, os.path.join(out, "hits.m8"), "--format-output",
                    "query,target,evalue,bits,prob,alntmscore,lddt,fident,qcov,tcov,qstart,qend,tstart,tend",
                    "--threads", str(threads)],
                   check=True)
    shutil.rmtree(tmp, ignore_errors=True)


def pfam(out, pfam_dir, threads=16, ncoils="ncoils"):
    """Repeat-like regions of the representatives: Pfam-A domains (repeat-like ones), tandem repeats
    and coiled coils. Hits on these regions are ignored when the donor group is assigned."""
    import pyhmmer

    hmm = _hmm_path(pfam_dir)
    meta = _pfam_meta(os.path.join(pfam_dir, "Pfam-A.hmm.dat.gz"))
    abc = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(os.path.join(out, "queries.faa"), digital=True, alphabet=abc) as sf:
        seqs = sf.read_block()
    t0, rows = time.time(), []
    with pyhmmer.plan7.HMMFile(hmm) as hf:
        for top in pyhmmer.hmmsearch(hf, seqs, cpus=threads, bit_cutoffs="gathering"):
            q = getattr(top, "query", None)
            acc = _s(getattr(q, "accession", None) if q is not None else getattr(top, "query_accession", None))
            acc = acc.split(".")[0]
            pid, tp, cl, rep = meta.get(acc, (_s(getattr(q, "name", None)), "", "", False))
            for hit in top:
                if not hit.included:
                    continue
                for d in hit.domains:
                    if d.included:
                        rows.append((_s(hit.name), acc, pid, tp, cl, "yes" if rep else "", d.env_from, d.env_to,
                                     d.i_evalue, d.score))
    genes = set(r[0] for r in rows)
    log("Pfam (%d queries, %.0f s): %d with a domain, %d with a repeat-like domain"
        % (len(seqs), time.time() - t0, len(genes), len(set(r[0] for r in rows if r[5]))))
    rows = [r[:8] + ("%.2g" % r[8], "%.1f" % r[9]) for r in rows]
    qs = dict((n.split()[0], v) for n, v in _fasta_gz(os.path.join(out, "queries.faa")))
    B = _blosum()
    nt = nc = 0
    for g, v in qs.items():
        for a, b in tandem_regions(v, B):
            rows.append((g, "-", "tandem_repeat", "Repeat", "", "yes", a, b, "NA", "NA"))
            nt += 1
    for g, iv in coil_regions(qs, ncoils).items():
        for a, b in iv:
            rows.append((g, "-", "coiled_coil", "Coiled-coil", "", "yes", a, b, "NA", "NA"))
            nc += 1
    rows.sort(key=lambda r: (r[0], r[6]))
    with open(os.path.join(out, "pfam.tsv"), "w") as fo:
        fo.write("gene_id\tpfam_acc\tpfam_id\ttype\tclan\trepeat_like\tstart\tend\ti_evalue\tscore\n")
        for r in rows:
            fo.write("%s\t%s\t%s\t%s\t%s\t%s\t%d\t%d\t%s\t%s\n" % r)
    log("tandem repeats in %d queries, coiled coils in %d queries"
        % (len(set(r[0] for r in rows if r[2] == "tandem_repeat")), len(set(r[0] for r in rows if r[2] == "coiled_coil"))))


def _target_ids(out):
    ids = set()
    if os.path.exists(os.path.join(out, "hits.m8")):
        ids.update(_acc(line.split("\t", 2)[1]) for line in open(os.path.join(out, "hits.m8")))
    if os.path.exists(os.path.join(out, "profile", "hits.m8")):
        ids.update(line.split("\t", 2)[1].rpartition("|")[0] for line in open(os.path.join(out, "profile", "hits.m8")))
    return ids


def target_repeats(out, pfam_dir, sprot_fasta, ref, threads=16, ncoils="ncoils"):
    """{target accession: repeat-like intervals (Pfam, tandem, coiled coil)} for every Foldseek and
    profile target."""
    import pyhmmer

    path = os.path.join(out, "repeat_targets.tsv")
    need = _target_ids(out)
    if os.path.exists(path):
        reg, seen = defaultdict(list), set(open(path + ".ids").read().split())
        if need <= seen:
            for r in csv.DictReader(open(path), delimiter="\t"):
                reg[r["target"]].append((int(r["start"]), int(r["end"])))
            return reg
    t0, seqs = time.time(), {}
    srcs = [(sprot_fasta, lambda h: h.split("|")[1] if "|" in h else h.split()[0])]
    if ref and os.path.exists(os.path.join(ref, "reference.faa")):
        srcs.append((os.path.join(ref, "reference.faa"), lambda h: h.split()[0].rpartition("|")[0]))
    for src, key in srcs:                              # stream: keep only the sequences that were hit
        op = gzip.open if src.endswith(".gz") else open
        with op(src, "rt") as fh:
            cur = None
            for line in fh:
                if line[0] == ">":
                    a = key(line[1:].rstrip("\n"))
                    cur = a if a in need and a not in seqs else None
                    if cur:
                        seqs[cur] = []
                elif cur:
                    seqs[cur].append(line.strip().replace("*", ""))
    tmp = os.path.join(out, "pfam_targets.faa")
    with open(tmp, "w") as fo:
        for a, v in seqs.items():
            fo.write(">%s\n%s\n" % (a, "".join(v)))
    meta = _pfam_meta(os.path.join(pfam_dir, "Pfam-A.hmm.dat.gz"))
    with pyhmmer.plan7.HMMFile(_hmm_path(pfam_dir)) as hf:
        hmms = [h for h in hf if meta.get(_s(h.accession).split(".")[0], (0, 0, 0, False))[3]]
    abc = pyhmmer.easel.Alphabet.amino()
    with pyhmmer.easel.SequenceFile(tmp, digital=True, alphabet=abc) as sf:
        block = sf.read_block()
    reg, n = defaultdict(list), 0
    with open(path + ".part", "w") as fo:
        fo.write("target\tpfam_id\tstart\tend\n")
        for top in pyhmmer.hmmsearch(hmms, block, cpus=threads, bit_cutoffs="gathering"):
            q = getattr(top, "query", None)
            pid = _s(getattr(q, "name", None) if q is not None else getattr(top, "query_name", None))
            for hit in top:
                if hit.included:
                    for d in hit.domains:
                        if d.included:
                            fo.write("%s\t%s\t%d\t%d\n" % (_s(hit.name), pid, d.env_from, d.env_to))
                            reg[_s(hit.name)].append((d.env_from, d.env_to))
                            n += 1
        npf, B = len(reg), _blosum()
        full = dict((a, "".join(v)) for a, v in seqs.items())
        nt = nc = 0
        for a, v in full.items():
            for x, y in tandem_regions(v, B):
                fo.write("%s\ttandem_repeat\t%d\t%d\n" % (a, x, y))
                reg[a].append((x, y))
                nt += 1
        for a, iv in coil_regions(full, ncoils).items():
            for x, y in iv:
                fo.write("%s\tcoiled_coil\t%d\t%d\n" % (a, x, y))
                reg[a].append((x, y))
                nc += 1
    os.replace(path + ".part", path)
    with open(path + ".ids", "w") as fo:
        fo.write("\n".join(sorted(seqs)) + "\n")
    os.remove(tmp)
    log("targets: %d with repeat-like Pfam domains (%d HMMs), %d tandem-repeat and %d coiled-coil segments; "
        "%d of %d targets carry some repeat (%.0f s)" % (npf, len(hmms), nt, nc, len(reg), len(seqs), time.time() - t0))
    return reg


def _load_pfam(out):
    """{gene: sorted repeat-like intervals}, {gene: domain list in sequence order}."""
    reg, doms = defaultdict(list), defaultdict(list)
    path = os.path.join(out, "pfam.tsv")
    if not os.path.exists(path):
        return reg, doms
    for r in csv.DictReader(open(path), delimiter="\t"):
        doms[r["gene_id"]].append(r["pfam_id"] + ("*" if r["repeat_like"] else ""))
        if r["repeat_like"]:
            reg[r["gene_id"]].append((int(r["start"]), int(r["end"])))
    return reg, doms


def _is_repeat(qreg, treg, f, qi, ti):
    """Aligned segment mostly on a repeat-like region of the query (cols qi, qi+1) or the target (ti, ti+1)."""
    if len(f) < ti + 2:
        return len(f) >= qi + 2 and _overlap(qreg, int(f[qi]), int(f[qi + 1])) >= MAX_REPEAT_OVERLAP
    return (_overlap(qreg, int(f[qi]), int(f[qi + 1])) >= MAX_REPEAT_OVERLAP
            or _overlap(treg, int(f[ti]), int(f[ti + 1])) >= MAX_REPEAT_OVERLAP)


def filter_profile(out, ref, treg=None, top=10):
    """Profile hits minus those on repeat-like Pfam regions, classified like `donors`; genes whose top
    outside hits are mostly repeat-driven are labelled "repeat-like" instead of getting a donor group."""
    from .donors import MIN_QCOV as SEQ_QCOV, classify as classify_seq
    pdir, ndir = os.path.join(out, "profile"), os.path.join(out, "profile_nr")
    tax, treg = taxonomy(ref), treg or {}
    reg, _ = _load_pfam(out)
    os.makedirs(ndir, exist_ok=True)
    shutil.copy(os.path.join(pdir, "queries.tsv"), os.path.join(ndir, "queries.tsv"))
    selft = {r["gene_id"]: int(r["self_taxid"])
             for r in csv.DictReader(open(os.path.join(pdir, "queries.tsv")), delimiter="\t")}
    by_q = defaultdict(list)
    for line in open(os.path.join(pdir, "hits.m8")):
        f = line.rstrip("\n").split("\t")
        if len(f) < 10:
            raise SystemExit("profile/hits.m8 has no query coordinates: delete it and rerun the profile step")
        pid = float(f[2]) / (100.0 if float(f[2]) > 1.0 else 1.0)
        rep = pid < KEEP_IDENTITY and _is_repeat(reg.get(f[0]), treg.get(f[1].rpartition("|")[0]), f, 8, 10)
        by_q[f[0]].append((float(f[5]), rep, f, line))
    n = k = 0
    dominated = {}
    with open(os.path.join(ndir, "hits.m8"), "w") as fo:
        for g, hs in by_q.items():
            hs.sort(key=lambda h: -h[0])
            st = selft.get(g)
            outside = [h for h in hs if (float(h[2][6]) / (100.0 if float(h[2][6]) > 1 else 1.0)) >= SEQ_QCOV
                       and st not in tax.lineage(int(h[2][1].rpartition("|")[2]))][:top]
            tb, rb = sum(h[0] for h in outside), sum(h[0] for h in outside if h[1])
            n += len(hs)
            k += sum(1 for h in hs if h[1])
            if outside and rb >= REPEAT_DOMINANCE * tb:
                dominated[g] = rb / tb
                continue
            for h in hs:
                if not h[1]:
                    fo.write(h[3])
    log("profile hits on repeat-like regions (identity < %.0f%%): %d of %d; genes dominated by them: %d"
        % (100 * KEEP_IDENTITY, k, n, len(dominated)))
    classify_seq(ndir, tax)
    path = os.path.join(ndir, "donors_by_gene.tsv")
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    for r in rows:
        if r["gene_id"] in dominated:
            r.update(top_group="repeat-like", top_group_share="%.2f" % dominated[r["gene_id"]])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    c = Counter(r["top_group"] for r in rows)
    log("profile genes after the repeat filter: " + ", ".join("%s %d" % kv for kv in c.most_common()))


def _acc(target):
    m = re.match(r"AF-([A-Z0-9]+)-F\d+", target)
    return m.group(1) if m else target.split(".")[0]


def _group(good, tax, top, none):
    w = Counter()
    for h, t in good[:top]:
        w[tax.group(t)] += h["bits"]
    tot = sum(w.values())
    return (w.most_common(1)[0][0], w.most_common(1)[0][1] / tot) if tot else (none, 0.0)


def classify(out, ref, sprot_fasta, pfam_dir=None, threads=16, top=10, ncoils="ncoils"):
    tax = taxonomy(ref)
    treg = target_repeats(out, pfam_dir, sprot_fasta, ref, threads, ncoils) if pfam_dir else {}
    taxid, pname = {}, {}
    with gzip.open(sprot_fasta, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                acc = line[1:].split("|")[1] if "|" in line else line[1:].split()[0]
                m = re.search(r" OX=(\d+)", line)
                if m:
                    taxid[acc] = int(m.group(1))
                pname[acc] = re.sub(r" OS=.*", "", line.rstrip("\n").split(" ", 1)[-1])
    plddt = {}
    with open(os.path.join(out, "predict.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            plddt[r["gene_id"]] = r["mean_plddt"]
    reg, doms = _load_pfam(out)
    hits = defaultdict(list)
    with open(os.path.join(out, "hits.m8")) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            h = dict(target=f[1], evalue=float(f[2]), bits=float(f[3]), prob=float(f[4]), tm=float(f[5]),
                     lddt=float(f[6]), fident=float(f[7]), qcov=float(f[8]))
            h["rep"] = h["fident"] < KEEP_IDENTITY and _is_repeat(reg.get(f[0]), treg.get(_acc(f[1])), f, 10, 12)
            hits[f[0]].append(h)
    rows = list(csv.DictReader(open(os.path.join(out, "queries.tsv")), delimiter="\t"))
    res = []
    for q in rows:
        st, L = int(q["self_taxid"]), int(q["length"])
        allg = []
        for h in sorted(hits.get(q["gene_id"], []), key=lambda h: -h["bits"]):
            t = taxid.get(_acc(h["target"]))
            if (t is None or h["evalue"] > MAX_EVALUE or h["prob"] < MIN_PROB or h["qcov"] < MIN_QCOV
                    or h["tm"] < MIN_TM or st in tax.lineage(t)):
                continue
            allg.append((h, t))
        good = [(h, t) for h, t in allg if not h["rep"]]
        grp_all, _ = _group(allg, tax, top, "no structural match")
        tb = sum(h["bits"] for h, t in allg[:top])
        rb = sum(h["bits"] for h, t in allg[:top] if h["rep"])
        if allg and rb >= REPEAT_DOMINANCE * tb:      # the evidence is mostly repeat folds: no donor call
            grp, share = "repeat-like", rb / tb
        else:
            grp, share = _group(good, tax, top, "no structural match")
        mam = next(((h, t) for h, t in good if 40674 in tax.lineage(t)), None)
        b = good[0] if good else None
        res.append(dict(q, mean_plddt=plddt.get(q["gene_id"], ""),
                        pfam=",".join(doms.get(q["gene_id"], [])),
                        repeat_frac="%.2f" % _overlap(reg.get(q["gene_id"]), 1, L),
                        n_struct_hits=len(good), n_repeat_hits=len(allg) - len(good), top_group=grp,
                        top_group_share="%.2f" % share, top_group_all=grp_all,
                        best_hit=_acc(b[0]["target"]) if b else "",
                        best_hit_name=pname.get(_acc(b[0]["target"]), "") if b else "",
                        best_hit_taxon=tax.name.get(b[1], "") if b else "",
                        best_hit_group=tax.group(b[1]) if b else "",
                        best_tm="%.2f" % b[0]["tm"] if b else "", best_prob="%.2f" % b[0]["prob"] if b else "",
                        best_fident="%.2f" % b[0]["fident"] if b else "",
                        mammal_hit=("%s (%s, TM %.2f, prob %.2f)" % (tax.name.get(mam[1], ""), _acc(mam[0]["target"]),
                                                                    mam[0]["tm"], mam[0]["prob"])) if mam else ""))
    cols = list(rows[0].keys()) + ["mean_plddt", "pfam", "repeat_frac", "n_struct_hits", "n_repeat_hits",
                                   "top_group", "top_group_share", "top_group_all", "best_hit", "best_hit_name",
                                   "best_hit_taxon", "best_hit_group", "best_tm", "best_prob", "best_fident",
                                   "mammal_hit"] if rows else []
    with open(os.path.join(out, "structure_donors.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    c = Counter(r["top_group"] for r in res)
    log("events by closest structural group (repeat-like hits left out): "
        + ", ".join("%s %d" % kv for kv in c.most_common()))
    log("events with a mammal structural match: %d (mammal top group %d; before the repeat filter %d)"
        % (sum(1 for r in res if r["mammal_hit"]), sum(1 for r in res if r["top_group"] == "Mammalia"),
           sum(1 for r in res if r["top_group_all"] == "Mammalia")))
    if (reg or treg) and os.path.exists(os.path.join(out, "profile", "hits.m8")):
        filter_profile(out, ref, treg)


def run_structure(a):
    steps = a.steps.split(",")
    if "queries" in steps:
        queries(a.donors, a.out)
    if "profile" in steps:
        profile(a.out, a.donors, a.ref, a.mmseqs, a.threads, gpu=not a.cpu)
    if "predict" in steps:
        predict(a.out, a.model, a.max_len)
    if "pfam" in steps:
        pfam(a.out, a.pfam, a.threads, a.ncoils)
    if "search" in steps:
        search(a.out, a.target, a.foldseek, a.threads, gpu=not a.cpu)
    if "classify" in steps:
        classify(a.out, a.ref, a.sprot, a.pfam, a.threads, ncoils=a.ncoils)
