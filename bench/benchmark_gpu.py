"""Benchmarks on the GPU box: homology-search engines and gene-tree methods.

Part A  search: N random proteins of the dataset vs the donor reference (UniProt reference
        proteomes) with MMseqs2-GPU, MMseqs2-CPU (-s 7.5) and DIAMOND --sensitive. Wall and CPU
        time, database build time, queries with hits, overlap of the hit lists, agreement of the
        donor group of the best hit outside Enterobacterales. MMseqs2-GPU also on all proteins.
Part B  trees: FastTree (-lg -gamma) vs IQ-TREE (LG+G4, UFBoot 1000) on families whose IQ-TREE
        tree is finished: normalised Robinson-Foulds distance, share of strongly supported
        IQ-TREE splits (UFBoot >= 95) that FastTree contradicts, CPU time per tree on a
        size-stratified subsample.
Part C  HGT calls (needs the species tree): reconcile the IQ-TREE tree (--collapse 95) and the
        FastTree tree (--collapse 0.95, SH-like supports) of the same families and compare the
        transfers they call.

usage: python bench/benchmark_gpu.py --part A|B|C --root /hy-tmp/hgtscan [--threads 16]
Writes bench/gpu/<part>.json and prints a summary.
"""
import argparse
import glob
import json
import os
import random
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from seeddrift.donors import _fasta_gz, _load_hits, taxonomy  # noqa: E402
from seeddrift.reconcile import DTLModel, reconcile_family  # noqa: E402
from seeddrift.tree import SpeciesTree, parse_newick, prune_to  # noqa: E402


def cpu_children():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime


def timed(fn, *a, **k):
    c0, t0 = cpu_children(), time.time()
    out = fn(*a, **k)
    return out, time.time() - t0, cpu_children() - c0


def sh(cmd, log=None):
    with open(log or os.devnull, "a") as fh:
        subprocess.run(cmd, check=True, stdout=fh, stderr=subprocess.STDOUT)


# -- part A --------------------------------------------------------------------

def part_a(root, threads, n=1000, seed=1):
    W = os.path.join(root, "real/enterobacterales")
    ref = os.path.join(root, "ref")
    E = os.path.join(root, "mamba/envs/hgt/bin")
    mm = os.path.join(root, "tools/mmseqs/bin/mmseqs")
    out = os.path.join(root, "bench/gpu/search")
    os.makedirs(out, exist_ok=True)
    logf = os.path.join(out, "tools.log")
    seqs = []
    for p in sorted(glob.glob(os.path.join(W, "proteomes/*.faa"))):
        seqs += list(_fasta_gz(p))
    rng = random.Random(seed)
    sub = rng.sample(seqs, n)
    qa, qall = os.path.join(out, "q1k.faa"), os.path.join(out, "qall.faa")
    for path, rows in ((qa, sub), (qall, seqs)):
        with open(path, "w") as fh:
            for name, s in rows:
                fh.write(">%s\n%s\n" % (name, s))
    res = {"n_queries": n, "n_all": len(seqs), "threads": threads}
    refdb, pad = os.path.join(ref, "refDB"), os.path.join(ref, "refDB_pad")
    if not os.path.exists(refdb + ".dbtype"):
        _, w, c = timed(sh, [mm, "createdb", os.path.join(ref, "reference.faa"), refdb], logf)
        res["mmseqs_createdb"] = {"wall": w, "cpu": c}
    if not os.path.exists(pad + ".dbtype"):
        _, w, c = timed(sh, [mm, "makepaddedseqdb", refdb, pad], logf)
        res["mmseqs_makepaddedseqdb"] = {"wall": w, "cpu": c}
    dmnd = os.path.join(ref, "ref.dmnd")
    if not os.path.exists(dmnd):
        _, w, c = timed(sh, [os.path.join(E, "diamond"), "makedb", "--in", os.path.join(ref, "reference.faa"),
                             "-d", dmnd, "--threads", str(threads)], logf)
        res["diamond_makedb"] = {"wall": w, "cpu": c}

    def mmseqs_search(qfa, target, tag, extra):
        tmp = os.path.join(out, "tmp_" + tag)
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        qdb, rdb = os.path.join(tmp, "q"), os.path.join(tmp, "r")
        sh([mm, "createdb", qfa, qdb], logf)
        sh([mm, "search", qdb, target, rdb, os.path.join(tmp, "t"), "-e", "1e-5", "--max-seqs", "2000",
            "--threads", str(threads)] + extra, logf)
        hits = os.path.join(out, tag + ".m8")
        sh([mm, "convertalis", qdb, target, rdb, hits, "--format-output",
            "query,target,pident,alnlen,evalue,bits,qcov,tcov", "--threads", str(threads)], logf)
        shutil.rmtree(tmp, ignore_errors=True)
        return hits

    def diamond(qfa, tag):
        hits = os.path.join(out, tag + ".m8")
        sh([os.path.join(E, "diamond"), "blastp", "-q", qfa, "-d", dmnd, "-o", hits, "--sensitive",
            "--max-target-seqs", "2000", "-e", "1e-5", "--threads", str(threads), "--outfmt", "6", "qseqid",
            "sseqid", "pident", "length", "evalue", "bitscore", "qcovhsp", "scovhsp"], logf)
        return hits

    runs = {}
    for tag, fn in (("mmseqs_gpu", lambda: mmseqs_search(qa, pad, "mmseqs_gpu", ["--gpu", "1"])),
                    ("mmseqs_cpu_s7.5", lambda: mmseqs_search(qa, refdb, "mmseqs_cpu_s7.5", ["-s", "7.5"])),
                    ("diamond_sensitive", lambda: diamond(qa, "diamond_sensitive"))):
        path, w, c = timed(fn)
        runs[tag] = path
        res[tag] = {"wall": w, "cpu": c}
        print("%-18s %7.1f s wall %8.1f s CPU" % (tag, w, c), flush=True)
    _, w, c = timed(mmseqs_search, qall, pad, "mmseqs_gpu_all", ["--gpu", "1"])
    res["mmseqs_gpu_all"] = {"wall": w, "cpu": c, "n": len(seqs)}
    print("%-18s %7.1f s wall %8.1f s CPU (%d queries)" % ("mmseqs_gpu_all", w, c, len(seqs)), flush=True)

    # agreement
    tax = taxonomy(ref)
    hits = {k: _load_hits(v) for k, v in runs.items()}
    names = [nm for nm, _ in sub]
    for k, h in hits.items():
        res[k]["queries_with_hits"] = sum(1 for q in names if h.get(q))
        res[k]["median_hits"] = sorted(len(h.get(q, [])) for q in names)[n // 2]

    def top_out(hl):
        for bits, ev, acc, t, pid, qcov in hl:
            if qcov >= 0.4 and 91347 not in tax.lineage(t):
                return tax.group(t)
        return None

    base = hits["diamond_sensitive"]
    for k in ("mmseqs_gpu", "mmseqs_cpu_s7.5"):
        h = hits[k]
        jac, same, both = [], 0, 0
        for q in names:
            a = set(x[2] for x in base.get(q, [])[:50])
            b = set(x[2] for x in h.get(q, [])[:50])
            if a or b:
                jac.append(len(a & b) / float(len(a | b)))
            ga, gb = top_out(base.get(q, [])), top_out(h.get(q, []))
            if ga and gb:
                both += 1
                same += ga == gb
        res[k]["top50_jaccard_vs_diamond"] = sum(jac) / len(jac) if jac else None
        res[k]["outside_group_agreement_vs_diamond"] = same / float(both) if both else None
        res[k]["outside_group_compared"] = both
    return res


# -- part B --------------------------------------------------------------------

def splits(root_node, leaves, min_support=None):
    """Non-trivial bipartitions of an unrooted tree, as frozensets of the side without `ref`."""
    ref = min(leaves)
    out = set()
    for nd in root_node.postorder():
        if not nd.children or nd.parent is None:
            continue
        s = frozenset(x.name for x in nd.leaves())
        if 1 < len(s) < len(leaves) - 1:
            if min_support is not None and (nd.support is None or nd.support < min_support):
                continue
            out.add(s if ref not in s else frozenset(leaves - s))
    return out


def compare_trees(t_iq, t_ft):
    a, b = parse_newick(t_iq), parse_newick(t_ft)
    la = set(x.name for x in a.leaves())
    lb = set(x.name for x in b.leaves())
    common = la & lb
    a, b = prune_to(a, common), prune_to(b, common)
    sa, sb = splits(a, common), splits(b, common)
    strong = splits(a, common, 95)
    rf = len(sa ^ sb) / float(len(sa) + len(sb)) if sa or sb else 0.0

    def compatible(s, others):
        for o in others:
            if not (s <= o or o <= s or not (s & o) or (s | o) == common):
                return False
        return True

    conflict = sum(1 for s in strong if not compatible(s, sb))
    return rf, len(strong), conflict


def _fasttree(job):
    aln, out, ft = job
    with open(out, "w") as fo:
        subprocess.run([ft, "-lg", "-gamma", "-quiet", aln], check=True, stdout=fo, stderr=subprocess.DEVNULL)
    return out


def _iqtree(job):
    aln, pre, iq = job
    subprocess.run([iq, "-s", aln, "-m", "LG+G4", "-B", "1000", "-T", "1", "--seed", "1", "--quiet", "-pre", pre],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pre + ".treefile"


def part_b(root, threads, n_timing=32):
    W = os.path.join(root, "real/enterobacterales")
    E = os.path.join(root, "mamba/envs/hgt/bin")
    out = os.path.join(root, "bench/gpu/trees")
    os.makedirs(os.path.join(out, "fasttree"), exist_ok=True)
    fams = sorted(os.path.basename(p)[:-9] for p in glob.glob(os.path.join(W, "gene_trees/*.treefile"))
                  if os.path.exists(os.path.join(W, "alignments", os.path.basename(p)[:-9] + ".faa")))
    nseq = {f: sum(1 for line in open(os.path.join(W, "alignments", f + ".faa")) if line.startswith(">"))
            for f in fams}
    jobs = [(os.path.join(W, "alignments", f + ".faa"), os.path.join(out, "fasttree", f + ".nwk"),
             os.path.join(E, "FastTree")) for f in fams]
    c0, t0 = cpu_children(), time.time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(_fasttree, jobs))
    res = {"families": len(fams), "fasttree_all": {"wall": time.time() - t0, "cpu": cpu_children() - c0}}
    rfs, strong, conflict = [], 0, 0
    for f in fams:
        rf, s, c = compare_trees(open(os.path.join(W, "gene_trees", f + ".treefile")).read(),
                                 open(os.path.join(out, "fasttree", f + ".nwk")).read())
        rfs.append(rf)
        strong += s
        conflict += c
    rfs.sort()
    res["rf_median"] = rfs[len(rfs) // 2]
    res["rf_mean"] = sum(rfs) / len(rfs)
    res["strong_splits"] = strong
    res["strong_splits_contradicted"] = conflict
    # CPU time per tree on a size-stratified subsample (both methods, same alignments)
    by_size = sorted(fams, key=lambda f: nseq[f])
    step = max(1, len(by_size) // n_timing)
    sample = by_size[::step][:n_timing]
    tdir = os.path.join(out, "iqtree_rerun")
    os.makedirs(tdir, exist_ok=True)
    c0, t0 = cpu_children(), time.time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(_iqtree, [(os.path.join(W, "alignments", f + ".faa"), os.path.join(tdir, f),
                               os.path.join(E, "iqtree3")) for f in sample]))
    res["iqtree_sample"] = {"n": len(sample), "wall": time.time() - t0, "cpu": cpu_children() - c0,
                            "median_seqs": nseq[sample[len(sample) // 2]]}
    c0, t0 = cpu_children(), time.time()
    with ThreadPoolExecutor(threads) as ex:
        list(ex.map(_fasttree, [(os.path.join(W, "alignments", f + ".faa"), os.path.join(tdir, f + ".ft.nwk"),
                                 os.path.join(E, "FastTree")) for f in sample]))
    res["fasttree_sample"] = {"n": len(sample), "wall": time.time() - t0, "cpu": cpu_children() - c0}
    res["cpu_ratio_iqtree_over_fasttree"] = res["iqtree_sample"]["cpu"] / max(1e-9, res["fasttree_sample"]["cpu"])
    res["sample"] = sample
    return res


# -- part C --------------------------------------------------------------------

def part_c(root, threads):
    W = os.path.join(root, "real/enterobacterales")
    out = os.path.join(root, "bench/gpu/trees")
    with open(os.path.join(W, "species_tree.nwk")) as fh:
        sp = SpeciesTree(parse_newick(fh.read()))
    model = DTLModel(sp, dated=sp.has_lengths())
    gene_sp = {}
    import csv
    with open(os.path.join(W, "genes.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            gene_sp[r["gene_id"]] = sp.leaf_index[r["species"]]
    fams = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(out, "fasttree/*.nwk")))
    agree = Counter()
    for f in fams:
        t_iq = open(os.path.join(W, "gene_trees", f + ".treefile")).read().strip()
        t_ft = open(os.path.join(out, "fasttree", f + ".nwk")).read().strip()
        leaves = [x.name for x in parse_newick(t_iq).leaves()]
        gsp = {g: gene_sp[g] for g in leaves if g in gene_sp}
        calls = []
        for txt, col in ((t_iq, 95.0), (t_ft, 0.95)):
            r = reconcile_family(f, [txt] * 11, gsp, model, 3, 0, None, col)
            calls.append(set(R for R, e in r["events"].items() if e["runs"] / float(max(1, r["runs"])) >= 0.5))
        a, b = calls
        agree["iq_calls"] += len(a)
        agree["ft_calls"] += len(b)
        agree["both"] += len(a & b)
        agree["families_identical"] += a == b
    agree["families"] = len(fams)
    res = dict(agree)
    res["recall_of_iqtree_calls"] = agree["both"] / float(max(1, agree["iq_calls"]))
    res["precision_vs_iqtree"] = agree["both"] / float(max(1, agree["ft_calls"]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B", "C"])
    ap.add_argument("--root", default="/hy-tmp/hgtscan")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    fn = {"A": part_a, "B": part_b, "C": part_c}[a.part]
    res = fn(a.root, a.threads)
    os.makedirs(os.path.join(a.root, "bench/gpu"), exist_ok=True)
    with open(os.path.join(a.root, "bench/gpu/part_%s.json" % a.part), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "sample"}, indent=2))
