"""Donor taxa for HGT candidates whose donor is outside the dataset (mammal? other bacteria? phage?).

A scan of a few related genomes can say "acquired from outside the sample" (gain events,
donor `unsampled`) but not from whom. This module compares the candidates with a broad,
taxonomically balanced reference and builds a phylogeny for each candidate event.

Steps (each skips work that is already done):
  reference  choose UniProt reference proteomes by NCBI taxonomy (Gammaproteobacteria: one per
             genus; other bacteria, archaea, mammals: one per family; other eukaryotes: one per
             order; viruses: one per genus), download them, write reference.faa (">accession|taxid")
  queries    candidate genes of a scan (P_HGT >= threshold; default: gain events) -> queries.faa
  search     MMseqs2 (GPU when available) or DIAMOND, queries vs reference
  classify   per gene: hits outside the recipient's own genus, Alien Index (closest hit inside the
             order vs outside it), donor group of the closest outside homologs, mammal hits
  trees      MAFFT + FastTree (default; IQ-TREE with --tree-method iqtree) of every candidate event
             with its closest outside homologs; donor group = nearest reference leaves to the
             recipient genes in the tree
"""
from __future__ import annotations

import csv
import glob
import gzip
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from .prepare import _read_aln, gap_filter, tool
from .reconcile import tree_distances
from .tree import SpeciesTree, parse_newick, unrooted_adjacency

UNIPROT_RP = "https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes"
TAXDUMP = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
EBI_TAXONOMY = "https://ftp.ebi.ac.uk/pub/databases/taxonomy/taxonomy.dat"   # NCBI taxonomy, EBI mirror

# donor groups, first match along the lineage wins (NCBI taxids)
GROUPS = [
    ("Mammalia", 40674), ("Vertebrata, other", 7742), ("Metazoa, other", 33208), ("Fungi", 4751),
    ("Viridiplantae", 33090), ("Eukaryota, other", 2759), ("Archaea", 2157),
    ("Enterobacterales, other genera", 91347), ("Gammaproteobacteria, other", 1236),
    ("Pseudomonadota, other", 1224), ("Bacillota", 1239), ("Actinomycetota", 201174),
    ("Bacteroidota", 976), ("Cyanobacteriota", 1117), ("Bacteria, other", 2),
    ("Phage (Caudoviricetes)", 2731619), ("Viruses, other", 10239),
]
MIN_QCOV = 0.4          # hits covering less of the query are domain-level matches: ignored


def log(msg):
    print("[donors] " + msg, file=sys.stderr, flush=True)


# -- NCBI taxonomy ------------------------------------------------------------

class Taxonomy:
    """NCBI taxonomy from taxdump (nodes/names/merged.dmp) or EBI's flat-file copy, cached as TSV."""

    def __init__(self, taxdir):
        self.parent, self.rank, self.name, self.merged = {}, {}, {}, {}
        self._lin = {}
        cache = os.path.join(taxdir, "taxonomy.tsv")
        merged = os.path.join(taxdir, "merged.dmp")
        if os.path.exists(merged):
            with open(merged) as fh:
                for line in fh:
                    f = line.split("\t|\t")
                    self.merged[int(f[0])] = int(f[1].split("\t")[0])
        if os.path.exists(cache):
            with open(cache) as fh:
                for line in fh:
                    t, p, r, n = line.rstrip("\n").split("\t")
                    t = int(t)
                    self.parent[t], self.rank[t], self.name[t] = int(p), r, n
            return
        if os.path.exists(os.path.join(taxdir, "nodes.dmp")):
            with open(os.path.join(taxdir, "nodes.dmp")) as fh:
                for line in fh:
                    f = line.split("\t|\t")
                    t = int(f[0])
                    self.parent[t] = int(f[1])
                    self.rank[t] = f[2]
            with open(os.path.join(taxdir, "names.dmp")) as fh:
                for line in fh:
                    f = line.rstrip("\t|\n").split("\t|\t")
                    if f[3] == "scientific name":
                        self.name[int(f[0])] = f[1]
        else:                              # EBI flat file: "ID : 2", "PARENT ID : ...", "//"
            t = None
            with open(os.path.join(taxdir, "taxonomy.dat")) as fh:
                for line in fh:
                    if line.startswith("//"):
                        t = None
                        continue
                    key, _, val = line.partition(":")
                    key, val = key.strip(), val.strip()
                    if key == "ID":
                        t = int(val)
                    elif t is None:
                        continue
                    elif key == "PARENT ID":
                        self.parent[t] = int(val)
                    elif key == "RANK":
                        self.rank[t] = val
                    elif key == "SCIENTIFIC NAME":
                        self.name[t] = val
        with open(cache + ".part", "w") as fo:
            for t, p in self.parent.items():
                fo.write("%d\t%d\t%s\t%s\n" % (t, p, self.rank.get(t, ""), self.name.get(t, "")))
        os.replace(cache + ".part", cache)

    def lineage(self, t):
        t = self.merged.get(t, t)
        if t in self._lin:
            return self._lin[t]
        out, x = [], t
        while x in self.parent and x not in out:
            out.append(x)
            if self.parent[x] in (x, 0):
                break
            x = self.parent[x]
        self._lin[t] = out
        return out

    def at_rank(self, t, rank):
        for x in self.lineage(t):
            if self.rank.get(x) == rank:
                return x
        return None

    def group(self, t):
        lin = set(self.lineage(t))
        for name, gid in GROUPS:
            if gid in lin:
                return name
        return "unclassified"

    def lca(self, taxids):
        lins = [self.lineage(t) for t in taxids]
        if not lins or not lins[0]:
            return None
        common = set(lins[0]).intersection(*map(set, lins[1:]))
        return next((x for x in lins[0] if x in common), None)


def _fetch(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    os.replace(tmp, dest)
    return dest


def _fetch_parallel(url, dest, parts=8):
    """Download a large file in byte ranges in parallel (slow single streams on some links)."""
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
        size = int(r.headers["Content-Length"])
    step = size // parts + 1
    ranges = [(i, min(size - 1, i + step - 1)) for i in range(0, size, step)]

    def one(k):
        a, b = ranges[k]
        part = "%s.%d" % (dest, k)
        if os.path.exists(part) and os.path.getsize(part) == b - a + 1:
            return
        req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b)})
        with urllib.request.urlopen(req, timeout=600) as r, open(part, "wb") as fh:
            shutil.copyfileobj(r, fh)

    with ThreadPoolExecutor(parts) as ex:
        list(ex.map(one, range(len(ranges))))
    with open(dest + ".part", "wb") as fo:
        for k in range(len(ranges)):
            with open("%s.%d" % (dest, k), "rb") as fi:
                shutil.copyfileobj(fi, fo)
    if os.path.getsize(dest + ".part") != size:
        raise IOError("incomplete download of %s" % url)
    os.replace(dest + ".part", dest)
    for k in range(len(ranges)):
        os.remove("%s.%d" % (dest, k))
    return dest


def taxonomy(ref):
    """Current NCBI taxdump (honours https_proxy); EBI's older flat-file copy as a fallback."""
    taxdir = os.path.join(ref, "taxdump")
    os.makedirs(taxdir, exist_ok=True)
    if not any(os.path.exists(os.path.join(taxdir, f)) for f in ("taxonomy.tsv", "nodes.dmp", "taxonomy.dat")):
        try:
            tgz = _fetch_parallel(TAXDUMP, os.path.join(taxdir, "taxdump.tar.gz"), parts=24)
            with tarfile.open(tgz) as tf:
                for m in ("nodes.dmp", "names.dmp", "merged.dmp"):
                    tf.extract(m, taxdir)
            os.remove(tgz)
        except Exception as e:             # noqa: BLE001
            log("NCBI taxdump not reachable (%s); using EBI's copy, which misses recently named taxa" % e)
            _fetch_parallel(EBI_TAXONOMY, os.path.join(taxdir, "taxonomy.dat"))
    return Taxonomy(taxdir)


# -- reference ----------------------------------------------------------------

def _readme_rows(path):
    rows, on = [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("Proteome_ID\t"):
                on = True
                continue
            if on:
                f = line.rstrip("\n").split("\t")
                if len(f) < 8 or not f[0].startswith("UP"):
                    if rows:
                        break
                    continue
                rows.append({"id": f[0], "taxid": int(f[1]), "regnum": f[3].lower(),
                             "n": int(f[4] or 0), "species": f[7]})
    return rows


def select_proteomes(rows, tax):
    """Taxonomically balanced subset: dense near the recipients, sparse far away.

    Gammaproteobacteria (closest relatives of Enterobacterales): one proteome per genus; other
    bacteria and archaea: one per family; mammals: one per family; other eukaryotes: one per
    order; viruses: one per genus. Within a slot the proteome with most proteins wins.
    """
    best = {}
    unknown = 0
    for r in rows:
        t = r["taxid"]
        lin = set(tax.lineage(t))
        if len(lin) < 2:                   # taxid unknown to this taxonomy release: cannot be placed
            unknown += 1
            continue
        if r["regnum"] == "bacteria":
            level = "genus" if 1236 in lin else "family"
        elif r["regnum"] == "archaea":
            level = "family"
        elif r["regnum"] == "viruses":
            level = "genus"
        elif 40674 in lin:
            level = "family"
        else:
            level = "order"
        slot = (tax.at_rank(t, level) or tax.at_rank(t, "family") or tax.at_rank(t, "genus") or t)
        key = (r["regnum"], level, slot)
        if key not in best or r["n"] > best[key]["n"]:
            best[key] = r
    if unknown:
        log("%d proteomes skipped: taxid not in the taxonomy" % unknown)
    return sorted(best.values(), key=lambda r: (r["regnum"], r["id"]))


def reference(ref, threads=16, max_proteins=25_000_000):
    os.makedirs(os.path.join(ref, "proteomes"), exist_ok=True)
    tax = taxonomy(ref)
    readme = os.path.join(ref, "README")
    if not os.path.exists(readme):
        _fetch(UNIPROT_RP + "/README", readme)
    rows = _readme_rows(readme)
    sel = select_proteomes(rows, tax)
    n = sum(r["n"] for r in sel)
    by = Counter(r["regnum"] for r in sel)
    mam = sum(1 for r in sel if 40674 in tax.lineage(r["taxid"]))
    log("UniProt reference proteomes: %d listed, %d selected (%s; mammals %d), %.1f M proteins"
        % (len(rows), len(sel), ", ".join("%s %d" % kv for kv in sorted(by.items())), mam, n / 1e6))
    if n > max_proteins:
        raise SystemExit("selection has %.1f M proteins > --max-proteins" % (n / 1e6))

    def one(r):
        fn = "%s_%d.fasta.gz" % (r["id"], r["taxid"])
        dest = os.path.join(ref, "proteomes", fn)
        if os.path.exists(dest):
            return "cached"
        url = "%s/%s/%s/%s" % (UNIPROT_RP, r["regnum"].capitalize(), r["id"], fn)
        for k in range(6):
            try:
                _fetch(url, dest)
                return "ok"
            except Exception as e:          # noqa: BLE001 - network hiccups / throttling: back off
                err = e
                time.sleep(3 * (k + 1))
        return "failed: %s" % err

    with ThreadPoolExecutor(threads) as ex:
        status = Counter(s.split(":")[0] for s in ex.map(one, sel))
    log("downloads: %s" % dict(status))
    if status.get("failed"):
        log("some proteomes failed to download; rerun the reference step to retry them")
    with open(os.path.join(ref, "proteomes.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["proteome", "taxid", "regnum", "n_proteins", "group", "species"])
        for r in sel:
            w.writerow([r["id"], r["taxid"], r["regnum"], r["n"], tax.group(r["taxid"]), r["species"]])
    out = os.path.join(ref, "reference.faa")
    if not os.path.exists(out):
        tmp, k = out + ".part", 0
        with open(tmp, "w") as fo:
            for r in sel:
                p = os.path.join(ref, "proteomes", "%s_%d.fasta.gz" % (r["id"], r["taxid"]))
                if not os.path.exists(p):
                    continue
                for name, seq in _fasta_gz(p):
                    if len(seq) < 30:
                        continue
                    acc = name.split()[0].split("|")
                    acc = acc[1] if len(acc) >= 2 else acc[0]
                    fo.write(">%s|%d\n%s\n" % (acc, r["taxid"], seq))
                    k += 1
        os.replace(tmp, out)
        log("reference.faa: %d proteins" % k)


def _fasta_gz(path):
    opener = gzip.open if path.endswith(".gz") else open
    name, seq = None, []
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].strip(), []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


# -- queries ------------------------------------------------------------------

def _genome_taxids(workdir, index):
    genomes = list(csv.DictReader(open(os.path.join(workdir, "genomes.tsv")), delimiter="\t"))
    tid = {}
    with open(index) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            tid[row["species"]] = int(row["taxonomy_id"])
    return {g["label"]: tid[g["ensembl_name"]] for g in genomes if g["ensembl_name"] in tid}


def queries(workdir, scan, out, index, tax, threshold=0.6, sources=("gain",)):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(scan, "species_tree.labeled.nwk")) as fh:
        sp = SpeciesTree(parse_newick(fh.read()))
    gtax = _genome_taxids(workdir, index)
    seqs = {}
    for p in glob.glob(os.path.join(workdir, "proteomes", "*.faa")):
        for name, s in _fasta_gz(p):
            seqs[name] = s
    rows = []
    with open(os.path.join(scan, "hgt_genes.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if float(r["P_HGT"]) >= threshold and r["source"] in sources and r["gene_id"] in seqs:
                rows.append(r)
    self_cache = {}
    with open(os.path.join(out, "queries.faa"), "w") as fa, open(os.path.join(out, "queries.tsv"), "w") as ft:
        ft.write("gene_id\tspecies\tfamily\tevent_id\tsource\tP_HGT\trecipient\tself_taxid\tself_name\n")
        for r in rows:
            node = r["recipient"].split("{")[0]
            if node not in self_cache:
                leaves = sp.leafsets[sp.index[node]] if node in sp.index else [r["species"]]
                t = tax.lca([gtax[x] for x in leaves if x in gtax])
                # at least genus: one reference proteome per genus cannot resolve finer
                lin = tax.lineage(t) if t else []
                g = next((x for x in lin if tax.rank.get(x) in ("genus", "family", "order", "class", "phylum")), t)
                self_cache[node] = g
            st = self_cache[node]
            fa.write(">%s\n%s\n" % (r["gene_id"], seqs[r["gene_id"]]))
            ft.write("\t".join([r["gene_id"], r["species"], r["family"], r["event_id"], r["source"], r["P_HGT"],
                                node, str(st), tax.name.get(st, "")]) + "\n")
    log("queries: %d genes in %d events" % (len(rows), len(set(r["event_id"] for r in rows))))


# -- search -------------------------------------------------------------------

def _gpu():
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).returncode == 0
    except OSError:
        return False


def search(ref, out, threads, engine="auto", mmseqs=None, tools_dir=None):
    hits = os.path.join(out, "hits.m8")
    if os.path.exists(hits):
        return hits
    q = os.path.join(out, "queries.faa")
    mm = mmseqs or shutil.which("mmseqs")
    if engine == "auto":
        engine = ("mmseqs-gpu" if _gpu() else "mmseqs") if mm else "diamond"
    tmp = tempfile.mkdtemp(prefix="donors_", dir=out)
    try:
        if engine.startswith("mmseqs"):
            gpu = engine == "mmseqs-gpu"
            db = os.path.join(ref, "refDB")
            target = db + ("_pad" if gpu else "")
            if not os.path.exists(db + ".dbtype"):
                subprocess.run([mm, "createdb", os.path.join(ref, "reference.faa"), db], check=True)
            if gpu and not os.path.exists(target + ".dbtype"):
                subprocess.run([mm, "makepaddedseqdb", db, target], check=True)
            qdb, res = os.path.join(tmp, "q"), os.path.join(tmp, "res")
            subprocess.run([mm, "createdb", q, qdb], check=True)
            cmd = [mm, "search", qdb, target, res, os.path.join(tmp, "t"), "-e", "1e-5", "--max-seqs", "2000",
                   "--threads", str(threads)]
            if gpu:
                cmd += ["--gpu", "1"]
            else:
                cmd += ["-s", "7.5"]
            subprocess.run(cmd, check=True)
            subprocess.run([mm, "convertalis", qdb, target, res, hits + ".part", "--format-output",
                            "query,target,pident,alnlen,evalue,bits,qcov,tcov", "--threads", str(threads)],
                           check=True)
        else:
            dmd = tool("diamond", tools_dir)
            db = os.path.join(ref, "ref.dmnd")
            if not os.path.exists(db):
                subprocess.run([dmd, "makedb", "--in", os.path.join(ref, "reference.faa"), "-d", db,
                                "--threads", str(threads)], check=True)
            subprocess.run([dmd, "blastp", "-q", q, "-d", db, "-o", hits + ".part", "--sensitive",
                            "--max-target-seqs", "2000", "-e", "1e-5", "--threads", str(threads), "--outfmt",
                            "6", "qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qcovhsp",
                            "scovhsp"], check=True)
        os.replace(hits + ".part", hits)
        log("search (%s) done -> %s" % (engine, hits))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return hits


# -- classify -----------------------------------------------------------------

def _load_hits(path):
    by_q = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            acc, _, t = f[1].rpartition("|")
            qcov = float(f[6])
            qcov = qcov / 100.0 if qcov > 1.0 else qcov          # DIAMOND reports percent
            by_q[f[0]].append((float(f[5]), float(f[4]), acc, int(t), float(f[2]), qcov))
    for v in by_q.values():
        v.sort(key=lambda x: -x[0])
    return by_q


def classify(out, tax, ingroup=91347, top=10):
    qrows = list(csv.DictReader(open(os.path.join(out, "queries.tsv")), delimiter="\t"))
    hits = _load_hits(os.path.join(out, "hits.m8"))
    res = []
    for q in qrows:
        st = int(q["self_taxid"])
        best_in = best_out = None
        outside, mammal = [], None
        for bits, ev, acc, t, pid, qcov in hits.get(q["gene_id"], []):
            if qcov < MIN_QCOV:
                continue
            lin = set(tax.lineage(t))
            if st in lin:
                continue                               # the recipient's own genus (or clade)
            outside.append((bits, ev, acc, t, pid))
            if ingroup in lin:
                best_in = best_in or (bits, ev, t, pid)
            else:
                best_out = best_out or (bits, ev, t, pid)
            if mammal is None and 40674 in lin:
                mammal = (bits, ev, t, pid, len(outside))
        e_in = best_in[1] if best_in else 1.0
        e_out = best_out[1] if best_out else 1.0
        ai = math.log(e_in + 1e-200) - math.log(e_out + 1e-200)
        w = Counter()
        for bits, _, _, t, _ in outside[:top]:
            w[tax.group(t)] += bits
        tot = sum(w.values())
        grp, share = (w.most_common(1)[0][0], w.most_common(1)[0][1] / tot) if tot else ("no outside homolog", 0.0)
        b = outside[0] if outside else None
        res.append(dict(q, n_outside=len(outside), AI="%.1f" % ai,
                        best_in_bits="%.0f" % best_in[0] if best_in else "",
                        best_out_bits="%.0f" % best_out[0] if best_out else "",
                        top_group=grp, top_group_share="%.2f" % share,
                        best_hit=b[2] if b else "", best_hit_taxon=tax.name.get(b[3], "") if b else "",
                        best_hit_group=tax.group(b[3]) if b else "", best_hit_pident="%.1f" % b[4] if b else "",
                        mammal_hit=("%s (%s, %.0f bits, %.0f%% id, rank %d)"
                                    % (tax.name.get(mammal[2], ""), mammal[2], mammal[0], mammal[3], mammal[4]))
                        if mammal else ""))
    cols = list(qrows[0].keys()) + ["n_outside", "AI", "best_in_bits", "best_out_bits", "top_group",
                                     "top_group_share", "best_hit", "best_hit_taxon", "best_hit_group",
                                     "best_hit_pident", "mammal_hit"] if qrows else []
    with open(os.path.join(out, "donors_by_gene.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(res)
    c = Counter(r["top_group"] for r in res)
    log("genes by closest outside group: " + ", ".join("%s %d" % kv for kv in c.most_common()))
    log("genes with any mammal hit (qcov >= %.1f): %d" % (MIN_QCOV, sum(1 for r in res if r["mammal_hit"])))
    return res


# -- trees --------------------------------------------------------------------

def _tree_job(job):
    ev, genes, refs, work, mafft, builder, seed = job
    method, prog = builder
    final = os.path.join(work, "..", ev + ".treefile")
    if os.path.exists(final):
        return ev, "cached"
    os.makedirs(work, exist_ok=True)
    try:
        fa = os.path.join(work, "in.faa")
        with open(fa, "w") as fh:
            for n, s in list(genes.items()) + list(refs.items()):
                fh.write(">%s\n%s\n" % (n, s))
        aln_path = os.path.join(work, "aln.faa")
        with open(aln_path, "w") as fh:
            subprocess.run([mafft, "--auto", "--thread", "1", "--quiet", fa], stdout=fh, check=True)
        aln = gap_filter(_read_aln(aln_path))
        if len(next(iter(aln.values()))) < 30:
            return ev, "short alignment"
        filt = os.path.join(work, "filtered.faa")
        with open(filt, "w") as fh:
            for n, s in aln.items():
                fh.write(">%s\n%s\n" % (n, s))
        pre = os.path.join(work, ev)
        if method == "iqtree":
            subprocess.run([prog, "-s", filt, "-m", "LG+G4", "-B", "1000", "-T", "1", "--seed", str(seed),
                            "--quiet", "-pre", pre], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:                              # FastTree: approximate ML, SH-like local supports
            with open(pre + ".treefile", "w") as fo:
                subprocess.run([prog, "-lg", "-gamma", "-quiet", filt], check=True, stdout=fo,
                               stderr=subprocess.DEVNULL)
        shutil.copy(pre + ".treefile", final + ".part")
        os.replace(final + ".part", final)
        return ev, "ok"
    except subprocess.CalledProcessError as e:
        return ev, "failed: %s" % e
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _nearest(tree_txt, queries, tax, k=5):
    """Donor group from the tree: reference leaves closest (patristic) to the recipient genes."""
    adj, names = unrooted_adjacency(parse_newick(tree_txt))
    q = [i for i, n in enumerate(names) if n in queries]
    r = [i for i, n in enumerate(names) if n and n not in queries]
    if not q or not r:
        return None
    acc = Counter()
    for i in q:
        d = tree_distances(adj, i)
        for j in r:
            acc[j] += d[j] / len(q)
    near = sorted(r, key=lambda j: acc[j])[:k]
    groups = Counter(tax.group(int(names[j].rsplit("|", 1)[1])) for j in near)
    g, n = groups.most_common(1)[0]
    taxa = [tax.name.get(int(names[j].rsplit("|", 1)[1]), "") for j in near]
    # do the recipient genes form one clan (a split separating them from all reference leaves)?
    clan = False
    qs = frozenset(q)
    seen = {}

    def below(u, frm):
        key = (frm, u)
        if key not in seen:
            s = {u} if names[u] else set()
            for v, _ in adj[u]:
                if v != frm:
                    s |= below(v, u)
            seen[key] = frozenset(s)
        return seen[key]

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10000))
    for u in range(len(adj)):
        for v, _ in adj[u]:
            leaves = below(v, u)
            if leaves and leaves <= qs and len(leaves) == len(qs):
                clan = True
                break
        if clan:
            break
    return g, n / float(len(near)), clan, "; ".join(taxa)


def trees(out, ref, tax, threads, tools_dir=None, max_refs=60, per_genus=3, min_refs=4, events=None,
          method="fasttree"):
    rows = list(csv.DictReader(open(os.path.join(out, "donors_by_gene.tsv")), delimiter="\t"))
    hits = _load_hits(os.path.join(out, "hits.m8"))
    by_ev = defaultdict(list)
    for r in rows:
        by_ev[r["event_id"]].append(r)
    if events:
        by_ev = {e: v for e, v in by_ev.items() if e in events}
    plan, need = {}, set()
    for ev, rs in by_ev.items():
        st = int(rs[0]["self_taxid"])
        pool = []
        for r in rs[:10]:
            pool += [h for h in hits.get(r["gene_id"], []) if h[5] >= MIN_QCOV and st not in tax.lineage(h[3])]
        pool.sort(key=lambda h: -h[0])
        chosen, per_g, seen = [], Counter(), set()
        for bits, e, acc, t, pid, qcov in pool:
            if acc in seen:
                continue
            gkey = tax.at_rank(t, "genus") or t
            if per_g[gkey] >= per_genus:
                continue
            seen.add(acc)
            per_g[gkey] += 1
            chosen.append("%s|%d" % (acc, t))
            if len(chosen) >= max_refs:
                break
        if len(chosen) >= min_refs:
            plan[ev] = ([r["gene_id"] for r in rs[:10]], chosen)
            need.update(chosen)
    log("trees: %d events with >= %d outside homologs" % (len(plan), min_refs))
    seqs = {}
    for name, s in _fasta_gz(os.path.join(ref, "reference.faa")):
        if name in need:
            seqs[name] = s
    qseq = {n: s for n, s in _fasta_gz(os.path.join(out, "queries.faa"))}
    tdir = os.path.join(out, "trees")
    os.makedirs(tdir, exist_ok=True)
    mafft = tool("mafft", tools_dir)
    builder = (("iqtree", tool(("iqtree2", "iqtree3", "iqtree"), tools_dir)) if method == "iqtree"
               else ("fasttree", tool(("FastTree", "fasttree"), tools_dir)))
    jobs = [(ev, {g: qseq[g] for g in gs}, {n: seqs[n] for n in refs if n in seqs},
             os.path.join(tdir, "work_" + ev), mafft, builder, 1) for ev, (gs, refs) in plan.items()]
    status = Counter()
    with ThreadPoolExecutor(threads) as ex:
        for i, (ev, st) in enumerate(ex.map(_tree_job, jobs), 1):
            status[st.split(":")[0]] += 1
            if i % max(1, len(jobs) // 10) == 0:
                log("trees %d/%d %s" % (i, len(jobs), dict(status)))
    with open(os.path.join(out, "donors_by_event.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["event_id", "recipient", "self_name", "n_genes", "P_HGT", "blast_top_group", "blast_share",
                    "mammal_hits", "tree_group", "tree_share", "recipient_clan", "nearest_taxa"])
        for ev, rs in sorted(by_ev.items(), key=lambda x: -float(x[1][0]["P_HGT"])):
            c = Counter(r["top_group"] for r in rs)
            g, n = c.most_common(1)[0]
            tr = os.path.join(tdir, ev + ".treefile")
            near = _nearest(open(tr).read(), set(r["gene_id"] for r in rs), tax) if os.path.exists(tr) else None
            w.writerow([ev, rs[0]["recipient"], rs[0]["self_name"], len(rs), rs[0]["P_HGT"], g,
                        "%.2f" % (n / float(len(rs))), sum(1 for r in rs if r["mammal_hit"]),
                        near[0] if near else "", "%.2f" % near[1] if near else "", int(near[2]) if near else "",
                        near[3] if near else ""])
    log("trees: %s -> donors_by_event.tsv" % dict(status))


def run_donors(a):
    steps = a.steps.split(",")
    tax = taxonomy(a.ref)
    if a.tools and os.path.isdir(a.tools):
        os.environ["PATH"] = a.tools + os.pathsep + os.environ.get("PATH", "")
    if "reference" in steps:
        reference(a.ref, a.download_threads, a.max_proteins)
    if "queries" in steps:
        queries(a.workdir, a.scan, a.out, a.index, tax, a.threshold, tuple(a.sources.split(",")))
    if "search" in steps:
        search(a.ref, a.out, a.threads, a.engine, a.mmseqs, a.tools)
    if "classify" in steps:
        classify(a.out, tax, a.ingroup)
    if "trees" in steps:
        trees(a.out, a.ref, tax, a.threads, a.tools, method=a.tree_method)
