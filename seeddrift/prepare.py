"""Build scan inputs from real genomes (Ensembl Genomes proteomes + annotation).

Steps (run in order; each skips work that is already done):
  fetch        download *.pep.all.fa.gz, *.cds.all.fa.gz and *.gff3.gz for every genome in genomes.tsv
  proteomes    one protein per gene -> proteomes/<label>.faa, gene coordinates,
               mobile elements from annotation (transposase / integrase / phage / plasmid / tRNA)
  composition  GC and GC3 of each chosen gene's CDS -> gene_composition.tsv
  orthogroups  OrthoFinder (-og) -> Orthogroups.tsv, then genes.tsv
  genetrees    MAFFT + gap-column filter + IQ-TREE (LG+G4, UFBoot) per family -> gene_trees/
  speciestree  concatenated single-copy families -> IQ-TREE -> outgroup root -> MPL ultrametric

External tools (OrthoFinder, MAFFT, IQ-TREE) are looked up in --tools (a bin directory),
then on PATH.
"""
from __future__ import annotations

import csv
import glob
import gzip
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from .tree import Node, parse_newick, read_trees, to_newick, unrooted_adjacency

ENSEMBL = "https://ftp.ensemblgenomes.ebi.ac.uk/pub/%s/current"
ME_RULES = [   # (type, regex on "symbol description"), first match wins
    ("integrase", re.compile(r"integrase(?!.*host factor)", re.I)),
    ("transposase", re.compile(r"transposase|transposon|insertion (sequence|element)|\bIS\d+|\bIS[A-Z][a-z]\d", re.I)),
    ("phage", re.compile(r"phage|terminase|portal protein|major capsid|tail fib(er|re)|tail tape|baseplate", re.I)),
    ("recombinase", re.compile(r"recombinase|resolvase|invertase", re.I)),
]
HOUSEKEEPING_RECOMBINASES = re.compile(r"\b(xerC|xerD|recA|recB|recC|recD|recF|recG|recN|recO|recR|ruvC)\b", re.I)


def log(msg):
    print("[prepare] " + msg, file=sys.stderr, flush=True)


def safe(x):
    return re.sub(r"[^A-Za-z0-9_-]", "_", x)


def tool(name, tools_dir):
    cands = [name] if isinstance(name, str) else list(name)
    for c in cands:
        if tools_dir and os.path.exists(os.path.join(tools_dir, c)):
            return os.path.join(tools_dir, c)
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("tool not found: %s (pass --tools <env>/bin)" % "/".join(cands))


def read_genomes(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


# -- fetch --------------------------------------------------------------------

def _listing(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")
    return re.findall(r'href="([^"?/][^"]*)"', html)


def _download(url, dest):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    os.replace(tmp, dest)


def fetch(genomes, index_path, workdir, division="bacteria"):
    coll = {}
    with open(index_path) as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            coll[row["species"]] = row["core_db"].split("_core_")[0]
    raw = os.path.join(workdir, "raw")
    os.makedirs(raw, exist_ok=True)
    base = ENSEMBL % division
    for g in genomes:
        name, label = g["ensembl_name"], g["label"]
        if name not in coll:
            raise SystemExit("%s not in Ensembl index" % name)
        for kind, sub, pat in (("pep", "fasta/%s/%s/pep/", r"\.pep\.all\.fa\.gz$"),
                               ("cds", "fasta/%s/%s/cds/", r"\.cds\.all\.fa\.gz$"),
                               ("gff3", "gff3/%s/%s/", r"\.gff3\.gz$")):
            dest = os.path.join(raw, "%s.%s.gz" % (label, kind))
            if os.path.exists(dest):
                continue
            url = base + "/" + sub % (coll[name], name)
            files = [f for f in _listing(url) if re.search(pat, f) and ".abinitio." not in f]
            if not files:
                raise SystemExit("no %s file under %s" % (kind, url))
            _download(url + files[0], dest)
            log("downloaded %s %s (%.1f MB)" % (label, kind, os.path.getsize(dest) / 1e6))


# -- proteomes ----------------------------------------------------------------

def _parse_header(h):
    toks = h.split()
    rec = {"pid": toks[0], "symbol": "", "desc": ""}
    for i, t in enumerate(toks[1:], 1):
        if t.count(":") >= 5 and "contig" not in rec:
            parts = t.split(":")
            rec.update(coord=parts[0], contig=":".join(parts[2:-3]), start=int(parts[-3]),
                       end=int(parts[-2]), strand="+" if parts[-1] == "1" else "-")
        elif t.startswith("gene:"):
            rec["gene"] = t[5:]
        elif t.startswith("transcript:"):
            rec["transcript"] = t[11:]
        elif t.startswith("gene_symbol:"):
            rec["symbol"] = t[12:]
        elif t.startswith("description:"):
            rec["desc"] = " ".join([t[12:]] + toks[i + 1:])
            break
    return rec


def _read_fasta_gz(path):
    name, seq = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(seq)
                name, seq = line[1:].strip(), []
            else:
                seq.append(line.strip())
    if name is not None:
        yield name, "".join(seq)


def me_type(symbol, desc):
    text = "%s %s" % (symbol, desc)
    if HOUSEKEEPING_RECOMBINASES.search(symbol or ""):
        return None
    for t, rx in ME_RULES:
        if rx.search(text):
            return t
    return None


def proteomes(genomes, workdir, min_len=30):
    raw, pdir = os.path.join(workdir, "raw"), os.path.join(workdir, "proteomes")
    os.makedirs(pdir, exist_ok=True)
    coords, mes = [], []
    for g in genomes:
        label = g["label"]
        best = {}
        for header, seq in _read_fasta_gz(os.path.join(raw, label + ".pep.gz")):
            r = _parse_header(header)
            seq = seq.rstrip("*").replace("*", "X")
            if "contig" not in r or len(seq) < min_len:
                continue
            key = r.get("gene", r["pid"])
            if key not in best or len(seq) > len(best[key][1]):
                best[key] = (r, seq)
        with open(os.path.join(pdir, label + ".faa"), "w") as fh:
            for r, seq in best.values():
                gid = "%s__%s" % (label, safe(r["pid"]))
                fh.write(">%s\n%s\n" % (gid, seq))
                coords.append([gid, label, r["contig"], r["start"], r["end"], r["strand"], r["symbol"], r["desc"]])
                t = me_type(r["symbol"], r["desc"])
                if t:
                    mes.append([label, r["contig"], r["start"], r["end"], t])
        regions, n_trna = {}, 0
        with gzip.open(os.path.join(raw, label + ".gff3.gz"), "rt") as fh:
            for line in fh:
                if line.startswith("##sequence-region"):
                    f = line.split()
                    regions[f[1]] = int(f[3])
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) == 9 and f[2] == "tRNA":     # absent from Prokka-based Ensembl builds
                    mes.append([label, f[0], int(f[3]), int(f[4]), "tRNA"])
                    n_trna += 1
        # plasmids: in a complete genome (few replicons), pXXX-named or small replicons
        plasmids = []
        if regions and len(regions) <= 10:
            big = max(regions.values())
            plasmids = [n for n, L in regions.items()
                        if L < big and (L < 0.25 * big or re.match(r"(plasmid|p[A-Z0-9])", n, re.I))]
        for p in plasmids:
            mes.append([label, p, 1, regions[p], "plasmid"])
        log("%-16s %5d proteins, %d replicon(s)/contig(s), plasmids: %s, tRNA: %d"
            % (label, len(best), len(regions), ",".join(plasmids) or "-", n_trna))
    with open(os.path.join(workdir, "gene_coords.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "species", "contig", "start", "end", "strand", "symbol", "description"])
        w.writerows(coords)
    with open(os.path.join(workdir, "mobile_elements.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "contig", "start", "end", "type"])
        w.writerows(mes)
    with open(os.path.join(workdir, "species.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "domain"])
        for g in genomes:
            w.writerow([g["label"], g.get("domain", "prok") or "prok"])
    log("mobile-element records: %d" % len(mes))


# -- orthogroups --------------------------------------------------------------

def orthogroups(genomes, workdir, tools_dir, threads):
    out = os.path.join(workdir, "Orthogroups.tsv")
    if not os.path.exists(out):
        # OrthoFinder builds shell commands without quoting: run it in a space-free temp dir
        tmp = tempfile.mkdtemp(prefix="hgt_orthofinder_")
        try:
            src = os.path.join(tmp, "proteomes")
            shutil.copytree(os.path.join(workdir, "proteomes"), src)
            ofdir = os.path.join(tmp, "out")
            cmd = [tool("orthofinder", tools_dir), "-f", src, "-og", "-t", str(threads),
                   "-a", str(threads), "-S", "diamond", "-o", ofdir, "-n", "run"]
            log("running OrthoFinder: " + " ".join(cmd))
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            # OrthoFinder 2: Orthogroups.tsv; OrthoFinder 3 with -og: MCL_Orthogroups.tsv
            res = (glob.glob(os.path.join(ofdir, "Results_run*", "Orthogroups", "Orthogroups.tsv")) or
                   glob.glob(os.path.join(ofdir, "Results_run*", "Orthogroups", "*Orthogroups.tsv")))
            if not res:
                raise SystemExit("OrthoFinder finished but no Orthogroups.tsv under %s" % ofdir)
            shutil.copy(res[0], out)
            un = glob.glob(os.path.join(os.path.dirname(res[0]), "*Orthogroups_UnassignedGenes.tsv"))
            if un:
                shutil.copy(un[0], os.path.join(workdir, "Orthogroups_UnassignedGenes.tsv"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    fam = {}
    for path in (out, os.path.join(workdir, "Orthogroups_UnassignedGenes.tsv")):
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            rd = csv.reader(fh, delimiter="\t")
            next(rd)
            for row in rd:
                for cell in row[1:]:
                    for g in cell.split(", "):
                        if g:
                            fam[g] = row[0]
    with open(os.path.join(workdir, "gene_coords.tsv")) as fh, \
            open(os.path.join(workdir, "genes.tsv"), "w", newline="") as oh:
        w = csv.writer(oh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "species", "family", "contig", "start", "end", "strand"])
        for r in csv.DictReader(fh, delimiter="\t"):
            w.writerow([r["gene_id"], r["species"], fam.get(r["gene_id"], "U_" + r["gene_id"]),
                        r["contig"], r["start"], r["end"], r["strand"]])
    log("genes.tsv written (%d genes in orthogroups)" % len(fam))


def _families(workdir):
    with open(os.path.join(workdir, "Orthogroups.tsv")) as fh:
        rd = csv.reader(fh, delimiter="\t")
        species = next(rd)[1:]
        for row in rd:
            members = {s: [g for g in cell.split(", ") if g] for s, cell in zip(species, row[1:])}
            yield row[0], members


# -- gene trees ---------------------------------------------------------------

def _seqs(workdir):
    seqs = {}
    for path in glob.glob(os.path.join(workdir, "proteomes", "*.faa")):
        name = None
        with open(path) as fh:
            for line in fh:
                if line.startswith(">"):
                    name = line[1:].strip()
                    seqs[name] = []
                else:
                    seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def _read_aln(path):
    out, name = {}, None
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].strip()
                out[name] = []
            elif name:
                out[name].append(line.strip())
    return {k: "".join(v).upper() for k, v in out.items()}


def gap_filter(aln, max_gap=0.5):
    names = list(aln)
    L = len(aln[names[0]])
    keep = [i for i in range(L) if sum(aln[n][i] == "-" for n in names) <= max_gap * len(names)]
    return {n: "".join(aln[n][i] for i in keep) for n in names}


def _one_tree(job):
    og, genes, seqs, outdir, tmpdir, mafft, builder, n_keep, seed = job
    method, prog = builder
    final = os.path.join(outdir, og + ".treefile")
    if os.path.exists(final):
        return og, "cached"
    work = os.path.join(tmpdir, og)
    os.makedirs(work, exist_ok=True)
    try:
        fa = os.path.join(work, "in.faa")
        with open(fa, "w") as fh:
            for g in genes:
                fh.write(">%s\n%s\n" % (g, seqs[g]))
        aln_path = os.path.join(work, "aln.faa")
        with open(aln_path, "w") as fh:
            subprocess.run([mafft, "--auto", "--thread", "1", "--quiet", fa], stdout=fh, check=True)
        aln = gap_filter(_read_aln(aln_path))
        if len(next(iter(aln.values()))) < 30:
            return og, "short alignment"
        filt = os.path.join(work, "filtered.faa")
        with open(filt, "w") as fh:
            for n, s in aln.items():
                fh.write(">%s\n%s\n" % (n, s))
        shutil.copy(filt, os.path.join(outdir, "..", "alignments", og + ".faa"))
        pre = os.path.join(work, og)
        if method == "fasttree":
            # approximate ML; SH-like local supports scaled to 0-100 so that --collapse 95
            # means the same threshold as for UFBoot
            res = subprocess.run([prog, "-lg", "-gamma", "-quiet", filt], check=True, capture_output=True,
                                 text=True)
            t = parse_newick(res.stdout.strip())
            for nd in t.preorder():
                if nd.support is not None:
                    nd.support = round(100.0 * nd.support, 1)
            with open(final + ".part", "w") as fh:
                fh.write(to_newick(t, support=True) + "\n")
            os.replace(final + ".part", final)
            return og, "ok"
        subprocess.run([prog, "-s", filt, "-m", "LG+G4", "-B", "1000", "--wbtl", "-T", "1",
                        "--seed", str(seed), "--quiet", "-pre", pre], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        boots = read_trees(pre + ".ufboot")
        rng = random.Random(seed)
        boots = rng.sample(boots, min(n_keep, len(boots)))
        with open(os.path.join(outdir, og + ".ufboot"), "w") as fh:
            fh.write("\n".join(boots) + "\n")
        shutil.copy(pre + ".treefile", final + ".part")
        os.replace(final + ".part", final)
        return og, "ok"
    except subprocess.CalledProcessError as e:
        return og, "failed: %s" % e
    finally:
        shutil.rmtree(work, ignore_errors=True)


def genetrees(workdir, tools_dir, threads, min_genes=4, max_genes=150, n_keep=50, method="iqtree"):
    """Gene trees per family; families with a finished tree are skipped (so a run can switch method
    midway). method: iqtree (LG+G4, UFBoot 1000, keeps n_keep bootstrap trees) or fasttree
    (-lg -gamma, ~100x less CPU; SH-like supports scaled to 0-100, no bootstrap trees)."""
    mafft = tool("mafft", tools_dir)
    builder = (("fasttree", tool(("FastTree", "fasttree"), tools_dir)) if method == "fasttree"
               else ("iqtree", tool(("iqtree2", "iqtree3", "iqtree"), tools_dir)))
    outdir = os.path.join(workdir, "gene_trees")
    for d in (outdir, os.path.join(workdir, "alignments")):
        os.makedirs(d, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="hgt_trees_")   # space-free scratch for mafft / iqtree
    seqs = _seqs(workdir)
    jobs = []
    for og, members in _families(workdir):
        genes = [g for gs in members.values() for g in gs]
        n_sp = sum(1 for gs in members.values() if gs)
        if min_genes <= len(genes) <= max_genes and n_sp >= 3:
            jobs.append((og, genes, seqs, outdir, tmp, mafft, builder, n_keep, 1))
    log("gene trees (%s) for %d families with %d-%d genes (%d threads)"
        % (builder[0], len(jobs), min_genes, max_genes, threads))
    t0, done, status = time.time(), 0, {}
    with ThreadPoolExecutor(threads) as ex:
        for og, st in ex.map(_one_tree, jobs):
            done += 1
            key = st.split(":")[0]
            status[key] = status.get(key, 0) + 1
            if done % 200 == 0 or done == len(jobs):
                el = time.time() - t0
                log("%d/%d trees (%s), %.0f s elapsed, ~%.0f min left"
                    % (done, len(jobs), status, el, el / done * (len(jobs) - done) / 60))
    shutil.rmtree(tmp, ignore_errors=True)


# -- species tree -------------------------------------------------------------

def _reroot_on_leaf(root, name):
    adj, names = unrooted_adjacency(root)
    leaf = names.index(name)
    other, L = adj[leaf][0]
    new = Node()
    new.add(Node(name, L / 2.0))
    nodes = {other: Node(None, L / 2.0)}
    new.add(nodes[other])
    stack = [(other, leaf)]
    while stack:
        u, frm = stack.pop()
        for v, lv in adj[u]:
            if v != frm:
                nodes[v] = Node(names[v], lv)
                nodes[u].add(nodes[v])
                stack.append((v, u))
    return new


def ultrametric_mpl(root):
    """Mean-path-length dating: node age = mean distance to its leaves; root age scaled to 1."""
    acc = {}
    for nd in root.postorder():
        if not nd.children:
            acc[id(nd)] = (0.0, 1)
        else:
            s = n = 0
            for c in nd.children:
                cs, cn = acc[id(c)]
                s, n = s + cs + cn * (c.length or 0.0), n + cn
            acc[id(nd)] = (s, n)
    age = {k: s / n for k, (s, n) in acc.items()}
    H = age[id(root)] or 1.0
    for nd in root.preorder():
        if nd.parent is not None:
            age[id(nd)] = min(age[id(nd)], age[id(nd.parent)] - 1e-4 * H) if nd.children else 0.0
            nd.length = max(1e-6, (age[id(nd.parent)] - age[id(nd)]) / H)
    return root


def speciestree(genomes, workdir, tools_dir, threads, max_families=300):
    iqtree = tool(("iqtree2", "iqtree3", "iqtree"), tools_dir)
    labels = [g["label"] for g in genomes]
    outgroup = [g["label"] for g in genomes if g.get("role") == "outgroup"]
    single = []
    for og, members in _families(workdir):
        if all(len(members.get(s, [])) == 1 for s in labels):
            path = os.path.join(workdir, "alignments", og + ".faa")
            if os.path.exists(path):
                single.append((og, members, path))
    random.Random(1).shuffle(single)
    single = single[:max_families]
    if len(single) < 20:
        raise SystemExit("only %d single-copy families with alignments; run genetrees first" % len(single))
    concat = {s: [] for s in labels}
    for og, members, path in single:
        aln = _read_aln(path)
        L = len(next(iter(aln.values())))
        by_sp = {}
        for g, s in aln.items():
            by_sp[g.split("__")[0]] = s
        for s in labels:
            concat[s].append(by_sp.get(s, "-" * L))
    d = os.path.join(workdir, "species_tree_build")
    os.makedirs(d, exist_ok=True)
    sup = os.path.join(d, "concat.faa")
    with open(sup, "w") as fh:
        for s in labels:
            fh.write(">%s\n%s\n" % (s, "".join(concat[s])))
    log("species tree from %d single-copy families (%d aa)" % (len(single), len("".join(concat[labels[0]]))))
    pre = os.path.join(d, "concat")
    if not os.path.exists(pre + ".treefile"):
        tmp = tempfile.mkdtemp(prefix="hgt_sptree_")
        try:
            shutil.copy(sup, os.path.join(tmp, "concat.faa"))
            subprocess.run([iqtree, "-s", os.path.join(tmp, "concat.faa"), "-m", "LG+F+R4", "-B", "1000",
                            "-T", str(threads), "--seed", "1", "--quiet", "-pre", os.path.join(tmp, "concat")],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for ext in (".treefile", ".iqtree", ".contree"):
                if os.path.exists(os.path.join(tmp, "concat" + ext)):
                    shutil.copy(os.path.join(tmp, "concat" + ext), pre + ext)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    with open(pre + ".treefile") as fh:
        ml = parse_newick(fh.read())
    rooted = _reroot_on_leaf(ml, outgroup[0]) if outgroup else ml
    with open(os.path.join(workdir, "species_tree.ml_rooted.nwk"), "w") as fh:
        fh.write(to_newick(rooted) + "\n")
    ultra = ultrametric_mpl(rooted)
    k = 0
    for nd in ultra.preorder():
        if nd.children:
            k += 1
            nd.name = "N%d" % k
    with open(os.path.join(workdir, "species_tree.nwk"), "w") as fh:
        fh.write(to_newick(ultra) + "\n")
    log("species_tree.nwk written (rooted on %s, MPL ultrametric)" % (outgroup[0] if outgroup else "IQ-TREE root"))


def gc_content(cds):
    """(GC, GC3) of a coding sequence; GC3 over third codon positions without the stop codon."""
    s = cds.upper()
    acgt = sum(s.count(b) for b in "ACGT")
    if acgt < 60:
        return None
    gc = (s.count("G") + s.count("C")) / float(acgt)
    third = s[2:len(s) - (3 if len(s) >= 6 else 0):3]
    t = sum(third.count(b) for b in "ACGT")
    gc3 = (third.count("G") + third.count("C")) / float(t) if t else gc
    return gc, gc3


def composition(genomes, workdir):
    """GC / GC3 of the CDS behind each protein kept by `proteomes` (matched by transcript, then gene)."""
    keep = set()
    with open(os.path.join(workdir, "gene_coords.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            keep.add(r["gene_id"])
    rows = []
    for g in genomes:
        label = g["label"]
        cds_path = os.path.join(workdir, "raw", label + ".cds.gz")
        if not os.path.exists(cds_path):
            log("%s: no CDS file (run the fetch step); composition skipped" % label)
            continue
        by_tx, by_gene = {}, {}
        for header, _ in _read_fasta_gz(os.path.join(workdir, "raw", label + ".pep.gz")):
            r = _parse_header(header)
            gid = "%s__%s" % (label, safe(r["pid"]))
            if gid in keep:
                by_tx[r.get("transcript", r["pid"])] = gid
                by_gene.setdefault(r.get("gene", ""), gid)
        n = 0
        seen = set()
        for header, seq in _read_fasta_gz(cds_path):
            r = _parse_header(header)
            gid = by_tx.get(r["pid"]) or by_gene.get(r.get("gene", ""))
            if gid is None or gid in seen:
                continue
            c = gc_content(seq)
            if c is None:
                continue
            seen.add(gid)
            rows.append([gid, "%.4f" % c[0], "%.4f" % c[1], len(seq)])
            n += 1
        log("%-16s composition for %d / %d genes" % (label, n, len(by_tx)))
    with open(os.path.join(workdir, "gene_composition.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["gene_id", "gc", "gc3", "cds_len"])
        w.writerows(rows)


def run_prepare(a):
    genomes = read_genomes(a.genomes)
    os.makedirs(a.workdir, exist_ok=True)
    if a.tools and os.path.isdir(a.tools):   # OrthoFinder calls diamond/mcl/fastme via PATH
        os.environ["PATH"] = a.tools + os.pathsep + os.environ.get("PATH", "")
    steps = a.steps.split(",")
    if "fetch" in steps:
        fetch(genomes, a.index, a.workdir, a.division)
    if "proteomes" in steps:
        proteomes(genomes, a.workdir)
    if "composition" in steps:
        composition(genomes, a.workdir)
    if "orthogroups" in steps:
        orthogroups(genomes, a.workdir, a.tools, a.threads)
    if "genetrees" in steps:
        genetrees(a.workdir, a.tools, a.threads, max_genes=a.max_genes, method=a.tree_method)
    if "speciestree" in steps:
        speciestree(genomes, a.workdir, a.tools, a.threads)
