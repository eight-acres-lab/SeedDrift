"""Stage islands (pipeline selfsup): the development chromosomes of the genomic-island reference set (bench/islands,
bench/hard) scored by the self-supervised model and by this line's own comparators, and evaluated.

  python -m ci.stages.islands RUN_DIR data     inputs, search and model arrays (CPU; runs beside the training)
  python -m ci.stages.islands RUN_DIR score    every variant of the model and the comparators score every gene (GPU lease)
  python -m ci.stages.islands RUN_DIR report   bench/hard/evaluate.py on the scores, and data/selfsup/report.json

Configuration (`islands:`): store (the benchmark's folder: dev_genomes/<acc>.gb, genes/, labels/, labels_panel/, the V1
views), genomes (default bench/islands/results/dev_genomes.tsv), sealed (default bench/hard/results/sealed.tsv),
views (default dev24_v/genus: hidden_<acc>.tsv), threads, merge (default Shigella=Escherichia), max_target_seqs (4,000),
wait_for_database_s (0: a machine still copying the reference database in waits for its full size), reuse_search_from
(runs whose island search to take whole: the hit files of a run with the same search key, see search_key(); the
partner search of objective C and the model's arrays are made again, as they depend on the corpus).

data   Only accessions of the development list; a sealed accession, or one outside the development list, stops the stage
       (and gate G3_islands checks the result again). Per chromosome: the benchmark's gene table genes/<acc>.tsv
       (ids, coordinates; the record is parsed the way bench/islands/labels.py parses it and must give the same table),
       proteins and coding sequences from the GenBank record, the taxid from the record. One DIAMOND search of all
       their proteins against the run's reference database (checked against the reference dataset) with the settings
       of the profile's search, but a list of islands.max_target_seqs (default 4,000) targets per gene, as long as the
       benchmark's own V1 views; the proteomes of the chromosome's view (V1: the benchmark's hidden_<acc>.tsv, measured
       closeness; plus its own taxid and genus) are dropped from that list before the nearest rows are chosen. Genes
       whose list was full and lost every row to the view are counted (a truncation the view cannot undo). Then the
       model's arrays (train_data.one_genome) and data/islands/index.json.
score  The model scores with selfsup.score_genomes; no label file is opened. The comparators of this line, from the
       same arrays: cmp_codon (divergence of the gene's codon frequencies from the rest of the chromosome), cmp_gc3
       (|GC3 of the gene - GC3 of the rest|), cmp_hits (minus the tracked hit rows under the V1 view), cmp_short (minus
       the gene's codons). The model must beat them; they are not the other line's comparators. (The model also hides
       its panel genomes' genera, which these comparators do not know of.)
report bench/hard/evaluate.py (gated labels, units by genus, the registered metrics) on every score; this line's
       contrasts (each model score against each comparator, paired over units) from its per-chromosome results.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from ci import ROOT
from ci import manifest as M
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import Stage, main_args, write_json
from ci.stages.search import FIELDS, flush, gene_ids
from ci.stages.train_data import RANKS, one_genome
from ci.taxonomy import Taxonomy

COMPARATORS = ("cmp_codon", "cmp_gc3", "cmp_hits", "cmp_short", "cmp_kmer", "cmp_pairnull", "cmp_triplet")
_BASES = np.array([0, 1, 2, 3])

# The search part of do_data (build_inputs, the hidden view, Keep, the DIAMOND call; ci/stages/search.py's flush and FIELDS
# enter the key through that file's hash): raise ISLAND_SEARCH_VERSION whenever it changes, so that
# islands.reuse_search_from never takes hits made by other search code. SEARCH_V1_FILES: sha256 of islands.py at the
# commits made before the constant existed whose search part is version 1 (144bb36, f63bca8, d823bfc, 46f8c7d, 01ea339
# share this file); runs of those commits record no version.
ISLAND_SEARCH_VERSION = 1
SEARCH_V1_FILES = {"0291a2e6a7b2044b3d2cd5fa22575b975feb2625fb198dc81139b7e484dc4ed6"}
SEARCH_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search.py")


def search_key(run_dir, cfg, accs, version, search_code):
    """What the island hit files depend on besides the per-chromosome inputs (compared separately, see
    search_inputs): the reference dataset, the taxonomy (the genus of the view, the genus kept per row), the DIAMOND
    version, the chromosomes in the order they were searched (a hit list can depend on the other queries of the run),
    the settings of the search including its threads, and its code."""
    ic, sc = cfg["islands"], cfg["search"]
    with open(os.path.join(run_dir, "data", "reference", "manifest.json")) as f:
        ref = json.load(f)["sha256"]
    with open(os.path.join(run_dir, "record.json")) as f:
        rec = json.load(f)
    return dict(reference=ref, taxdump=rec.get("taxdump_sha256"), diamond=(rec.get("env_snapshot") or {}).get("diamond_version.txt"), version=version,
                search_code=sha256_file(search_code, use_sidecar=False), order=[acc for _, acc in accs],
                islands=sha256_json(dict(max_target_seqs=int(ic.get("max_target_seqs", 4000)), views=ic.get("views", "dev24_v/genus"),
                                         threads=int(ic.get("threads", sc.get("threads", 2))))),
                search=sha256_json({k: sc.get(k) for k in ("sensitivity", "evalue", "block_size", "nearest", "keep_sseq")}))


def search_inputs(genomes, folders):
    """Per chromosome what its search read: the sha256 of its query proteins, its taxid, lineage and hidden taxids
    (the view). For our run from the inputs just built; for a candidate from its manifest and index (what it searched,
    not the benchmark files as they are now)."""
    return {g["id"]: dict(proteins=sha256_file(os.path.join(folders[g["id"]], "proteins.faa"), use_sidecar=False), taxid=int(g["taxid"]),
                          ranks=[int(x) for x in g["ranks"]], hidden=sorted(int(x) for x in g["hidden_taxids"])) for g in genomes}


def candidate_inputs(cman, cidx):
    prot = {e["id"]: [fe["sha256"] for fe in e["files"] if fe["path"].endswith("/proteins.faa")] for e in cman["entries"]}
    return {g["id"]: dict(proteins=(prot.get(g["id"]) or [None])[0], taxid=int(g["taxid"]), ranks=[int(x) for x in g["ranks"]],
                          hidden=sorted(int(x) for x in g["hidden_taxids"])) for g in cidx["genomes"]}


def search_candidates(run_dir, cfg):
    """(run, its key, its per-chromosome inputs, its island manifest, its index) for every run of
    islands.reuse_search_from whose island data stage passed and whose island manifest and index are the files that
    stage reported; the code of a candidate is read from its worktree (<root of the run>/wt/<run>/ci/stages/), as
    search.reuse_from does; a candidate with a file missing is skipped."""
    for c in (cfg["islands"].get("reuse_search_from") or []):
        cand = c if os.path.isabs(c) else os.path.join(ROOT, c)
        sp, mp, ip = (os.path.join(cand, "stages", "islands_data.json"), os.path.join(cand, "data", "islands", "manifest.json"),
                      os.path.join(cand, "data", "islands", "index.json"))
        code_dir = os.path.join(os.path.dirname(os.path.dirname(cand)), "wt", os.path.basename(cand), "ci", "stages")
        need = [sp, mp, ip, os.path.join(cand, "config.resolved.yaml"), os.path.join(cand, "record.json"),
                os.path.join(cand, "data", "reference", "manifest.json"), os.path.join(code_dir, "search.py")]
        if not all(os.path.exists(x) for x in need):
            continue
        with open(sp) as f:
            rep = json.load(f)
        out = {o["name"]: o["sha256"] for o in rep.get("outputs") or []}
        if rep.get("status") != "passed" or out.get("islands_manifest") != sha256_file(mp, use_sidecar=False) \
                or out.get("islands_index") != sha256_file(ip, use_sidecar=False):
            continue
        version = rep.get("search_version")                               # stage extras sit at the top of the report
        if version is None and os.path.exists(os.path.join(code_dir, "islands.py")) and \
                sha256_file(os.path.join(code_dir, "islands.py"), use_sidecar=False) in SEARCH_V1_FILES:
            version = 1
        if version is None:
            continue
        with open(os.path.join(cand, "config.resolved.yaml")) as f:
            ccfg = yaml.safe_load(f)
        with open(mp) as f:
            man = json.load(f)
        with open(ip) as f:
            idx = json.load(f)
        accs = [(g["genus"], g["id"]) for g in idx["genomes"]]
        yield cand, search_key(cand, ccfg, accs, int(version), os.path.join(code_dir, "search.py")), candidate_inputs(man, idx), man, idx


def link_reused_hits(run_dir, cand, cman, cidx, ids):
    """Hard-link (or copy, across file systems) the island hit files of run `cand` (its island manifest `cman` and
    index `cidx`) into run_dir, each checked against the manifest's sha256; the chromosomes must be exactly `ids`.
    Returns the rows its view dropped per chromosome."""
    cdropped = {g["id"]: int(g.get("hidden_rows_dropped", 0)) for g in cidx["genomes"]}
    files = {e["id"]: [fe for fe in e["files"] if fe["path"].startswith("data/islands/hits/")] for e in cman["entries"]}
    if sorted(files) != sorted(ids) or any(len(v) != 1 for v in files.values()):
        raise SystemExit("islands: the reused run %s has other chromosomes or hit files" % cand)
    for gid in ids:
        fe = files[gid][0]
        dst = os.path.join(run_dir, fe["path"])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(os.path.join(cand, fe["path"]), dst)
        except OSError:
            shutil.copyfile(os.path.join(cand, fe["path"]), dst)
        if sha256_file(dst, use_sidecar=False) != fe["sha256"]:
            raise SystemExit("islands: reused hit file %s does not match its manifest" % dst)
    return {g: n for g, n in cdropped.items() if n}


def _tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def accessions(st, ic):
    """(genus, accession) of the configured chromosomes; any sealed or non-development accession stops the stage."""
    rows = _tsv(st.repo(ic.get("genomes", "bench/islands/results/dev_genomes.tsv")))
    sealed = {r["accession"] for r in _tsv(st.repo(ic.get("sealed", "bench/hard/results/sealed.tsv")))}
    dev = {r["accession"] for r in _tsv(st.repo("bench/islands/results/dev_genomes.tsv"))} - sealed
    bad = sorted(r["accession"] for r in rows if r["accession"] in sealed or r["accession"] not in dev)
    if bad:
        raise SystemExit("islands: not development chromosomes (sealed or not listed): %s" % " ".join(bad))
    return [(r["genus"], r["accession"]) for r in rows]


def clean_names(seqs, min_len=30):
    """seeddrift.pipeline.clean_proteins (the names the benchmark's gene tables use): {original: new}."""
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


def build_inputs(store, acc, out_dir):
    """genes.tsv / proteins.faa / cds.fna of one chromosome in the corpus layout; returns (taxid, genes)."""
    from seeddrift.inputs import from_genbank
    with open(os.path.join(store, "dev_genomes", acc + ".gb")) as f:
        text = f.read()
    m = re.search(r'/db_xref="taxon:(\d+)"', text)
    if not m:
        raise SystemExit("islands: no taxon in the record of %s" % acc)
    g = from_genbank(text)
    prot, ren = clean_names(g.proteins)
    rows = []
    for gid, new in ren.items():
        contig, s, e, strand, _ = g.coords[gid]
        rows.append((new, contig, str(s), str(e), strand, gid))
    table = [(r["gene_id"], r["contig"], r["start"], r["end"], r["strand"]) for r in _tsv(os.path.join(store, "genes", acc + ".tsv"))]
    if [r[:5] for r in rows] != table:
        raise SystemExit("islands: the genes of the record of %s differ from the benchmark's genes/%s.tsv" % (acc, acc))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "genes.tsv"), "w") as f:
        f.write("gene_id\treplicon\tstart\tend\tstrand\n")
        for r in rows:
            f.write("\t".join(r[:5]) + "\n")
    with open(os.path.join(out_dir, "proteins.faa"), "w") as f:
        for r in rows:
            f.write(">%s__%s\n%s\n" % (acc, r[0], prot[r[0]]))
    with open(os.path.join(out_dir, "cds.fna"), "w") as f:
        for r in rows:
            f.write(">%s__%s\n%s\n" % (acc, r[0], g.cds.get(r[5], "")))
    return int(m.group(1)), len(rows)


class Keep:
    """The rows of one gene that search.select() can choose, kept while DIAMOND streams them in its order (non-increasing
    bit score): the first 2 x nearest rows and every later row tying with the last of those, and the best rows of every
    reference genus (ties included); the aligned residues only for the first group. Bounds the memory of 4,000-target
    lists without changing what select() returns."""

    def __init__(self, nearest, genus_of_prot):
        self.n, self.gp = nearest, genus_of_prot
        self.rows, self.best, self.k, self.cut = [], {}, 0, None

    def add(self, r):
        self.k += 1
        bits, g = r[9], self.gp[r[1]]
        top = self.k <= 2 * self.n or (self.cut is not None and bits >= self.cut)
        if self.k == 2 * self.n:
            self.cut = bits
        gb = self.best.get(g)
        if top or gb is None or bits >= gb:
            if gb is None or bits > gb:
                self.best[g] = bits
            self.rows.append(r if top or len(r) < 12 else r[:11] + ("",))


def hidden_of(store, ic, acc):
    return sorted({int(r["taxid"]) for r in _tsv(os.path.join(store, ic.get("views", "dev24_v/genus"), "hidden_%s.tsv" % acc))})


def do_data(run_dir):
    with Stage(run_dir, "islands_data", requires=("pre",)) as st:
        cfg = st.config
        ic, sc, dc = cfg["islands"], cfg["search"], cfg.get("train_data") or {}
        store = ic["store"]
        accs = accessions(st, ic)
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        with open(os.path.join(run_dir, "data", "reference", "manifest.json")) as f:
            ref = json.load(f)
        db = ref["database"] if os.path.isabs(ref["database"]) else os.path.join(run_dir, ref["database"])
        dbe = [e for e in ref["entries"][0]["files"] if os.path.abspath(e["path"]) == os.path.abspath(db) or os.path.join(run_dir, e["path"]) == db]
        idx = np.load(os.path.join(run_dir, "data", "reference", "index.npz"))
        acc_s, acc_tax = idx["acc"], idx["taxid"]
        lib = np.load(os.path.join(run_dir, "data", "reference", "library.npz"))
        lib_tax, lib_ranks = lib["taxid"], lib["ranks"]
        genus_of_prot = {i: (int(g) if g else -int(t)) for i, (t, g) in enumerate(zip(lib_tax, lib_ranks[:, 4]))}
        base = st.path("islands", "x")[:-2]
        genomes, hidden_mask = [], {}
        q = os.path.join(base, "queries.faa")
        with open(q + ".part", "w") as fo:
            for genus, acc in accs:
                folder = os.path.join(base, "inputs", acc)
                taxid, n = build_inputs(store, acc, folder)
                ranks = [tax.at_rank(taxid, r) for r in RANKS]
                hid = hidden_of(store, ic, acc)
                m = np.isin(lib_tax, sorted(set(hid) | {taxid}))
                if ranks[4]:
                    m |= lib_ranks[:, 4] == ranks[4]
                hidden_mask[acc] = m
                genomes.append(dict(id=acc, genus=genus, taxid=taxid, ranks=ranks, group="bacteria", genes=n, hidden_taxids=hid,
                                    hidden_proteomes=int(m.sum()), depth="genus", folder=os.path.relpath(folder, run_dir)))
                with open(os.path.join(folder, "proteins.faa")) as f:
                    fo.write(f.read())
        os.replace(q + ".part", q)
        keep = int(sc.get("keep_sseq", 0))
        hits_dir = os.path.join(base, "hits")
        os.makedirs(hits_dir, exist_ok=True)
        folders = {g["id"]: os.path.join(run_dir, g["folder"]) for g in genomes}
        # islands.reuse_search_from: the hit files of a run with this search's key and the same per-chromosome inputs,
        # taken whole (the reference database is then not read: a machine without it can score the islands)
        ours, ours_in = search_key(run_dir, cfg, accs, ISLAND_SEARCH_VERSION, SEARCH_CODE), search_inputs(genomes, folders)
        cands = list(search_candidates(run_dir, cfg))
        reuse = next(((c, m, x) for c, k, ki, m, x in cands if k == ours and ki == ours_in), None)
        st.extra.update(search_version=ISLAND_SEARCH_VERSION, search_key=ours)
        if reuse is None and ic.get("reuse_search_required"):
            diffs = {os.path.basename(c): sorted(x for x in ours if k.get(x) != ours[x]) + ([] if ki == ours_in else ["inputs"]) for c, k, ki, _, _ in cands}
            raise SystemExit("islands.reuse_search_required: no run of islands.reuse_search_from has this search's key and inputs; differing parts: %s"
                             % (diffs or "no candidate"))
        if reuse is None:
            # islands.wait_for_database_s: on a machine where the database is still being copied in, wait (up to that
            # many seconds) until it has its full size; the hash below decides whether it is the dataset's database
            t0 = time.time()
            while dbe and time.time() - t0 < float(ic.get("wait_for_database_s", 0)) and not (os.path.exists(db) and os.path.getsize(db) == dbe[0]["bytes"]):
                time.sleep(60)
            st.extra["waited_for_database_s"] = round(time.time() - t0)
            if dbe and (not os.path.exists(db) or sha256_file(db) != dbe[0]["sha256"]):
                raise SystemExit("islands: the reference database %s is missing or differs from the reference dataset" % db)
        if reuse is not None:
            dropped = link_reused_hits(run_dir, reuse[0], reuse[1], reuse[2], [g["id"] for g in genomes])
            st.extra.update(search_reused_from=os.path.basename(reuse[0]))
        else:
            dropped = search_islands(cfg, ic, sc, q, db, keep, hits_dir, folders, genomes, acc_s, acc_tax, lib_tax, hidden_mask, genus_of_prot)
        cap = int(ic.get("max_target_seqs", 4000))
        for g in genomes:                                                  # full lists that the view emptied
            h = np.load(os.path.join(hits_dir, g["id"] + ".npz"))
            g["full_and_emptied"] = int(np.sum((h["nraw"].astype(np.int64) >= min(cap, 65535)) & (np.diff(h["off"]) == 0)))
        _islands_rest(run_dir, st, cfg, ic, sc, dc, tax, ref, base, q, hits_dir, folders, genomes, dropped, cap)


def search_islands(cfg, ic, sc, q, db, keep, hits_dir, folders, genomes, acc_s, acc_tax, lib_tax, hidden_mask, genus_of_prot):
    """The island search (version ISLAND_SEARCH_VERSION): DIAMOND with islands.max_target_seqs targets per gene, the
    chromosome's view dropped before the nearest rows are chosen; one hit file per chromosome. Returns the rows the view
    dropped per chromosome."""
    cmd = [cfg["tools"]["diamond"], "blastp", "-q", q, "-d", db, "-o", "/dev/stdout", "--" + sc.get("sensitivity", "sensitive"),
           "--max-target-seqs", str(ic.get("max_target_seqs", 4000)), "--max-hsps", "1", "-e", str(sc.get("evalue", 1e-5)),
           "--threads", str(ic.get("threads", sc.get("threads", 2))), "-b", str(sc.get("block_size", 4)), "-c", "1",
           "--tmpdir", os.environ.get("TMPDIR", "/tmp"), "--quiet", "--outfmt", "6"] + FIELDS + (["sseq"] if keep else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    cur, by_gene, nraw, done, dropped = None, {}, {}, set(), {}
    try:
        for line in p.stdout:
            f = line.rstrip("\n").split("\t")
            gid, gene = f[0].split("__", 1)
            if gid != cur:
                if cur is not None:
                    flush(os.path.join(hits_dir, cur + ".npz"), gene_ids(folders[cur]), {k: v.rows for k, v in by_gene.items()}, nraw, genus_of_prot,
                          sc.get("nearest", 32), keep)
                    done.add(cur)
                if gid in done:
                    raise SystemExit("search output of %s is not contiguous" % gid)
                cur, by_gene, nraw = gid, {}, {}
            acc, _, t = f[1].partition("|")
            i = int(np.searchsorted(acc_s, acc.encode()))
            if i >= len(acc_s) or acc_s[i] != acc.encode():
                raise SystemExit("target %s is not in the reference index" % f[1])
            pr = int(np.searchsorted(lib_tax, acc_tax[i]))
            nraw[gene] = nraw.get(gene, 0) + 1
            if hidden_mask[gid][pr]:                                   # the view: hidden before the nearest are chosen
                dropped[gid] = dropped.get(gid, 0) + 1
                continue
            if gene not in by_gene:
                by_gene[gene] = Keep(int(sc.get("nearest", 32)), genus_of_prot)
            by_gene[gene].add((i, pr, int(round(float(f[2]) * 10)), int(f[3]), int(f[4]), int(f[5]), int(f[6]), int(f[7]), int(f[8]),
                               float(f[9]), f[10]) + ((f[11],) if keep else ()))
        if p.wait() != 0:
            raise SystemExit("diamond exited with %d" % p.returncode)
        if cur is not None:
            flush(os.path.join(hits_dir, cur + ".npz"), gene_ids(folders[cur]), {k: v.rows for k, v in by_gene.items()}, nraw, genus_of_prot,
                  sc.get("nearest", 32), keep)
            done.add(cur)
    finally:
        if p.poll() is None:
            p.kill()
    for g in genomes:
        if g["id"] not in done:
            flush(os.path.join(hits_dir, g["id"] + ".npz"), gene_ids(folders[g["id"]]), {}, {}, genus_of_prot, sc.get("nearest", 32), keep)
    return dropped


def _islands_rest(run_dir, st, cfg, ic, sc, dc, tax, ref, base, q, hits_dir, folders, genomes, dropped, cap):
    """After the search: the partner search of objective C (it depends on the corpus), the model's arrays, the island
    manifest and index."""
    if cfg.get("pairs"):                                              # objective C: the chromosomes against the corpus
        from ci.stages import pairs as P
        pdir = os.path.join(base, "pairs")
        os.makedirs(pdir, exist_ok=True)
        with open(os.path.join(run_dir, "data", "pairs", "genomes.json")) as f:
            order = json.load(f)["genomes"]
        corpus_dirs = dict(P.corpus(run_dir))
        names = {g: gene_ids(corpus_dirs[g]) for g in order}
        # the view for partners too: corpus genomes of a hidden proteome's taxon, of the chromosome's own taxon or of
        # its genus (merged as in the evaluation) are not partners, left out before the nearest genomes are chosen
        with open(os.path.join(run_dir, "data", "corpus", "manifest.json")) as f:
            ctax = {e["id"]: int(e["taxid"]) for e in json.load(f)["entries"]}
        merge = dict(x.split("=", 1) for x in ic.get("merge", ["Shigella=Escherichia"]))
        unit = lambda name: merge.get(name, name)
        cgenus = {g: tax.name(tax.at_rank(t, "genus")) if tax.at_rank(t, "genus") else "" for g, t in ctax.items()}
        skip = {}
        for g in genomes:
            hid = set(g["hidden_taxids"]) | {g["taxid"]}
            skip[g["id"]] = {i for i, c in enumerate(order) if ctax[c] in hid or (cgenus[c] and unit(cgenus[c]) == unit(g["genus"]))}
            g["partners_skipped"] = len(skip[g["id"]])
        P.search(cfg["tools"]["diamond"], q, os.path.join(run_dir, "data", "pairs", "db.dmnd"), dict(cfg["pairs"], threads=ic.get("threads", 2)),
                 {g: i for i, g in enumerate(order)}, {g: {n: i for i, n in enumerate(v)} for g, v in names.items()},
                 {g["id"]: gene_ids(folders[g["id"]]) for g in genomes}, pdir, own=False, skip=skip)
        for g in genomes:
            g["pairs"] = os.path.relpath(os.path.join(pdir, g["id"] + ".npz"), run_dir)
    os.remove(q)
    entries = []
    arr_dir = os.path.join(base, "train_data")
    os.makedirs(arr_dir, exist_ok=True)
    for g in genomes:
        arrays, n_genes, n_hits = one_genome(folders[g["id"]], os.path.join(hits_dir, g["id"] + ".npz"), g["id"], int(dc.get("max_codons", 1000)), int(dc.get("hits", 32)))
        from seeddrift.train.logger import save_npz
        path = os.path.join(arr_dir, g["id"] + ".npz")
        save_npz(path, compress=True, **arrays)
        g.update(path=os.path.relpath(path, run_dir), tracked_hits=n_hits, hidden_rows_dropped=dropped.get(g["id"], 0))
        entries.append(dict(id=g["id"], files=[M.file_entry(os.path.join(folders[g["id"]], n), run_dir) for n in ("genes.tsv", "proteins.faa", "cds.fna")]
                            + [M.file_entry(os.path.join(hits_dir, g["id"] + ".npz"), run_dir), M.file_entry(path, run_dir)]
                            + ([M.file_entry(os.path.join(run_dir, g["pairs"]), run_dir)] if g.get("pairs") else [])))
    man = M.build("islands", "islands", entries, meta=dict(reference=ref["sha256"], search=dict({k: sc.get(k) for k in ("sensitivity", "evalue", "nearest", "keep_sseq")},
                                                                                                   max_target_seqs=cap),
                                                            views=ic.get("views", "dev24_v/genus")))
    sha = M.write(man, os.path.join(run_dir, "manifests"))
    write_json(os.path.join(base, "manifest.json"), dict(sha256=sha, dataset_id=M.dataset_id(man), **man))
    ip = os.path.join(base, "index.json")
    write_json(ip, dict(genomes=genomes, dataset_id=M.dataset_id(man)))
    st.output("islands_index", ip)
    st.output("islands_manifest", os.path.join(base, "manifest.json"))
    st.extra.update(chromosomes=len(genomes), genes=sum(g["genes"] for g in genomes), hidden_rows_dropped=sum(dropped.values()),
                    full_lists_emptied_by_the_view=sum(g["full_and_emptied"] for g in genomes))


def comparators(gen):
    """This line's comparators of one chromosome (a source_model Genome): higher = more foreign."""
    n = gen.n
    total = gen.ccount.astype(np.float64)
    out = {k: np.zeros(n) for k in COMPARATORS}
    gc3_codon = np.zeros(66)
    for c in range(64):
        gc3_codon[c + 1] = 1.0 if (c % 4) in (1, 2) else 0.0            # third base C or G (ACGT order: 1 = C, 2 = G)
    valid = np.zeros(66, bool)
    valid[1:65] = True
    for i in range(n):
        own = gen.gene_count(i).astype(np.float64)
        rest = total - own
        a, b = own[valid] + 0.5, rest[valid] + 0.5
        pa, pb = a / a.sum(), b / b.sum()
        out["cmp_codon"][i] = float(np.sum(pa * np.log(pa / pb)))
        ga = (own * gc3_codon).sum() / max(own[valid].sum(), 1)
        gb = (rest * gc3_codon).sum() / max(rest[valid].sum(), 1)
        out["cmp_gc3"][i] = abs(ga - gb)
        out["cmp_hits"][i] = -float(gen.hoff[i + 1] - gen.hoff[i])
        out["cmp_short"][i] = -float(gen.coff[i + 1] - gen.coff[i])
    out["cmp_kmer"] = kmer_distance(gen)
    out["cmp_pairnull"] = np.full(n, np.nan)
    out["cmp_triplet"] = np.full(n, np.nan)
    return out


def kmer_distance(gen, k=4):
    """cmp_kmer: KL divergence of the gene's overlapping k-mer frequencies (its coding sequence rebuilt from the codon
    tokens, 1..64 in ACGT order) from those of the rest of the chromosome's genes; higher = more foreign."""
    n = gen.n
    counts = np.zeros((n, 4 ** k))
    for i in range(n):
        t = gen.codons[gen.coff[i]:gen.coff[i + 1]].astype(np.int64)
        t = t[(t >= 1) & (t <= 64)] - 1
        if not len(t):
            continue
        nt = np.stack([t // 16, (t // 4) % 4, t % 4], 1).reshape(-1)
        if len(nt) < k:
            continue
        idx = np.zeros(len(nt) - k + 1, np.int64)
        for j in range(k):
            idx = idx * 4 + nt[j:len(nt) - k + 1 + j]
        counts[i] = np.bincount(idx, minlength=4 ** k)
    total = counts.sum(0)
    out = np.full(n, np.nan)
    for i in range(n):
        if counts[i].sum() == 0:
            continue
        a, b = counts[i] + 0.5, total - counts[i] + 0.5
        pa, pb = a / a.sum(), b / b.sum()
        out[i] = float(np.sum(pa * np.log(pa / pb)))
    return out


def pair_comparators(path, gen, min_support=10):
    """cmp_pairnull and cmp_triplet from the chromosome's partner rows (the stage's search against the corpus; per
    gene the best row in each of the nearest corpus genomes): similarity = bit score per query residue.
      cmp_pairnull  per gene the largest robust z of its similarity to a partner genome against the similarities of all
                    the chromosome's genes to that genome (median, 1.4826 MAD): a gene far closer to a genome than the
                    chromosome's genes usually are
      cmp_triplet   per gene with two partners or more, the share of its partner pairs whose order (which genome the gene
                    is closer to) contradicts the chromosome's order of the same two genomes (their median similarities)
    NaN for genes without partner rows (or with one, for cmp_triplet). The null of a genome is censored to the genes
    that kept it among their nearest partner genomes (the stage keeps those only); a genome with fewer than min_support
    supporting genes enters neither score. The query length is the protein's (the gene table's coordinates), not the
    stored codons, which stop at max_codons."""
    z = np.load(path)
    off, tg, bits = z["off"].astype(np.int64), z["tgenome"].astype(np.int64), z["bits"].astype(np.float64)
    n = gen.n
    if getattr(gen, "start", None) is not None and getattr(gen, "end", None) is not None and len(gen.start) == n:
        qlen = np.maximum((np.asarray(gen.end, np.float64) - np.asarray(gen.start, np.float64) + 1) / 3.0 - 1.0, 1.0)
    else:
        qlen = np.maximum(np.diff(gen.coff).astype(np.float64), 1.0)
    gene_of = np.repeat(np.arange(len(off) - 1), np.diff(off))
    sim = bits / qlen[gene_of] if len(gene_of) else np.zeros(0)
    med, mad = {}, {}
    for t in np.unique(tg):
        v = sim[tg == t]
        if len(v) < min_support:
            continue
        med[int(t)] = float(np.median(v))
        mad[int(t)] = max(1.4826 * float(np.median(np.abs(v - med[int(t)]))), 0.05 * abs(med[int(t)]), 1e-3)
    pn, tr = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(min(n, len(off) - 1)):
        a, b = int(off[i]), int(off[i + 1])
        keep = [r for r in range(a, b) if int(tg[r]) in med]
        if not keep:
            continue
        s_i, t_i = sim[keep], tg[keep]
        zz = np.array([(x - med[int(t)]) / mad[int(t)] for x, t in zip(s_i, t_i)])
        pn[i] = float(zz.max())
        a, b = 0, len(keep)
        if b - a >= 2:
            c = np.array([med[int(t)] for t in t_i])
            ds, dc = s_i[:, None] - s_i[None, :], c[:, None] - c[None, :]
            iu = np.triu_indices(b - a, 1)
            prod = ds[iu] * dc[iu]
            tr[i] = float((prod < 0).sum() / len(prod))
    return dict(cmp_pairnull=pn, cmp_triplet=tr)


def write_scores(path, gene_names, cols):
    with open(path + ".part", "w") as f:
        f.write("gene_id\t" + "\t".join(cols) + "\n")
        for i, g in enumerate(gene_names):
            f.write(g + "\t" + "\t".join("" if not np.isfinite(cols[k][i]) else repr(float(cols[k][i])) for k in cols) + "\n")
    os.replace(path + ".part", path)


def do_score(run_dir):
    with Stage(run_dir, "islands_score", requires=("pre", "islands")) as st:
        cfg = st.config
        from ci import gpu_lease
        from ci.stages.train import model_env, selfsup_confs, selfsup_env
        from seeddrift.train import source_model as S
        tc = cfg["train"]
        py = tc.get("python", "/usr/local/bin/python3")
        model_env(st, py, tc.get("python_lock", "ci/envs/model-torch.pip.txt"))
        _, confs = selfsup_confs(tc, run_dir, cfg["git"]["sha"], has_pairs=bool(cfg.get("pairs")))
        base = os.path.join(run_dir, "data", "islands")
        with open(os.path.join(base, "index.json")) as f:
            idx = json.load(f)
        env = selfsup_env(tc)                                              # train.device cpu: the GPUs hidden
        st.extra["device"] = tc.get("device", "auto")
        script = st.repo("seeddrift/train/selfsup.py")
        with gpu_lease.hold("islands-" + os.path.basename(run_dir), enabled=bool(tc.get("gpu_lease", True))) as lease:
            st.extra["gpu_lease"] = lease
            rd = tc.get("readouts") or {}
            for c in confs:
                conf = dict(c, genomes_index=os.path.relpath(os.path.join(base, "index.json"), run_dir), genomes_out=os.path.join("data", "islands", "raw", c["name"]))
                if c["name"] in (rd.get("variants") or []):                  # train.readouts: round 1 of the new objectives
                    conf["readouts"] = {k: v for k, v in rd.items() if k != "variants"}
                cp = st.path("islands", "score-%s.json" % c["name"])
                write_json(cp, conf)
                with open(st.path("islands", "score-%s.log" % c["name"]), "w") as lg:
                    pr = subprocess.Popen([py, script, "genomes", run_dir, cp], stdout=lg, stderr=subprocess.STDOUT, env=env)
                    lease.add(pr.pid)
                    rc = pr.wait()
                if rc != 0:
                    raise SystemExit("scoring the islands with variant %s failed (exit %d); see %s" % (c["name"], rc, cp[:-5] + ".log"))
        n_genes = 0
        for g in idx["genomes"]:
            names = gene_ids(os.path.join(run_dir, g["folder"]))
            n_genes += len(names)
            for c in confs:
                z = np.load(os.path.join(base, "raw", c["name"], g["id"] + ".npz"))
                if len(z["gene"]) != len(names) or not np.array_equal(z["gene"], np.arange(len(names))):
                    raise SystemExit("islands: the scores of %s (%s) do not cover its genes in order" % (g["id"], c["name"]))
                d = os.path.join(base, "scores", c["name"])
                os.makedirs(d, exist_ok=True)
                cols = {"%s_%s" % (c["name"], k): z[k] for k in ("A", "B1", "B2", "C", "comb") if k in z.files and k in set(c["objectives"]) | {"comb"}}
                cols.update({"%s_%s" % (c["name"], k): z[k] for k in sorted(z.files) if k.startswith("R_")})
                write_scores(os.path.join(d, "scores_%s.tsv" % g["id"]), names, cols)
            gen = S.Genome(dict(id=g["id"], taxid=g["taxid"], ranks=g["ranks"], group=g["group"]), os.path.join(run_dir, g["path"]))
            d = os.path.join(base, "scores", "comparators")
            os.makedirs(d, exist_ok=True)
            cmp_cols = comparators(gen)
            if g.get("pairs"):
                cmp_cols.update(pair_comparators(os.path.join(run_dir, g["pairs"]), gen))
            write_scores(os.path.join(d, "scores_%s.tsv" % g["id"]), names, cmp_cols)
        out = os.path.join(base, "scores.json")
        write_json(out, dict(variants=[c["name"] for c in confs], comparators=list(COMPARATORS), chromosomes=[g["id"] for g in idx["genomes"]], genes=n_genes,
                             folders=sorted(os.listdir(os.path.join(base, "scores")))))
        st.output("islands_scores", out)


def contrasts(per, units, methods, comps):
    """Paired over units: for every model score and comparator, the mean difference of the unit means of a metric with a
    t-interval (95%) and the sign-flip p over units (bench/hard/metrics.py, as the benchmark's evaluator)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bench", "hard"))
    import metrics as BM
    out = []
    for metric in ("r1", "auroc"):
        for a in methods:
            for b in comps:
                d = []
                for u in units:
                    va = [per[x][a][metric] for x in per if per[x]["_unit"] == u and a in per[x] and metric in per[x][a]]
                    vb = [per[x][b][metric] for x in per if per[x]["_unit"] == u and b in per[x] and metric in per[x][b]]
                    if va and vb:
                        d.append(float(np.mean(va)) - float(np.mean(vb)))
                if not d:
                    continue
                mean, lo, hi = BM.t_interval(d)
                out.append(dict(metric=metric, a=a, b=b, n_units=len(d), mean=mean, lo=lo, hi=hi, p_signflip=BM.sign_flip_p(d), per_unit=list(d)))
    return out


def protocol_report(run_dir, base, sj, store, per, units_of, model_cols):
    """The shared test protocol (ci/stages/protocol.py) on every model column and comparator, gated labels, the
    chromosomes the evaluator admits at gene level; and the leak audit of objective C's partners."""
    from ci.stages import protocol as PR
    if not per:
        return dict(note="no chromosome admitted at gene level with gated labels", scores={})
    table = {}
    for acc in sorted(per):
        lab = {r["gene_id"]: r["class"] for r in _tsv(os.path.join(store, "labels_panel", acc + ".tsv"))}
        cols = {}
        names = None
        for d in sj["folders"]:
            rows = _tsv(os.path.join(base, "scores", d, "scores_%s.tsv" % acc))
            names = [r["gene_id"] for r in rows]
            for k in rows[0]:
                if k != "gene_id":
                    cols[k] = np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
        y = np.array([1 if lab.get(g) == "island" else 0 if lab.get(g) == "native" else -1 for g in names])
        table[acc] = dict(y=y, cols=cols, unit=units_of[acc])
    units = sorted({t["unit"] for t in table.values()})
    scores = list(model_cols) + list(COMPARATORS)
    ev = PR.evaluate(table, units, scores, list(COMPARATORS))
    names = [n for n in scores if all(n in t["cols"] for t in table.values())]
    spear = PR.spearman_matrix(table, names)
    against = list(PR.KNOWN[:3]) + ["cmp_hits", "cmp_short"] + [n for n in model_cols if n.endswith(("_A", "_B1", "_C")) and "_R_" not in n]
    leak = leak_audit(run_dir, base, sj, table)
    return dict(note=PR.__doc__.strip().splitlines()[0], scores=ev, spearman=spear, collapsed=PR.collapsed(spear, names, against),
                leak_audit=leak, chromosomes=sorted(table), units=units, not_implemented=list(PR.NOT_IMPLEMENTED))


def leak_audit(run_dir, base, sj, table):
    """Where objective C's chosen partners come from (by corpus group) for island and native genes, and the share of
    island genes with any partner row to a cellular genome (not phage, plasmid or virus)."""
    gp = os.path.join(run_dir, "data", "pairs", "genomes.json")
    if not os.path.exists(gp):
        return {}
    with open(gp) as f:
        order = json.load(f)["genomes"]
    with open(os.path.join(run_dir, "data", "train_data", "index.json")) as f:
        grp = {g["id"]: g["group"] for g in json.load(f)["genomes"]}
    cellular = lambda gid: not grp.get(gid, "").startswith(("viruses.", "mobile."))
    out = {}
    for v in sj["variants"]:
        cnt = {1: {}, 0: {}}
        for acc, t in table.items():
            p = os.path.join(base, "raw", v, acc + ".npz")
            if not os.path.exists(p):
                continue
            z = np.load(p)
            if "c_partner" not in z.files or not np.isfinite(z["C"]).any():
                continue
            for cls in (1, 0):
                for tgi in z["c_partner"][t["y"] == cls]:
                    key = grp.get(order[int(tgi)], "?") if tgi >= 0 else "none"
                    cnt[cls][key] = cnt[cls].get(key, 0) + 1
        if cnt[1] or cnt[0]:
            out[v] = dict(island=cnt[1], native=cnt[0])
    any_cell = {1: [0, 0], 0: [0, 0]}
    with open(os.path.join(base, "index.json")) as f:
        pairs_of = {g["id"]: g.get("pairs") for g in json.load(f)["genomes"]}
    for acc, t in table.items():
        if not pairs_of.get(acc):
            continue
        z = np.load(os.path.join(run_dir, pairs_of[acc]))
        off, tg = z["off"], z["tgenome"]
        for i, y in enumerate(t["y"]):
            if y < 0 or i >= len(off) - 1:
                continue
            rows = tg[off[i]:off[i + 1]]
            any_cell[y][0] += int(any(cellular(order[int(r)]) for r in rows))
            any_cell[y][1] += 1
    out["share_with_a_cellular_partner"] = {("island" if k else "native"): round(a / max(b, 1), 4) for k, (a, b) in any_cell.items()}
    return out


def do_report(run_dir):
    with Stage(run_dir, "islands_report", requires=("pre", "islands")) as st:
        cfg = st.config
        ic = cfg["islands"]
        store = ic["store"]
        base = os.path.join(run_dir, "data", "islands")
        with open(os.path.join(base, "scores.json")) as f:
            sj = json.load(f)
        ev = st.path("islands", "eval", "x")[:-2]
        merge = ic.get("merge", ["Shigella=Escherichia"])
        cmd = [sys.executable, st.repo("bench/hard/evaluate.py"), "--labels", os.path.join(store, "labels_panel"), "--published", os.path.join(store, "labels"),
               "--genomes", st.repo(ic.get("genomes", "bench/islands/results/dev_genomes.tsv")), "--merge"] + merge + \
              ["--scores"] + [os.path.join(base, "scores", d) for d in sj["folders"]] + ["--shifts", str(int(ic.get("shifts", 200))), "--out", os.path.join(ev, "dev")]
        with open(os.path.join(ev, "evaluate.log"), "w") as lg:
            r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=st.repo("."))
        if r.returncode != 0:
            raise SystemExit("bench/hard/evaluate.py failed (exit %d); see %s" % (r.returncode, os.path.join(ev, "evaluate.log")))
        with open(os.path.join(ev, "dev.json")) as f:
            res = json.load(f)
        # this line's contrasts: chromosomes the evaluator admits at gene level (dev_gene), gated labels, view V1; the
        # gene's own score and its window (_nb) compared like with like
        per = {}
        for acc, pc in res["per_chromosome"].items():
            gated = (pc.get("v1") or {}).get("gated")
            if not res["eligibility"][acc].get("dev_gene") or not gated:
                continue
            per[acc] = dict(gated, _unit=res["units"][acc])
        units = sorted({v["_unit"] for v in per.values()})
        model_cols = []                                                    # the columns the variants have (their objectives)
        for v in sj["variants"]:
            with open(os.path.join(base, "scores", v, "scores_%s.tsv" % sj["chromosomes"][0])) as f:
                model_cols += f.readline().rstrip("\n").split("\t")[1:]
        rows = contrasts(per, units, model_cols, list(COMPARATORS)) + contrasts(per, units, [m + "_nb" for m in model_cols], [c + "_nb" for c in COMPARATORS])
        table = {}
        for acc, v in per.items():
            for m, mv in v.items():
                if m not in ("_unit", "_n") and isinstance(mv, dict):
                    table.setdefault(m, {})[acc] = {k: mv.get(k) for k in ("r1", "auroc", "mr5")}
        out = st.path("islands", "contrasts.json")
        write_json(out, dict(units=units, chromosomes=sorted(per), contrasts=rows, metrics=table,
                             note="development chromosomes only; units are genera (merged as configured); t-intervals over units"))
        prot = protocol_report(run_dir, base, sj, store, per, res["units"], model_cols)
        pp = st.path("islands", "protocol.json")
        write_json(pp, prot)
        st.output("islands_protocol", pp)
        st.extra["evaluate"] = sorted(os.listdir(ev))
        st.output("islands_eval", os.path.join(ev, "dev.json"))
        st.output("islands_contrasts", out)


def main():
    run_dir = main_args()
    {"data": do_data, "score": do_score, "report": do_report}[sys.argv[2]](run_dir)


if __name__ == "__main__":
    main()
