"""Stage islands (pipeline selfsup): the development chromosomes of the genomic-island reference set (bench/islands,
bench/hard) scored by the self-supervised model and by this line's own comparators, and evaluated.

  python -m ci.stages.islands RUN_DIR data     inputs, search and model arrays (CPU; runs beside the training)
  python -m ci.stages.islands RUN_DIR score    every variant of the model and the comparators score every gene (GPU lease)
  python -m ci.stages.islands RUN_DIR report   bench/hard/evaluate.py on the scores, and data/selfsup/report.json

Configuration (`islands:`): store (the benchmark's folder: dev_genomes/<acc>.gb, genes/, labels/, labels_panel/, the V1
views), genomes (default bench/islands/results/dev_genomes.tsv), sealed (default bench/hard/results/sealed.tsv),
views (default dev24_v/genus: hidden_<acc>.tsv), threads, merge (default Shigella=Escherichia).

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
import subprocess
import sys

import numpy as np

from ci import manifest as M
from ci.hashing import sha256_file
from ci.stages.common import Stage, main_args, write_json
from ci.stages.search import FIELDS, flush, gene_ids
from ci.stages.train_data import RANKS, one_genome
from ci.taxonomy import Taxonomy

COMPARATORS = ("cmp_codon", "cmp_gc3", "cmp_hits", "cmp_short")


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
        if dbe and (not os.path.exists(db) or sha256_file(db) != dbe[0]["sha256"]):
            raise SystemExit("islands: the reference database %s is missing or differs from the reference dataset" % db)
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
        cmd = [cfg["tools"]["diamond"], "blastp", "-q", q, "-d", db, "-o", "/dev/stdout", "--" + sc.get("sensitivity", "sensitive"),
               "--max-target-seqs", str(ic.get("max_target_seqs", 4000)), "--max-hsps", "1", "-e", str(sc.get("evalue", 1e-5)),
               "--threads", str(ic.get("threads", sc.get("threads", 2))), "-b", str(sc.get("block_size", 4)), "-c", "1",
               "--tmpdir", os.environ.get("TMPDIR", "/tmp"), "--quiet", "--outfmt", "6"] + FIELDS + (["sseq"] if keep else [])
        hits_dir = os.path.join(base, "hits")
        os.makedirs(hits_dir, exist_ok=True)
        folders = {g["id"]: os.path.join(run_dir, g["folder"]) for g in genomes}
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
        cur, by_gene, nraw, done, dropped = None, {}, {}, set(), {}
        try:
            for line in p.stdout:
                f = line.rstrip("\n").split("\t")
                gid, gene = f[0].split("__", 1)
                if gid != cur:
                    if cur is not None:
                        flush(os.path.join(hits_dir, cur + ".npz"), gene_ids(folders[cur]), by_gene, nraw, genus_of_prot, sc.get("nearest", 32), keep)
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
                by_gene.setdefault(gene, []).append((i, pr, int(round(float(f[2]) * 10)), int(f[3]), int(f[4]), int(f[5]), int(f[6]), int(f[7]), int(f[8]),
                                                     float(f[9]), f[10]) + ((f[11],) if keep else ()))
            if p.wait() != 0:
                raise SystemExit("diamond exited with %d" % p.returncode)
            if cur is not None:
                flush(os.path.join(hits_dir, cur + ".npz"), gene_ids(folders[cur]), by_gene, nraw, genus_of_prot, sc.get("nearest", 32), keep)
                done.add(cur)
        finally:
            if p.poll() is None:
                p.kill()
        for g in genomes:
            if g["id"] not in done:
                flush(os.path.join(hits_dir, g["id"] + ".npz"), gene_ids(folders[g["id"]]), {}, {}, genus_of_prot, sc.get("nearest", 32), keep)
        cap = int(ic.get("max_target_seqs", 4000))
        for g in genomes:                                                  # full lists that the view emptied
            h = np.load(os.path.join(hits_dir, g["id"] + ".npz"))
            g["full_and_emptied"] = int(np.sum((h["nraw"].astype(np.int64) >= min(cap, 65535)) & (np.diff(h["off"]) == 0)))
        if cfg.get("pairs"):                                              # objective C: the chromosomes against the corpus
            from ci.stages import pairs as P
            pdir = os.path.join(base, "pairs")
            os.makedirs(pdir, exist_ok=True)
            with open(os.path.join(run_dir, "data", "pairs", "genomes.json")) as f:
                order = json.load(f)["genomes"]
            corpus_dirs = dict(P.corpus(run_dir))
            names = {g: gene_ids(corpus_dirs[g]) for g in order}
            P.search(cfg["tools"]["diamond"], q, os.path.join(run_dir, "data", "pairs", "db.dmnd"), dict(cfg["pairs"], threads=ic.get("threads", 2)),
                     {g: i for i, g in enumerate(order)}, {g: {n: i for i, n in enumerate(v)} for g, v in names.items()},
                     {g["id"]: gene_ids(folders[g["id"]]) for g in genomes}, pdir, own=False)
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
    return out


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
        from ci.stages.train import model_env, selfsup_confs
        from seeddrift.train import source_model as S
        tc = cfg["train"]
        py = tc.get("python", "/usr/local/bin/python3")
        model_env(st, py, tc.get("python_lock", "ci/envs/model-torch.pip.txt"))
        _, confs = selfsup_confs(tc, run_dir, cfg["git"]["sha"], has_pairs=bool(cfg.get("pairs")))
        base = os.path.join(run_dir, "data", "islands")
        with open(os.path.join(base, "index.json")) as f:
            idx = json.load(f)
        env = dict(os.environ, PYTHONHASHSEED="0", PYTORCH_CUDA_ALLOC_CONF=str(tc.get("cuda_alloc_conf", "expandable_segments:True")))
        script = st.repo("seeddrift/train/selfsup.py")
        with gpu_lease.hold("islands-" + os.path.basename(run_dir), enabled=bool(tc.get("gpu_lease", True))) as lease:
            st.extra["gpu_lease"] = lease
            for c in confs:
                conf = dict(c, genomes_index=os.path.relpath(os.path.join(base, "index.json"), run_dir), genomes_out=os.path.join("data", "islands", "raw", c["name"]))
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
                write_scores(os.path.join(d, "scores_%s.tsv" % g["id"]), names, {"%s_%s" % (c["name"], k): z[k] for k in ("A", "B1", "B2", "C", "comb")
                                                                                    if k in z.files and k in set(c["objectives"]) | {"comb"}})
            gen = S.Genome(dict(id=g["id"], taxid=g["taxid"], ranks=g["ranks"], group=g["group"]), os.path.join(run_dir, g["path"]))
            d = os.path.join(base, "scores", "comparators")
            os.makedirs(d, exist_ok=True)
            write_scores(os.path.join(d, "scores_%s.tsv" % g["id"]), names, comparators(gen))
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
                if m != "_unit" and isinstance(mv, dict):
                    table.setdefault(m, {})[acc] = {k: mv.get(k) for k in ("r1", "auroc", "mr5")}
        out = st.path("islands", "contrasts.json")
        write_json(out, dict(units=units, chromosomes=sorted(per), contrasts=rows, metrics=table,
                             note="development chromosomes only; units are genera (merged as configured); t-intervals over units"))
        st.extra["evaluate"] = sorted(os.listdir(ev))
        st.output("islands_eval", os.path.join(ev, "dev.json"))
        st.output("islands_contrasts", out)


def main():
    run_dir = main_args()
    {"data": do_data, "score": do_score, "report": do_report}[sys.argv[2]](run_dir)


if __name__ == "__main__":
    main()
