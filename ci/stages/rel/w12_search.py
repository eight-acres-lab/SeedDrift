"""Stage rel_w12_search (package W12; pipeline rel_tables): the search of the corpus without the caps of stage pairs
(seeddrift/train/rel/tables/v1_hits.py holds the rows). `python -m ci.stages.rel.w12_search RUN_DIR`.

Scope (task breakdown section 8.1, item 2): the database holds the proteins of the training genomes only; the queries are
the proteins of the training and held-out genomes; every held-out genome is also searched against its own proteins (a
training genome meets its own proteins in the database), so every genome's other copies of a gene are found. Both
lists pass the sealed check before any protein is read; the sealed test split is never opened. Every target and every
alignment of a pair is kept (DIAMOND -k 0 --max-hsps 0); stage pairs' caps (64 genomes, the best row per genome, one
alignment per pair, 500 targets, no own genome) do not apply. No score or rank is stored.

Inputs: the corpus proteins and gene tables of a passed selfsup run (rel_w12_search.source; each file checked against
the run's corpus manifest), its training-data index (the splits) and gene order (genes.tsv, the order of the training
arrays). With rel_w12_search.islands (task breakdown section 8.8), the development chromosomes of the island set that
the island stage of a passed run prepared (islands_from, default the source) are extra queries, split "island": every
accession must be a development chromosome and none a sealed one (ci/stages/rel/common.py development_chromosomes),
every file is checked against the island manifest, and each is searched against the training database and against its
own proteins. They are queries only, never rows of a training table (they are of the order the corpus excludes).

Configuration (profile block rel_w12_search): source, expect ({split: genomes}), islands (true: the development
chromosomes as extra queries), islands_from, sample (share of the query proteins searched, drawn per group, the island
chromosomes as a group of their own; 1.0 = all), seed, threads, block_size, sensitivity, evalue, chunk_proteins (query proteins
per DIAMOND call), self_batch_proteins (proteins of the held-out genomes whose own databases one call searches),
probe_proteins (a first, discarded call of that many query proteins that measures DIAMOND's fixed cost; default 200 in
a sample run, 0 otherwise), wall_s (stop before a call that would end after it; the run is resumed later, finished
calls are kept), reserve_gb.
Outputs under data/rel_w12/search/: <genome>.npz per query genome (v1_hits.py), genomes.json (the genome list the
arrays index), index/ (the lookup index of all of them, memory-mappable: v1_hits.Index), report.md and manifest.json (written last: settings, sizes, wall times, rows, bytes and, for a sample
run, the linear extrapolation to the full search with the disk it needs).
Gate G4_rel_w12_search (gate()): every query genome has its file, sound and as listed; the query genomes are the open
splits of the source and the database genomes its training split; the files are the ones the manifest names.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, read_fasta, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from seeddrift.train.rel.tables import v0_build as V
from seeddrift.train.rel.tables import v1_hits as H

OUT = os.path.join("data", "rel_w12", "search")
KEYS = {"source", "expect", "islands", "islands_from", "sample", "seed", "threads", "block_size", "sensitivity", "evalue", "chunk_proteins", "self_batch_proteins",
        "probe_proteins", "wall_s", "reserve_gb"}


def settings(cfg) -> dict:
    c = cfg.get("rel_w12_search") or {}
    unknown = set(c) - KEYS
    if unknown or not c.get("source"):
        raise SystemExit("rel_w12_search: needs `source`; unknown keys %s" % sorted(unknown))
    sample = float(c.get("sample", 1.0))
    out = dict(source=str(c["source"]), expect={str(k): int(v) for k, v in (c.get("expect") or {}).items()}, islands=bool(c.get("islands", False)),
               islands_from=str(c.get("islands_from") or c["source"]), sample=sample, seed=int(c.get("seed", cfg.get("seed", 1))),
               threads=int(c.get("threads", 12)), block_size=float(c.get("block_size", 4)), sensitivity=str(c.get("sensitivity", "sensitive")),
               evalue=float(c.get("evalue", 1e-5)), chunk_proteins=int(c.get("chunk_proteins", 60000)), self_batch_proteins=int(c.get("self_batch_proteins", 30000)),
               probe_proteins=int(c.get("probe_proteins", 200 if sample < 1 else 0)), wall_s=float(c["wall_s"]) if c.get("wall_s") else None,
               reserve_gb=float(c.get("reserve_gb", 30)), max_left_out=20)
    if not (0 < sample <= 1 and 1 <= out["threads"] <= 64 and out["chunk_proteins"] >= 1 and out["self_batch_proteins"] >= 1 and out["probe_proteins"] >= 0
            and out["sensitivity"] in ("fast", "mid-sensitive", "sensitive", "more-sensitive", "very-sensitive", "ultra-sensitive")):
        raise SystemExit("rel_w12_search: settings out of range: %r" % out)
    return out


def diamond_search(diamond, queries, db, s, out_lines):
    """Run DIAMOND blastp of `queries` against `db` and hand its output lines to out_lines(iterable); wall seconds."""
    cmd = [diamond, "blastp", "-q", queries, "-d", db, "-o", "/dev/stdout", "--" + s["sensitivity"], "-k", "0", "--max-hsps", "0", "-e", str(s["evalue"]),
           "--threads", str(s["threads"]), "-b", str(s["block_size"]), "-c", "1", "--tmpdir", os.environ.get("TMPDIR", "/tmp"), "--quiet", "--outfmt", "6"] + H.FIELDS
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1 << 20)
    try:
        result = out_lines(p.stdout)
        if p.wait() != 0:
            raise SystemExit("diamond exited with %d" % p.returncode)
    finally:
        if p.poll() is None:
            p.kill()
    return time.time() - t0, result


def makedb(diamond, records, faa, db):
    """A DIAMOND database of the records [(name, sequence)]; wall seconds."""
    t0 = time.time()
    with open(faa + ".part", "w") as f:
        for name, seq in records:
            f.write(">%s\n%s\n" % (name, seq))
    os.replace(faa + ".part", faa)
    subprocess.run([diamond, "makedb", "--in", faa, "-d", db[:-5], "--quiet"], check=True)
    return time.time() - t0


def chunks(order, sizes, limit):
    """Consecutive runs of `order` whose sizes add up to at most `limit` (an item larger than the limit alone)."""
    out, cur, n = [], [], 0
    for g in order:
        if cur and n + sizes[g] > limit:
            out.append(cur)
            cur, n = [], 0
        cur.append(g)
        n += sizes[g]
    if cur:
        out.append(cur)
    return [c for c in out if sum(sizes[g] for g in c) > 0]


def sample_genes(genomes, names, sample, seed):
    """{genome: uint8 mask of the genes searched}: all of them, or per group (the island chromosomes as a group of their
    own) a share `sample` of the group's proteins drawn at random (at least one per group)."""
    if sample >= 1:
        return {g["id"]: np.ones(len(names[g["id"]]), dtype=np.uint8) for g in genomes}
    out = {g["id"]: np.zeros(len(names[g["id"]]), dtype=np.uint8) for g in genomes}
    stratum = lambda g: "island" if g["split"] == "island" else g["group"]      # noqa: E731
    groups = sorted({stratum(g) for g in genomes})
    for k, grp in enumerate(groups):
        pool = [(g["id"], i) for g in genomes if stratum(g) == grp for i in range(len(names[g["id"]]))]
        if not pool:
            continue
        take = max(1, int(round(sample * len(pool))))
        for x in np.random.default_rng([int(seed), 12, k]).choice(len(pool), size=take, replace=False):
            gid, i = pool[int(x)]
            out[gid][i] = 1
    return out


def island_genomes(run, sealed):
    """([dict(id, group, split "island", taxid, ranks, folder, genes)], provenance) of the development chromosomes the
    island stage of the passed run `run` prepared, each accession checked (development, not sealed) and each file
    against the island manifest."""
    from ci import manifest as M
    from seeddrift.train.rel import sealed as SL
    ip = os.path.join(run, "data", "islands", "index.json")
    sha = W._output(run, "islands_data", "islands_index", ip)
    with open(ip) as f:
        idx = json.load(f)
    accs = [g["id"] for g in idx["genomes"]]
    common.development_chromosomes(accs)
    try:
        SL.check_ids(accs, sealed)
    except SL.SealedError as err:
        raise SystemExit("rel_w12_search: %s" % err)
    with open(os.path.join(run, "data", "islands", "manifest.json")) as f:
        msha = json.load(f)["sha256"]
    man = M.load(os.path.join(run, "manifests", msha + ".json"))
    want = {f["path"]: f["sha256"] for e in man["entries"] for f in e.get("files", [])}
    out, bad = [], []
    for g in idx["genomes"]:
        for n in ("genes.tsv", "proteins.faa"):
            p = os.path.join(g["folder"], n)
            if want.get(p) is None or sha256_file(os.path.join(run, p), use_sidecar=False) != want[p]:
                bad.append(p)
        out.append(dict(id=g["id"], group=g["group"], split="island", taxid=int(g["taxid"]), ranks=g.get("ranks"), folder=os.path.join(run, g["folder"]),
                        genes=int(g["genes"])))
    if bad:
        raise SystemExit("rel_w12_search: island files of %s differ from its manifest: %s" % (run, bad[:5]))
    return out, dict(run_id=os.path.basename(os.path.normpath(run)), index_sha256=sha, manifest=msha, chromosomes=len(out))


def build(st, cfg, run_dir):
    s, t0 = settings(cfg), time.time()
    say = lambda text: print("rel_w12_search: %s (%.0f s)" % (text, time.time() - t0), flush=True)
    diamond = cfg["tools"]["diamond"]
    src = W.source(s["source"], pairs=False)              # W12 reads the training data only
    sealed = common.sealed_set(cfg)
    metas, left, sizes = W.open_genomes(src, s, sealed)
    from ci.taxonomy import Taxonomy
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    st.extra["sealed_check"] = common.sealed_check(cfg, [W.entry(src, m) for m in metas], splits={m["id"]: m["split"] for m in metas}, sealed=sealed, tax=tax)
    st.extra.update(source=src["provenance"], split_sizes=sizes, left_out=left)
    run = src["run"]
    # the corpus files of the open genomes, against the source's corpus manifest
    paths = []
    for m in metas:
        files = {os.path.basename(f["path"]): f for f in src["corpus"][m["id"]].get("files") or []}
        for name in ("genes.tsv", "proteins.faa"):
            if name not in files:
                raise SystemExit("rel_w12_search: the corpus manifest names no %s of %s" % (name, m["id"]))
            paths.append((files[name]["path"], files[name]["sha256"]))
    bad = [p for (p, want), got in zip(paths, V.hash_files([os.path.join(run, p) for p, _ in paths], min(s["threads"], 8))) if got != want]
    if bad:
        raise SystemExit("rel_w12_search: corpus files of %s differ from its manifest: %s" % (run, bad[:5]))
    from ci.stages.search import gene_ids
    names, prots = {}, {}
    islands, island_prov = island_genomes(s["islands_from"], sealed) if s["islands"] else ([], None)
    for m in metas + islands:
        folder = m.get("folder") or os.path.join(run, "data", "corpus", m["id"])
        names[m["id"]] = gene_ids(folder)
        if len(names[m["id"]]) != int(m["genes"]):
            raise SystemExit("rel_w12_search: %s has %d genes in genes.tsv and %d in the training arrays" % (m["id"], len(names[m["id"]]), m["genes"]))
        fa = read_fasta(os.path.join(folder, "proteins.faa"))
        prots[m["id"]] = [fa["%s__%s" % (m["id"], n)] for n in names[m["id"]]]
    say("%d genomes and %d development chromosomes, %d proteins, files checked" % (len(metas), len(islands), sum(len(v) for v in prots.values())))
    genomes = [dict(id=m["id"], group=m["group"], split=m["split"], taxid=int(m["taxid"]), genes=len(names[m["id"]]),
                    letters=int(sum(len(x) for x in prots[m["id"]]))) for m in metas + islands]
    gix = {g["id"]: i for i, g in enumerate(genomes)}
    gene_index = {g: {n: i for i, n in enumerate(v)} for g, v in names.items()}
    searched = sample_genes(genomes, names, s["sample"], s["seed"])
    for g in genomes:
        g["searched"] = int(searched[g["id"]].sum())
        g["searched_letters"] = int(sum(len(prots[g["id"]][i]) for i in np.flatnonzero(searched[g["id"]])))
    out = os.path.join(run_dir, OUT)
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    write_json(os.path.join(out, "genomes.json"), dict(genomes=genomes, note="the genome list the arrays index (tgenome)"))
    train = [g["id"] for g in genomes if g["split"] == "train"]
    timing = dict(db_s=0.0, probe_s=None, main_s=[], self_s=[], self_db_s=[])
    db = os.path.join(parts, "db.dmnd")
    if not os.path.exists(db):
        timing["db_s"] = makedb(diamond, [("%s__%s" % (g, n), p) for g in train for n, p in zip(names[g], prots[g])], os.path.join(parts, "db.faa"), db)
        say("database of %d training genomes" % len(train))
    n_genes = {g["id"]: g["genes"] for g in genomes}

    def query_file(path, gids):
        with open(path + ".part", "w") as f:
            for g in gids:
                for i in np.flatnonzero(searched[g]):
                    f.write(">%s__%s\n%s\n" % (g, names[g][i], prots[g][i]))
        os.replace(path + ".part", path)

    if s["probe_proteins"]:                                           # DIAMOND's fixed cost of one call against the database
        pool = [(g, i) for g in [x["id"] for x in genomes] for i in np.flatnonzero(searched[g])]
        pick = sorted(int(x) for x in np.random.default_rng([s["seed"], 13]).choice(len(pool), size=min(s["probe_proteins"], len(pool)), replace=False))
        pq = os.path.join(parts, "probe.faa")
        with open(pq, "w") as f:
            for x in pick:
                g, i = pool[x]
                f.write(">%s__%s\n%s\n" % (g, names[g][i], prots[g][i]))
        timing["probe_s"], probe_rows = diamond_search(diamond, pq, db, s, lambda lines: sum(1 for _ in lines))
        timing["probe_proteins"], timing["probe_rows"] = len(pick), probe_rows
        say("probe: %d proteins, %d rows" % (len(pick), probe_rows))

    def done(marker):
        p = os.path.join(parts, marker + ".json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            d = json.load(f)
        ok = all(os.path.exists(os.path.join(parts, fn)) and sha256_file(os.path.join(parts, fn), use_sidecar=False) == sha for fn, sha in d["files"].items())
        return d if ok else None

    def run_calls(kind, batches, call):
        for k, gids in enumerate(batches):
            marker = "%s_%04d" % (kind, k)
            d = done(marker)
            if d is not None:
                timing[kind + "_s"].append(d["seconds"])
                continue
            spent = time.time() - t0
            est = max(timing[kind + "_s"] or [0.0])
            if s["wall_s"] and spent + est > s["wall_s"]:
                raise SystemExit("rel_w12_search: stopped before %s call %d of %d to stay within wall_s %.0f; resume the run" % (kind, k + 1, len(batches), s["wall_s"]))
            files, sec = call(k, gids)
            timing[kind + "_s"].append(sec)
            write_json(os.path.join(parts, marker + ".json"), dict(genomes=gids, seconds=sec,
                                                                   files={fn: sha256_file(os.path.join(parts, fn), use_sidecar=False) for fn in files}))
            say("%s call %d of %d: %d genomes, %.0f s" % (kind, k + 1, len(batches), len(gids), sec))

    def main_call(k, gids):
        q = os.path.join(parts, "main_%04d.faa" % k)
        query_file(q, gids)
        written = []

        def write(gid, arrays):
            H.save(os.path.join(parts, gid + ".main.npz"), arrays)
            written.append(gid)
        keep = lambda qg, qi, tg, tj: not (tg == qg and tj == qi)   # noqa: E731 - the query gene itself
        sec, _ = diamond_search(diamond, q, db, s, lambda lines: H.stream(lines, gix, gene_index, keep, write, n_genes, searched))
        for g in gids:
            if g not in written:
                H.save(os.path.join(parts, g + ".main.npz"), H.empty(n_genes[g], searched[g]))
        os.remove(q)
        return [g + ".main.npz" for g in gids], sec

    def self_call(k, gids):
        sdb = os.path.join(parts, "self_%04d.dmnd" % k)
        dbs = makedb(diamond, [("%s__%s" % (g, n), p) for g in gids for n, p in zip(names[g], prots[g])], os.path.join(parts, "self_%04d.faa" % k), sdb)
        timing["self_db_s"].append(dbs)
        q = os.path.join(parts, "self_%04d.q.faa" % k)
        query_file(q, gids)
        written = []

        def write(gid, arrays):
            H.save(os.path.join(parts, gid + ".self.npz"), arrays)
            written.append(gid)
        keep = lambda qg, qi, tg, tj: tg == qg and tj != qi          # noqa: E731 - the own genome only, not the gene itself
        sec, _ = diamond_search(diamond, q, sdb, s, lambda lines: H.stream(lines, gix, gene_index, keep, write, n_genes, searched))
        for g in gids:
            if g not in written:
                H.save(os.path.join(parts, g + ".self.npz"), H.empty(n_genes[g], searched[g]))
        for x in (q, sdb, os.path.join(parts, "self_%04d.faa" % k)):
            os.remove(x)
        return [g + ".self.npz" for g in gids], sec + dbs

    order = [g["id"] for g in genomes]
    run_calls("main", chunks(order, {g: int(searched[g].sum()) for g in order}, s["chunk_proteins"]), main_call)
    held = [g["id"] for g in genomes if g["split"] in ("heldout", "island")]           # not in the database: their own proteins apart
    run_calls("self", chunks(held, {g: len(names[g]) if searched[g].any() else 0 for g in held}, s["self_batch_proteins"]), self_call)
    # every query genome: its rows of the main search, and of its own proteins for a held-out genome
    files, stats = [], dict(pairs=0, hsps=0, self_pairs=0, genes_searched=0, genes_with_pairs=0)
    for g in genomes:
        mp_ = os.path.join(parts, g["id"] + ".main.npz")
        a = H.load(mp_) if os.path.exists(mp_) else H.empty(g["genes"], searched[g["id"]])     # a genome with no gene searched
        if g["split"] in ("heldout", "island") and os.path.exists(os.path.join(parts, g["id"] + ".self.npz")):
            a = H.merge(a, H.load(os.path.join(parts, g["id"] + ".self.npz")))
        bad = H.check(a, len(genomes))
        if bad:
            raise SystemExit("rel_w12_search: the rows of %s are not sound: %s" % (g["id"], bad))
        others = a["tgenome"] != gix[g["id"]]
        if g["split"] != "train" and others.any() and set(np.unique(a["tgenome"][others]).tolist()) - {gix[t] for t in train}:
            raise SystemExit("rel_w12_search: %s has rows to genomes outside the training split" % g["id"])
        p = os.path.join(out, g["id"] + ".npz")
        H.save(p, a)
        g.update(pairs=int(len(a["tgenome"])), hsps=int(len(a["qs"])), self_pairs=int((~others).sum()))
        for k in ("pairs", "hsps", "self_pairs"):
            stats[k] += g[k]
        stats["genes_searched"] += g["searched"]
        stats["genes_with_pairs"] += int((np.diff(a["off"]) > 0).sum())
        files.append(p)
    gp = os.path.join(out, "genomes.json")
    write_json(gp, dict(genomes=genomes, note="the genome list the arrays index (tgenome)"))
    files.append(gp)
    shutil.rmtree(parts)
    # the lookup index of every query genome's rows (memory-mapped lookups by query genome, target genome, query gene)
    ix_dir = os.path.join(out, "index")
    index_sizes = H.write_index(ix_dir, [(lambda gid=g["id"]: H.load(os.path.join(out, gid + ".npz"))) for g in genomes])
    index_files = [os.path.join(ix_dir, k + ".npy") for k in H.INDEX_FIELDS]
    files += index_files
    say("lookup index: %d genes, %d pairs, %d alignments" % (index_sizes["genes"], index_sizes["pairs"], index_sizes["hsps"]))
    entries = [dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)) for p in files]
    index_bytes = sum(e["bytes"] for e in entries if os.path.basename(os.path.dirname(e["path"])) == "index")
    nbytes = sum(e["bytes"] for e in entries)
    total_letters = sum(g["letters"] for g in genomes)
    searched_letters = sum(g["searched_letters"] for g in genomes)
    timing.update(main_total_s=float(sum(timing["main_s"])), self_total_s=float(sum(timing["self_s"])), wall_s=round(time.time() - t0, 1))
    rep = dict(format="rel-w12-search", version=1, note=__doc__.strip().splitlines()[0], source=src["provenance"],
               settings={k: s[k] for k in sorted(s) if k not in ("max_left_out",)},
               database=dict(genomes=len(train), proteins=int(sum(len(prots[g]) for g in train)), letters=int(sum(len(x) for g in train for x in prots[g]))),
               queries=dict(genomes=len(genomes), proteins=int(sum(g["genes"] for g in genomes)), letters=total_letters, searched_proteins=stats["genes_searched"],
                            searched_letters=searched_letters, per_split={sp: sum(1 for g in genomes if g["split"] == sp) for sp in V.OPEN_SPLITS + ("island",)}),
               islands=island_prov,
               index=dict(index_sizes, folder=os.path.relpath(ix_dir, run_dir), bytes=index_bytes),
               rows=dict(stats, bytes=nbytes, index_bytes=index_bytes, bytes_per_hsp=(nbytes / stats["hsps"]) if stats["hsps"] else None,
                         hsps_per_searched_gene=(stats["hsps"] / stats["genes_searched"]) if stats["genes_searched"] else None),
               timing=timing, left_out=[dict(group=x["group"], split=x["split"], reasons=x["reasons"]) for x in left],
               sealed_check=dict(passed=True, **st.extra["sealed_check"]))
    free = shutil.disk_usage(run_dir).free
    rep["disk"] = dict(free_gb=round(free / 1e9, 1), reserve_gb=s["reserve_gb"])
    if s["sample"] < 1 and searched_letters:
        f = total_letters / searched_letters
        calls_full = math.ceil(rep["queries"]["proteins"] / s["chunk_proteins"])
        fixed = timing["probe_s"] or 0.0
        main_marginal = max(0.0, timing["main_total_s"] - fixed * len(timing["main_s"]))
        self_calls = len(timing["self_s"])
        self_fixed = min(fixed, timing["self_total_s"] / self_calls) if self_calls else 0.0
        self_marginal = max(0.0, timing["self_total_s"] - self_fixed * self_calls)
        est_bytes = nbytes * f
        rep["extrapolation"] = dict(
            factor_letters=f, note="linear in the query letters; DIAMOND's fixed cost per call (the probe) counted once per call",
            main_calls=calls_full, main_s=fixed * calls_full + main_marginal * f, self_calls=self_calls, self_s=self_fixed * self_calls + self_marginal * f,
            hsps=stats["hsps"] * f, pairs=stats["pairs"] * f, bytes=est_bytes,
            disk_needed_gb=round(est_bytes * 2 / 1e9 + 1.0, 1), disk_fits=bool(est_bytes * 2 + 1e9 + s["reserve_gb"] * 1e9 < free))
        e = rep["extrapolation"]
        e["total_h"] = round((e["main_s"] + e["self_s"] + timing["db_s"]) / 3600, 2)
        e["segments_of_3h"] = int(math.ceil(e["total_h"] / 3.0)) if e["total_h"] else 0
    page = os.path.join(out, "report.md")
    with open(page + ".part", "w") as fh:
        fh.write(page_of(rep))
    os.replace(page + ".part", page)
    rep["files"] = entries + [dict(path=os.path.relpath(page, run_dir), sha256=sha256_file(page, use_sidecar=False), bytes=os.path.getsize(page))]
    mp = os.path.join(out, "manifest.json")
    write_json(mp, rep)
    st.output("search_manifest", mp)
    st.extra.update(rows=rep["rows"], timing=timing, extrapolation=rep.get("extrapolation"))
    return rep


def _h(x):
    return "n/a" if x is None else "%.2f h" % (x / 3600)


def page_of(rep):
    q, r, t = rep["queries"], rep["rows"], rep["timing"]
    L = ["# The uncapped search (package W12)", "", rep["note"], "",
         "Source run `%s`. Database: %d training genomes, %s proteins. Queries: %d genomes (%s), %s proteins, %s of them searched in this run (%.2f%% of the "
         "letters)." % (rep["source"]["run_id"], rep["database"]["genomes"], "{:,}".format(rep["database"]["proteins"]), q["genomes"],
                        ", ".join("%s %d" % kv for kv in q["per_split"].items()), "{:,}".format(q["proteins"]), "{:,}".format(q["searched_proteins"]),
                        100.0 * q["searched_letters"] / max(q["letters"], 1)), "",
         "Settings: %s, -k 0, --max-hsps 0, e-value %g, %d threads, block size %g, %d query proteins per call." % (
             rep["settings"]["sensitivity"], rep["settings"]["evalue"], rep["settings"]["threads"], rep["settings"]["block_size"], rep["settings"]["chunk_proteins"]), "",
         "| | value |", "|---|---|",
         "| database build | %.0f s |" % t["db_s"],
         "| probe (fixed cost of one call) | %s |" % ("n/a" if t.get("probe_s") is None else "%.0f s (%d proteins, %d rows)" % (t["probe_s"], t["probe_proteins"], t["probe_rows"])),
         "| main search | %.0f s in %d calls |" % (t["main_total_s"], len(t["main_s"])),
         "| own-genome search of held-out genomes | %.0f s in %d calls |" % (t["self_total_s"], len(t["self_s"])),
         "| pairs (query gene, target gene) | %s, %s of them to the own genome |" % ("{:,}".format(r["pairs"]), "{:,}".format(r["self_pairs"])),
         "| alignments | %s, %.1f per searched gene |" % ("{:,}".format(r["hsps"]), r["hsps_per_searched_gene"] or 0.0),
         "| searched genes with at least one pair | %s of %s |" % ("{:,}".format(r["genes_with_pairs"]), "{:,}".format(r["genes_searched"])),
         "| bytes on disk | %.2f GB, %.1f per alignment (the lookup index %.2f GB of it) |" % (r["bytes"] / 1e9, r["bytes_per_hsp"] or 0.0, r.get("index_bytes", 0) / 1e9),
         "| free disk | %.0f GB (reserve %.0f GB) |" % (rep["disk"]["free_gb"], rep["disk"]["reserve_gb"])]
    e = rep.get("extrapolation")
    if e:
        L += ["", "## The full search, extrapolated linearly", "", e["note"] + ".", "",
              "| | estimate |", "|---|---|",
              "| main search | %s in %d calls |" % (_h(e["main_s"]), e["main_calls"]),
              "| own-genome search | %s |" % _h(e["self_s"]),
              "| total | %.2f h, %d segments of at most 3 h |" % (e["total_h"], e["segments_of_3h"]),
              "| alignments | %s |" % "{:,}".format(int(e["hsps"])),
              "| pairs | %s |" % "{:,}".format(int(e["pairs"])),
              "| disk | %.1f GB of output, %.1f GB needed while it is written; %s |" % (e["bytes"] / 1e9, e["disk_needed_gb"], "fits" if e["disk_fits"] else "DOES NOT FIT")]
    L += ["", "Development diagnosis. The sealed test split was not opened; no score or rank is stored.", ""]
    return "\n".join(L)


def gate(st, cfg, run_dir):
    s = settings(cfg)
    p = os.path.join(run_dir, OUT, "manifest.json")
    with open(p) as f:
        rep = json.load(f)
    problems = []
    for e in rep.get("files") or []:
        full = os.path.join(run_dir, e["path"])
        if not os.path.exists(full) or os.path.getsize(full) != e["bytes"] or sha256_file(full, use_sidecar=False) != e["sha256"]:
            problems.append(dict(file=e["path"], error="missing or not the file the manifest names"))
    with open(os.path.join(run_dir, OUT, "genomes.json")) as f:
        genomes = json.load(f)["genomes"]
    src = W.source(s["source"], pairs=False)              # W12 reads the training data only
    want = sorted(m["id"] for m in src["index"]["genomes"] if m.get("source", "corpus") == "corpus" and m["split"] in V.OPEN_SPLITS)
    have = sorted(g["id"] for g in genomes if g["split"] != "island")
    if any(g["split"] not in V.OPEN_SPLITS + ("island",) for g in genomes) or not set(have) <= set(want) or len(want) - len(have) != len(rep.get("left_out") or []):
        problems.append(dict(error="the query genomes are not the open splits of the source (less the ones left out)"))
    isl = [g["id"] for g in genomes if g["split"] == "island"]
    if isl and not s["islands"]:
        problems.append(dict(error="island chromosomes among the queries of a run without islands"))
    if isl:
        try:
            common.development_chromosomes(isl)
        except SystemExit as err:
            problems.append(dict(error=str(err)[:300]))
    if rep.get("database", {}).get("genomes") != sum(1 for g in genomes if g["split"] == "train"):
        problems.append(dict(error="the database is not the training genomes"))
    listed = {os.path.basename(e["path"]) for e in rep.get("files") or []}
    for g in genomes:
        if g["id"] + ".npz" not in listed:
            problems.append(dict(genome=g["id"], error="no rows file"))
            continue
        a = H.load(os.path.join(run_dir, OUT, g["id"] + ".npz"))
        bad = H.check(a, len(genomes))
        if bad or len(a["off"]) - 1 != g["genes"] or int(a["searched"].sum()) != g["searched"] or len(a["qs"]) != g["hsps"]:
            problems.append(dict(genome=g["id"], error="rows not sound or not as listed", problems=bad[:3]))
    ix = rep.get("index") or {}
    try:
        idx = H.Index(os.path.join(run_dir, ix.get("folder", os.path.join(OUT, "index"))))
        gb = np.asarray(idx.a["gene_base"])
        if len(gb) != len(genomes) + 1 or list(np.diff(gb)) != [g["genes"] for g in genomes] or len(idx.a["qs"]) != sum(g["hsps"] for g in genomes):
            problems.append(dict(error="the lookup index does not hold every query genome's rows"))
        for qg in range(0, len(genomes), max(1, len(genomes) // 20)):    # a sample of genomes: their pairs as in their own files
            a = H.load(os.path.join(run_dir, OUT, genomes[qg]["id"] + ".npz"))
            k0 = int(gb[qg])
            if not np.array_equal(np.asarray(idx.a["off"][k0:k0 + genomes[qg]["genes"] + 1]) - int(idx.a["off"][k0]), a["off"]):
                problems.append(dict(genome=genomes[qg]["id"], error="the lookup index differs from the genome's rows"))
    except (OSError, ValueError, KeyError) as err:
        problems.append(dict(error="the lookup index cannot be read: %s" % err))
    require(GateResult("G4_rel_w12_search", "fail" if problems else "pass", dict(problems=problems[:50], genomes=len(genomes), rows=rep.get("rows"),
                                                                                 extrapolation=rep.get("extrapolation"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "manifest.json")) as f:
        rep = json.load(f)
    return dict(json=dict(rows=rep["rows"], timing={k: v for k, v in rep["timing"].items() if not isinstance(v, list)}, extrapolation=rep.get("extrapolation"),
                          queries=rep["queries"], database=rep["database"], disk=rep["disk"]),
                md=page_of(rep).splitlines()[2:])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w12_search") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
