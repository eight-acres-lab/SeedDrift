"""Stage rel_w18_rulers (package W18; pipeline rel_tables): the implant tables, the first ruler set of the relational line
(bench/rel/implant.py). `python -m ci.stages.rel.w18_rulers RUN_DIR`. An evaluation stage: the set is kind ruler and
purpose ruler, never a training table and never an item of a selection round.

Inputs: the passed selfsup run the tables of version 0 came from (its training arrays with the three-way split and its
homolog rows of stage pairs, every file checked against its records, as stage w2_v0 takes them), and a passed run of
stage w2_dist on the same source (the distances between genomes, every file checked against its report and the report
against its stage record). Only genomes of the train and held-out splits are opened; the sealed test split is neither
read nor named. The sealed check runs on every open genome before any is opened.

Eligible as the examined genome A and as the donor D: held-out genomes of the standard genetic code, that is the
translation table their corpus source declares and the nuclear code the NCBI taxonomy gives their taxon are both 1 or
11 (the same codon to amino-acid map). Other rows are training genomes only.

Profile block rel_w18_rulers:
  version            0 (default): homolog rows of stage pairs of the source, tables of version 0; 1: the homolog rows of
                     a passed full run of stage w12_search (search_from) on the same source, the fields of version 1
  search_from        (version 1) the passed full run of stage w12_search; its rows files are checked against its
                     manifest as they are read; the source is the one the search read (its manifest's settings, which
                     may be a corpus of stage w13_build), and source, when given, must be that one
  source             the passed selfsup run (version 0)
  dist               the passed run of stage w2_dist on the same source (version 0; version 1 without it computes the
                     distances here, as stage w12_v1 does: the same k-mers, k and sketch size)
  expect             optional {split: genomes}
  rows               rows per table (16)
  per_tier           triples per eligible A and donor tier (2); group_tier_cap: at most this many per (group, tier) (30)
  scenarios          true: the copies of scenario donor_unsampled too
  edges              the tier edges of the distance ([0.05, 0.2, 0.5, 1.0]; the protocol's)
  k                  the k-mer length of the distance matrix read (the distance run's primary k, 6)
  min_codons, length_tol, attempts, donor_cap, min_aligned, aligned_share    the generator's draws (64, 0.15, 40, 3, 10, 0.3)
  shard_tables, processes, max_left_out
Outputs under data/rel_w18/rulers/: tables/ (the shards: the contract's fields written by W2's writer, and beside each
shard W18's meta fields <shard>.w18_<name>.npy, role meta), index.json (the genome list, groups, edges, the triples and
their draws, the genomes left out by the genetic code) and tableset.json (the manifest, written last).
Gate G4_rel_w18_rulers: the manifest is sound and of this stage, every table is one of a complete triple, examined
genomes and donors are eligible held-out genomes, other rows training genomes, the three tables of a triple share their
rows and every row has content on the target block, and no genome outside the open splits is named.
"""
from __future__ import annotations

import gzip
import json
import os
import time

import numpy as np
import yaml

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.kingdom import KingdomMap
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from seeddrift.train import selfsup as SS
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.tables import v0_build as V
from seeddrift.train.rel.tables import v0_io as IO

OUT = os.path.join("data", "rel_w18", "rulers")
KEYS = {"version", "search_from", "source", "dist", "expect", "rows", "per_tier", "group_tier_cap", "scenarios", "edges", "k", "min_codons", "length_tol", "attempts",
        "donor_cap", "min_aligned", "aligned_share", "shard_tables", "processes", "max_left_out"}
COVERAGE_NOTE = ("Implant tables (ruler, evaluation only): the target block of a held-out genome A replaced by a gene of another held-out genome D of "
                 "the standard code, with two controls at the same locus (A's own gene, another gene of A). Rows are training genomes with a homolog row "
                 "on all three target genes; before the corpus is searched again without caps, having such a row is decided by a similarity ranking, "
                 "and the stored columns come from alignments anchored on the target gene, so readings on these tables are preliminary.")
NOT_COVERED = ["donors of the training split", "close relatives (same species or genus) beyond what the medium corpus holds", "non-standard genetic codes",
               "intergenic sequence, partial genes, implants longer than one target block", "genes without homolog rows in the training genomes",
               "the sealed test split"]


def settings(cfg) -> dict:
    c = cfg.get("rel_w18_rulers") or {}
    unknown = set(c) - KEYS
    v1 = int(c.get("version", 0)) == 1
    if unknown or (not v1 and (not c.get("source") or not c.get("dist"))):
        raise SystemExit("rel_w18_rulers: needs source and dist (version 0) or search_from (version 1); unknown keys %s" % sorted(unknown))
    out = dict(version=int(c.get("version", 0)), search_from=c.get("search_from"), source=str(c.get("source") or ""), dist=str(c.get("dist") or ""), expect={str(k): int(v) for k, v in (c.get("expect") or {}).items()},
               rows=int(c.get("rows", 16)), per_tier=int(c.get("per_tier", 2)), group_tier_cap=int(c.get("group_tier_cap", 30)),
               scenarios=bool(c.get("scenarios", True)), min_aligned=int(c.get("min_aligned", 10)), aligned_share=float(c.get("aligned_share", 0.3)),
               edges=[float(x) for x in c.get("edges") or [0.05, 0.2, 0.5, 1.0]], k=int(c.get("k", 6)), min_codons=int(c.get("min_codons", 64)),
               length_tol=float(c.get("length_tol", 0.15)), attempts=int(c.get("attempts", 40)), donor_cap=int(c.get("donor_cap", 3)),
               shard_tables=int(c.get("shard_tables", 720)), processes=int(c.get("processes", 4)), max_left_out=int(c.get("max_left_out", 20)),
               seed=int(cfg.get("seed", 1)))
    if out["version"] not in (0, 1) or (out["version"] == 1 and not out["search_from"]):
        raise SystemExit("rel_w18_rulers: version is 0 or 1, and version 1 needs search_from")
    if not (2 <= out["rows"] <= 64 and 1 <= out["per_tier"] <= 10 and 1 <= out["group_tier_cap"] <= 1000 and out["edges"] == sorted(out["edges"])
            and 1 <= out["min_codons"] <= 256 and 1 <= out["min_aligned"] and 0 <= out["aligned_share"] < 1
            and 0 < out["length_tol"] < 1 and 1 <= out["attempts"] <= 1000 and 1 <= out["donor_cap"] and 3 <= out["shard_tables"] <= 20000
            and out["shard_tables"] % 3 == 0 and 1 <= out["processes"] <= 16):
        raise SystemExit("rel_w18_rulers: settings out of range: %r" % out)
    return out


def distance_run(run, k):
    """(genome ids, [G, G] distances, the report) of a passed run of stage w2_dist, every file checked."""
    p = os.path.join(run, "data", "rel_w2", "dist", "report.json")
    W._output(run, "rel_w2_dist", "dist_report", p)
    with open(p) as f:
        rep = json.load(f)
    for e in rep.get("files") or []:
        full = os.path.join(run, e["path"])
        if not os.path.exists(full) or sha256_file(full, use_sidecar=False) != e["sha256"]:
            raise SystemExit("rel_w18_rulers: %s is not the file the distance report names" % e["path"])
    if int(k) not in [int(x) for x in rep.get("ks") or []]:
        raise SystemExit("rel_w18_rulers: the distance run has no k = %d (it has %s)" % (k, rep.get("ks")))
    with open(os.path.join(run, "data", "rel_w2", "dist", "genomes.json")) as f:
        ids = [g["id"] for g in json.load(f)["genomes"]]
    dist = np.load(os.path.join(run, "data", "rel_w2", "dist", "distances_k%d.npy" % int(k)))
    if dist.shape != (len(ids), len(ids)):
        raise SystemExit("rel_w18_rulers: the distance matrix does not fit its genome list")
    return ids, dist, rep


def declared_tables(spec_path):
    """{genome id: the translation table its corpus source declares} (corpus.py's defaults: 1 for Ensembl coding
    sequences, 11 otherwise)."""
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    out = {}
    for s in spec.get("sources") or []:
        out[str(s["id"])] = int(s.get("table", 1 if s.get("source_kind") == "annotated_cds" else 11))
    return out


def nuclear_codes(taxdump, taxids, tax=None):
    """{taxid: the nuclear genetic code NCBI gives the taxon (nodes.dmp, column 7)}; merged taxids resolved."""
    want = {int(t): (tax.resolve(int(t)) if tax is not None else int(t)) for t in taxids}
    need = set(want.values())
    path = os.path.join(taxdump, "nodes.dmp")
    opener = (lambda: gzip.open(path + ".gz", "rt")) if not os.path.exists(path) and os.path.exists(path + ".gz") else (lambda: open(path))
    got = {}
    with opener() as f:
        for line in f:
            parts = line.split("\t|\t", 7)
            t = int(parts[0])
            if t in need:
                got[t] = int(parts[6]) if len(parts) > 6 and parts[6].strip().isdigit() else -1     # -1: the dump gives no code
    return {t: got.get(r, -1) for t, r in want.items()}


def eligibility(metas, declared, codes):
    """(eligible held-out ids, {group: genomes left out by the code}) for A and D."""
    ok, left = set(), {}
    for m in metas:
        if m["split"] != "heldout":
            continue
        dt, nc = declared.get(m["id"], -1), codes.get(int(m["taxid"]), -1)
        if dt in (1, 11) and nc in (1, 11):
            ok.add(m["id"])
        else:
            left.setdefault(m["group"], []).append(dict(id=m["id"], declared=dt, ncbi=nc))
    return ok, left


def source_of(s):
    """(the source run's records, the search run's (version 1) or None): version 0 reads source; version 1 the source
    the search read, which source, when given, must be."""
    if s["version"] == 1:
        from ci.stages.rel import w12_v1
        smeta = w12_v1.search_run(s["search_from"])
        path = str((smeta[0].get("settings") or {}).get("source", ""))
        if s["source"] and os.path.normpath(path) != os.path.normpath(s["source"]):
            raise SystemExit("rel_w18_rulers: the search run read another source (%s)" % path)
        return W.source(path, pairs=False), smeta
    return W.source(s["source"]), None


def search_source(s, smeta, metas):
    """(the search's genome list, rows_of, provenance) of the full search run s["search_from"] (version 1; smeta its
    w12_v1.search_run): every open genome is in its list with its split, and every rows file is checked against the
    search's manifest when it is read."""
    from ci.stages.rel import w12_v1
    from seeddrift.train.rel.tables import v1_hits as H
    man, sgen, files, prov = smeta
    split = {g["id"]: g["split"] for g in sgen}
    bad = [m["id"] for m in metas if split.get(m["id"]) != m["split"]]
    if bad:
        raise SystemExit("rel_w18_rulers: open genomes missing from the search's list or with another split: %s" % bad[:5])
    checked = {}

    def rows_of(gid):
        rel = os.path.join(w12_v1.SEARCH, gid + ".npz")
        full = os.path.join(s["search_from"], rel)
        if gid not in checked:
            if rel not in files or sha256_file(full, use_sidecar=False) != files[rel]["sha256"]:
                raise SystemExit("rel_w18_rulers: %s is not the rows file the search's manifest names" % rel)
            checked[gid] = True
        return H.load(full)
    return sgen, rows_of, prov


def mask_digests(out, meta):
    """({triple: the sha256 of the cells its passes hide}, [triples whose three tables differ]): the masks are hashed on
    the triple, so the three tables of a triple hide the same cells (the target blocks have one length)."""
    from seeddrift.train.rel.readouts import w18_passes as P
    got, bad = {}, []
    for t in range(len(out["tok"])):
        tab = {k: out[k][t] for k in P.FIELDS}
        d = P.mask_digest(tab, meta["row_tier"][t], t, int(meta["parent"][t]))
        s = int(meta["triple"][t])
        if got.setdefault(s, d) != d and s not in bad:
            bad.append(s)
    return got, bad


def tier_expectation(capacity, groups, edges):
    """The protocol's pre-declaration per donor tier: tiers 0 and 1 "expected too few" (the split keeps close clusters
    on one side); tier 2 as well unless every block has at least 3 groups with at least 5 eligible A that have a
    candidate donor in it; the other tiers "expected readable"."""
    from seeddrift.train.rel.evalh import score as SC
    out = {}
    for t in range(len(edges) + 2):
        if t <= 1:
            out[str(t)] = "expected too few"
            continue
        ok = all(sum(1 for g in groups if SC.block_of(g) == block and capacity.get("%s|%d" % (g, t), 0) >= 5) >= 3 for block in SC.BLOCKS)
        out[str(t)] = "expected readable" if (t > 2 or ok) else "expected too few"
    return out


def write_meta(path, meta):
    """W18's meta of every table, in the order of the set, as one uncompressed .npz (the evaluator's reader of shards
    reads <shard>.<field>.npy files only, so this file never reaches a table reader; readers of the ruler set take it
    through seeddrift/train/rel/readouts/w18_passes.load_meta)."""
    with open(path + ".part", "wb") as f:
        np.savez(f, **{k: np.ascontiguousarray(v) for k, v in meta.items()})
    os.replace(path + ".part", path)
    return path


def build(st, cfg, run_dir):
    from bench.rel import implant as I
    from ci.taxonomy import Taxonomy
    s, t0, say = settings(cfg), time.time(), lambda text: print("rel_w18_rulers: " + text, flush=True)
    src, smeta = source_of(s)
    run = src["run"]
    groups = sorted(KingdomMap().names)
    sealed = common.sealed_set(cfg)
    metas, left, sizes = W.open_genomes(src, dict(s, expect=s["expect"]), sealed)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    ents = {m["id"]: W.entry(src, m) for m in metas}
    st.extra["sealed_check"] = common.sealed_record(cfg, list(ents.values()), splits={m["id"]: m["split"] for m in metas}, sealed=sealed, tax=tax)
    st.extra.update(source=src["provenance"], split_sizes=sizes, left_out=left)
    if s["dist"]:
        dist_ids, dist, dist_rep = distance_run(s["dist"], s["k"])
        if dist_rep.get("source") != src["provenance"]:
            raise SystemExit("rel_w18_rulers: the distance run was made on another source")
    else:
        dist_ids, dist, dist_rep = None, None, None                   # version 1: computed below, once the arrays are loaded
    declared = declared_tables(st.repo(cfg["corpus_spec"]))
    codes = nuclear_codes(st.repo(cfg["taxdump"]), {int(m["taxid"]) for m in metas}, tax)
    eligible, by_code = eligibility(metas, declared, codes)
    say("%d held-out genomes of the standard code eligible as A and D; left out by the code: %s" % (len(eligible), {g: len(v) for g, v in by_code.items()}))
    paths = [os.path.join("data", d, m["id"] + ".npz") for d in (("train_data", "pairs") if s["version"] == 0 else ("train_data",)) for m in metas]
    shas = V.hash_files([os.path.join(run, p) for p in paths], s["processes"])
    bad = [p for p, sha in zip(paths, shas) if src["files"].get(p) != sha]
    if bad:
        raise SystemExit("rel_w18_rulers: files of %s differ from its manifests: %s" % (run, bad[:5]))
    genomes, _ = V.load_genomes(os.path.join(run, "data", "train_data"), metas, s["processes"])
    say("%d files checked, %d genomes loaded, %.0f s" % (len(paths), len(genomes), time.time() - t0))
    if dist is None:
        from ci.stages.rel import w12_v1
        dist_ids = [m["id"] for m in metas]
        dist = w12_v1.distances(genomes, [dict(id=x) for x in dist_ids], dict(dist_k=s["k"], dist_size=4096, processes=s["processes"]))
        say("distances of %d genomes computed, %.0f s" % (len(dist_ids), time.time() - t0))
    ids = [m["id"] for m in metas]
    train = [m["id"] for m in metas if m["split"] == "train"]
    units = I.split_units(metas)
    draws = dict(rows=s["rows"], min_codons=s["min_codons"], length_tol=s["length_tol"], attempts=s["attempts"], donor_cap=s["donor_cap"],
                 min_aligned=s["min_aligned"], aligned_share=s["aligned_share"])
    if s["version"] == 0:
        imp = I.Implanter(SS.Pairs(run), genomes, train, ids, dist, dist_ids, s["edges"], eligible, units, **draws)
    else:
        sgen, rows_of, st.extra["search"] = search_source(s, smeta, metas)
        imp = I.ImplanterV1(sgen, genomes, rows_of, train, ids, dist, dist_ids, s["edges"], eligible, units, **draws)
    out, meta, triples, counts, capacity = imp.build(metas, groups, s["seed"], per_tier=s["per_tier"], group_tier_cap=s["group_tier_cap"],
                                                     scenarios=s["scenarios"], progress=say)
    gix = {g: i for i, g in enumerate(ids)}
    problems = I.verify(out, meta, {gix[g] for g in eligible}, {gix[g] for g in train}, {gix[g]: u for g, u in units.items()})
    digests, unequal = mask_digests(out, meta)
    if unequal:
        problems.append("triples whose three tables hide different cells: %s" % unequal[:5])
    if problems:
        raise SystemExit("rel_w18_rulers: the implant tables are not sound: %s" % problems[:10])
    for t in triples:
        t["mask_sha256"] = digests[t["triple"]]
    say("%d triples (%d copies of scenario donor_unsampled), %d tables, %.0f s" % (len(triples), counts["copies"], len(out["tok"]), time.time() - t0))
    base = os.path.join(run_dir, OUT)
    folder = os.path.join(base, "tables")
    shards = IO.write(folder, out, s["shard_tables"], s["version"])
    files = []
    for name, _ in shards:
        for field in IO.fields(s["rows"], s["version"]):
            p = os.path.join(folder, "%s.%s.npy" % (name, field))
            files.append(dict(path=os.path.relpath(p, run_dir), role="shard" if field in IO.model_fields(s["version"]) else "meta"))
    files.append(dict(path=os.path.relpath(write_meta(os.path.join(base, "w18_meta.npz"), meta), run_dir), role="meta"))
    for f, sha in zip(files, V.hash_files([os.path.join(run_dir, f["path"]) for f in files], s["processes"])):
        f.update(sha256=sha, bytes=os.path.getsize(os.path.join(run_dir, f["path"])))
    used = sorted({int(x) for x in np.unique(out["genome"]) if x >= 0} | {int(x) for x in np.unique(meta["donor"])})
    present = sorted({t["group"] for t in triples})
    tver = I.TABLE_VERSION + ("-v1" if s["version"] else "")
    index = dict(format="rel-w18-rulers-index", version=1, table_version=tver, tables_of_version=s["version"], api_version=V.API_VERSION, rows=s["rows"], groups=groups,
                 edges=s["edges"], tiers=["<%g" % s["edges"][0]] + ["%g-%g" % (a, b) for a, b in zip(s["edges"], s["edges"][1:])] + [">=%g" % s["edges"][-1], "no shared k-mer"],
                 k=s["k"], variants=list(I.VARIANTS), meta_fields={k: v[2] for k, v in I.FIELDS.items()}, settings=s, counts=counts,
                 triples=[dict(t, a=gix[t["a"]], d=gix[t["d"]]) for t in triples], units=imp.unit_names, capacity=capacity,
                 tier_expectation=tier_expectation(capacity, groups, s["edges"]),
                 left_out_by_code={g: len(v) for g, v in by_code.items()}, code_detail=by_code, eligible=sorted(gix[g] for g in eligible),
                 meta_file="w18_meta.npz",
                 genomes=[dict(id=m["id"], group=m["group"], split=m["split"], taxid=int(m["taxid"])) for m in metas],
                 distance_run=(dict(run_id=os.path.basename(os.path.normpath(s["dist"])), report_sha256=sha256_file(os.path.join(s["dist"], "data", "rel_w2", "dist", "report.json"), use_sidecar=False))
                               if s["dist"] else dict(computed="here, as stage w12_v1 computes them", k=s["k"], size=4096)))
    ip = os.path.join(base, "index.json")
    write_json(ip, index)
    files.append(dict(path=os.path.relpath(ip, run_dir), role="index", sha256=sha256_file(ip, use_sidecar=False), bytes=os.path.getsize(ip)))
    st.output("w18_rulers_index", ip)
    n = len(out["tok"])
    part = dict(split=V.SPLITS["heldout"], tables=n, folder=os.path.relpath(folder, run_dir), shards=[dict(name=nm, tables=m) for nm, m in shards],
                sha256=sha256_json(sorted([f["path"], f["sha256"]] for f in files if f["role"] != "index")))
    doc = handoff.finish(dict(
        format="rel-tableset", format_version=1, name="w18-implant%s-r%d-s%d" % ("-v1" if s["version"] else "", s["rows"], s["seed"]), kind=api.TABLE_RULER, purpose="ruler",
        api_version=V.API_VERSION, table_version=tver, tables_of_version=s["version"], layout=api.LAYOUT_STORED,
        coverage=dict(groups=present, scenarios=["an implant at the locus of a held-out gene, from a held-out donor of the standard code, every donor distance the held-out split offers"],
                      not_covered=NOT_COVERED, note=COVERAGE_NOTE),
        tables=n, rows=s["rows"], seed=s["seed"], parts=dict(ruler=part), files=files, group_names=groups,
        genomes=[dict(id=ids[i], taxid=int(metas[i]["taxid"]), split=metas[i]["split"]) for i in used],
        left_out=[dict(group=x["group"], split=x["split"], reasons=x["reasons"]) for x in left], source=src["provenance"],
        distance_run=index["distance_run"], sealed_check=st.extra["sealed_check"], index=os.path.relpath(ip, run_dir),
        producer=dict(stage="w18_rulers", package="W18", run_id=os.path.basename(run_dir), commit=cfg["git"]["sha"])))
    p = os.path.join(base, "tableset.json")
    write_json(p, doc)
    st.output("tableset", p)
    st.extra.update(tables=n, triples=len(triples), counts=counts, eligible=len(eligible), tableset_sha256=doc["sha256"], seconds=round(time.time() - t0, 1))
    return doc


def _load(run_dir, doc, names):
    part = doc["parts"]["ruler"]
    out = {}
    for name in names:
        out[name] = np.concatenate([np.load(os.path.join(run_dir, part["folder"], "%s.%s.npy" % (sh["name"], name))) for sh in part["shards"]])
    return out


def gate(st, cfg, run_dir):
    from bench.rel import implant as I
    s = settings(cfg)
    base = os.path.join(run_dir, OUT)
    p = os.path.join(base, "tableset.json")
    with open(p) as f:
        doc = json.load(f)
    with open(os.path.join(base, "index.json")) as f:
        index = json.load(f)
    problems = [dict(e, part="manifest") for e in handoff.tableset_problems(doc, run_dir)]
    if doc.get("kind") != api.TABLE_RULER or doc.get("purpose") != "ruler" or (doc.get("producer") or {}).get("stage") != "w18_rulers":
        problems.append(dict(error="not a ruler set of stage w18_rulers"))
    listed = index.get("genomes") or []
    src, _ = source_of(s)
    by = {m["id"]: m for m in src["index"]["genomes"]}
    wrong = [g["id"] for g in listed if g["id"] not in by or by[g["id"]]["split"] != g.get("split") or g.get("split") not in V.OPEN_SPLITS]
    if wrong:
        problems.append(dict(error="genomes listed outside the open splits of the source, or with another split", genomes=wrong[:10]))
    out = _load(run_dir, doc, list(IO.fields(s["rows"], s["version"])))
    with np.load(os.path.join(base, "w18_meta.npz")) as z:
        meta = {k: z[k] for k in z.files}
    if sorted(meta) != sorted(I.FIELDS) or any(len(v) != len(out["tok"]) for v in meta.values()):
        problems.append(dict(error="the meta file does not hold every field of every table"))
    held = {i for i, g in enumerate(listed) if g["split"] == "heldout"}
    eligible = set(index.get("eligible") or []) & held
    train = {i for i, g in enumerate(listed) if g["split"] == "train"}
    units = I.split_units([by[g["id"]] for g in listed if g["id"] in by])
    problems += [dict(error=e) for e in I.verify(out, meta, eligible, train, {i: units.get(g["id"]) for i, g in enumerate(listed)})]
    digests, unequal = mask_digests(out, meta)
    if unequal or any(t.get("mask_sha256") != digests.get(t["triple"]) for t in index.get("triples") or []):
        problems.append(dict(error="the cells the passes hide differ between the tables of a triple, or from the index", triples=unequal[:10]))
    sealed_ids = {m["id"] for m in src["index"]["genomes"] if m["split"] not in V.OPEN_SPLITS}
    if sealed_ids & ({g["id"] for g in listed} | {g["id"] for g in doc.get("genomes") or []}):
        problems.append(dict(error="genomes outside the open splits are named"))
    require(GateResult("G4_rel_w18_rulers", "fail" if problems else "pass", dict(
        problems=problems[:50], tables=doc.get("tables"), triples=(index.get("counts") or {}).get("triples"), counts=index.get("counts"),
        left_out_by_code=index.get("left_out_by_code"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False), sha256_file(os.path.join(base, "index.json"), use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "tableset.json")) as f:
        doc = json.load(f)
    with open(os.path.join(run_dir, OUT, "index.json")) as f:
        index = json.load(f)
    c = index.get("counts") or {}
    tiers, rows = {}, {}
    for t in index.get("triples") or []:
        if t.get("scenario") == "in_place":
            tiers[t["tier"]] = tiers.get(t["tier"], 0) + 1
            rows[min(t["rows"], 8)] = rows.get(min(t["rows"], 8), 0) + 1
    js = dict(name=doc["name"], tables=doc["tables"], triples=c.get("triples"), copies=c.get("copies"), failed=c.get("failed"), zero_rows=c.get("zero_rows"),
              phases={k: c.get(k) for k in ("phase1", "phase2", "phase3")}, context_homolog=c.get("context_homolog"),
              donor_tiers={str(k): v for k, v in sorted(tiers.items())}, rows_on_target={str(k): v for k, v in sorted(rows.items())},
              tier_expectation=index.get("tier_expectation"), left_out_by_code=index.get("left_out_by_code"), tableset_sha256=doc["sha256"], folder=OUT)
    md = ["Implant tables (`%s`, kind ruler, construct only): %d triples in place and %d copies with the donor's lineage unsampled, %d tables; "
          "phases (n = 256 / matched / zero rows) %s; failed %s; triples flagged by rule 9 %s." % (
              doc["name"], c.get("triples", 0), c.get("copies", 0), doc["tables"], "/".join(str(c.get(k)) for k in ("phase1", "phase2", "phase3")),
              c.get("failed"), c.get("context_homolog")),
          "Donor tiers (tier: triples) %s, pre-declared %s; rows on the target (8 = 8 or more) %s; held-out genomes left out by the genetic code %s." % (
              js["donor_tiers"], js["tier_expectation"], js["rows_on_target"], js["left_out_by_code"]),
          "Manifest `%s`." % doc["sha256"]]
    return dict(json=js, md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w18_rulers") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
