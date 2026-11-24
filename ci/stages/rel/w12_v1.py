"""Stage rel_w12_v1 (package W12; pipeline rel_tables): the tables of version 1 (seeddrift/train/rel/tables/v1_build.py)
from a passed full run of stage rel_w12_search, training and held-out parts, with their census.
`python -m ci.stages.rel.w12_v1 RUN_DIR`. The module also builds the tables of the development chromosomes (task
breakdown 8.8) for stage rel_w12_islands (ci/stages/rel/w12_islands.py): the same code, its own table set.

Inputs, each checked before use: the search run (rel_w12_v1.search_from; passed, not a sample run; its manifest the
output of its stage, every rows file read checked against it), the training arrays of its source run (every file
against the source's manifests) and, for the development chromosomes, the arrays the island stage of that run
prepared (against the island manifest). The sealed check runs on every genome first (the development chromosomes
through common.development_chromosomes); the sealed test split is never opened.
Distances: the k-mer distances of stage rel_w2_dist (seeddrift/train/rel/tables/v0_dist.py, the same k and sketch
size), computed here for every genome of the search's list (the development chromosomes are not in the distance
stage's list); tiers by api.TIER_EDGES.

Configuration (profile block rel_w12_v1, for stage rel_w12_islands rel_w12_islands): search_from, expect, rows (32),
train_per_group, heldout_per_group (rel_w12_v1), per_chromosome (rel_w12_islands), shard_tables, processes, dist_k (6),
dist_size (4096), seed.
Outputs under data/rel_w12/v1/ (data/rel_w12/islands/): the shards of every part (<part>/<shard>.<field>.npy, the fields
of version 1), index.json (the genome list: the search's), counts.json, census.json and census.md, tableset.json (the
table-set manifest, written last; the development chromosomes' set has purpose heldout and not_for_training).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ci.gates import GateFailure, GateResult, run
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from seeddrift.train import source_model as SM
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.tables import v0_build as V0
from seeddrift.train.rel.tables import v0_census as C
from seeddrift.train.rel.tables import v0_dist as D
from seeddrift.train.rel.tables import v1_build as V1
from seeddrift.train.rel.tables import v1_hits as H

WHICH = dict(v1=dict(block="rel_w12_v1", stage="w12_v1", out=os.path.join("data", "rel_w12", "v1"), parts=("train", "heldout")),
             islands=dict(block="rel_w12_islands", stage="w12_islands", out=os.path.join("data", "rel_w12", "islands"), parts=("island",)),
             x=dict(block="rel_w12_v1x", stage="w12_v1x", out=os.path.join("data", "rel_w12", "v1x"), parts=("heldout",)))
SEARCH = os.path.join("data", "rel_w12", "search")
KEYS = {"search_from", "expect", "rows", "train_per_group", "heldout_per_group", "per_chromosome", "shard_tables", "processes", "dist_k", "dist_size", "seed",
        "frame_index", "name", "train_sampling", "heldout_equals", "max_uses_per_target", "heldout_from", "new_since"}


def settings(cfg, which="v1") -> dict:
    w = WHICH[which]
    c = cfg.get(w["block"]) or {}
    unknown = set(c) - KEYS
    if unknown or not c.get("search_from"):
        raise SystemExit("%s: needs `search_from`; unknown keys %s" % (w["block"], sorted(unknown)))
    out = dict(search_from=str(c["search_from"]), expect={str(k): int(v) for k, v in (c.get("expect") or {}).items()}, rows=int(c.get("rows", 32)),
               train_per_group=int(c.get("train_per_group", 8400)), heldout_per_group=int(c.get("heldout_per_group", 250)),
               per_chromosome=int(c.get("per_chromosome", 500)), shard_tables=int(c.get("shard_tables", 2400)), processes=int(c.get("processes", 8)),
               dist_k=int(c.get("dist_k", 6)), dist_size=int(c.get("dist_size", 4096)), seed=int(c.get("seed", cfg.get("seed", 1))),
               frame_index=str(c.get("frame_index", "none")), name=str(c["name"]) if c.get("name") else None,
               train_sampling=str(c.get("train_sampling", "genome")), heldout_equals=dict(c["heldout_equals"]) if c.get("heldout_equals") else None,
               max_uses_per_target=int(c["max_uses_per_target"]) if c.get("max_uses_per_target") else None,
               heldout_from=dict(c["heldout_from"]) if c.get("heldout_from") else None, new_since=dict(c["new_since"]) if c.get("new_since") else None,
               max_left_out=20)
    for k in ("heldout_from", "new_since"):
        if out[k] is not None and set(out[k]) != {"run", "sha256"}:
            raise SystemExit("%s: %s takes {run, sha256} (a passed table set of version 1)" % (w["block"], k))
    if out["heldout_from"] and which != "v1":
        raise SystemExit("%s: heldout_from belongs to the tables of version 1" % w["block"])
    if (which == "x") != bool(out["new_since"]):
        raise SystemExit("%s: new_since (the table set whose genomes are not examined) goes with the set of new held-out genomes, and only with it" % w["block"])
    if out["max_uses_per_target"] is not None and (out["train_sampling"] != "gene" or out["max_uses_per_target"] < 1):
        raise SystemExit("%s: max_uses_per_target (a positive count) goes with train_sampling gene" % w["block"])
    he = out["heldout_equals"]
    if he is not None and (set(he) != {"run", "sha256"} or "heldout" not in w["parts"]):
        raise SystemExit("%s: heldout_equals takes {run, sha256} (a table set with a held-out part)" % w["block"])
    if not (2 <= out["rows"] <= 64 and out["shard_tables"] >= 1 and 1 <= out["processes"] <= 16) or out["frame_index"] not in ("none", "target", "all") \
            or out["train_sampling"] not in ("genome", "gene"):
        raise SystemExit("%s: settings out of range: %r" % (w["block"], out))
    return out


def search_run(path):
    """(the search's manifest, its genome list) of a passed full run of stage rel_w12_search, checked."""
    rp = os.path.join(path, "record.json")
    if not os.path.exists(rp):
        raise SystemExit("w12 tables: %s holds no run record" % path)
    with open(rp) as f:
        rec = json.load(f)
    if rec.get("status") != "passed":
        raise SystemExit("w12 tables: the search run %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
    mp = os.path.join(path, SEARCH, "manifest.json")
    W._output(path, "rel_w12_search", "search_manifest", mp)
    with open(mp) as f:
        man = json.load(f)
    if float((man.get("settings") or {}).get("sample", 0)) != 1.0:
        raise SystemExit("w12 tables: %s is a sample run of the search (sample %s), not the full search" % (rec.get("run_id"), (man.get("settings") or {}).get("sample")))
    if not (man.get("sealed_check") or {}).get("passed"):
        raise SystemExit("w12 tables: the search run records no passed sealed check")
    files = {e["path"]: e for e in man.get("files") or []}
    gp = os.path.join(SEARCH, "genomes.json")
    if gp not in files or sha256_file(os.path.join(path, gp), use_sidecar=False) != files[gp]["sha256"]:
        raise SystemExit("w12 tables: the search's genome list is not the file its manifest names")
    with open(os.path.join(path, gp)) as f:
        genomes = json.load(f)["genomes"]
    return man, genomes, files, dict(run_id=rec.get("run_id") or os.path.basename(path), commit=(rec.get("git") or {}).get("sha"), manifest_sha256=sha256_file(mp, use_sidecar=False),
                                     genomes_sha256=files[gp]["sha256"])


def reference_set(ref, w_out=None):
    """(manifest, index, counts) of the table set of version 1 of run ref["run"] whose manifest sha256 is ref["sha256"],
    checked (its manifest made again gives that sha256)."""
    out = w_out or WHICH["v1"]["out"]
    rp = os.path.join(ref["run"], out, "tableset.json")
    if not os.path.exists(rp):
        raise SystemExit("w12 tables: the reference table set %s is missing" % rp)
    with open(rp) as f:
        rdoc = json.load(f)
    if rdoc.get("sha256") != ref["sha256"] or handoff.finish(rdoc)["sha256"] != ref["sha256"]:
        raise SystemExit("w12 tables: the reference table set's manifest is not sha256 %s" % ref["sha256"][:12])
    files = {e["path"]: e for e in rdoc.get("files") or []}
    loaded = []
    for name in ("index.json", "counts.json"):
        p = os.path.join(out, name)
        if p not in files or sha256_file(os.path.join(ref["run"], p), use_sidecar=False) != files[p]["sha256"]:
            raise SystemExit("w12 tables: the reference's %s is not the file its manifest names" % name)
        with open(os.path.join(ref["run"], p)) as f:
            loaded.append(json.load(f))
    return rdoc, loaded[0], loaded[1]


def copy_heldout(ref, genomes, groups, rows, base, w_out):
    """The held-out part of the reference table set, copied file by file (the programme lead, 2026-11-24: round 2's
    3,000 held-out tables are round 1's, byte for byte), when it reads the same here: the same groups and rows, and
    every genome its tables name at the same place of this search's genome list. Returns (part entry, counts, files,
    the genome indices used)."""
    import shutil
    rdoc, ridx, rcounts = reference_set(ref)
    if ridx["groups"] != groups or int(ridx["rows"]) != int(rows):
        raise SystemExit("w12 tables: the reference's held-out tables have other groups or rows")
    mark = "/heldout/"
    mine = [e for e in rdoc["files"] if mark in "/" + e["path"]]
    folder = os.path.join(base, "heldout")
    os.makedirs(folder, exist_ok=True)
    files = []
    for e in mine:
        dst = os.path.join(folder, os.path.basename(e["path"]))
        shutil.copyfile(os.path.join(ref["run"], e["path"]), dst)
        if sha256_file(dst, use_sidecar=False) != e["sha256"]:
            raise SystemExit("w12 tables: the copy of %s is not the reference's file" % e["path"])
        files.append(dict(e, path=os.path.join(w_out, "heldout", os.path.basename(e["path"]))))
    part = ridx["parts"]["heldout"]
    used = set()
    for sh in part["shards"]:
        used |= {int(x) for x in np.unique(np.load(os.path.join(folder, "%s.genome.npy" % sh["name"]))) if x >= 0}
    moved = [i for i in sorted(used) if i >= len(genomes) or genomes[i]["id"] != ridx["genomes"][i]["id"] or genomes[i]["split"] != ridx["genomes"][i]["split"]]
    if moved:
        raise SystemExit("w12 tables: genomes of the reference's held-out tables are elsewhere in this search's list: %s" % [ridx["genomes"][i]["id"] for i in moved[:5]])
    return dict(part, folder=os.path.join(w_out, "heldout"), copied_from=dict(ref, run_id=os.path.basename(os.path.normpath(ref["run"])))), rcounts["parts"]["heldout"], files, used


def _strip(gen):
    for name in ("hoff", "hprot", "hqs", "track", "toff"):
        setattr(gen, name, None)
    return gen


def load_arrays(man, genomes, cfg, s, st, with_islands=False):
    """{id: Genome} of the genomes of the search's list (the development chromosomes only with_islands, for the island
    set's tables), every file checked; the sealed checks first. The record of the id check covers every genome loaded,
    the development chromosomes included (handoff.sealed_problems: at least the genomes a manifest lists)."""
    src = W.source(man["settings"]["source"], pairs=False)         # the source run the search read (its own settings)
    sealed = common.sealed_set(cfg)
    corpus = [g for g in genomes if g["split"] != "island"]
    islands = [g for g in genomes if g["split"] == "island"] if with_islands else []
    metas, left, _ = W.open_genomes(src, dict(s, expect=s["expect"]), sealed)
    by = {m["id"]: m for m in metas}
    bad = [g["id"] for g in corpus if g["id"] not in by or by[g["id"]]["split"] != g["split"]]
    if bad:
        raise SystemExit("w12 tables: genomes of the search that are not open genomes of its source with the same split: %s" % bad[:5])
    from ci.taxonomy import Taxonomy
    ev = common.sealed_check(cfg, [W.entry(src, by[g["id"]]) for g in corpus], splits={g["id"]: g["split"] for g in corpus}, sealed=sealed,
                             tax=Taxonomy(st.repo(cfg["taxdump"])))
    paths = [os.path.join("data", "train_data", g["id"] + ".npz") for g in corpus]
    wrong = [p for p, h in zip(paths, V0.hash_files([os.path.join(src["run"], p) for p in paths], s["processes"])) if src["files"].get(p) != h]
    if wrong:
        raise SystemExit("w12 tables: training arrays of %s differ from its manifests: %s" % (src["run"], wrong[:5]))
    arrays = {g["id"]: _strip(SM.Genome(dict(by[g["id"]]), os.path.join(src["run"], p))) for g, p in zip(corpus, paths)}
    island_prov = None
    if islands:
        from ci import manifest as M
        from seeddrift.train.rel import sealed as SL
        isl = man.get("islands") or {}
        run = man["settings"].get("islands_from") or man["settings"]["source"]
        common.development_chromosomes([g["id"] for g in islands])
        try:                                                        # one record for the corpus genomes and the development chromosomes
            ids = SL.check_ids([g["id"] for g in corpus + islands], sealed, splits={g["id"]: g["split"] for g in corpus + islands},
                               allowed=tuple(SL.ALLOWED) + ("island",))
        except SL.SealedError as err:
            raise SystemExit("w12 tables: %s" % err)
        ev = dict(ev, ids=ids)
        ip = os.path.join(run, "data", "islands", "index.json")
        if sha256_file(ip, use_sidecar=False) != isl.get("index_sha256"):
            raise SystemExit("w12 tables: the island index of %s is not the one the search read" % run)
        with open(ip) as f:
            idx = {g["id"]: g for g in json.load(f)["genomes"]}
        mman = M.load(os.path.join(run, "manifests", isl["manifest"] + ".json"))
        want = {f["path"]: f["sha256"] for e in mman["entries"] for f in e.get("files", [])}
        for g in islands:
            p = idx[g["id"]]["path"]
            if want.get(p) is None or sha256_file(os.path.join(run, p), use_sidecar=False) != want[p]:
                raise SystemExit("w12 tables: the island arrays %s differ from the island manifest" % p)
            arrays[g["id"]] = _strip(SM.Genome(dict(id=g["id"], ranks=idx[g["id"]]["ranks"], group=g["group"]), os.path.join(run, p)))
        island_prov = dict(isl, run=os.path.basename(os.path.normpath(run)))
    return arrays, dict(ids=ev["ids"], lineage=ev["lineage"], left_out=len(left), islands=dict(checked=len(islands), development_chromosomes=True) if islands else 0), island_prov


def distances(arrays, genomes, s):
    """The k-mer distances of the genomes whose arrays are loaded; inf for a genome not loaded (the development
    chromosomes in the tables of version 1, which never show them)."""
    have = [i for i, g in enumerate(genomes) if g["id"] in arrays]
    sk = [D.sketch(arrays[genomes[i]["id"]].codons, arrays[genomes[i]["id"]].coff, s["dist_k"], s["dist_size"]) for i in have]
    jac, _ = D.matrix([x for x, _ in sk], [n for _, n in sk], s["dist_size"], s["processes"])
    dist = np.full((len(genomes), len(genomes)), np.inf)
    dist[np.ix_(have, have)] = D.mash(jac, s["dist_k"])
    np.fill_diagonal(dist, 0.0)
    return dist


def build(st, cfg, run_dir, which="v1"):
    w, s, t0 = WHICH[which], settings(cfg, which), time.time()
    say = lambda text: print("%s: %s (%.0f s)" % (w["block"], text, time.time() - t0), flush=True)
    man, genomes, files, prov = search_run(s["search_from"])
    arrays, sealed_ev, island_prov = load_arrays(man, genomes, cfg, s, st, with_islands=which == "islands")
    gix = {g["id"]: i for i, g in enumerate(genomes)}
    plan_genomes, since = genomes, None
    if which == "x":                                               # the held-out genomes the reference set did not have
        _, sidx, _ = reference_set(s["new_since"])
        old_ids = {g["id"] for g in sidx["genomes"]}
        plan_genomes = [dict(g, split="heldout_before") if g["split"] == "heldout" and g["id"] in old_ids else g for g in genomes]
        since = dict(s["new_since"], run_id=os.path.basename(os.path.normpath(s["new_since"]["run"])), heldout_before=sum(1 for g in plan_genomes if g["split"] == "heldout_before"))
    examined = [g for g in plan_genomes if g["split"] in w["parts"]]
    if not examined:
        raise SystemExit("%s: no genome of the parts %s in the search" % (w["block"], w["parts"]))
    need = [os.path.join(SEARCH, g["id"] + ".npz") for g in examined]
    wrong = [p for p, h in zip(need, V0.hash_files([os.path.join(s["search_from"], p) for p in need], s["processes"])) if (files.get(p) or {}).get("sha256") != h]
    if wrong:
        raise SystemExit("%s: rows files of the search differ from its manifest: %s" % (w["block"], wrong[:5]))
    say("%d genomes, %d examined; files checked" % (len(genomes), len(examined)))
    dist = distances(arrays, genomes, s)
    say("distances")
    train = [i for i, g in enumerate(genomes) if g["split"] == "train"]
    rows_of = lambda gid: H.load(os.path.join(s["search_from"], SEARCH, gid + ".npz"))         # noqa: E731
    builder = V1.Builder(genomes, arrays, rows_of, lambda a, b: float(dist[a, b]), train, s["rows"])
    groups = sorted({g["group"] for g in genomes if g["split"] != "island"})
    base = os.path.join(run_dir, w["out"])
    parts, counts, out_files, used = {}, {}, [], set()
    rel_of = {g["id"]: g.get("relation", "") for g in W.source(man["settings"]["source"], pairs=False)["index"]["genomes"]}
    for part in w["parts"]:
        if part == "heldout" and s["heldout_from"]:
            parts[part], counts[part], copied, used_here = copy_heldout(s["heldout_from"], genomes, groups, s["rows"], base, w["out"])
            out_files += copied
            used |= used_here
            say("part heldout: %d tables copied from run %s" % (parts[part]["tables"], parts[part]["copied_from"]["run_id"]))
            continue
        per = s["per_chromosome"] * sum(1 for g in genomes if g["split"] == "island") if part == "island" else s[part + "_per_group"]
        mine = [g for g in plan_genomes if g["split"] == part]
        valid = {g["id"]: np.flatnonzero(np.diff(arrays[g["id"]].coff) > 0) for g in mine if g["id"] in arrays}
        by_gene = part == "train" and s["train_sampling"] == "gene"
        plan = V1.plan_by_gene(plan_genomes, groups, part, per, s["seed"], valid, s["max_uses_per_target"]) if by_gene else V1.plan_set(plan_genomes, groups, part, per, s["seed"])
        folder = os.path.join(base, part)
        shards, census = V1.build_set(builder, groups, part, plan, folder, shard_tables=s["shard_tables"], seed=s["seed"], processes=s["processes"], progress=say)
        problems = V1.verify_set(folder, shards, s["rows"], part, {gix[g["id"]] for g in plan_genomes if g["split"] == part}, set(train), len(groups))
        if problems:
            raise SystemExit("%s: the %s tables are not sound: %s" % (w["block"], part, problems[:10]))
        hashes = _hash(folder, shards, s["rows"], s["processes"])
        added_shard = set(getattr(api, "SHARD_FIELDS_ADDED", {}))
        for name, _ in shards:
            for field in V1.fields(s["rows"]):
                role = "shard" if field in api.SHARD_FIELDS or field in added_shard else "meta"
                out_files.append(dict(path=os.path.join(w["out"], part, "%s.%s.npy" % (name, field)), role=role, **hashes[name][field]))
        g_all = np.concatenate([np.load(os.path.join(folder, "%s.genome.npy" % name)).ravel() for name, _ in shards])
        used |= {int(x) for x in np.unique(g_all) if x >= 0}
        tgt = np.concatenate([np.load(os.path.join(folder, "%s.gene.npy" % name), mmap_mode="r")[:, V1.TARGET] for name, _ in shards]).astype(np.int64)
        pairs = np.asarray([gix[g] for g in plan[0]], dtype=np.int64) * (1 << 32) + tgt
        _, reps = np.unique(pairs, return_counts=True)
        targets = dict(tables=int(len(pairs)), distinct=int(len(reps)), max_times=int(reps.max()) if len(reps) else 0, sampling="gene" if by_gene else "genome",
                       note=("target genes drawn over the whole group without replacement: every gene of the group once before any twice (plan_by_gene)"
                             if by_gene else "target genes: each genome's genes in a shuffled order, a gene again only after every gene of its genome"),
                       by_group=V1.target_stats(plan, groups, tgt, {k: v for k, v in rel_of.items() if v}, valid, {g["id"]: g["group"] for g in mine}))
        targets.update(max_uses_per_target=s["max_uses_per_target"] if by_gene else None,
                       tables_by_group={groups[i]: sum(1 for x in plan[1] if x == i) for i in sorted(set(plan[1]))})
        parts[part] = dict(split=V1.SPLIT_CODE[part], tables=len(plan[0]), per_group=per, groups_without_a_genome=plan[2], folder=os.path.join(w["out"], part), targets=targets,
                           shards=[dict(name=n, tables=m) for n, m in shards], sha256=sha256_json(sorted([n, f, h["sha256"]] for n, fs in hashes.items() for f, h in fs.items())))
        counts[part] = {k: v.tolist() for k, v in census.to_arrays().items()}
        say("part %s: %d tables" % (part, len(plan[0])))
    frame_ix = None
    if s["frame_index"] != "none":                                  # the pairs the online frame layout reads, for a machine without the search's index
        fi = os.path.join(base, "frame_index")
        sizes = V1.frame_index(os.path.join(s["search_from"], (man.get("index") or {}).get("folder", os.path.join(SEARCH, "index"))),
                               [(os.path.join(base, p), [x["name"] for x in parts[p]["shards"]]) for p in parts], fi,
                               blocks=(V1.TARGET,) if s["frame_index"] == "target" else None, processes=s["processes"])
        for k in sorted(os.listdir(fi)):
            fp = os.path.join(fi, k)
            out_files.append(dict(path=os.path.relpath(fp, run_dir), sha256=sha256_file(fp, use_sidecar=False), bytes=os.path.getsize(fp), role="index"))
        frame_ix = dict(sizes, folder=os.path.relpath(fi, run_dir), blocks=s["frame_index"],
                        note="the pairs of the search's index that align.frame_layout_online looks up for these tables (v1_hits.Index)")
        say("frame index: %d pairs" % sizes["pairs"])
    index = dict(format="rel-w12-v1-index", version=1, table_version=V1.TABLE_VERSION, api_version=api.API, rows=s["rows"], groups=groups, parts=parts,
                 search=prov, tier_edges=list(V1.TIER_EDGES), genomes=[dict(id=g["id"], group=g["group"], split=g["split"], taxid=g.get("taxid")) for g in genomes])
    cen = census_doc(counts, groups, s["rows"], prov)
    for name, obj, role in (("index.json", index, "index"), ("counts.json", dict(rows=s["rows"], groups=groups, parts=counts), "meta"), ("census.json", cen, "meta")):
        p = os.path.join(base, name)
        write_json(p, obj)
        out_files.append(dict(path=os.path.join(w["out"], name), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p), role=role))
    page = os.path.join(base, "census.md")
    with open(page, "w") as f:
        f.write(census_page(cen))
    st.output(w["stage"] + "_census", page)
    listed = [dict(id=genomes[i]["id"], taxid=genomes[i].get("taxid"), split=genomes[i]["split"]) for i in sorted(used)]
    n_close = sum(1 for g in genomes if rel_of.get(g["id"]) in ("same species", "same genus"))
    island = which == "islands"
    xset = which == "x"
    doc = handoff.finish(dict(
        format="rel-tableset", format_version=1, name=s["name"] or "w12-%s-r%d-s%d" % ("v1x" if xset else which, s["rows"], s["seed"]), kind=api.TABLE_REAL,
        api_version=V0.API_VERSION, table_version=V1.TABLE_VERSION, layout=api.LAYOUT_STORED, purpose="heldout" if island or xset else None,
        not_for_training=(island or xset) or None, new_since=since,
        coverage=dict(groups=["bacteria"] if island else groups,
                      scenarios=["the island set's development chromosomes as examined genomes, other rows training genomes (task breakdown 8.8)" if island
                                 else "far relatives of the medium corpus, rows drawn by distance tier among the genomes with a homolog (the uncapped search)"]
                      + (["close relatives: %d genomes of the same species or genus as a corpus genome (package W13's first batch)" % n_close] if n_close else []),
                      not_covered=([] if n_close else ["close relatives (the corpus has few)"]) + ["the sealed test split and the sealed chromosomes"] + (
                          ["any training: these tables are for diagnosis only"] if island else []),
                      note=("Development chromosomes as row 0; never read by any training (not_for_training)." if island else
                            ("Held-out genomes new since table set %s (its held-out genomes are not examined here): read and reported apart from its "
                             "held-out tables; never read by any training (not_for_training)." % (since or {}).get("run_id") if xset else
                             "Tables of version 1: rows from the uncapped search; stored placement from the alignment anchored on the examined gene."))),
        tables=sum(p["tables"] for p in parts.values()), rows=s["rows"], seed=s["seed"], parts=parts, files=out_files, group_names=groups, genomes=listed,
        search=prov, islands=island_prov, sealed_check=dict(passed=True, **sealed_ev), index=os.path.join(w["out"], "index.json"), frame_index=frame_ix,
        producer=dict(stage=w["stage"], package="W12", run_id=os.path.basename(run_dir), commit=cfg["git"]["sha"])))
    doc = handoff.finish({k: v for k, v in doc.items() if v is not None})
    p = os.path.join(base, "tableset.json")
    write_json(p, doc)
    st.output("tableset", p)
    st.extra.update(tables={k: v["tables"] for k, v in parts.items()}, tableset_sha256=doc["sha256"], seconds=round(time.time() - t0, 1))
    return doc


def _hash(folder, shards, rows, processes):
    paths = [(n, f, os.path.join(folder, "%s.%s.npy" % (n, f))) for n, _ in shards for f in V1.fields(rows)]
    out = {}
    for (n, f, p), h in zip(paths, V0.hash_files([p for _, _, p in paths], processes)):
        out.setdefault(n, {})[f] = dict(sha256=h, bytes=os.path.getsize(p))
    return out


def census_doc(counts, groups, rows, prov):
    out = dict(format="rel-w12-v1-census", version=1, search=prov, rows=rows, groups=groups, tiers=[*("tier %d" % t for t in range(V1.N_TIERS - 1)), "no shared k-mer"],
               distance_bins=list(V1.DIST_EDGES), sets={})
    for part, arrays in counts.items():
        c = V1.Census1.from_arrays(arrays, rows)
        a = c.a
        out["sets"][part] = dict(census=C.summarise(c.base, groups), tier_rows={g: [int(x) for x in a["tier_rows"][k]] for k, g in enumerate(groups)},
                                 self_rows=[int(x) for x in a["self_rows"]], tables_with_self=[int(x) for x in a["tables_with_self"]], dropped_alignments=int(a["dropped"].sum()),
                                 candidates_per_table=float(a["candidates_sum"].sum()) / max(int(c.base.a["tables"].sum()), 1),
                                 row_distances={g: [int(x) for x in a["dist_rows"][k]] for k, g in enumerate(groups)})
    return out


def census_page(cen):
    L = ["# Census of the tables of version 1", "", "Search run `%s`; %d rows stored." % (cen["search"]["run_id"], cen["rows"]), ""]
    for part, d in cen["sets"].items():
        a = d["census"]["all"]
        L += ["## %s: %s tables" % (part, "{:,}".format(a["tables"])), "",
              "Other rows per table %.2f; tables without other rows %s; cells with codons on the blocks other than the target %s; kinds there: %s." % (
                  a["other_rows"]["mean"] or 0, _p(a["other_rows"]["share_of_tables_without"]), _p(a["filled"]["not_target"]["share"]),
                  ", ".join("%s %s" % kv for kv in a["kinds"]["not_target"].items())),
              "Tables with a self-copy row: %s; candidate genomes on the target per table: %.1f; alignments dropped when merging: %s." % (
                  "{:,}".format(sum(d["tables_with_self"])), d["candidates_per_table"], "{:,}".format(d["dropped_alignments"])), "",
              "| group | rows per tier (%s) | rows' distance to row 0 (%s, none) |" % (", ".join(cen["tiers"]), ", ".join("%.2f" % x for x in cen["distance_bins"])),
              "|---|---|---|"]
        for g in cen["groups"]:
            if sum(d["tier_rows"][g]):
                L.append("| %s | %s | %s |" % (g, " / ".join(str(x) for x in d["tier_rows"][g]), " / ".join(str(x) for x in d["row_distances"][g])))
        L.append("")
    return "\n".join(L) + "\n"


def _p(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def gate(st, cfg, run_dir, which="v1"):
    w, s = WHICH[which], settings(cfg, which)
    p = os.path.join(run_dir, w["out"], "tableset.json")
    with open(p) as f:
        doc = json.load(f)
    problems = [dict(e, part="manifest") for e in handoff.tableset_problems(doc, run_dir)]
    if doc.get("kind") != api.TABLE_REAL or (doc.get("producer") or {}).get("stage") != w["stage"]:
        problems.append(dict(error="not a real table set of stage %s" % w["stage"]))
    splits = {g.get("split") for g in doc.get("genomes") or []}
    if which == "islands":
        if doc.get("purpose") != "heldout" or doc.get("not_for_training") is not True:
            problems.append(dict(error="the development chromosomes' tables must be marked purpose heldout and not_for_training"))
        isl = [g["id"] for g in doc.get("genomes") or [] if g.get("split") == "island"]
        try:
            common.development_chromosomes(isl)
        except SystemExit as err:
            problems.append(dict(error=str(err)[:300]))
        if splits - {"island", "train"}:
            problems.append(dict(error="genomes of other splits than the development chromosomes and the training rows"))
    elif splits - {"train", "heldout"}:
        problems.append(dict(error="genomes outside the open splits"))
    if which == "x":
        if doc.get("purpose") != "heldout" or doc.get("not_for_training") is not True:
            problems.append(dict(error="the set of new held-out genomes must be marked purpose heldout and not_for_training"))
        _, sidx, _ = reference_set(s["new_since"])
        old = {g["id"] for g in sidx["genomes"]}
        gl = [g["id"] for g in json.load(open(os.path.join(run_dir, w["out"], "index.json")))["genomes"]]
        ex = set()
        for sh in (doc.get("parts") or {}).get("heldout", {}).get("shards") or []:
            ex |= {int(x) for x in np.load(os.path.join(run_dir, w["out"], "heldout", "%s.genome.npy" % sh["name"]))[:, 0] if x >= 0}
        if any(gl[i] in old for i in ex):
            problems.append(dict(error="examined genomes that the reference set already had"))
    man, genomes, _, prov = search_run(s["search_from"])
    if doc.get("search") != prov:
        problems.append(dict(error="the manifest names another search than the configured run's"))
    if s["heldout_equals"]:
        problems += heldout_problems(run_dir, w, s["heldout_equals"])
    results = [run(GateResult("G4_rel_%s" % w["stage"], "fail" if problems else "pass", dict(problems=problems[:50], tables={k: v["tables"] for k, v in (doc.get("parts") or {}).items()}))
                   , run_dir)]
    if any(r.status == "fail" for r in results):
        raise GateFailure("gates failed: G4_rel_%s" % w["stage"])
    return sha256_json([sha256_file(p, use_sidecar=False)])


def heldout_problems(run_dir, w, ref):
    """Problems unless the held-out part of this run's table set is the reference's, file for file: the reference is
    the table set of run ref["run"] whose manifest sha256 is ref["sha256"]; every held-out file of it must be here with
    the same name, and the file here, hashed again, must have the reference's sha256 (the programme lead, 2026-11-23:
    the 3,000 held-out tables of round 2's training set are round 1's, byte for byte)."""
    rp = os.path.join(ref["run"], w["out"], "tableset.json")
    if not os.path.exists(rp):
        return [dict(check="heldout_equals", error="the reference table set %s is missing" % rp)]
    with open(rp) as f:
        rdoc = json.load(f)
    if rdoc.get("sha256") != ref["sha256"] or handoff.finish(rdoc)["sha256"] != ref["sha256"]:
        return [dict(check="heldout_equals", error="the reference table set's manifest is not sha256 %s" % ref["sha256"][:12])]
    mark = "/%s/" % "heldout"
    want = {os.path.basename(e["path"]): e["sha256"] for e in rdoc.get("files") or [] if mark in "/" + e["path"]}
    folder = os.path.join(run_dir, w["out"], "heldout")
    have = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
    differ = sorted(n for n in want if n not in have or sha256_file(os.path.join(folder, n), use_sidecar=False) != want[n])
    extra = sorted(n for n in have if n.endswith(".npy") and n not in want)
    if not want or differ or extra:
        return [dict(check="heldout_equals", error="the held-out tables are not the reference's (%d of %d files differ, %d extra)" % (len(differ), len(want), len(extra)),
                     files=(differ + extra)[:20])]
    return []


def summary(run_dir, which="v1"):
    w = WHICH[which]
    with open(os.path.join(run_dir, w["out"], "tableset.json")) as f:
        doc = json.load(f)
    with open(os.path.join(run_dir, w["out"], "census.md")) as f:
        md = f.read().splitlines()
    return dict(json=dict(tables={k: v["tables"] for k, v in doc["parts"].items()}, tableset_sha256=doc["sha256"], search=doc["search"]["run_id"]), md=md[2:60])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w12_v1") as st:
        build(st, st.config, run_dir, "v1")


if __name__ == "__main__":
    main()
