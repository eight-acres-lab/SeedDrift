"""Stage rel_w12_compo (package W12; pipeline rel_tables): who the training and held-out genomes of a few groups are, in
built sets of tables (the programme lead, 2026-11-23: plants and algae overfit in round 1a for no reason found yet;
fungi and other vertebrates, of about as many genomes, did not). `python -m ci.stages.rel.w12_compo RUN_DIR`. Reads
the sets' manifests and indexes, the taxonomy and the codon arrays of the genomes of these groups (for the k-mer
distance of package W2); trains nothing, writes no table.

Per set and group: every genome of the training and held-out parts with its species, order, class and kind (plants:
land plant, green alga, charophyte alga, red alga, glaucophyte; fungi: phylum; vertebrates: class), the tables it is row
0 of, its split unit (genus, else species; its near-relative cluster when it has one); for every held-out genome the
nearest training genome by taxonomy (the lowest rank they share: species, genus, family, order, class, phylum, or none)
and by W2's k-mer distance (MinHash over codon k-mers, seeddrift/train/rel/tables/v0_dist.py), with both distances.
The test part is not read.

Configuration (profile block rel_w12_compo): sets ([{name, run, which}], which v0 or v1: a passed run of stage w2_v0 or
w12_v1), groups (default eukaryota.plants_algae, eukaryota.fungi, eukaryota.other_vertebrates), dist_k (6), dist_size
(4096), processes (2). Outputs under data/rel_w12/compo/: report.json (written last) and report.md. Gate
G4_rel_w12_compo.
"""
from __future__ import annotations

import json
import os

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w2_v0 as W
from ci.stages.rel import w12_tstats as WT
from ci.stages.rel import w12_v1 as WV
from seeddrift.train.rel.tables import v0_dist as D
from seeddrift.train.rel.tables import v0_io as IO

OUT = os.path.join("data", "rel_w12", "compo")
KEYS = {"sets", "groups", "dist_k", "dist_size", "processes"}
GROUPS = ("eukaryota.plants_algae", "eukaryota.fungi", "eukaryota.other_vertebrates")
RANKS = ("phylum", "class", "order", "family", "genus", "species")
PLANT_KINDS = ((3193, "land plant"), (3041, "green alga (Chlorophyta)"), (35493, "charophyte alga"), (2763, "red alga"), (38254, "glaucophyte"))


def settings(cfg) -> dict:
    c = cfg.get("rel_w12_compo") or {}
    unknown = set(c) - KEYS
    sets = [dict(name=str(x["name"]), run=str(x["run"]), which=str(x.get("which", "v1"))) for x in c.get("sets") or []]
    if unknown or not sets or any(x["which"] not in ("v0", "v1") for x in sets):
        raise SystemExit("rel_w12_compo: needs sets [{name, run, which v0|v1}]; unknown keys %s" % sorted(unknown))
    return dict(sets=sets, groups=[str(g) for g in c.get("groups") or GROUPS], dist_k=int(c.get("dist_k", 6)), dist_size=int(c.get("dist_size", 4096)),
                processes=int(c.get("processes", 2)))


def kind_of(tax, taxid, group):
    lin = set(tax.lineage(int(taxid)))
    if group == "eukaryota.plants_algae":
        for t, name in PLANT_KINDS:
            if t in lin:
                return name
        return "other"
    rank = "phylum" if group == "eukaryota.fungi" else "class"
    t = tax.at_rank(int(taxid), rank)
    return name_of(tax, t) if t else "(no %s)" % rank


def name_of(tax, t):
    try:
        return tax.name(int(t)) if t else ""
    except Exception:  # noqa: BLE001 - a rank taxid the taxonomy does not know: its number
        return str(t)


def shared_rank(a, b):
    """The lowest of RANKS two lineages ([phylum .. species] taxids) share, or "none"."""
    for k in range(len(RANKS) - 1, -1, -1):
        if a[k] and a[k] == b[k]:
            return RANKS[k]
    return "none"


def one_set(entry, groups, tax, s):
    w = WT.SETS[entry["which"]]
    run = entry["run"]
    rec = json.load(open(os.path.join(run, "record.json")))
    if rec.get("status") != "passed":
        raise SystemExit("rel_w12_compo: %s is not a passed run" % rec.get("run_id"))
    tp = os.path.join(run, w["out"], "tableset.json")
    W._output(run, "rel_" + w["stage"], "tableset", tp)
    doc = json.load(open(tp))
    index = json.load(open(os.path.join(run, w["out"], "index.json")))
    runs = os.path.dirname(os.path.normpath(run))
    if entry["which"] == "v0":
        src = W.source(os.path.join(runs, index["source"]["run_id"]), pairs=False)
    else:
        man, _, _, _ = WV.search_run(os.path.join(runs, doc["search"]["run_id"]))
        src = W.source(man["settings"]["source"], pairs=False)
    meta = {m["id"]: m for m in src["index"]["genomes"]}
    genomes = index["genomes"]
    tables = {}
    for part, p in doc["parts"].items():
        folder = os.path.join(run, p["folder"])
        g0 = np.concatenate([np.load(IO.shard_path(folder, x["name"], "genome"), mmap_mode="r")[:, 0] for x in p["shards"]]).astype(np.int64)
        ids, n = np.unique(g0, return_counts=True)
        for i, c in zip(ids, n):
            tables[(part, genomes[int(i)]["id"])] = int(c)
    out = {}
    for grp in groups:
        mine = [m for m in src["index"]["genomes"] if m.get("source", "corpus") == "corpus" and m["group"] == grp and m["split"] in ("train", "heldout")]
        from seeddrift.train import source_model as SM
        arrays = {m["id"]: SM.Genome(dict(m), os.path.join(src["run"], "data", "train_data", m["id"] + ".npz")) for m in mine}
        sk = [D.sketch(arrays[m["id"]].codons, arrays[m["id"]].coff, s["dist_k"], s["dist_size"]) for m in mine]
        jac, _ = D.matrix([x for x, _ in sk], [k for _, k in sk], s["dist_size"], s["processes"])
        dist = D.mash(jac, s["dist_k"])
        rows = []
        for i, m in enumerate(mine):
            r = m["ranks"]
            rows.append(dict(id=m["id"], split=m["split"], taxid=m["taxid"], species=name_of(tax, r[5]), order=name_of(tax, r[2]),
                             **{"class": name_of(tax, r[1])}, kind=kind_of(tax, m["taxid"], grp), relation=m.get("relation") or "anchor",
                             tables=tables.get((m["split"], m["id"]), 0), unit=("cluster " + m["cluster"]) if m.get("cluster") else
                             ("genus %s" % name_of(tax, r[4]) if r[4] else "species %s" % (name_of(tax, r[5]) or m["id"]))))
        tr = [i for i, m in enumerate(mine) if m["split"] == "train"]
        for i, m in enumerate(mine):
            if m["split"] != "heldout" or not tr:
                continue
            j = min(tr, key=lambda k: (float(dist[i, k]), mine[k]["id"]))
            depth = {x: RANKS.index(x) if x != "none" else -1 for x in RANKS + ("none",)}
            jt = max(tr, key=lambda k: (depth[shared_rank(meta[m["id"]]["ranks"], meta[mine[k]["id"]]["ranks"])], -float(dist[i, k]), mine[k]["id"]))
            rows[i].update(nearest_by_distance=dict(id=mine[j]["id"], distance=round(float(dist[i, j]), 4),
                                                    shared_rank=shared_rank(m["ranks"], mine[j]["ranks"])),
                           nearest_by_taxonomy=dict(id=mine[jt]["id"], shared_rank=shared_rank(m["ranks"], mine[jt]["ranks"]), distance=round(float(dist[i, jt]), 4)))
        kinds = {}
        for x in rows:
            k = kinds.setdefault(x["kind"], dict(train_genomes=0, heldout_genomes=0, train_tables=0, heldout_tables=0))
            k[x["split"] + "_genomes"] += 1
            k[x["split"] + "_tables"] += x["tables"]
        out[grp] = dict(genomes=rows, by_kind=kinds)
    return dict(name=entry["name"], run=rec.get("run_id"), which=entry["which"], tableset_sha256=doc.get("sha256"), source=src["provenance"]["run_id"],
                groups=out)


def build(st, cfg, run_dir):
    from ci.taxonomy import Taxonomy
    s = settings(cfg)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    sets = [one_set(e, s["groups"], tax, s) for e in s["sets"]]
    listed = {(g["id"], int(g["taxid"]), g["split"]) for x in sets for grp in x["groups"].values() for g in grp["genomes"]}
    st.extra["sealed_check"] = common.sealed_check(cfg, [dict(id=i, taxid=t) for i, t, _ in sorted(listed)], splits={i: sp for i, _, sp in listed})
    rep = dict(format="rel-w12-compo", version=1, note=__doc__.strip().splitlines()[0], groups=s["groups"], dist=dict(k=s["dist_k"], size=s["dist_size"]), sets=sets)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    page = os.path.join(folder, "report.md")
    with open(page, "w") as f:
        f.write(page_of(rep))
    st.output("compo_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("compo_report", p)
    return rep


def page_of(rep):
    L = ["# Who the training and held-out genomes of a few groups are (tables of version 0 and 1)", "", rep["note"], ""]
    for x in rep["sets"]:
        L += ["## %s: run `%s` (tables of version %s, source `%s`)" % (x["name"], x["run"], x["which"][1:], x["source"]), ""]
        for grp, d in x["groups"].items():
            L += ["### %s" % grp, "", "| kind | training genomes | their tables | held-out genomes | their tables |", "|---|---|---|---|---|"]
            for k, v in sorted(d["by_kind"].items()):
                L.append("| %s | %d | %s | %d | %s |" % (k, v["train_genomes"], "{:,}".format(v["train_tables"]), v["heldout_genomes"], "{:,}".format(v["heldout_tables"])))
            L += ["", "| split | genome | species | order | class | kind | tables | unit | nearest training genome by distance (distance, shared rank) | by taxonomy (shared rank, distance) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for g in sorted(d["genomes"], key=lambda g: (g["split"] != "heldout", g["kind"], g["order"], g["species"])):
                a, b = g.get("nearest_by_distance"), g.get("nearest_by_taxonomy")
                L.append("| %s | %s | %s | %s | %s | %s | %d | %s | %s | %s |" % (
                    g["split"], g["id"], g["species"], g["order"], g["class"], g["kind"], g["tables"], g["unit"],
                    "%s (%.3f, %s)" % (a["id"], a["distance"], a["shared_rank"]) if a else "", "%s (%s, %.3f)" % (b["id"], b["shared_rank"], b["distance"]) if b else ""))
            L.append("")
    return "\n".join(L)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    rep = json.load(open(p))
    problems = []
    for x in rep.get("sets") or []:
        for grp, d in x["groups"].items():
            if any(g["split"] not in ("train", "heldout") for g in d["genomes"]):
                problems.append(dict(set=x["name"], group=grp, error="a genome outside the open splits"))
            if any(g["split"] == "heldout" and "nearest_by_distance" not in g for g in d["genomes"]) and any(g["split"] == "train" for g in d["genomes"]):
                problems.append(dict(set=x["name"], group=grp, error="a held-out genome without its nearest training genome"))
    require(GateResult("G4_rel_w12_compo", "fail" if problems else "pass", dict(problems=problems[:50])), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    rep = json.load(open(os.path.join(run_dir, OUT, "report.json")))
    return dict(json={x["name"]: {g: d["by_kind"] for g, d in x["groups"].items()} for x in rep["sets"]}, md=page_of(rep).splitlines()[2:60])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w12_compo") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
