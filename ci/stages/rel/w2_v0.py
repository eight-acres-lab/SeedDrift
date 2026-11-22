"""Stage rel_w2_v0 (package W2; pipeline rel_tables): the tables of version 0, training and held-out, from the arrays of a
passed run of pipeline selfsup (seeddrift/train/rel/tables/v0_build.py). `python -m ci.stages.rel.w2_v0 RUN_DIR`.

No old stage runs again: the training-data arrays (with the three-way split) and the homolog rows of stage pairs are
taken from the source run, every file checked against the sha256 its stage record and manifest name (as
ci/stages/codon.py takes a passed run's outputs). Only genomes of the train and held-out splits are opened; the sealed
test split is neither read nor named in any output.

Before any genome is opened: the sealed check (ci/stages/rel/common.py) on every genome of the two open splits, and
gates G1 and G2 as functions on the training genomes alone (the floors of the corpus spec, the exclusions by full
lineage). A genome of an open split that the sealed check refuses by its name (the assembly of a sealed test genome,
such as a plasmid of a sealed chromosome's assembly) is left out of the tables (the stage report names it, the manifest
counts it by group and split); more than `max_left_out` of them stop the stage. Gate phase rel_w2_v0 decides G1_rel_w2_v0 and G2_rel_w2_v0 again for the record
and G4_rel_w2_v0 on what was written.

Configuration (profile block rel_w2_v0):
  source             the run directory of the passed selfsup run
  expect             optional {split: genomes}: the stage stops when the source's split sizes differ
  rows               rows stored per table (16)
  train_per_group    training tables per group (the examined genome is a training genome)
  heldout_per_group  held-out tables per group (the examined genome is a held-out genome; other rows training genomes)
  shard_tables       tables per shard (a multiple of the number of groups keeps the groups equal in every shard)
  processes          worker processes
  max_left_out       genomes the sealed check may refuse by name before the stage stops (20)
Outputs under data/rel_w2/v0/: train/ and heldout/ (the shards: <shard>.<field>.npy, the shard and meta fields of the
contract), index.json (rows, groups, the genome list the meta field `genome` indexes, the parts and their shards),
counts.json (the census counts and the codon counts per genome; never a model input) and tableset.json (the table-set
manifest of seeddrift/train/rel/handoff.py, kind real, written last).
"""
from __future__ import annotations

import json
import os
import time

import yaml

from ci import manifest as M
from ci.gates import GateFailure, GateResult, g1_kingdom, g2_exclusion, run
from ci.hashing import sha256_file, sha256_json
from ci.kingdom import KingdomMap
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.taxonomy import Taxonomy
from seeddrift.train import selfsup as SS
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.tables import v0_build as V

OUT = os.path.join("data", "rel_w2", "v0")
KEYS = {"source", "expect", "rows", "train_per_group", "heldout_per_group", "shard_tables", "processes", "max_left_out"}
PARTS = ("train", "heldout")
COVERAGE_NOTE = ("Tables of version 0 for the first read-outs. Other rows are training genomes drawn among those with a homolog row on the target gene; "
                 "before the corpus is searched again without caps, having such a row is decided by a similarity ranking (stage pairs keeps the 64 "
                 "nearest genomes of a gene), so what relatives are worth is overestimated. The stored columns come from the alignment anchored on "
                 "the examined gene: a rendering that uses them is an upper bound.")
NOT_COVERED = ["close relatives beyond what the medium corpus holds (same species or genus)", "self copies and cells searched and absent",
               "genes after the first 1,500 of a eukaryote and codons after the source's codon limit of a gene", "intergenic sequence",
               "chromosome and plasmid of one assembly as one row", "distance tiers (every tier is -1)", "the sealed test split"]


def settings(cfg) -> dict:
    vc = cfg.get("rel_w2_v0") or {}
    unknown = set(vc) - KEYS
    if unknown or not vc.get("source"):
        raise SystemExit("rel_w2_v0: needs `source`; unknown keys %s" % sorted(unknown))
    out = dict(source=str(vc["source"]), expect={str(k): int(v) for k, v in (vc.get("expect") or {}).items()}, rows=int(vc.get("rows", 16)),
               train_per_group=int(vc.get("train_per_group", 8400)), heldout_per_group=int(vc.get("heldout_per_group", 250)),
               shard_tables=int(vc.get("shard_tables", 2400)), processes=int(vc.get("processes", 4)), max_left_out=int(vc.get("max_left_out", 20)),
               seed=int(cfg.get("seed", 1)))
    if not (2 <= out["rows"] <= 32 and out["train_per_group"] >= 1 and out["heldout_per_group"] >= 1 and 1 <= out["shard_tables"] <= 20000
            and 1 <= out["processes"] <= 16):
        raise SystemExit("rel_w2_v0: settings out of range: %r" % out)
    return out


def _output(run, stage, name, path):
    """The sha256 of `path`, which must be output `name` of stage `stage` of `run`, and that stage must have passed."""
    p = os.path.join(run, "stages", stage + ".json")
    if not os.path.exists(p):
        raise SystemExit("rel_w2_v0: %s has no stage %s" % (run, stage))
    with open(p) as f:
        rep = json.load(f)
    if rep.get("status") != "passed":
        raise SystemExit("rel_w2_v0: stage %s of %s did not pass (%s)" % (stage, run, rep.get("status")))
    out = {o["name"]: o["sha256"] for o in rep.get("outputs") or []}
    sha = sha256_file(path, use_sidecar=False)
    if out.get(name) != sha:
        raise SystemExit("rel_w2_v0: %s is not the output %s of stage %s of %s" % (path, name, stage, run))
    return sha


def _json(path):
    with open(path) as f:
        return json.load(f)


def source(run) -> dict:
    """The source run's records, checked: it passed; its training-data index, the manifest and genome order of stage
    pairs and its corpus manifest are the outputs its stage records name; the file manifests match their names."""
    rp = os.path.join(run, "record.json")
    if not os.path.exists(rp):
        raise SystemExit("rel_w2_v0: %s holds no run record" % run)
    rec = _json(rp)
    if rec.get("status") != "passed":
        raise SystemExit("rel_w2_v0: the source run %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
    ip = os.path.join(run, "data", "train_data", "index.json")
    index_sha = _output(run, "train_data", "train_data_index", ip)
    index = _json(ip)
    pmp = os.path.join(run, "data", "pairs", "manifest.json")
    pm_sha = _output(run, "pairs", "pairs_manifest", pmp)
    pg_sha = _output(run, "pairs", "pairs_genomes", os.path.join(run, "data", "pairs", "genomes.json"))
    pm = _json(pmp)
    cmp_ = os.path.join(run, "data", "corpus", "manifest.json")
    cm_sha = _output(run, "corpus", "corpus_manifest", cmp_)
    corpus = _json(cmp_)
    files = {}
    for sha in (index["sha256"], pm["sha256"]):
        man = M.load(os.path.join(run, "manifests", sha + ".json"))
        files.update({f["path"]: f["sha256"] for e in man["entries"] for f in e.get("files", [])})
    return dict(run=run, index=index, files=files, corpus={e["id"]: e for e in corpus["entries"]},
                provenance=dict(run_id=rec.get("run_id") or os.path.basename(run), commit=(rec.get("git") or {}).get("sha"), ci_version=rec.get("ci_version"),
                                train_data_index_sha256=index_sha, train_data_manifest=index["sha256"], train_data_dataset=index.get("dataset_id"),
                                pairs_manifest_sha256=pm_sha, pairs_manifest=pm["sha256"], pairs_dataset=pm.get("dataset_id"), pairs_genomes_sha256=pg_sha,
                                pairs_settings=(pm.get("meta") or {}).get("settings"), corpus_manifest_sha256=cm_sha, corpus_dataset=corpus.get("dataset_id")))


def entry(src, m):
    """A genome as the gates read it: the corpus manifest's taxid, source kind, proteins and accessions."""
    e = src["corpus"].get(m["id"])
    if e is None or int(e["taxid"]) != int(m["taxid"]):
        raise SystemExit("rel_w2_v0: %s of the training-data index is not a genome of the corpus manifest with the same taxid" % m["id"])
    return dict(id=m["id"], taxid=int(e["taxid"]), source_kind=e.get("source_kind", ""), proteins=int(e.get("proteins", 0)),
                accession=e.get("accession", ""), paired_accession=e.get("paired_accession", ""))


def training_gates(cfg, entries, tax, repo):
    """Gates G1 (the floors of the corpus spec) and G2 (the exclusions by full lineage) on exactly these genomes."""
    with open(repo(cfg["corpus_spec"])) as f:
        spec = yaml.safe_load(f)
    g1 = g1_kingdom.built(spec, entries, tax, cfg.get("tier"), KingdomMap())
    g1.gate = "G1_rel_w2_v0"
    g2 = g2_exclusion.entries(entries, tax, extra_paths=[repo(x) for x in cfg.get("extra_exclusions") or []])
    g2.gate = "G2_rel_w2_v0"
    return g1, g2


def open_genomes(src, s, sealed):
    """(index entries of the corpus genomes of the open splits that the sealed check does not refuse by name, the
    refused ones with the reasons, split sizes of the source)."""
    corpus = [m for m in src["index"]["genomes"] if m.get("source", "corpus") == "corpus"]
    sizes = {}
    for m in corpus:
        sizes[m["split"]] = sizes.get(m["split"], 0) + 1
    bad = {k: (sizes.get(k, 0), v) for k, v in s["expect"].items() if sizes.get(k, 0) != v}
    if bad:
        raise SystemExit("rel_w2_v0: split sizes of the source differ from the expected ones (found, expected): %s" % bad)
    metas, left = [], []
    for m in corpus:
        if m["split"] not in V.OPEN_SPLITS:
            continue
        why = sealed.reasons(m["id"])
        if why:
            left.append(dict(id=m["id"], split=m["split"], group=m["group"], reasons=why))
        else:
            metas.append(m)
    if len(left) > s["max_left_out"]:
        raise SystemExit("rel_w2_v0: the sealed check refuses %d genomes of the open splits by name (at most %d are left out): %s" % (
            len(left), s["max_left_out"], [x["id"] for x in left[:10]]))
    return metas, left, sizes


def build(st, cfg, run_dir):
    s, t0, say = settings(cfg), time.time(), lambda text: print("rel_w2_v0: " + text, flush=True)
    src = source(s["source"])
    run = src["run"]
    groups = sorted(KingdomMap().names)
    sealed = common.sealed_set(cfg)
    metas, left, sizes = open_genomes(src, s, sealed)
    unknown = sorted({m["group"] for m in metas} - set(groups))
    if unknown:
        raise SystemExit("rel_w2_v0: groups outside configs/kingdom_map.yml: %s" % unknown)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    ents = {m["id"]: entry(src, m) for m in metas}
    st.extra["sealed_check"] = common.sealed_check(cfg, list(ents.values()), splits={m["id"]: m["split"] for m in metas}, sealed=sealed, tax=tax)
    train = [m["id"] for m in metas if m["split"] == "train"]
    g1, g2 = training_gates(cfg, [ents[g] for g in train], tax, st.repo)
    if g1.status == "fail" or g2.status == "fail":
        raise SystemExit("rel_w2_v0: gates on the training genomes failed: %s" % json.dumps(dict(G1=g1.evidence.get("problems"), G2=dict(
            problems=g2.evidence.get("problems"), hits=g2.evidence.get("hits"))))[:2000])
    st.extra.update(source=src["provenance"], split_sizes=sizes, left_out=left, training_gates=dict(G1=g1.status, G2=g2.status, genomes=len(train)))
    # every file that will be read, against the source's manifests (training arrays and homolog rows of the open genomes only)
    order = SS.Pairs(run).order
    missing = sorted(m["id"] for m in metas if m["id"] not in set(order))
    if missing:
        raise SystemExit("rel_w2_v0: genomes without homolog rows in stage pairs: %s" % missing[:5])
    paths = [os.path.join("data", d, m["id"] + ".npz") for d in ("train_data", "pairs") for m in metas]
    shas = V.hash_files([os.path.join(run, p) for p in paths], s["processes"])
    bad = [p for p, sha in zip(paths, shas) if src["files"].get(p) != sha]
    if bad:
        raise SystemExit("rel_w2_v0: files of %s differ from its manifests: %s" % (run, bad[:5]))
    say("%d files of the source match its manifests, %.0f s" % (len(paths), time.time() - t0))
    genomes, stats = V.load_genomes(os.path.join(run, "data", "train_data"), metas, s["processes"])
    say("%d genomes loaded, %.0f s" % (len(genomes), time.time() - t0))
    ids = [m["id"] for m in metas]
    gix = {g: i for i, g in enumerate(ids)}
    base = os.path.join(run_dir, OUT)
    parts, counts, files, used = {}, {}, [], set()
    for part in PARTS:
        plan = V.plan_set(metas, groups, part, s[part + "_per_group"], s["seed"])
        folder = os.path.join(base, part)
        shards, census = V.build_set(SS.Pairs(run), genomes, train, ids, groups, part, plan, folder, rows=s["rows"], shard_tables=s["shard_tables"],
                                     seed=s["seed"], processes=s["processes"], progress=say)
        problems = V.verify_set(folder, shards, s["rows"], part, {gix[m["id"]] for m in metas if m["split"] == part}, {gix[g] for g in train}, len(groups),
                                s["processes"])
        if problems:
            raise SystemExit("rel_w2_v0: the %s tables are not sound: %s" % (part, problems[:10]))
        hashes = V.hash_shards(folder, shards, s["rows"], s["processes"])
        for name, _ in shards:
            for field in V.fields(s["rows"]):
                files.append(dict(path=os.path.join(OUT, part, "%s.%s.npy" % (name, field)), role="shard" if field in api.SHARD_FIELDS else "meta", **hashes[name][field]))
        examined, other = V.genomes_in(folder, shards)
        used |= {int(x) for x in examined} | {int(x) for x in other}
        parts[part] = dict(split=V.SPLITS[part], tables=len(plan[0]), per_group=s[part + "_per_group"], groups_without_a_genome=plan[2],
                           folder=os.path.join(OUT, part), shards=[dict(name=name, tables=m) for name, m in shards],
                           sha256=sha256_json(sorted([name, f, h["sha256"]] for name, fs in hashes.items() for f, h in fs.items())),
                           examined_genomes=len(examined), other_row_genomes=len(other), seconds=round(time.time() - t0, 1))
        counts[part] = {k: v.tolist() for k, v in census.to_arrays().items()}
        say("part %s: %d tables in %d shards, %.0f s" % (part, len(plan[0]), len(shards), time.time() - t0))
    if parts["train"]["groups_without_a_genome"]:
        raise SystemExit("rel_w2_v0: groups without a training genome: %s" % parts["train"]["groups_without_a_genome"])
    index = dict(format="rel-w2-v0-index", version=1, table_version=V.TABLE_VERSION, api_version=V.API_VERSION, rows=s["rows"], row_len=api.ROW_LEN,
                 blocks=api.N_BLOCKS, seed=s["seed"], file_pattern="<part folder>/<shard>.<field>.npy", shard_fields=sorted(api.SHARD_FIELDS),
                 meta_fields=sorted(api.META_FIELDS), groups=groups, parts=parts, source=src["provenance"],
                 kinds={"0": "nothing: no such row or no such block", str(api.KIND_PRESENT): "codons", str(api.KIND_NOT_SEARCHED): "no homolog row in stage pairs",
                        str(api.KIND_TRUNCATED): "a homolog row whose aligned codons lie outside the stored parts"},
                 genomes=[dict(id=m["id"], group=m["group"], split=m["split"], taxid=int(m["taxid"])) for m in metas])
    doc_counts = dict(format="rel-w2-v0-counts", version=1, rows=s["rows"], groups=groups, max_codons=src["index"].get("max_codons"), genomes=stats, parts=counts)
    for name, obj, role in (("index.json", index, "index"), ("counts.json", doc_counts, "meta")):
        p = os.path.join(base, name)
        write_json(p, obj)
        files.append(dict(path=os.path.join(OUT, name), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p), role=role))
        st.output("v0_" + name.split(".")[0], p)
    present = [g for g in groups if g not in parts["heldout"]["groups_without_a_genome"]]
    doc = handoff.finish(dict(
        format="rel-tableset", format_version=1, name="w2-v0-r%d-s%d" % (s["rows"], s["seed"]), kind=api.TABLE_REAL, api_version=V.API_VERSION,
        table_version=V.TABLE_VERSION, layout=api.LAYOUT_STORED,
        coverage=dict(groups=present, scenarios=["far relatives only: the medium corpus as it is, rows among the 64 nearest genomes of a gene"],
                      not_covered=NOT_COVERED, note=COVERAGE_NOTE),
        tables=sum(p["tables"] for p in parts.values()), rows=s["rows"], seed=s["seed"], parts=parts, files=files,
        genomes=[dict(id=ids[i], taxid=int(metas[i]["taxid"]), split=metas[i]["split"]) for i in sorted(used)],
        left_out=[dict(group=x["group"], split=x["split"], reasons=x["reasons"]) for x in left], source=src["provenance"],
        sealed_check=dict(passed=True, **st.extra["sealed_check"]),       # the record readers on every machine require (handoff.py)
        index=os.path.join(OUT, "index.json"),
        producer=dict(stage="w2_v0", package="W2", run_id=os.path.basename(run_dir), commit=cfg["git"]["sha"])))
    p = os.path.join(base, "tableset.json")
    write_json(p, doc)
    st.output("tableset", p)
    st.extra.update(tables={k: v["tables"] for k, v in parts.items()}, bytes=sum(f["bytes"] for f in files), tableset_sha256=doc["sha256"],
                    heldout_sha256=parts["heldout"]["sha256"], seconds=round(time.time() - t0, 1), processes=s["processes"])
    return doc


def gate(st, cfg, run_dir):
    import numpy as np
    s = settings(cfg)
    base = os.path.join(run_dir, OUT)
    p = os.path.join(base, "tableset.json")
    doc, index = _json(p), _json(os.path.join(base, "index.json"))
    src = source(s["source"])
    by = {m["id"]: m for m in src["index"]["genomes"]}
    listed = index.get("genomes") or []
    train = [g["id"] for g in listed if g.get("split") == "train"]
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    g1, g2 = training_gates(cfg, [entry(src, by[g]) for g in train if g in by], tax, st.repo)
    results = [run(g1, run_dir), run(g2, run_dir)]
    problems = [dict(e, part="manifest") for e in handoff.tableset_problems(doc)]
    if doc.get("kind") != api.TABLE_REAL or (doc.get("producer") or {}).get("stage") != "w2_v0" or doc.get("api_version") != V.API_VERSION:
        problems.append(dict(error="not a real table set of stage w2_v0 for this version of the contract"))
    if doc.get("source") != src["provenance"]:
        problems.append(dict(error="the manifest names another source than the configured run's records"))
    wrong = [g["id"] for g in listed if g["id"] not in by or by[g["id"]]["split"] != g.get("split") or g.get("split") not in V.OPEN_SPLITS]
    if wrong:
        problems.append(dict(error="genomes listed outside the open splits of the source, or with another split", genomes=wrong[:10]))
    paths = {f["path"]: f["role"] for f in doc.get("files") or [] if isinstance(f, dict)}
    occur = set()
    for part in PARTS:
        d = (doc.get("parts") or {}).get(part) or {}
        want = s[part + "_per_group"] * (len(index.get("groups") or []) - len(d.get("groups_without_a_genome") or []))
        if d.get("tables") != want or sum(x["tables"] for x in d.get("shards") or []) != want:
            problems.append(dict(part=part, error="%s tables, %d configured" % (d.get("tables"), want)))
        if part == "train" and d.get("groups_without_a_genome"):
            problems.append(dict(part=part, error="groups without training tables", groups=d["groups_without_a_genome"]))
        for x in d.get("shards") or []:
            for field in V.fields(s["rows"]):
                fp = os.path.join(OUT, part, "%s.%s.npy" % (x["name"], field))
                if paths.get(fp) != ("shard" if field in api.SHARD_FIELDS else "meta"):
                    problems.append(dict(part=part, file=fp, error="not in the manifest with its role"))
            g = np.load(os.path.join(base, part, x["name"] + ".genome.npy"))
            if g.shape != (x["tables"], s["rows"]) or g.max() >= len(listed):
                problems.append(dict(part=part, shard=x["name"], error="the genome field does not fit the genome list"))
                continue
            ex, other = {listed[i]["split"] for i in np.unique(g[:, 0])}, {listed[i]["split"] for i in np.unique(g[:, 1:]) if i >= 0}
            if ex != {part} or not other <= {"train"}:
                problems.append(dict(part=part, shard=x["name"], error="examined genomes of splits %s, other rows of splits %s" % (sorted(ex), sorted(other))))
            occur |= {int(i) for i in np.unique(g) if i >= 0}
    if sorted(g["id"] for g in doc.get("genomes") or []) != sorted(listed[i]["id"] for i in occur):
        problems.append(dict(error="the manifest's genomes are not the genomes of the rows"))
    sealed_ids = {m["id"] for m in src["index"]["genomes"] if m["split"] not in V.OPEN_SPLITS}
    named = sealed_ids & ({g["id"] for g in listed} | {g["id"] for g in doc.get("genomes") or []})
    if named:
        problems.append(dict(error="genomes outside the open splits are named", n=len(named)))
    results.append(run(GateResult("G4_rel_w2_v0", "fail" if problems else "pass", dict(
        problems=problems[:50], tables={k: (doc.get("parts") or {}).get(k, {}).get("tables") for k in PARTS}, genomes=len(occur), training_genomes=len(train),
        left_out=len(doc.get("left_out") or []), heldout_sha256=((doc.get("parts") or {}).get("heldout") or {}).get("sha256"))), run_dir))
    bad = [r.gate for r in results if r.status == "fail"]
    if bad:
        raise GateFailure("gates failed: %s" % ", ".join(bad))
    return sha256_json([sha256_file(p, use_sidecar=False), sha256_file(os.path.join(base, "index.json"), use_sidecar=False)])


def summary(run_dir):
    doc = _json(os.path.join(run_dir, OUT, "tableset.json"))
    size = sum(f["bytes"] for f in doc["files"])
    parts = doc["parts"]
    js = dict(name=doc["name"], rows=doc["rows"], tables={k: v["tables"] for k, v in parts.items()}, shards={k: len(v["shards"]) for k, v in parts.items()},
              bytes=size, genomes=len(doc["genomes"]), left_out=len(doc.get("left_out") or []), heldout_sha256=parts["heldout"]["sha256"],
              tableset_sha256=doc["sha256"], source_run=doc["source"]["run_id"], folder=OUT)
    md = ["Tables of version 0 (`%s`, %d rows stored): %s training tables in %d shards and %s held-out tables in %d shards under `%s/`, %.2f GB." % (
              doc["name"], doc["rows"], "{:,}".format(parts["train"]["tables"]), len(parts["train"]["shards"]), "{:,}".format(parts["heldout"]["tables"]),
              len(parts["heldout"]["shards"]), OUT, size / 1e9),
          "Held-out part sha256 `%s`; table-set manifest `%s`; %d genomes in rows; left out by the sealed check: %d." % (
              parts["heldout"]["sha256"], doc["sha256"], len(doc["genomes"]), len(doc.get("left_out") or []))]
    return dict(json=js, md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w2_v0") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
