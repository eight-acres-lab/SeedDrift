"""Stage rel_w3_offline (package W3; pipeline rel_tables): the strict offline rendering of the held-out evaluation
items. `python -m ci.stages.rel.w3_offline RUN_DIR`.

For every kind of mask named (scattered, suffix, whole_gene), the held-out tables of a passed tables run are written
again as one part of a table set. With scattered cells or a suffix hidden, the correspondence is aligned anew wherever
the condition hides residues of the frame row: the hidden residues are unknown to the aligner (seeddrift/train/rel/tables/align.py: BLOSUM62, gaps 11 +
1 per position, an unknown residue -1 against anything, semi-global), so the correspondence depends on the length of a
hidden stretch and never on its content. Evaluation renders such a part with the leak-free layout
(seeddrift/train/rel/tables/render.py) and the same plan (seeddrift/train/rel/evalh/conditions.py): the scored cells and
their truth are those of the stored layout (the frame row keeps its own columns), only the placement of the relatives
changes. With the whole gene hidden, the target block is laid on its frame relative (the first other row, in the order
of the table, with a stretch there: frame_row of the block), the examined row from the first column, every other row
by its alignment to the frame relative, never through the examined row (AL.frame_layout); the re-paired condition
takes this part, the conditions without relatives the stored tables. Everything else of a table (its rows, blocks,
tokens, metadata) is copied as it is.

The fourth validity test runs on every table of every part: the hidden residues are replaced by random codons and every
stored position of a block with a hidden residue is erased, the correspondence is aligned again from the changed table
and the table is rendered: the grid and the cells laid by extension must be the same, cell for cell.

Profile block rel_w3_offline: tables_from (the run directory of the passed tables run), tablesets [{path, sha256}] (its
manifest), conditions (default [scattered, suffix, whole_gene]), seed (the plan's, default conditions.SEED), check_seed
(the random codons of the test, drawn per table), shard_tables (2400), allow_kinds (default [real]; [synthetic] for
tests; a ruler set is refused), processes (1: the tables of a part rendered by that many processes; the output is the
same whatever the number). Ruler sets (W18's implant tables) are rendered by stage rel_w18_offline alone: its passes have
masks of their own (hashed on the triple), which the plans of this stage do not reproduce.
Tables of version 0 and 1 alike (32 rows, ABSENT and TRUNCATED blocks, self-copy rows, the fields the contract adds):
every row with codons on a block (KIND_PRESENT or KIND_SELF_COPY) is a relative to align and may be the frame relative;
a block without codons is rendered by its kind. Every field the source holds is copied (with the added fields of
version 1, written as version 1), and every other file its manifest lists (the source's index and census, say) is
copied unchanged under source/ (manifest key source_files: where each came from). The manifest takes the source's
name (w3-strict-<source>-s<seed>), kind, purpose and group names.
Outputs under data/rel_w3/offline/: <condition>/ the shards of each part (the shard and meta fields of the contract, the
disk layout of seeddrift/train/rel/tables/v0_io.py), report.json, and tableset.json (written last: the table-set
manifest, kind real, one part per condition with its plan and the digest of its scored cells, the sealed record).
Gate G4_rel_w3_offline: the manifest is complete and matches its files, every part holds every held-out table, the
fourth test passed on every table, and the scored cells of every part are those of the stored tables under its plan.
"""
from __future__ import annotations

import json
import os
import shutil
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel import api, handoff
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.tables import align as AL
from seeddrift.train.rel.tables import render as RD
from seeddrift.train.rel.tables import strict as ST
from seeddrift.train.rel.tables import v0_io as IO

STAGE = "rel_w3_offline"
OUT = os.path.join("data", "rel_w3", "offline")
KEYS = {"tables_from", "tablesets", "conditions", "seed", "check_seed", "shard_tables", "allow_kinds", "processes"}
FIELDS = tuple(api.SHARD_FIELDS) + tuple(api.META_FIELDS)
KINDS = (api.TABLE_REAL, api.TABLE_SYNTHETIC)                     # a ruler set is W18's (stage rel_w18_offline)
RULER_REFUSED = ("a ruler set is rendered by stage rel_w18_offline, with the masks of W18's passes (hashed on the triple); "
                 "the plans of this stage would place other cells")
TABLE_VERSION = "w3-strict-%s" % AL.VERSION
CHUNK = 64


def settings(cfg) -> dict:
    c = dict(cfg.get(STAGE) or {})
    unknown = set(c) - KEYS
    if unknown or not c.get("tables_from") or not c.get("tablesets"):
        raise SystemExit("%s: the profile names tables_from and tablesets (keys: %s): %s" % (STAGE, sorted(KEYS), sorted(unknown)))
    conds = list(c.get("conditions") or [cond.SCATTERED, cond.SUFFIX, cond.WHOLE_GENE])
    for x in conds:
        mask, mode = cond.parse(x)
        if mode:
            raise SystemExit("%s: a part is rendered for a mask (scattered, suffix, whole_gene), not for %r: re-paired and "
                             "no-relatives conditions take the whole_gene part and the stored tables" % (STAGE, x))
    kinds = list(c.get("allow_kinds") or [api.TABLE_REAL])
    if any(k not in KINDS for k in kinds):
        raise SystemExit("%s: allow_kinds takes %s, not %s: %s" % (STAGE, list(KINDS), kinds, RULER_REFUSED))
    return dict(tables_from=str(c["tables_from"]), tablesets=list(c["tablesets"]), conditions=conds, seed=int(c.get("seed", cond.SEED)),
                check_seed=int(c.get("check_seed", 20261123)), shard_tables=int(c.get("shard_tables", 2400)),
                allow_kinds=kinds, processes=max(1, int(c.get("processes", 1))))


def heldout(s):
    """The manifest and the held-out part of the tables run."""
    docs = []
    for t in s["tablesets"]:
        try:
            doc = handoff.from_run(s["tables_from"], t["path"], t.get("sha256"))
        except handoff.HandoffError as err:
            raise SystemExit("%s: %s" % (STAGE, err))
        if doc["kind"] == api.TABLE_RULER:
            raise SystemExit("%s: table set %s is a ruler set: %s" % (STAGE, doc["name"], RULER_REFUSED))
        if doc["kind"] not in s["allow_kinds"]:
            raise SystemExit("%s: table set %s is %s; this profile accepts %s" % (STAGE, doc["name"], doc["kind"], s["allow_kinds"]))
        docs.append(doc)
    try:
        sets = shards.by_split(s["tables_from"], docs)
    except ValueError as err:
        raise SystemExit("%s: %s" % (STAGE, err))
    if shards.SPLIT_HELDOUT not in sets:
        raise SystemExit("%s: the tables run has no held-out part" % STAGE)
    return docs, sets[shards.SPLIT_HELDOUT]


def one(tables, k):
    tab = {f: tables[f][k] for f in ("tok", "n", "col", "kind")}
    tab["frame_len"], tab["frame_row"] = tables["frame_len"][k], tables["frame_row"][k]
    return tab


def render_strict(tab, visible, target, condition=cond.SCATTERED):
    """The strict rendering of one table (seeddrift/train/rel/tables/strict.py, the one the training uses): with the
    whole gene hidden, the target block laid on its frame relative; otherwise the correspondence aligned with the hidden
    residues unknown. Then leak-free. Returns the strict table and grid, truth, ext."""
    grid, truth, ext, placed = ST.render(tab, visible, target, whole_gene=cond.parse(condition)[0] == cond.WHOLE_GENE)
    return placed, grid, truth, ext


def render_tables(arrays, condition, seed, check_seed, first, last):
    """The strict rendering of tables first..last-1 of a part under one condition, with the fourth test on each: the
    new col, frame_row and frame_len, the tables that failed the test and the counts of the report. The test's random
    codons are drawn per table (check_seed, table), so the result does not depend on how the tables are split."""
    n = last - first
    col = np.array(arrays["col"][first:last], copy=True)
    frame_row, frame_len = np.array(arrays["frame_row"][first:last], copy=True), np.array(arrays["frame_len"][first:last], copy=True)
    out = dict(failures=[], kept=0, anchors=0, dropped=0, ext=0, codons=0)
    for a in range(first, last, CHUNK):
        b = min(a + CHUNK, last)
        batch = cond.materialise({f: arrays[f][a:b] for f in ("tok", "n", "col", "kind")}, np.arange(a, b), condition, seed)
        for k in range(b - a):
            t = a + k
            tab = one(arrays, t)
            vis, target = batch["visible"][k], batch["target"][k]
            strict, grid, truth, ext = render_strict(tab, vis, target, condition)
            new = strict["col"]
            col[t - first], frame_row[t - first], frame_len[t - first] = new, strict.get("frame_row", tab["frame_row"]), strict["frame_len"]
            out["dropped"] += RD.dropped(strict, vis, target)
            out["ext"] += int(ext[1:].sum())
            out["codons"] += int(((grid[1:] >= api.TARGET_MIN) & (grid[1:] <= api.TARGET_MAX)).sum())
            old_anchor, new_anchor = _anchors(tab["col"], tab, vis), _anchors(new, strict, vis)
            out["anchors"] += int(old_anchor.sum())
            out["kept"] += int((old_anchor & new_anchor & (tab["col"] == new)).sum())
            if not ST.fourth_test(tab, vis, target, whole_gene=cond.parse(condition)[0] == cond.WHOLE_GENE, key=t, seed=check_seed)["passed"]:
                out["failures"].append(t)                                       # the fourth test (tables/strict.py)
    out.update(col=col, frame_row=frame_row, frame_len=frame_len, tables=n)
    return out


_SHARED = {}


def _render_unit(args):
    return render_tables(_SHARED["arrays"], *args)


def render_part(arrays, condition, seed, check_seed, processes=1):
    """render_tables over every table of a part, in `processes` processes (forked, the arrays shared): the same result
    whatever the number."""
    N = int(arrays["tok"].shape[0])
    if processes <= 1 or N <= CHUNK:
        return render_tables(arrays, condition, seed, check_seed, 0, N)
    import multiprocessing as mp
    step = max(CHUNK, -(-N // (4 * processes)))
    units = [(condition, seed, check_seed, a, min(a + step, N)) for a in range(0, N, step)]
    _SHARED["arrays"] = arrays
    try:
        with mp.get_context("fork").Pool(processes) as pool:
            got = pool.map(_render_unit, units)
    finally:
        _SHARED.clear()
    out = dict(failures=sorted(t for g in got for t in g["failures"]))
    for k in ("kept", "anchors", "dropped", "ext", "codons"):
        out[k] = sum(g[k] for g in got)
    for k in ("col", "frame_row", "frame_len"):
        out[k] = np.concatenate([g[k] for g in got])
    return out


def build(st, cfg, run_dir):
    s = settings(cfg)
    t0 = time.time()
    docs, held = heldout(s)
    genomes = {g["id"]: g for d in docs for g in d.get("genomes") or []}
    record = common.sealed_record(cfg, list(genomes.values()), splits={k: g.get("split") for k, g in genomes.items()}) if genomes else None
    arrays = held.load(0, held.tables, list(held.fields))                # every field the set holds
    if (arrays["split"] != shards.SPLIT_HELDOUT).any():
        raise SystemExit("%s: a table to render is not a held-out table" % STAGE)
    added = [f for f in arrays if f in shards.ADDED]
    version = 1 if added else 0
    if added and sorted(added) != sorted(shards.ADDED):
        raise SystemExit("%s: the set holds some of the fields of table version 1 and not the others: %s" % (STAGE, sorted(added)))
    kinds = {d["kind"] for d in docs}
    if len(kinds) != 1:
        raise SystemExit("%s: the table sets read are of different kinds: %s" % (STAGE, sorted(kinds)))
    base = os.path.join(run_dir, OUT)
    parts, files, report = {}, [], dict(conditions={})
    for ci, c in enumerate(s["conditions"]):
        tc = time.time()
        digest = cond.ItemsDigest(c, s["seed"])
        for a in range(0, held.tables, 512):
            digest.add(cond.materialise({f: arrays[f][a:a + 512] for f in ("tok", "n", "col", "kind")}, np.arange(a, min(a + 512, held.tables)), c, s["seed"]))
        got = render_part(arrays, c, s["seed"], s["check_seed"] + 1000003 * ci, s["processes"])
        col, frame_row, frame_len = got["col"], got["frame_row"], got["frame_len"]
        failures, kept, anchors, dropped, ext_cells, codons = (got[k] for k in ("failures", "kept", "anchors", "dropped", "ext", "codons"))
        out = dict(arrays, col=col.astype(np.int16), frame_row=frame_row.astype(np.int8), frame_len=frame_len.astype(np.int16))
        folder = os.path.join(base, c)
        written = IO.write(folder, out, s["shard_tables"], version)
        for name, _ in written:
            for f in arrays:
                p = IO.shard_path(folder, name, f)
                files.append(dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p),
                                  role="shard" if f in shards.SHARD_KNOWN else "meta"))
        parts[c] = dict(folder=os.path.relpath(folder, run_dir), split=shards.SPLIT_HELDOUT, tables=int(held.tables),
                        shards=[dict(name=n, tables=m) for n, m in written], condition=c, seed=s["seed"], plan_version=cond.VERSION,
                        items_sha256=digest.hexdigest(), layout=api.LAYOUT_LEAKFREE)
        report["conditions"][c] = dict(tables=int(held.tables), fourth_test_failures=len(failures), failed_tables=failures[:20],
                                       dropped_targets=int(dropped), stored_anchors=int(anchors), stored_anchors_kept=int(kept),
                                       other_row_codons_shown=int(codons), cells_laid_by_extension=int(ext_cells),
                                       items_sha256=digest.hexdigest(), seconds=round(time.time() - tc, 1))
    source_files = []
    for d in docs:                                                       # the source's other files, unchanged
        for e in d.get("files") or []:
            if e.get("role") == "shard" or (e.get("role") == "meta" and e["path"].endswith(".npy") and len(os.path.basename(e["path"]).split(".")) == 3):
                continue
            to = os.path.join(base, "source", d["name"], e["path"])
            os.makedirs(os.path.dirname(to), exist_ok=True)
            shutil.copyfile(os.path.join(s["tables_from"], e["path"]), to)
            files.append(dict(path=os.path.relpath(to, run_dir), sha256=sha256_file(to, use_sidecar=False), bytes=os.path.getsize(to), role=e["role"]))
            if files[-1]["sha256"] != e.get("sha256", files[-1]["sha256"]):
                raise SystemExit("%s: %s of %s is not the file its manifest lists" % (STAGE, e["path"], d["name"]))
            source_files.append(dict(path=files[-1]["path"], source=e["path"], tableset=d["name"], role=e["role"]))
    src = docs[0]
    left = [] if cond.WHOLE_GENE in s["conditions"] else ["the whole gene hidden (no part rendered on the frame relative)"]
    doc = handoff.finish(dict(
        format="rel-tableset", format_version=1, name="w3-strict-%s-s%d" % ("+".join(d["name"] for d in docs), s["seed"]), kind=src["kind"],
        api_version=api.API, rows=int(arrays["tok"].shape[1]), source_table_version=version, source_files=source_files,
        **{k: src[k] for k in ("purpose", "not_for_training", "group_names") if src.get(k) is not None},
        table_version="%s on %s" % (TABLE_VERSION, src["table_version"]), layout=api.LAYOUT_LEAKFREE,
        coverage=dict(groups=list((src.get("coverage") or {}).get("groups") or []), scenarios=list((src.get("coverage") or {}).get("scenarios") or []),
                      not_covered=list((src.get("coverage") or {}).get("not_covered") or []) + left,
                      note="the held-out tables of %s, one part per evaluation condition, their correspondence aligned with the condition's hidden "
                           "residues unknown; render with the leak-free layout and the same plan" % src["name"]),
        tables=int(held.tables) * len(parts), parts=parts, files=files, genomes=list(genomes.values()),
        aligner=dict(version=AL.VERSION, matrix="BLOSUM62", gap_open=AL.GAP_OPEN, gap_extend=AL.GAP_EXTEND, unknown=AL.UNKNOWN, mode="semi-global"),
        renderer=dict(version=RD.VERSION, layout=api.LAYOUT_LEAKFREE),
        source=dict(run=os.path.basename(os.path.normpath(s["tables_from"])), tablesets=[dict(name=d["name"], sha256=d["sha256"]) for d in docs]),
        **({"sealed_check": record} if record else {}),
        producer=dict(stage="w3_offline", package="W3", run_id=os.path.basename(run_dir), commit=(cfg.get("git") or {}).get("sha", ""))))
    report.update(seconds=round(time.time() - t0, 1), tables=int(held.tables), manifest_sha256=doc["sha256"])
    rp = os.path.join(base, "report.json")
    write_json(rp, report)
    st.output("report", rp)
    mp = os.path.join(base, "tableset.json")
    write_json(mp, doc)
    st.output("tableset", mp)
    st.extra.update(conditions=s["conditions"], fourth_test_failures={c: r["fourth_test_failures"] for c, r in report["conditions"].items()})
    return doc


def _anchors(col, tab, vis):
    """[R, ROW_LEN] the pairs of col whose two residues are visible (a block's frame row excluded)."""
    n = np.asarray(tab["n"]).astype(np.int64)
    cells = RD.cells_of(n)
    c = np.asarray(col).astype(np.int64)
    f = np.asarray(tab["frame_len"]).astype(np.int64)[RD.BLOCK_OF]
    fr = RD.frame_rows(tab)[RD.BLOCK_OF]
    ok = cells & (c >= 0) & (c < f[None, :])
    dest = np.where(ok, RD.START[None, :] + np.maximum(c, 0), 0)
    v = np.asarray(vis, bool)
    out = ok & v & v[fr[None, :], dest]
    out[np.arange(c.shape[0])[:, None] == fr[None, :]] = False
    return out


def gate(st, cfg, run_dir):
    base = os.path.join(run_dir, OUT)
    with open(os.path.join(base, "tableset.json")) as f:
        doc = json.load(f)
    with open(os.path.join(base, "report.json")) as f:
        rep = json.load(f)
    problems = [dict(e, check="manifest") for e in handoff.tableset_problems(doc, run_dir)]
    s = settings(cfg)
    try:
        sets = shards.parts(run_dir, doc)
    except ValueError as err:
        sets = {}
        problems.append(dict(error=str(err)))
    _, held = heldout(s)
    for c in s["conditions"]:
        r, p = (rep.get("conditions") or {}).get(c) or {}, (doc.get("parts") or {}).get(c) or {}
        if not r or r.get("fourth_test_failures") != 0:
            problems.append(dict(condition=c, error="the fourth test did not pass on every table", failures=r.get("fourth_test_failures"), tables=r.get("failed_tables")))
        if c not in sets or sets[c].tables != held.tables or p.get("tables") != held.tables:
            problems.append(dict(condition=c, error="the part does not hold every held-out table"))
            continue
        mine, stored = cond.ItemsDigest(c, p.get("seed", s["seed"])), cond.ItemsDigest(c, p.get("seed", s["seed"]))
        for a in range(0, held.tables, 512):
            b = min(a + 512, held.tables)
            mine.add(cond.materialise(sets[c].load(a, b, ("tok", "n", "col", "kind")), np.arange(a, b), c, p.get("seed", s["seed"])))
            stored.add(cond.materialise(held.load(a, b, ("tok", "n", "col", "kind")), np.arange(a, b), c, p.get("seed", s["seed"])))
        if not (mine.hexdigest() == stored.hexdigest() == p.get("items_sha256")):
            problems.append(dict(condition=c, error="the scored cells differ from those of the stored tables under the same plan"))
    require(GateResult("G4_rel_w3_offline", "fail" if problems else "pass", dict(problems=problems[:50], conditions=s["conditions"],
                                                                                 report={c: {k: v for k, v in r.items() if k != "failed_tables"}
                                                                                         for c, r in (rep.get("conditions") or {}).items()})), run_dir)
    return sha256_file(os.path.join(base, "tableset.json"), use_sidecar=False)


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        rep = json.load(f)
    md = ["| condition | tables | fourth test failures | stored anchors kept | cells laid by extension | dropped targets |", "|---|---|---|---|---|---|"]
    for c, r in rep["conditions"].items():
        md.append("| %s | %d | %d | %.3f | %d | %d |" % (c, r["tables"], r["fourth_test_failures"], r["stored_anchors_kept"] / max(r["stored_anchors"], 1),
                                                       r["cells_laid_by_extension"], r["dropped_targets"]))
    return dict(json={c: {k: v for k, v in r.items() if k != "failed_tables"} for c, r in rep["conditions"].items()}, md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
