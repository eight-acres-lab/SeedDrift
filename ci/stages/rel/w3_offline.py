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
manifest), conditions (default [scattered, suffix]), seed (the plan's, default conditions.SEED), check_seed (the random
codons of the test), shard_tables (2400).
Outputs under data/rel_w3/offline/: <condition>/ the shards of each part (the shard and meta fields of the contract, the
disk layout of seeddrift/train/rel/tables/v0_io.py), report.json, and tableset.json (written last: the table-set
manifest, kind real, one part per condition with its plan and the digest of its scored cells, the sealed record).
Gate G4_rel_w3_offline: the manifest is complete and matches its files, every part holds every held-out table, the
fourth test passed on every table, and the scored cells of every part are those of the stored tables under its plan.
"""
from __future__ import annotations

import json
import os
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
from seeddrift.train.rel.tables import v0_io as IO

STAGE = "rel_w3_offline"
OUT = os.path.join("data", "rel_w3", "offline")
KEYS = {"tables_from", "tablesets", "conditions", "seed", "check_seed", "shard_tables", "allow_kinds"}
FIELDS = tuple(api.SHARD_FIELDS) + tuple(api.META_FIELDS)
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
    return dict(tables_from=str(c["tables_from"]), tablesets=list(c["tablesets"]), conditions=conds, seed=int(c.get("seed", cond.SEED)),
                check_seed=int(c.get("check_seed", 20261123)), shard_tables=int(c.get("shard_tables", 2400)),
                allow_kinds=list(c.get("allow_kinds") or [api.TABLE_REAL]))


def heldout(s):
    """The manifest and the held-out part of the tables run."""
    docs = []
    for t in s["tablesets"]:
        try:
            doc = handoff.from_run(s["tables_from"], t["path"], t.get("sha256"))
        except handoff.HandoffError as err:
            raise SystemExit("%s: %s" % (STAGE, err))
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
    """The strict rendering of one table: with the whole gene hidden, the target block laid on its frame relative
    (AL.frame_layout); otherwise the correspondence aligned with the hidden residues unknown (AL.strict). Then leak-free.
    Returns the strict table and grid, truth, ext."""
    if cond.parse(condition)[0] == cond.WHOLE_GENE:
        col, frame_row, frame_len = AL.frame_layout(tab)
        strict = dict(tab, col=col, frame_row=frame_row, frame_len=frame_len)
    else:
        strict = dict(tab, col=AL.strict(tab, visible))
    grid, truth, ext = RD.render(strict, visible, target, RD.LEAKFREE)
    return strict, grid, truth, ext


def build(st, cfg, run_dir):
    s = settings(cfg)
    t0 = time.time()
    docs, held = heldout(s)
    genomes = {g["id"]: g for d in docs for g in d.get("genomes") or []}
    record = common.sealed_record(cfg, list(genomes.values()), splits={k: g.get("split") for k, g in genomes.items()}) if genomes else None
    arrays = held.load(0, held.tables, FIELDS)
    if (arrays["split"] != shards.SPLIT_HELDOUT).any():
        raise SystemExit("%s: a table to render is not a held-out table" % STAGE)
    base = os.path.join(run_dir, OUT)
    rng = np.random.default_rng(s["check_seed"])
    parts, files, report = {}, [], dict(conditions={})
    for c in s["conditions"]:
        tc = time.time()
        col = np.array(arrays["col"], copy=True)
        frame_row, frame_len = np.array(arrays["frame_row"], copy=True), np.array(arrays["frame_len"], copy=True)
        digest = cond.ItemsDigest(c, s["seed"])
        failures, kept, anchors, dropped, ext_cells, codons = [], 0, 0, 0, 0, 0
        for a in range(0, held.tables, CHUNK):
            b = min(a + CHUNK, held.tables)
            batch = cond.materialise({f: arrays[f][a:b] for f in ("tok", "n", "col", "kind")}, np.arange(a, b), c, s["seed"])
            digest.add(batch)
            for k in range(b - a):
                t = a + k
                tab = one(arrays, t)
                vis, target = batch["visible"][k], batch["target"][k]
                strict, grid, truth, ext = render_strict(tab, vis, target, c)
                new = strict["col"]
                col[t], frame_row[t], frame_len[t] = new, strict.get("frame_row", tab["frame_row"]), strict["frame_len"]
                dropped += RD.dropped(strict, vis, target)
                ext_cells += int(ext[1:].sum())
                codons += int(((grid[1:] >= api.TARGET_MIN) & (grid[1:] <= api.TARGET_MAX)).sum())
                old_anchor, new_anchor = _anchors(tab["col"], tab, vis), _anchors(new, strict, vis)
                anchors += int(old_anchor.sum())
                kept += int((old_anchor & new_anchor & (tab["col"] == new)).sum())
                gone = np.array(tab["col"], copy=True)                           # the fourth test: the hidden residues changed and
                for bb, (_, s0, e0) in enumerate(api.block_slices()):           # every stored position of a block with any of them
                    f0 = int(tab["frame_len"][bb])
                    if f0 and not vis[0, s0:s0 + f0].all():
                        gone[1:, s0:e0] = -1
                other = dict(tab, tok=np.where(target, rng.integers(1, 66, tab["tok"].shape), tab["tok"]).astype(np.uint8), col=gone)
                _, g2, _, e2 = render_strict(other, vis, target, c)
                if not (np.array_equal(grid, g2) and np.array_equal(ext, e2)):
                    failures.append(t)
        out = dict(arrays, col=col.astype(np.int16), frame_row=frame_row.astype(np.int8), frame_len=frame_len.astype(np.int16))
        folder = os.path.join(base, c)
        written = IO.write(folder, out, s["shard_tables"])
        for name, _ in written:
            for f in FIELDS:
                p = IO.shard_path(folder, name, f)
                files.append(dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p),
                                  role="shard" if f in api.SHARD_FIELDS else "meta"))
        parts[c] = dict(folder=os.path.relpath(folder, run_dir), split=shards.SPLIT_HELDOUT, tables=int(held.tables),
                        shards=[dict(name=n, tables=m) for n, m in written], condition=c, seed=s["seed"], plan_version=cond.VERSION,
                        items_sha256=digest.hexdigest(), layout=api.LAYOUT_LEAKFREE)
        report["conditions"][c] = dict(tables=int(held.tables), fourth_test_failures=len(failures), failed_tables=failures[:20],
                                       dropped_targets=int(dropped), stored_anchors=int(anchors), stored_anchors_kept=int(kept),
                                       other_row_codons_shown=int(codons), cells_laid_by_extension=int(ext_cells),
                                       items_sha256=digest.hexdigest(), seconds=round(time.time() - tc, 1))
    src = docs[0]
    doc = handoff.finish(dict(
        format="rel-tableset", format_version=1, name="w3-strict-v0-s%d" % s["seed"], kind=api.TABLE_REAL, api_version=api.API,
        table_version="%s on %s" % (TABLE_VERSION, src["table_version"]), layout=api.LAYOUT_LEAKFREE,
        coverage=dict(groups=list((src.get("coverage") or {}).get("groups") or []), scenarios=list((src.get("coverage") or {}).get("scenarios") or []),
                      not_covered=list((src.get("coverage") or {}).get("not_covered") or []) + ["the whole gene hidden (step two: the frame relative)"],
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
