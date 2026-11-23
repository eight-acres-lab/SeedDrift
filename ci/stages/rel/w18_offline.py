"""Stage rel_w18_offline (package W18; pipeline rel_tables, CPU only): the strict offline rendering of a ruler set
(protocol w18-v3), built before any reading. `python -m ci.stages.rel.w18_offline RUN_DIR`. Nothing is trained and no
model is run; every result on it stays "construct only".

For every table of the ruler set and every mask family its passes use (seeddrift/train/rel/readouts/w18_passes.py:
T scattered target, C scattered context, H1 and H2 either half of the target, W the whole target; the masks are the
passes' own, hashed on the triple, so the three tables of a triple stay comparable cell for cell), the placement of the
other rows is made by package W3's functions (seeddrift/train/rel/tables/align.py; no second aligner): align.strict
for T, C, H1 and H2 (the correspondence aligned to row 0 again wherever the mask hides residues of row 0, those
residues unknown), align.frame_layout for W (the target block laid on its frame relative). Every other row is aligned
to row 0 or to the frame relative alone, so a pass that removes rows keeps the placement of the rows it keeps.

W3's fourth validity test runs on every table and family (the hidden residues replaced by random codons, every stored
position of a block where row 0 hides a residue erased, the placement made again: W3's leak-free grid and its extension
marks the same, cell for cell); a table that fails stops the stage.

Profile block rel_w18_offline: tables {run, path[, sha256]} (the ruler set, a passed run of stage rel_w18_rulers, read
through common.read_tableset), processes (8), chunk (64 tables per task).
Outputs under data/rel_w18/offline/: the placements (w18_passes.offline_arrays: has.npy, <family>.col.npy,
W.frame_row.npy, W.frame_len.npy) and offline.json (written last: the ruler set, the plan, the aligner, every file with
its sha256, the fourth test, counts per family). run_ruler reads them (w18_passes.load_offline) for a leak-free reading.
Gate G4_rel_w18_offline: the files are the ones offline.json names, they cover every table of the ruler set, every
table holds the placement of every mask family its passes use and of no other (the plan recomputed table by table), and
no table failed the fourth test. What the test does not cover (VALIDITY_LIMITS): which rows a block holds, their kinds
and stretch lengths come from the search, which saw the hidden gene; only the presence passes measure them.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.readouts import w18_passes as P

STAGE = "rel_w18_offline"
OUT = P.OFFLINE_DIR
KEYS = {"tables", "processes", "chunk"}
FIELDS = P.FIELDS


def settings(cfg) -> dict:
    c = cfg.get("rel_w18_offline") or {}
    unknown = set(c) - KEYS
    t = c.get("tables") or {}
    if unknown or not t.get("run") or not t.get("path"):
        raise SystemExit("%s: needs tables {run, path[, sha256]}; unknown keys %s" % (STAGE, sorted(unknown)))
    out = dict(tables=dict(run=str(t["run"]), path=str(t["path"]), sha256=t.get("sha256")), processes=int(c.get("processes", 8)),
               chunk=int(c.get("chunk", 64)))
    if not (1 <= out["processes"] <= 16 and 1 <= out["chunk"] <= 4096):
        raise SystemExit("%s: settings out of range: %r" % (STAGE, out))
    return out


def ruler(s):
    """(manifest, TableSet, W18 meta) of the ruler set, read through the registry's reader."""
    doc = common.read_tableset("w18_offline", s["tables"]["run"], s["tables"]["path"], s["tables"]["sha256"])
    if doc.get("kind") != "ruler":
        raise SystemExit("%s: the table set is kind %r, not ruler" % (STAGE, doc.get("kind")))
    sets = shards.by_split(s["tables"]["run"], [doc])
    if list(sets) != [1]:
        raise SystemExit("%s: a ruler set of held-out tables only (splits %s)" % (STAGE, sorted(sets)))
    return doc, sets[1], P.load_meta(s["tables"]["run"], doc)


BLOCK_START = np.array([P.SLICES[b][1] for b in P.BLOCK_OF], dtype=np.int64)     # the first slot of every slot's block


def _anchors(col, tab, vis):
    """[R, ROW_LEN] the stored pairs of rows 1.. (col >= 0, inside the frame) whose two residues are visible."""
    n = np.asarray(tab["n"]).astype(np.int64)
    c = np.asarray(col).astype(np.int64)
    f = np.asarray(tab["frame_len"]).astype(np.int64)[P.BLOCK_OF]
    ok = (P.OFFSET_OF[None, :] < n[:, P.BLOCK_OF]) & (c >= 0) & (c < f[None, :])
    ok[0] = False
    rr, ss = np.nonzero(ok)
    fv = np.zeros_like(ok)
    fv[rr, ss] = vis[0, BLOCK_START[ss] + c[rr, ss]]
    return ok & vis & fv


def chunk(job):
    """The placements and the fourth test of tables a..a+k-1: (a, [ {family: (compact placement, passed, counts)} ])."""
    a, fields, row_tier, parent = job
    out = []
    for k in range(len(row_tier)):
        t = a + k
        tab = {f: fields[f][k] for f in FIELDS}
        got = {}
        for fam, p in P.family_passes(tab, row_tier[k], t, int(parent[k])).items():
            pl = P.strict_placement(tab, fam, p.visible)
            ok, grid, ext = P.fourth_test(tab, fam, p, pl, t)
            counts = dict(ext=int(ext[1:].sum()), shown=int(((grid[1:] >= 1) & (grid[1:] <= 65)).sum()))
            if fam != "W":
                old = _anchors(tab["col"], tab, p.visible)
                counts["stored"] = int(old.sum())
                counts["kept"] = int((old & (np.asarray(pl["col"]) == np.asarray(tab["col"]))).sum())
            else:
                counts["frame_relative"] = int(pl["frame_row"][P.TARGET_BLOCK])
            one = P.offline_arrays(1, tab["tok"].shape[0])
            P.offline_put(one, 0, fam, pl, tab)
            got[fam] = ({kk: v[0] for kk, v in one.items()}, ok, counts)
        out.append(got)
    return a, out


def build(st, cfg, run_dir):
    s, t0 = settings(cfg), time.time()
    say = lambda text: print("%s: %s" % (STAGE, text), flush=True)          # noqa: E731
    doc, ts, meta = ruler(s)
    N = ts.tables
    R = int(ts.load(0, 1, ("tok",))["tok"].shape[1])
    arrays = P.offline_arrays(N, R)
    report = {f: dict(tables=0, failures=0, failed_tables=[], ext_cells=0, other_row_codons_shown=0, stored_pairs=0, stored_pairs_kept=0,
                      frame_relative_not_row1=0) for f in P.FAMILIES}
    jobs = []
    for a in range(0, N, s["chunk"]):
        b = min(a + s["chunk"], N)
        jobs.append((a, ts.load(a, b, FIELDS), np.asarray(meta["row_tier"][a:b]), np.asarray(meta["parent"][a:b])))
    if s["processes"] > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(s["processes"]) as pool:
            results = pool.imap_unordered(chunk, jobs)
            done = _collect(results, arrays, report, N, say)
    else:
        done = _collect(map(chunk, jobs), arrays, report, N, say)
    failures = sum(r["failures"] for r in report.values())
    if failures:
        raise SystemExit("%s: %d placements failed W3's fourth validity test (tables %s)" % (
            STAGE, failures, sorted({t for r in report.values() for t in r["failed_tables"]})[:20]))
    base = os.path.join(run_dir, OUT)
    os.makedirs(base, exist_ok=True)
    files = []
    for name, v in arrays.items():
        p = os.path.join(base, name + ".npy")
        np.save(p, v)
        files.append(dict(name=name, path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)))
        st.output("w18_offline_" + name.replace(".", "_"), p)
    from seeddrift.train.rel.tables import align as AL
    from seeddrift.train.rel.tables import render as RD
    head = dict(format=P.OFFLINE_FORMAT, protocol="w18-v3", run=os.path.basename(os.path.normpath(run_dir)),
                ruler=dict(run=os.path.basename(os.path.normpath(s["tables"]["run"])), path=s["tables"]["path"], sha256=doc.get("sha256"),
                           name=doc.get("name"), tables=int(N), rows=R),
                plan=dict(version=P.VERSION, seed=P.SEED, rate=P.RATE), families=list(P.FAMILIES),
                aligner=dict(version=AL.VERSION, module="seeddrift/train/rel/tables/align.py", strict="align.strict (T, C, H1, H2)",
                             whole="align.frame_layout (W)"),
                render=dict(version=RD.VERSION, layout=RD.LEAKFREE),
                fourth_test=dict(failures=int(failures), checked=int(done)), validity_limits=VALIDITY_LIMITS,
                families_report=report, files=files, seconds=round(time.time() - t0, 1))
    p = os.path.join(base, P.OFFLINE_FILE)
    write_json(p, head)
    st.output("w18_offline", p)
    say("%d tables, %d placements checked, no failure, %.0f s" % (N, done, time.time() - t0))


def _collect(results, arrays, report, N, say):
    done, seen = 0, 0
    for a, got in results:
        for k, fams in enumerate(got):
            t = a + k
            for fam, (one, ok, counts) in fams.items():
                arrays["has"][t, P.FAMILIES.index(fam)] = True
                for name, v in one.items():
                    if name == "has":
                        continue
                    if name.startswith(fam + ".") or (fam == "W" and name.startswith("W.")):
                        arrays[name][t] = v
                r = report[fam]
                r["tables"] += 1
                done += 1
                if not ok:
                    r["failures"] += 1
                    if len(r["failed_tables"]) < 20:
                        r["failed_tables"].append(int(t))
                r["ext_cells"] += counts["ext"]
                r["other_row_codons_shown"] += counts["shown"]
                r["stored_pairs"] += counts.get("stored", 0)
                r["stored_pairs_kept"] += counts.get("kept", 0)
                r["frame_relative_not_row1"] += int(fam == "W" and counts.get("frame_relative", 1) != 1)
        seen += len(got)
        if seen and seen % 1000 < len(got):
            say("%d of %d tables" % (seen, N))
    return done


VALIDITY_LIMITS = ("W3's fourth test covers the placement only (col, frame_row, frame_len and the cells laid by extension): which rows a block "
                   "holds, their kinds and the lengths of their stretches come from the search, which was run with the hidden gene, and stay "
                   "as they are in the test; only the presence passes measure what they carry (protocol w3-offline-v0 version 3, validity_limits)")


def missing_families(ts, meta, has, chunk=256):
    """[(table, family)] where the plan of the passes makes a family whose placement `has` (offline arrays' has) lacks,
    or holds a placement of a family the plan does not make: every table and every mask family (protocol w18-v3)."""
    out = []
    for a in range(0, ts.tables, chunk):
        b = min(a + chunk, ts.tables)
        got = ts.load(a, b, FIELDS)
        for k in range(b - a):
            t = a + k
            planned = set(P.family_passes({f: got[f][k] for f in FIELDS}, meta["row_tier"][t], t, int(meta["parent"][t])))
            held = {f for i, f in enumerate(P.FAMILIES) if has[t, i]}
            out += [(int(t), f) for f in sorted(planned ^ held)]
    return out


def gate(st, cfg, run_dir):
    s = settings(cfg)
    p = os.path.join(run_dir, OUT, P.OFFLINE_FILE)
    problems = []
    with open(p) as f:
        head = json.load(f)
    doc, ts, meta = ruler(s)
    try:
        arrays, _, _ = P.load_offline(run_dir, doc.get("sha256"), passed=False)
    except ValueError as err:
        arrays = None
        problems.append(dict(error=str(err)))
    if arrays is not None:
        if arrays["has"].shape[0] != ts.tables or arrays["T.col"].shape[0] != ts.tables:
            problems.append(dict(error="the placements do not cover every table of the ruler set"))
        else:
            miss = missing_families(ts, meta, np.asarray(arrays["has"]))
            if miss:
                problems.append(dict(error="tables whose placements are not those of the families their passes use", tables=len({t for t, _ in miss}),
                                     first=miss[:10]))
    if int((head.get("fourth_test") or {}).get("failures", -1)) != 0:
        problems.append(dict(error="a placement failed W3's fourth validity test"))
    require(GateResult("G4_rel_w18_offline", "fail" if problems else "pass", dict(problems=problems[:20], tables=(head.get("ruler") or {}).get("tables"),
                                                                              checked=(head.get("fourth_test") or {}).get("checked"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, P.OFFLINE_FILE)) as f:
        head = json.load(f)
    fam = head["families_report"]
    md = ["W18 strict offline rendering of `%s` (%d tables, protocol w18-v3, construct only): %d placements, W3's fourth test passed on all." % (
              head["ruler"]["name"], head["ruler"]["tables"], head["fourth_test"]["checked"]),
          "Stored pairs kept by the aligner (T, C, H1, H2): " + "; ".join(
              "%s %s" % (f, ("%.3f" % (fam[f]["stored_pairs_kept"] / fam[f]["stored_pairs"])) if fam[f]["stored_pairs"] else "-")
              for f in P.FAMILIES if f != "W"),
          "What the test does not cover: " + VALIDITY_LIMITS + "."]
    return dict(json=dict(ruler=head["ruler"], fourth_test=head["fourth_test"]), md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
