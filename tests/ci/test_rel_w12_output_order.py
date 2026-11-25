"""An expanded corpus can preserve old table indices without changing a DIAMOND call or alignment."""
import json
import os
import stat
import sys

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w12_search as T12
import test_rel_w2_stage as TS
from ci.gates import GateFailure
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import w12_search as WS
from seeddrift.train.rel import handoff
from seeddrift.train.rel.tables import v1_hits as H


def reference(tmp, genomes):
    run = tmp / "old-tables"
    base = run / "data/rel_w12/v1"
    base.mkdir(parents=True)
    (run / "stages").mkdir()
    idx = base / "index.json"
    idx.write_text(json.dumps(dict(genomes=[{k: g[k] for k in ("id", "group", "split", "taxid")} for g in genomes])))
    counts = base / "counts.json"
    counts.write_text("{}")
    doc = handoff.finish(dict(files=[dict(path=str(p.relative_to(run)), sha256=sha256_file(str(p), use_sidecar=False)) for p in (idx, counts)]))
    manifest = base / "tableset.json"
    manifest.write_text(json.dumps(doc))
    (run / "record.json").write_text(json.dumps(dict(run_id=run.name, status="passed")))
    (run / "stages/rel_w12_v1.json").write_text(json.dumps(dict(status="passed", outputs=[dict(name="tableset", sha256=sha256_file(str(manifest), use_sidecar=False))])))
    return dict(run=str(run), sha256=doc["sha256"])


def alignments(run):
    genomes = json.load(open(os.path.join(run, WS.OUT, "genomes.json")))["genomes"]
    ids = [g["id"] for g in genomes]
    out = {}
    index = H.Index(os.path.join(run, WS.OUT, "index"))
    for qg, g in enumerate(genomes):
        a = H.load(os.path.join(run, WS.OUT, g["id"] + ".npz"))
        for qi in range(g["genes"]):
            rows = H.gene_rows(a, qi)
            out[g["id"], qi] = sorted((ids[r[0]], *r[1:]) for r in rows)
            for tg in set(r[0] for r in rows):
                # The lookup index uses exactly the same renumbering as the per-genome arrays.
                expected = sorted(tuple(r[1:]) for r in rows if r[0] == tg)
                actual = sorted((tj, qs, qe, ss, se, cig) for tj, hsps in index.lookup(qg, tg, qi) for qs, qe, ss, se, cig in hsps)
                assert actual == expected
    return out, genomes


def test_output_renumbering_preserves_calls_alignments_self_copies_and_islands(tmp_path, monkeypatch):
    src = str(tmp_path / "source")
    os.makedirs(src)
    metas = TS.make_source(src)
    T12.add_corpus(src, metas)
    T12.add_islands(src, T12.dev_accessions(2), next(m["id"] for m in metas if m["split"] == "train"))
    fake = tmp_path / "diamond"
    fake.write_text(T12.FAKE % dict(python=sys.executable))
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    source = dict(tmp=tmp_path, src=src, metas=metas, diamond=str(fake))
    calls = []
    original = WS.diamond_search

    def capture(diamond, queries, db, settings, out_lines):
        calls.append((open(queries, "rb").read(), open(db, "rb").read()))
        return original(diamond, queries, db, settings, out_lines)

    monkeypatch.setattr(WS, "diamond_search", capture)
    baseline, _, _ = T12.run_stage(source, tmp_path / "base", "20261125T000001-base", islands=True)
    expected, current = alignments(baseline)
    baseline_calls = list(calls)
    calls.clear()
    prior = list(reversed(current[4:]))  # old prefix includes both island chromosomes; four sources are new
    ref = reference(tmp_path, prior)
    reordered, cfg, rep = T12.run_stage(source, tmp_path / "ordered", "20261125T000002-order", islands=True, output_order_from=ref)
    actual, genomes = alignments(reordered)
    assert actual == expected and calls == baseline_calls
    assert [g["id"] for g in genomes] == [g["id"] for g in prior + current[:4]]
    assert rep["output_order"]["preserved_genomes"] == len(prior)
    assert rep["output_order"]["appended_genomes"] == 4
    TS.patch(monkeypatch, tmp_path)
    partial, partial_cfg = RF.rel_run(tmp_path / "resume", stages=("w12_search",), run_id="20261125T000003-resume",
                                     tools=dict(diamond=str(fake), python=sys.executable), rel_w12_search=cfg["rel_w12_search"])
    calls.clear()

    def interrupt_third(*args, **kwargs):
        if len(calls) == 2:
            raise RuntimeError("interrupted before third call")
        return capture(*args, **kwargs)

    monkeypatch.setattr(WS, "diamond_search", interrupt_third)
    with pytest.raises(RuntimeError, match="third call"):
        with Stage(partial, "rel_w12_search") as st:
            WS.build(st, partial_cfg, partial)
    monkeypatch.setattr(WS, "diamond_search", capture)
    with Stage(partial, "rel_w12_search") as st:
        WS.build(st, partial_cfg, partial)
    assert calls == baseline_calls
    for g in genomes:
        fresh = H.load(os.path.join(reordered, WS.OUT, g["id"] + ".npz"))
        resumed = H.load(os.path.join(partial, WS.OUT, g["id"] + ".npz"))
        assert all(np.array_equal(fresh[k], resumed[k]) for k in H.KEYS)
    with Stage(reordered, "gates_rel_w12_search") as st:
        WS.gate(st, cfg, reordered)
    # A copied claim about the reference cannot replace the checked ordering provenance.
    path = os.path.join(reordered, WS.OUT, "manifest.json")
    rep["output_order"]["preserved_genomes"] -= 1
    with open(path, "w") as f:
        json.dump(rep, f)
    with pytest.raises(GateFailure):
        with Stage(reordered, "gates_rel_w12_search") as st:
            WS.gate(st, cfg, reordered)


@pytest.mark.parametrize("change", ["missing", "group", "split", "taxid", "duplicate", "duplicate_prior", "not_passed", "manifest_sha", "index_hash"])
def test_output_order_refuses_invalid_reference(tmp_path, change):
    genomes = [dict(id="a", group="bacteria", split="train", taxid=1), dict(id="b", group="bacteria", split="heldout", taxid=2)]
    ref = reference(tmp_path, genomes + [dict(genomes[0])] if change == "duplicate_prior" else genomes)
    current = [dict(g) for g in genomes]
    if change == "missing":
        current.pop()
    elif change in ("group", "split", "taxid"):
        current[0][change] = {"group": "archaea", "split": "heldout", "taxid": 99}[change]
    elif change == "duplicate":
        current.append(dict(current[0]))
    elif change == "duplicate_prior":
        pass
    elif change == "manifest_sha":
        ref["sha256"] = "0" * 64
    elif change == "not_passed":
        (tmp_path / "old-tables/record.json").write_text('{"status":"failed"}')
    else:
        with (tmp_path / "old-tables/data/rel_w12/v1/index.json").open("a") as f:
            f.write(" ")
    with pytest.raises(SystemExit):
        WS.output_order(current, ref)


def test_output_order_identity_and_config_validation():
    genomes = [dict(id="a")]
    same, mapping, provenance = WS.output_order(genomes, None)
    assert same is genomes and mapping is None and provenance is None
    for ref in ("old", {}, dict(run="old", sha256="x"), dict(run="old", sha256="a" * 64, extra=True)):
        with pytest.raises(SystemExit, match="output_order_from"):
            WS.settings(dict(rel_w12_search=dict(source="new", output_order_from=ref)))
