"""The evaluation items (seeddrift/train/rel/evalh/items.py) and their stage rel_w5_items (ci/stages/rel/w5_items.py):
per group the same number of items, a gene used once, every stretch inside the stored part of its gene with three bases
per codon, halves by genus, the same items in every numpy; the stage on a synthetic source run (the fixture of the
tables package), its gate, and what it refuses."""
import json
import os

import numpy as np
import pytest

import rel_fixtures as RF
import test_rel_w2_stage as W2S
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import w5_items
from seeddrift.train.rel import api
from seeddrift.train.rel.evalh import items as I


def genomes(n_groups=3, per_group=4, genera=2, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_groups):
        for k in range(per_group):
            n = int(rng.integers(20, 60))
            lengths = rng.integers(5, 400, n)
            start = np.cumsum(rng.integers(10, 3000, n)) + 1
            out.append(dict(id="grp%d_genome%d" % (g, k), group="group%d" % g, genus="genus:%d" % (10 * g + k % genera), lengths=lengths, start=start,
                            end=start + 3 * lengths - 1, strand=rng.choice([-1, 1], n), replicon=np.zeros(n, np.int64)))
    return out


def test_items_follow_the_rules():
    gs = genomes()
    items, short = I.plan(gs, per_group=30)
    assert short == {} and len(items) == 90 and [it["item"] for it in items] == list(range(90))
    assert I.problems(items, short, 30, ["group0", "group1", "group2"], {(g["id"], j): int(x) for g in gs for j, x in enumerate(g["lengths"])}) == []
    by = {g["id"]: g for g in gs}
    for it in items:
        g = by[it["genome"]]
        part = min(int(g["lengths"][it["gene"]]), api.LEN_TARGET)
        assert it["codons"] == min(I.MAX_CODONS, part) and 0 <= it["first"] <= part - it["codons"] and g["lengths"][it["gene"]] >= I.MIN_CODONS
        assert it["base_last"] - it["base_first"] + 1 == 3 * it["codons"]
        if it["strand"] > 0:
            assert it["base_first"] == g["start"][it["gene"]] + 3 * it["first"]
        else:
            assert it["base_last"] == g["end"][it["gene"]] - 3 * it["first"]
    for grp in ("group0", "group1", "group2"):                         # genomes take turns; halves split by genus
        mine = [it for it in items if it["group"] == grp]
        per_genome = np.bincount([int(it["genome"][-1]) for it in mine], minlength=4)
        assert per_genome.max() - per_genome.min() <= 1
        assert {it["half"] for it in mine} == set(I.HALVES) and all(len({it["half"] for it in mine if it["genus"] == u}) == 1 for u in {it["genus"] for it in mine})
    assert I.plan(gs, per_group=30) == (items, short) and I.digest(items) == I.digest(I.plan(gs, per_group=30)[0])
    assert I.digest(I.plan(gs, per_group=30, seed=5)[0]) != I.digest(items)
    pinned = [(it["genome"], it["gene"], it["first"]) for it in items[:3]]
    assert pinned == [(it["genome"], it["gene"], it["first"]) for it in I.plan(list(reversed(gs)), per_group=30)[0][:3]]   # input order is free


def test_short_groups_and_one_genus_are_problems_fails():
    gs = genomes(n_groups=2, per_group=2, genera=1)
    for g in gs:
        g["lengths"] = np.minimum(g["lengths"], 300)
    items, short = I.plan(gs, per_group=10)
    assert short == {}
    bad = I.problems(items, short, 10, ["group0", "group1"])
    assert any("a half without items" in e["error"] for e in bad)                    # one genus: no confirmation half
    few = genomes(n_groups=1, per_group=1)
    few[0]["lengths"] = np.array([3, 200, 8, 120])                                     # two genes long enough
    items, short = I.plan(few, per_group=5)
    assert len(items) == 2 and short == {"group0": 3} and any("short" in e["error"] for e in I.problems(items, short, 5))
    broken = [dict(it, first=api.LEN_TARGET) for it in items]
    assert any("not inside" in e["error"] for e in I.problems(broken, {}, 2))
    assert I.genus_of(dict(id="x", ranks=[1, 2, 3, 4, 77, 9])) == "genus:77" and I.genus_of(dict(id="x", ranks=[1, 2, 3, 4, 0, 9])) == "genome:x"


GENOMES = ([("%s_%s%d" % (g.replace(".", "_"), s[:2], k), g, s) for g in sorted(W2S.TINY) for s, n in (("train", 1), ("heldout", 2)) for k in range(n)]
           + [(W2S.SEALED[0], "bacteria", "test")])


@pytest.fixture
def source(tmp_path, monkeypatch):
    src = str(tmp_path / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    monkeypatch.setattr(W2S, "GENOMES", GENOMES)
    metas = W2S.make_source(src)
    W2S.patch(monkeypatch, tmp_path)
    return src, metas


def test_stage_writes_items_of_the_held_out_split(source, tmp_path):
    src, metas = source
    run_dir, cfg = RF.rel_run(tmp_path, stages=("w5_items",), rel_w5_items=dict(source=src, per_group=2, min_codons=12))    # the fixture's viruses have few, short genes
    with Stage(run_dir, w5_items.STAGE) as st:
        doc = w5_items.build(st, cfg, run_dir)
    split = {m["id"]: m["split"] for m in metas}
    assert doc["problems"] == [] and len(doc["items"]) == 24 and doc["short"] == {} and len(doc["groups"]) == 12
    assert {split[it["genome"]] for it in doc["items"]} == {"heldout"} and W2S.SEALED[0] not in json.dumps(doc)
    assert all(c == dict(screen=1, confirm=1) for c in doc["counts"].values())                # two genera per group, items in turns
    with Stage(run_dir, "gates_rel_w5_items", requires=("pre",)) as st:
        w5_items.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w5_items.json")))["status"] == "pass"
    assert w5_items.summary(run_dir)["json"]["items"] == 24
    p = os.path.join(run_dir, w5_items.OUT, "items.json")
    d = json.load(open(p))
    d["items"][0]["half"] = "confirm" if d["items"][0]["half"] == "screen" else "screen"
    json.dump(d, open(p, "w"))
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w5_items", requires=("pre",)) as st:
            w5_items.gate(st, cfg, run_dir)


def test_stage_refuses_a_changed_source_fails(source, tmp_path):
    src, metas = source
    held = [m["id"] for m in metas if m["split"] == "heldout"][0]
    with open(os.path.join(src, "data", "train_data", held + ".npz"), "ab") as f:
        f.write(b"\0")
    run_dir, cfg = RF.rel_run(tmp_path, stages=("w5_items",), rel_w5_items=dict(source=src, per_group=6))
    with pytest.raises(SystemExit):
        with Stage(run_dir, w5_items.STAGE) as st:
            w5_items.build(st, cfg, run_dir)
    with pytest.raises(SystemExit):
        w5_items.settings(dict(rel_w5_items=dict(source=src, per_group=6, max_codons=500)))
