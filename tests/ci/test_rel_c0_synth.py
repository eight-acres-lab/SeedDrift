"""The generator of synthetic tables (ci/stages/rel/synth.py) against its specification: random symbols; 16 to 32 rows,
each copied from another at a substitution rate between 0.1% and 30%; known relations; the two capability tasks (the
whole locus masked, so the source must be found elsewhere; a switch of source inside the locus); the correct sources
for tests only. And the development stage that writes them as a table set (ci/stages/rel/c0_synth.py)."""
import json
import os

import numpy as np
import pytest

import rel_fixtures as RF
from ci.stages.common import Stage
from ci.stages.rel import c0_synth, common, synth


def test_tables_are_reproducible_from_seed_and_index():
    a, b = synth.generate(5, 3), synth.generate(5, 3)
    assert a["truth"] == b["truth"] and all((x == y).all() for x, y in zip(a["blocks"], b["blocks"]))
    c = synth.generate(6, 3)
    assert a["id"] != c["id"] and not all(x.shape == y.shape and (x == y).all() for x, y in zip(a["blocks"], c["blocks"]))


def test_shapes_symbols_and_row_counts():
    rows = []
    for t in RF.synthetic(200, seed=2):
        n = t["blocks"][0].shape[0]
        rows.append(n)
        assert t["roles"][:3] == ["target", "left", "right"] and len(t["roles"]) == 19 == len(t["blocks"])
        assert 64 <= t["blocks"][0].shape[1] <= 256 and all(32 <= b.shape[1] <= 128 for b in t["blocks"][1:3]) and all(b.shape[1] == 32 for b in t["blocks"][3:])
        assert all(b.shape[0] == n and b.dtype == np.uint8 and b.min() >= 1 and b.max() <= 64 for b in t["blocks"])
        tr = t["truth"]
        assert len(tr["parent"]) == n and sum(1 for p in tr["parent"] if p < 0) == 1 and tr["parent"][0] == tr["source"] >= 1
        assert all(0.001 <= r <= 0.3 for i, r in enumerate(tr["rate"]) if tr["parent"][i] >= 0)
    assert min(rows) == 16 and max(rows) == 32


def test_every_row_differs_from_its_parent_at_its_rate():
    z = []
    for t in RF.synthetic(150, seed=4):
        tr = t["truth"]
        full = np.concatenate(t["blocks"], axis=1)
        rest = np.concatenate(t["blocks"][1:], axis=1)                     # the switch task changes the examined target block
        for i, p in enumerate(tr["parent"]):
            if p < 0:
                continue
            x, y = (rest[i], rest[p]) if i == 0 else (full[i], full[p])
            r = tr["rate"][i]
            z.append(((x != y).mean() - r) / np.sqrt(r * (1 - r) / x.size))
    z = np.array(z)
    assert abs(z.mean()) < 0.1 and 0.9 < z.std() < 1.1 and np.abs(z).max() < 6


def test_locus_task_source_is_the_nearest_row_elsewhere():
    n, top = 0, 0
    for t in RF.synthetic(200, seed=5, switch_share=0.0):
        tr = t["truth"]
        assert t["task"] == "locus" and int(np.argmin(tr["divergence"][1:])) + 1 == tr["source"]
        rest = np.concatenate(t["blocks"][1:], axis=1)                     # everything but the target block
        top += int(np.argmax((rest[1:] == rest[0]).mean(1))) + 1 == tr["source"]
        n += 1
    assert top / n > 0.85                                                  # found from elsewhere in most tables


def test_switch_task_follows_the_second_source_after_the_switch():
    before, after = [], []
    for t in RF.synthetic(120, seed=6, switch_share=1.0):
        tr, T = t["truth"], t["blocks"][0]
        k, a, b = tr["switch_at"], tr["source"], tr["second"]
        assert t["task"] == "switch" and a != b and T.shape[1] // 4 <= k <= 3 * T.shape[1] // 4
        before.append((T[0, :k] == T[a, :k]).mean() - (T[0, :k] == T[b, :k]).mean())
        after.append((T[0, k:] == T[b, k:]).mean() - (T[0, k:] == T[a, k:]).mean())
    assert np.mean(before) > 0.5 and np.mean(after) > 0.5


def test_bad_parameters_fail():
    for bad in (dict(rows_min=1), dict(rate_max=1.0), dict(rows_min=40), dict(nothing=1)):
        with pytest.raises(ValueError):
            synth.params(**bad)


def test_development_stage_writes_a_table_set_that_passes_its_gate(tmp_path, monkeypatch):
    run_dir, cfg = RF.rel_run(tmp_path, rel_c0_synth=dict(tables=6, seed=2, per_shard=4))

    def fake_writer(tables, folder, per_shard):                            # the shard format comes with the contract
        out = []
        for k in range(0, len(tables), per_shard):
            p = os.path.join(folder, "shard-%05d.npy" % (k // per_shard))
            np.save(p, np.stack([t["blocks"][0][0, :8] for t in tables[k:k + per_shard]]))
            out.append((p, "shard"))
        return out
    monkeypatch.setattr(c0_synth, "write_shards", fake_writer)
    monkeypatch.setattr(common, "sealed_set", lambda cfg, spec=None: RF.sealed())

    class St:
        extra, outputs = {}, []

        def output(self, name, path):
            self.outputs.append((name, path))
    doc = c0_synth.build(St(), cfg, run_dir)
    assert doc["kind"] == "synthetic" and doc["tables"] == 6 and sorted({f["role"] for f in doc["files"]}) == ["meta", "shard", "truth"]
    with Stage(run_dir, "gates_rel_c0_synth") as st:                        # the plain Stage: no machine check here
        c0_synth.gate(st, cfg, run_dir)
    g = json.load(open(os.path.join(run_dir, "gates", "G4_rel_c0_synth.json")))
    assert g["status"] == "pass", g
    assert c0_synth.summary(run_dir)["json"]["tables"] == 6
    cfg_more = dict(cfg, rel_c0_synth=dict(tables=7, seed=2, per_shard=4))
    from ci.gates import GateFailure
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_c0_synth") as st:
            c0_synth.gate(st, cfg_more, run_dir)                             # another table count than configured


def test_development_stage_refuses_unknown_settings_fails():
    with pytest.raises(SystemExit):
        c0_synth.settings(dict(rel_c0_synth=dict(tables=5, colour="blue")))
