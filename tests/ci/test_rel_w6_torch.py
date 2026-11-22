"""The trainer of the relational line end to end on small random tables (package W6): preflight with its divergence
guard, a few steps with the warm-up and the moving average, the held-out evaluation at the nodes of tables seen and at
the end, the result rows and the digests of the cells scored. torch only: skipped where torch is not installed (the CI
environment), run in the model interpreter by the stage rel_w6_selftest (the trainer's selftest mode loads this file
by its path) or by hand, `python tests/ci/test_rel_w6_torch.py`."""
import csv
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

import test_rel_w6_data as D  # noqa: E402

from seeddrift.train.rel import api as A  # noqa: E402
from seeddrift.train.rel.data import tables as T  # noqa: E402
from seeddrift.train.rel.evalh import conditions as cond  # noqa: E402
from seeddrift.train.rel.evalh import score as SC  # noqa: E402

if HAVE_TORCH:
    from seeddrift.train.rel.trainer import loop  # noqa: E402
else:
    import pytest
    pytestmark = pytest.mark.skip(reason="torch is not installed here; the model interpreter runs these tests")


def _config(root, kind=A.TABLE_SYNTHETIC, **over):
    tables = D.write_tableset(root, [("train/s0000", D.random_tables(24, rows=4, seed=1)), ("heldout/s0000", D.random_tables(12, rows=4, seed=2, split=1))],
                              kind=kind)
    cfg = dict(name="tiny", author="w6", model="axial", size=dict(layers=1, width=256), objective="mlm", lr=1e-3, seed=1, wall_s=900,
               train_tables=tables, train_root=root, heldout_tables=tables, heldout_root=root, out=os.path.join(root, "out"), probe=True,
               rows=[2, 4], per_step=2, micro=1, max_steps=6, warmup_steps=3, eval_first=4, eval_per_group=1, preflight_steps=8,
               preflight_warmup=3, preflight_guard="reduce", device="cpu", threads=2, run="run-x", commit="abc", group_names={"0": "bacteria"})
    cfg.update(over)
    return cfg


def _digest(cfg, root, condition):
    held = T.open_split(root, cfg["heldout_tables"], 1)
    data = held.load(T.MODEL_FIELDS)
    d = cond.ItemsDigest(condition)
    d.add(cond.materialise(data, np.arange(len(held)), condition))
    return d.hexdigest()


def test_tiny_fit():
    root = tempfile.mkdtemp()
    cfg = _config(root)
    pre = loop.preflight(cfg)
    assert pre["ok"] and pre["micro"] == 1 and pre["params"] > 1e6 and pre["cells_per_s"] > 0 and set(pre["eval_bits"]) == set(cond.CONDITIONS)
    assert pre["guard"]["steps"] == 8 and pre["lr"] in (1e-3, 3e-4) and pre["lr_reduced"] == (pre["lr"] != 1e-3)
    out = loop.fit(cfg)
    assert out["status"] == "ok" and out["steps"] == 6 and out["ended_by"] == "max_steps" and out["tables_seen"] == 12 and out["cells_seen"] > 0
    assert [e["tables_seen"] for e in out["evals"]] == [0, 4, 8] and out["weights_read"] == "ema" and out["eval_final"] == 12
    assert abs(out["evals"][0]["bits"]["scattered"] - math.log2(65)) < 0.3                         # an untrained model is near uniform
    rows = SC.read_rows(os.path.join(cfg["out"], "results.tsv"))
    assert set(rows[0]) == set(A.RESULT_ROW)
    assert {r["weights"] for r in rows} == {"ema@tables=0", "ema@tables=4", "ema@tables=8", "ema@final"}
    final = [r for r in rows if r["weights"] == "ema@final"]
    assert {r["condition"] for r in final} == set(cond.CONDITIONS) and {r["item_half"] for r in final} == {"all"}
    assert {r["item_half"] for r in rows if r["weights"] != "ema@final"} == {"subset"}
    assert all(r["layout"] == A.LAYOUT_STORED and r["table_kind"] == A.TABLE_SYNTHETIC and r["variant"] == "tiny" and r["run"] == "run-x" for r in rows)
    top = [r for r in final if r["group"] == "all" and r["stratum"] == "all"]
    assert len(top) == 5 and all(0 < float(r["bits"]) < 8 and int(r["cells_seen"]) == out["cells_seen"] for r in top)
    assert {r["group"] for r in final} >= {"all", "bacteria", "11"} and {r["stratum"] for r in final} >= {"all", "other_rows>=2"} | set(SC.PARTS)
    for c in cond.CONDITIONS:                                     # the two parts add up to the bits of the codon
        whole, aa, rest = (float(next(r["bits"] for r in final if r["condition"] == c and r["group"] == "all" and r["stratum"] == x))
                           for x in ("all",) + tuple(SC.PARTS))
        assert abs(aa + rest - whole) < 1e-4 and 0 < aa < whole, c
    for c in cond.CONDITIONS:                                     # the cells scored are the evaluator's, table for table
        assert out["final_items"][c] == _digest(cfg, root, c), c
    with open(os.path.join(cfg["out"], "steps.csv")) as f:
        steps = list(csv.DictReader(f))
    lr = [float(s["lr"]) for s in steps]
    assert len(steps) == 6 and lr[0] < lr[1] < lr[2] == lr[3] == lr[5] == 1e-3 and all(float(s["loss_bits"]) > 0 for s in steps)
    assert int(steps[-1]["tables_seen"]) == 12 and int(steps[-1]["cells_seen"]) == out["cells_seen"] and int(steps[-1]["target_cells"]) == out["target_cells"]
    assert not os.path.exists(os.path.join(cfg["out"], "final.pt"))                                # probe mode keeps no checkpoint
    with open(os.path.join(cfg["out"], "evals.jsonl")) as f:
        ev = [json.loads(line) for line in f]
    assert [e.get("final", False) for e in ev] == [False, False, False, True] and ev[-1]["items"] == out["final_items"]
    try:                                                                                         # a second training in the same folder
        loop.fit(cfg)
    except SystemExit:
        pass
    else:
        raise AssertionError("a training started in a folder that holds one")


def test_preflight_guard_stops_by_default():
    root = tempfile.mkdtemp()
    try:
        loop.preflight(_config(root, lr=10.0, preflight_guard="stop"))
    except SystemExit as err:
        assert "did not train" in str(err)
    else:
        raise AssertionError("a preflight whose loss did not fall passed")
    with open(os.path.join(root, "out", "preflight.json")) as f:
        assert json.load(f)["ok"] is False


def test_anyorder_fill_in_trains():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, masks=["anyorder"], max_steps=3))
    assert out["status"] == "ok" and out["steps"] == 3 and all(math.isfinite(v) for v in out["final"].values() if v is not None)


def test_moving_average():
    root = tempfile.mkdtemp()
    run = loop.Run(_config(root))
    w0 = [p.detach().clone() for p in run.ema_weights]
    run.step(run.sampler.draw(0), 1e-3)
    run.update_ema(1)
    d = 2.0 / 11.0
    for e, a, w in zip(run.ema_weights, w0, run.weights):
        assert torch.allclose(e, d * a + (1 - d) * w.detach(), atol=1e-6)
    assert any(not torch.equal(e, w) for e, w in zip(run.ema_weights, run.weights))
    assert all(not p.requires_grad for p in run.ema_weights)


def test_tiny_fit_without_relatives():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, relatives=False, max_steps=2, per_step=4, micro=4, ema=False))
    assert out["status"] == "ok" and out["conditions"] == [c for c in cond.CONDITIONS if "no_relatives" in c] and out["cells_seen"] == 2 * 4 * A.ROW_LEN
    rows = SC.read_rows(os.path.join(root, "out", "results.tsv"))
    assert {r["condition"] for r in rows} == {"scattered+no_relatives", "whole_gene+no_relatives"} and {r["weights"] for r in rows} == {
        "raw@tables=0", "raw@tables=4", "raw@tables=8", "raw@final"}
    assert {r["budget"] for r in rows if r["weights"] == "raw@tables=4"} <= {"tables=4;wall_s=%d" % s for s in range(0, 900)}
    assert "final_best" not in out                                                              # probe mode keeps no best weights


def test_the_best_node_of_an_arm_without_relatives_is_read_on_every_table():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, kind=A.TABLE_REAL, probe=False, relatives=False, max_steps=4, per_step=4, micro=4, eval_first=4))
    rows = SC.read_rows(os.path.join(root, "out", "results.tsv"))
    if out["best"]["tables_seen"] < out["tables_seen"]:
        best = [r for r in rows if r["weights"] == "ema@best"]
        assert out["final_best"]["tables_seen"] == out["best"]["tables_seen"] and {r["item_half"] for r in best} == {"all"}
        assert {r["budget"].split(";")[0] for r in best} == {"tables=%d" % out["best"]["tables_seen"]}
        assert out["final_best"]["items"] == out["final_items"]                                  # the same cells as the end
    else:
        assert "final_best" not in out and not any(r["weights"] == "ema@best" for r in rows)


def test_saved_weights_evaluated_again():
    """An arm's final.pt evaluated again gives the bits and the digests of the training's own final evaluation; best.pt
    gives rows ema@best."""
    root = tempfile.mkdtemp()
    cfg = _config(root, kind=A.TABLE_REAL, probe=False, relatives=False, max_steps=4, per_step=4, micro=4, eval_first=4)
    out = loop.fit(cfg)
    folder = os.path.join(root, "again")
    best = out["best"]
    again = loop.evaluate_checkpoints(dict(cfg, out=folder, final=os.path.join(root, "out", "final.pt"), best=os.path.join(root, "out", "best.pt"),
                                           final_tables=out["tables_seen"], final_cells=out["cells_seen"], final_wall_s=out["wall_s"],
                                           best_tables=best["tables_seen"], best_cells=best["cells_seen"], best_wall_s=best["wall_s"]))
    assert again["status"] == "ok" and again["final_items"] == out["final_items"] and again["tables_seen"] == out["tables_seen"]
    assert all(abs(again["final"][c] - out["final"][c]) < 1e-5 for c in out["final"]) and "final_best" in again
    rows = SC.read_rows(os.path.join(folder, "results.tsv"))
    assert {r["weights"] for r in rows} == {"ema@final", "ema@best"} and {r["item_half"] for r in rows} == {"all"}
    assert {r["budget"].split(";")[0] for r in rows if r["weights"] == "ema@final"} == {"tables=%d" % out["tables_seen"]}


def test_split_and_gradient_checks():
    root = tempfile.mkdtemp()
    try:
        loop.Run(_config(root, train_split=1))
    except SystemExit as err:
        assert "split" in str(err)
    else:
        raise AssertionError("a training on split 1 was accepted")
    run = loop.Run(_config(tempfile.mkdtemp()))
    with torch.no_grad():
        next(run.model.parameters()).fill_(float("inf"))
    before = [p.detach().clone() for p in run.model.parameters()]
    try:
        run.step(run.sampler.draw(0), 1e-3)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("a step with a gradient that is not finite was taken")
    assert all(torch.equal(a, b) or not torch.isfinite(a).all() for a, b in zip(before, run.model.parameters()))


def test_real_tables_only_outside_probe_mode():
    root = tempfile.mkdtemp()
    try:
        loop.fit(_config(root, probe=False))
    except SystemExit as err:
        assert "probe" in str(err)
    else:
        raise AssertionError("a training outside probe mode took synthetic tables")


def test_budget_ends_the_run_and_checkpoints_are_kept():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, kind=A.TABLE_REAL, probe=False, max_steps=None, budget_cells=30000))
    assert out["ended_by"] == "budget" and out["cells_seen"] >= 30000 and out["status"] == "ok"
    assert os.path.exists(os.path.join(root, "out", "final.pt")) and os.path.exists(os.path.join(root, "out", "best.pt")) and out["best"] is not None
    ck = torch.load(os.path.join(root, "out", "final.pt"), map_location="cpu", weights_only=True)
    assert ck["cells_seen"] == out["cells_seen"] and any(k.startswith("blocks.0.row.") for k in ck["model"]) and set(ck["ema"]) == set(ck["model"])
    best = torch.load(os.path.join(root, "out", "best.pt"), map_location="cpu", weights_only=True)
    assert "model" not in best and best["tables_seen"] == out["best"]["tables_seen"]
    z = np.load(os.path.join(root, "out", "heldout_final.npz"))
    assert len(z["bits_whole_gene"]) == len(z["table_whole_gene"]) > 0 and set(z["table_scattered"].tolist()) <= set(range(12))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
