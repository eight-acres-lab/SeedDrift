"""The trainer of the relational line end to end on a small random table set (package W6): preflight, a few steps,
the held-out evaluation on the way and at the end, the result rows. torch only: skipped where torch is not installed
(the CI environment), run in the model interpreter by the stage rel_w6_selftest (the trainer's selftest mode loads
this file by its path) or by hand, `python tests/ci/test_rel_w6_torch.py`."""
import csv
import json
import math
import os
import sys
import tempfile

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
from seeddrift.train.rel.data import heldout_plan as H  # noqa: E402

if HAVE_TORCH:
    from seeddrift.train.rel.trainer import loop  # noqa: E402
    from seeddrift.train.rel.trainer import rows as W  # noqa: E402
else:
    import pytest
    pytestmark = pytest.mark.skip(reason="torch is not installed here; the model interpreter runs these tests")


def _config(root, kind=A.TABLE_SYNTHETIC, **over):
    tables = D.write_tableset(root, [("t0", D.random_tables(24, rows=4, seed=1)), ("h0", D.random_tables(12, rows=4, seed=2, split=1))], kind=kind)
    cfg = dict(name="tiny", author="w6", model="axial", size=dict(layers=1, width=256), objective="mlm", lr=1e-3, seed=1, wall_s=900,
               train_tables=tables, train_root=root, train_split=0, heldout_tables=tables, heldout_root=root, heldout_split=1,
               out=os.path.join(root, "out"), probe=True, rows=[2, 4], per_step=2, micro=1, max_steps=6, eval_every_cells=8000, eval_per_group=1,
               device="cpu", threads=2, run="run-x", commit="abc", preflight_s=1)
    cfg.update(over)
    return cfg


def test_tiny_fit():
    root = tempfile.mkdtemp()
    cfg = _config(root)
    pre = loop.preflight(cfg)
    assert pre["ok"] and pre["micro"] == 1 and pre["params"] > 1e6 and pre["cells_per_s"] > 0 and set(pre["eval_bits"]) == set(H.CONDITIONS)
    out = loop.fit(cfg)
    assert out["status"] == "ok" and out["steps"] == 6 and out["ended_by"] == "max_steps" and out["cells_seen"] > 0
    assert len(out["evals"]) >= 2 and out["evals"][0]["step"] == 0 and out["evals"][0]["cells_seen"] == 0
    assert abs(out["evals"][0]["bits"]["scattered"] - math.log2(65)) < 0.3                       # an untrained model is near uniform
    rows = W.read_rows(os.path.join(cfg["out"], "results.tsv"))
    final = [r for r in rows if r["weights"] == "final"]
    assert {r["condition"] for r in final} == set(H.CONDITIONS) and {r["item_half"] for r in final} == {"all"}
    assert all(r["layout"] == A.LAYOUT_STORED and r["table_kind"] == A.TABLE_SYNTHETIC and r["variant"] == "tiny" and r["run"] == "run-x" for r in rows)
    top = [r for r in final if r["group"] == "all" and r["stratum"] == "all"]
    assert len(top) == 5 and all(0 < float(r["bits"]) < 8 and int(r["cells_seen"]) == out["cells_seen"] for r in top)
    assert {r["group"] for r in final} >= {"all", "g00", "g11"} and {r["stratum"] for r in final} <= set(H.STRATA) | {"all"}
    with open(os.path.join(cfg["out"], "steps.csv")) as f:
        steps = list(csv.DictReader(f))
    assert len(steps) == 6 and float(steps[0]["lr"]) < float(steps[-1]["lr"]) <= 1e-3 and all(float(s["loss_bits"]) > 0 for s in steps)
    assert int(steps[-1]["cells_seen"]) == out["cells_seen"] and int(steps[-1]["target_cells"]) == out["target_cells"]
    assert not os.path.exists(os.path.join(cfg["out"], "final.pt"))                               # probe mode keeps no checkpoint
    with open(os.path.join(cfg["out"], "summary.json")) as f:
        assert json.load(f)["status"] == "ok"
    try:                                                                                         # a second training in the same folder
        loop.fit(cfg)
    except SystemExit:
        pass
    else:
        raise AssertionError("a training started in a folder that holds one")


def test_tiny_fit_without_relatives():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, relatives=False, max_steps=2, per_step=4, micro=4, eval_every_cells=None))
    assert out["status"] == "ok" and out["conditions"] == [c for c in H.CONDITIONS if "no_relatives" in c] and out["cells_seen"] == 2 * 4 * A.ROW_LEN
    rows = W.read_rows(os.path.join(root, "out", "results.tsv"))
    assert {r["condition"] for r in rows} == {"scattered+no_relatives", "whole_gene+no_relatives"}


def test_real_tables_only_outside_probe_mode():
    root = tempfile.mkdtemp()
    try:
        loop.fit(_config(root, probe=False))
    except SystemExit as err:
        assert "probe" in str(err)
    else:
        raise AssertionError("a training outside probe mode took synthetic tables")


def test_budget_ends_the_run_and_checkpoints_are_kept():
    import numpy as np
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, kind=A.TABLE_REAL, probe=False, max_steps=None, budget_cells=30000, eval_every_cells=12000))
    assert out["ended_by"] == "budget" and out["cells_seen"] >= 30000 and out["status"] == "ok"
    assert os.path.exists(os.path.join(root, "out", "final.pt")) and out["best"] is not None
    ck = torch.load(os.path.join(root, "out", "final.pt"), map_location="cpu")
    assert ck["cells_seen"] == out["cells_seen"] and any(k.startswith("blocks.0.row.") for k in ck["model"])
    with open(os.path.join(root, "out", "steps.csv")) as f:
        lr = [float(r["lr"]) for r in csv.DictReader(f)]
    assert lr[-1] < max(lr) and max(lr) <= 1e-3                                                   # warm-up, then decay towards the budget's end
    z = np.load(os.path.join(root, "out", "heldout_final.npz"))
    assert len(z["table"]) == 12 and (z["cells_scattered"] >= 0).all() and z["bits_whole_gene"].sum() > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
