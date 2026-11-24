"""Arms S1 and S2 of decision D7 (seeddrift/train/rel/models/source.py, package W5) through package W6's trainer end to
end on small random tables: preflight, a few steps in the leak-free layout, the held-out evaluation, the result rows
whose parts add up, the cells scored the evaluator's, and the source read-out on a held-out batch. torch only: skipped
where torch is not installed (the CI environment), run in the model interpreter by the stage rel_w6_selftest (the
trainer's selftest mode loads this file by its path) or by hand, `python tests/ci/test_rel_w5_source_torch.py`."""
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
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from seeddrift.train.rel import api as A  # noqa: E402
from seeddrift.train.rel.evalh import conditions as cond  # noqa: E402
from seeddrift.train.rel.evalh import score as SC  # noqa: E402

if HAVE_TORCH:
    import test_rel_w6_torch as W6  # noqa: E402
    from seeddrift.train.rel.models import source as SR  # noqa: E402
    from seeddrift.train.rel.trainer import loop  # noqa: E402
else:
    import pytest
    pytestmark = pytest.mark.skip(reason="torch is not installed here; the model interpreter runs these tests")


def _cfg(root, arm, kw=None, **over):
    size = dict(dict(layers=2, width=256, arm=arm, self_layers=1), **(kw or {}))
    return W6._config(root, model="source", author="w5", size=size, layout=A.LAYOUT_LEAKFREE, whole_gene_layout=A.LAYOUT_STORED,
                      max_steps=4, eval_first=4, **over)


def test_both_arms_train_and_are_scored_as_the_reference_arm_is():
    for arm, kw in SR.TESTED:                                     # S1, S2 and S2 with the amino-acid level (copy-aa30m)
        root = tempfile.mkdtemp()
        cfg = _cfg(root, arm, kw)
        pre = loop.preflight(cfg)
        assert pre["ok"] and pre["params"] == SR.count_parameters(2, 256, arm, ext=True, **kw), (arm, kw, pre["params"])
        out = loop.fit(cfg)
        assert out["status"] == "ok" and out["steps"] == 4 and all(math.isfinite(v) for v in out["final"].values() if v is not None), arm
        start = out["evals"][0]["bits"]                           # untrained: uniform alone, the copy prior of the start with relatives
        assert all(abs(start[c] - math.log2(65)) < 0.6 for c in start if cond.NO_RELATIVES in c), (arm, start)
        assert start["scattered"] < math.log2(65) + 0.6, (arm, start)
        final = [r for r in SC.read_rows(os.path.join(cfg["out"], "results.tsv")) if r["weights"] == "ema@final" and r["group"] == "all"]
        for c in cond.CONDITIONS:                                 # the parts of the bits add up to the bits of the codon
            whole, aa, rest = (float(next(r["bits"] for r in final if r["condition"] == c and r["stratum"] == x)) for x in ("all",) + tuple(SC.PARTS))
            assert abs(aa + rest - whole) < 1e-4 and 0 < whole < 9, (arm, c, whole)
            assert out["final_items"][c] == W6._digest(cfg, root, c), (arm, c)
        run = loop.Run(dict(cfg, out=os.path.join(root, "other")))
        assert run.size == dict(cfg["size"], ext=True) and run.info["factorisation"] == "independent" and run.info["arm"] == arm
        assert run.info["aa_level"] == bool(kw.get("aa_level"))


def test_the_source_readout_on_a_trained_model():
    root = tempfile.mkdtemp()
    cfg = _cfg(root, "s1")
    loop.fit(cfg)
    model = SR.build(dict(cfg["size"], ext=True))
    grid = SR._grid(6, 3)
    row, col = SR._cells(50, 6, 4)
    grid[row, col] = A.MASK
    r = SR.readout(model, grid, [4, 2], row, col, ext=torch.zeros_like(grid, dtype=torch.bool), token=torch.randint(1, 66, (50,)))
    assert torch.allclose(r["weights"].sum(-1), torch.ones(50), atol=1e-5) and float(r["joint_nats"]) > 0
    assert torch.allclose(r["given_truth"].sum(-1), torch.ones(50), atol=1e-5)


def test_the_sources_readout_on_saved_weights():
    """readouts/w5_sources on the final weights of tiny trainings of both arms: the cells read are those of the
    training's final evaluation (the evaluator's digest), every per-cell quantity is in its range, and the bits are the
    model's own."""
    import numpy as np
    from seeddrift.train.rel.readouts import w5_sources as RO
    for arm, kw in SR.TESTED:
        root = tempfile.mkdtemp()
        cfg = W6._config(root, kind=A.TABLE_REAL, probe=False, model="source", author="w5", size=dict(layers=2, width=256, arm=arm, self_layers=1, **kw),
                         max_steps=4, per_step=4, micro=4, eval_first=4)
        out = loop.fit(cfg)
        import json
        with open(os.path.join(root, "out", "conf.json"), "w") as f:              # as stage w6_fit leaves it beside the arm
            json.dump(cfg, f)
        folder = tempfile.mkdtemp()
        head = RO.read_arm(arm, os.path.join(root, "out"), {"scattered": "stored", "whole_gene": "stored"}, 0, 32, folder, log=lambda s: None)
        assert head["arm"] == arm and head["tables_seen"] == out["tables_seen"]
        for c in ("scattered", "whole_gene"):
            h = head["conditions"][c]
            assert h["items_sha256"] == out["final_items"][c] and h["cells"] > 0, (arm, c)
            z = np.load(os.path.join(folder, "%s.%s.npz" % (arm, c)))
            assert len(z["bits"]) == h["cells"] and np.isfinite(z["bits"]).all() and (z["bits"] >= 0).all()
            assert ((z["none"] >= 0) & (z["none"] <= 1 + 1e-6)).all() and (z["entropy"] >= -1e-9).all()
            copy = z["none"] < 1 - 1e-6
            assert np.allclose(z["level"][copy].sum(axis=1), 1.0, atol=1e-5) and z["level"].shape[1] == SR.LEVELS + int(bool(kw.get("aa_level")))
            ti = z["top_identity"][np.isfinite(z["top_identity"])]
            assert ((ti >= 0) & (ti <= 1)).all()
            assert ("none_given_truth" in z.files) == (arm == "s1")
            if c == "scattered":                                                        # one pass of the trunk gives source.readout's weights
                cfg2, model, _ = RO.load_arm(os.path.join(root, "out"))
                held, data, meta = RO.held_data(cfg2, c, "stored")
                ex, _ = RO.examples_of(data, RO.E.subset(held, 0)[:3], c, "stored", False)
                got = RO.read_batch(model, ex, meta["identity"])
                b = RO.E.to_device(RO.S.collate([{k: e[k] for k in ("grid", "truth")} for e in ex]), torch.device("cpu"))
                ref = SR.readout(model, b["grid"], b["sizes"], b["row"], b["col"], token=b["token"])
                assert np.allclose(got["none"], ref["weights"][:, -1].double().numpy(), atol=1e-6)
                if arm == "s1":
                    assert np.allclose(got["none_given_truth"], ref["given_truth"][:, -1].double().numpy(), atol=1e-6)
            fin = np.load(os.path.join(root, "out", "heldout_final.npz"))                # the final evaluation's cells: same order, same bits
            assert np.array_equal(z["table"], fin["table_%s" % c]) and np.allclose(z["bits"], fin["bits_%s" % c], atol=1e-4), (arm, c)


def test_both_arms_train_on_the_strict_layout():
    """Round 1's third batch (protocol round1, version 9): S1, S2 and copy-aa30m (S2 with the amino-acid level) trained
    on W3's strict rendering (done by the trainer's sampler; the module unchanged), the whole gene on its frame relative;
    the nodes read scattered strictly online, the end on W3's offline rendering; W3's fourth test on the training's
    sample 0 passes."""
    import csv
    root = tempfile.mkdtemp()
    D = W6.D
    train, held = D.v1_tables(24, rows=8, seed=11), D.v1_tables(12, rows=8, seed=12, split=1)
    tables = D.write_tableset(root, [("train/s0000", train), ("heldout/s0000", held)], kind=A.TABLE_REAL, sub="v1")
    off = W6._offline(root, held)
    four = [cond.SCATTERED, cond.WHOLE_GENE, cond.SCATTERED + "+" + cond.NO_RELATIVES, cond.WHOLE_GENE + "+" + cond.NO_RELATIVES]
    for arm, kw in SR.TESTED:
        cfg = W6._config(root, train_tables=tables, heldout_tables=tables, probe=False, rows=[2, 8], layout="strict", whole_gene_layout="frame",
                         eval_online=True, heldout_offline_tables=off, heldout_offline_root=root, only_conditions=four, max_steps=6, eval_first=4,
                         model="source", author="w5", size=dict(layers=2, width=256, arm=arm, self_layers=1, **kw),
                         out=os.path.join(root, "out-%s%s" % (arm, "-aa" if kw else "")))
        pre = loop.preflight(cfg)
        assert pre["ok"] and pre["sampler"]["strict_tables"] > 0 and pre["params"] == SR.count_parameters(2, 256, arm, ext=True, **kw), (arm, kw, pre)
        out = loop.fit(cfg)
        assert out["status"] == "ok" and out["traincheck"]["failures"] == 0 and out["traincheck"]["checked"] >= 1, (arm, out.get("traincheck"))
        assert all(math.isfinite(v) for v in out["final"].values() if v is not None), (arm, out["final"])
        lay = {(r["weights"] == "ema@final", r["condition"]): r["layout"] for r in SC.read_rows(os.path.join(cfg["out"], "results.tsv"))}
        assert lay[(False, cond.SCATTERED)] == "leakfree" and lay[(True, cond.SCATTERED)] == "leakfree", (arm, lay)
        with open(os.path.join(cfg["out"], "steps.csv")) as f:
            assert list(csv.DictReader(f))[-1]["traincheck_failures"] == "0"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("passed", name)
