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


def _cfg(root, arm, **over):
    size = dict(layers=2, width=256, arm=arm, self_layers=1)
    return W6._config(root, model="source", author="w5", size=size, layout=A.LAYOUT_LEAKFREE, whole_gene_layout=A.LAYOUT_STORED,
                      max_steps=4, eval_first=4, **over)


def test_both_arms_train_and_are_scored_as_the_reference_arm_is():
    for arm in SR.ARMS:
        root = tempfile.mkdtemp()
        cfg = _cfg(root, arm)
        pre = loop.preflight(cfg)
        assert pre["ok"] and pre["params"] == SR.count_parameters(2, 256, arm, ext=True), (arm, pre["params"])
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("passed", name)
