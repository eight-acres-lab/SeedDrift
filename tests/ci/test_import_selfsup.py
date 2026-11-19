"""train.models_from (ci/stages/train.py import_selfsup): variants trained by one run imported by another.

compare_heldout runs everywhere; the end-to-end import needs torch and runs with the model's Python on the GPU machine
(PYTHONPATH=. python tests/ci/test_import_selfsup.py), on the CPU."""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

try:
    import pytest
except ImportError:                                                    # the model's Python has no pytest
    pytest = None

from ci import ROOT
from ci.hashing import sha256_file, sha256_json
from ci.stages import train as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_selfsup import make_run  # noqa: E402


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        if pytest is not None:
            pytest.skip("torch is not installed")
        raise


def _scores(path, **cols):
    n = len(next(iter(cols.values())))
    base = dict(genome=np.array([b"g"] * n), gene=np.arange(n), b1_n=np.ones(n), b2_n=np.ones(n), c_rows=np.zeros(n))
    base.update(cols)
    np.savez(path, **base)
    return path


def test_compare_heldout():
    with tempfile.TemporaryDirectory() as d:
        a = _scores(os.path.join(d, "a.npz"), A=np.linspace(0, 3, 50), C=np.r_[np.full(10, np.nan), np.linspace(-1, 1, 40)])
        probs, cols = T.compare_heldout(a, a)
        assert probs == [] and cols["A"]["max_rel"] == 0 and cols["A"]["corr"] > 0.999999 and cols["C"]["n"] == 40
        z = dict(np.load(a))
        b = _scores(os.path.join(d, "b.npz"), A=z["A"] + 1e-3, C=z["C"])
        probs, cols = T.compare_heldout(a, b)
        assert not probs and 0 < cols["A"]["max_rel"] <= 1e-3
        c = _scores(os.path.join(d, "c.npz"), A=z["A"], C=np.r_[np.full(11, np.nan), z["C"][11:]])
        assert any("finite entries of C" in p for p in T.compare_heldout(a, c)[0])
        e = _scores(os.path.join(d, "e.npz"), A=z["A"], C=z["C"], b1_n=np.r_[2.0, np.ones(49)])
        assert any("b1_n" in p for p in T.compare_heldout(a, e)[0])
        f = _scores(os.path.join(d, "f.npz"), A=z["A"][::-1].copy(), C=z["C"])
        probs, cols = T.compare_heldout(a, f)
        assert not probs and cols["A"]["corr"] < 0 and cols["A"]["max_rel"] > 0.5


def test_compare_summary():
    s = dict(genomes=3, genes=15, panel=["g0", "g1"], objectives=["A"], A_mean_neglogp_own=1.25, A_chance_neglogp=1.0986)
    assert T.compare_summary(s, dict(s), 0.05) == []
    assert T.compare_summary(s, dict(s, A_mean_neglogp_own=1.26), 0.05) == []
    assert any("A_mean_neglogp_own" in p for p in T.compare_summary(s, dict(s, A_mean_neglogp_own=1.5), 0.05))
    assert any("genes" in p for p in T.compare_summary(s, dict(s, genes=14), 0.05))
    assert any("panel" in p for p in T.compare_summary(s, dict(s, panel=["g0"]), 0.05))
    assert any("B1_lr_mean" in p for p in T.compare_summary(s, dict(s, B1_lr_mean=0.1), 0.05))


TC = dict(model="selfsup_v0", seed=1, steps=3, batch=4, lr=1e-3, eval_every=2, ckpt_every=2, width=8, max_codons=64, hits_centre=4, max_track=64,
          source_genes=4, source_codons=64, source_hits=4, b2_hits=2, held_batches=1, held_genes=5, dev_seed=0,
          variants=[dict(name="v1", seed=1, objectives=["A", "B1"]), dict(name="v2", seed=2, objectives=["A"])])


class FakeStage:
    def __init__(self, run_dir, config):
        self.run_dir, self.config, self.extra, self.outputs = run_dir, config, {}, []

    def path(self, *parts):
        p = os.path.join(self.run_dir, "data", *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def repo(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)

    def output(self, name, path):
        self.outputs.append(dict(name=name, path=os.path.relpath(path, self.run_dir), sha256=sha256_file(path)))


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


def _source(src, lock):
    """A trained source run: the variants fitted in process, with the records fit_selfsup and the CI write."""
    from seeddrift.train import selfsup as SS
    base, confs = T.selfsup_confs(TC, src, "src-commit")
    outputs = []
    for c in confs:
        _write(os.path.join(src, "data", "train", "selfsup-%s.json" % c["name"]), c)
        SS.fit(src, c)
        for kind, rel in (("heldout_scores_", os.path.join(c["out"], "heldout_scores.npz")), ("data_seen_", os.path.join(c["logs"], "data_seen.json"))):
            outputs.append(dict(name=kind + c["name"], path=rel, sha256=sha256_file(os.path.join(src, rel), use_sidecar=False)))
    outputs.append(dict(name="model", path="data/model/model.npz", sha256=sha256_file(os.path.join(src, confs[0]["out"], "model.npz"), use_sidecar=False)))
    with open(lock) as f:
        env_sha = sha256_json(T.pip_lines(f.read()))
    _write(os.path.join(src, "data", "model", "model.json"), dict(model_id="mdl-test", kind="selfsup_v0", corpus_dataset="ds-corpus", train_data="ds-test",
                                                                  steps=TC["steps"], python_lock=lock, model_env_sha256=env_sha,
                                                                  base={k: v for k, v in base.items() if k != "commit"},
                                                                  variants={c["name"]: {k: c[k] for k in ("seed", "train_fraction", "width", "objectives", "trim", "out", "logs")}
                                                                            for c in confs}))
    _write(os.path.join(src, "stages", "train.json"), dict(name="train", status="passed", outputs=outputs))
    _write(os.path.join(src, "record.json"), dict(run_id="src-run", pipeline="selfsup", profile="p", git=dict(sha="src-commit")))
    return confs


def _run(run, src, lock, **tc):
    ids = os.path.join(src, "data", "train", "ids.json")
    if os.path.exists(ids):
        os.makedirs(os.path.join(run, "data", "train"), exist_ok=True)
        shutil.copyfile(ids, os.path.join(run, "data", "train", "ids.json"))
    cfg = dict(train=dict(TC, models_from=src, python=sys.executable, python_lock=lock, device="cpu", cpu_threads=2, **tc), git=dict(sha="run-commit"))
    st = FakeStage(run, cfg)
    T.import_selfsup(st, cfg, run, dict(corpus_dataset="ds-corpus"))
    return st


def _fails(fn, text):
    try:
        fn()
    except SystemExit as e:
        assert text in str(e), str(e)
        return
    raise AssertionError("no failure with %r" % text)


def test_import_end_to_end_on_cpu():
    _torch()
    old = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = ROOT + os.pathsep + (old or "")
    real_env = T.model_env
    try:
        with tempfile.TemporaryDirectory() as d:
            lock = os.path.join(d, "lock.txt")
            with open(lock, "w") as f:
                f.write("numpy==1\ntorch==2\n")
            T.model_env = lambda st, py, lock_path: T.pip_lines(open(lock_path).read() + "extra==3\n")
            src, run = os.path.join(d, "src"), os.path.join(d, "run")
            make_run(src)
            make_run(run)                                              # the same seed: the same data
            _write(os.path.join(src, "data", "train", "ids.json"), dict(genomes=["g0"], corpus_dataset="ds-corpus"))
            confs = _source(src, lock)
            st = _run(run, src, lock)
            assert [o["name"] for o in st.outputs][:2] == ["model", "imported_from"]
            for c in confs:
                cols = st.extra["recheck"][c["name"]]["columns"]
                assert cols["A"]["n"] > 0 and cols["A"]["max_rel"] < 1e-4 and not st.extra["recheck"][c["name"]]["problems"]
                assert sha256_file(os.path.join(run, c["out"], "model.npz"), use_sidecar=False) == sha256_file(os.path.join(src, c["out"], "model.npz"), use_sidecar=False)
            mj = json.load(open(os.path.join(run, "data", "model", "model.json")))
            assert mj["model_id"] == "mdl-test" and mj["imported_from"]["run"] == "src-run" and mj["imported_from"]["commit"] == "src-commit"
            assert os.path.samefile(os.path.join(run, "data", "model", "model.npz"), os.path.join(run, confs[0]["out"], "model.npz"))
            # another configuration, other data, a swapped model, a missing copy: refused
            r2 = os.path.join(d, "r2")
            make_run(r2)
            _fails(lambda: _run(r2, src, lock, lr=2e-3), "configuration differs")
            r3 = os.path.join(d, "r3")
            make_run(r3)
            idx = json.load(open(os.path.join(r3, "data", "train_data", "index.json")))
            _write(os.path.join(r3, "data", "train_data", "index.json"), dict(idx, dataset_id="ds-other"))
            _fails(lambda: _run(r3, src, lock), "training data")
            shutil.copyfile(os.path.join(src, confs[0]["out"], "model.npz"), os.path.join(src, confs[1]["out"], "model.npz"))
            r4 = os.path.join(d, "r4")
            make_run(r4)
            _fails(lambda: _run(r4, src, lock), "do not reproduce")
            _fails(lambda: _run(r4, os.path.join(d, "nothing"), lock), "lacks")
    finally:
        T.model_env = real_env
        if old is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
