import json
import os

import numpy as np
import pytest

from seeddrift.train.logger import Diverged, TrainLogger, save_npz


def test_npz_is_byte_identical(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    save_npz(str(a), x=np.arange(5), y=np.ones(3))
    save_npz(str(b), y=np.ones(3), x=np.arange(5))
    assert a.read_bytes() == b.read_bytes()


def test_nan_loss_fails_as_diverged(tmp_path):
    log = TrainLogger(str(tmp_path))
    log.step(1, 0.5)
    with pytest.raises(Diverged):
        log.step(2, float("nan"))
    rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    assert rows[-1]["diverged"] is True


def test_checkpoints_are_thinned_and_resumable(tmp_path):
    log = TrainLogger(str(tmp_path), keep_last=2, keep_every=100)
    for s in range(1, 6):
        log.checkpoint(s, {"W": np.full(2, s)}, val_metric=float(s))
    kept = sorted(n for n in os.listdir(tmp_path) if n.startswith("ckpt_step") and n.endswith(".npz"))
    assert kept == ["ckpt_step1.npz", "ckpt_step4.npz", "ckpt_step5.npz"]
    assert int(log.last()["step"]) == 5 and json.load(open(tmp_path / "best.json"))["step"] == 5
    (tmp_path / "metrics.jsonl").write_text("")
    with pytest.raises(FileExistsError):
        TrainLogger(str(tmp_path))
