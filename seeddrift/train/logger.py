"""TrainLogger: the record of a training run, kept even when the run is killed.

  metrics.jsonl   one JSON line every `every` steps: step, epoch, wall time, loss and its parts, loss per group,
                  learning rate, gradient norm, throughput, memory; appended, flushed and fsynced at once
  eval.jsonl      periodic validation metrics; probes.jsonl: a fixed probe set scored at every evaluation
  data_seen.json  how many examples of each group the loader actually served, and a hash of the order
  checkpoints     ckpt_step<N>.npz (atomic: written to a temporary name, then renamed) with a sidecar JSON (step,
                  sha256, validation metric, RNG state hash, commit, dataset ids); last.npz carries everything a
                  resume needs; best.npz the best validation metric; older periodic checkpoints are thinned to the
                  last `keep_last` plus every `keep_every`-th
A NaN or infinite loss stops the run (status "diverged") and leaves the last good checkpoint in place.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import time
import zipfile

import numpy as np


class Diverged(RuntimeError):
    pass


def save_npz(path, compress=False, **arrays):
    """Like numpy.savez, but byte-identical for identical arrays: members in sorted order with a fixed timestamp
    (numpy stamps the current time into the archive, which would make every hash of an unchanged result differ).
    Written to a temporary name and renamed."""
    tmp = path + ".part"
    method = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(tmp, "w", compression=method) as z:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = method
            info.external_attr = 0o644 << 16
            z.writestr(info, buf.getvalue())
    os.replace(tmp, path)


def _append(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class TrainLogger:
    def __init__(self, folder, commit="", dataset_ids=(), keep_last=3, keep_every=50, resume=False):
        self.dir = folder
        os.makedirs(folder, exist_ok=True)
        self.commit, self.dataset_ids = commit, list(dataset_ids)
        self.keep_last, self.keep_every = keep_last, keep_every
        self.t0 = time.time()
        self.seen = {}
        self.best = None
        if not resume:
            for name in ("metrics.jsonl", "eval.jsonl", "probes.jsonl"):
                p = os.path.join(folder, name)
                if os.path.exists(p):
                    raise FileExistsError("%s exists: start a new run directory or resume" % p)
        else:
            last = os.path.join(folder, "last.json")
            step = json.load(open(last))["step"] if os.path.exists(last) else 0
            for name in ("metrics.jsonl", "eval.jsonl", "probes.jsonl"):     # drop what was logged after the last checkpoint
                p = os.path.join(folder, name)
                if os.path.exists(p):
                    with open(p) as f:
                        rows = [l for l in f if l.strip() and json.loads(l).get("step", 0) <= step]
                    with open(p + ".part", "w") as f:
                        f.writelines(rows)
                    os.replace(p + ".part", p)
            best = os.path.join(folder, "best.json")
            if os.path.exists(best):
                self.best = json.load(open(best)).get("val_metric")

    def step(self, step, loss, **fields):
        if not math.isfinite(float(loss)):
            _append(os.path.join(self.dir, "metrics.jsonl"), dict(step=step, loss=None, diverged=True, wall_s=round(time.time() - self.t0, 3)))
            raise Diverged("loss is %s at step %d" % (loss, step))
        row = dict(step=step, loss=float(loss), wall_s=round(time.time() - self.t0, 3))
        row.update({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in fields.items()})
        _append(os.path.join(self.dir, "metrics.jsonl"), row)

    def evaluate(self, step, **metrics):
        _append(os.path.join(self.dir, "eval.jsonl"), dict(step=step, **{k: float(v) for k, v in metrics.items()}))

    def probes(self, step, scores):
        _append(os.path.join(self.dir, "probes.jsonl"), dict(step=step, scores=[round(float(x), 6) for x in scores]))

    def count(self, groups):
        for g in groups:
            self.seen[g] = self.seen.get(g, 0) + 1

    def save_seen(self, order_hash):
        tmp = os.path.join(self.dir, "data_seen.json.part")
        with open(tmp, "w") as f:
            json.dump(dict(counts=self.seen, order_sha256=order_hash), f, sort_keys=True, indent=1)
        os.replace(tmp, os.path.join(self.dir, "data_seen.json"))

    def checkpoint(self, step, arrays, val_metric=None, rng_state=None, final=False):
        """arrays: dict of numpy arrays (weights, optimizer state, loader position...). Returns the checkpoint path."""
        rng_hash = hashlib.sha256(json.dumps(rng_state, sort_keys=True, default=str).encode()).hexdigest() if rng_state is not None else ""
        name = "ckpt_step%d.npz" % step
        path = os.path.join(self.dir, name)
        save_npz(path, step=np.int64(step), **arrays)
        side = dict(step=step, sha256=sha256_file(path), val_metric=val_metric, rng_state_sha256=rng_hash, commit=self.commit, dataset_ids=self.dataset_ids)
        with open(path[:-4] + ".json", "w") as f:
            json.dump(side, f, sort_keys=True, indent=1)
        for alias in (("last.npz",) + (("best.npz",) if val_metric is not None and (self.best is None or val_metric > self.best) else ()) + (("final.npz",) if final else ())):
            if alias == "best.npz":
                self.best = val_metric
            dst = os.path.join(self.dir, alias)
            tmp2 = dst + ".part.npz"
            with open(path, "rb") as fi, open(tmp2, "wb") as fo:
                fo.write(fi.read())
            os.replace(tmp2, dst)
            with open(dst[:-4] + ".json", "w") as f:
                json.dump(side, f, sort_keys=True, indent=1)
        self._thin()
        return path

    def _thin(self):
        """Keep the last keep_last checkpoints and every one whose step is a multiple of keep_every."""
        steps = sorted(int(n[9:-4]) for n in os.listdir(self.dir) if n.startswith("ckpt_step") and n.endswith(".npz"))
        keep = set(steps[-self.keep_last:]) | {s for s in steps if s % self.keep_every == 0}
        for s in steps:
            if s not in keep:
                for ext in (".npz", ".json"):
                    p = os.path.join(self.dir, "ckpt_step%d%s" % (s, ext))
                    if os.path.exists(p):
                        os.remove(p)

    def last(self):
        p = os.path.join(self.dir, "last.npz")
        return dict(np.load(p)) if os.path.exists(p) else None
