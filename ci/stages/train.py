"""Stage train: `python -m ci.stages.train RUN_DIR ids|fit`.

ids   writes data/train/ids.json, the exact genome and gene list the loader will read (gate G1/G2 run on it next)
fit   trains the model of the profile and records everything with seeddrift.train.logger.TrainLogger

Bootstrap model ("toy_source_id", a pipeline check, not a scientific model): softmax regression from a gene's
coding-DNA 3-mer profile to the genome it came from, i.e. "which genome does this gene belong to" in miniature. Its
foreignness score for a gene is 1 - p(the gene's own genome). It exists so that every part of the training path
(loader, gates on the loaded ids, logging, checkpoints, resume, divergence stop, model id) is exercised by every run.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from itertools import product

import numpy as np

from ci.hashing import sha256_file, sha256_json
from ci.kingdom import KingdomMap
from ci.stages.common import Stage, read_fasta, write_json
from ci.taxonomy import Taxonomy
from seeddrift.train.logger import Diverged, TrainLogger, save_npz

KMERS = {"".join(k): i for i, k in enumerate(product("ACGT", repeat=3))}


def features(cds: str) -> np.ndarray:
    v = np.zeros(65)
    s = cds.upper()
    for i in range(len(s) - 2):
        j = KMERS.get(s[i:i + 3])
        if j is not None:
            v[j] += 1
    t = v[:64].sum()
    if t:
        v[:64] /= t
    v[:64] = (v[:64] - 1 / 64) * 64
    v[64] = 1.0
    return v


def load_genes(run_dir, ids):
    X, y, gid = [], [], []
    for k, g in enumerate(ids):
        cds = read_fasta(os.path.join(run_dir, "data", "corpus", g, "cds.fna"))
        for name in sorted(cds):
            X.append(features(cds[name]))
            y.append(k)
            gid.append(name)
    return np.array(X), np.array(y), gid


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def model_id(cfg, dataset_ids):
    return "mdl-toy-" + sha256_json([cfg["git"]["sha"], cfg["config_sha256"], sorted(dataset_ids), cfg["train"]["seed"]])[:12]


def do_ids(run_dir):
    with Stage(run_dir, "train_ids", requires=("pre", "corpus")) as st:
        with open(os.path.join(run_dir, "data", "corpus", "manifest.json")) as f:
            man = json.load(f)
        ids = [e["id"] for e in man["entries"]]
        p = st.path("train", "ids.json")
        write_json(p, dict(genomes=ids, corpus_dataset=man["dataset_id"]))
        st.output("train_ids", p)


def do_fit(run_dir):
    with Stage(run_dir, "train", requires=("pre", "corpus", "train")) as st:
        cfg = st.config
        tc = cfg["train"]
        ids_path = os.path.join(run_dir, "data", "train", "ids.json")
        with open(ids_path) as f:
            ids_doc = json.load(f)
        ids = ids_doc["genomes"]
        with open(os.path.join(run_dir, "gates", "train.ok")) as f:
            gate = json.load(f)
        with open(os.path.join(run_dir, "data", "corpus", "manifest.json")) as f:
            man = json.load(f)
        if gate.get("inputs") != sha256_json([man["sha256"], sha256_file(ids_path, use_sidecar=False)]):
            raise SystemExit("the training ids or the corpus changed after gate G2_train decided on them")
        from ci import manifest as M
        used = dict(man, entries=[e for e in man["entries"] if e["id"] in set(ids)])
        bad = M.verify(used, run_dir)
        if bad:
            raise SystemExit("corpus files differ from their manifest: %s" % bad[:5])
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        km = KingdomMap()
        group_of = {e["id"]: km.group(tax, e["taxid"], e["source_kind"]) for e in man["entries"]}
        X, y, names = load_genes(run_dir, ids)
        val = np.array([int(hashlib.sha256(n.encode()).hexdigest(), 16) % 10 == 0 for n in names])
        tr = np.flatnonzero(~val)
        va = np.flatnonzero(val)
        probes = va[:8]
        folder = st.path("train", "logs", "x")[:-2]
        resume = bool(tc.get("resume")) and os.path.exists(os.path.join(folder, "last.npz"))
        log = TrainLogger(folder, commit=cfg["git"]["sha"], dataset_ids=[ids_doc["corpus_dataset"]], resume=resume)
        rng = np.random.default_rng(int(tc["seed"]))
        W = np.zeros((X.shape[1], len(ids)))
        start = 0
        lr, batch, steps = float(tc["lr"]), int(tc["batch"]), int(tc["steps"])
        order = hashlib.sha256()
        if resume:
            last = log.last()
            W, start = last["W"], int(last["step"])
            for _ in range(start):                          # replay the batch draws: same data order, same counts, same order hash
                b = rng.choice(tr, size=min(batch, len(tr)), replace=False)
                order.update(b.tobytes())
                log.count(group_of[ids[y[i]]] for i in b)
            st.extra["resumed_from_step"] = start
        stop_at = int(os.environ.get("CI_STOP_AFTER_STEPS", "0"))     # test hook for kill and resume; not part of the configuration
        for step in range(start + 1, steps + 1):
            t = time.time()
            b = rng.choice(tr, size=min(batch, len(tr)), replace=False)
            order.update(b.tobytes())
            P = softmax(X[b] @ W)
            onehot = np.zeros_like(P)
            onehot[np.arange(len(b)), y[b]] = 1
            per = -np.log(np.maximum(P[np.arange(len(b)), y[b]], 1e-12))
            loss = per.mean()
            grad = X[b].T @ (P - onehot) / len(b)
            W -= lr * grad
            by_group = {}
            for i, v in zip(b, per):
                by_group.setdefault(group_of[ids[y[i]]], []).append(v)
            log.count(group_of[ids[y[i]]] for i in b)
            try:
                log.step(step, loss, lr=lr, grad_norm=float(np.linalg.norm(grad)), examples_per_s=len(b) / max(time.time() - t, 1e-9),
                         loss_by_group={k: float(np.mean(v)) for k, v in sorted(by_group.items())})
            except Diverged:
                st.extra["status_note"] = "diverged"
                raise
            if step % int(tc.get("eval_every", 5)) == 0 or step == steps:
                Pv = softmax(X[va] @ W)
                acc = float((Pv.argmax(1) == y[va]).mean()) if len(va) else 0.0
                vloss = float(-np.log(np.maximum(Pv[np.arange(len(va)), y[va]], 1e-12)).mean()) if len(va) else 0.0
                log.evaluate(step, val_accuracy=acc, val_loss=vloss)
                log.probes(step, softmax(X[probes] @ W)[np.arange(len(probes)), y[probes]])
            if step % int(tc.get("ckpt_every", 5)) == 0 or step == steps:
                Pv = softmax(X[va] @ W) if len(va) else None
                acc = float((Pv.argmax(1) == y[va]).mean()) if len(va) else None
                log.checkpoint(step, dict(W=W), val_metric=acc, rng_state=rng.bit_generator.state, final=step == steps)
            if stop_at and step == stop_at:
                st.extra["stopped_at_step"] = step
                raise SystemExit("stopped after step %d on request (kill-and-resume test)" % step)
        log.save_seen(order.hexdigest())
        out = st.path("model", "model.npz")
        mid = model_id(cfg, [ids_doc["corpus_dataset"]])
        write_json(st.path("model", "model.json"), dict(model_id=mid, kind="toy_source_id", note="pipeline check, not a scientific model", steps=steps,
                                                     classes=ids, corpus_dataset=ids_doc["corpus_dataset"]))
        save_npz(out, W=W, classes=np.array(ids))                  # the rule's target last
        st.output("model", out)
        st.output("data_seen", os.path.join(folder, "data_seen.json"))
        st.extra.update(model_id=mid, steps=steps)


if __name__ == "__main__":
    {"ids": do_ids, "fit": do_fit}[sys.argv[2]](sys.argv[1])
