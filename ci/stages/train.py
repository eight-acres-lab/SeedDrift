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
        if tc.get("model") == "source_v0":
            return fit_source(st, cfg, run_dir, ids_doc, tax)
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


MODEL_LOCK = os.path.join("ci", "envs", "model-torch.pip.txt")


def pip_lines(text):
    return sorted(l.strip() for l in text.splitlines() if l.strip() and not l.startswith("#"))


def fit_source(st, cfg, run_dir, ids_doc, tax):
    """Source model v0 (strategy: model-pilot), trained by seeddrift/train/source_model.py in the CUDA Python of the server.
    That interpreter's packages must equal ci/envs/model-torch.pip.txt (else the stage fails: the model environment is
    locked like the CI's own)."""
    import subprocess
    tc = cfg["train"]
    py = tc.get("python", "/usr/local/bin/python3")
    r = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(st.repo(MODEL_LOCK)) as f:
        want = pip_lines(f.read())
    have = pip_lines(r.stdout)
    if r.returncode != 0 or have != want:
        diff = sorted(set(have) ^ set(want))[:20]
        raise SystemExit("the model environment %s differs from %s: %s" % (py, MODEL_LOCK, diff))
    import csv as _csv
    hidden = {}
    sc = cfg.get("scenario") or {}
    if sc.get("truth"):
        with open(st.repo(sc["truth"])) as f:
            for t in _csv.DictReader(f, delimiter="\t"):
                if t.get("donor_taxid"):
                    g = tax.at_rank(int(t["donor_taxid"]), "genus")
                    if g:
                        hidden.setdefault(t["scenario"], set()).add(g)
    seeds = [int(x) for x in (tc.get("seeds") or [tc["seed"]])]
    base = dict(lr=float(tc["lr"]), steps=int(tc["steps"]), batch=int(tc["batch"]), eval_every=int(tc.get("eval_every", 500)),
                ckpt_every=int(tc.get("ckpt_every", 1000)), commit=cfg["git"]["sha"], hidden_genera={k: sorted(v) for k, v in hidden.items()},
                max_codons=int(tc.get("max_codons", 512)), hits_centre=int(tc.get("hits_centre", 16)), hits_neighbour=int(tc.get("hits_neighbour", 4)),
                max_track=int(tc.get("max_track", 512)), dev_seed=int(tc.get("dev_seed", 0)), baseline_per_genome=int(tc.get("baseline_per_genome", 60)))
    script = os.path.join(st.repo("seeddrift/train/source_model.py"))
    env = dict(os.environ, PYTHONHASHSEED="0")

    def run(mode, conf, name):
        cp = st.path("train", name + ".json")
        write_json(cp, conf)
        logp = st.path("train", name + ".log")
        lg = open(logp, "a")
        return subprocess.Popen([py, script, mode, run_dir, cp], stdout=lg, stderr=subprocess.STDOUT, env=env), logp, lg

    def wait(jobs):
        bad = []
        for pr, logp, lg in jobs:
            rc = pr.wait()
            lg.close()
            if rc != 0:
                with open(logp) as f:
                    bad.append("%s (exit %d): %s" % (os.path.basename(logp), rc, f.read()[-1500:]))
        if bad:
            raise SystemExit("source model failed: %s" % " | ".join(bad))
    wait([run("sets", base, "source_sets")])                    # the constructed example lists (dev set, baseline training set)
    # one folder per seed (data/model/seed-<s>, data/train/logs/seed-<s>); with one seed the plain folders, as before
    multi = len(seeds) > 1
    confs = []
    for sd in seeds:
        sub = "seed-%d" % sd if multi else ""
        logs = os.path.join("data", "train", "logs", sub) if multi else os.path.join("data", "train", "logs")
        out = os.path.join("data", "model", sub) if multi else os.path.join("data", "model")
        confs.append(dict(base, seed=sd, logs=logs, out=out,
                          resume=bool(tc.get("resume")) and os.path.exists(os.path.join(run_dir, logs, "last.npz"))))
    # ablations (train.ablations: hits, codons): the same network with one input channel cut, first seed, own folders
    abl = []
    for a in tc.get("ablations") or []:
        abl.append(dict(base, seed=seeds[0], ablate=str(a), logs=os.path.join("data", "train", "logs", "ablate-%s" % a),
                        out=os.path.join("data", "model", "ablate-%s" % a), resume=False))
    jobs = [(c, "source_config" + ("-seed-%d" % c["seed"] if multi else "")) for c in confs] + [(c, "source_config-ablate-%s" % c["ablate"]) for c in abl]
    par = max(1, int(tc.get("parallel", 1)))
    for k in range(0, len(jobs), par):
        wait([run("fit", c, name) for c, name in jobs[k:k + par]])
    model_dir = os.path.join(run_dir, "data", "model")
    if multi:                                                   # the first seed is the run's model; the others sit beside it
        for name in ("model.npz", "eval_scores.npz", "dev_eval.npz"):
            dst = os.path.join(model_dir, name)
            if os.path.exists(dst):
                os.remove(dst)
            os.link(os.path.join(run_dir, confs[0]["out"], name), dst)
    with open(os.path.join(run_dir, "data", "train_data", "index.json")) as f:
        tdi = json.load(f)
    mid = "mdl-source-v0-" + sha256_json([cfg["git"]["sha"], cfg["config_sha256"], [ids_doc["corpus_dataset"], tdi["dataset_id"]], tc["seed"]])[:12]
    write_json(st.path("model", "model.json"), dict(model_id=mid, kind="source_v0", note="pilot: development diagnosis only (charter section 8)",
                                                  steps=int(tc["steps"]), corpus_dataset=ids_doc["corpus_dataset"], train_data=tdi["dataset_id"],
                                                  model_env_sha256=sha256_json(have), seeds=seeds, seed_dirs=[c["out"] for c in confs],
                                                  ablations={c["ablate"]: c["out"] for c in abl}))
    for name in ("model.npz", "eval_scores.npz", "dev_eval.npz", "dev_set.npz", "baseline_train.npz"):
        st.output(name.split(".")[0], os.path.join(model_dir, name))
    for c in confs[1:] if multi else []:
        st.output("dev_eval_seed_%d" % c["seed"], os.path.join(run_dir, c["out"], "dev_eval.npz"))
    for c in abl:
        st.output("dev_eval_ablate_%s" % c["ablate"], os.path.join(run_dir, c["out"], "dev_eval.npz"))
    st.output("data_seen", os.path.join(run_dir, confs[0]["logs"], "data_seen.json"))
    st.extra.update(model_id=mid, steps=int(tc["steps"]), model_env=py, seeds=seeds)


if __name__ == "__main__":
    {"ids": do_ids, "fit": do_fit}[sys.argv[2]](sys.argv[1])
