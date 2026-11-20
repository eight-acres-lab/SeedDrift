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
import re
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
        if tc.get("model") == "selfsup_v0":
            return fit_selfsup(st, cfg, run_dir, ids_doc)
        if tc.get("model") not in (None, "toy_source_id"):
            raise SystemExit("train.model %r is not known (source_v0, selfsup_v0, or toy_source_id, the pipeline check)" % tc.get("model"))
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


def run_slots(jobs, start, parallel, exclusive_width=0, poll=5.0):
    """Run (config, name) jobs, up to `parallel` at once, each started when a slot frees; a job whose config is at least
    `exclusive_width` wide runs alone (its activations need most of the GPU's memory). start(config, name) returns
    (process, log path, open log). Returns [(name, jobs running when it started)]; raises after all have ended if any
    failed (a failure stops new starts)."""
    running, bad, started = [], [], []

    def reap(block):
        while running:
            for job in list(running):
                pr, logp, lg = job
                if pr.poll() is not None:
                    running.remove(job)
                    lg.close()
                    if pr.returncode != 0:
                        with open(logp) as f:
                            bad.append("%s (exit %d): %s" % (os.path.basename(logp), pr.returncode, f.read()[-1500:]))
                    if not block:
                        return
            time.sleep(poll)
    for c, name in jobs:
        alone = bool(exclusive_width) and int(c.get("width", 128)) >= exclusive_width
        while running and (alone or len(running) >= parallel):
            reap(block=alone)
        if bad:
            break
        started.append((name, len(running)))
        running.append(start(c, name))
        if alone:
            reap(block=True)
    reap(block=True)
    if bad:
        raise SystemExit("source model failed: %s" % " | ".join(bad))
    return started


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
                max_track=int(tc.get("max_track", 512)), dev_seed=int(tc.get("dev_seed", 0)), baseline_per_genome=int(tc.get("baseline_per_genome", 60)),
                genome_composition=bool(tc.get("genome_composition", False)), width=int(tc.get("width", 128)),
                baseline_depth=str(tc.get("baseline_depth", "genus")))
    variants = [dict(v) for v in tc.get("variants") or []]
    for v in variants:
        if set(v) - {"name", "seed", "fraction", "width", "ablate"} or not v.get("name") or "seed" not in v:
            raise SystemExit("train.variants: each has a name and a seed, and only fraction, width, ablate besides: %r" % v)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(v["name"])) or not 0 < float(v.get("fraction", 1.0)) <= 1 or int(v.get("width", 128)) % 4 \
                or v.get("ablate", "") not in ("", "hits", "codons"):
            raise SystemExit("train.variants: a name of letters, digits, . _ -; a fraction in (0, 1]; a width divisible by 4; ablate hits or codons: %r" % v)
    if len({v["name"] for v in variants}) != len(variants):
        raise SystemExit("train.variants: names repeat")
    base["fractions"] = sorted({float(v.get("fraction", 1.0)) for v in variants} | {1.0})
    script = os.path.join(st.repo("seeddrift/train/source_model.py"))
    # expandable segments: the CUDA allocator grows its segments instead of reserving new blocks; without it a width-128
    # training reserved 30 GB for 10 GB of tensors and four at once ran out of memory (measured 2026-11-18). Memory
    # layout only, not the numbers.
    alloc = str(tc.get("cuda_alloc_conf", "expandable_segments:True"))
    env = dict(os.environ, PYTHONHASHSEED="0", PYTORCH_CUDA_ALLOC_CONF=alloc)
    st.extra["cuda_alloc_conf"] = alloc

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
    if variants:
        # train.variants: a grid (data fraction, width, seed, ablation), each in data/model/<name> and data/train/logs/<name>;
        # the first variant is the run's model
        multi = True
        confs = []
        for v in variants:
            logs, out = os.path.join("data", "train", "logs", v["name"]), os.path.join("data", "model", v["name"])
            confs.append(dict(base, seed=int(v["seed"]), train_fraction=float(v.get("fraction", 1.0)), width=int(v.get("width", base["width"])),
                              ablate=str(v.get("ablate", "")), name=v["name"], logs=logs, out=out,
                              resume=bool(tc.get("resume")) and os.path.exists(os.path.join(run_dir, logs, "last.npz"))))
        abl = []
        jobs = [(c, "source_config-%s" % c["name"]) for c in confs]
        seeds = [c["seed"] for c in confs]
    else:
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
    log_ = run_slots(jobs, lambda c, name: run("fit", c, name), max(1, int(tc.get("parallel", 1))), int(tc.get("exclusive_width", 0) or 0))
    st.extra["schedule"] = log_
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
                                                  model_env_sha256=sha256_json(have), seeds=[] if variants else seeds,
                                                  seed_dirs=[] if variants else [c["out"] for c in confs], ablations={c["ablate"]: c["out"] for c in abl},
                                                  variants={c["name"]: dict(seed=c["seed"], fraction=c["train_fraction"], width=c["width"], ablate=c["ablate"],
                                                                            out=c["out"], logs=c["logs"]) for c in confs} if variants else {}))
    for name in ("model.npz", "eval_scores.npz", "dev_eval.npz", "dev_set.npz", "baseline_train.npz", "fractions.json"):
        st.output(name.split(".")[0], os.path.join(model_dir, name))
    for f in base["fractions"]:
        if f < 1.0:
            st.output("baseline_train_f%g" % f, os.path.join(model_dir, "baseline_train_f%g.npz" % f))
    for c in (confs[1:] if multi else []):
        st.output("dev_eval_%s" % (c["name"] if variants else "seed_%d" % c["seed"]), os.path.join(run_dir, c["out"], "dev_eval.npz"))
    for c in abl:
        st.output("dev_eval_ablate_%s" % c["ablate"], os.path.join(run_dir, c["out"], "dev_eval.npz"))
    st.output("data_seen", os.path.join(run_dir, confs[0]["logs"], "data_seen.json"))
    st.extra.update(model_id=mid, steps=int(tc["steps"]), model_env=py, seeds=seeds)


SELFSUP_KEYS = ("objectives", "width", "fraction", "trim")


def model_env(st, py, lock=MODEL_LOCK):
    """The packages of the model interpreter; they must equal the lock (train.python_lock, default
    ci/envs/model-torch.pip.txt), else the stage fails."""
    import subprocess
    r = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(st.repo(lock)) as f:
        want = pip_lines(f.read())
    have = pip_lines(r.stdout)
    if r.returncode != 0 or have != want:
        raise SystemExit("the model environment %s differs from %s: %s" % (py, lock, sorted(set(have) ^ set(want))[:20]))
    return have


def gpu_free_mb():
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=30, check=True).stdout
        return sum(float(a) - float(b) for a, b in (line.split(",") for line in out.strip().splitlines()))
    except Exception:
        return None


def selfsup_confs(tc, run_dir, commit, has_pairs=False):
    """The configurations of the self-supervised variants (train.variants: name, seed, and optionally objectives of A,
    B1, B2, width, fraction, trim; without variants one model "main" with train.seed)."""
    base = dict(lr=float(tc["lr"]), steps=int(tc["steps"]), batch=int(tc["batch"]), eval_every=int(tc.get("eval_every", 500)),
                ckpt_every=int(tc.get("ckpt_every", 1000)), commit=commit, max_codons=int(tc.get("max_codons", 512)),
                hits_centre=int(tc.get("hits_centre", 32)), max_track=int(tc.get("max_track", 512)), dev_seed=int(tc.get("dev_seed", 0)),
                width=int(tc.get("width", 128)), objectives=list(tc.get("objectives") or ["A", "B1", "B2"]), trim=float(tc.get("trim", 0.1)),
                trim_genes=float(tc.get("trim_genes", 0.2)), source_genes=int(tc.get("source_genes", 64)), source_codons=int(tc.get("source_codons", 256)),
                source_hits=int(tc.get("source_hits", 8)), b2_hits=int(tc.get("b2_hits", 8)), held_batches=int(tc.get("held_batches", 8)),
                held_genes=int(tc.get("held_genes", 200)), c_batch=int(tc.get("c_batch", tc["batch"])), c_partners=int(tc.get("c_partners", 16)),
                pairs_dir=os.path.join("data", "pairs"))
    # train.train_splits: the splits the model trains on (default train; [train, heldout]: the whole corpus, the held-out
    # diagnosis then in-sample). The sealed test part is never loaded by the model.
    ts = tc.get("train_splits")
    if ts is not None:
        if not isinstance(ts, list) or not ts or set(ts) - {"train", "heldout"} or len(set(ts)) != len(ts):
            raise SystemExit("train.train_splits: a list of distinct splits among train, heldout: %r" % (ts,))
        base["train_splits"] = list(ts)
    variants = [dict(v) for v in tc.get("variants") or [dict(name="main", seed=int(tc["seed"]))]]
    for v in variants:
        if set(v) - {"name", "seed"} - set(SELFSUP_KEYS) or not v.get("name") or "seed" not in v:
            raise SystemExit("train.variants: each has a name and a seed, and only %s besides: %r" % (", ".join(SELFSUP_KEYS), v))
        obj = v.get("objectives", base["objectives"])
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(v["name"])) or not 0 < float(v.get("fraction", 1.0)) <= 1 or int(v.get("width", 128)) % 4 \
                or not obj or set(obj) - {"A", "B1", "B2", "C"} or not 0 <= float(v.get("trim", base["trim"])) < 1:
            raise SystemExit("train.variants: a name of letters, digits, . _ -; a fraction in (0, 1]; a width divisible by 4; objectives of A, B1, "
                             "B2, C; a trim in [0, 1): %r" % v)
        if "C" in obj and not has_pairs:
            raise SystemExit("train.variants: objective C needs the pairs stage (a `pairs:` block in the profile): %r" % v)
    if len({v["name"] for v in variants}) != len(variants):
        raise SystemExit("train.variants: names repeat")
    confs = []
    for v in variants:
        logs, out = os.path.join("data", "train", "logs", v["name"]), os.path.join("data", "model", v["name"])
        confs.append(dict(base, seed=int(v["seed"]), name=v["name"], train_fraction=float(v.get("fraction", 1.0)), width=int(v.get("width", base["width"])),
                          objectives=list(v.get("objectives", base["objectives"])), trim=float(v.get("trim", base["trim"])), logs=logs, out=out,
                          resume=bool(tc.get("resume")) and os.path.exists(os.path.join(run_dir, logs, "last.npz"))))
    return base, confs


def selfsup_env(tc):
    """The environment of the model's processes. train.device: auto (default; the GPU when there is one) or cpu (the
    GPUs hidden, CUDA_VISIBLE_DEVICES empty, and train.cpu_threads threads for torch, default 8)."""
    env = dict(os.environ, PYTHONHASHSEED="0", PYTORCH_CUDA_ALLOC_CONF=str(tc.get("cuda_alloc_conf", "expandable_segments:True")))
    device = str(tc.get("device", "auto"))
    if device not in ("auto", "cpu"):
        raise SystemExit("train.device is auto or cpu, not %r" % device)
    if device == "cpu":
        n = str(int(tc.get("cpu_threads", 8)))
        env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=n, MKL_NUM_THREADS=n, OPENBLAS_NUM_THREADS=n)
    return env


def fit_selfsup(st, cfg, run_dir, ids_doc):
    """Self-supervised model v0 (truth-labels line; seeddrift/train/selfsup.py) in the CUDA Python of the machine, whose
    packages are locked like those of the source model. The training and the scoring of the held-out genomes hold the
    machine's GPU lease (ci/gpu_lease.py: on a machine where another service holds the GPU, that service is stopped for
    this part and started again afterwards); with the lease held, the free GPU memory must reach gpu_mem_min_mb.
    With train.models_from the variants are imported from another run instead (import_selfsup)."""
    import subprocess
    from ci import gpu_lease
    tc = cfg["train"]
    if tc.get("models_from"):
        return import_selfsup(st, cfg, run_dir, ids_doc)
    py = tc.get("python", "/usr/local/bin/python3")
    have = model_env(st, py, tc.get("python_lock", MODEL_LOCK))
    base, confs = selfsup_confs(tc, run_dir, cfg["git"]["sha"], has_pairs=bool(cfg.get("pairs")))
    script = st.repo("seeddrift/train/selfsup.py")
    env = selfsup_env(tc)
    st.extra["cuda_alloc_conf"] = env["PYTORCH_CUDA_ALLOC_CONF"]

    def start(conf, name):
        cp = st.path("train", name + ".json")
        write_json(cp, conf)
        logp = st.path("train", name + ".log")
        lg = open(logp, "a")
        pr = subprocess.Popen([py, script, "fit", run_dir, cp], stdout=lg, stderr=subprocess.STDOUT, env=env)
        lease.add(pr.pid)                                       # the lease stays held while the training lives
        return pr, logp, lg
    with gpu_lease.hold("train-" + os.path.basename(run_dir), enabled=bool(tc.get("gpu_lease", True))) as lease:
        st.extra["gpu_lease"] = lease
        need, free = int(cfg.get("gpu_mem_min_mb", 0)), gpu_free_mb()
        if need and (free is None or free < need):
            raise SystemExit("with the GPU lease held %.0f MB of GPU memory are free; the profile needs %d" % (free or 0, need))
        st.extra["schedule"] = run_slots([(c, "selfsup-%s" % c["name"]) for c in confs], start, max(1, int(tc.get("parallel", 1))),
                                         int(tc.get("exclusive_width", 0) or 0))
    model_dir = os.path.join(run_dir, "data", "model")
    for name in ("model.npz", "heldout_scores.npz", "heldout_summary.json"):          # the first variant is the run's model
        dst = os.path.join(model_dir, name)
        if os.path.exists(dst):
            os.remove(dst)
        os.link(os.path.join(run_dir, confs[0]["out"], name), dst)
    with open(os.path.join(run_dir, "data", "train_data", "index.json")) as f:
        tdi = json.load(f)
    mid = "mdl-selfsup-v0-" + sha256_json([cfg["git"]["sha"], cfg["config_sha256"], [ids_doc["corpus_dataset"], tdi["dataset_id"]], [c["seed"] for c in confs]])[:12]
    write_json(st.path("model", "model.json"), dict(model_id=mid, kind="selfsup_v0", note="truth-labels line: development diagnosis only",
                                                  steps=int(tc["steps"]), corpus_dataset=ids_doc["corpus_dataset"], train_data=tdi["dataset_id"],
                                                  model_env_sha256=sha256_json(have), python=py, python_lock=tc.get("python_lock", MODEL_LOCK), base={k: v for k, v in base.items() if k != "commit"},
                                                  variants={c["name"]: {k: c[k] for k in ("seed", "train_fraction", "width", "objectives", "trim", "out", "logs")}
                                                            for c in confs}))
    st.output("model", os.path.join(model_dir, "model.npz"))
    for c in confs:
        st.output("heldout_scores_%s" % c["name"], os.path.join(run_dir, c["out"], "heldout_scores.npz"))
        st.output("data_seen_%s" % c["name"], os.path.join(run_dir, c["logs"], "data_seen.json"))
    st.extra.update(model_id=mid, steps=int(tc["steps"]), model_env=py, variants=[c["name"] for c in confs])


RECHECK_EXACT = ("b1_n", "b2_n", "c_rows")                            # counts the data and the code fix
RECHECK_SCORES = ("A", "b1_aware", "b1_agn", "b2_aware", "b2_agn", "C")


def compare_heldout(theirs, ours):
    """Held-out scores recomputed here (ours) against the imported ones (theirs): the same genes, equal counts, finite
    scores in the same places; per score column the largest and the 99th-percentile relative deviation
    |ours - theirs| / (1 + |theirs|) and the correlation. Returns (problems, columns)."""
    a, b = np.load(theirs), np.load(ours)
    if not (np.array_equal(a["genome"], b["genome"]) and np.array_equal(a["gene"], b["gene"])):
        return ["the held-out genes differ"], {}
    problems, cols = [], {}
    for k in RECHECK_EXACT:
        if k in a.files and not np.array_equal(np.nan_to_num(a[k], nan=-1.0), np.nan_to_num(b[k], nan=-1.0)):
            problems.append("%s differs" % k)
    for k in RECHECK_SCORES:
        if k not in a.files:
            continue
        fa, fb = np.isfinite(a[k]), np.isfinite(b[k])
        if not np.array_equal(fa, fb):
            problems.append("the finite entries of %s differ (%d against %d)" % (k, int(fa.sum()), int(fb.sum())))
            continue
        if not fa.any():
            continue
        x, y = a[k][fa].astype(np.float64), b[k][fa].astype(np.float64)
        rel = np.abs(y - x) / (1.0 + np.abs(x))
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(x) > 0 and np.std(y) > 0 else None
        cols[k] = dict(n=int(fa.sum()), max_rel=float(rel.max()), p99_rel=float(np.quantile(rel, 0.99)), corr=corr)
    return problems, cols


def compare_summary(theirs, ours, max_rel):
    """heldout_summary.json recomputed here (ours) against the imported one (theirs): the same keys, equal counts and
    lists, and the means within the relative deviation max_rel (|ours - theirs| / (1 + |theirs|))."""
    problems = []
    for k in sorted(set(theirs) | set(ours)):
        x, y = theirs.get(k), ours.get(k)
        if isinstance(x, float) or isinstance(y, float):
            if not (isinstance(x, (int, float)) and isinstance(y, (int, float))) or abs(y - x) > max_rel * (1.0 + abs(x)):
                problems.append("summary %s: %r here, %r imported" % (k, y, x))
        elif x != y:
            problems.append("summary %s differs" % k)
    return problems


def import_selfsup(st, cfg, run_dir, ids_doc):
    """train.models_from: the variants trained by another run of this pipeline, for a machine that does not train them
    (another machine holds the GPU or the data cannot move). The value is a folder holding a copy of that run's
    record.json, stages/train.json, data/train (ids.json, the variants' configurations and data_seen.json),
    data/model (model.json, each variant's model.npz, heldout_scores.npz, heldout_summary.json) and the manifests of
    data/train_data, data/pairs and data/corpus. Imported only when
      - its stage train passed and its record is of pipeline selfsup;
      - it trained on the same data: corpus dataset, training-data dataset, split (data/train/ids.json) and, with
        pairs, pair dataset equal to this run's;
      - every variant's configuration equals this run's (all keys but the commit and the resume flag);
      - its model environment is its lock (the repository's file it names) and that lock's packages are in this run's
        model environment;
      - the files its stage record lists (the first variant's model, every variant's held-out scores and data counts)
        have their recorded hashes;
      - every variant, loaded here, reproduces its held-out scores: the same genes and counts, finite in the same
        places, and per score a relative deviation (|ours - theirs| / (1 + |theirs|)) of at most
        train.recheck_max_rel (default 0.05) and a correlation of at least train.recheck_min_corr (default 0.999).
        (Another seed or variant gives other scores; a GPU and a CPU differ in rounding only.) Its heldout_summary.json,
        which the report publishes, matches the recomputed one (compare_summary, the same relative deviation).
    The recomputed scores stay in data/train/recheck/<variant>; the deviations are in the stage record.
    train.models_wait_s (default 0): while the copy is still arriving, wait up to that many seconds for every file
    named here to exist (a copy that renames each file when it is complete, as rsync does)."""
    import shutil
    import subprocess
    tc = cfg["train"]
    src = os.path.abspath(tc["models_from"])
    names = [os.path.join("data", "train", "selfsup-%s.json" % v["name"]) for v in tc.get("variants") or []]
    for v in tc.get("variants") or []:
        names += [os.path.join("data", "model", v["name"], n) for n in ("model.npz", "heldout_scores.npz", "heldout_summary.json")]
        names.append(os.path.join("data", "train", "logs", v["name"], "data_seen.json"))
    names += ["record.json", os.path.join("stages", "train.json"), os.path.join("data", "model", "model.json"), os.path.join("data", "train", "ids.json")]
    t0 = time.time()
    while time.time() - t0 < float(tc.get("models_wait_s", 0)) and not all(os.path.exists(os.path.join(src, n)) for n in names):
        time.sleep(60)
    st.extra["waited_for_models_s"] = round(time.time() - t0)
    missing = [n for n in names if not os.path.exists(os.path.join(src, n))]
    if missing:
        raise SystemExit("train.models_from: %s lacks %s" % (src, ", ".join(missing[:8])))

    def load(*parts):
        with open(os.path.join(src, *parts)) as f:
            return json.load(f)

    def same_file(a, b):
        return sha256_file(a, use_sidecar=False) == sha256_file(b, use_sidecar=False)

    srec, strain, smodel = load("record.json"), load("stages", "train.json"), load("data", "model", "model.json")
    problems = []
    if strain.get("status") != "passed" or srec.get("pipeline") != "selfsup" or smodel.get("kind") != "selfsup_v0":
        raise SystemExit("train.models_from: %s is not a passed training of pipeline selfsup (status %s, pipeline %s, kind %s)"
                         % (src, strain.get("status"), srec.get("pipeline"), smodel.get("kind")))
    with open(os.path.join(run_dir, "data", "train_data", "index.json")) as f:
        tdi = json.load(f)
    if smodel.get("corpus_dataset") != ids_doc["corpus_dataset"]:
        problems.append("corpus dataset %s, here %s" % (smodel.get("corpus_dataset"), ids_doc["corpus_dataset"]))
    if smodel.get("train_data") != tdi["dataset_id"]:
        problems.append("training data %s, here %s" % (smodel.get("train_data"), tdi["dataset_id"]))
    if not same_file(os.path.join(src, "data", "train", "ids.json"), os.path.join(run_dir, "data", "train", "ids.json")):
        problems.append("data/train/ids.json differs")
    has_pairs = bool(cfg.get("pairs"))
    if has_pairs:
        theirs, ours = load("data", "pairs", "manifest.json"), json.load(open(os.path.join(run_dir, "data", "pairs", "manifest.json")))
        if theirs.get("sha256") != ours.get("sha256"):
            problems.append("pair dataset %s, here %s" % (theirs.get("dataset_id"), ours.get("dataset_id")))
    base, confs = selfsup_confs(tc, run_dir, cfg["git"]["sha"], has_pairs=has_pairs)
    if smodel.get("base") != {k: v for k, v in base.items() if k != "commit"}:
        problems.append("base configuration differs: %s" % sorted(k for k in set(base) | set(smodel.get("base") or {})
                                                                    if k != "commit" and base.get(k) != (smodel.get("base") or {}).get(k)))
    if sorted(smodel.get("variants") or {}) != sorted(c["name"] for c in confs):
        problems.append("variants %s, here %s" % (sorted(smodel.get("variants") or {}), [c["name"] for c in confs]))
    for c in confs:
        p = os.path.join(src, "data", "train", "selfsup-%s.json" % c["name"])
        theirs = load("data", "train", "selfsup-%s.json" % c["name"]) if os.path.exists(p) else {}
        drop = ("commit", "resume")
        if {k: v for k, v in theirs.items() if k not in drop} != {k: v for k, v in c.items() if k not in drop}:
            problems.append("variant %s: configuration differs in %s" % (c["name"], sorted(k for k in set(theirs) | set(c) if k not in drop and theirs.get(k) != c.get(k))))
    py = tc.get("python", "/usr/local/bin/python3")
    have = model_env(st, py, tc.get("python_lock", MODEL_LOCK))
    slock = smodel.get("python_lock") or MODEL_LOCK
    with open(st.repo(slock)) as f:
        want = pip_lines(f.read())
    if sha256_json(want) != smodel.get("model_env_sha256"):
        problems.append("the model environment of the source is not %s of this commit" % slock)
    elif set(want) - set(have):
        problems.append("packages of the source's model environment missing here: %s" % sorted(set(want) - set(have))[:10])
    if problems:
        raise SystemExit("train.models_from %s: %s" % (src, "; ".join(problems)))
    recorded = {o["path"]: o["sha256"] for o in strain["outputs"]}
    model_dir = os.path.join(run_dir, "data", "model")
    for c in confs:                                                    # the variants' files, checked where recorded
        for rel in [os.path.join(c["out"], n) for n in ("model.npz", "heldout_scores.npz", "heldout_summary.json")] + [os.path.join(c["logs"], "data_seen.json")]:
            dst = os.path.join(run_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(os.path.join(src, rel), dst)
            if rel in recorded and sha256_file(dst, use_sidecar=False) != recorded[rel]:
                raise SystemExit("train.models_from: %s does not have the hash of the source's stage record" % rel)
    first = os.path.join(confs[0]["out"], "model.npz")
    if recorded.get("data/model/model.npz") != sha256_file(os.path.join(run_dir, first), use_sidecar=False):
        raise SystemExit("train.models_from: %s is not the model the source's stage record names" % first)
    env = selfsup_env(tc)
    script = st.repo("seeddrift/train/selfsup.py")
    max_rel, min_corr = float(tc.get("recheck_max_rel", 0.05)), float(tc.get("recheck_min_corr", 0.999))
    recheck, failed = {}, []
    for c in confs:                                                    # every variant reproduces its held-out scores
        out = os.path.join("data", "train", "recheck", c["name"])
        cp = st.path("train", "recheck-%s.json" % c["name"])
        write_json(cp, dict(c, score_out=out))
        with open(st.path("train", "recheck-%s.log" % c["name"]), "w") as lg:
            rc = subprocess.run([py, script, "score", run_dir, cp], stdout=lg, stderr=subprocess.STDOUT, env=env).returncode
        if rc != 0:
            raise SystemExit("train.models_from: rescoring the held-out genes with %s failed (exit %d); see %s" % (c["name"], rc, cp[:-5] + ".log"))
        probs, cols = compare_heldout(os.path.join(run_dir, c["out"], "heldout_scores.npz"), os.path.join(run_dir, out, "heldout_scores.npz"))
        with open(os.path.join(run_dir, c["out"], "heldout_summary.json")) as f:
            s_theirs = json.load(f)
        with open(os.path.join(run_dir, out, "heldout_summary.json")) as f:
            probs += compare_summary(s_theirs, json.load(f), max_rel)          # the summary the report publishes
        bad = probs +["%s: max relative deviation %.4g" % (k, v["max_rel"]) for k, v in cols.items() if v["max_rel"] > max_rel] \
            + ["%s: correlation %.6f" % (k, v["corr"]) for k, v in cols.items() if v["corr"] is not None and v["corr"] < min_corr]
        recheck[c["name"]] = dict(columns=cols, problems=bad)
        if bad:
            failed.append("%s (%s)" % (c["name"], "; ".join(bad)))
    st.extra["recheck"] = recheck
    if failed:
        raise SystemExit("train.models_from: the imported variants do not reproduce their held-out scores: %s" % ", ".join(failed))
    for name in ("model.npz", "heldout_scores.npz", "heldout_summary.json"):          # the first variant is the run's model
        dst = os.path.join(model_dir, name)
        if os.path.exists(dst):
            os.remove(dst)
        os.link(os.path.join(run_dir, confs[0]["out"], name), dst)
    imported = dict(run=srec.get("run_id"), commit=(srec.get("git") or {}).get("sha"), profile=srec.get("profile"), folder=src,
                    stage_record_sha256=sha256_file(os.path.join(src, "stages", "train.json"), use_sidecar=False),
                    model_env_here=sha256_json(have), python_here=py, device_here=tc.get("device", "auto"),
                    recheck=dict(max_rel=max_rel, min_corr=min_corr))
    write_json(st.path("model", "model.json"), dict(smodel, imported_from=imported))
    write_json(st.path("train", "imported_from.json"), dict(imported, recheck=recheck))
    st.output("model", os.path.join(model_dir, "model.npz"))
    st.output("imported_from", st.path("train", "imported_from.json"))
    for c in confs:
        st.output("heldout_scores_%s" % c["name"], os.path.join(run_dir, c["out"], "heldout_scores.npz"))
        st.output("data_seen_%s" % c["name"], os.path.join(run_dir, c["logs"], "data_seen.json"))
    st.extra.update(model_id=smodel["model_id"], steps=smodel.get("steps"), model_env=py, variants=[c["name"] for c in confs], imported_from=src,
                    imported_run=imported["run"])


if __name__ == "__main__":
    {"ids": do_ids, "fit": do_fit}[sys.argv[2]](sys.argv[1])
