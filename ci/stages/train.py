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
    # the partner's (donor's or decoy's) clade hidden at its own depth, below the rank it shares with the recipient
    # (seeddrift/train/source_model.py partner_depth); empty: at the recipient's depth, as before
    pds = tc.get("partner_depths") or []
    if not isinstance(pds, list):
        raise SystemExit("train.partner_depths: a list of ranks: %r" % (pds,))
    pds = [str(x) for x in pds]
    if set(pds) - {"species", "genus", "family", "order", "class", "phylum"} or len(set(pds)) != len(pds):
        raise SystemExit("train.partner_depths: distinct ranks among species, genus, family, order, class, phylum: %r" % pds)
    if pds:
        base["partner_depths"] = pds
    variants = [dict(v) for v in tc.get("variants") or []]
    for v in variants:
        if set(v) - {"name", "seed", "fraction", "width", "ablate", "partner_depths"} or not v.get("name") or "seed" not in v:
            raise SystemExit("train.variants: each has a name and a seed, and only fraction, width, ablate, partner_depths besides: %r" % v)
        if "partner_depths" in v:
            pv = v["partner_depths"] or []
            if not isinstance(pv, list) or set(map(str, pv)) - {"species", "genus", "family", "order", "class", "phylum"} or len(set(pv)) != len(pv):
                raise SystemExit("train.variants: partner_depths a list of distinct ranks among species, genus, family, order, class, phylum: %r" % v)
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
            c = dict(base, seed=int(v["seed"]), train_fraction=float(v.get("fraction", 1.0)), width=int(v.get("width", base["width"])),
                     ablate=str(v.get("ablate", "")), name=v["name"], logs=logs, out=out,
                     resume=bool(tc.get("resume")) and os.path.exists(os.path.join(run_dir, logs, "last.npz")))
            if "partner_depths" in v:                           # this variant trains on its own construction (a control arm);
                c["partner_depths"] = [str(x) for x in v["partner_depths"] or []]   # scored on the run's development set
            confs.append(c)
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
                                                  variants={c["name"]: dict(dict(seed=c["seed"], fraction=c["train_fraction"], width=c["width"], ablate=c["ablate"],
                                                                                 out=c["out"], logs=c["logs"]),
                                                                            **({"partner_depths": c["partner_depths"]} if "partner_depths" in v else {}))
                                                            for c, v in zip(confs, variants)} if variants else {}))
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


if __name__ == "__main__":
    {"ids": do_ids, "fit": do_fit}[sys.argv[2]](sys.argv[1])
