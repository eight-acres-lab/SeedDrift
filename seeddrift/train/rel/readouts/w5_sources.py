"""Readout w5_sources (package W5; a diagnosis of decision D7's arms, never a selection): what arms S1 and S2 take from
their sources, on the held-out tables they were scored on, with the moving average of their final weights (as their
final evaluation reads them). Runs in the model interpreter (torch), on the CPU; nothing is trained and no model code
is changed: the arm is built from its training's conf.json through the registry and read with
models/source.py readout().

`python -m seeddrift.train.rel.readouts.w5_sources CONF_JSON`; CONF_JSON holds arms [{name, fit}] (fit: the arm's
folder of a training run, holding conf.json and final.pt), conditions ({condition: layout}: scattered on W3's strict
offline rendering, the layout of its final evaluation; whole_gene in the stored layout the arms were trained in),
per_group (the evaluator's subset: the first per_group held-out tables of every group), max_rows, threads, out.

Rendering is the evaluator's (trainer/heldout.score): the condition's plan (conditions.materialise), the present rows,
tables.render (stored) or W3's renderer (leak-free, with the cells laid by extension). Per scored cell it writes
(out/<arm>.<condition>.npz): table (position in the held-out set), bits (-log2 p(truth)), none (the weight of no
source), entropy (of the weights over the sources and none, in bits), level (the share of every level of the
substitution among the copy weight), top (the weight of the largest source), top_identity (that row's identity with
row 0 at the target locus, nan without a source), top_is_nearest (it is the row of the largest identity), and for
S1 none_given_truth (the weight of no source given the true codon). out/<arm>.json: the arm, its weights, the
digest of the cells read (the evaluator's, to check against the training's final evaluation) and the seconds.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

from seeddrift.train.rel import api as A
from seeddrift.train.rel import registry
from seeddrift.train.rel.data import sampler as S
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.models import source as SR
from seeddrift.train.rel.trainer import heldout as E

VERSION = 1
OFFLINE_PART = {cond.SCATTERED: "scattered", cond.WHOLE_GENE: "whole_gene", "suffix": "suffix"}


def load_arm(fit):
    """(the training's configuration, the model with the moving average of final.pt, the checkpoint's step and tables)."""
    from seeddrift.train.rel.trainer import loop
    with open(os.path.join(fit, "conf.json")) as f:
        cfg = dict(loop.DEFAULTS, **json.load(f))                          # the trainer's defaults where the file leaves a key out
    module = registry.load("model", cfg["model"])
    if module is not SR:
        raise SystemExit("arm %s is model %s: this readout reads the source module's arms only" % (cfg.get("name"), cfg["model"]))
    model = module.build(loop.size_of(module, cfg))
    ck = torch.load(os.path.join(fit, "final.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(ck["ema"])
    model.eval()
    return cfg, model, dict(step=int(ck.get("step", -1)), tables_seen=int(ck.get("tables_seen", -1)))


def held_data(cfg, condition, layout):
    """(held-out set, arrays of PLAN_FIELDS the condition is rendered from, meta identity/group of the held-out set)."""
    held = T.open_split(cfg["heldout_root"], cfg["heldout_tables"], int(cfg.get("heldout_split", 1)))
    meta = held.load(("identity",))
    if layout == A.LAYOUT_STORED or not cfg.get("heldout_offline_tables"):  # no strict offline rendering: rendered online, as the evaluator
        return held, held.load(E.PLAN_FIELDS), meta
    doc = T.read_manifest(cfg["heldout_offline_tables"])
    part = shards.parts(cfg["heldout_offline_root"], doc)[OFFLINE_PART[cond.parse(condition)[0]]]
    if int(part.tables) != len(held):
        raise SystemExit("the offline part holds %d tables, the held-out set %d" % (part.tables, len(held)))
    return held, part.load(0, part.tables, E.PLAN_FIELDS, verify=False), meta


def examples_of(data, ids, condition, layout, with_ext, chunk=32):
    """The rendered tables of ids under the condition, as the evaluator renders them: dict(grid, truth[, ext], table,
    rows (the original row of every grid row)); and the evaluator's digest of the cells."""
    from seeddrift.train.rel.tables import render as RD
    digest = cond.ItemsDigest(condition)
    out = []
    for at in range(0, len(ids), chunk):
        part = np.asarray(ids[at:at + chunk], dtype=np.int64)
        plan = cond.materialise({f: data[f][part] for f in E.PLAN_FIELDS}, part, condition)
        digest.add(plan)
        for k in range(len(part)):
            if not plan["has"][k]:
                continue
            planned = dict(tok=plan["tok"][k], n=plan["n"][k], col=plan["col"][k], kind=plan["kind"][k], frame_len=plan["frame_len"][k],
                           frame_row=plan["frame_row"][k])
            keep = T.present_rows(planned)
            tab = T.take_rows(planned, keep)
            if layout == A.LAYOUT_STORED:
                grid, truth = T.render(tab, plan["visible"][k][keep], plan["target"][k][keep])
                ext = None
            else:
                grid, truth, ext = RD.render(tab, plan["visible"][k][keep], plan["target"][k][keep], layout=layout)
                ext = ext if with_ext else None
            if not truth.any():
                continue
            rows = np.asarray(keep, dtype=np.int64)                   # present_rows: the stored rows kept, row 0 first (no row moves here)
            e = dict(grid=grid, truth=truth, table=int(part[k]), rows=rows)
            if ext is not None:
                e["ext"] = ext
            out.append(e)
    return out, digest.hexdigest()


@torch.no_grad()
def read_batch(model, examples, identity):
    """Per scored cell of a micro-batch (cells in the order of S.collate): the quantities the module's docstring names."""
    b = E.to_device(S.collate([{k: e[k] for k in ("grid", "truth", "ext") if k in e} for e in examples]), torch.device("cpu"))
    kw = E.ext_of(b, model)
    token = b["token"]
    r = SR.readout(model, b["grid"], b["sizes"], b["row"], b["col"], token=token if model.arm == "s1" else None, **kw)
    lp = model.logp(b["grid"], b["sizes"], b["row"], b["col"], **kw)
    bits = (-lp.gather(1, (token - A.TARGET_MIN)[:, None])[:, 0] / math.log(2.0)).double().numpy()
    p = model.pieces(b["grid"], b["sizes"], b["row"], b["col"], **kw)
    logw, _ = model.weights(p)
    w_rl = logw.exp()                                                         # [M, R, L]
    copy = w_rl.sum(dim=(1, 2), keepdim=False)
    level = (w_rl.sum(dim=1) / copy.clamp_min(1e-30)[:, None]).double().numpy()
    w = r["weights"].double()                                                 # [M, R + 1], last no source
    ent = (-(w * torch.log2(w.clamp_min(1e-300))).sum(dim=1)).numpy()
    src = w[:, :-1]
    top_w, top_j = src.max(dim=1)
    glob = r["rows"].gather(1, top_j[:, None])[:, 0].numpy()                  # the batch row of the largest source
    ex = b["example"].numpy()
    starts = np.cumsum([0] + [int(e["grid"].shape[0]) for e in examples])
    tables = np.array([examples[k]["table"] for k in ex], dtype=np.int64)
    top_id = np.full(len(ex), np.nan)
    top_near = np.zeros(len(ex), dtype=bool)
    for m in range(len(ex)):
        if glob[m] < 0 or float(top_w[m]) <= 0:
            continue
        k = int(ex[m])
        orig = int(examples[k]["rows"][int(glob[m]) - starts[k]])
        ident = np.asarray(identity[examples[k]["table"]], dtype=np.float64)
        top_id[m] = ident[orig]
        other = np.where(np.isfinite(ident[1:]), ident[1:], -1.0)
        top_near[m] = orig >= 1 and other.size > 0 and int(np.argmax(other)) + 1 == orig
    out = dict(table=tables, bits=bits, none=w[:, -1].numpy(), entropy=ent, level=level, top=top_w.numpy(), top_identity=top_id,
               top_is_nearest=top_near)
    if model.arm == "s1":
        out["none_given_truth"] = r["given_truth"][:, -1].double().numpy()
    return out


def read_arm(name, fit, conditions, per_group, max_rows, out_dir, log=print):
    cfg, model, ck = load_arm(fit)
    head = dict(name=name, fit=fit, arm=model.arm, variant=cfg.get("name"), weights="ema@final", step=ck["step"], tables_seen=ck["tables_seen"],
                version=VERSION, source_version=SR.SOURCE_VERSION, conditions={})
    for condition, layout in conditions.items():
        t = time.time()
        held, data, meta = held_data(cfg, condition, layout)
        ids = E.subset(held, per_group)
        examples, digest = examples_of(data, ids, condition, layout, getattr(model, "ext", None) is not None)
        parts, pending, rows = [], [], 0
        for e in examples:
            if pending and rows + e["grid"].shape[0] > max_rows:
                parts.append(read_batch(model, pending, meta["identity"]))
                pending, rows = [], 0
            pending.append(e)
            rows += e["grid"].shape[0]
        if pending:
            parts.append(read_batch(model, pending, meta["identity"]))
        cells = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(os.path.join(out_dir, "%s.%s.npz" % (name, condition.replace("+", "__"))), **cells)
        head["conditions"][condition] = dict(layout=layout, tables=len(ids), cells=int(len(cells["bits"])), items_sha256=digest,
                                             mean_bits=float(cells["bits"].mean()), seconds=round(time.time() - t, 1))
        log("%s %s: %d tables, %d cells, %.3f bits, %.0f s" % (name, condition, len(ids), len(cells["bits"]), cells["bits"].mean(), time.time() - t))
    with open(os.path.join(out_dir, "%s.json" % name), "w") as f:
        json.dump(head, f, indent=1, sort_keys=True)
    return head


def main(conf_path):
    with open(conf_path) as f:
        conf = json.load(f)
    torch.set_num_threads(int(conf.get("threads", 4)))
    os.makedirs(conf["out"], exist_ok=True)
    done = {}
    for a in conf["arms"]:
        done[a["name"]] = read_arm(a["name"], a["fit"], dict(conf["conditions"]), int(conf.get("per_group", 50)), int(conf.get("max_rows", 32)),
                                   conf["out"], log=lambda s: print(s, flush=True))
    with open(os.path.join(conf["out"], "done.json"), "w") as f:
        json.dump(dict(arms=done, torch=torch.__version__, gpu_hidden=not torch.cuda.is_available()), f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main(sys.argv[1])
