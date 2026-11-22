"""Training of one variant of the relational line, with its held-out evaluation on the way (today's slice).

Protocol: AdamW (betas 0.9 and 0.98, weight decay 0.01 on the weight matrices), warm-up over the first 5% of the
progress, then linear decay to zero at its end, gradient clipping 1.0, per_step tables per step (accumulated over
micro-batches of `micro` tables when they do not fit at once), trunk in bfloat16 and output layer in float32 on a GPU,
no compilation. Every arm is trained from scratch.

Progress is the larger of (cells seen / budget_cells) and (training time / the time the wall limit leaves), so the
learning rate reaches zero at the end of the budget or at the wall limit, whichever comes first, and the run ends
there in the normal way: the final checkpoint, then the evaluation on every held-out table. Without budget_cells the
budget is the wall limit. Which of the two ended the run is in summary.json (ended_by).

The held-out tables are scored every eval_every_cells cells seen on a subset (eval_per_group tables of every group)
and all of them at the end. Files in cfg["out"]: steps.csv (every step: loss in bits per target cell, learning rate,
cells per second, memory peak), results.tsv (rows of api.RESULT_ROW), preflight.json, summary.json, final.pt and
best.pt (the two checkpoints kept; none in probe mode).

Probe mode (cfg["probe"]) takes only tables that are not real and keeps no checkpoint; otherwise only real tables are
accepted. This module never imports a generator of synthetic tables or of rulers.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from seeddrift.train.rel import api as A
from seeddrift.train.rel import registry
from seeddrift.train.rel.data import heldout_plan as H
from seeddrift.train.rel.data import sampler as S
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.models.common import layers as C
from seeddrift.train.rel.trainer import heldout as E
from seeddrift.train.rel.trainer import rows as W

TRAINER_VERSION = "1"
MB = float(1 << 20)
DEFAULTS = dict(objective="mlm", masks=["scattered", "span", "whole_gene"], mask_weights=None, rows=[4, 16], relatives=True, per_step=4, micro=4,
                warmup=0.05, weight_decay=0.01, betas=[0.9, 0.98], clip=1.0, budget_cells=None, eval_every_cells=None, eval_per_group=15,
                eval_max_rows=64, eval_points=14, final_per_group=0, probe=False, device="auto", threads=2, gpu_mb=None, max_steps=None,
                author="", train_split=None, heldout_split=None, run="", commit="", layout=A.LAYOUT_STORED, columns=A.ROW_LEN)


def settings(cfg):
    out = dict(DEFAULTS)
    out.update(cfg)
    for k in ("name", "model", "size", "lr", "seed", "wall_s", "train_tables", "train_root", "heldout_tables", "heldout_root", "out"):
        if out.get(k) is None:
            raise SystemExit("the trainer's configuration lacks %s" % k)
    if int(out.get("api", T.API_VERSION)) != T.API_VERSION:
        raise SystemExit("variant %s is written for contract version %s; this trainer has version %s" % (out["name"], out.get("api"), T.API_VERSION))
    if out["layout"] != A.LAYOUT_STORED or int(out["columns"]) != A.ROW_LEN:
        raise SystemExit("variant %s: this trainer renders the stored placement on %d columns only" % (out["name"], A.ROW_LEN))
    return out


def write_json(path, obj):
    with open(path + ".part", "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(path + ".part", path)


def versions():
    return "api=%s;common=%s;trainer=%s;reader=%s;sampler=%s;render=%s;plan=%s;eval=%s" % (
        T.API_VERSION, C.COMMON_VERSION, TRAINER_VERSION, T.READER_VERSION, S.SAMPLER_VERSION, T.RENDER_VERSION, H.PLAN_VERSION, E.EVAL_VERSION + "/" + W.ROWS_VERSION)


class Run:
    """Everything a preflight and a training share: tables, model, optimiser, the step."""

    def __init__(self, cfg):
        self.cfg = cfg = settings(cfg)
        torch.set_num_threads(int(cfg["threads"]))
        torch.manual_seed(int(cfg["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device("cuda" if cfg["device"] != "cpu" and torch.cuda.is_available() else "cpu")
        if cfg["device"] == "cuda" and self.device.type != "cuda":
            raise SystemExit("the configuration asks for a GPU and there is none")
        self.train = T.open_tableset(cfg["train_tables"], cfg["train_root"], cfg["train_split"])
        self.held = T.open_tableset(cfg["heldout_tables"], cfg["heldout_root"], cfg["heldout_split"])
        for ts in (self.train, self.held):
            if bool(cfg["probe"]) == (ts.kind == A.TABLE_REAL):
                raise SystemExit("table set %s is of kind %r: probe mode takes only tables that are not real, every other run only real tables"
                                 % (ts.name, ts.kind))
        self.sampler = S.Sampler(self.train, cfg["seed"], masks=cfg["masks"], weights=cfg["mask_weights"], rows=cfg["rows"],
                                 relatives=cfg["relatives"], per_step=cfg["per_step"])
        self.module = registry.load("model", cfg["model"])
        self.objective = registry.load("objective", cfg["objective"])
        self.model = self.module.build(cfg["size"]).to(self.device)
        self.info = self.module.info(cfg["size"], rows=int(cfg["rows"][1]) if cfg["relatives"] else 1)
        self.params = sum(p.numel() for p in self.model.parameters())
        if self.params != self.info["params"]:
            raise SystemExit("the model has %d parameters, its module declares %d" % (self.params, self.info["params"]))
        decay = [p for p in self.model.parameters() if p.ndim >= 2]
        rest = [p for p in self.model.parameters() if p.ndim < 2]
        self.opt = torch.optim.AdamW([dict(params=decay, weight_decay=float(cfg["weight_decay"])), dict(params=rest, weight_decay=0.0)],
                                     lr=float(cfg["lr"]), betas=tuple(float(b) for b in cfg["betas"]), fused=self.device.type == "cuda")
        self.micro = max(1, min(int(cfg["micro"]), int(cfg["per_step"])))
        # the conditions this arm is scored under: an arm trained without relatives is read only without them
        self.conditions = [c for c in H.CONDITIONS if cfg["relatives"] or H.NO_REL in c.split("+")]
        self.main = self.conditions[0]
        self.group_names = self.held.group_names

    def peak_mb(self):
        if self.device.type != "cuda":
            return 0.0, 0.0
        return torch.cuda.max_memory_allocated() / MB, torch.cuda.max_memory_reserved() / MB

    def step(self, examples, lr):
        """One update on the examples. Returns the loss in bits per target cell, the gradient norm, and per mask
        scheme the summed bits and the number of target cells."""
        n_targets = sum(int((e["truth"] > 0).sum()) for e in examples)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        k = len(self.sampler.masks)
        scheme_bits = torch.zeros(k, dtype=torch.float64, device=self.device)
        scheme_cells = torch.zeros(k, dtype=torch.float64, device=self.device)
        total = 0.0
        for at in range(0, len(examples), self.micro):
            chunk = examples[at:at + self.micro]
            b = E.to_device(S.collate(chunk), self.device)
            summed, per = self.objective.loss(self.model, b)
            (summed / n_targets).backward()
            scheme = torch.tensor([e["scheme"] for e in chunk], device=self.device)[b["example"]]
            scheme_bits.index_add_(0, scheme, per.detach().double())
            scheme_cells.index_add_(0, scheme, torch.ones_like(per, dtype=torch.float64))
            total += float(summed)
        if not math.isfinite(total):
            raise FloatingPointError("the loss is %s" % total)
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.cfg["clip"])))
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        return total / n_targets / math.log(2.0), norm, (scheme_bits / math.log(2.0)).tolist(), scheme_cells.tolist()

    def fixed_columns(self, weights, item_half, cells_seen, target_cells, cells_per_s, status):
        cfg = self.cfg
        budget = "cells=%s;wall_s=%s" % (cfg["budget_cells"] if cfg["budget_cells"] else "none", int(cfg["wall_s"]))
        return dict(run=cfg["run"], commit=cfg["commit"], variant=cfg["name"], author=cfg["author"], module=self.info["module"], params=self.params,
                    seed=int(cfg["seed"]), table_set=self.held.name, table_kind=self.held.kind, items_sha256=self.held.sha256, item_half=item_half,
                    layout=A.LAYOUT_STORED, budget=budget, weights=weights, cells_seen=int(cells_seen), target_cells=int(target_cells),
                    cells_per_s="%.1f" % cells_per_s, peak_mb="%.0f" % self.peak_mb()[1], status=status, versions=versions())

    def evaluate(self, ids, fixed, path, strata=None, groups=None):
        """Scores the held-out tables ids under every condition of this arm and appends the rows to path, condition
        by condition. Returns {condition: bits of (all groups, all strata)} and the per-table sums."""
        strata = W.strata_of(self.held, ids) if strata is None else strata
        groups = self.held.group[ids] if groups is None else groups
        main, tables = {}, {}
        for cond in self.conditions:
            bits, cells = E.score(self.model, self.held, ids, cond, self.device, max_rows=int(self.cfg["eval_max_rows"]))
            agg = W.aggregate(bits, cells, groups, strata, self.group_names)
            W.write_rows(path, W.result_rows(fixed, cond, agg))
            main[cond] = next((x for g, s, x, _ in agg if g == W.ALL and s == W.ALL), None)       # None: no table has this condition
            tables[cond] = (bits, cells)
        return main, tables


def worst_case(examples, rows):
    """The examples with their rows repeated up to `rows` rows each: the largest micro-batch a training can meet."""
    out = []
    for e in examples:
        k = -(-rows // e["grid"].shape[0])
        out.append(dict(e, grid=np.concatenate([e["grid"]] * k)[:rows], truth=np.concatenate([e["truth"]] * k)[:rows]))
    return out


def preflight(cfg):
    """A variant before its training: the model is built on the real tables, its parameters counted, the largest
    micro-batch run forward and backward (with fewer tables per micro-batch when it does not fit the GPU or the
    variant's memory allowance), a few ordinary steps timed, two held-out tables scored under every condition."""
    cfg = settings(cfg)
    os.makedirs(cfg["out"], exist_ok=True)
    out = dict(name=cfg["name"], ok=False, torch=torch.__version__, trainer=TRAINER_VERSION)
    path = os.path.join(cfg["out"], "preflight.json")
    try:
        run = Run(cfg)
        out.update(params=run.params, device=run.device.type, flops_per_cell=run.info["flops_per_cell"], train_tables=len(run.train),
                   heldout_tables=len(run.held), table_rows=run.train.rows, conditions=run.conditions,
                   gpu=torch.cuda.get_device_name(0) if run.device.type == "cuda" else "")
        rows_max = min(int(cfg["rows"][1]), run.train.rows) if cfg["relatives"] else 1
        examples = worst_case(run.sampler.draw(0), rows_max)
        allowance = float(cfg["gpu_mb"]) if cfg["gpu_mb"] else None
        tried = []
        micro = run.micro
        while True:
            run.micro = micro
            try:
                if run.device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    run.step(examples, 0.0)
                alloc, reserved = run.peak_mb()
                fits = allowance is None or reserved <= allowance
                tried.append(dict(micro=micro, peak_mb=round(alloc), reserved_mb=round(reserved), fits=fits))
            except torch.cuda.OutOfMemoryError:
                fits = False
                tried.append(dict(micro=micro, error="out of memory"))
                run.opt.zero_grad(set_to_none=True)
            if fits or micro == 1:
                break
            micro = max(1, micro // 2)
        out.update(micro=micro, memory=tried)
        if not fits:
            raise SystemExit("one table of %d rows does not fit: %s" % (rows_max, tried[-1]))
        out.update(peak_mb=tried[-1]["peak_mb"], reserved_mb=tried[-1]["reserved_mb"])
        t0, cells, steps = time.time(), 0, 0
        while steps < 3 or (time.time() - t0 < float(cfg.get("preflight_s", 20)) and steps < 200):
            ex = run.sampler.draw(steps)
            bits, _, _, _ = run.step(ex, 0.0)
            cells += sum(e["grid"].size for e in ex)
            steps += 1
        out.update(cells_per_s=round(cells / (time.time() - t0), 1), steps_timed=steps, loss_bits=round(bits, 4))
        ids = W.subset(run.held, 1)[:2]
        t0 = time.time()
        scored = {c: E.score(run.model, run.held, ids, c, run.device, max_rows=int(cfg["eval_max_rows"])) for c in run.conditions}
        out.update(eval_s_per_table=round((time.time() - t0) / max(1, len(ids)), 3),
                   eval_bits={c: round(float(b.sum() / max(1, n.sum())), 4) for c, (b, n) in scored.items()})
        if not all(math.isfinite(v) for v in out["eval_bits"].values()):
            raise SystemExit("the held-out scores are not finite: %s" % out["eval_bits"])
        out["ok"] = True
    except (Exception, SystemExit) as err:
        out["error"] = "%s: %s" % (type(err).__name__, str(err)[:2000])
        write_json(path, out)
        raise
    write_json(path, out)
    return out


def nice(x):
    """x rounded down to 1, 2 or 5 times a power of ten."""
    p = 10.0 ** math.floor(math.log10(max(x, 1.0)))
    return int(max(m for m in (1, 2, 5) if m * p <= max(x, 1.0)) * p)


def fit(cfg):
    began = time.time()
    run = Run(cfg)
    cfg = run.cfg
    os.makedirs(cfg["out"], exist_ok=True)
    results = os.path.join(cfg["out"], "results.tsv")
    for name in ("results.tsv", "steps.csv", "summary.json"):
        if os.path.exists(os.path.join(cfg["out"], name)):
            raise SystemExit("%s exists in %s: a training starts in an empty folder" % (name, cfg["out"]))
    W.write_rows(results, [], header=True)
    sub = W.subset(run.held, int(cfg["eval_per_group"]))
    everything = W.subset(run.held, int(cfg["final_per_group"]))
    sub_strata, sub_groups = W.strata_of(run.held, sub), run.held.group[sub]
    state = dict(name=cfg["name"], status="running", params=run.params, device=run.device.type, micro=run.micro, conditions=run.conditions,
                 main_condition=run.main, versions=versions(), evals=[], best=None)
    summary = os.path.join(cfg["out"], "summary.json")

    def save(name):
        if not cfg["probe"]:
            p = os.path.join(cfg["out"], name)
            torch.save(dict(model=run.model.state_dict(), name=cfg["name"], size=cfg["size"], module=cfg["model"], step=step, cells_seen=cells_seen,
                            commit=cfg["commit"]), p + ".part")
            os.replace(p + ".part", p)

    def evaluate_subset():
        t = time.time()
        rate = cells_seen / train_time if train_time > 0 else 0.0
        fixed = run.fixed_columns("raw", "subset", cells_seen, target_seen, rate, "running")
        main, _ = run.evaluate(sub, fixed, results, strata=sub_strata, groups=sub_groups)
        took = time.time() - t
        state["evals"].append(dict(step=step, cells_seen=cells_seen, bits=main, eval_s=round(took, 1), wall_s=round(time.time() - began, 1)))
        if main[run.main] is not None and (state["best"] is None or main[run.main] < state["best"]["bits"]):
            state["best"] = dict(step=step, cells_seen=cells_seen, bits=main[run.main], condition=run.main)
            if step:
                save("best.pt")
        write_json(summary, state)
        return took

    step, cells_seen, target_seen, train_time = 0, 0, 0, 0.0
    eval0 = evaluate_subset()                                   # the untrained model: the start of every curve
    reserve = eval0 * len(everything) / max(1, len(sub)) * 1.5 + 30.0
    t0 = time.time()
    train_s = float(cfg["wall_s"]) - (t0 - began) - reserve
    if train_s <= 0:
        raise SystemExit("the wall limit of %s s leaves no time to train (the final evaluation needs about %.0f s)" % (cfg["wall_s"], reserve))
    budget = float(cfg["budget_cells"]) if cfg["budget_cells"] else None
    every = int(cfg["eval_every_cells"]) if cfg["eval_every_cells"] else None
    points = int(max(4, min(int(cfg["eval_points"]), 0.08 * float(cfg["wall_s"]) / max(eval0, 1e-3))))
    next_eval = every
    state.update(train_s=round(train_s, 1), final_eval_reserve_s=round(reserve, 1), eval_every_cells=every)
    warm, peak_lr = float(cfg["warmup"]), float(cfg["lr"])
    names = run.sampler.masks
    log = open(os.path.join(cfg["out"], "steps.csv"), "w")
    log.write("step,cells_seen,target_cells,loss_bits,lr,cells_per_s,peak_mb,grad_norm,wall_s," + ",".join("bits_" + m for m in names) + "\n")
    ended_by, last_flush = None, time.time()
    try:
        while True:
            now = time.time()
            p_time = (now - t0) / train_s
            p_cells = cells_seen / budget if budget else 0.0
            p = max(p_time, p_cells)
            if p >= 1.0 or (cfg["max_steps"] and step >= int(cfg["max_steps"])):
                ended_by = "max_steps" if p < 1.0 else ("budget" if p_cells >= p_time else "wall")
                break
            lr = peak_lr * (p / warm if p < warm else (1.0 - p) / (1.0 - warm))
            examples = run.sampler.draw(step)
            bits, norm, sb, sc = run.step(examples, lr)
            took = time.time() - now
            step += 1
            cells = sum(e["grid"].size for e in examples)
            cells_seen += cells
            target_seen += int(sum(sc))
            train_time += took
            log.write("%d,%d,%d,%.5f,%.4g,%.0f,%.0f,%.4g,%.1f,%s\n" % (step, cells_seen, target_seen, bits, lr, cells / took, run.peak_mb()[0], norm,
                                                                 time.time() - began, ",".join("%.5f" % (b / c) if c else "" for b, c in zip(sb, sc))))
            if time.time() - last_flush > 30:
                log.flush()
                last_flush = time.time()
            if every is None and train_time >= min(120.0, 0.02 * train_s):
                # the spacing of the evaluations, from the throughput the training really has
                expected = cells_seen / (time.time() - t0) * train_s
                every = nice(min(expected, budget or expected) / points)
                next_eval = every
                state["eval_every_cells"] = every
            if every is not None and cells_seen >= next_eval:
                evaluate_subset()
                while next_eval <= cells_seen:
                    next_eval += every
    except FloatingPointError as err:
        log.close()
        state.update(status="diverged", error=str(err), steps=step, cells_seen=cells_seen)
        write_json(summary, state)
        raise SystemExit("variant %s diverged at step %d: %s" % (cfg["name"], step, err))
    log.close()
    rate = cells_seen / train_time if train_time > 0 else 0.0
    alloc, reserved = run.peak_mb()
    state.update(status="evaluating", ended_by=ended_by, steps=step, cells_seen=cells_seen, target_cells=target_seen, cells_per_s=round(rate, 1),
                 train_time_s=round(train_time, 1), peak_mb=round(alloc), reserved_mb=round(reserved))
    save("final.pt")
    write_json(summary, state)
    t = time.time()
    fixed = run.fixed_columns("final", "all", cells_seen, target_seen, rate, "ok")
    main, tables = run.evaluate(everything, fixed, results)
    np.savez(os.path.join(cfg["out"], "heldout_final.npz"), table=everything,
             **{"%s_%s" % (k, c.replace("+", "__")): v for c, bc in tables.items() for k, v in zip(("bits", "cells"), bc)})
    state.update(status="ok", final=main, final_eval_s=round(time.time() - t, 1), wall_s=round(time.time() - began, 1))
    write_json(summary, state)
    return state
