"""Training of one arm of the relational line with its held-out evaluation on the way (today's slice: package W6
with the minimal trainer of W4).

Protocol: AdamW (betas 0.9 and 0.98, weight decay 0.01 on the weight matrices), gradient clipping 1.0, per_step
tables per step (accumulated over micro-batches of `micro` tables when they do not fit at once), trunk in bfloat16 and
output layer in float32 on a GPU, no compilation, every arm from scratch. Learning rate: a linear warm-up over the
first warmup_steps steps (1,000), then constant to the end. The end is the wall-clock limit wall_s (evaluations
included; the time of the final evaluation is reserved) or, when given, budget_cells cells seen, whichever comes
first; the run then ends in the normal way. Evaluation reads the moving average of the weights, updated every step
with the coefficient min(0.999, (1 + t) / (10 + t)), t the step (ema: false reads the raw weights instead).

Held-out evaluation (seeddrift/train/rel/evalh: the plan of every condition, the strata, the rows): at the start, at
every node of tables seen (eval_first, then doubling: 1,000, 2,000, 4,000 ...) on the first eval_per_group tables of
every group, and at the end on all held-out tables (final_per_group: 0). Conditions: the five of the evaluator; an arm
trained without relatives only the two without relatives. Result rows: weights "ema@tables=<n>" at a node and
"ema@final" at the end; item_half "subset" at a node and "all" at the end; items_sha256 the held-out table set's
manifest (the digest of the cells scored under every condition is in evals.jsonl and summary.json).

Files in cfg["out"]: preflight.json; steps.csv (every step: tables, cells and target cells seen, loss in bits per
target cell, learning rate, cells per second, memory, gradient norm, wall time, the loss of every mask scheme);
evals.jsonl (every evaluation); results.tsv; summary.json; heldout_final.npz (every scored cell of the final
evaluation); final.pt (the raw and the averaged weights at the end) and best.pt (the averaged weights at the node
with the lowest bits under the arm's main condition). No checkpoint in probe mode, which takes only tables that are
not real; every other run takes only real tables. This module never imports a generator of synthetic tables or rulers.
"""
from __future__ import annotations

import copy
import json
import math
import os
import time

import numpy as np
import torch

from seeddrift.train.rel import api as A
from seeddrift.train.rel import registry
from seeddrift.train.rel.data import sampler as S
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.models.common import layers as C
from seeddrift.train.rel.trainer import heldout as E

TRAINER_VERSION = "2"
MB = float(1 << 20)
DEFAULTS = dict(objective="mlm", masks=["scattered", "span", "whole_gene"], mask_weights=None, rows=[4, 16], relatives=True, per_step=4, micro=4,
                warmup_steps=1000, weight_decay=0.01, betas=[0.9, 0.98], clip=1.0, budget_cells=None, ema=True, eval_first=1000,
                eval_per_group=50, final_per_group=0, eval_max_rows=64, final_best=None, preflight_steps=200, preflight_warmup=50, preflight_lr_scale=0.3,
                preflight_guard="stop", alone_share=None,
                probe=False, device="auto", threads=2, gpu_mb=None, max_steps=None, author="", group_names=None, train_split=0, heldout_split=1,
                run="", commit="", layout=A.LAYOUT_STORED, columns=A.ROW_LEN)
NEEDED = ("name", "model", "size", "lr", "seed", "wall_s", "train_tables", "train_root", "heldout_tables", "heldout_root", "out")


def settings(cfg):
    out = dict(DEFAULTS)
    out.update(cfg)
    missing = [k for k in NEEDED if out.get(k) is None]
    if missing:
        raise SystemExit("the trainer's configuration lacks %s" % ", ".join(missing))
    if int(out.get("api", T.API_VERSION)) != T.API_VERSION:
        raise SystemExit("variant %s is written for contract version %s; this trainer has version %s" % (out["name"], out.get("api"), T.API_VERSION))
    if out["layout"] != A.LAYOUT_STORED or int(out["columns"]) != A.ROW_LEN:
        raise SystemExit("variant %s: this trainer renders the stored placement on %d columns only" % (out["name"], A.ROW_LEN))
    return out


def write_json(path, obj):
    with open(path + ".part", "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
    os.replace(path + ".part", path)


def append_json(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def versions():
    return "api=%s;common=%s;trainer=%s;reader=%s;sampler=%s;render=%s;evalh=%s/%s;eval=%s" % (
        T.API_VERSION, C.COMMON_VERSION, TRAINER_VERSION, T.READER_VERSION, S.SAMPLER_VERSION, T.RENDER_VERSION, cond.VERSION, SC.VERSION, E.EVAL_VERSION)


class Run:
    """What a preflight and a training share: the tables, the model and its moving average, the optimiser."""

    def __init__(self, cfg):
        self.cfg = cfg = settings(cfg)
        torch.set_num_threads(int(cfg["threads"]))
        torch.manual_seed(int(cfg["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device("cuda" if cfg["device"] != "cpu" and torch.cuda.is_available() else "cpu")
        if cfg["device"] == "cuda" and self.device.type != "cuda":
            raise SystemExit("the configuration asks for a GPU and there is none")
        if int(cfg["train_split"]) != 0 or int(cfg["heldout_split"]) != 1:
            raise SystemExit("the training tables are split 0 and the held-out tables split 1 (given %s and %s)" % (cfg["train_split"], cfg["heldout_split"]))
        self.train = T.open_split(cfg["train_root"], cfg["train_tables"], 0)
        self.held = T.open_split(cfg["heldout_root"], cfg["heldout_tables"], 1)
        for ts, want in ((self.train, 0), (self.held, 1)):
            got = np.unique(np.asarray(ts.load(("split",))["split"]))
            if got.tolist() != [want]:
                raise SystemExit("table set %s: the tables read as split %d carry splits %s" % (ts.name, want, got.tolist()))
        for ts in (self.train, self.held):
            if bool(cfg["probe"]) == (ts.kind == A.TABLE_REAL):
                raise SystemExit("table set %s is of kind %r: probe mode takes only tables that are not real, every other run only real tables"
                                 % (ts.name, ts.kind))
        self.sampler = S.Sampler(self.train, cfg["seed"], masks=cfg["masks"], weights=cfg["mask_weights"], rows=cfg["rows"],
                                 relatives=cfg["relatives"], per_step=cfg["per_step"], alone_share=cfg["alone_share"])
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
        self.weights = list(self.model.parameters())
        self.ema = copy.deepcopy(self.model).requires_grad_(False) if cfg["ema"] else None
        self.ema_weights = list(self.ema.parameters()) if self.ema is not None else []
        self.micro = max(1, min(int(cfg["micro"]), int(cfg["per_step"])))
        self.conditions = E.conditions_of(cfg["relatives"])
        self.main = self.conditions[0]
        self.group_names = {int(k): str(v) for k, v in (cfg["group_names"] or {}).items()}
        self.n_groups = int(self.held.group.max()) + 1
        self.rows = []                                                  # every result row written so far
        self._held = None

    def held_data(self):
        """The arrays of every held-out table (read once), and its metadata for the strata."""
        if self._held is None:
            data = self.held.load(E.PLAN_FIELDS)
            meta = self.held.load(("identity",))
            self._held = (data, np.asarray(meta["identity"]), E.other_rows(data["kind"]))
        return self._held

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
        if not math.isfinite(norm):                                    # never an update from a gradient that is not finite
            self.opt.zero_grad(set_to_none=True)
            raise FloatingPointError("the gradient norm is %s" % norm)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        return total / n_targets / math.log(2.0), norm, (scheme_bits / math.log(2.0)).tolist(), scheme_cells.tolist()

    @torch.no_grad()
    def update_ema(self, t):
        """The moving average after step t: coefficient min(0.999, (1 + t) / (10 + t)); no autograd graph."""
        if self.ema is not None:
            d = min(0.999, (1.0 + t) / (10.0 + t))
            torch._foreach_mul_(self.ema_weights, d)
            torch._foreach_add_(self.ema_weights, self.weights, alpha=1.0 - d)

    def reader(self):
        return self.ema if self.ema is not None else self.model

    def fixed_columns(self, weights, item_half, cells_seen, cells_per_s, status, tables_seen=0, wall_s=0.0):
        """The columns of a result row that are the same for every row of one evaluation; budget: the point of the
        weights evaluated, tables seen and wall seconds since the start of the training."""
        cfg = self.cfg
        budget = "tables=%d;wall_s=%d" % (int(tables_seen), int(round(wall_s)))
        return dict(run=cfg["run"], commit=cfg["commit"], variant=cfg["name"], author=cfg["author"], module="models." + self.info["module"],
                    params=self.params, seed=int(cfg["seed"]), table_set=self.held.name, table_kind=self.held.kind, items_sha256=self.held.sha256,
                    item_half=item_half, layout=A.LAYOUT_STORED, budget=budget, weights=weights, cells_seen=int(cells_seen),
                    cells_per_s=round(cells_per_s, 1), peak_mb=round(self.peak_mb()[1]), tier=getattr(SC, "TIER", SC.ALL), passes=1, status=status,
                    versions=versions())

    def evaluate(self, ids, fixed, path):
        """The held-out tables ids under every condition of this arm, read by the moving average. The rows go to
        self.rows and the whole table to path after every condition. Returns {condition: bits over all groups and
        strata, None when nothing was scored}, {condition: (bits, cell_table)} and {condition: items digest}."""
        data, identity, n_other = self.held_data()
        part = {f: v[ids] for f, v in data.items()}
        strata = SC.strata_of(identity[ids], n_other[ids])
        groups = self.held.group[ids]
        main, cells, digests = {}, {}, {}
        for c in self.conditions:
            bits, cell_table, digest = E.score(self.reader(), part, ids, c, self.device, max_rows=int(self.cfg["eval_max_rows"]))
            agg = SC.aggregate(bits, cell_table, groups, strata, self.n_groups, self.group_names)
            self.rows += [SC.result_row(condition=c, group=r["group"], stratum=r["stratum"], bits=r["bits"], target_cells=r["target_cells"],
                                        n_items=r["n_items"], **fixed) for r in agg]
            SC.write_rows(path + ".part", self.rows)
            os.replace(path + ".part", path)
            main[c] = next((r["bits"] for r in agg if r["group"] == SC.ALL and r["stratum"] == SC.ALL), None)
            cells[c], digests[c] = (bits, cell_table), digest
        return main, cells, digests


def worst_case(examples, rows):
    """The examples with their rows repeated up to `rows` rows each: the largest micro-batch a training can meet."""
    out = []
    for e in examples:
        k = -(-rows // e["grid"].shape[0])
        out.append(dict(e, grid=np.concatenate([e["grid"]] * k)[:rows], truth=np.concatenate([e["truth"]] * k)[:rows]))
    return out


def preflight(cfg):
    """An arm before its training: the model built on the real tables and its parameters counted; the largest
    micro-batch run forward and backward, with fewer tables per micro-batch while it does not fit the GPU or the arm's
    memory allowance (gpu_mb); then, on a fresh model, preflight_steps steps (warm-up over preflight_warmup steps to
    the peak learning rate): when the loss is not finite or its mean over the last 20 steps is not below that of the
    first 5, the preflight fails (preflight_guard "stop", the default: an implementation problem to look at, the
    learning rate is never lowered on its own) or, with preflight_guard "reduce", the arm trains at preflight_lr_scale
    times its peak, which is recorded; 72 held-out tables scored under every condition (evaluation batches at their
    largest). Writes preflight.json in every case."""
    cfg = settings(cfg)
    os.makedirs(cfg["out"], exist_ok=True)
    out = dict(name=cfg["name"], ok=False, torch=torch.__version__, trainer=TRAINER_VERSION, lr_peak=float(cfg["lr"]))
    path = os.path.join(cfg["out"], "preflight.json")
    try:
        run = Run(cfg)
        out.update(params=run.params, device=run.device.type, flops_per_cell=run.info["flops_per_cell"], train_tables=len(run.train),
                   heldout_tables=len(run.held), table_rows=run.train.rows, conditions=run.conditions,
                   gpu=torch.cuda.get_device_name(0) if run.device.type == "cuda" else "")
        rows_max = min(int(cfg["rows"][1]), run.train.rows) if cfg["relatives"] else 1
        examples = worst_case(run.sampler.draw(0), rows_max)
        allowance = float(cfg["gpu_mb"]) if cfg["gpu_mb"] else None
        tried, micro = [], run.micro
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
        del run, examples
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run = Run(dict(cfg, micro=micro))                              # the guard on a fresh model
        peak, warm, losses = float(cfg["lr"]), max(1, int(cfg["preflight_warmup"])), []
        t0, cells, timed = None, 0, 0
        try:
            for s in range(1, int(cfg["preflight_steps"]) + 1):
                if s == 6:
                    t0 = time.time()                                    # the first steps warm the kernels up
                ex = run.sampler.draw(s - 1)
                bits, _, _, _ = run.step(ex, peak * min(1.0, s / warm))
                run.update_ema(s)
                losses.append(bits)
                if t0 is not None:
                    cells += sum(e["grid"].size for e in ex)
                    timed += 1
        except FloatingPointError:
            losses.append(float("nan"))
        start, end = float(np.mean(losses[:5])), float(np.mean(losses[-20:]))
        bad = not all(math.isfinite(x) for x in losses) or not end < start
        if cfg["preflight_guard"] not in ("stop", "reduce"):
            raise SystemExit("preflight_guard is stop or reduce, not %r" % cfg["preflight_guard"])
        lr = peak * float(cfg["preflight_lr_scale"]) if bad and cfg["preflight_guard"] == "reduce" else peak
        took = time.time() - t0 if t0 is not None else 0.0
        alloc, reserved = run.peak_mb()
        out.update(guard=dict(steps=len(losses), warmup=warm, start_bits=round(start, 4), end_bits=round(end, 4) if math.isfinite(end) else None,
                              failed=bad, lr=lr), lr=lr, lr_reduced=bad, cells_per_s=round(cells / took, 1) if took > 0 else None,
                   steps_per_s=round(timed / took, 3) if took > 0 else None, peak_mb=max(round(alloc), max(t.get("peak_mb", 0) for t in tried)),
                   reserved_mb=max(round(reserved), max(t.get("reserved_mb", 0) for t in tried)))
        if bad and cfg["preflight_guard"] == "stop":
            raise SystemExit("the preflight's %d steps did not train: the loss went from %.4f to %s bits (finite: %s); stopped, the learning rate is "
                             "not lowered on its own" % (len(losses), start, "%.4f" % end if math.isfinite(end) else end, all(math.isfinite(x) for x in losses)))
        data, _, _ = run.held_data()
        ids = E.subset(run.held, 6)[:72]                              # 72 tables: an evaluation batch at its largest (64 rows)
        t0 = time.time()
        part = {f: v[ids] for f, v in data.items()}
        scored = {c: E.score(run.reader(), part, ids, c, run.device, max_rows=int(cfg["eval_max_rows"])) for c in run.conditions}
        out.update(eval_s_per_table=round((time.time() - t0) / max(1, len(ids)), 3),
                   eval_bits={c: round(float(b.mean()), 4) if len(b) else None for c, (b, _, _) in scored.items()})
        if not all(v is None or math.isfinite(v) for v in out["eval_bits"].values()):
            raise SystemExit("the held-out scores are not finite: %s" % out["eval_bits"])
        out["row_key_mask"] = dict(C.ROW_MASK)
        out["ok"] = True
    except (Exception, SystemExit) as err:
        out["error"] = "%s: %s" % (type(err).__name__, str(err)[:2000])
        write_json(path, out)
        raise
    write_json(path, out)
    return out


def fit(cfg):
    began = time.time()
    run = Run(cfg)
    cfg = run.cfg
    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    for name in ("results.tsv", "steps.csv", "summary.json", "evals.jsonl"):
        if os.path.exists(os.path.join(out_dir, name)):
            raise SystemExit("%s exists in %s: a training starts in an empty folder" % (name, out_dir))
    results, evals_path, summary = (os.path.join(out_dir, n) for n in ("results.tsv", "evals.jsonl", "summary.json"))
    sub = E.subset(run.held, int(cfg["eval_per_group"]))
    final_ids = E.subset(run.held, int(cfg["final_per_group"]))
    reads = "ema" if run.ema is not None else "raw"
    state = dict(name=cfg["name"], status="running", params=run.params, device=run.device.type, micro=run.micro, conditions=run.conditions,
                 main_condition=run.main, versions=versions(), lr_peak=float(cfg["lr"]), warmup_steps=int(cfg["warmup_steps"]), weights_read=reads,
                 eval_subset=len(sub), eval_final=len(final_ids), evals=[], best=None)
    k = dict(step=0, tables=0, cells=0, targets=0, train_s=0.0, eval_s=0.0, lr=0.0)

    def save(name, ema_only):
        if cfg["probe"]:
            return
        body = dict(name=cfg["name"], size=cfg["size"], module=cfg["model"], step=k["step"], tables_seen=k["tables"], cells_seen=k["cells"],
                    commit=cfg["commit"], ema=run.reader().state_dict())
        if not ema_only:
            body["model"] = run.model.state_dict()
        p = os.path.join(out_dir, name)
        torch.save(body, p + ".part")
        os.replace(p + ".part", p)

    def node():
        t = time.time()
        rate = k["cells"] / k["train_s"] if k["train_s"] > 0 else 0.0
        fixed = run.fixed_columns("%s@tables=%d" % (reads, k["tables"]), "subset", k["cells"], rate, "running", k["tables"], time.time() - began)
        main, _, digests = run.evaluate(sub, fixed, results)
        took = time.time() - t
        k["eval_s"] += took
        rec = dict(tables_seen=k["tables"], step=k["step"], cells_seen=k["cells"], target_cells=k["targets"], lr=k["lr"], wall_s=round(time.time() - began, 1),
                   eval_s=round(took, 1), tables_scored=len(sub), bits=main, items=digests)
        append_json(evals_path, rec)
        state["evals"].append({x: rec[x] for x in rec if x != "items"})
        m = main.get(run.main)
        if m is not None and k["step"] and (state["best"] is None or m < state["best"]["bits"]):
            state["best"] = dict(step=k["step"], tables_seen=k["tables"], cells_seen=k["cells"], bits=m, condition=run.main, wall_s=rec["wall_s"])
            save("best.pt", ema_only=True)
        write_json(summary, state)
        return took

    eval0 = node()                                                   # the untrained model: the start of every curve
    # final_best (default: the arms trained without relatives): the weights of the best node evaluated on the final
    # tables too, so that an arm's best can be set against another arm's end on the same tables
    best_too = (not cfg["relatives"]) if cfg["final_best"] is None else bool(cfg["final_best"])
    reserve = eval0 * len(final_ids) / max(1, len(sub)) * 1.3 * (2 if best_too else 1) + 60.0
    deadline = began + float(cfg["wall_s"]) - reserve
    if deadline <= time.time():
        raise SystemExit("the wall limit of %s s leaves no time to train (the final evaluation needs about %.0f s)" % (cfg["wall_s"], reserve))
    state.update(final_eval_reserve_s=round(reserve, 1), deadline_s=round(deadline - began, 1))
    budget = float(cfg["budget_cells"]) if cfg["budget_cells"] else None
    peak, warm = float(cfg["lr"]), max(1, int(cfg["warmup_steps"]))
    next_node = int(cfg["eval_first"])
    names = run.sampler.masks
    log = open(os.path.join(out_dir, "steps.csv"), "w")
    log.write("step,tables_seen,cells_seen,target_cells,loss_bits,lr,cells_per_s,peak_mb,grad_norm,wall_s," + ",".join("bits_" + m for m in names) + "\n")
    ended_by, last_flush = None, time.time()
    try:
        while True:
            now = time.time()
            if now >= deadline:
                ended_by = "wall"
            elif budget and k["cells"] >= budget:
                ended_by = "budget"
            elif cfg["max_steps"] and k["step"] >= int(cfg["max_steps"]):
                ended_by = "max_steps"
            if ended_by:
                break
            k["lr"] = lr = peak * min(1.0, (k["step"] + 1) / warm)
            examples = run.sampler.draw(k["step"])
            bits, norm, sb, sc = run.step(examples, lr)
            k["step"] += 1
            run.update_ema(k["step"])
            took = time.time() - now
            cells = sum(e["grid"].size for e in examples)
            k["tables"] += len(examples)
            k["cells"] += cells
            k["targets"] += int(sum(sc))
            k["train_s"] += took
            log.write("%d,%d,%d,%d,%.5f,%.4g,%.0f,%.0f,%.4g,%.1f,%s\n" % (k["step"], k["tables"], k["cells"], k["targets"], bits, lr, cells / took,
                                                                       run.peak_mb()[0], norm, time.time() - began,
                                                                       ",".join("%.5f" % (b / c) if c else "" for b, c in zip(sb, sc))))
            if time.time() - last_flush > 30:
                log.flush()
                last_flush = time.time()
            if k["tables"] >= next_node:
                node()
                while next_node <= k["tables"]:
                    next_node *= 2
    except FloatingPointError as err:
        log.close()
        state.update(status="diverged", error=str(err), steps=k["step"], tables_seen=k["tables"], cells_seen=k["cells"])
        write_json(summary, state)
        raise SystemExit("arm %s diverged at step %d: %s" % (cfg["name"], k["step"], err))
    log.close()
    rate = k["cells"] / k["train_s"] if k["train_s"] > 0 else 0.0
    alloc, reserved = run.peak_mb()
    state.update(status="evaluating", ended_by=ended_by, steps=k["step"], tables_seen=k["tables"], cells_seen=k["cells"], target_cells=k["targets"],
                 cells_per_s=round(rate, 1), tables_per_s=round(k["tables"] / k["train_s"], 3) if k["train_s"] > 0 else None, train_s=round(k["train_s"], 1),
                 eval_s=round(k["eval_s"], 1), eval_overhead=round(k["eval_s"] / k["train_s"], 4) if k["train_s"] > 0 else None,
                 peak_mb=round(alloc), reserved_mb=round(reserved), lr_last=k["lr"])
    save("final.pt", ema_only=False)
    write_json(summary, state)
    t = time.time()
    fixed = run.fixed_columns("%s@final" % reads, "all" if len(final_ids) == len(run.held) else "subset", k["cells"], rate, "ok", k["tables"],
                              time.time() - began)
    main, cells, digests = run.evaluate(final_ids, fixed, results)
    np.savez(os.path.join(out_dir, "heldout_final.npz"),
             **{"%s_%s" % (x, c.replace("+", "__")): v for c, (b, tb) in cells.items() for x, v in (("bits", b), ("table", final_ids[tb]))})
    state.update(final=main, final_items=digests, final_tables=len(final_ids), final_eval_s=round(time.time() - t, 1))
    best = state["best"]
    if best_too and best is not None and best["tables_seen"] < k["tables"] and not cfg["probe"]:
        t = time.time()
        reader = run.reader()
        keep = {n: v.detach().clone() for n, v in reader.state_dict().items()}
        reader.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=run.device, weights_only=True)["ema"])
        fixed = run.fixed_columns("%s@best" % reads, "all" if len(final_ids) == len(run.held) else "subset", best["cells_seen"], rate, "ok",
                                  best["tables_seen"], best["wall_s"])
        main_b, _, digests_b = run.evaluate(final_ids, fixed, results)
        reader.load_state_dict(keep)
        state["final_best"] = dict(tables_seen=best["tables_seen"], step=best["step"], bits=main_b, items=digests_b, eval_s=round(time.time() - t, 1))
    state.update(status="ok", wall_s=round(time.time() - began, 1), row_key_mask=dict(C.ROW_MASK))
    append_json(evals_path, dict(final=True, tables_seen=k["tables"], step=k["step"], cells_seen=k["cells"], target_cells=k["targets"], lr=k["lr"],
                                 wall_s=state["wall_s"], eval_s=state["final_eval_s"], tables_scored=len(final_ids), bits=main, items=digests))
    write_json(summary, state)
    return state
