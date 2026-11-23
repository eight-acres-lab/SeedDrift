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
evaluation); node_cells.npz (at every node, per held-out table of the nodes' subset and condition: the summed bits
and the number of scored cells, keys t<tables>_<condition>_bits and _cells, "+" written "__"; tables: their ids); final.pt (the raw and the averaged weights at the end) and best.pt (the averaged weights at the node
with the lowest bits under the arm's main condition). No checkpoint in probe mode, which takes only tables that are
not real; every other run takes only real tables. This module never imports a generator of synthetic tables or rulers.
"""
from __future__ import annotations

import copy
import inspect
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
from seeddrift.train.rel.evalh import shards
from seeddrift.train.rel.models.common import layers as C
from seeddrift.train.rel.tables import align as AL
from seeddrift.train.rel.trainer import heldout as E

TRAINER_VERSION = "2"
RULER = getattr(A, "TABLE_RULER", "ruler")      # the kind of the ruler table sets (package W18): evaluation only
MB = float(1 << 20)
DEFAULTS = dict(objective="mlm", masks=["scattered", "span", "whole_gene"], mask_weights=None, rows=[4, 16], relatives=True, per_step=4, micro=4,
                warmup_steps=1000, weight_decay=0.01, betas=[0.9, 0.98], clip=1.0, budget_cells=None, ema=True, eval_first=1000,
                eval_per_group=50, final_per_group=0, eval_max_rows=64, final_best=None, preflight_steps=200, preflight_warmup=50, preflight_lr_scale=0.3,
                preflight_guard="stop", alone_share=None,
                probe=False, device="auto", threads=2, gpu_mb=None, max_steps=None, author="", group_names=None, train_split=0, heldout_split=1,
                run="", commit="", layout=A.LAYOUT_STORED, whole_gene_layout=None, columns=A.ROW_LEN, references_at_nodes=False,
                pair_index=None, eval_online=False, end_controls=[], keep_node_weights=False, nodes_every=None, nodes_every_from=100000,
                data_workers=0)
CONTROLS = ("shuffle", "presence")              # package W3's controls of the renderer, read at the final evaluation
TRAINED_LAYOUTS = (A.LAYOUT_STORED, A.LAYOUT_LEAKFREE)          # the layouts a training renders its tables in
FRAME = S.FRAME                                 # whole_gene_layout: the whole gene laid by W3's strict aligner (align.frame_layout)
ONLINE = "leakfree-online"                      # the layout column of rows rendered leak-free online (not W3's offline rendering)
NODE_WEIGHTS = "ema_tables%d.pt"                # keep_node_weights: the moving average at a node of that many tables seen
# the part of package W3's strict offline rendering a condition of a leak-free arm is scored on (made for its mask)
OFFLINE_PART = {cond.SCATTERED: "scattered", cond.WHOLE_GENE: "whole_gene", cond.WHOLE_GENE + "+" + cond.REPAIRED: "whole_gene",
                getattr(cond, "SUFFIX", "suffix"): "suffix"}
# the conditions a leak-free arm also reads in the stored layout its whole-gene samples were trained in
STORED_TOO = (cond.WHOLE_GENE, cond.WHOLE_GENE + "+" + cond.REPAIRED)
NEEDED = ("name", "model", "size", "lr", "seed", "wall_s", "train_tables", "train_root", "heldout_tables", "heldout_root", "out")


def settings(cfg):
    out = dict(DEFAULTS)
    out.update(cfg)
    missing = [k for k in NEEDED if out.get(k) is None]
    if missing:
        raise SystemExit("the trainer's configuration lacks %s" % ", ".join(missing))
    if int(out.get("api", T.API_VERSION)) != T.API_VERSION:
        raise SystemExit("variant %s is written for contract version %s; this trainer has version %s" % (out["name"], out.get("api"), T.API_VERSION))
    if out["layout"] not in TRAINED_LAYOUTS or (out["whole_gene_layout"] or out["layout"]) not in TRAINED_LAYOUTS + (FRAME,) or int(out["columns"]) != A.ROW_LEN:
        raise SystemExit("variant %s: this trainer renders the layouts %s on %d columns, the whole gene also %s (given %s, whole gene %s)"
                         % (out["name"], TRAINED_LAYOUTS, A.ROW_LEN, FRAME, out["layout"], out["whole_gene_layout"]))
    if out["whole_gene_layout"] == FRAME and out["layout"] == A.LAYOUT_STORED:
        raise SystemExit("variant %s: the whole gene on its frame relative is read leak-free; the variant's layout is leakfree" % out["name"])
    if out["nodes_every"] is not None and (int(out["nodes_every"]) <= 0 or int(out["nodes_every_from"]) <= 0):
        raise SystemExit("nodes_every and nodes_every_from are positive numbers of tables (given %s, %s)" % (out["nodes_every"], out["nodes_every_from"]))
    return out


def node_after(cfg, tables):
    """The next evaluation node after `tables` tables seen: eval_first, then every doubling; with nodes_every also every
    nodes_every tables from nodes_every_from on (the long arms of round 1: every 50,000 after 100,000)."""
    nxt = int(cfg["eval_first"])
    while nxt <= tables:
        nxt *= 2
    if cfg["nodes_every"]:
        every, start = int(cfg["nodes_every"]), int(cfg["nodes_every_from"])
        nxt = min(nxt, start if tables < start else start + ((tables - start) // every + 1) * every)
    return nxt


def _same(x):
    return x


class Feed:
    """The examples of steps start, start + 1, ... in order: drawn here (workers 0), or ahead by `workers` background
    processes of torch's data loader (each draws every workers-th step; a step's examples depend on (seed, step) only,
    so they are the same either way). wait_s: the seconds the trainer waited for them."""

    def __init__(self, sampler, start=0, workers=0):
        self.sampler, self.next_step, self.workers, self.wait_s = sampler, int(start), int(workers), 0.0
        self._it = None
        if self.workers > 0:
            self._it = iter(torch.utils.data.DataLoader(_Steps(sampler, self.next_step, self.workers), batch_size=None, num_workers=self.workers,
                                                        collate_fn=_same, prefetch_factor=4, multiprocessing_context="fork"))

    def get(self, step):
        if int(step) != self.next_step:
            raise RuntimeError("the feed is at step %d, step %d was asked" % (self.next_step, step))
        t = time.time()
        if self._it is None:
            out = self.sampler.draw(self.next_step)
        else:
            got, out = next(self._it)
            if got != self.next_step:
                raise RuntimeError("the data worker sent step %d for step %d" % (got, self.next_step))
        self.wait_s += time.time() - t
        self.next_step += 1
        return out

    def close(self):
        self._it = None                                                # the loader's workers stop with their iterator


class _Steps(torch.utils.data.IterableDataset):
    def __init__(self, sampler, start, workers):
        self.sampler, self.start, self.workers = sampler, int(start), int(workers)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        w = info.id if info is not None else 0
        s = self.start + w
        while True:
            yield s, self.sampler.draw(s)
            s += self.workers


def layouts_of(cfg):
    """{mask scheme: layout}: the variant's layout, the whole gene's own where whole_gene_layout names one."""
    return {m: (cfg["whole_gene_layout"] or cfg["layout"]) if m == "whole_gene" else cfg["layout"] for m in cfg["masks"]}


def size_of(module, cfg):
    """The size the model is built with: the variant's, with ext on where a layout other than stored is rendered."""
    size = cfg["size"]
    if all(x == A.LAYOUT_STORED for x in layouts_of(cfg).values()):
        return size
    base = dict(getattr(module, "SIZES", {})[size]) if isinstance(size, str) else dict(size)
    return dict(base, ext=True)


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

    def __init__(self, cfg, evaluate_only=False):
        """evaluate_only (evaluate_checkpoints alone): the held-out set may be a ruler set (package W18) of the split it
        carries; a training or preflight reads real held-out tables of split 1 only."""
        self.cfg = cfg = settings(cfg)
        torch.set_num_threads(int(cfg["threads"]))
        torch.manual_seed(int(cfg["seed"]))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device("cuda" if cfg["device"] != "cpu" and torch.cuda.is_available() else "cpu")
        if cfg["device"] == "cuda" and self.device.type != "cuda":
            raise SystemExit("the configuration asks for a GPU and there is none")
        held_split = int(cfg["heldout_split"])
        if int(cfg["train_split"]) != 0 or (held_split != 1 and not evaluate_only):
            raise SystemExit("the training tables are split 0 and the held-out tables split 1 (given %s and %s)" % (cfg["train_split"], cfg["heldout_split"]))
        self.train = T.open_split(cfg["train_root"], cfg["train_tables"], 0)
        self.held = T.open_split(cfg["heldout_root"], cfg["heldout_tables"], held_split)
        ruler = evaluate_only and self.held.kind == RULER and not cfg["probe"]
        if self.held.kind == A.TABLE_REAL and held_split != 1:
            raise SystemExit("real held-out tables are split 1, not %d" % held_split)
        for ts, want in ((self.train, 0), (self.held, held_split)):
            got = np.unique(np.asarray(ts.load(("split",))["split"]))
            if got.tolist() != [want]:
                raise SystemExit("table set %s: the tables read as split %d carry splits %s" % (ts.name, want, got.tolist()))
        for ts in (self.train, self.held):
            if ts is self.held and ruler:
                continue                                              # a ruler set, read by an evaluation only
            if bool(cfg["probe"]) == (ts.kind == A.TABLE_REAL):
                raise SystemExit("table set %s is of kind %r: probe mode takes only tables that are not real, every other run only real tables"
                                 % (ts.name, ts.kind))
        self.frame_index = None                                         # the pair index of W12's uncapped search
        self.frame_strict = layouts_of(cfg).get("whole_gene") == FRAME   # the whole gene laid by W3's strict aligner instead
        self._frames = {}                                               # its layouts of the held-out tables, (table, condition) -> arrays
        if layouts_of(cfg).get("whole_gene", A.LAYOUT_STORED) not in (A.LAYOUT_STORED, FRAME):
            if cfg["pair_index"]:
                from seeddrift.train.rel.tables import v1_hits
                ix = cfg["pair_index"]                                  # a folder of W12's run; an index object in the tests
                self.frame_index = v1_hits.Index(ix) if isinstance(ix, str) else ix
            elif not cfg["probe"]:
                raise SystemExit("variant %s lays the whole gene on its frame relative online: the pair index of the search (pair_index)" % cfg["name"])
        if set(cfg["end_controls"] or []) - set(CONTROLS):
            raise SystemExit("end_controls are among %s, not %s" % (CONTROLS, cfg["end_controls"]))
        self.sampler = S.Sampler(self.train, cfg["seed"], masks=cfg["masks"], weights=cfg["mask_weights"], rows=cfg["rows"],
                                 relatives=cfg["relatives"], per_step=cfg["per_step"], alone_share=cfg["alone_share"], layouts=layouts_of(cfg),
                                 frame_index=self.frame_index)
        self.module = registry.load("model", cfg["model"])
        self.objective = registry.load("objective", cfg["objective"])
        self.size = size_of(self.module, cfg)
        self.model = self.module.build(self.size).to(self.device)
        self.info = self.module.info(self.size, rows=int(cfg["rows"][1]) if cfg["relatives"] else 1)
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
        self.conditions = E.conditions_of(cfg["relatives"], cfg.get("extra_conditions") or (), cfg.get("only_conditions"))
        if not self.conditions:
            raise SystemExit("arm %s has no condition to be scored under (only_conditions %s)" % (cfg["name"], cfg.get("only_conditions")))
        self.main = self.conditions[0]
        self.group_names = {int(k): str(v) for k, v in (cfg["group_names"] or {}).items()}
        self.n_groups = int(self.held.group.max()) + 1
        self.rows = []                                                  # every result row written so far
        self._held = None
        self._parts = {}
        self.offline = None                                             # package W3's strict offline rendering of the held-out set
        if cfg.get("heldout_offline_tables"):
            doc = T.read_manifest(cfg["heldout_offline_tables"])
            self.offline = dict(doc=doc, parts=shards.parts(cfg["heldout_offline_root"], doc))
            for name, ts in self.offline["parts"].items():
                if int(ts.tables) != len(self.held):
                    raise SystemExit("offline part %s holds %d tables, the held-out set %d" % (name, ts.tables, len(self.held)))

    def held_data(self):
        """The arrays of every held-out table (read once), and its metadata for the strata."""
        if self._held is None:
            data = self.held.load(E.PLAN_FIELDS)
            meta = self.held.load(("identity",))
            self._held = (data, np.asarray(meta["identity"]), E.other_rows(data["kind"]))
        return self._held

    def eval_source(self, condition, at_node=False):
        """(layout, arrays of every held-out table, label) a condition is scored in, label the result rows' layout
        column: an arm trained in the stored layout, and the conditions without relatives, read the stored tables in
        the stored layout; an arm trained leak-free reads the other conditions in the leak-free layout, from the part of
        the strict offline rendering made for that condition's mask (OFFLINE_PART, label leakfree) when the run names
        one, except at the nodes with eval_online; else from the stored tables rendered online (label leakfree-online:
        the anchors of scattered and span the stored pairs; the whole gene laid as the arm was trained, by W3's strict
        aligner with whole_gene_layout frame)."""
        data = self.held_data()[0]
        if self.cfg["layout"] == A.LAYOUT_STORED or cond.NO_RELATIVES in condition:
            return A.LAYOUT_STORED, data, A.LAYOUT_STORED
        part = OFFLINE_PART.get(condition)
        if self.offline is None or part is None or (at_node and self.cfg["eval_online"]):
            if not self.cfg["probe"] and part is not None and not self.cfg["eval_online"]:
                raise SystemExit("a leak-free arm is scored under %s on the strict offline rendering (heldout_offline_tables), or "
                                 "online where the run says so (eval_online)" % condition)
            return A.LAYOUT_LEAKFREE, data, ONLINE
        if part not in self._parts:
            ts = self.offline["parts"][part]
            self._parts[part] = ts.load(0, ts.tables, E.PLAN_FIELDS, verify=False)
        return A.LAYOUT_LEAKFREE, self._parts[part], A.LAYOUT_LEAKFREE

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
        budget = "tables=%d;wall_s=%d;micro=%d" % (int(tables_seen), int(round(wall_s)), self.micro)   # the tables per micro-batch trained with
        return dict(run=cfg["run"], commit=cfg["commit"], variant=cfg["name"], author=cfg["author"], module="models." + self.info["module"],
                    params=self.params, seed=int(cfg["seed"]), table_set=self.held.name, table_kind=self.held.kind, items_sha256=self.held.sha256,
                    item_half=item_half, layout=A.LAYOUT_STORED, budget=budget, weights=weights, cells_seen=int(cells_seen),
                    cells_per_s=round(cells_per_s, 1), peak_mb=round(self.peak_mb()[1]), tier=getattr(SC, "TIER", SC.ALL), passes=1, status=status,
                    versions=versions())

    def eval_plan(self, at_node=False):
        """[(key, condition, layout, arrays, control, label)]: every condition of the arm in the layout eval_source
        gives it (label: the rows' layout column); an arm trained leak-free with the whole gene stored also reads the
        whole gene (and its re-pairing) in the stored layout it was trained in (key "<condition>@stored"; the programme
        lead's ruling for round 1b: that reading is a reference only). The references (those, and the extra conditions
        such as suffix) are read at the final evaluation only, not at the nodes, unless references_at_nodes (the ruling
        of 2026-11-23). The controls of end_controls: at the final evaluation only, under scattered and whole_gene (key
        "<condition>@<control>", label "<label>+<control>")."""
        refs = not at_node or bool(self.cfg["references_at_nodes"])
        out = []
        for c in self.conditions:
            if c not in cond.CONDITIONS and not refs:
                continue
            layout, data, label = self.eval_source(c, at_node)
            out.append((c, c, layout, data, None, label))
            if refs and layout != A.LAYOUT_STORED and c in STORED_TOO and self.cfg["whole_gene_layout"] in (None, A.LAYOUT_STORED):
                out.append(("%s@%s" % (c, A.LAYOUT_STORED), c, A.LAYOUT_STORED, self.held_data()[0], None, A.LAYOUT_STORED))
            if not at_node and layout != A.LAYOUT_STORED and c in (cond.SCATTERED, cond.WHOLE_GENE):
                for ctl in self.cfg["end_controls"] or []:               # the controls: at the final evaluation only
                    out.append(("%s@%s" % (c, ctl), c, layout, data, ctl, "%s+%s" % (label, ctl)))
        return out

    def held_meta(self):
        """The meta fields of every held-out table the online frame layout reads (read once)."""
        if getattr(self, "_meta", None) is None:
            self._meta = self.held.load(T.FRAME_META)
        return self._meta

    def frame_tiers(self, which="heldout", examples=None):
        """{label: tables}: the frame relative of the whole gene by its distance to row 0 (round 1's ruling of 2026-11-23,
        to set the training's frames beside the evaluation's): self_copy (a row of the examined genome's own copies),
        tier_<t> (the row's distance tier, META tier; -1 not defined), none (no other row has a stretch in the target
        block). heldout: every held-out table laid whole (align.frame_relative of the stored table); train: the
        examples' frame_from (the rows drawn)."""
        ts = self.held if which == "heldout" else self.train
        if getattr(self, "_tiers", {}).get(which) is None:
            self._tiers = dict(getattr(self, "_tiers", {}), **{which: (np.asarray(ts.load(("tier",))["tier"]), np.asarray(ts.load(("kind",))["kind"]))})
        tier, kind = self._tiers[which]

        def label(i, r):
            if r is None or r < 0:
                return "none"
            if (kind[i, r] == A.KIND_SELF_COPY).any():
                return "self_copy"
            return "tier_%d" % int(tier[i, r])
        out = {}
        if which == "heldout":
            n = self.held_data()[0]["n"]
            pairs = [(i, AL.frame_relative(dict(n=n[i]))) for i in range(len(self.held))]
        else:
            pairs = [(int(e["table"]), int(e["frame_from"])) for e in examples or [] if "frame_from" in e]
        for i, r in pairs:
            k = label(i, r)
            out[k] = out.get(k, 0) + 1
        return out

    def strata(self, ids):
        """The evaluator's strata of the held-out tables ids; with the kinds of their rows where the evaluator reads them
        (the three relatives strata of round 1: other-genome rows, self copies only, none)."""
        _, identity, n_other = self.held_data()
        if "kind" in inspect.signature(SC.strata_of).parameters:
            return SC.strata_of(identity[ids], n_other[ids], np.asarray(self.held_data()[0]["kind"])[ids])
        return SC.strata_of(identity[ids], n_other[ids])

    def evaluate(self, ids, fixed, path, at_node=False):
        """The held-out tables ids under every condition of this arm (eval_plan), read by the moving average. The rows
        go to self.rows and the whole table to path after every condition; their layout column says how the grids were
        rendered. Returns {key: bits over all groups and strata, None when nothing was scored}, {key: (bits,
        cell_table)} and {key: items digest}, the keys of eval_plan."""
        strata = self.strata(ids)
        groups = self.held.group[ids]
        main, cells, digests = {}, {}, {}
        for key, c, layout, data, ctl, label in self.eval_plan(at_node):
            part = {f: v[ids] for f, v in data.items()}
            fixed = dict(fixed, layout=label)
            frame = None
            online = data is self.held_data()[0]                         # rendered here, not W3's offline rendering
            if online and self.frame_strict and layout != A.LAYOUT_STORED and cond.parse(c)[0] == cond.WHOLE_GENE:
                frame = dict(strict=True, cache=self._frames)            # as the arm was trained: W3's strict aligner
            elif online and self.frame_index is not None and layout != A.LAYOUT_STORED and cond.parse(c)[0] == cond.WHOLE_GENE:
                frame = dict(index=self.frame_index, meta={f: np.asarray(v)[ids] for f, v in self.held_meta().items()})
            bits, cell_table, digest, aa = E.score(self.reader(), part, ids, c, self.device, max_rows=int(self.cfg["eval_max_rows"]), layout=layout,
                                                   frame=frame, control=ctl)
            agg = SC.aggregate(bits, cell_table, groups, strata, self.n_groups, self.group_names)
            for label, b in zip(SC.PARTS, (aa, bits - aa)):           # the two parts of the bits, per group and over the groups
                agg += [dict(r, stratum=label) for r in SC.aggregate(b, cell_table, groups, [], self.n_groups, self.group_names) if r["stratum"] == SC.ALL]
            self.rows += [SC.result_row(condition=c, group=r["group"], stratum=r["stratum"], bits=r["bits"], target_cells=r["target_cells"],
                                        n_items=r["n_items"], **fixed) for r in agg]
            SC.write_rows(path + ".part", self.rows)
            os.replace(path + ".part", path)
            main[key] = next((r["bits"] for r in agg if r["group"] == SC.ALL and r["stratum"] == SC.ALL), None)
            cells[key], digests[key] = (bits, cell_table), digest
        return main, cells, digests


def worst_case(examples, rows):
    """The examples with their rows repeated up to `rows` rows each: the largest micro-batch a training can meet."""
    out = []
    for e in examples:
        k = -(-rows // e["grid"].shape[0])
        grown = {f: np.concatenate([e[f]] * k)[:rows] for f in ("grid", "truth", "ext", "genome_of_row") if f in e}   # every per-row array
        out.append(dict(e, **grown))
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
        t1 = time.time()                                                # the sampler's cost, drawn here: 20 steps past the guard's
        f0 = (run.sampler.frame_s, run.sampler.frame_tables)
        for s in range(20):
            run.sampler.draw(int(cfg["preflight_steps"]) + s)
        t1, nf = time.time() - t1, run.sampler.frame_tables - f0[1]
        out.update(sampler=dict(steps=20, s_per_step=round(t1 / 20, 4), frame_tables=nf,
                                frame_s_per_table=round((run.sampler.frame_s - f0[0]) / nf, 4) if nf else None, data_workers=int(cfg["data_workers"])))
        peak, warm, losses = float(cfg["lr"]), max(1, int(cfg["preflight_warmup"])), []
        t0, cells, timed = None, 0, 0
        feed = Feed(run.sampler, 0, int(cfg["data_workers"]))
        try:
            for s in range(1, int(cfg["preflight_steps"]) + 1):
                if s == 6:
                    t0 = time.time()                                    # the first steps warm the kernels up
                    w0 = feed.wait_s
                ex = feed.get(s - 1)
                bits, _, _, _ = run.step(ex, peak * min(1.0, s / warm))
                run.update_ema(s)
                losses.append(bits)
                if t0 is not None:
                    cells += sum(e["grid"].size for e in ex)
                    timed += 1
        except FloatingPointError:
            losses.append(float("nan"))
        feed.close()
        start, end = float(np.mean(losses[:5])), float(np.mean(losses[-20:]))
        bad = not all(math.isfinite(x) for x in losses) or not end < start
        if cfg["preflight_guard"] not in ("stop", "reduce"):
            raise SystemExit("preflight_guard is stop or reduce, not %r" % cfg["preflight_guard"])
        lr = peak * float(cfg["preflight_lr_scale"]) if bad and cfg["preflight_guard"] == "reduce" else peak
        took = time.time() - t0 if t0 is not None else 0.0
        alloc, reserved = run.peak_mb()
        out.update(guard=dict(steps=len(losses), warmup=warm, start_bits=round(start, 4), end_bits=round(end, 4) if math.isfinite(end) else None,
                              failed=bad, lr=lr), lr=lr, lr_reduced=bad, cells_per_s=round(cells / took, 1) if took > 0 else None,
                   steps_per_s=round(timed / took, 3) if took > 0 else None, tables_per_s=round(timed * int(cfg["per_step"]) / took, 3) if took > 0 else None,
                   data_wait_s_per_step=round((feed.wait_s - w0) / timed, 4) if timed else None,
                   peak_mb=max(round(alloc), max(t.get("peak_mb", 0) for t in tried)),
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
                   eval_bits={c: round(float(b.mean()), 4) if len(b) else None for c, (b, _, _, _) in scored.items()})
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


def evaluate_checkpoints(cfg):
    """The held-out evaluation of an arm's saved weights, again: the moving average of final.pt (rows "ema@final") and,
    when cfg["best"] names it, of best.pt (rows "ema@best"), on the first final_per_group tables of every group (0: all),
    under the arm's conditions, with the evaluator's plan, strata and rows. For an arm whose final evaluation the wall
    limit cut short, or for all held-out tables after a training that scored a subset. cfg: the arm's training
    configuration plus final (the path of the weights read: final.pt, or with weights "best" or "node@<tables>"
    best.pt or ema_tables<tables>.pt, rows "ema@best" or "ema@tables=<n>"), best (that of best.pt or None), out (an
    empty folder), and the tables seen, cells seen and wall seconds of the weights read (final_tables, final_cells,
    final_wall_s) and of the best node (best_tables, best_cells, best_wall_s). Writes results.tsv, heldout_final.npz
    (heldout_best.npz for the best weights) and summary.json in out."""
    began = time.time()
    run = Run(dict(cfg, out=cfg["out"]), evaluate_only=True)            # a ruler set of package W18 may be the held-out set here
    cfg = dict(run.cfg, **{k: v for k, v in cfg.items() if k not in run.cfg})
    if cfg.get("readout"):
        return read_ruler(run, cfg, began)
    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    for name in ("results.tsv", "summary.json"):
        if os.path.exists(os.path.join(out_dir, name)):
            raise SystemExit("%s exists in %s: an evaluation starts in an empty folder" % (name, out_dir))
    results, summary = os.path.join(out_dir, "results.tsv"), os.path.join(out_dir, "summary.json")
    final_ids = E.subset(run.held, int(cfg["final_per_group"]))
    half = "all" if len(final_ids) == len(run.held) else "subset"
    reader = run.reader()
    weights = str(cfg.get("weights") or "final")                      # the weights in cfg["final"]: final, best or node@<tables>
    label = "tables=%s" % weights.split("@")[1] if weights.startswith("node@") else weights
    state = dict(name=cfg["name"], status="evaluating", conditions=run.conditions, main_condition=run.main, versions=versions(), params=run.params,
                 eval_final=len(final_ids), weights_read="ema", weights=weights)
    for which, key in (("final", "final"), ("best", "best")):
        path = cfg.get(key)
        if not path:
            continue
        t = time.time()
        ck = torch.load(path, map_location=run.device, weights_only=True)
        reader.load_state_dict(ck["ema"])
        fixed = run.fixed_columns("ema@%s" % (label if which == "final" else which), half, int(cfg["%s_cells" % which]), 0.0, "ok",
                                  int(cfg["%s_tables" % which]), float(cfg["%s_wall_s" % which]))
        main, cells, digests = run.evaluate(final_ids, fixed, results)
        np.savez(os.path.join(out_dir, "heldout_%s.npz" % which),                # the best weights' cells too (paired intervals)
                 **{"%s_%s" % (x, c.replace("+", "__")): v for c, (b, tb) in cells.items() for x, v in (("bits", b), ("table", final_ids[tb]))})
        if which == "final":
            state.update(final=main, final_items=digests, final_tables=len(final_ids), final_eval_s=round(time.time() - t, 1), step=int(ck["step"]),
                         tables_seen=int(ck["tables_seen"]), cells_seen=int(ck["cells_seen"]))
        else:
            state["final_best"] = dict(tables_seen=int(cfg["best_tables"]), bits=main, items=digests, eval_s=round(time.time() - t, 1))
        write_json(summary, state)
    state.update(status="ok", wall_s=round(time.time() - began, 1), row_key_mask=dict(C.ROW_MASK))
    write_json(summary, state)
    return state


UNTRAINED = "untrained"


def read_ruler(run, cfg, began):
    """The passes of a readout module (cfg["readout"]: module, presence, max_rows, layout; package W18's
    readouts/w18_passes.run_ruler) on the ruler set held as run.held, scored by the moving average of final.pt, or by
    the arm's model as built with its seed and never trained (cfg["final"] == "untrained"). The module writes its
    per-cell file into cfg["out"] (no result rows: a ruler set is no item of a round); summary.json beside it."""
    import importlib
    ro = dict(cfg["readout"])
    out_dir = cfg["out"]
    os.makedirs(out_dir, exist_ok=True)
    summary = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary):
        raise SystemExit("summary.json exists in %s: a readout starts in an empty folder" % out_dir)
    reader = run.reader()
    untrained = cfg.get("final") == UNTRAINED
    ck = None
    if not untrained:
        ck = torch.load(cfg["final"], map_location=run.device, weights_only=True)
        reader.load_state_dict(ck["ema"])
    else:                                                             # the weights read are kept, as a checkpoint of the index
        body = dict(name=cfg["name"], size=run.size, module=cfg["model"], step=0, tables_seen=0, cells_seen=0, commit=cfg["commit"], ema=reader.state_dict())
        torch.save(body, os.path.join(out_dir, "untrained.pt.part"))
        os.replace(os.path.join(out_dir, "untrained.pt.part"), os.path.join(out_dir, "untrained.pt"))
    layout = str(ro.get("layout") or A.LAYOUT_STORED)
    if layout == A.LAYOUT_STORED:
        render, version = None, T.RENDER_VERSION
    elif ro.get("offline"):                                           # W18's strict offline rendering: the module renders (W3, leak-free)
        from seeddrift.train.rel.tables import render as RD
        render, version = None, "w3-%s" % RD.VERSION
    else:
        from seeddrift.train.rel.tables import render as RD
        with_ext = getattr(reader, "ext", None) is not None            # an arm trained on stored tables reads no ext cells

        def render(tab, visible, target):
            grid, truth, ext = RD.render(tab, visible, target, layout=layout)
            return (grid, truth, ext) if with_ext else (grid, truth)
        version = "w3-%s" % RD.VERSION
    max_rows = int(ro.get("max_rows", 32))
    info = dict(arm=cfg["name"], weights=UNTRAINED if untrained else str(cfg.get("weights") or "final"), relatives=bool(cfg["relatives"]), layout=layout,
                render_version=version,
                run=cfg["run"], commit=cfg["commit"], root=cfg["heldout_root"])
    extra = {}
    if ro.get("in_place_only"):                                       # the protocol's switch of compute: in-place implants only
        extra = dict(scenarios=[0]) if untrained else (dict(presence_scenarios=[0]) if ro.get("presence") else {})
    if ro.get("adapt_v0"):                                            # tables of version 1 read by an arm trained on version 0
        extra["adapt_v0"] = True
    if ro.get("offline"):                                             # the run of stage rel_w18_offline on this ruler set
        extra["offline"] = str(ro["offline"])
    head = importlib.import_module(str(ro["module"])).run_ruler(lambda examples: E.score_cells(reader, examples, run.device, max_rows=max_rows),
                                                                run.held, info, out_dir, render=render, presence=bool(ro.get("presence")),
                                                                max_rows=max_rows, **extra)
    state = dict(name=cfg["name"], status="ok", readout=str(ro["module"]), weights=info["weights"], layout=layout, presence=bool(ro.get("presence")),
                 in_place_only={k: v for k, v in extra.items() if k != "offline"}, offline=extra.get("offline"),
                 tables=len(run.held), table_set=run.held.name, table_kind=run.held.kind, versions=versions(), params=run.params,
                 step=int(ck["step"]) if ck else 0, tables_seen=int(ck["tables_seen"]) if ck else 0, cells_seen=int(ck["cells_seen"]) if ck else 0,
                 ruler={k: v for k, v in (head or {}).items() if isinstance(v, (str, int, float, bool)) or v is None},
                 wall_s=round(time.time() - began, 1), row_key_mask=dict(C.ROW_MASK))
    write_json(summary, state)
    return state


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
    k = dict(step=0, tables=0, cells=0, targets=0, train_s=0.0, eval_s=0.0, lr=0.0, alone=0)
    state["alone_share"] = cfg["alone_share"]                         # the cap on tables without other rows (None: as they come)

    def save(name, ema_only):
        if cfg["probe"]:
            return
        body = dict(name=cfg["name"], size=run.size, module=cfg["model"], step=k["step"], tables_seen=k["tables"], cells_seen=k["cells"],
                    commit=cfg["commit"], ema=run.reader().state_dict())
        if not ema_only:
            body["model"] = run.model.state_dict()
        p = os.path.join(out_dir, name)
        torch.save(body, p + ".part")
        os.replace(p + ".part", p)

    node_cells = {}                                                    # node_cells.npz: tables (ids), t<n>_<condition>_bits / _cells

    def node():
        t = time.time()
        rate = k["cells"] / k["train_s"] if k["train_s"] > 0 else 0.0
        fixed = run.fixed_columns("%s@tables=%d" % (reads, k["tables"]), "subset", k["cells"], rate, "running", k["tables"], time.time() - began)
        main, cells_n, digests = run.evaluate(sub, fixed, results, at_node=True)
        for key, (b, tb) in cells_n.items():                           # every node's bits per held-out table (paired intervals between nodes)
            c = key.replace("+", "__")
            node_cells["t%d_%s_bits" % (k["tables"], c)] = np.bincount(tb, weights=b, minlength=len(sub))
            node_cells["t%d_%s_cells" % (k["tables"], c)] = np.bincount(tb, minlength=len(sub)).astype(np.int64)
        np.savez(os.path.join(out_dir, "node_cells.npz.part.npz"), tables=sub, **node_cells)
        os.replace(os.path.join(out_dir, "node_cells.npz.part.npz"), os.path.join(out_dir, "node_cells.npz"))
        took = time.time() - t
        k["eval_s"] += took
        rec = dict(tables_seen=k["tables"], step=k["step"], cells_seen=k["cells"], target_cells=k["targets"], lr=k["lr"], wall_s=round(time.time() - began, 1),
                   eval_s=round(took, 1), tables_scored=len(sub), bits=main, items=digests,
                   alone_share_seen=round(k["alone"] / k["tables"], 4) if k["tables"] else None)
        append_json(evals_path, rec)
        state["evals"].append({x: rec[x] for x in rec if x != "items"})
        m = main.get(run.main)
        if m is not None and k["step"] and (state["best"] is None or m < state["best"]["bits"]):
            state["best"] = dict(step=k["step"], tables_seen=k["tables"], cells_seen=k["cells"], bits=m, condition=run.main, wall_s=rec["wall_s"])
            save("best.pt", ema_only=True)
        if cfg["keep_node_weights"] and k["step"]:                    # the moving average at every node (for readings at a common node)
            save(NODE_WEIGHTS % k["tables"], ema_only=True)
            state.setdefault("node_weights", []).append(k["tables"])
        write_json(summary, state)
        return took

    eval0 = node()                                                   # the untrained model: the start of every curve
    # final_best (default: the arms trained without relatives): the weights of the best node evaluated on the final
    # tables too, so that an arm's best can be set against another arm's end on the same tables
    best_too = (not cfg["relatives"]) if cfg["final_best"] is None else bool(cfg["final_best"])
    more = len(run.eval_plan(False)) / max(1, len(run.eval_plan(True)))   # the final evaluation reads the references too
    reserve = eval0 * len(final_ids) / max(1, len(sub)) * more * 1.3 * (2 if best_too else 1) + 60.0
    deadline = began + float(cfg["wall_s"]) - reserve
    if deadline <= time.time():
        raise SystemExit("the wall limit of %s s leaves no time to train (the final evaluation needs about %.0f s)" % (cfg["wall_s"], reserve))
    state.update(final_eval_reserve_s=round(reserve, 1), deadline_s=round(deadline - began, 1))
    budget = float(cfg["budget_cells"]) if cfg["budget_cells"] else None
    peak, warm = float(cfg["lr"]), max(1, int(cfg["warmup_steps"]))
    next_node = node_after(cfg, 0)
    names = run.sampler.masks
    shares = {m: dict(tables=0, target_cells=0, bits=0.0) for m in names}  # what every mask scheme took of the cells and of the loss
    framed = run.frame_strict and cfg["relatives"]
    frames = {}                                                         # the whole gene's frame relatives in training, by distance
    if framed:
        state["frame_tiers"] = dict(heldout=run.frame_tiers("heldout"), train=frames)
    log = open(os.path.join(out_dir, "steps.csv"), "w")
    log.write("step,tables_seen,cells_seen,target_cells,alone_tables,loss_bits,lr,cells_per_s,peak_mb,grad_norm,wall_s," + ",".join("bits_" + m for m in names) +
              "," + ",".join("cells_" + m for m in names) + ",data_wait_s\n")
    ended_by, last_flush = None, time.time()
    feed = Feed(run.sampler, 0, int(cfg["data_workers"]))
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
            waited = feed.wait_s
            examples = feed.get(k["step"])
            waited = feed.wait_s - waited
            bits, norm, sb, sc = run.step(examples, lr)
            for e in examples:
                shares[names[e["scheme"]]]["tables"] += 1
            if framed:
                for lab, n in run.frame_tiers("train", examples).items():
                    frames[lab] = frames.get(lab, 0) + n
            for m, b, c in zip(names, sb, sc):
                shares[m]["target_cells"] += int(c)
                shares[m]["bits"] += float(b)
            k["step"] += 1
            run.update_ema(k["step"])
            took = time.time() - now
            cells = sum(e["grid"].size for e in examples)
            k["tables"] += len(examples)
            k["alone"] += sum(1 for e in examples if e.get("alone"))
            k["cells"] += cells
            k["targets"] += int(sum(sc))
            k["train_s"] += took
            log.write("%d,%d,%d,%d,%d,%.5f,%.4g,%.0f,%.0f,%.4g,%.1f,%s,%s,%.4f\n" % (k["step"], k["tables"], k["cells"], k["targets"], k["alone"], bits, lr,
                                                                                 cells / took, run.peak_mb()[0], norm, time.time() - began,
                                                                                 ",".join("%.5f" % (b / c) if c else "" for b, c in zip(sb, sc)),
                                                                                 ",".join("%d" % c for c in sc), waited))
            if time.time() - last_flush > 30:
                log.flush()
                last_flush = time.time()
            if k["tables"] >= next_node:
                node()
                next_node = node_after(cfg, k["tables"])
    except FloatingPointError as err:
        log.close()
        feed.close()
        state.update(status="diverged", error=str(err), steps=k["step"], tables_seen=k["tables"], cells_seen=k["cells"])
        write_json(summary, state)
        raise SystemExit("arm %s diverged at step %d: %s" % (cfg["name"], k["step"], err))
    log.close()
    feed.close()
    rate = k["cells"] / k["train_s"] if k["train_s"] > 0 else 0.0
    alloc, reserved = run.peak_mb()
    cells_all, bits_all = sum(x["target_cells"] for x in shares.values()), sum(x["bits"] for x in shares.values())
    state["schemes"] = {m: dict(tables=x["tables"], target_cells=x["target_cells"], cell_share=round(x["target_cells"] / cells_all, 4) if cells_all else None,
                                loss_share=round(x["bits"] / bits_all, 4) if bits_all else None,
                                bits_per_cell=round(x["bits"] / x["target_cells"], 4) if x["target_cells"] else None) for m, x in shares.items()}
    state.update(data_workers=int(cfg["data_workers"]), data_wait_s=round(feed.wait_s, 1))
    state.update(status="evaluating", ended_by=ended_by, steps=k["step"], tables_seen=k["tables"], cells_seen=k["cells"], target_cells=k["targets"],
                 cells_per_s=round(rate, 1), tables_per_s=round(k["tables"] / k["train_s"], 3) if k["train_s"] > 0 else None, train_s=round(k["train_s"], 1),
                 eval_s=round(k["eval_s"], 1), eval_overhead=round(k["eval_s"] / k["train_s"], 4) if k["train_s"] > 0 else None,
                 peak_mb=round(alloc), reserved_mb=round(reserved), lr_last=k["lr"],
                 alone_share_seen=round(k["alone"] / k["tables"], 4) if k["tables"] else None)
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
