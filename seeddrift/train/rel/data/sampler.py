"""The training examples of a step (numpy only).

Every step draws its tables, their rows and their masks from (seed, step), so a step is the same whatever came before
it. Tables: a group drawn uniformly, then one of its tables; with alone_share (arms with relatives only), the tables of
a group that have no other row are drawn at most at that share (their own share where it is lower). Rows: row 0 and a random choice of the other rows that
exist, so that the table has between rows[0] and rows[1] rows (with relatives = False every other row is removed).
Mask scheme: one per table, drawn among `masks` (modules of seeddrift.train.rel.data.masks) with `weights`
(uniform when not given). Layout: per mask scheme (`layouts`, default stored for every scheme): the stored placement
(tables.render) or another layout of package W3's renderer (tables/render.py, e.g. leakfree), which also marks the
cells it laid by extension (ext); or frame (whole gene only): the table, after its rows are drawn, laid on its frame
relative by package W3's strict aligner (tables/align.py frame_layout, the function the evaluation uses), then
rendered leak-free (the example's frame_from: the stored row its frame relative is, -1 for none); or strict (scattered
and span): W3's strict aligner on the rows drawn and the cells hidden (tables/strict.py render), then leak-free. One
sample in CHECK_EVERY of these two layouts (by its number over the training) is checked by W3's fourth test (the
example's traincheck: 0 passed, 1 failed; only for an arm with a strict layout). The sampler counts the seconds it spends in frame_layout and in the strict
rendering and the tables it laid (frame_s, frame_tables, strict_s, strict_tables), in the process that draws.

An example is a dict: grid uint8 [R, ROW_LEN] (what the model reads), truth uint8 [R, ROW_LEN] (0 = no target),
scheme (index into masks), table, group, layout; and ext bool [R, ROW_LEN] when the layout is not stored.
"""
from __future__ import annotations

import time

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel import registry
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.tables import align as AL
from seeddrift.train.rel.tables import render as RD

SAMPLER_VERSION = "5"
FRAME = "frame"                 # the whole gene laid on its frame relative by W3's strict aligner, then rendered leak-free
STRICT = "strict"               # scattered and span laid by W3's strict aligner (tables/strict.py, aligner version 2), then leak-free
CHECK_EVERY, CHECK_SEED = 1000, 20261124        # the fourth test on one sample in 1,000 of the strict and frame layouts


def mask_scheme(name):
    return registry.load("mask", name)


class Sampler:
    def __init__(self, tables, seed, masks=("scattered", "span", "whole_gene"), weights=None, rows=(4, 16), relatives=True, per_step=4, alone_share=None,
                 layouts=None, frame_index=None):
        """frame_index: the pair index of the uncapped search (package W2's v1_hits.Index): a whole-gene sample in a
        layout other than stored is placed on its frame relative online (package W3's align.frame_layout_online), after
        the rows are drawn; without it such a sample is laid from the first column of the block (W3's fallback)."""
        self.tables, self.seed, self.per_step, self.relatives = tables, int(seed), int(per_step), bool(relatives)
        self.frame_index = frame_index
        self.masks = [str(m) for m in masks]
        self.layouts = [str((layouts or {}).get(m, A.LAYOUT_STORED)) for m in self.masks]
        if set(layouts or {}) - set(self.masks) or set(self.layouts) - set(RD.LAYOUTS) - {FRAME, STRICT} or \
                any(x == FRAME and m != "whole_gene" for m, x in zip(self.masks, self.layouts)) or \
                any(x == STRICT and m == "whole_gene" for m, x in zip(self.masks, self.layouts)):
            raise ValueError("layouts %r: a layout of %s for mask schemes among %r (frame: whole_gene only; strict: not whole_gene)"
                             % (layouts, RD.LAYOUTS, self.masks))
        self.frame_s, self.frame_tables, self.strict_s, self.strict_tables = 0.0, 0, 0.0, 0
        self.checks = STRICT in self.layouts                            # an arm trained strictly checks its strict and frame samples
        modules = [mask_scheme(m) for m in self.masks]
        self.plans = [m.plan for m in modules]
        weighted = {bool(getattr(m, "WEIGHTED", False)) for m in modules}
        if len(weighted) > 1:
            raise ValueError("mask schemes %r mix weighted and unweighted schemes" % (self.masks,))
        self.weighted = weighted == {True}
        w = np.ones(len(self.plans)) if weights is None else np.asarray(weights, dtype=np.float64)
        if len(w) != len(self.plans) or not len(w) or (w < 0).any() or w.sum() <= 0:
            raise ValueError("mask weights %r do not fit the schemes %r" % (weights, self.masks))
        self.weights = w / w.sum()
        self.rows = (int(rows[0]), int(rows[1]))
        if not 1 <= self.rows[0] <= self.rows[1]:
            raise ValueError("rows %r: at least one row and a range" % (rows,))
        groups = sorted(set(tables.group.tolist()))
        self.by_group = [np.flatnonzero(tables.group == g) for g in groups]
        if not groups:
            raise ValueError("the table set has no tables")
        self.alone = None
        if alone_share is not None and self.relatives:
            if not 0 <= float(alone_share) <= 1:
                raise ValueError("alone_share %r is a share" % (alone_share,))
            kind = tables.load(("kind",))["kind"]
            has = (np.asarray(kind) != T.KIND_NONE).any(axis=2)[:, 1:].any(axis=1)
            self.alone = []
            for ids in self.by_group:
                lone, other = ids[~has[ids]], ids[has[ids]]
                p = min(len(lone) / len(ids), float(alone_share)) if len(other) else 1.0
                self.alone.append((lone, other, p))

    def example(self, rng, index=None):
        """One example; index (the sample's number over the training: step x per_step + its place in the step) marks
        the one sample in CHECK_EVERY of the strict and frame layouts that the fourth test checks (traincheck)."""
        for _ in range(16):
            g = int(rng.integers(len(self.by_group)))
            if self.alone is None:
                ids = self.by_group[g]
            else:
                lone, other, p = self.alone[g]
                ids = lone if len(lone) and rng.random() < p else other
            i = int(ids[rng.integers(len(ids))])
            tab = self.tables.table(i)
            others = T.present_rows(tab)[1:]
            frames = np.unique([f for f in np.asarray(tab.get("frame_row", [0])).reshape(-1).tolist() if f > 0])
            frames = frames[np.isin(frames, others)].astype(np.int64)  # a block framed on another row keeps that row
            if self.relatives and len(frames):
                k = int(rng.integers(self.rows[0], self.rows[1] + 1)) - 1
                rest = np.setdiff1d(others, frames)
                m = min(max(k - len(frames), 0), len(rest))
                keep = np.sort(np.concatenate([frames, rng.choice(rest, size=m, replace=False) if m else rest[:0]]))
            elif self.relatives:
                k = int(rng.integers(self.rows[0], self.rows[1] + 1)) - 1
                keep = np.sort(rng.choice(others, size=min(k, len(others)), replace=False)) if len(others) and k > 0 else others[:0]
            else:
                keep = others[:0]
            chosen = np.concatenate([[0], keep]).astype(np.int64)
            tab = T.take_rows(tab, chosen)
            s = int(rng.choice(len(self.plans), p=self.weights))
            visible, target = self.plans[s](dict(n=tab["n"], kind=tab["kind"]), rng)
            lay = self.layouts[s]
            if lay == A.LAYOUT_STORED:
                grid, truth = T.render(tab, visible, target)
                ext = None
            elif lay == FRAME:                                          # W3's strict aligner on the rows drawn, then leak-free
                t0 = time.perf_counter()
                col, frame_row, frame_len = AL.frame_layout(tab)
                self.frame_s += time.perf_counter() - t0
                self.frame_tables += 1
                grid, truth, ext = RD.render(dict(tab, col=col, frame_row=frame_row, frame_len=frame_len), visible, target, layout=A.LAYOUT_LEAKFREE)
                fr = AL.frame_relative(tab)                             # the frame relative of the target block, as a row of the stored table
                frame_from = int(chosen[fr]) if fr is not None else -1
            elif lay == STRICT:                                         # scattered and span: W3's strict aligner, then leak-free
                from seeddrift.train.rel.tables import strict as ST
                t0 = time.perf_counter()
                grid, truth, ext, _ = ST.render(tab, visible, target, whole_gene=False)
                self.strict_s += time.perf_counter() - t0
                self.strict_tables += 1
            else:
                if self.masks[s] == "whole_gene" and self.frame_index is not None:   # the whole gene on its frame relative
                    col, frame_row, frame_len, _ = AL.frame_layout_online(tab, T.take_meta(self.tables.meta(i), chosen), self.frame_index)
                    tab = dict(tab, col=col, frame_row=frame_row, frame_len=frame_len)
                grid, truth, ext = RD.render(tab, visible, target, layout=lay)
            if truth.any():
                out = dict(grid=grid, truth=truth, scheme=s, table=i, group=int(self.tables.group[i]), alone=not len(others),
                           cells=int(T.inside(tab).sum()), layout=lay, rows=chosen)
                if self.checks and index is not None and lay in (FRAME, STRICT) and int(index) % CHECK_EVERY == 0:
                    from seeddrift.train.rel.tables import strict as ST     # the fourth test on this rendering (W3)
                    got = ST.fourth_test(tab, visible, target, whole_gene=lay == FRAME, key=int(index), seed=CHECK_SEED)
                    out["traincheck"] = 0 if got["passed"] else 1
                if ext is not None:
                    out["ext"] = ext
                if lay == FRAME:
                    out["frame_from"] = frame_from                       # -1: no other row has a stretch in the target block
                if "genome_of_row" in tab:
                    out["genome_of_row"] = np.asarray(tab["genome_of_row"]).astype(np.int64)
                return out
        raise RuntimeError("sixteen tables in a row gave no target cell: the table set or the mask schemes are wrong")

    def draw(self, step):
        rng = np.random.default_rng([self.seed, int(step)])
        out = [self.example(rng, int(step) * self.per_step + j) for j in range(self.per_step)]
        if self.weighted:
            # weight of a table's cells: its cells over its target cells (the inverse of the ratio it realised), scaled so
            # that the weights of the step's target cells sum to their number (the trainer divides by that number)
            t = [int((e["truth"] > 0).sum()) for e in out]
            scale = sum(t) / max(1, sum(e["cells"] for e in out))
            for e, n in zip(out, t):
                e["weight"] = e["cells"] / n * scale
        return out


def collate(examples):
    """A micro-batch: grid uint8 [N, ROW_LEN] (the rows of all its tables), sizes (rows per table), and for the target
    cells row (into N), column, truth token and the index of their example."""
    grid = np.concatenate([e["grid"] for e in examples], axis=0)
    sizes = [int(e["grid"].shape[0]) for e in examples]
    truth = np.concatenate([e["truth"] for e in examples], axis=0)
    row, col = np.nonzero(truth)
    first = np.cumsum([0] + sizes)
    example = (np.searchsorted(first, row, side="right") - 1).astype(np.int64)
    out = dict(grid=grid, sizes=sizes, row=row.astype(np.int64), col=col.astype(np.int64), token=truth[row, col].astype(np.int64), example=example)
    if all("genome_of_row" in e for e in examples):                 # the genome of every row (read as equal or not only)
        out["genome"] = np.concatenate([np.asarray(e["genome_of_row"], dtype=np.int64) for e in examples])
    if any("ext" in e for e in examples):                            # the cells a renderer laid by extension
        out["ext"] = np.concatenate([np.asarray(e["ext"], dtype=bool) if "ext" in e else np.zeros(e["grid"].shape, dtype=bool) for e in examples], axis=0)
    if all("weight" in e for e in examples):
        out["weight"] = np.asarray([e["weight"] for e in examples], dtype=np.float32)[example]
    return out
