"""The training examples of a step (numpy only).

Every step draws its tables, their rows and their masks from (seed, step), so a step is the same whatever came before
it. Tables: a group drawn uniformly, then one of its tables; with alone_share (arms with relatives only), the tables of
a group that have no other row are drawn at most at that share (their own share where it is lower). Rows: row 0 and a random choice of the other rows that
exist, so that the table has between rows[0] and rows[1] rows (with relatives = False every other row is removed).
Mask scheme: one per table, drawn among `masks` (modules of seeddrift.train.rel.data.masks) with `weights`
(uniform when not given). Layout: per mask scheme (`layouts`, default stored for every scheme): the stored placement
(tables.render) or another layout of package W3's renderer (tables/render.py, e.g. leakfree), which also marks the
cells it laid by extension (ext).

An example is a dict: grid uint8 [R, ROW_LEN] (what the model reads), truth uint8 [R, ROW_LEN] (0 = no target),
scheme (index into masks), table, group, layout; and ext bool [R, ROW_LEN] when the layout is not stored.
"""
from __future__ import annotations

import numpy as np

from seeddrift.train.rel import api as A
from seeddrift.train.rel import registry
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.tables import render as RD

SAMPLER_VERSION = "2"


def mask_scheme(name):
    return registry.load("mask", name)


class Sampler:
    def __init__(self, tables, seed, masks=("scattered", "span", "whole_gene"), weights=None, rows=(4, 16), relatives=True, per_step=4, alone_share=None,
                 layouts=None):
        self.tables, self.seed, self.per_step, self.relatives = tables, int(seed), int(per_step), bool(relatives)
        self.masks = [str(m) for m in masks]
        self.layouts = [str((layouts or {}).get(m, A.LAYOUT_STORED)) for m in self.masks]
        if set(layouts or {}) - set(self.masks) or set(self.layouts) - set(RD.LAYOUTS):
            raise ValueError("layouts %r: a layout of %s for mask schemes among %r" % (layouts, RD.LAYOUTS, self.masks))
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

    def example(self, rng):
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
            if self.relatives:
                k = int(rng.integers(self.rows[0], self.rows[1] + 1)) - 1
                keep = np.sort(rng.choice(others, size=min(k, len(others)), replace=False)) if len(others) and k > 0 else others[:0]
            else:
                keep = others[:0]
            tab = T.take_rows(tab, np.concatenate([[0], keep]).astype(np.int64))
            s = int(rng.choice(len(self.plans), p=self.weights))
            visible, target = self.plans[s](dict(n=tab["n"], kind=tab["kind"]), rng)
            lay = self.layouts[s]
            if lay == A.LAYOUT_STORED:
                grid, truth = T.render(tab, visible, target)
                ext = None
            else:
                grid, truth, ext = RD.render(tab, visible, target, layout=lay)
            if truth.any():
                out = dict(grid=grid, truth=truth, scheme=s, table=i, group=int(self.tables.group[i]), alone=not len(others),
                           cells=int(T.inside(tab).sum()), layout=lay)
                if ext is not None:
                    out["ext"] = ext
                return out
        raise RuntimeError("sixteen tables in a row gave no target cell: the table set or the mask schemes are wrong")

    def draw(self, step):
        rng = np.random.default_rng([self.seed, int(step)])
        out = [self.example(rng) for _ in range(self.per_step)]
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
    if any("ext" in e for e in examples):                            # the cells a renderer laid by extension
        out["ext"] = np.concatenate([np.asarray(e["ext"], dtype=bool) if "ext" in e else np.zeros(e["grid"].shape, dtype=bool) for e in examples], axis=0)
    if all("weight" in e for e in examples):
        out["weight"] = np.asarray([e["weight"] for e in examples], dtype=np.float32)[example]
    return out
