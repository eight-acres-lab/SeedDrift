"""Held-out evaluation of a model during and after its training, and the result rows it writes.

The plan of every condition is seeddrift.train.rel.data.heldout_plan (shared with the baselines): only cells of the
examined row in its target block are scored, under the stored placement. bits = sum of -log2 p(true token) over the
scored cells / their number, pooled inside a (group, stratum); the row of group "all" is the plain mean of the groups
that have scored cells. The metadata (group, identity) is read here for the report only and never reaches the model.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from seeddrift.train.rel import api as A
from seeddrift.train.rel.data import heldout_plan as H
from seeddrift.train.rel.data import sampler as S
from seeddrift.train.rel.data import tables as T

EVAL_VERSION = "1"


def to_device(batch, device):
    out = dict(sizes=batch["sizes"])
    for k in ("grid", "row", "col", "token", "example"):
        out[k] = torch.from_numpy(batch[k]).to(device, non_blocking=True).long()
    return out


@torch.no_grad()
def score(model, tables, ids, cond, device, max_rows=64):
    """bits (float64 [len(ids)], the sum over the scored cells) and cells (int64 [len(ids)]) of condition cond; a
    table without that condition has 0 cells."""
    bits, cells = np.zeros(len(ids)), np.zeros(len(ids), dtype=np.int64)
    pending, rows = [], 0

    def flush():
        nonlocal pending, rows
        if pending:
            b = to_device(S.collate([dict(grid=g, truth=t) for _, g, t in pending]), device)
            logp = model.logp(b["grid"], b["sizes"], b["row"], b["col"])
            per = -logp.gather(1, (b["token"] - A.TARGET_MIN)[:, None])[:, 0].double() / math.log(2.0)
            tot = torch.zeros(len(pending), dtype=torch.float64, device=per.device).index_add_(0, b["example"], per).cpu().numpy()
            cnt = torch.bincount(b["example"], minlength=len(pending)).cpu().numpy()
            for (k, _, _), x, c in zip(pending, tot, cnt):
                bits[k], cells[k] = x, c
        pending, rows = [], 0

    was_training = model.training
    model.eval()
    for k, i in enumerate(ids):
        tab = tables.table(int(i))
        plan = H.condition(T.take_rows(tab, T.present_rows(tab)), cond, int(i))
        if plan is None:
            continue
        grid, truth = T.render(*plan)
        if not truth.any():
            continue
        if rows and rows + grid.shape[0] > max_rows:
            flush()
        pending.append((k, grid, truth))
        rows += grid.shape[0]
    flush()
    model.train(was_training)
    return bits, cells
