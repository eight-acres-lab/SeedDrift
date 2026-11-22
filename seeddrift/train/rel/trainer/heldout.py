"""Held-out evaluation of a model during and after its training.

What is hidden and scored, the strata, the averaging and the result rows are the evaluator's
(seeddrift/train/rel/evalh: conditions.materialise, score.strata_of, score.aggregate, score.result_row), the one
implementation every arm and every classical baseline shares: only cells of the examined row in its target block are
scored, a table's id is its position in the held-out set, and the digest of what was scored (conditions.ItemsDigest)
equals the baselines' when the same tables were scored. This module renders the planned tables in the model's grid
(data/tables.render), runs the model and returns the bits of every scored cell. The metadata (group, identity) is
used for the report only and never reaches the model.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from seeddrift.train.rel import api as A
from seeddrift.train.rel.data import sampler as S
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score as SC

EVAL_VERSION = "4"
PLAN_FIELDS = ("tok", "n", "col", "kind", "frame_len")


def to_device(batch, device):
    out = dict(sizes=batch["sizes"])
    for k in ("grid", "row", "col", "token", "example"):
        out[k] = torch.from_numpy(batch[k]).to(device, non_blocking=True).long()
    if "weight" in batch:
        out["weight"] = torch.from_numpy(batch["weight"]).to(device, non_blocking=True).float()
    if "ext" in batch:
        out["ext"] = torch.from_numpy(batch["ext"]).to(device, non_blocking=True).bool()
    return out


def ext_of(batch):
    """The keyword a model reads the cells laid by extension with: none for a batch without them (the stored layout),
    so a model that does not know the keyword is called as before."""
    return {"ext": batch["ext"]} if "ext" in batch else {}


def subset(tables, per_group):
    """The evaluator's subset (conditions.subset): the first per_group tables of every group, in storage order; 0: all
    tables. Positions in the table set."""
    if not per_group:
        return np.arange(len(tables), dtype=np.int64)
    return np.asarray(cond.subset(tables.group, int(per_group)), dtype=np.int64)


def conditions_of(relatives):
    """The conditions an arm is scored under: an arm trained without relatives is read only without them."""
    return [c for c in cond.CONDITIONS if relatives or cond.parse(c)[1] == cond.NO_RELATIVES]


def other_rows(kind):
    """(N,) the number of other rows every table has (kind (N, R, B))."""
    return cond.present_rows(kind)[:, 1:].sum(axis=1)


@torch.no_grad()
def score(model, data, ids, condition, device, max_rows=64, chunk=32):
    """The tables ids (positions in the held-out set; data: their arrays PLAN_FIELDS, in the same order) under one
    condition. Returns bits (float64 [C], one per scored cell), cell_table (int64 [C], the index into ids of the
    cell's table), the digest of what was scored, and the bits of the amino acid of every scored cell (the first of the
    two parts of the evaluator's score.amino_acid_bits; reporting only, the genetic code never reaches the model)."""
    digest = cond.ItemsDigest(condition)
    bits, cell_table, aa_bits = [], [], []
    pending, rows = [], 0
    classes = torch.from_numpy(SC.amino_acid_classes()[1:66]).to(device)                      # class of every token 1..65
    onehot = torch.nn.functional.one_hot(classes, int(classes.max()) + 1).double()

    def flush():
        nonlocal pending, rows
        if pending:
            b = to_device(S.collate([dict(grid=g, truth=t) for _, g, t in pending]), device)
            logp = model.logp(b["grid"], b["sizes"], b["row"], b["col"], **ext_of(b))
            per = -logp.gather(1, (b["token"] - A.TARGET_MIN)[:, None])[:, 0].double() / math.log(2.0)
            bits.append(per.cpu().numpy())
            p_class = logp.double().exp() @ onehot
            aa_bits.append((-torch.log2(p_class.gather(1, classes[b["token"] - A.TARGET_MIN][:, None])[:, 0])).cpu().numpy())
            cell_table.append(np.array([k for k, _, _ in pending], dtype=np.int64)[b["example"].cpu().numpy()])
        pending, rows = [], 0

    was_training = model.training
    model.eval()
    ids = np.asarray(ids, dtype=np.int64)
    for at in range(0, len(ids), chunk):
        part = slice(at, min(at + chunk, len(ids)))
        plan = cond.materialise({f: data[f][part] for f in PLAN_FIELDS}, ids[part], condition)
        digest.add(plan)
        for k in range(len(ids[part])):
            if not plan["has"][k]:
                continue
            planned = dict(tok=plan["tok"][k], n=plan["n"][k], col=plan["col"][k], kind=plan["kind"][k], frame_len=plan["frame_len"][k])
            keep = T.present_rows(planned)
            grid, truth = T.render(T.take_rows(planned, keep), plan["visible"][k][keep], plan["target"][k][keep])
            if int((truth > 0).sum()) != int(plan["target"][k].sum()):
                raise ValueError("table %d under %s: the grid scores %d cells, the evaluator planned %d"
                                 % (int(ids[at + k]), condition, int((truth > 0).sum()), int(plan["target"][k].sum())))
            if not truth.any():
                continue
            if rows and rows + grid.shape[0] > max_rows:
                flush()
            pending.append((at + k, grid, truth))
            rows += grid.shape[0]
    flush()
    model.train(was_training)
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)      # noqa: E731
    return cat(bits, np.float64), cat(cell_table, np.int64), digest.hexdigest(), cat(aa_bits, np.float64)
