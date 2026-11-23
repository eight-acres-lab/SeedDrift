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
from seeddrift.train.rel.tables import align as AL

EVAL_VERSION = "5"
PLAN_FIELDS = ("tok", "n", "col", "kind", "frame_len", "frame_row", "genome_of_row")


def to_device(batch, device):
    out = dict(sizes=batch["sizes"])
    for k in ("grid", "row", "col", "token", "example"):
        out[k] = torch.from_numpy(batch[k]).to(device, non_blocking=True).long()
    if "weight" in batch:
        out["weight"] = torch.from_numpy(batch["weight"]).to(device, non_blocking=True).float()
    if "ext" in batch:
        out["ext"] = torch.from_numpy(batch["ext"]).to(device, non_blocking=True).bool()
    if "genome" in batch:
        out["genome"] = torch.from_numpy(batch["genome"]).to(device, non_blocking=True).long()
    return out


def ext_of(batch):
    """The keywords a model reads the cells laid by extension and the genome of every row with: none for a batch
    without them, so a model that does not know the keywords is called as before."""
    return {k: batch[k] for k in ("ext", "genome") if k in batch}


@torch.no_grad()
def score_cells(model, examples, device, max_rows=32):
    """The target cells of grids rendered elsewhere (the passes of package W18's readouts): examples [dict(grid uint8
    [R, ROW_LEN], truth uint8 [R, ROW_LEN] (0: not scored), ext optional)], scored in micro-batches of at most max_rows
    rows. Returns dict of numpy arrays, one entry per scored cell in the order of numpy.nonzero(truth) of every example,
    examples in order: bits (-log2 p(truth)), aa_bits (-log2 of the probability of the truth's class: the evaluator's
    20 amino acids, the stops, OTHER), entropy (of the distribution over the residues 1..65, in bits), example."""
    classes = torch.from_numpy(SC.amino_acid_classes()[1:66]).to(device)
    onehot = torch.nn.functional.one_hot(classes, int(classes.max()) + 1).double()
    out = dict(bits=[], aa_bits=[], entropy=[], example=[])
    was_training = model.training
    model.eval()
    rows, pending = 0, []

    def flush():
        nonlocal pending, rows
        if pending:
            b = to_device(S.collate([examples[k] for k in pending]), device)
            if len(b["row"]):
                logp = model.logp(b["grid"], b["sizes"], b["row"], b["col"], **ext_of(b)).double()
                idx = (b["token"] - A.TARGET_MIN)[:, None]
                out["bits"].append((-logp.gather(1, idx)[:, 0] / math.log(2.0)).cpu().numpy())
                p_class = logp.exp() @ onehot
                out["aa_bits"].append((-torch.log2(p_class.gather(1, classes[idx[:, 0]][:, None])[:, 0])).cpu().numpy())
                out["entropy"].append((-(logp.exp() * logp).sum(dim=1) / math.log(2.0)).cpu().numpy())
                out["example"].append(np.asarray(pending, dtype=np.int64)[b["example"].cpu().numpy()])
        pending, rows = [], 0

    for k, e in enumerate(examples):
        r = int(np.asarray(e["grid"]).shape[0])
        if pending and rows + r > max_rows:
            flush()
        pending.append(k)
        rows += r
    flush()
    model.train(was_training)
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)      # noqa: E731
    return dict(bits=cat(out["bits"], np.float64), aa_bits=cat(out["aa_bits"], np.float64), entropy=cat(out["entropy"], np.float64),
                example=cat(out["example"], np.int64))


def subset(tables, per_group):
    """The evaluator's subset (conditions.subset): the first per_group tables of every group, in storage order; 0: all
    tables. Positions in the table set."""
    if not per_group:
        return np.arange(len(tables), dtype=np.int64)
    return np.asarray(cond.subset(tables.group, int(per_group)), dtype=np.int64)


def conditions_of(relatives, extra=(), only=None):
    """The conditions an arm is scored under: today's five (cond.CONDITIONS) and the evaluator's extra conditions named
    in `extra` (cond.EXTRA_CONDITIONS, e.g. suffix), or only those named in `only`; an arm trained without relatives
    is read only without them."""
    allowed = list(cond.CONDITIONS) + list(getattr(cond, "EXTRA_CONDITIONS", ()))
    named = list(only) if only else list(cond.CONDITIONS) + [c for c in extra if c not in cond.CONDITIONS]
    bad = [c for c in named if c not in allowed]
    if bad:
        raise ValueError("conditions %s are not the evaluator's (%s)" % (bad, allowed))
    return [c for c in named if relatives or cond.parse(c)[1] == cond.NO_RELATIVES]


def other_rows(kind):
    """(N,) the number of other rows every table has (kind (N, R, B))."""
    return cond.present_rows(kind)[:, 1:].sum(axis=1)


@torch.no_grad()
def score(model, data, ids, condition, device, max_rows=64, chunk=32, layout=A.LAYOUT_STORED, frame=None, control=None):
    """The tables ids (positions in the held-out set; data: their arrays PLAN_FIELDS, in the same order) under one
    condition, rendered in `layout` (stored: tables.render; another: package W3's renderer, with the cells it laid by
    extension where the model reads them). Returns bits (float64 [C], one per scored cell), cell_table (int64 [C], the
    index into ids of the cell's table), the digest of what was scored, and the bits of the amino acid of every scored
    cell (the first of the two parts of the evaluator's score.amino_acid_bits; reporting only, the genetic code never
    reaches the model). frame: None; {strict: True, cache} (a whole-gene condition in a layout other than stored laid
    on its frame relative by package W3's strict aligner, align.frame_layout, as the training lays it; cache, a dict
    or None, keeps the layout of every (table id, condition)); or {index, meta} (the pair index of the uncapped search
    and the tables' meta fields genome, row_gene, row_first, in the order of data): placed online by the stored pairs
    (package W3's align.frame_layout_online).
    control: None, or a control of W3's renderer (shuffle: the rows' correspondence permuted by column; presence:
    only which cells exist), keyed by the table's id."""
    if layout != A.LAYOUT_STORED:
        from seeddrift.train.rel.tables import render as RD
        with_ext = getattr(model, "ext", None) is not None
    mask = cond.parse(condition)[0]
    digest = cond.ItemsDigest(condition)
    bits, cell_table, aa_bits = [], [], []
    pending, rows = [], 0
    classes = torch.from_numpy(SC.amino_acid_classes()[1:66]).to(device)                      # class of every token 1..65
    onehot = torch.nn.functional.one_hot(classes, int(classes.max()) + 1).double()

    def flush():
        nonlocal pending, rows
        if pending:
            b = to_device(S.collate([dict(dict(grid=g, truth=t), **({} if x is None else dict(ext=x)), **({} if gr is None else dict(genome_of_row=gr)))
                                     for _, g, t, x, gr in pending]), device)
            logp = model.logp(b["grid"], b["sizes"], b["row"], b["col"], **ext_of(b))
            per = -logp.gather(1, (b["token"] - A.TARGET_MIN)[:, None])[:, 0].double() / math.log(2.0)
            bits.append(per.cpu().numpy())
            p_class = logp.double().exp() @ onehot
            aa_bits.append((-torch.log2(p_class.gather(1, classes[b["token"] - A.TARGET_MIN][:, None])[:, 0])).cpu().numpy())
            cell_table.append(np.array([k for k, _, _, _, _ in pending], dtype=np.int64)[b["example"].cpu().numpy()])
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
            planned = dict(tok=plan["tok"][k], n=plan["n"][k], col=plan["col"][k], kind=plan["kind"][k], frame_len=plan["frame_len"][k],
                           frame_row=plan["frame_row"][k])
            keep = T.present_rows(planned)
            gor = None
            if "genome_of_row" in plan:                                # a row keeps its genome; one row left: row 0's
                g_all = np.asarray(plan["genome_of_row"][k]).astype(np.int64)
                rows_k = np.asarray(plan["rows"][k])
                g_k = g_all[:len(rows_k)] if len(rows_k) == len(g_all) else g_all[rows_k]
                gor = g_k[keep]
            ext = None
            if layout == A.LAYOUT_STORED:
                grid, truth = T.render(T.take_rows(planned, keep), plan["visible"][k][keep], plan["target"][k][keep])
            else:
                tab = T.take_rows(planned, keep)
                if frame is not None and frame.get("strict") and mask == cond.WHOLE_GENE and cond.NO_RELATIVES not in condition:
                    ck = (int(ids[at + k]), condition)                 # W3's strict aligner on the rows scored, once per table
                    if frame.get("cache") is None or ck not in frame["cache"]:
                        got = AL.frame_layout(tab)
                        if frame.get("cache") is not None:
                            frame["cache"][ck] = got
                    col, frame_row, frame_len = got if frame.get("cache") is None else frame["cache"][ck]
                    tab = dict(tab, col=col, frame_row=frame_row, frame_len=frame_len)
                elif frame is not None and mask == cond.WHOLE_GENE and cond.NO_RELATIVES not in condition:    # the rows stay: their meta holds
                    meta_k = {f: np.asarray(frame["meta"][f][at + k]) for f in T.FRAME_META}
                    col, frame_row, frame_len, _ = AL.frame_layout_online(tab, T.take_meta(meta_k, keep), frame["index"])
                    tab = dict(tab, col=col, frame_row=frame_row, frame_len=frame_len)
                grid, truth, ext = RD.render(tab, plan["visible"][k][keep], plan["target"][k][keep], layout=layout, control=control,
                                             key=int(ids[at + k]))
                ext = ext if with_ext else None
            if int((truth > 0).sum()) != int(plan["target"][k].sum()):
                raise ValueError("table %d under %s: the grid scores %d cells, the evaluator planned %d"
                                 % (int(ids[at + k]), condition, int((truth > 0).sum()), int(plan["target"][k].sum())))
            if not truth.any():
                continue
            if rows and rows + grid.shape[0] > max_rows:
                flush()
            pending.append((at + k, grid, truth, ext, gor))
            rows += grid.shape[0]
    flush()
    model.train(was_training)
    cat = lambda parts, dtype: np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)      # noqa: E731
    return cat(bits, np.float64), cat(cell_table, np.int64), digest.hexdigest(), cat(aa_bits, np.float64)
