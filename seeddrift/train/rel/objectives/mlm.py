"""Objective "mlm": bidirectional fill-in. Every hidden residue (a token 1..65 of its own row) is predicted on its own
from the grid; the loss is the cross-entropy over those 65 tokens, computed in float32. When the batch carries a weight
per target cell (a weighted mask scheme such as anyorder), the summed loss is weighted by it; the loss of every cell is
returned unweighted, for the logs."""
from __future__ import annotations

import torch.nn.functional as F

from seeddrift.train.rel import api as A

NAME = "mlm"
FACTORISATION = "independent"


def loss(model, batch):
    """batch: grid long [N, C], sizes, and for the target cells row, col and token (1..65). Returns the summed loss in
    nats and the loss of every target cell (float32 [M], in the order of the batch's cells)."""
    extra = {k: batch[k] for k in ("ext", "genome") if k in batch}   # cells laid by extension; the genome of every row
    logits = model(batch["grid"], batch["sizes"], batch["row"], batch["col"], **extra)
    per = F.cross_entropy(logits, batch["token"] - A.TARGET_MIN, reduction="none")
    w = batch.get("weight")
    return (per.sum() if w is None else (per * w).sum()), per
