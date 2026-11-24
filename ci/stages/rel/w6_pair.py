"""Verify the denominator and the strict-online control of the round-1 paired report."""

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from seeddrift.train.rel.evalh import conditions as cond


CONDITIONS = (cond.SCATTERED, cond.WHOLE_GENE, cond.SUFFIX)
ONLINE = (cond.SCATTERED, cond.SUFFIX)


def planned_cells(tables):
    """Rebuild mask positions and their digest on the report machine's copy of split 1."""
    counts = {c: np.zeros(len(tables), dtype=np.int64) for c in CONDITIONS}
    digests = {c: cond.ItemsDigest(c) for c in CONDITIONS}
    for start in range(0, len(tables), 32):
        stop = min(start + 32, len(tables))
        data = tables.load(("tok", "n", "col", "kind"), start, stop)
        for c in CONDITIONS:
            plan = cond.materialise(data, np.arange(start, stop), c)
            digests[c].add(plan)
            counts[c][start:stop] = plan["target"].sum(axis=(1, 2))
    return counts, {c: digest.hexdigest() for c, digest in digests.items()}


def verify(tables, expected_tables, first, second, online):
    """Return an audit row per condition or reject the comparison before it produces a verdict."""
    entries = (first, second, online)
    if len(tables) != expected_tables or any(entry["tables"] != expected_tables for entry in entries):
        raise ValueError("paired ends: held-out table denominator disagrees with the profile")
    if not second["checkpoint"] or second["checkpoint"] != online["checkpoint"]:
        raise ValueError("paired ends: strict and online final checkpoint digests differ or are missing")

    counts, digests = planned_cells(tables)
    results = []
    with ExitStack() as files:
        scores = [files.enter_context(np.load(Path(entry["folder"]) / "heldout_final.npz", allow_pickle=False))
                  for entry in entries]
        for c in CONDITIONS:
            expected = np.repeat(np.arange(expected_tables), counts[c])
            if not len(expected):
                raise ValueError(f"paired ends: no planned cells for {c}")
            losses = []
            n_entries = 3 if c in ONLINE else 2
            for entry, score in zip(entries[:n_entries], scores[:n_entries]):
                if entry["final_items"].get(c) != digests[c]:
                    raise ValueError(f"paired ends: {c} target digest differs from the held-out mask plan in {entry['folder']}")
                ids = score[f"table_{c}"]
                bits = score[f"bits_{c}"]
                if ids.dtype.kind not in "iu" or not np.array_equal(ids, expected):
                    raise ValueError(f"paired ends: {c} per-table cell denominator or order differs in {entry['folder']}")
                if bits.shape != expected.shape or not np.isfinite(bits).all():
                    raise ValueError(f"paired ends: missing or non-finite {c} cell scores in {entry['folder']}")
                losses.append(bits)

            row = dict(condition=c, heldout_tables=expected_tables, scored_tables=int(np.count_nonzero(counts[c])),
                       scored_cells=len(expected), items_sha256=digests[c])
            if c in ONLINE:
                delta = losses[2] - losses[1]
                max_delta = float(np.max(np.abs(delta)))
                if max_delta >= 0.0005:
                    raise ValueError(f"paired ends: {c} strict-online cell difference {max_delta:.9g} exceeds zero at report precision")
                row.update(online_minus_offline_mean=float(np.mean(delta)), max_abs_cell_delta=max_delta,
                           nonzero_cell_differences=int(np.count_nonzero(delta)))
            results.append(row)
    return results
