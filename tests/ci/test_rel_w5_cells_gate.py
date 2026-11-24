"""W5's gate reconciles the saved cell scores and the full and subset result rows."""
import json
from pathlib import Path

import numpy as np
import pytest

from ci.gates import GateFailure
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import w5_baselines
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score
from test_rel_w5_stage import baselines_run, tables_run


@pytest.fixture
def scored_run(tmp_path, monkeypatch):
    src, tables = tables_run(tmp_path)
    run_dir, cfg = baselines_run(tmp_path, monkeypatch, src, tables)
    w5_baselines.main()
    return run_dir, cfg


def run_gate(scored_run):
    run_dir, cfg = scored_run
    with Stage(run_dir, "gates_rel_w5_baselines", requires=("pre",)) as stage:
        return w5_baselines.gate(stage, cfg, run_dir)


def test_unmodified_results_pass(scored_run):
    assert run_gate(scored_run)


@pytest.mark.parametrize("damage", ["swap_bins", "missing_aa", "missing_aa_index", "nan_aa"])
def test_rejects_changed_cells_with_matching_file_hash(scored_run, damage):
    run_dir, cfg = scored_run
    folder = Path(run_dir) / w5_baselines.OUT
    doc_path = folder / "results.json"
    doc = json.loads(doc_path.read_text())
    cells_path = folder / "cells.npz"
    with np.load(cells_path) as archive:
        arrays = dict(archive)
    condition = doc["cells"]["conditions"][cond.SCATTERED]
    baseline = condition["baselines"]["copy_nearest"]
    aa_key = baseline["amino_acid_bits"]
    if damage == "swap_bins":
        # Exchange two complete (codon, amino acid) scores in one group across two identity bins.
        # Each group's two marginal means and the headline stay exactly the same.
        held = Path(cfg[w5_baselines.STAGE]["tables_from"]) / "data/rel_w2/v0/heldout"
        identity, kind, n, col = (np.load(held / f"s0000.{name}.npy") for name in ("identity", "kind", "n", "col"))
        bins = score.fine_stratum(identity, kind, score.row_coverage(n, col))
        assert bins[0] != bins[3] and arrays["group"][0] == arrays["group"][3]
        cell_tables = arrays[condition["table"]]
        first = np.flatnonzero(cell_tables == 0)
        second = np.flatnonzero(cell_tables == 3)
        bits = arrays[baseline["bits"]]
        left, right = first[np.argmax(bits[first])], second[np.argmin(bits[second])]
        assert abs(bits[left] - bits[right]) > 1
        for key in (baseline["bits"], aa_key):
            arrays[key][[left, right]] = arrays[key][[right, left]]
    elif damage == "missing_aa":
        del arrays[aa_key]
    elif damage == "missing_aa_index":
        del arrays[aa_key]
        del baseline["amino_acid_bits"]
    else:
        arrays[aa_key][0] = np.nan
    np.savez(cells_path, **arrays)
    doc["cells"]["sha256"] = sha256_file(str(cells_path), use_sidecar=False)
    doc_path.write_text(json.dumps(doc))
    with pytest.raises(GateFailure):
        run_gate(scored_run)


@pytest.mark.parametrize(("file_name", "damage"), [
    ("rows.tsv", "missing"),
    ("rows.tsv", "duplicate"),
    ("rows_subset.tsv", "missing"),
    ("rows_subset.tsv", "wrong_bits"),
])
def test_rejects_changed_bin_rows_with_matching_count_and_hash(scored_run, file_name, damage):
    run_dir, _ = scored_run
    folder = Path(run_dir) / w5_baselines.OUT
    rows_path = folder / file_name
    rows = score.read_rows(rows_path)
    row = next(row for row in rows if row["group"] == "arch" and row["condition"] == cond.SCATTERED
               and row["variant"] == "copy_nearest" and row["stratum"] in score.FINE)
    if damage == "missing":
        rows.remove(row)
    elif damage == "duplicate":
        rows.append(row.copy())
    else:
        row["bits"] = str(float(row["bits"]) + 0.5)
    score.write_rows(rows_path, rows)
    doc_path = folder / "results.json"
    doc = json.loads(doc_path.read_text())
    index = doc["subset"] if file_name == "rows_subset.tsv" else doc
    index.update(rows=len(rows), rows_sha256=sha256_file(str(rows_path), use_sidecar=False))
    doc_path.write_text(json.dumps(doc))
    with pytest.raises(GateFailure):
        run_gate(scored_run)


def test_subset_row_in_full_file_fails_after_both_ledgers_are_updated(scored_run):
    run_dir, _ = scored_run
    folder = Path(run_dir) / w5_baselines.OUT
    doc_path = folder / "results.json"
    doc = json.loads(doc_path.read_text())
    full_path = folder / "rows.tsv"
    subset_path = folder / "rows_subset.tsv"
    full = score.read_rows(full_path)
    subset = score.read_rows(subset_path)
    idx = next(i for i, row in enumerate(subset) if row["group"] != score.ALL and row["stratum"] in score.FINE)
    moved = subset.pop(idx)
    assert moved["item_half"] == w5_baselines.SUBSET
    full.append(moved)
    for rows_path, rows, index in ((full_path, full, doc), (subset_path, subset, doc["subset"])):
        score.write_rows(rows_path, rows)
        index.update(rows=len(rows), rows_sha256=sha256_file(str(rows_path), use_sidecar=False))
    doc_path.write_text(json.dumps(doc))
    with pytest.raises(GateFailure):
        run_gate(scored_run)
