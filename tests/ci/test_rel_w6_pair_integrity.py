"""W6's strict/online pair on small held-out shards and the final files written by trainer.loop."""
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import pytest

from ci.stages.common import Stage
from ci.stages.rel import w6_fit, w6_pair, w6_report
from seeddrift.train.rel import api
from seeddrift.train.rel.data import tables as T
from seeddrift.train.rel.evalh import conditions as cond
from test_rel_w6_data import v1_tables, write_tableset
from test_rel_w6_stages import _finished_fit


@pytest.fixture
def pair(tmp_path):
    size = 5
    arrays = v1_tables(size, rows=5, seed=83, split=1)
    root = tmp_path / "i1" / "heldout"
    manifest = write_tableset(root, [("heldout/s0000", arrays)], kind=api.TABLE_REAL)
    tables = T.open_split(root, manifest, 1)
    entries = []
    for name in ("reference", "strict", "strict.online"):
        folder = tmp_path / ("runs" if name == "reference" else "imports/runs") / name
        folder.mkdir(parents=True)
        items, files = {}, {}
        conditions = [cond.SCATTERED, cond.SUFFIX]
        if name != "strict.online":
            conditions.append(cond.WHOLE_GENE)
        for condition in conditions:
            plan = cond.materialise(arrays, np.arange(size), condition)
            digest = cond.ItemsDigest(condition)
            digest.add(plan)
            items[condition] = digest.hexdigest()
            counts = plan["target"].sum(axis=(1, 2))
            ids = np.repeat(np.arange(size), counts)
            files["table_" + condition] = ids
            # Arbitrary finite losses, with both between-model and between-cell variation.
            files["bits_" + condition] = 3.0 + (name == "reference") + np.arange(len(ids)) / 1000
        np.savez(folder / "heldout_final.npz", **files)
        entries.append(dict(folder=str(folder), final_items=items, tables=size,
                            checkpoint=("1" if name == "reference" else "2") * 64))
    return tables, size, *entries


def change_cells(entry, change):
    path = Path(entry["folder"]) / "heldout_final.npz"
    with np.load(path) as saved:
        files = {key: saved[key] for key in saved.files}
    change(files)
    np.savez(path, **files)


def test_complete_pair_accepts_imported_evaluations_and_distinct_model_scores(pair):
    audit = {r["condition"]: r for r in w6_pair.verify(*pair)}
    assert set(audit) == {cond.SCATTERED, cond.SUFFIX, cond.WHOLE_GENE}
    for condition, row in audit.items():
        assert row["heldout_tables"] == row["scored_tables"] == pair[1]
        with np.load(Path(pair[2]["folder"]) / "heldout_final.npz") as scored:
            assert row["scored_cells"] == len(scored["table_" + condition])
        if condition != cond.WHOLE_GENE:
            assert row["max_abs_cell_delta"] == 0


@pytest.mark.parametrize("entries", [(2,), (2, 3, 4)], ids=["only-reference", "all-three"])
def test_rejects_a_missing_table_even_if_every_input_lost_it(pair, entries):
    def remove_last_table(files):
        keep = files["table_scattered"] != pair[1] - 1
        assert not keep.all()
        for field in ("bits_scattered", "table_scattered"):
            files[field] = files[field][keep]

    for index in entries:
        change_cells(pair[index], remove_last_table)
    with pytest.raises((ValueError, SystemExit)):
        w6_pair.verify(*pair)


def test_rejects_equal_counts_but_different_scored_positions(pair):
    # A run with a different target residue has the same number of scored cells.
    # All three arms can agree with each other and still disagree with the local held-out set.
    tables, size, *entries = pair
    changed = tables.load(("tok", "n", "col", "kind", "frame_len", "frame_row", "genome_of_row"))
    plan = cond.materialise(changed, np.arange(size), cond.SCATTERED)
    plan["tok"] = plan["tok"].copy()
    cell = tuple(np.argwhere(plan["target"])[0])
    plan["tok"][cell] = 1 if plan["tok"][cell] != 1 else 2
    digest = cond.ItemsDigest(cond.SCATTERED)
    digest.add(plan)
    for entry in entries:
        assert digest.hexdigest() != entry["final_items"][cond.SCATTERED]
        entry["final_items"][cond.SCATTERED] = digest.hexdigest()
    with pytest.raises((ValueError, SystemExit)):
        w6_pair.verify(*pair)


def test_rejects_cell_errors_that_cancel_within_one_table(pair):
    def change_two_scores(files):
        indices = np.flatnonzero(files["table_suffix"] == 0)
        assert len(indices) >= 2
        files["bits_suffix"][indices[:2]] += [0.125, -0.125]

    change_cells(pair[-1], change_two_scores)
    with pytest.raises((ValueError, SystemExit)):
        w6_pair.verify(*pair)


def test_rejects_online_scores_from_another_checkpoint(pair):
    pair[-1]["checkpoint"] = pair[2]["checkpoint"]
    with pytest.raises((ValueError, SystemExit)):
        w6_pair.verify(*pair)


@pytest.mark.parametrize("delta,accepted", [(0.0004, True), (0.0006, False)])
def test_strict_zero_precision_is_applied_to_each_cell(pair, delta, accepted):
    def shift_one_score(files):
        files["bits_scattered"][0] += delta

    change_cells(pair[-1], shift_one_score)
    if accepted:
        audit = {r["condition"]: r for r in w6_pair.verify(*pair)}
        assert audit[cond.SCATTERED]["max_abs_cell_delta"] == pytest.approx(delta)
    else:
        with pytest.raises(ValueError, match="scattered"):
            w6_pair.verify(*pair)


def test_build_reads_local_split_after_copy_back(pair, tmp_path, monkeypatch):
    """Imported arm configs name i-2 paths. Exercise the report's entry point with local split-1 shards."""
    tables, size, *entries = pair
    run_dir, config, index, _ = _finished_fit(tmp_path / "report", statuses=("ok", "failed", "ok"))
    fit = Path(run_dir) / w6_fit.OUT
    names = ["w6-ref30m-s1", "w6-ref10m-s1", "w6-ref10m-s1.online"]
    variants = {v["name"]: v for v in index["variants"]}
    variants[names[2]] = dict(variants[names[1]], name=names[2])
    for name, entry in zip(names, entries):
        folder = fit / name
        folder.mkdir(exist_ok=True)
        copyfile(Path(entry["folder"]) / "heldout_final.npz", folder / "heldout_final.npz")
        (folder / "conf.json").write_text(json.dumps(dict(heldout_root="/i2-only",
                                                         heldout_tables="/i2-only/tableset.json")))
        render = "online" if name.endswith(".online") else "offline"
        variants[name]["info"] = dict(variants[name]["info"], folder=str(folder.relative_to(run_dir)),
                                      render=render, eval_final=size, final_items=entry["final_items"],
                                      evaluated_again=dict(checkpoints=dict(final=entry["checkpoint"])))
    (fit / "models.json").write_text(json.dumps(dict(index, evaluated_on=dict(sha256=tables.sha256, split=1), variants=list(variants.values()))))

    local = tmp_path / "i1/heldout"
    def units(conf, source):
        assert Path(conf["heldout_root"]) == local
        assert Path(conf["heldout_tables"]) == local / "tables/tableset.json"
        return ["bacteria"] * len(tables), [f"unit-{i}" for i in range(len(tables))]

    monkeypatch.setattr(w6_report, "held_units", units)
    config["rel_w6_report"] = dict(
        heldout_tables=dict(run=str(local), path="tables/tableset.json"), units_source=str(local),
        paired_ends=[dict(name="layout pair", first=names[0], second=names[1], threshold=0.009,
                          strict_online_tables=size)])
    def build():
        return w6_report.build(Stage(run_dir, "rel_w6_report"), config, run_dir)

    comparison = build()["paired_ends"][0]
    assert len(comparison["integrity"]) == 3

    # The same files pass the original intersection-only comparison after losing this
    # table. Build must reject before it returns paired rows from that subset.
    path = fit / names[1] / "heldout_final.npz"
    with np.load(path) as saved:
        damaged = {key: saved[key] for key in saved.files}
    keep = damaged["table_whole_gene"] != size - 1
    for key in ("table_whole_gene", "bits_whole_gene"):
        damaged[key] = damaged[key][keep]
    np.savez(path, **damaged)
    with pytest.raises(ValueError, match="whole_gene"):
        build()
