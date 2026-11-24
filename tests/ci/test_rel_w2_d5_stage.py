"""Trace the D5 baseline's permitted inputs, verify digests, and challenge its plan."""
import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
import pytest

from ci.stages.rel import w2_d5_census as stage


def write_json(path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(item).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path, monkeypatch, *, swapped=False, split_error=False, bad_anchor=False):
    run = tmp_path.resolve() / stage.TABLE_RUN
    path = run / stage.V1
    out = path / "heldout"
    out.mkdir(parents=True)
    genomes = [dict(id=f"g{i}", group=f"g{i % 12}", split="train" if i < 1093 else "heldout", taxid=i + 7)
               for i in range(1515)]
    if split_error:
        genomes[-1]["split"] = "test"
    genomes.extend(dict(id=f"i{i}", group="bacteria", split="island", taxid=i) for i in range(24))
    search = list(genomes)
    if swapped:
        search[0], search[1] = search[1], search[0]
    sfile = run.parent / stage.SEARCH_RUN / stage.SEARCH_ROSTER
    search_sha = write_json(sfile, dict(genomes=search))
    provenance = dict(run_id=stage.SEARCH_RUN, genomes_sha256=search_sha)
    row0 = []
    shards = []
    files = []
    allowed = {sfile, path / "tableset.json", path / "index.json"}
    for name, count in stage.SHARDS:
        fname = out / f"{name}.genome.npy"
        others = np.full((count, 32), -2147483648, dtype="int32")  # opaque columns after row zero
        others[:, 0] = np.arange(count) % 20 + 1093
        row0.extend(genomes[i]["id"] for i in others[:, 0])
        if bad_anchor:
            others[-1, 0] = 0
        np.save(fname, others)
        files.append(dict(path=str(fname.relative_to(run)), sha256=hashlib.sha256(fname.read_bytes()).hexdigest()))
        shards.append(dict(name=name, tables=count))
        allowed.add(fname)
    doc = dict(files=files, producer=dict(run_id=stage.TABLE_RUN), search=provenance)
    ix = dict(genomes=genomes, rows=32, parts={"heldout": dict(tables=3000, shards=shards)}, search=provenance)
    monkeypatch.setattr(stage, "ROW0_SHA", stage.sha256_json(row0))
    monkeypatch.setattr(stage, "SEARCH_SHA", search_sha)
    monkeypatch.setattr(stage, "TABLESET_SHA", write_json(path / "tableset.json", doc))
    monkeypatch.setattr(stage, "INDEX_SHA", write_json(path / "index.json", ix))
    return dict(rel_w2_d5_census=dict(tables_from=str(run))), allowed


def test_inventory_opens_only_five_pinned_inputs(tmp_path, monkeypatch):
    cfg, allowed = fixture(tmp_path, monkeypatch)
    opened = set()
    def guard(original):
        def checked(file, *args, **kwargs):
            if not isinstance(file, (str, os.PathLike)) or Path(file) not in allowed:
                raise AssertionError(f"unexpected read: {file}")
            opened.add(Path(file))
            return original(file, *args, **kwargs)
        return checked

    with monkeypatch.context() as m:
        m.setattr(builtins, "open", guard(builtins.open))
        m.setattr(io, "open", guard(io.open))
        rep = stage.inventory(cfg)
    assert opened == allowed
    assert rep["heldout_tables"] == 3000 and rep["census"]["distinct_row0"] == 20
    assert rep["census"]["units"] is None


@pytest.mark.parametrize("key", ["TABLESET_SHA", "INDEX_SHA", "SEARCH_SHA"])
def test_bad_pinned_digest_fails(tmp_path, monkeypatch, key):
    cfg, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(stage, key, "0" * 64)
    with pytest.raises(ValueError, match="checksum"):
        stage.inventory(cfg)


def test_same_roster_with_different_row0_fails(tmp_path, monkeypatch):
    cfg, _ = fixture(tmp_path, monkeypatch)
    shard = Path(cfg["rel_w2_d5_census"]["tables_from"]) / stage.V1 / "heldout/s0001.genome.npy"
    matrix = np.load(shard, mmap_mode="r+")
    matrix[0, 0] += 1
    matrix.flush()
    with pytest.raises(ValueError, match="row-zero target checksum"):
        stage.inventory(cfg)


def test_other_columns_are_opaque_and_the_declared_digests_are_labelled(tmp_path, monkeypatch):
    cfg, _ = fixture(tmp_path, monkeypatch)
    shard = Path(cfg["rel_w2_d5_census"]["tables_from"]) / stage.V1 / "heldout/s0001.genome.npy"
    recorded_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    matrix = np.load(shard, mmap_mode="r+")
    matrix[0, 1] = 21
    matrix.flush()
    assert hashlib.sha256(shard.read_bytes()).hexdigest() != recorded_sha
    result = stage.inventory(cfg)
    assert result["row0_plan_sha256"] == stage.ROW0_SHA
    assert recorded_sha in result["shard_file_sha256_as_recorded"].values()


@pytest.mark.parametrize("problem, reason", [
    ("swapped", "ordered table and search rosters differ"),
    ("split_error", "unexpected split"),
    ("bad_anchor", "outside the heldout split"),
])
def test_wrong_order_split_or_target_fails(tmp_path, monkeypatch, problem, reason):
    cfg, _ = fixture(tmp_path, monkeypatch, **{problem: True})
    with pytest.raises(ValueError, match=reason):
        stage.inventory(cfg)


def test_unlisted_input_cannot_be_enabled(tmp_path):
    with pytest.raises(ValueError, match="tables_from input only"):
        stage.inventory(dict(rel_w2_d5_census=dict(tables_from=str(tmp_path), dist_from="other")))
    with pytest.raises(ValueError, match="baseline run"):
        stage.inventory(dict(rel_w2_d5_census=dict(tables_from=str(tmp_path))))


def test_rejects_redirected_shard(tmp_path, monkeypatch):
    cfg, _ = fixture(tmp_path, monkeypatch)
    original = Path(cfg["rel_w2_d5_census"]["tables_from"]) / stage.V1 / "heldout/s0000.genome.npy"
    redirected = tmp_path / "unlisted"
    original.rename(redirected)
    original.symlink_to(redirected)
    with pytest.raises(ValueError, match="canonical"):
        stage.inventory(cfg)
