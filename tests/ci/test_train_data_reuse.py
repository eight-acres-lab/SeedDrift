"""Stage train_data taken over whole from a run with the same key (ci/stages/train_data_or_reuse.py), and the
manifest-only search reuse it allows (ci/stages/search_or_reuse.py)."""
import json
import os
import shutil
import sys

import numpy as np
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ci import manifest as M  # noqa: E402
from ci.hashing import sha256_file  # noqa: E402
from ci.stages import search_or_reuse as SR  # noqa: E402
from ci.stages import train_data_or_reuse as R  # noqa: E402


def _write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj))


def _run(root, name, train_data=None, search=None, pipeline="selfsup", code_root=None):
    """A run folder with the four dataset manifests, its record and configuration, the gates the stage needs, and the
    code of its worktree."""
    run = root / "runs" / name
    for d in ("gates", "stages"):
        (run / d).mkdir(parents=True, exist_ok=True)
    for g in ("pre", "corpus"):
        (run / "gates" / (g + ".ok")).write_text("")
    for ds, sha in (("corpus", "c1"), ("eval_inputs", "e1"), ("reference", "r1"), ("hits", "h1")):
        _write(run / "data" / ds / "manifest.json", dict(sha256=sha, entries=[]))
    _write(run / "record.json", dict(taxdump_sha256="t1", env_snapshot={"diamond_version.txt": "d1"}))
    cfg = dict(pipeline=pipeline, train_data=dict(train_data or dict(hits=32)), search=dict(search or dict(sensitivity="sensitive", nearest=32)))
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    code = root / "wt" / name
    for c in list(R.CODE) + sorted({f for f, _ in R.FUNCS}):
        (code / c).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(code_root or ROOT, c), code / c)
    return run


def _with_train_data(run):
    """Training data as stage train_data writes it: arrays, the content-addressed manifest, the index naming it, and a
    passed stage report with the index's sha256."""
    entries = []
    for gid in ("G1", "G2"):
        p = run / "data" / "train_data" / (gid + ".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, codons=np.arange(10, dtype=np.uint8) + len(gid))
        entries.append(dict(id=gid, files=[M.file_entry(str(p), str(run))]))
    man = M.build("train-data", "train_data", entries, meta=dict(max_codons=1000, hits=32))
    sha = M.write(man, str(run / "manifests"))
    ip = run / "data" / "train_data" / "index.json"
    _write(ip, dict(sha256=sha, genomes=[dict(id="G1", hits=3, split="train"), dict(id="G2", hits=0, split="heldout")]))
    _write(run / "stages" / "train_data.json", dict(status="passed", outputs=[dict(name="train_data_index", sha256=sha256_file(str(ip), use_sidecar=False))]))
    return run


def _main(monkeypatch, run, mod):
    monkeypatch.setattr(sys, "argv", ["x", str(run)])
    mod.main()


def test_train_data_is_taken_whole_from_a_run_with_the_same_key(tmp_path, monkeypatch):
    old = _with_train_data(_run(tmp_path, "old"))
    new = _run(tmp_path, "new", train_data=dict(hits=32, reuse_from=[str(old)], reuse_required=True))
    _main(monkeypatch, new, R)
    rep = json.loads((new / "stages" / "train_data.json").read_text())
    assert rep["status"] == "passed" and rep["reused_from"] == "old" and rep["genomes"] == 2 and rep["with_hits"] == 1
    assert (new / "data" / "train_data" / "index.json").read_bytes() == (old / "data" / "train_data" / "index.json").read_bytes()
    assert os.stat(new / "data" / "train_data" / "G1.npz").st_ino == os.stat(old / "data" / "train_data" / "G1.npz").st_ino
    idx = json.loads((new / "data" / "train_data" / "index.json").read_text())
    assert M.digest(M.load(str(new / "manifests" / ("%s.json" % idx["sha256"])))) == idx["sha256"]


@pytest.mark.parametrize("change", ["settings", "corpus", "eval_inputs", "reference", "hits", "taxdump", "code", "map", "helper", "file"])
def test_train_data_reuse_refuses_any_difference(tmp_path, monkeypatch, change):
    old = _with_train_data(_run(tmp_path, "old"))
    new = _run(tmp_path, "new", train_data=dict(hits=16 if change == "settings" else 32, reuse_from=[str(old)], reuse_required=True))
    if change in ("corpus", "eval_inputs", "reference", "hits"):
        _write(new / "data" / change / "manifest.json", dict(sha256="x2", entries=[]))
    if change == "taxdump":
        _write(new / "record.json", dict(taxdump_sha256="t2"))
    if change == "code":
        p = tmp_path / "wt" / "old" / "ci" / "stages" / "train_data.py"
        p.write_text(p.read_text() + "\n# changed\n")
    if change == "map":                                  # the kingdom map sets the groups, so the split
        p = tmp_path / "wt" / "old" / "configs" / "kingdom_map.yml"
        p.write_text(p.read_text() + "\n# changed\n")
    if change == "helper":                               # a helper whose behaviour shapes the written bytes
        p = tmp_path / "wt" / "old" / "seeddrift" / "train" / "logger.py"
        p.write_text(p.read_text().replace("date_time=(1980, 1, 1, 0, 0, 0)", "date_time=(1981, 1, 1, 0, 0, 0)"))
    if change == "file":                                 # a training-data file that no longer matches its manifest
        (old / "data" / "train_data" / "G2.npz").write_bytes(b"x")
        with pytest.raises(SystemExit, match="does not match its manifest"):
            _main(monkeypatch, new, R)
        return
    with pytest.raises(SystemExit, match="reuse_required"):
        _main(monkeypatch, new, R)
    assert not (new / "data" / "train_data" / "index.json").exists()


def test_helper_changes_elsewhere_in_a_shared_module_keep_the_key(tmp_path):
    """Only the listed functions of the shared modules enter the key: another function of ci/stages/common.py may change."""
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    p = tmp_path / "wt" / "b" / "ci" / "stages" / "common.py"
    p.write_text(p.read_text() + "\n\ndef unrelated():\n    return 1\n")
    cfg = yaml.safe_load((a / "config.resolved.yaml").read_text())
    assert R.key(str(a), cfg, str(tmp_path / "wt" / "a")) == R.key(str(b), cfg, str(tmp_path / "wt" / "b"))
    assert R.function_source(str(p), "read_fasta").startswith("def read_fasta")


def test_train_data_reuse_needs_a_passed_stage_with_its_index(tmp_path):
    old = _with_train_data(_run(tmp_path, "old"))
    assert R.candidate_ok(str(old))
    rep = json.loads((old / "stages" / "train_data.json").read_text())
    rep["status"] = "failed"
    _write(old / "stages" / "train_data.json", rep)
    assert R.candidate_ok(str(old)) is None
    _with_train_data(old)
    _write(old / "data" / "train_data" / "index.json", dict(sha256="0" * 64, genomes=[]))        # an index the stage did not report
    assert R.candidate_ok(str(old)) is None


def _search_candidate(root, name):
    """A passed search run whose hits manifest names hit files that are not there (they live on another machine)."""
    run = _run(root, name)
    ents = [dict(id="G1", files=[dict(path="data/hits/G1.npz", sha256="a" * 64, bytes=1)])]
    _write(run / "data" / "hits" / "manifest.json", dict(sha256="h1", **M.build("hits", "hits", ents)))
    _write(run / "data" / "corpus" / "manifest.json", dict(sha256="c1", entries=[dict(id="G1", files=[dict(path="data/corpus/G1/genes.tsv", sha256="b" * 64)])]))
    _write(run / "stages" / "search.json", dict(status="passed", outputs=[dict(name="hits_manifest", sha256=sha256_file(str(run / "data" / "hits" / "manifest.json"), use_sidecar=False))]))
    code = root / "wt" / name / "ci" / "stages"
    code.mkdir(parents=True, exist_ok=True)
    shutil.copy(SR.SEARCH_CODE, code / "search.py")
    return run


def test_manifest_only_search_reuse(tmp_path, monkeypatch):
    cand = _search_candidate(tmp_path, "cand")
    # not allowed without a required training-data reuse, nor outside pipeline selfsup
    for td, pipe in ((dict(hits=32), "selfsup"), (dict(hits=32, reuse_from=["x"], reuse_required=True), "train")):
        new = _run(tmp_path, "new-%s-%d" % (pipe, len(td)), train_data=td, pipeline=pipe,
                   search=dict(sensitivity="sensitive", nearest=32, reuse_from=[str(cand)], reuse_manifest_only=True))
        _write(new / "data" / "corpus" / "manifest.json", dict(sha256="c1", entries=[dict(id="G1", files=[dict(path="data/corpus/G1/genes.tsv", sha256="b" * 64)])]))
        with pytest.raises(SystemExit, match="reuse_manifest_only needs"):
            _main(monkeypatch, new, SR)
    new = _run(tmp_path, "new", train_data=dict(hits=32, reuse_from=["x"], reuse_required=True),
               search=dict(sensitivity="sensitive", nearest=32, reuse_from=[str(cand)], reuse_manifest_only=True))
    _write(new / "data" / "corpus" / "manifest.json", dict(sha256="c1", entries=[dict(id="G1", files=[dict(path="data/corpus/G1/genes.tsv", sha256="b" * 64)])]))
    (new / "data" / "hits" / "manifest.json").unlink()
    _main(monkeypatch, new, SR)
    rep = json.loads((new / "stages" / "search.json").read_text())
    assert rep["status"] == "passed" and rep["manifest_only"] is True and rep["reused_from"] == "cand"
    assert json.loads((new / "data" / "hits" / "manifest.json").read_text()) == json.loads((cand / "data" / "hits" / "manifest.json").read_text())
    assert not (new / "data" / "hits" / "G1.npz").exists()
    # the reuse keys are not part of the search key: the candidate (without them) and this run match
    ccfg = yaml.safe_load((cand / "config.resolved.yaml").read_text())
    ncfg = yaml.safe_load((new / "config.resolved.yaml").read_text())
    assert SR.key(str(cand), ccfg["search"], SR.SEARCH_CODE) == SR.key(str(new), ncfg["search"], SR.SEARCH_CODE)
    # a different search setting finds no candidate, and manifest-only never searches: the stage fails
    other = _run(tmp_path, "other", train_data=dict(hits=32, reuse_from=["x"], reuse_required=True),
                 search=dict(sensitivity="fast", nearest=32, reuse_from=[str(cand)], reuse_manifest_only=True))
    _write(other / "data" / "corpus" / "manifest.json", dict(sha256="c1", entries=[dict(id="G1", files=[dict(path="data/corpus/G1/genes.tsv", sha256="b" * 64)])]))
    with pytest.raises(SystemExit, match="no run of search.reuse_from"):
        _main(monkeypatch, other, SR)
    # training data is never built from absent hit files
    (new / "config.resolved.yaml").write_text(yaml.safe_dump(dict(ncfg, train_data=dict(hits=32))))
    with pytest.raises(SystemExit, match="hit files are absent"):
        _main(monkeypatch, new, R)
