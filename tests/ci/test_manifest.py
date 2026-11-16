import pytest

from ci import manifest as M


def test_digest_does_not_depend_on_entry_order():
    a = M.build("x", "corpus", [{"id": "b", "taxid": 2}, {"id": "a", "taxid": 1}])
    b = M.build("x", "corpus", [{"id": "a", "taxid": 1}, {"id": "b", "taxid": 2}])
    assert M.digest(a) == M.digest(b) and M.dataset_id(a).startswith("ds-x-")


def test_duplicate_ids_fail():
    with pytest.raises(M.ManifestError):
        M.build("x", "corpus", [{"id": "a"}, {"id": "a"}])


def test_verify_detects_a_changed_file(tmp_path):
    f = tmp_path / "g.tsv"
    f.write_text("one")
    m = M.build("x", "corpus", [{"id": "a", "files": [M.file_entry(str(f), str(tmp_path))]}])
    assert M.verify(m, str(tmp_path)) == []
    f.write_text("two")
    assert M.verify(m, str(tmp_path)) == ["g.tsv"]
    sha = M.write(m, str(tmp_path / "manifests"))
    assert M.load(str(tmp_path / "manifests" / (sha + ".json"))) == m
