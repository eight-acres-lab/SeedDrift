from ci.hashing import canonical_json, files_hash, sha256_bytes, sha256_file, stage_key


def test_canonical_json_ignores_key_order():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


def test_stage_key_changes_with_every_part():
    base = stage_key("r", "1", "c", {"x": 1}, ["h1", "h2"], "e")
    assert base == stage_key("r", "1", "c", {"x": 1}, ["h2", "h1"], "e")
    for k in (stage_key("r", "2", "c", {"x": 1}, ["h1", "h2"], "e"), stage_key("r", "1", "d", {"x": 1}, ["h1", "h2"], "e"),
              stage_key("r", "1", "c", {"x": 2}, ["h1", "h2"], "e"), stage_key("r", "1", "c", {"x": 1}, ["h1"], "e"),
              stage_key("r", "1", "c", {"x": 1}, ["h1", "h2"], "f")):
        assert k != base


def test_files_hash_and_file_sha(tmp_path):
    (tmp_path / "a").write_text("1")
    (tmp_path / "b").write_text("2")
    h = files_hash(str(tmp_path), ["a", "b"])
    assert h == files_hash(str(tmp_path), ["b", "a"])
    (tmp_path / "b").write_text("3")
    assert h != files_hash(str(tmp_path), ["a", "b"])
    assert sha256_file(str(tmp_path / "a")) == sha256_bytes(b"1")
