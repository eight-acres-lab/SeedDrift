from ci.gates import g7_env

LOCK = "name: x\ndependencies:\n  - python=3.11.9=h1\n  - numpy=2.4.6=py311h2\n"


def test_same_packages_pass(tmp_path):
    p = tmp_path / "lock.yml"
    p.write_text(LOCK)
    assert g7_env.lock_matches(LOCK, str(p)).status == "pass"


def test_changed_package_fails(tmp_path):
    p = tmp_path / "lock.yml"
    p.write_text(LOCK)
    r = g7_env.lock_matches(LOCK.replace("2.4.6", "2.4.7"), str(p))
    assert r.status == "fail" and r.evidence["missing"] and r.evidence["extra"]


def test_temp_dir_outside_the_data_root_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", "/somewhere/else")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("PIP_CACHE_DIR", str(tmp_path))
    r = g7_env.resources(str(tmp_path), need_disk_gb=0, reserve_gb=0)
    assert r.status == "fail" and r.evidence["problems"][0]["var"] == "TMPDIR"


def test_resources_pass(tmp_path, monkeypatch):
    for v in ("TMPDIR", "XDG_CACHE_HOME", "PIP_CACHE_DIR"):
        monkeypatch.setenv(v, str(tmp_path))
    assert g7_env.resources(str(tmp_path), need_disk_gb=0, reserve_gb=0, root_min_gb=0).status == "pass"


def test_relocated_env_fails(tmp_path):
    from collections import namedtuple
    VP = namedtuple("VP", "openssl_cafile")
    env = tmp_path / "env"
    (env / "conda-meta").mkdir(parents=True)
    (env / "ssl").mkdir()
    (env / "ssl" / "cert.pem").write_text("x")
    assert g7_env.relocation(str(env), VP(str(env / "ssl" / "cert.pem"))).status == "pass"
    moved = g7_env.relocation(str(env), VP(str(tmp_path / "env.build.1" / "ssl" / "cert.pem")))
    assert moved.status == "fail"
    assert g7_env.relocation(str(env), VP(str(env / "ssl" / "gone.pem"))).status == "fail"
    assert g7_env.relocation(str(tmp_path)).status == "pass"            # not a conda prefix: the lock gate decides
