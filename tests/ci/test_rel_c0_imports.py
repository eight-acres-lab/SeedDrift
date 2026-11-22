"""Import rules of the relational-model line, checked on every file (the import scanner of gate G2: absolute,
relative and literal dynamic imports):
- no rel module imports ci/gpu_lease (it stops the co-tenant service of a machine; the new line never uses it);
- the train side (everything under seeddrift/train/rel/ and the stage modules configs/rel/kinds.yaml marks side
  train) never imports the generator of synthetic tables (ci/stages/rel/synth.py), the ruler generators (bench/rel/)
  or the test fixtures;
- no rel module or import is named construct (the static check of gate G2 keys on that name);
- files both interpreters run never import torch (the CI environment has none)."""
import os

from conftest import ROOT
from ci.gates.g2_exclusion import _imports
from ci.stages.rel import kinds

REL_DIRS = ("seeddrift/train/rel", "ci/stages/rel", "bench/rel")
GENERATORS = ("ci.stages.rel.synth", "bench.rel", "rel_fixtures", "tests.ci.rel_fixtures")
BOTH_SIDES = ("seeddrift/train/rel/__init__.py", "seeddrift/train/rel/api.py", "seeddrift/train/rel/registry.py", "seeddrift/train/rel/sealed.py",
              "seeddrift/train/rel/handoff.py", "seeddrift/train/rel/models/uniform.py", "seeddrift/train/rel/trainer/tolerance.py")


def _py(folder, root=ROOT):
    full = os.path.join(root, folder)
    return sorted(os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs if f.endswith(".py"))


def _hits(files, names, root=ROOT):
    bad = []
    for p in files:
        for name in _imports(p, root):
            if any(name == g or name.startswith(g + ".") for g in names):
                bad.append((os.path.relpath(p, root), name))
    return bad


def train_side(root=ROOT):
    stages = [os.path.join(root, *d["module"].split(".")) + ".py" for d in kinds.all_stages().values() if d["side"] == "train"]
    return _py("seeddrift/train/rel", root) + [p for p in stages if os.path.exists(p)]


def test_no_rel_module_imports_the_gpu_lease():
    files = [p for d in REL_DIRS for p in _py(d)] + [os.path.join(ROOT, "tests", "ci", "rel_fixtures.py")]
    assert _hits(files, ("ci.gpu_lease",)) == []
    for f in os.listdir(os.path.join(ROOT, "ci", "rules")):
        if f.startswith("rel_"):
            assert "gpu_lease" not in open(os.path.join(ROOT, "ci", "rules", f)).read(), f


def test_train_side_does_not_import_generators():
    assert _hits(train_side(), GENERATORS) == []


def test_generator_import_on_the_train_side_fails(tmp_path):
    (tmp_path / "seeddrift" / "train" / "rel" / "data").mkdir(parents=True)
    (tmp_path / "seeddrift" / "train" / "rel" / "data" / "reader.py").write_text("from ci.stages.rel import synth\n")
    (tmp_path / "seeddrift" / "train" / "rel" / "ok.py").write_text("import numpy\n")
    (tmp_path / "seeddrift" / "train" / "rel" / "dyn.py").write_text("import importlib\nimportlib.import_module('bench.rel.splice')\n")
    bad = _hits(_py("seeddrift/train/rel", str(tmp_path)), GENERATORS, str(tmp_path))
    assert sorted(b[0] for b in bad) == [os.path.join("seeddrift", "train", "rel", "data", "reader.py"), os.path.join("seeddrift", "train", "rel", "dyn.py")]


def test_nothing_is_named_construct():
    files = [p for d in REL_DIRS for p in _py(d)]
    assert [p for p in files if os.path.basename(p) == "construct.py"] == []
    assert [(p, n) for p in files for n in _imports(p, ROOT) if n.split(".")[-1] == "construct"] == []


def test_files_of_both_interpreters_do_not_import_torch():
    assert _hits([os.path.join(ROOT, p) for p in BOTH_SIDES], ("torch",)) == []
