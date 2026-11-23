"""The variant files of arms S1 and S2 (package W5, decision D7) in the CI environment (no torch): they load as the
trainer loads them, differ from the reference arm they are set against only in their model, and name a size the
source module accepts. The module's own tests (torch) run in the model interpreter: seeddrift/train/rel/models/source.py."""
import ast
import os

import yaml

from conftest import ROOT
from ci.stages.common import Stage
from ci.stages.rel import w6_fit
from seeddrift.train.rel import registry

FOLDER = os.path.join(ROOT, "configs", "rel", "variants")
ARMS = {"w5-src30m-s1": ("s1", "w6-ref30m-leakfree-lr1.5e-4"), "w5-src30m-s2": ("s2", "w6-ref30m-leakfree-lr1.5e-4"),    # tables of version 0
        "w5-src30m-s1-v1": ("s1", "w6-ref30m-v1"), "w5-src30m-s2-v1": ("s2", "w6-ref30m-v1"),                        # round 1
        "w5-src30m-s1-strict-v1": ("s1", "w6-ref30m-strict-v1"), "w5-src30m-s2-strict-v1": ("s2", "w6-ref30m-strict-v1"),  # round 1, batch 3
        "w5-copy-aa30m": ("s2", "w6-ref30m-leakfree-lr1.5e-4")}                                                     # S2 with the amino-acid level
SWITCHES = {"w5-copy-aa30m": {"aa_level": True}}
SOURCE = os.path.join(ROOT, "seeddrift", "train", "rel", "models", "source.py")


def _load(name):
    with open(os.path.join(FOLDER, name + ".yaml")) as f:
        return yaml.safe_load(f)


def _module_constant(name):
    tree = ast.parse(open(SOURCE).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("source.py has no constant %s" % name)


def test_variants_load_and_follow_the_reference_arm():
    st = Stage.__new__(Stage)
    for name, (arm, reference) in ARMS.items():
        ref = _load(reference)
        v = w6_fit.load_variant(st, "configs/rel/variants/%s.yaml" % name)
        assert v["name"] == name and v["author"] == "w5" and v["model"] == "source" and "source" in registry.available("model")
        rest = {k: x for k, x in v.items() if k not in ("name", "author", "model", "size", "gpu_mb")}
        assert rest == {k: x for k, x in ref.items() if k not in ("name", "author", "model", "size", "gpu_mb")}, name
        size = v["size"]
        assert size == dict({"layers": 12, "width": 384, "arm": arm, "self_layers": 4}, **SWITCHES.get(name, {})), size
        assert set(size) <= set(_module_constant("KNOWN")) and size["arm"] in _module_constant("ARMS")
        assert 1 <= size["self_layers"] < size["layers"] and v["gpu_mb"] >= ref["gpu_mb"]


def test_the_source_module_imports_only_torch_and_the_line():
    tree = ast.parse(open(SOURCE).read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0] if not node.module.startswith("seeddrift") else node.module)
    allowed = {"__future__", "math", "itertools", "types", "torch", "numpy", "seeddrift.train.rel", "seeddrift.train.rel.models",
               "seeddrift.train.rel.models.common", "seeddrift.train.rel.data"}
    assert names <= allowed, sorted(names - allowed)
