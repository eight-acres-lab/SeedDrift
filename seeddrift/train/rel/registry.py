"""Finds the parts of an arm by module name (owned by the coordinator). Both interpreters import it: it never imports
torch itself; a model module it loads may.

  CATEGORIES        category -> package: model (models/), objective (objectives/), mask (data/masks/), baseline
                    (baselines/), readout (readouts/)
  available(cat)    the module names of a category, without importing any of them (packages such as models/common
                    and names starting with _ are not modules of the category)
  load(cat, name)   the module seeddrift.train.rel.<package>.<name>, checked for what the contract asks of the category
                    (model: logp and info; objective: loss; mask: plan)
  compose(model=, objective=, mask=)
                    the three modules of an arm (a variant file names one of each), as a dict
Names are lower case letters, digits and underscores. A name never reaches outside its package, so the generators of
synthetic tables (ci/stages/rel/synth.py) and rulers (bench/rel/) cannot be loaded through here, and no module may
be called construct (the static check of gate G2 keys on that name).
"""
from __future__ import annotations

import importlib
import os
import pkgutil
import re

CATEGORIES = dict(model="seeddrift.train.rel.models", objective="seeddrift.train.rel.objectives", mask="seeddrift.train.rel.data.masks",
                  baseline="seeddrift.train.rel.baselines", readout="seeddrift.train.rel.readouts")
REQUIRED = dict(model=("logp", "info"), objective=("loss",), mask=("plan",), baseline=(), readout=())
NAME_RX = re.compile(r"^[a-z][a-z0-9_]*$")
FORBIDDEN = ("construct",)
HERE = os.path.dirname(os.path.abspath(__file__))


class RegistryError(LookupError):
    pass


def _folder(category):
    if category not in CATEGORIES:
        raise RegistryError("unknown category %r (known: %s)" % (category, ", ".join(sorted(CATEGORIES))))
    return os.path.join(HERE, *CATEGORIES[category].split(".")[3:])


def available(category) -> list:
    return sorted(m.name for m in pkgutil.iter_modules([_folder(category)]) if not m.ispkg and not m.name.startswith("_"))


def load(category, name):
    folder = _folder(category)
    if not isinstance(name, str) or not NAME_RX.match(name) or name in FORBIDDEN:
        raise RegistryError("a %s name is lower case letters, digits and underscores (not %s): %r" % (category, ", ".join(FORBIDDEN), name))
    if not os.path.exists(os.path.join(folder, name + ".py")):
        raise RegistryError("no %s named %r (available: %s)" % (category, name, ", ".join(available(category)) or "none"))
    mod = importlib.import_module("%s.%s" % (CATEGORIES[category], name))
    missing = [a for a in REQUIRED[category] if not hasattr(mod, a)]
    if missing:
        raise RegistryError("%s %s lacks %s" % (category, name, ", ".join(missing)))
    return mod


def compose(model, objective, mask) -> dict:
    return dict(model=load("model", model), objective=load("objective", objective), mask=load("mask", mask))
