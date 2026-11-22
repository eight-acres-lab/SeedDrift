"""Profiles of the relational-model line: every rel-dev-* profile takes the select lock (the exact line
`needs_gpu: false`, which ci/server_launch.sh greps) and runs on the CPU; every rel profile names a machine of
configs/rel/machines.json, never asks for the GPU lease, and lists stages of its own kind; profiles of a select kind
carry `needs_gpu: false`, and round profiles of a GPU kind do not."""
import os
import re

import pytest
import yaml

from conftest import ROOT
from ci.stages.rel import common, kinds

PROFILES = os.path.join(ROOT, "ci", "profiles")


def _profiles():
    out = []
    for name in sorted(os.listdir(PROFILES)):
        p = os.path.join(PROFILES, name, "config.yaml")
        text = open(p).read()
        cfg = yaml.safe_load(text)
        if name.startswith("rel-") or kinds.is_kind(cfg.get("pipeline")):
            out.append((name, text, cfg))
    return out


def problems(name, text, cfg):
    out = []
    select_line = re.search(r"^needs_gpu: false$", text, re.MULTILINE) is not None
    if not kinds.is_kind(cfg.get("pipeline")):
        return ["a rel-* profile runs a rel kind"]
    if name.startswith("rel-dev-"):
        if not select_line:
            out.append("a rel-dev profile has the line needs_gpu: false")
        if (cfg.get("train") or {}).get("device") != "cpu":
            out.append("a rel-dev profile sets train.device: cpu")
    elif not re.match(r"^rel-[a-z0-9]+-i[12]$", name):
        out.append("a rel profile is rel-dev-<package>-<purpose> or rel-<round>-<machine>")
    if (cfg.get("train") or {}).get("gpu_lease") is not False:
        out.append("a rel profile sets train.gpu_lease: false")
    if (cfg.get("rel") or {}).get("machine") not in common.machines():
        out.append("rel.machine names a machine of configs/rel/machines.json")
    try:
        p = kinds.pipeline(cfg)
    except SystemExit as err:
        return out + [str(err)]
    if p["lock"] == "select" and not select_line:
        out.append("a profile of a select kind has the line needs_gpu: false")
    if p["lock"] == "gpu" and select_line and not name.startswith("rel-dev-"):
        out.append("a round profile of a GPU kind takes the GPU lock")
    return out


def test_rel_profiles_exist_for_every_kind():
    kinds_seen = {cfg["pipeline"] for name, _, cfg in _profiles() if name.startswith("rel-dev-c0-dry-")}
    assert kinds_seen == set(kinds.load()["kinds"])


@pytest.mark.parametrize("name,text,cfg", _profiles(), ids=[p[0] for p in _profiles()])
def test_rel_profile_rules(name, text, cfg):
    assert problems(name, text, cfg) == []


def test_profile_rules_fail_on_gpu_lock_lease_or_device():
    text = "pipeline: rel_tables\ntier: tiny\nrel: {machine: i1, stages: [c0_synth]}\ntrain: {gpu_lease: false, device: cpu}\n"
    assert problems("rel-dev-c0-x", text, yaml.safe_load(text)) != []                 # no needs_gpu: false line
    text2 = "pipeline: rel_tables\nneeds_gpu: false\nrel: {machine: i1, stages: [c0_synth]}\ntrain: {device: cpu}\n"
    assert problems("rel-dev-c0-x", text2, yaml.safe_load(text2)) != []               # no gpu_lease: false
    text3 = "pipeline: rel_tables\nneeds_gpu: false\nrel: {machine: i1, stages: [c0_synth]}\ntrain: {gpu_lease: false, device: auto}\n"
    assert problems("rel-dev-c0-x", text3, yaml.safe_load(text3)) != []               # not on the CPU
    text4 = "pipeline: rel_traineval\nneeds_gpu: false\nrel: {machine: i3, stages: [w4_train]}\ntrain: {gpu_lease: false, device: cpu}\n"
    assert problems("rel-r1-i1", text4, yaml.safe_load(text4)) != []                  # unknown machine; a round without the GPU lock
