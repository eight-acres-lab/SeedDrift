"""Genome groups of the multi-kingdom quota gate (G1).

The group of a genome comes from its taxid's lineage (configs/kingdom_map.yml), never from a file name or a label in a
spec. Mobile elements are the one exception: a plasmid carries its host's taxid, so its group comes from its source kind.
An unknown taxid or a lineage that matches no group is an error (fail closed).
"""
from __future__ import annotations

import os

import yaml

from ci import ROOT
from ci.taxonomy import Taxonomy

DEFAULT_MAP = os.path.join(ROOT, "configs", "kingdom_map.yml")


class NoGroup(ValueError):
    pass


class KingdomMap:
    def __init__(self, path: str = DEFAULT_MAP):
        with open(path) as f:
            d = yaml.safe_load(f)
        self.groups = d["groups"]              # ordered: the first matching group wins
        self.by_source = d.get("by_source", {})
        self.names = [g["name"] for g in self.groups] + sorted(set(self.by_source.values()))

    def group(self, tax: Taxonomy, taxid: int, source_kind: str = "") -> str:
        if source_kind in self.by_source:
            return self.by_source[source_kind]
        lin = set(tax.lineage(taxid))
        for g in self.groups:
            if lin & set(g["include"]) and not lin & set(g.get("exclude", [])):
                return g["name"]
        raise NoGroup("taxid %s (%s) matches no group" % (taxid, tax.name(taxid)))
