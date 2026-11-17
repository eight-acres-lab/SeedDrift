"""Genome groups of the multi-kingdom quota gate (G1).

The group of a genome comes from its taxid's lineage (configs/kingdom_map.yml), never from a file name or a label in a
spec. Mobile elements are the one exception: a plasmid carries its host's taxid, so its group comes from its source kind.
A virus with a known host is placed by its host (`virus_hosts`): the taxon lists cannot name every family of
prokaryote viruses (archaeal viruses, unclassified phages), so a host in Bacteria or Archaea makes it a prokaryote
virus, a host in Eukaryota a eukaryotic virus (giant viruses stay giant). A virus of a listed prokaryote-virus taxon
whose host is a eukaryote (or the reverse for giant viruses) is a conflict, not a guess. An unknown taxid, a lineage
that matches no group, or a conflict is an error (fail closed).
"""
from __future__ import annotations

import os

import yaml

from ci import ROOT
from ci.taxonomy import Taxonomy

DEFAULT_MAP = os.path.join(ROOT, "configs", "kingdom_map.yml")
VIRUSES = 10239


class NoGroup(ValueError):
    pass


class GroupConflict(NoGroup):
    pass


class KingdomMap:
    def __init__(self, path: str = DEFAULT_MAP):
        with open(path) as f:
            d = yaml.safe_load(f)
        self.groups = d["groups"]              # ordered: the first matching group wins
        self.by_source = d.get("by_source", {})
        self.virus_hosts = d.get("virus_hosts") or {}
        self.names = [g["name"] for g in self.groups] + sorted(set(self.by_source.values()))

    def _by_taxon(self, lin, taxid, tax):
        for g in self.groups:
            if lin & set(g["include"]) and not lin & set(g.get("exclude", [])):
                return g["name"]
        raise NoGroup("taxid %s (%s) matches no group" % (taxid, tax.name(taxid)))

    def host_class(self, tax: Taxonomy, host_taxid) -> str:
        """'prokaryote', 'eukaryote' or '' (no host, or a host in neither class)."""
        if not host_taxid:
            return ""
        hl = set(tax.lineage(int(host_taxid)))
        for cls in ("prokaryote", "eukaryote"):
            rule = self.virus_hosts.get(cls)
            if rule and hl & set(rule["include"]):
                return cls
        return ""

    def group(self, tax: Taxonomy, taxid: int, source_kind: str = "", host_taxid=None) -> str:
        if source_kind in self.by_source:
            return self.by_source[source_kind]
        lin = set(tax.lineage(taxid))
        by_taxon = self._by_taxon(lin, taxid, tax)
        if VIRUSES not in lin or not host_taxid or not self.virus_hosts:
            return by_taxon
        cls = self.host_class(tax, host_taxid)
        if not cls:
            return by_taxon
        rule = self.virus_hosts[cls]
        if by_taxon in rule.get("keep", []):
            return by_taxon
        if by_taxon in rule.get("conflict", []):
            raise GroupConflict("virus %s (%s) is in group %s by its taxon but its host %s is a %s" % (taxid, tax.name(taxid), by_taxon, host_taxid, cls))
        return rule["group"]
