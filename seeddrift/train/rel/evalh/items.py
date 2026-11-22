"""The evaluation items (package W5): which stretch of which held-out genome is scored, and in which half.

An item is a held-out genome, one of its genes, the codons [first, first + codons) of that gene, its half (screening
or confirmation) and the stretch's base range on its replicon (the contract defines items by bases; for a spliced
eukaryotic gene the range is as approximate as the stored gene coordinates). The tables of the items (the context
rows and relation loci) are added by the tables package with the code that builds its training tables.

The rules use the examined genome's own gene lengths and its taxonomy only, never another genome or any cell:
- per group PER_GROUP items; the held-out genomes of the group take turns in a hashed order, each turn one gene of
  that genome not used yet, the next in a hashed order of its genes with at least MIN_CODONS codons; a genome without
  such genes left drops out of the turns;
- the stretch lies in the part of the gene a target block stores (its first LEN_TARGET codons): all of that part when
  it holds at most MAX_CODONS codons, else MAX_CODONS codons from a hashed start;
- halves by genus (the genus taxid of the genome; a genome without one is its own unit): the units of a group in a
  hashed order, each to the half with fewer items so far (screening on ties). The two halves share no genus.
Every random choice is the 64-bit hash of evalh.conditions (seed, stream, indices), so the items are the same in
every numpy. Genomes are keyed by their rank among the sorted ids given.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.evalh import conditions as cond

VERSION = 1
SEED = 20261122
PER_GROUP = 100
MAX_CODONS = 128
MIN_CODONS = 16
HALVES = ("screen", "confirm")
STREAM_GENOMES, STREAM_GENES, STREAM_WINDOW, STREAM_UNITS = 11, 12, 13, 14


def _order(seed, stream, key, n):
    """A hashed order of range(n) under one key."""
    return np.argsort(cond.hash_u01(seed, stream, int(key), np.arange(n)), kind="stable")


def bases(start, end, strand, first, codons):
    """(first base, last base) of codons [first, first + codons) of a gene with stored coordinates start..end (both
    inclusive, the convention of the training arrays), on the forward strand of its replicon."""
    if strand >= 0:
        return int(start) + 3 * int(first), int(start) + 3 * (int(first) + int(codons)) - 1
    return int(end) - 3 * (int(first) + int(codons)) + 1, int(end) - 3 * int(first)


def plan(genomes, seed=SEED, per_group=PER_GROUP, max_codons=MAX_CODONS, min_codons=MIN_CODONS):
    """genomes: [dict(id, group, genus, lengths, start, end, strand, replicon)] of the held-out genomes (lengths: codons
    of every gene; the other arrays per gene as stored). Returns (items, short): items as dicts in group order, and
    {group: items missing} for a group whose genes ran out."""
    ids = sorted(g["id"] for g in genomes)
    if len(set(ids)) != len(ids):
        raise ValueError("a genome is given twice")
    key = {g: k for k, g in enumerate(ids)}
    by_group = {}
    for g in genomes:
        by_group.setdefault(str(g["group"]), []).append(g)
    groups = sorted(by_group)
    items, short = [], {}
    for gi, group in enumerate(groups):
        members = sorted(by_group[group], key=lambda g: key[g["id"]])
        turns = [members[int(k)] for k in _order(seed, STREAM_GENOMES, gi, len(members))]
        queues = []
        for g in turns:
            lengths = np.asarray(g["lengths"]).astype(np.int64)
            ok = np.flatnonzero(lengths >= int(min_codons))
            queues.append(list(ok[_order(seed, STREAM_GENES, key[g["id"]], ok.shape[0])]))
        mine, at = [], 0
        while len(mine) < int(per_group) and any(queues):
            k = at % len(turns)
            at += 1
            if not queues[k]:
                continue
            g, j = turns[k], int(queues[k].pop(0))
            part = min(int(g["lengths"][j]), api.LEN_TARGET)
            n = min(int(max_codons), part)
            u = cond.hash_u01(seed, STREAM_WINDOW, key[g["id"]], j)[0]
            first = int(np.floor(u * (part - n + 1)))
            lo, hi = bases(g["start"][j], g["end"][j], int(g["strand"][j]), first, n)
            mine.append(dict(group=group, genome=g["id"], genus=str(g["genus"]), gene=j, first=first, codons=n, replicon=int(g["replicon"][j]),
                             strand=int(g["strand"][j]), base_first=lo, base_last=hi))
        if len(mine) < int(per_group):
            short[group] = int(per_group) - len(mine)
        units = sorted({it["genus"] for it in mine})
        count = {u: sum(it["genus"] == u for it in mine) for u in units}
        half, size = {}, [0, 0]
        for k in _order(seed, STREAM_UNITS, gi, len(units)):
            u = units[int(k)]
            h = 0 if size[0] <= size[1] else 1
            half[u] = HALVES[h]
            size[h] += count[u]
        for it in mine:
            it["half"] = half[it["genus"]]
        items += mine
    for k, it in enumerate(items):
        it["item"] = k
    return items, short


def genus_of(meta, ranks=("phylum", "class", "order", "family", "genus", "species")):
    """The unit of the halves of a genome from its training-data metadata: its genus taxid, else its own id."""
    r = list(meta.get("ranks") or [])
    g = int(r[ranks.index("genus")]) if len(r) > ranks.index("genus") else 0
    return "genus:%d" % g if g > 0 else "genome:%s" % meta["id"]


def digest(items):
    """sha256 of the items as a list (the identity of the item set)."""
    keys = ("item", "group", "genome", "gene", "first", "codons", "replicon", "strand", "base_first", "base_last", "half")
    text = json.dumps([[it[k] for k in keys] for it in items], separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def problems(items, short, per_group=PER_GROUP, groups=None, lengths=None):
    """What is wrong with a set of items: groups missing or short, an empty half in a group, a genus in both halves,
    a stretch outside the stored part of its gene or longer than MAX_CODONS (lengths: {(genome, gene): codons})."""
    out = []
    have = sorted({it["group"] for it in items})
    if groups is not None and sorted(groups) != have:
        out.append(dict(error="groups without items", groups=sorted(set(groups) - set(have))))
    for g, n in sorted(short.items()):
        out.append(dict(group=g, error="%d items short: the group ran out of genes" % n))
    for g in have:
        mine = [it for it in items if it["group"] == g]
        if len(mine) != per_group and g not in short:
            out.append(dict(group=g, error="%d items, %d expected" % (len(mine), per_group)))
        if {it["half"] for it in mine} != set(HALVES):
            out.append(dict(group=g, error="a half without items (one genus only?)", halves=sorted({it["half"] for it in mine})))
        both = {u for u in {it["genus"] for it in mine} if len({it["half"] for it in mine if it["genus"] == u}) > 1}
        if both:
            out.append(dict(group=g, error="a genus in both halves", genera=sorted(both)[:5]))
    seen = set()
    for it in items:
        k = (it["genome"], it["gene"])
        if k in seen:
            out.append(dict(item=it["item"], error="a gene used twice"))
        seen.add(k)
        if not 1 <= it["codons"] <= MAX_CODONS or it["first"] < 0 or it["first"] + it["codons"] > api.LEN_TARGET:
            out.append(dict(item=it["item"], error="the stretch is not inside the stored part of its gene"))
        if lengths is not None and it["first"] + it["codons"] > lengths.get(k, -1):
            out.append(dict(item=it["item"], error="the stretch runs past the end of its gene"))
        if it["base_last"] - it["base_first"] + 1 != 3 * it["codons"]:
            out.append(dict(item=it["item"], error="the base range is not three bases per codon"))
    return out
