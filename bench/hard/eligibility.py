#!/usr/bin/env python
"""Label gates, eligibility and the split into Test A and Reserve B (PROTOCOL.md sections 2.4 and 3).

Reads label files only: no score, no hit table. It may therefore run on the sealed chromosomes before the freeze.

  gates        on the development units with at least 3 other-species panel genomes:
                 gate 1  median over the chromosomes of a unit of [single-copy published backbone genes with f >= 0.8]
                         is at least 80%, in every such unit
                 gate 2  c = island-labelled genes with f >= 0.8 / island-labelled genes; in every such unit the median
                         c over its chromosomes with more backbone than island genes is at most 5%, and the median c
                         over its other chromosomes (if any) is at most 20%
  per genus    panel unreliable (under 70% of the single-copy backbone genes in the positional core, or none to
               judge by), label implausible (c above 20%), counts, eligibility of P1, P2 and S-pub
  split        the two fixed lists of family units; with n_P1 >= 14 the first 2 single-genus units of the high list
               and the first 3 units of the low list that are P1-eligible form Reserve B; with 9 to 13 there is no
               reserve; under 9, or when a gate fails, the fallback PUB applies (published labels, no verdicts)

Writes results/eligibility.tsv, results/split.json and (with --write-sealed) the tiers of results/sealed.tsv.

usage:
  python bench/hard/eligibility.py --labels DIR/labels_panel --published DIR/labels \\
      --development bench/islands/results/dev_genomes.tsv --sealed-genomes bench/islands/results/genomes.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import chromosome, eligibility  # noqa: E402

HIGH = [("Mycobacterium",), ("Acinetobacter",), ("Corynebacterium",), ("Rhizobium", "Sinorhizobium"), ("Pseudomonas",), ("Burkholderia",), ("Bacillus",),
        ("Bradyrhizobium", "Rhodopseudomonas")]
LOW = [("Streptococcus",), ("Staphylococcus",), ("Shewanella",), ("Haemophilus",), ("Prochlorococcus",), ("Vibrio",), ("Rickettsia",)]
DROPPED = {"Brucella": "no largest replicon in the island set (both records are chromosome II)"}
DEV_UNITS = {"Shigella": "Escherichia"}


def table(genomes, labels, published, step2=""):
    rows = []
    for g in genomes:
        acc = g["accession"]
        if not os.path.exists(os.path.join(labels, acc + ".tsv")):
            rows.append(dict(accession=acc, genus=g["genus"], missing=True))
            continue
        e = eligibility(chromosome(acc, labels, published))
        if step2 and os.path.exists(os.path.join(step2, acc + ".tsv")):
            e2 = eligibility(chromosome(acc, step2, published))
            e["native_blocks_step2"], e["p2_step2"] = e2["native_blocks"], e2["p2"]
        pc, c = e["backbone_positional_core"], e["conflict_share"]
        e["panel_unreliable"] = bool(e["other_species"] >= 3 and (pc is None or pc < 0.7))
        e["label_implausible"] = bool(c is not None and c > 0.2)
        rows.append(dict(accession=acc, genus=g["genus"], missing=False, **e))
    return rows


def gates(dev):
    units, out, ok = {}, {}, True
    for r in dev:
        if not r["missing"] and r["other_species"] >= 3:
            units.setdefault(DEV_UNITS.get(r["genus"], r["genus"]), []).append(r)
    for u, rs in sorted(units.items()):
        g1 = float(np.median([r["backbone_positional_core"] for r in rs if r["backbone_positional_core"] is not None]))
        more_bb = [r["conflict_share"] for r in rs if not r["count_flagged"] and r["conflict_share"] is not None]
        others = [r["conflict_share"] for r in rs if r["count_flagged"] and r["conflict_share"] is not None]
        g2a = float(np.median(more_bb)) if more_bb else None
        g2b = float(np.median(others)) if others else None
        passed = g1 >= 0.8 and (g2a is None or g2a <= 0.05) and (g2b is None or g2b <= 0.20)
        ok = ok and passed
        out[u] = dict(chromosomes=len(rs), gate1_median_backbone_in_core=round(g1, 4), gate2_median_conflict_backbone_majority=None if g2a is None else round(g2a, 4),
                      gate2_median_conflict_other=None if g2b is None else round(g2b, 4), passed=bool(passed))
    return out, ok and bool(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True); ap.add_argument("--published", required=True)
    ap.add_argument("--labels-step2", default="", help="panel labels built with --block-min 7 (the second step of the ladder)")
    ap.add_argument("--development", required=True); ap.add_argument("--sealed-genomes", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results")); ap.add_argument("--write-sealed", action="store_true")
    a = ap.parse_args()
    dev = table(list(csv.DictReader(open(a.development), delimiter="\t")), a.labels, a.published, a.labels_step2)
    sealed = table(list(csv.DictReader(open(a.sealed_genomes), delimiter="\t")), a.labels, a.published, a.labels_step2)
    gate, gates_ok = gates(dev)
    by_genus = {r["genus"]: r for r in sealed}
    for r in sealed:
        r["dropped"] = DROPPED.get(r["genus"], "")
        if r["missing"] or r["dropped"]:
            r["p1_final"] = r["p2_final"] = False
        else:
            r["p1_final"] = bool(r["p1"] and not r["panel_unreliable"] and not r["label_implausible"])
            r["p2_final"] = bool(r["p2"] and not r["panel_unreliable"] and not r["label_implausible"])
    complete = all(not r["missing"] for r in sealed if not r["dropped"])
    n_p1 = sum(r["p1_final"] for r in sealed)
    ok_unit = lambda unit: any(by_genus[g]["p1_final"] for g in unit if g in by_genus)
    split = dict(n_p1=int(n_p1), gates=gate, gates_passed=bool(gates_ok), labels_complete=bool(complete), high=["+".join(u) for u in HIGH], low=["+".join(u) for u in LOW],
                 dropped=DROPPED)
    reserve = []
    if not complete:
        split["rule"] = "not decided: label files of some sealed genera are missing"
    elif not gates_ok or n_p1 < 9:
        split["rule"] = "fallback PUB: published labels, all genera Test A, no verdicts"
    elif n_p1 >= 14:
        reserve = [u for u in HIGH if len(u) == 1 and ok_unit(u)][:2] + [u for u in LOW if ok_unit(u)][:3]
        split["rule"] = "n_P1 >= 14: Reserve B = first 2 single-genus units of the high list and first 3 units of the low list that are P1-eligible"
    else:
        split["rule"] = "9 <= n_P1 <= 13: no Reserve B, all genera Test A"
    in_b = {g for u in reserve for g in u}
    for r in sealed:
        r["tier"] = "secondary" if r["dropped"] else "sealed" if not complete else "B" if r["genus"] in in_b else "A"
    test_a = [r for r in sealed if r["tier"] == "A"]
    split["reserve_b"] = sorted(in_b)
    split["test_a"] = [r["genus"] for r in test_a]
    split["p1_units_test_a"] = [r["genus"] for r in test_a if r["p1_final"]]
    step1 = [r["genus"] for r in test_a if r["p2_final"]]
    split["p2_units_test_a_step1"] = step1
    if a.labels_step2:
        step2 = [r["genus"] for r in test_a if r.get("p2_step2") and not r["panel_unreliable"] and not r["label_implausible"]]
        split["p2_units_test_a_step2"] = step2
        split["block_rule"] = "9 of 11" if len(step1) >= 8 else "7 of 11" if len(step2) >= 8 else "9 of 11, P2 without verdicts"
    os.makedirs(a.out, exist_ok=True)
    cols = ["set", "genus", "accession", "tier", "other_species", "backbone_positional_core", "conflict_share", "panel_unreliable", "label_implausible", "natives",
            "deep_natives", "island_genes", "island_regions", "native_blocks", "native_blocks_step2", "published_island", "published_backbone", "p1", "p2", "pub_gene",
            "pub_block", "count_flagged", "dropped"]
    with open(os.path.join(a.out, "eligibility.tsv"), "w") as fo:
        fo.write("\t".join(cols) + "\n")
        for name, rows in (("development", dev), ("sealed", sealed)):
            for r in rows:
                r = dict(r, set=name, tier=r.get("tier", "open"), p1=r.get("p1_final", r.get("p1")), p2=r.get("p2_final", r.get("p2")))
                fo.write("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + "\n")
    json.dump(split, open(os.path.join(a.out, "split.json"), "w"), indent=1)
    if a.write_sealed and complete:
        with open(os.path.join(a.out, "sealed.tsv"), "w") as fo:
            fo.write("accession\tgenus\ttier\n")
            for r in sealed:
                fo.write("%s\t%s\t%s\n" % (r["accession"], r["genus"], r["tier"]))
    print(json.dumps(split, indent=1))
    print("\n" + "\t".join(cols[:16]))
    for name, rows in (("development", dev), ("sealed", sealed)):
        for r in rows:
            if not r["missing"]:
                r = dict(r, set=name, tier=r.get("tier", "open"))
                print("\t".join("" if r.get(c) is None else str(r.get(c)) for c in cols[:16]))


if __name__ == "__main__":
    main()
