"""Event registry drafts (package W15): read, check, group and count. Standard library only; no scores anywhere.

usage: python bench/rel/registry/registry.py REGISTRY.tsv [--md OUT.md] [--json OUT.json]

The rules are bench/rel/registry/RULES.md (rules version 0.1). This module checks a draft against them, computes the
groups (records whose exclusion radii overlap, or that share a paper or an assembly) and writes the three reports of
the first version: the association graph by exclusion lineage, the counts per grade and per scenario cell (direction x
recipient group), and the share of records with recheckable evidence. It never reads model output, and it is never
imported by training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict

RULES_VERSION = "0.1"

COLUMNS = (
    "event_id", "record_kind", "claim", "recipient", "recipient_group", "region", "gene_family", "donor_claimed",
    "donor_group", "direction", "scenario_tags", "category", "claims", "exclusion_taxon", "exclusion_lineage",
    "recipient_exclusion_taxid", "exclusion_basis", "papers", "year", "db_date", "assembly", "discovery", "ev_direct",
    "ev_position", "ev_phylogeny", "ev_population", "position_rule", "grade_transfer", "grade_donor", "status",
    "p_transfer", "p_donor", "truth_class", "recheckable", "later", "exposure", "in_pretraining", "sealed_overlap",
    "provisional_cover", "role", "verification", "notes",
)
GROUPS = ("bacteria", "archaea", "mammals", "other_vertebrates", "invertebrates", "plants_algae", "fungi", "protists",
          "phage", "eukaryotic_virus", "giant_virus", "plasmid")
DIRECTIONS = ("prok>prok", "prok>euk", "euk>prok", "euk>euk", "virus>host", "host>virus", "virus>virus")
VOCAB = {
    "record_kind": ("event", "claim_set"),
    "recipient_group": GROUPS,
    "donor_group": GROUPS + ("?",),
    "direction": DIRECTIONS,
    "category": ("protein_coding", "organelle_derived", "eukaryotic_transposon", "endogenous_viral_element"),
    "discovery": ("S", "N", "?"),
    "position_rule": ("integration", "replacement", "virus_chimera", "none"),
    "grade_transfer": ("A", "B", "C", "D"),
    "grade_donor": ("A", "B", "C", "D", "-"),
    "status": ("standing", "replicated", "disputed", "overturned", "retracted", "narrowed"),
    "truth_class": ("transfer", "contamination", "native", "artefact", "unresolved"),
    "exposure": ("L0", "L1", "L2", "L3", "L4", "L5", "?"),
    "in_pretraining": ("y", "n", "?"),
    "sealed_overlap": ("y", "n", "?"),
    "provisional_cover": ("y", "partial", "n"),
    "role": ("pool", "sealed", "prospective", "unassigned"),
}
SCENARIO_TAGS = ("near", "replacement", "ancient", "donor_unsampled", "partial", "serial", "high_foreign",
                 "chimeric_virus", "contamination_vs_integration", "multicopy", "intron_gain", "large_insert")
CLAIM_KEYS = ("foreign", "integrated", "germline", "donor", "timing")
EVIDENCE = ("ev_direct", "ev_position", "ev_phylogeny", "ev_population")
# RULES.md section 6: grade -> (transfer interval, donor interval); statuses that override the grade
INTERVALS = {
    "A": ((0.85, 0.99), (0.80, 0.99)),
    "B": ((0.70, 0.97), (0.60, 0.95)),
    "C": ((0.40, 0.90), (0.30, 0.85)),
    "D": ((0.10, 0.70), (0.05, 0.60)),
}
STATUS_INTERVALS = {"disputed": ((0.20, 0.80), (0.10, 0.70)), "overturned": ((0.01, 0.30), (0.01, 0.30))}
NEGATIVE_STATUS = ("disputed", "overturned")


def read(path):
    """The records of a registry draft (comment lines start with '#')."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader((line for line in f if not line.startswith("#")), delimiter="\t"))
    return rows


def _flag(value):
    return (value or "?").split(":", 1)[0].strip()


def derived_grade(r):
    """The grade the evidence lines imply (RULES.md section 4); no position evidence applies -> at most C."""
    direct, pos, phy = (_flag(r.get(k)) == "y" for k in ("ev_direct", "ev_position", "ev_phylogeny"))
    if r.get("position_rule") == "none":
        pos = False
    if direct and pos:
        return "A"
    if phy and pos:
        return "B"
    if phy:
        return "C"
    return "D"


def expected_interval(grade, status, which):
    """The interval RULES.md section 6 gives (which: 0 transfer, 1 donor); None when no donor is claimed."""
    if which == 1 and grade == "-":
        return None
    if status in STATUS_INTERVALS:
        return STATUS_INTERVALS[status][which]
    lo, hi = INTERVALS[grade][which]
    if status == "retracted":
        return (round(max(0.05, lo - 0.30), 2), hi)
    return (lo, hi)


def parse_interval(text):
    m = re.fullmatch(r"\s*(\d*\.\d+)\s*-\s*(\d*\.\d+)\s*", text or "")
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def check(rows):
    """(errors, warnings): errors break the rules; warnings are for the reviewer (for example a grade that differs
    from the one the evidence lines imply)."""
    errors, warnings = [], []
    seen = set()
    for i, r in enumerate(rows, 1):
        eid = r.get("event_id") or "row %d" % i
        missing = [c for c in COLUMNS if c not in r]
        if missing:
            errors.append("%s: missing columns %s" % (eid, ", ".join(missing)))
            continue
        if eid in seen:
            errors.append("%s: duplicate event_id" % eid)
        seen.add(eid)
        for col, allowed in VOCAB.items():
            if r[col] not in allowed:
                errors.append("%s: %s=%r not in %s" % (eid, col, r[col], "/".join(allowed)))
        for tag in filter(None, (t.strip() for t in r["scenario_tags"].split(";"))):
            if tag not in SCENARIO_TAGS:
                errors.append("%s: unknown scenario tag %r" % (eid, tag))
        claims = dict(kv.split("=", 1) for kv in r["claims"].split(";") if "=" in kv)
        if sorted(claims) != sorted(CLAIM_KEYS) or any(v not in ("y", "n", "?") for v in claims.values()):
            errors.append("%s: claims must give %s, each y/n/?" % (eid, "/".join(CLAIM_KEYS)))
        for col in EVIDENCE:
            if _flag(r[col]) not in ("y", "n", "?"):
                errors.append("%s: %s must start with y, n or ?" % (eid, col))
        if r["grade_transfer"] in ("A", "B") and r["position_rule"] == "none":
            errors.append("%s: grade %s needs position evidence, but position_rule is none" % (eid, r["grade_transfer"]))
        if r["grade_transfer"] in VOCAB["grade_transfer"] and r["position_rule"] in VOCAB["position_rule"]:
            d = derived_grade(r)
            if d != r["grade_transfer"]:
                warnings.append("%s: grade_transfer %s, the evidence lines imply %s" % (eid, r["grade_transfer"], d))
        for which, col, gcol in ((0, "p_transfer", "grade_transfer"), (1, "p_donor", "grade_donor")):
            if r[gcol] not in VOCAB[gcol] or r["status"] not in VOCAB["status"]:
                continue
            want = expected_interval(r[gcol], r["status"], which)
            if want is None:
                if r[col] not in ("", "-"):
                    errors.append("%s: %s given but no donor is claimed" % (eid, col))
                continue
            got = parse_interval(r[col])
            if got is None or not 0 < got[0] <= got[1] < 1:
                errors.append("%s: %s=%r is not an interval inside (0, 1)" % (eid, col, r[col]))
            elif got != want:
                errors.append("%s: %s=%s, the rules give %.2f-%.2f" % (eid, col, r[col], want[0], want[1]))
        if r["status"] in NEGATIVE_STATUS and r["truth_class"] == "transfer":
            warnings.append("%s: status %s with truth_class transfer" % (eid, r["status"]))
        if r["role"] != "unassigned":
            errors.append("%s: role %s, but roles are assigned only by the seeded hash (RULES.md section 7)" % (eid, r["role"]))
        if not re.fullmatch(r".+ \([a-z ]+\)", r["exclusion_taxon"]):
            errors.append("%s: exclusion_taxon must read 'name (rank)'" % eid)
        elif _taxon(r) not in _lineage(r):
            errors.append("%s: exclusion_lineage must end with (contain) the exclusion taxon" % eid)
        if r["exclusion_taxon"].endswith(("(species)", "(subspecies)", "(strain)")):
            errors.append("%s: the exclusion radius is at least the genus" % eid)
    return errors, warnings


def _taxon(r):
    return r["exclusion_taxon"].rsplit(" (", 1)[0].strip()


def _lineage(r):
    return [x.strip() for x in r["exclusion_lineage"].split(";") if x.strip()]


def _items(text, prefix=None):
    out = set()
    for x in (text or "").split(";"):
        x = x.strip()
        if prefix and x.startswith(prefix):
            x = x[len(prefix):]
        if x and x not in ("?", "-", "none") and (prefix is None or ":" not in x):
            out.add(x.lower())
    return out


def links(rows):
    """Edges (i, j, reason) of the association graph, i < j by row order."""
    edges = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            why = []
            ta, tb = _taxon(a), _taxon(b)
            if ta in _lineage(b) or tb in _lineage(a):
                why.append("radius %s / %s" % (ta, tb))
            shared = _items(a["papers"]) & _items(b["papers"])
            if shared:
                why.append("paper " + ", ".join(sorted(shared)))
            shared = _items(a["assembly"]) & _items(b["assembly"])
            if shared:
                why.append("assembly " + ", ".join(sorted(shared)))
            if why:
                edges.append((i, j, "; ".join(why)))
    return edges


def groups(rows, edges=None):
    """Connected components of the association graph, as lists of row indices, numbered by their first row."""
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in (links(rows) if edges is None else edges):
        parent[find(i)] = find(j)
    comp = defaultdict(list)
    for i in range(len(rows)):
        comp[find(i)].append(i)
    return sorted(comp.values(), key=lambda m: m[0])


def label(r):
    """The label kind a record contributes: a positive grade (A, B), C, D, or a negative status."""
    if r["status"] in NEGATIVE_STATUS:
        return r["status"]
    return r["grade_transfer"]


def summarise(rows):
    edges = links(rows)
    comps = groups(rows, edges)
    gid = {}
    for k, members in enumerate(comps, 1):
        for i in members:
            gid[i] = "G%02d" % k
    labels = ("A", "B", "C", "D", "disputed", "overturned")
    per_label = Counter(label(r) for r in rows)
    per_status = Counter(r["status"] for r in rows)
    cells = defaultdict(Counter)
    for r in rows:
        cells[(r["direction"], r["recipient_group"])][label(r)] += 1
    tags = defaultdict(Counter)
    for r in rows:
        for t in filter(None, (x.strip() for x in r["scenario_tags"].split(";"))):
            tags[t][label(r)] += 1
    by_route = defaultdict(Counter)
    for r in rows:
        by_route[r["discovery"]][label(r)] += 1
    recheck = [i for i, r in enumerate(rows) if _items(r["recheckable"]) - {"none"}]
    group_recheck = {gid[i] for i in recheck}
    by_label_recheck = Counter(label(rows[i]) for i in recheck)
    return dict(
        rules_version=RULES_VERSION,
        records=len(rows),
        groups=len(comps),
        group_members={"G%02d" % k: [rows[i]["event_id"] for i in m] for k, m in enumerate(comps, 1)},
        group_labels={"G%02d" % k: dict(Counter(label(rows[i]) for i in m)) for k, m in enumerate(comps, 1)},
        edges=[dict(a=rows[i]["event_id"], b=rows[j]["event_id"], why=w) for i, j, w in edges],
        per_label={lab: per_label.get(lab, 0) for lab in labels},
        groups_with_label={lab: len({gid[i] for i, r in enumerate(rows) if label(r) == lab}) for lab in labels},
        per_status=dict(per_status),
        discovery=dict(Counter(r["discovery"] for r in rows)),
        truth_class=dict(Counter(r["truth_class"] for r in rows)),
        category=dict(Counter(r["category"] for r in rows)),
        cells={"%s|%s" % k: dict(v) for k, v in sorted(cells.items())},
        scenario_tags={t: dict(tags.get(t, {})) for t in SCENARIO_TAGS},
        discovery_by_label={k: dict(v) for k, v in sorted(by_route.items())},
        recheckable=dict(records=len(recheck), share=round(len(recheck) / len(rows), 3) if rows else 0.0,
                         groups=len(group_recheck), by_label={lab: by_label_recheck.get(lab, 0) for lab in labels}),
        provisional_cover=dict(Counter(r["provisional_cover"] for r in rows)),
        in_pretraining=dict(Counter(r["in_pretraining"] for r in rows)),
        sealed_overlap=dict(Counter(r["sealed_overlap"] for r in rows)),
        exposure=dict(Counter(r["exposure"] for r in rows)),
        group_of={rows[i]["event_id"]: gid[i] for i in range(len(rows))},
    )


def markdown(rows, s, errors=(), warnings=()):
    labels = ("A", "B", "C", "D", "disputed", "overturned")
    out = ["# Event registry report (rules version %s; no scores)" % s["rules_version"], ""]
    out.append("%d records in %d groups (groups: exclusion radii that overlap, a shared paper or a shared assembly)."
               % (s["records"], s["groups"]))
    out += ["", "## 1. Association graph by exclusion lineage", "",
            "| group | records | labels | links inside the group |", "|---|---|---|---|"]
    by_id = {r["event_id"]: r for r in rows}
    for g, members in s["group_members"].items():
        inner = [e for e in s["edges"] if e["a"] in members]
        lab = ", ".join("%s %d" % (k, v) for k, v in sorted(s["group_labels"][g].items()))
        desc = "; ".join("%s (%s)" % (m, _taxon(by_id[m])) for m in members)
        out.append("| %s | %s | %s | %s |" % (g, desc, lab, "; ".join("%s-%s: %s" % (e["a"], e["b"], e["why"]) for e in inner) or "-"))
    out += ["", "## 2. Counts", "", "| label | records | groups |", "|---|---|---|"]
    for lab in labels:
        out.append("| %s | %d | %d |" % (lab, s["per_label"][lab], s["groups_with_label"][lab]))
    out += ["", "Label: the transfer grade, except that disputed and overturned records count under their status. "
            "Statuses: " + ", ".join("%s %d" % kv for kv in sorted(s["per_status"].items())) + ". Discovery route: "
            + ", ".join("%s %d" % kv for kv in sorted(s["discovery"].items())) + ". Truth class: "
            + ", ".join("%s %d" % kv for kv in sorted(s["truth_class"].items())) + ". Category: "
            + ", ".join("%s %d" % kv for kv in sorted(s["category"].items())) + "."]
    out += ["", "Scenario cells (direction x recipient group). Each filled cell lists its records by label; an empty "
            "cell has no record yet.", ""]
    out.append("| direction | " + " | ".join(GROUPS) + " |")
    out.append("|---" * (len(GROUPS) + 1) + "|")
    for d in DIRECTIONS:
        cells = []
        for g in GROUPS:
            c = s["cells"].get("%s|%s" % (d, g), {})
            cells.append(" ".join("%s%d" % (lab, c[lab]) for lab in labels if c.get(lab)) or "")
        out.append("| %s | %s |" % (d, " | ".join(cells)))
    out += ["", "Complex scenarios of charter section 1 (scenario tags) and discovery routes, by label:", "",
            "| tag | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for t in SCENARIO_TAGS:
        c = s["scenario_tags"][t]
        out.append("| %s | %s |" % (t, " | ".join(str(c.get(lab, "")) for lab in labels)))
    for k, c in s["discovery_by_label"].items():
        out.append("| route %s | %s |" % (k, " | ".join(str(c.get(lab, "")) for lab in labels)))
    rc = s["recheckable"]
    out += ["", "## 3. Recheckable evidence", "",
            "%d of %d records (%.0f%%) in %d of %d groups name evidence a fixed check can re-examine (read archives, "
            "a second assembly, population data). By label: %s." % (
                rc["records"], s["records"], 100 * rc["share"], rc["groups"], s["groups"],
                ", ".join("%s %d of %d" % (lab, rc["by_label"][lab], s["per_label"][lab]) for lab in labels if s["per_label"][lab]))]
    out += ["", "## 4. Exclusion and exposure", "",
            "Provisional list covers the radius: %s. Genomes inside the radius in the medium corpus: %s. Overlap with "
            "sealed genera or the sealed test part: %s. Exposure: %s." % tuple(
                ", ".join("%s %d" % kv for kv in sorted(s[k].items()))
                for k in ("provisional_cover", "in_pretraining", "sealed_overlap", "exposure"))]
    if errors or warnings:
        out += ["", "## Checks", ""]
        out += ["- error: " + e for e in errors]
        out += ["- warning: " + w for w in warnings]
    return "\n".join(out) + "\n"


DOUBLE_FIELDS = ("grade_transfer", "grade_donor", "status", "discovery", "exclusion_taxon", "truth_class")


def agreement(path):
    """Agreement of a double registration (RULES.md section 9): a table event_id, field, first, second, note."""
    per = defaultdict(lambda: [0, 0])
    events, differ = set(), []
    for r in read(path):
        if r["field"] not in DOUBLE_FIELDS:
            raise ValueError("double registration field %r is not one of %s" % (r["field"], ", ".join(DOUBLE_FIELDS)))
        events.add(r["event_id"])
        same = r["first"].strip() == r["second"].strip()
        per[r["field"]][0] += same
        per[r["field"]][1] += 1
        if not same:
            differ.append(dict(event_id=r["event_id"], field=r["field"], first=r["first"], second=r["second"], note=r["note"]))
    agree = sum(v[0] for v in per.values())
    cells = sum(v[1] for v in per.values())
    return dict(records=len(events), cells=cells, agree=agree,
                per_field={f: dict(agree=per[f][0], of=per[f][1]) for f in DOUBLE_FIELDS if f in per}, differ=differ)


def markdown_double(d):
    out = ["", "## 5. Double registration", "",
           "%d records registered twice; %d of %d fields agree (%.0f%%). The first registration is taken as compiled, "
           "before any reconciliation." % (d["records"], d["agree"], d["cells"], 100.0 * d["agree"] / max(1, d["cells"])), "",
           "| field | agree |", "|---|---|"]
    out += ["| %s | %d of %d |" % (f, v["agree"], v["of"]) for f, v in d["per_field"].items()]
    out += ["", "| record | field | first | second | note |", "|---|---|---|---|---|"]
    out += ["| %s | %s | %s | %s | %s |" % (x["event_id"], x["field"], x["first"], x["second"], x["note"]) for x in d["differ"]]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("registry")
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--double", help="a double-registration table (event_id, field, first, second, note)")
    a = ap.parse_args(argv)
    rows = read(a.registry)
    errors, warnings = check(rows)
    s = summarise(rows) if not any("missing columns" in e for e in errors) else {}
    s["errors"], s["warnings"] = errors, warnings
    double = agreement(a.double) if a.double else None
    if double:
        s["double"] = double
    if a.json:
        with open(a.json, "w") as f:
            json.dump(s, f, indent=1, sort_keys=True)
            f.write("\n")
    if a.md and "records" in s:
        with open(a.md, "w") as f:
            f.write(markdown(rows, s, errors, warnings) + (markdown_double(double) if double else ""))
    for e in errors:
        print("error:", e, file=sys.stderr)
    for w in warnings:
        print("warning:", w, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
