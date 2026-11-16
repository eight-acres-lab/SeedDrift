"""G2: nothing the benchmark judges enters training, checked by full NCBI lineage.

bench/hard/EXCLUDE.tsv (append-only) lists:
  genus   a sealed genus: excluded with its whole family (each name must resolve to exactly one bacterial or archaeal genus)
  order   an excluded order (the development order)
  rule    "panel genomes" (configs/exclusions/panel_assemblies.tsv: assemblies and their species),
          "test recipients" / "test donors" (bench/catalogue/events.tsv, once it exists), "generators" (static check)

entries(...) checks every genome, query, donor or catalogue id of a run: the taxid must resolve (fail closed), no node of
its lineage may be excluded, its accession must not be a panel assembly and its species must not be a panel species.
static(...) runs in T0: EXCLUDE.tsv parses, only grows against the previous ci-v* tag, and no training code imports a
benchmark generator.
"""
from __future__ import annotations

import ast
import csv
import io
import os
import subprocess

from ci import ROOT
from ci.gates import GateResult
from ci.hashing import sha256_file, sha256_json
from ci.taxonomy import UnknownTaxid

EXCLUDE = os.path.join(ROOT, "bench", "hard", "EXCLUDE.tsv")
PANELS = os.path.join(ROOT, "configs", "exclusions", "panel_assemblies.tsv")
CATALOGUE = os.path.join(ROOT, "bench", "catalogue", "events.tsv")
GENERATOR_MODULES = ("bench.hard.construct",)                 # evaluation generators: never imported by training code
TRAINING_CODE = ("research", "seeddrift/train", "ci/stages/train.py")
KNOWN_RULES = ("panel genomes", "generators", "test recipients", "test donors")


def parse_exclude(text: str):
    rows = [r for r in csv.DictReader(io.StringIO("".join(l for l in text.splitlines(True) if not l.startswith("#"))), delimiter="\t")]
    for r in rows:
        if r.get("kind") not in ("genus", "order", "rule") or not r.get("name") or not r.get("date"):
            raise ValueError("bad EXCLUDE.tsv row: %r" % r)
        if r["kind"] == "rule" and r["name"] not in KNOWN_RULES:
            raise ValueError("unknown rule in EXCLUDE.tsv: %s" % r["name"])
    return rows


def resolve(tax, rows):
    """(excluded taxids with reasons, problems)."""
    nodes, problems = {}, []
    for r in rows:
        if r["kind"] == "genus":
            ts = tax.genus_by_name(r["name"])
            if len(ts) != 1:
                problems.append(dict(row=r["name"], error="genus resolves to %s" % ts))
                continue
            fam = tax.at_rank(ts[0], "family")
            if not fam:
                problems.append(dict(row=r["name"], error="genus has no family"))
                continue
            nodes[ts[0]] = "sealed genus %s" % r["name"]
            nodes.setdefault(fam, "family of sealed genus %s" % r["name"])
        elif r["kind"] == "order":
            ts = tax.taxa_by_name(r["name"], "order")
            if len(ts) != 1:
                problems.append(dict(row=r["name"], error="order resolves to %s" % ts))
                continue
            nodes[ts[0]] = "excluded order %s" % r["name"]
    return nodes, problems


def read_panels(path=PANELS):
    acc, species = set(), set()
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            acc.add(r["assembly"].split(".")[0][4:])
            if r.get("species_taxid"):
                species.add(int(r["species_taxid"]))
    return acc, species


def read_catalogue(path=CATALOGUE):
    """Recipient taxids of catalogued events (the exclusion radius is applied by the catalogue's own column)."""
    if not os.path.exists(path):
        return None
    out = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out.append(r)
    return out


def entries(entries_, tax, exclude_path=EXCLUDE, panels_path=PANELS, catalogue_path=CATALOGUE) -> GateResult:
    """entries_: dicts with id, taxid, and optionally accession / paired_accession."""
    with open(exclude_path) as f:
        rows = parse_exclude(f.read())
    nodes, problems = resolve(tax, rows)
    rules = {r["name"] for r in rows if r["kind"] == "rule"}
    pan_acc, pan_species = set(), set()
    if "panel genomes" in rules:
        if not os.path.exists(panels_path):
            problems.append(dict(error="EXCLUDE.tsv has the panel rule but %s is missing" % os.path.relpath(panels_path, ROOT)))
        else:
            pan_acc, pan_species = read_panels(panels_path)
    cat = read_catalogue(catalogue_path)
    radius_nodes = {}
    if cat:
        for ev in cat:
            for col in ("recipient_exclusion_taxid",):
                if ev.get(col):
                    radius_nodes[int(ev[col])] = "test recipient of %s" % ev.get("event_id", "?")
    hits = []
    for e in entries_:
        try:
            lin = tax.lineage(int(e["taxid"]))
        except (UnknownTaxid, KeyError, ValueError) as err:
            hits.append(dict(id=e.get("id"), reason="taxid does not resolve: %s" % err))
            continue
        for t in lin:
            if t in nodes or t in radius_nodes:
                hits.append(dict(id=e.get("id"), taxid=int(e["taxid"]), node=t, reason=nodes.get(t) or radius_nodes.get(t)))
                break
        else:
            accs = [a for a in (e.get("accession"), e.get("paired_accession")) if a]
            if any(a.split(".")[0][4:] in pan_acc for a in accs):
                hits.append(dict(id=e.get("id"), reason="panel assembly"))
            elif pan_species and tax.at_rank(int(e["taxid"]), "species") in pan_species:
                hits.append(dict(id=e.get("id"), reason="species of a panel assembly"))
    status = "fail" if (hits or problems) else "pass"
    ev = dict(mode="entries", checked=len(entries_), excluded_nodes=len(nodes), panel_assemblies=len(pan_acc), panel_species=len(pan_species),
              catalogue="absent" if cat is None else len(cat), problems=problems, hits=hits[:200], n_hits=len(hits))
    return GateResult("G2", status, ev, sha256_json([sha256_file(exclude_path), sorted((e.get("id"), int(e.get("taxid", 0))) for e in entries_)]))


def _imports(path):
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
            for a in node.names:
                yield node.module + "." + a.name


def generator_imports(root=ROOT, training=TRAINING_CODE):
    bad = []
    for t in training:
        full = os.path.join(root, t)
        files = [full] if os.path.isfile(full) else [os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs if f.endswith(".py")] if os.path.isdir(full) else []
        for p in files:
            try:
                for name in _imports(p):
                    tails = {g.split(".")[-1] for g in GENERATOR_MODULES}          # also catches "import construct" after a path hack
                    if any(name == g or name.startswith(g + ".") for g in GENERATOR_MODULES) or name.split(".")[-1] in tails:
                        bad.append(dict(file=os.path.relpath(p, root), imports=name))
            except SyntaxError as err:
                bad.append(dict(file=os.path.relpath(p, root), error="syntax error: %s" % err))
    return bad


def previous_tag(root=ROOT):
    try:
        out = subprocess.run(["git", "describe", "--tags", "--abbrev=0", "--match", "ci-v*", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except subprocess.CalledProcessError:
        return ""


def append_only(old: str, new: str):
    """Problems if a data row of the old EXCLUDE.tsv is missing from the new one, or dates go backwards."""
    old_rows = [l for l in old.splitlines() if l and not l.startswith("#")]
    new_rows = [l for l in new.splitlines() if l and not l.startswith("#")]
    problems = [dict(removed=l) for l in old_rows if l not in new_rows]
    dates = [l.split("\t")[0] for l in new_rows[1:]]
    if dates != sorted(dates):
        problems.append(dict(error="dates are not in order"))
    return problems


def static(root=ROOT) -> GateResult:
    problems = []
    with open(os.path.join(root, "bench", "hard", "EXCLUDE.tsv")) as f:
        text = f.read()
    try:
        parse_exclude(text)
    except ValueError as err:
        problems.append(dict(error=str(err)))
    tag = previous_tag(root)
    if tag:
        try:
            old = subprocess.run(["git", "show", "%s:bench/hard/EXCLUDE.tsv" % tag], cwd=root, capture_output=True, text=True, check=True).stdout
            problems += append_only(old, text)
        except subprocess.CalledProcessError:
            problems.append(dict(error="cannot read EXCLUDE.tsv at %s" % tag))
    problems += generator_imports(root)
    return GateResult("G2", "fail" if problems else "pass", dict(mode="static", previous_tag=tag or "none (bootstrap)", problems=problems), sha256_json(text))
