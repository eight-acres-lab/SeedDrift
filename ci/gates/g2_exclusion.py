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
TRAINING_CODE = ("research", "seeddrift", "ci/stages/train.py", "ci/stages/corpus.py")
KNOWN_RULES = ("panel genomes", "generators", "test recipients", "test donors")
REQUIRED_RULES = ("panel genomes", "generators")                 # removing either would switch a check off


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
        for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
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


class Excluder:
    """The exclusion rules resolved against one taxonomy: reason(taxid, accessions) is None for an allowed genome and
    the reason otherwise; host_reason(taxid) checks only the lineage (the host of a virus, which need not be a species)."""

    def __init__(self, tax, exclude_path=EXCLUDE, panels_path=PANELS, catalogue_path=CATALOGUE, extra_paths=()):
        self.tax, self.exclude_path, self.extra_paths = tax, exclude_path, list(extra_paths)
        with open(exclude_path) as f:
            rows = parse_exclude(f.read())
        self.nodes, self.problems = resolve(tax, rows)
        self.rules = {r["name"] for r in rows if r["kind"] == "rule"}
        for rule in REQUIRED_RULES:
            if rule not in self.rules:
                self.problems.append(dict(error="EXCLUDE.tsv lacks the rule %r" % rule))
        self.pan_acc, self.pan_species = set(), set()
        if "panel genomes" in self.rules:
            if not os.path.exists(panels_path):
                self.problems.append(dict(error="EXCLUDE.tsv has the panel rule but %s is missing" % os.path.relpath(panels_path, ROOT)))
            else:
                self.pan_acc, raw_species = read_panels(panels_path)
                for t in raw_species:                            # merged ids follow their new node; unknown ones stay as given
                    try:
                        self.pan_species.add(tax.resolve(t))
                    except Exception:  # noqa: BLE001
                        self.pan_species.add(t)
        self.catalogue = read_catalogue(catalogue_path)
        self.radius_nodes = {}
        for ev in self.catalogue or []:
            for col in ("recipient_exclusion_taxid",):
                if ev.get(col):
                    try:
                        self.radius_nodes[tax.resolve(int(ev[col]))] = "test recipient of %s" % ev.get("event_id", "?")
                    except Exception:  # noqa: BLE001 - a catalogue taxid that does not resolve fails the gate
                        self.problems.append(dict(error="catalogue taxid %s does not resolve" % ev[col]))
        # extra lists named by a profile (configs/exclusions/pilot_recipients.tsv: provisional test recipients of pilot runs);
        # they only add exclusions, never remove one
        for path in self.extra_paths:
            if not os.path.exists(path):
                self.problems.append(dict(error="extra exclusion list %s is missing" % path))
                continue
            with open(path) as f:
                for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
                    try:
                        self.radius_nodes.setdefault(tax.resolve(int(r["taxid"])), "provisional test recipient %s (%s)" % (r.get("name", ""), os.path.basename(path)))
                    except Exception:  # noqa: BLE001
                        self.problems.append(dict(error="extra exclusion taxid %s (%s) does not resolve" % (r.get("taxid"), r.get("name"))))

    def _lineage_hit(self, taxid):
        for t in self.tax.lineage(int(taxid)):
            if t in self.nodes or t in self.radius_nodes:
                return t, self.nodes.get(t) or self.radius_nodes.get(t)
        return None

    def reason(self, taxid, accessions=()):
        """None, or (reason, node) for an excluded genome; a taxid that does not resolve or lies above species rank is excluded."""
        try:
            self.tax.lineage(int(taxid))
        except (UnknownTaxid, KeyError, ValueError) as err:
            return "taxid does not resolve: %s" % err, None
        if not self.tax.at_rank(int(taxid), "species"):
            return "taxid is above species rank (the panel-species check needs a species)", None
        hit = self._lineage_hit(taxid)
        if hit:
            return hit[1], hit[0]
        if any(a.split(".")[0][4:] in self.pan_acc for a in accessions if a):
            return "panel assembly", None
        if self.pan_species and self.tax.at_rank(int(taxid), "species") in self.pan_species:
            return "species of a panel assembly", None
        return None

    def host_reason(self, taxid):
        try:
            hit = self._lineage_hit(taxid)
        except (UnknownTaxid, KeyError, ValueError) as err:
            return "host taxid does not resolve: %s" % err
        return "host excluded: %s" % hit[1] if hit else None


def entries(entries_, tax, exclude_path=EXCLUDE, panels_path=PANELS, catalogue_path=CATALOGUE, excluder=None, extra_paths=()) -> GateResult:
    """entries_: dicts with id, taxid, and optionally accession / paired_accession / host_taxid (a virus's host is
    checked by lineage too: a phage of an excluded host shares sequence with it)."""
    ex = excluder or Excluder(tax, exclude_path, panels_path, catalogue_path, extra_paths)
    hits = []
    for e in entries_:
        r = ex.reason(e["taxid"], [e.get("accession"), e.get("paired_accession")])
        if r:
            reason, node = r
            if node is not None:                                 # an excluded node of the lineage
                hits.append(dict(id=e.get("id"), taxid=int(e["taxid"]), node=node, reason=reason))
            elif reason.startswith("taxid is above species"):
                hits.append(dict(id=e.get("id"), taxid=int(e["taxid"]), reason=reason))
            else:                                                # unresolved taxid, panel assembly or panel species
                hits.append(dict(id=e.get("id"), reason=reason))
            continue
        if e.get("host_taxid"):
            h = ex.host_reason(e["host_taxid"])
            if h:
                hits.append(dict(id=e.get("id"), host_taxid=int(e["host_taxid"]), reason=h))
    status = "fail" if (hits or ex.problems) else "pass"
    cat = ex.catalogue
    ev = dict(mode="entries", checked=len(entries_), excluded_nodes=len(ex.nodes), panel_assemblies=len(ex.pan_acc), panel_species=len(ex.pan_species),
              catalogue="absent" if cat is None else len(cat), problems=ex.problems, hits=hits[:200], n_hits=len(hits))
    ev["extra_exclusions"] = [os.path.basename(p) for p in ex.extra_paths]
    return GateResult("G2", status, ev, sha256_json([sha256_file(ex.exclude_path), [sha256_file(p) for p in ex.extra_paths if os.path.exists(p)],
                                                     sorted((e.get("id"), int(e.get("taxid", 0))) for e in entries_)]))


def _imports(path, root):
    """Module names imported by a file: absolute and relative imports (resolved against the file's package) and
    literal importlib.import_module / __import__ calls."""
    with open(path) as f:
        tree = ast.parse(f.read(), filename=path)
    pkg = os.path.relpath(os.path.dirname(path), root).replace(os.sep, ".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = pkg.split(".")
                parts = parts[:len(parts) - (node.level - 1)] if node.level > 1 else parts
                base = ".".join([p for p in parts if p and p != "."] + ([node.module] if node.module else []))
            if base:
                yield base
            for a in node.names:
                yield (base + "." if base else "") + a.name
        elif isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
            if name in ("import_module", "__import__", "run_module") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                yield node.args[0].value


def generator_imports(root=ROOT, training=TRAINING_CODE):
    bad = []
    for t in training:
        full = os.path.join(root, t)
        files = [full] if os.path.isfile(full) else [os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs if f.endswith(".py")] if os.path.isdir(full) else []
        for p in files:
            try:
                for name in _imports(p, root):
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
