"""Stage register (pipeline register): the register of the island reference set for benchmark v2, under the rule of
bench/hard/V2_REGISTER.md (committed before any register run). Metadata only: the island tables are read for their
accessions and organisms (no row is counted, no coordinate kept), the taxa from the store's taxid cache.

  parts          development (bench/islands/results/dev_genomes.tsv), development_unlisted (a development genus,
                 bench/islands/fetch.py DEV_GENERA, not in that list), sealed_v1 (bench/hard/results/sealed.tsv
                 without NC_010167.1), sealed_secondary (NC_010167.1), rest
  levels         the declared exposures (bench/hard/results/v2_exposure_declared.tsv: selector all, rest, sealed_v1,
                 or an accession; level L0-L5) and the names-only scan (register.scan_roots: paths naming a chromosome,
                 L3, L4 under score folders, L5 under evaluation folders; never file contents); a scan finding above a
                 chromosome's declared level stops the stage until it is declared in a new commit (V2_REGISTER.md rule 3)
  status         rest below L2: candidate; rest at L2 or above: demoted; the other parts keep their part
Outputs: data/register/reference_set.tsv (every chromosome: accession, organism, genus, family, order, part, level,
status), data/register/candidates.tsv (the candidates), data/register/report.json (counts per part and status, the
sha256 of the inputs, the tables' members and the outputs). Gate G3_register checks them.

Configuration (`register:`): store (the island store), scan_roots (default the store and the CI runs next to this
run), expect (the number of chromosomes, default 104), declared (default the file above).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import tarfile

from ci.hashing import sha256_bytes, sha256_file
from ci.stages.common import Stage, main_args, write_json

LEVELS = ("L0", "L1", "L2", "L3", "L4", "L5")
DEMOTE_FROM = "L2"                                   # bench/hard/V2_REGISTER.md, rule 1
SECONDARY = "NC_010167.1"
DECLARED = "bench/hard/results/v2_exposure_declared.tsv"
RULE = "bench/hard/V2_REGISTER.md"
TABLES = ("SupTable1.txt", "SupTable2.txt")


def _rows(text):
    return list(csv.DictReader(io.StringIO(text.replace("\r\n", "\n").replace("\r", "\n")), delimiter="\t"))


def reference_accessions(tar_path, want_sha):
    """{accession: organism} of the island tables (SupTable1 islands, SupTable2 backbone) read from the archive in
    memory: the accession and organism fields only, as a set (no row is counted, no coordinate kept). The archive must
    have the sha256 of provenance.json. Returns the map and the members' sha256."""
    if sha256_file(tar_path, use_sidecar=False) != want_sha:
        raise SystemExit("register: %s is not the archive of provenance.json (%s)" % (tar_path, want_sha))
    out, shas = {}, {}
    with tarfile.open(tar_path) as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if base in TABLES:
                raw = tf.extractfile(m).read()
                shas[base] = sha256_bytes(raw)
                for r in _rows(raw.decode("utf-8", errors="replace")):
                    acc, org = (r.get("Accession") or "").strip(), (r.get("Organism") or "").strip()
                    if acc:
                        if acc in out and out[acc] != org:
                            raise SystemExit("register: %s has two organisms in the tables" % acc)
                        out[acc] = org
    if set(shas) != set(TABLES):
        raise SystemExit("register: the archive lacks %s" % sorted(set(TABLES) - set(shas)))
    return out, shas


def load_declared(path):
    """Rows of the declared exposures: selector, level, source, evidence."""
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        if r["level"] not in LEVELS:
            raise SystemExit("register: unknown level %r in %s" % (r["level"], path))
    return rows


def parts(accs, organism, dev, sealed, dev_genera):
    """The part of every accession (see the module docstring)."""
    out = {}
    for a in accs:
        genus = organism[a].split()[0] if organism[a] else ""
        if a in dev:
            out[a] = "development"
        elif a == SECONDARY:
            out[a] = "sealed_secondary"
        elif a in sealed:
            out[a] = "sealed_v1"
        elif genus in dev_genera:
            out[a] = "development_unlisted"
        else:
            out[a] = "rest"
    return out


def levels(accs, part, declared):
    """The highest declared level of every accession, with the rows that set it."""
    lv = {a: (-1, []) for a in accs}
    for r in declared:
        sel = r["selector"]
        hit = [a for a in accs if sel == "all" or sel == a or sel == part[a]]
        if sel not in ("all", "rest", "sealed_v1", "development", "development_unlisted", "sealed_secondary") and sel not in accs:
            raise SystemExit("register: declared selector %r is neither a part nor an accession of the set" % sel)
        for a in hit:
            k = LEVELS.index(r["level"])
            if k > lv[a][0]:
                lv[a] = (k, [r["source"]])
            elif k == lv[a][0]:
                lv[a][1].append(r["source"])
    return {a: (LEVELS[k] if k >= 0 else "L0", src) for a, (k, src) in lv.items()}


L4_PARTS = ("/scores/", "/scores_", "/raw/", "/views/", "/dev_composition/", "/dev24_v/", "/dev_views/")
L5_PARTS = ("/eval/",)


def path_level(path):
    """The level of a path that names a chromosome (V2_REGISTER.md, Scans)."""
    p = "/" + path.replace(os.sep, "/") + "/"
    if any(x in p for x in L5_PARTS):
        return "L5"
    if any(x in p for x in L4_PARTS):
        return "L4"
    return "L3"


def scan_names(roots, accs, skip=()):
    """Every path (files and folders) under roots whose name contains an accession of accs (with or without its
    version): {accession: [(path, level)]}, and the number of paths walked. Names only; no file is opened."""
    base = {a.split(".")[0]: a for a in accs}
    rx = re.compile(r"(?<![A-Za-z0-9])(%s)(?:\.\d+)?(?![0-9])" % "|".join(sorted(map(re.escape, base))))
    found, walked = {}, 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in skip]
            for name in dirnames + filenames:
                walked += 1
                for m in rx.finditer(name):
                    p = os.path.join(dirpath, name)
                    found.setdefault(base[m.group(1)], []).append((p, path_level(os.path.relpath(p, root))))
    return found, walked


def main():
    run_dir = main_args()
    with Stage(run_dir, "register", requires=("pre",)) as st:
        cfg = st.config
        rc = cfg.get("register") or {}
        store = rc["store"]
        with open(st.repo("bench/islands/results/provenance.json")) as f:
            prov = json.load(f)
        organism, member_sha = reference_accessions(os.path.join(store, "Sup_tables.tar.gz"), prov["tables_sha256"])
        accs = sorted(organism)
        expect = int(rc.get("expect", 104))
        if len(accs) != expect:
            raise SystemExit("register: %d chromosomes in the tables, expected %d" % (len(accs), expect))
        extracted = {}
        for t in TABLES:                                  # the extracted copies the stages read: equal to the archive?
            p = os.path.join(store, t)
            extracted[t] = os.path.exists(p) and sha256_file(p, use_sidecar=False) == member_sha[t]
        with open(st.repo("bench/islands/results/dev_genomes.tsv")) as f:
            dev = {r["accession"] for r in csv.DictReader(f, delimiter="\t")}
        with open(st.repo("bench/hard/results/sealed.tsv")) as f:
            sealed = {r["accession"] for r in csv.DictReader(f, delimiter="\t")}
        import importlib.util
        spec = importlib.util.spec_from_file_location("bench_islands_fetch", st.repo("bench/islands/fetch.py"))
        fetch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fetch)
        part = parts(accs, organism, dev, sealed, set(fetch.DEV_GENERA))
        declared_path = st.repo(rc.get("declared", DECLARED))
        lv = levels(accs, part, load_declared(declared_path))
        roots = list(rc.get("scan_roots") or [store, os.path.dirname(run_dir)])
        found, walked = scan_names(roots, accs, skip={run_dir})          # this run's own outputs name every chromosome
        unexplained = []
        for a, hits in sorted(found.items()):
            top = max(LEVELS.index(lvl) for _, lvl in hits)
            if part[a] == "rest" and top > LEVELS.index(lv[a][0]):
                unexplained += [dict(accession=a, path=p, level=lvl) for p, lvl in hits if LEVELS.index(lvl) > LEVELS.index(lv[a][0])]
        if unexplained:
            raise SystemExit("register: exposures the scan found above the declared levels (declare them in a new commit, V2_REGISTER.md "
                             "rule 3): %s" % json.dumps(unexplained[:20]))
        with open(os.path.join(store, "panel", "island_set_taxa.json")) as f:
            taxid = {k: int(v) for k, v in json.load(f).items()}
        missing = sorted(set(accs) - set(taxid))
        if missing:
            raise SystemExit("register: no taxid for %s" % missing[:10])
        from ci.taxonomy import Taxonomy
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        name = lambda t, rank: (tax.name(tax.at_rank(t, rank)) if tax.at_rank(t, rank) else "")
        rows = []
        for a in accs:
            level, src = lv[a]
            status = part[a]
            if part[a] == "rest":
                status = "demoted" if LEVELS.index(level) >= LEVELS.index(DEMOTE_FROM) else "candidate"
            rows.append(dict(accession=a, organism=organism[a], genus=name(taxid[a], "genus"), family=name(taxid[a], "family"),
                             order=name(taxid[a], "order"), part=part[a], level=level, status=status, level_from="; ".join(sorted(set(src)))))
        cols = ("accession", "organism", "genus", "family", "order", "part", "level", "status", "level_from")
        base = os.path.join(run_dir, "data", "register")
        os.makedirs(base, exist_ok=True)
        ref = os.path.join(base, "reference_set.tsv")
        with open(ref, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(r[c] for c in cols) + "\n")
        cand = os.path.join(base, "candidates.tsv")
        with open(cand, "w") as f:
            f.write("accession\torganism\tgenus\tfamily\n")
            for r in rows:
                if r["status"] == "candidate":
                    f.write("\t".join(r[c] for c in ("accession", "organism", "genus", "family")) + "\n")
        count = lambda key: {v: sum(1 for r in rows if r[key] == v) for v in sorted({r[key] for r in rows})}
        scan = dict(roots=roots, paths_walked=walked, findings={part_name: sum(len(v) for a, v in found.items() if part[a] == part_name)
                                                                for part_name in sorted(set(part.values()))},
                    rest_findings={a: v for a, v in sorted(found.items()) if part[a] == "rest"})
        rep = dict(note=__doc__.strip().splitlines()[0], chromosomes=len(rows), parts=count("part"), status=count("status"), levels=count("level"), scan=scan,
                   demote_from=DEMOTE_FROM, tables_sha256=prov["tables_sha256"], members_sha256=member_sha, extracted_equal=extracted,
                   inputs={p: sha256_file(st.repo(p), use_sidecar=False) for p in (RULE, rc.get("declared", DECLARED), "bench/islands/results/dev_genomes.tsv",
                                                                                     "bench/hard/results/sealed.tsv", "bench/islands/fetch.py")},
                   taxid_cache_sha256=sha256_file(os.path.join(store, "panel", "island_set_taxa.json"), use_sidecar=False),
                   outputs={os.path.basename(p): sha256_file(p, use_sidecar=False) for p in (ref, cand)},
                   sealed_v1_genera=sorted({r["genus"] for r in rows if r["part"] == "sealed_v1"}))
        rp = os.path.join(base, "report.json")
        write_json(rp, rep)
        st.output("register_reference_set", ref)
        st.output("register_candidates", cand)
        st.output("register_report", rp)
        st.extra.update(register_parts=rep["parts"], register_status=rep["status"])


if __name__ == "__main__":
    main()
