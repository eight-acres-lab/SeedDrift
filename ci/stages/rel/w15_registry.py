"""Stage rel_w15_registry (package W15; pipeline rel_events): the event registry checked on the server, without any
score. `python -m ci.stages.rel.w15_registry RUN_DIR`.

Four things, all from the registry draft the profile names (bench/rel/registry/events_*.tsv) and none from any model:
1. the draft's own checks (bench/rel/registry/registry.py, the rules of RULES.md);
2. every exclusion taxon looked up in NCBI taxonomy (the run's taxdump): its taxid, rank and lineage, and the names of
   the draft's lineage that NCBI does not have;
3. whether genomes inside each exclusion radius are in the medium corpus. Only the train and held-out splits of the
   source's training-data index are read; the sealed test split is never looked at (its genomes are not opened, and
   their taxa are not counted per radius). Overlap with the sealed island genera (and their families) of
   bench/hard/EXCLUDE.tsv is reported from that list alone;
4. with `fetch`, the papers: Europe PMC metadata of every DOI in the draft, the open full text where Europe PMC has
   it, and the supplementary files of the DOIs the profile lists, each file with its URL, size and sha256. Papers are
   stored under the run's data folder, never in the repository.

Profile block rel_w15_registry: registry (repository path of the draft, or a list of drafts read as one registry), source (the run directory of a passed selfsup
run, for its training-data index; optional), fetch (bool), supplements (DOIs whose supplements are fetched),
max_supplement_mb (default 50), pause_s (default 0.5).
Output data/rel_w15/registry/report.json and report.md; papers in data/rel_w15/registry/papers/.
Gate G4_rel_w15_registry: the draft breaks no rule, every role is unassigned, nothing sits at bench/catalogue/events.tsv,
the report's sha256 of the draft matches the draft.
"""
from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from bench.rel.registry import registry as reg
from ci.gates import GateResult, require
from ci.hashing import sha256_file
from ci.stages.common import main_args, write_json
from ci.stages.rel import common

STAGE = "rel_w15_registry"
OUT = os.path.join("data", "rel_w15", "registry")
KEYS = {"registry", "source", "fetch", "supplements", "max_supplement_mb", "pause_s"}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
USER_AGENT = "literature-registry-fetch/0.2 (python urllib)"
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;()\[\]]+")
CATALOGUE = os.path.join("bench", "catalogue", "events.tsv")
PRETRAIN_SPLITS = ("train", "heldout")


def settings(cfg) -> dict:
    c = dict(cfg.get(STAGE) or {})
    unknown = set(c) - KEYS
    if unknown or not c.get("registry"):
        raise SystemExit("%s: the profile names registry (and only %s): %s" % (STAGE, sorted(KEYS), sorted(unknown)))
    paths = [str(x) for x in (c["registry"] if isinstance(c["registry"], list) else [c["registry"]])]
    if any(os.path.normpath(x) == CATALOGUE for x in paths):
        raise SystemExit("%s: the draft never lives at %s (gate G2 reads that file)" % (STAGE, CATALOGUE))
    return dict(registry=paths, source=c.get("source"), fetch=bool(c.get("fetch", False)),
                supplements=[d.lower() for d in c.get("supplements") or []],
                max_supplement_mb=float(c.get("max_supplement_mb", 50)), pause_s=float(c.get("pause_s", 0.5)))


def dois(rows) -> list:
    """Every DOI named in the draft's papers and later columns, lower case, trailing punctuation removed."""
    out = []
    for r in rows:
        for col in ("papers", "later", "exclusion_basis", "notes"):
            for m in DOI_RE.finditer(r.get(col) or ""):
                d = m.group(0).rstrip(".,:").lower()
                if d not in out:
                    out.append(d)
    return out


def _rank(r) -> str:
    m = re.search(r"\(([a-z ]+)\)\s*$", r["exclusion_taxon"])
    return m.group(1) if m else ""


def taxa(rows, tax) -> dict:
    """event_id -> the NCBI reading of the exclusion taxon: status ok / rank_differs / ambiguous / unresolved."""
    out = {}
    for r in rows:
        name, rank = reg._taxon(r), _rank(r)
        cands = list(tax.by_name.get(name, []))
        if rank and rank != "no rank":
            same = [t for t in cands if tax.rank.get(t) == rank]
        else:
            same = cands
        wanted = set(reg._lineage(r))
        scored = sorted(((len(wanted & {tax.name(x) for x in tax.lineage(t)}), t) for t in same), reverse=True)
        rec = dict(name=name, rank=rank, candidates=len(cands))
        if not cands:
            rec["status"] = "unresolved"
        elif not same:
            rec.update(status="rank_differs", ncbi_ranks=sorted({tax.rank.get(t, "?") for t in cands}))
        elif len(scored) > 1 and scored[0][0] == scored[1][0]:
            rec.update(status="ambiguous", taxids=[t for _, t in scored])
        else:
            t = scored[0][1]
            lin = [tax.name(x) for x in reversed(tax.lineage(t))]
            rec.update(status="ok", taxid=t, ncbi_rank=tax.rank.get(t), ncbi_lineage=lin,
                       names_not_in_ncbi=[x for x in reg._lineage(r) if x not in lin])
            stated = (r.get("recipient_exclusion_taxid") or "").strip()
            if stated and int(stated) != t:
                rec["stated_taxid_differs"] = int(stated)
        out[r["event_id"]] = rec
    return out


def pretraining(found, index, tax) -> dict:
    """event_id -> genomes inside the radius per split of the training-data index, train and held-out splits only."""
    genomes = [(m["split"], int(m["taxid"])) for m in index["genomes"] if m.get("split") in PRETRAIN_SPLITS]
    lineages = []
    for split, t in genomes:
        try:
            lineages.append((split, set(tax.lineage(t))))
        except Exception:  # noqa: BLE001 - a genome whose taxid does not resolve is counted apart
            lineages.append((split, None))
    out = {}
    for eid, rec in found.items():
        if rec.get("status") != "ok":
            continue
        counts = {s: 0 for s in PRETRAIN_SPLITS}
        for split, lin in lineages:
            if lin and rec["taxid"] in lin:
                counts[split] += 1
        out[eid] = dict(counts, in_pretraining="y" if counts["train"] else "n",
                        unresolved_genomes=sum(1 for _, lin in lineages if lin is None))
    return out


def sealed_genera(tax, exclude_path) -> dict:
    """Taxid -> reason, for the sealed island genera of EXCLUDE.tsv and their families."""
    out = {}
    with open(exclude_path) as f:
        rows = csv.DictReader((x for x in f if not x.startswith("#")), delimiter="\t")
        for r in rows:
            if r["kind"] != "genus" or not r["reason"].startswith("sealed genus"):
                continue
            for g in tax.genus_by_name(r["name"]):
                out[g] = "sealed genus %s" % r["name"]
                fam = tax.at_rank(g, "family")
                if fam:
                    out.setdefault(fam, "family of sealed genus %s" % r["name"])
    return out


def sealed_overlap(found, sealed, tax) -> dict:
    """event_id -> y when the radius contains a sealed genus or family, or lies inside one."""
    out = {}
    for eid, rec in found.items():
        if rec.get("status") != "ok":
            continue
        t = rec["taxid"]
        hits = [why for s, why in sealed.items() if s in set(tax.lineage(t)) or t in set(tax.lineage(s))]
        out[eid] = dict(sealed_overlap="y" if hits else "n", reasons=hits)
    return out


def _get(url, opener, max_bytes=None):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener(req, timeout=60) as resp:
        data = resp.read(max_bytes + 1 if max_bytes else -1)
        return resp.status, data


def _store(dest, name, data):
    os.makedirs(dest, exist_ok=True)
    p = os.path.join(dest, name)
    with open(p, "wb") as f:
        f.write(data)
    return dict(file=name, bytes=len(data), sha256=sha256_file(p, use_sidecar=False))


def fetch(doi, dest, opener=urllib.request.urlopen, supplements=False, max_supplement_mb=50.0, pause_s=0.5) -> dict:
    """Europe PMC metadata for one DOI, the open full text when Europe PMC has it, the supplements when asked."""
    rec = dict(doi=doi, when=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    url = EPMC + "search?" + urllib.parse.urlencode(dict(query='DOI:"%s"' % doi, resultType="core", format="json"))
    try:
        status, data = _get(url, opener)
        rec["meta"] = dict(_store(dest, "meta.json", data), url=url, status=status)
        hits = (json.loads(data).get("resultList") or {}).get("result") or []
        hit = hits[0] if hits else {}
        rec.update(found=bool(hits), pmcid=hit.get("pmcid"), open_access=hit.get("isOpenAccess"), pmid=hit.get("pmid"),
                   title=hit.get("title"), year=hit.get("pubYear"))
    except (urllib.error.URLError, OSError, ValueError) as err:
        rec["meta"] = dict(url=url, error=str(err)[:200])
        return rec
    for what, path, cap in (("fulltext", "fullTextXML", None), ("supplements", "supplementaryFiles", max_supplement_mb)):
        if not rec.get("pmcid") or (what == "supplements" and not supplements):
            continue
        time.sleep(pause_s)
        url = EPMC + "%s/%s" % (rec["pmcid"], path)
        max_bytes = int(cap * 1e6) if cap else None
        try:
            status, data = _get(url, opener, max_bytes)
            if max_bytes and len(data) > max_bytes:
                rec[what] = dict(url=url, error="larger than %.0f MB, not stored" % cap)
            else:
                rec[what] = dict(_store(dest, "fulltext.xml" if what == "fulltext" else "supplements.zip", data), url=url, status=status)
        except urllib.error.HTTPError as err:
            rec[what] = dict(url=url, status=err.code, error=str(err)[:200])
        except (urllib.error.URLError, OSError) as err:
            rec[what] = dict(url=url, error=str(err)[:200])
    return rec


def _folder(doi):
    return re.sub(r"[^a-z0-9._-]+", "_", doi.lower())


def build(st, cfg, run_dir):
    s = settings(cfg)
    t0 = time.time()
    paths = [st.repo(x) for x in s["registry"]]
    rows = [r for x in paths for r in reg.read(x)]
    errors, warnings = reg.check(rows)
    summary_ = reg.summarise(rows) if not errors else {}
    from ci.taxonomy import Taxonomy
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    found = taxa(rows, tax)
    pre = {}
    if s["source"]:
        from ci.stages.rel import w2_v0
        src = w2_v0.source(s["source"])
        pre = pretraining(found, src["index"], tax)
        st.extra["source"] = src["provenance"]
    overlap = sealed_overlap(found, sealed_genera(tax, st.repo(os.path.join("bench", "hard", "EXCLUDE.tsv"))), tax)
    papers = []
    if s["fetch"]:
        base = st.path("rel_w15", "registry", "papers", "x")
        for d in dois(rows):
            papers.append(fetch(d, os.path.join(os.path.dirname(base), _folder(d)), supplements=d in s["supplements"],
                                max_supplement_mb=s["max_supplement_mb"], pause_s=s["pause_s"]))
            time.sleep(s["pause_s"])
    doc = dict(format="rel-w15-registry", rules_version=reg.RULES_VERSION, registry=s["registry"],
               registry_sha256={x: sha256_file(st.repo(x), use_sidecar=False) for x in s["registry"]}, records=len(rows), errors=errors, warnings=warnings,
               summary=summary_, taxa=found, pretraining=pre, sealed_overlap=overlap, papers=papers,
               note="no model output and no score is read; the sealed test split of the medium corpus is not looked at",
               seconds=round(time.time() - t0, 1))
    out = st.path("rel_w15", "registry", "report.json")
    write_json(out, doc)
    st.output("report", out)
    md = st.path("rel_w15", "registry", "report.md")
    with open(md, "w") as f:
        f.write(reg.markdown(rows, summary_, errors, warnings) if summary_ else "# Event registry report\n\nThe draft breaks rules.\n")
        f.write(markdown_server(doc))
    st.output("report_md", md)
    st.extra.update(records=len(rows), errors=len(errors), papers=len(papers),
                    taxa={k: sum(1 for v in found.values() if v["status"] == k) for k in ("ok", "rank_differs", "ambiguous", "unresolved")})
    return doc


def markdown_server(doc) -> str:
    out = ["", "## Server checks", "", "| record | NCBI | taxid | names not in NCBI | train | held-out | sealed genera |",
           "|---|---|---|---|---|---|---|"]
    for eid, rec in sorted(doc["taxa"].items()):
        pre = doc["pretraining"].get(eid, {})
        ov = doc["sealed_overlap"].get(eid, {})
        out.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            eid, rec["status"], rec.get("taxid", ""), "; ".join(rec.get("names_not_in_ncbi", [])),
            pre.get("train", ""), pre.get("heldout", ""), ov.get("sealed_overlap", "")))
    if doc["papers"]:
        ok = sum(1 for p in doc["papers"] if (p.get("fulltext") or {}).get("sha256"))
        out += ["", "%d DOIs looked up; %d with an open full text stored; %d with supplements stored." % (
            len(doc["papers"]), ok, sum(1 for p in doc["papers"] if (p.get("supplements") or {}).get("sha256")))]
    return "\n".join(out) + "\n"


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        doc = json.load(f)
    bad = [dict(error=e) for e in doc.get("errors") or []]
    for rel, sha in (doc.get("registry_sha256") or {}).items():
        path = rel if os.path.isabs(rel) else os.path.join(common.ROOT, rel)
        if sha256_file(path, use_sidecar=False) != sha:
            bad.append(dict(error="%s is not the file the report was made from" % rel))
    if os.path.exists(os.path.join(common.ROOT, CATALOGUE)):
        bad.append(dict(error="%s exists: the draft must not live there" % CATALOGUE))
    require(GateResult("G4_rel_w15_registry", "fail" if bad else "pass",
                       dict(problems=bad[:50], records=doc.get("records"), warnings=len(doc.get("warnings") or []),
                            taxa={k: sum(1 for v in doc["taxa"].values() if v["status"] == k) for k in ("ok", "rank_differs", "ambiguous", "unresolved")})),
            run_dir)
    return sha256_file(p, use_sidecar=False)


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        doc = json.load(f)
    s = doc.get("summary") or {}
    md = ["| label | records |", "|---|---|"] + ["| %s | %d |" % kv for kv in (s.get("per_label") or {}).items()]
    return dict(json=dict(records=doc["records"], errors=len(doc["errors"]), papers=len(doc["papers"]),
                          rules_version=doc["rules_version"]), md=md)


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, STAGE) as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
