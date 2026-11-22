"""Stage rel_w13_scaffold (package W13; pipeline rel_corpus): a snapshot of the prokaryote assemblies at the scaffold
level, metadata only, with the quality of every one judged by fixed thresholds (task breakdown 8.11: "原核 scaffold 级，
同时只拉元数据，带组装质量的门槛"). `python -m ci.stages.rel.w13_scaffold RUN_DIR`. No sequence is downloaded.

Source: NCBI's GenBank assembly summaries of bacteria and archaea (genomes/genbank/<group>/assembly_summary.txt: one row
per assembly, with its level, size, contig count and the reasons RefSeq excludes it), streamed into the CI's content
store (<data_root>/cas) with their size and sha256 checked against the server's length; the lock records URL, server
date, bytes and sha256. A row at the scaffold level (latest version, full representation) passes when
  - RefSeq names none of the quality reasons (`quality_reasons`) among its exclusions (`excluded_from_refseq`);
  - it has at most `max_contigs` contigs;
  - its length is within `length_band` of the median length of the complete and chromosome-level assemblies of its
    species that pass the same exclusion rule, or, when its species has none, within `genus_band` of its genus's
    median (taxonomy of the profile); a row with neither is left out (no length to compare with).
Every scaffold row is kept with its verdict (`quality`: pass, or the first reason it fails), so a census can count both.
Environmental samples (an exclusion "derived from metagenome" or "derived from environmental source") are marked, not
left out. Outputs under data/rel_w13/scaffold/: <group>_scaffold.tsv (the fields of stage rel_w13_catalog and contigs,
excluded_from_refseq, reference_length, length_ratio, quality), lock.yaml and manifest.json (written last). Gate
G4_rel_w13_scaffold.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import io
import json
import os
import shutil
import statistics
import time
import urllib.request

import yaml

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w13_catalog

OUT = os.path.join("data", "rel_w13", "scaffold")
SUMMARY = "https://ftp.ncbi.nlm.nih.gov/genomes/genbank/%s/assembly_summary.txt"
GROUPS = ("bacteria", "archaea")
QUALITY_REASONS = ("abnormal gene to sequence ratio", "chimeric", "contaminated", "fragmented assembly", "genome length too large", "genome length too small",
                   "hybrid", "low contig N50", "low gene count", "low quality sequence", "many frameshifted proteins", "misassembled", "missing ribosomal protein genes",
                   "missing rRNA genes", "missing tRNA genes", "mixed culture", "partial", "sequence duplications", "unverified source organism", "refseq annotation failed")
ENVIRONMENTAL = ("derived from metagenome", "derived from environmental source")
FIELDS = list(w13_catalog.FIELDS) + ["contigs", "excluded_from_refseq", "reference_length", "length_ratio", "quality"]
KEYS = {"groups", "max_contigs", "length_band", "genus_band", "quality_reasons"}


def settings(cfg) -> dict:
    c = cfg.get("rel_w13_scaffold") or {}
    unknown = set(c) - KEYS
    if unknown:
        raise SystemExit("rel_w13_scaffold: unknown keys %s" % sorted(unknown))
    s = dict(groups=[str(g) for g in c.get("groups") or GROUPS], max_contigs=int(c.get("max_contigs", 100)),
             length_band=[float(x) for x in c.get("length_band") or (0.85, 1.15)], genus_band=[float(x) for x in c.get("genus_band") or (0.7, 1.3)],
             quality_reasons=[str(x) for x in c.get("quality_reasons") or QUALITY_REASONS])
    if set(s["groups"]) - set(GROUPS) or not (0 < s["length_band"][0] <= 1 <= s["length_band"][1]) or not (0 < s["genus_band"][0] <= 1 <= s["genus_band"][1]):
        raise SystemExit("rel_w13_scaffold: groups of %s and bands around 1: %r" % (GROUPS, s))
    return s


def download(url, cas, tries=3):
    """(path in the store, sha256, bytes, server date) of a large file, streamed; complete when it has the announced length."""
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req, timeout=600) as r:
                n, date = int(r.headers.get("Content-Length") or 0), r.headers.get("Last-Modified") or ""
                os.makedirs(cas, exist_ok=True)
                tmp = os.path.join(cas, "w13_scaffold.part%d" % os.getpid())
                h, got = hashlib.sha256(), 0
                with open(tmp, "wb") as f:
                    for block in iter(lambda: r.read(1 << 22), b""):
                        f.write(block)
                        h.update(block)
                        got += len(block)
            if n and got != n:
                raise IOError("%d of %d bytes" % (got, n))
            sha = h.hexdigest()
            path = os.path.join(cas, sha[:2], sha + ".txt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            os.replace(tmp, path)
            return path, sha, got, date
        except Exception as err:  # noqa: BLE001 - a network failure: tried again
            last = err
            time.sleep(10 * (k + 1))
    raise SystemExit("rel_w13_scaffold: %s could not be read whole (%s)" % (url, last))


def rows_of(path):
    """The rows of an assembly summary (the header is its second comment line)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        header = None
        for line in f:
            if line.startswith("#assembly_accession") or (line.startswith("#") and "assembly_accession" in line):
                header = line.lstrip("#").rstrip("\n").split("\t")
                break
        if header is None:
            raise SystemExit("rel_w13_scaffold: %s has no header" % path)
        for line in f:
            v = line.rstrip("\n").split("\t")
            if len(v) == len(header):
                yield dict(zip(header, v))


def reasons(r):
    x = (r.get("excluded_from_refseq") or "").strip()
    return [] if x in ("", "na") else [y.strip() for y in x.split(";") if y.strip()]


def judge(rows, s, genus_of):
    """(scaffold rows with their verdict, counts of the reference rows): rows are the dicts of an assembly summary."""
    bad = set(s["quality_reasons"])
    by_sp, scaff = {}, []
    n_ref = 0
    for r in rows:
        if r.get("version_status") != "latest" or (r.get("genome_rep") or "").lower() != "full":
            continue
        lv = r.get("assembly_level")
        size = int(r["genome_size"]) if (r.get("genome_size") or "").isdigit() else 0
        if lv in ("Complete Genome", "Chromosome"):
            if size and not set(reasons(r)) & bad:
                sp = int(r["species_taxid"]) if r.get("species_taxid", "").isdigit() else 0
                by_sp.setdefault(sp, []).append(size)
                n_ref += 1
        elif lv == "Scaffold":                                  # the fields used below only (the summaries are large)
            scaff.append({k: r.get(k, "") for k in ("assembly_accession", "taxid", "species_taxid", "organism_name", "infraspecific_name", "genome_size",
                                                    "contig_count", "excluded_from_refseq", "seq_rel_date")})
    med_sp = {k: statistics.median(v) for k, v in by_sp.items() if k}
    gsizes = {}
    for sp, v in by_sp.items():
        ge = genus_of(sp) if sp else 0
        if ge:
            gsizes.setdefault(ge, []).extend(v)
    med_ge = {k: statistics.median(v) for k, v in gsizes.items()}
    out = []
    for r in scaff:
        why = None
        ex = reasons(r)
        q = [x for x in ex if x in bad]
        contigs = int(r["contig_count"]) if (r.get("contig_count") or "").isdigit() else 0
        size = int(r["genome_size"]) if (r.get("genome_size") or "").isdigit() else 0
        sp = int(r["species_taxid"]) if r.get("species_taxid", "").isdigit() else 0
        ref, band = (med_sp.get(sp), s["length_band"]) if sp in med_sp else (med_ge.get(genus_of(sp) if sp else 0), s["genus_band"])
        ratio = size / ref if ref and size else 0.0
        if q:
            why = "excluded by RefSeq: %s" % q[0]
        elif not contigs or contigs > s["max_contigs"]:
            why = "more than %d contigs" % s["max_contigs"] if contigs else "no contig count"
        elif not ref:
            why = "no complete genome of its species or genus to compare the length with"
        elif not band[0] <= ratio <= band[1]:
            why = "length %.2f of the %s median" % (ratio, "species" if sp in med_sp else "genus")
        acc, _, ver = r["assembly_accession"].partition(".")
        out.append(dict(accession=acc, tax_id=r.get("taxid", ""), scientific_name=r.get("organism_name", ""), strain=r.get("infraspecific_name", ""),
                        assembly_level="scaffold", genome_representation="full", base_count=str(size), completeness_score="", contamination_score="",
                        host_tax_id="", environmental_sample="true" if set(ex) & set(ENVIRONMENTAL) else "false", version=ver, last_updated=r.get("seq_rel_date", ""),
                        contigs=str(contigs), excluded_from_refseq="; ".join(ex), reference_length=str(int(ref)) if ref else "", length_ratio="%.3f" % ratio,
                        quality=why or "pass"))
    return out, dict(reference_rows=n_ref, species_with_a_reference=len(med_sp), genera_with_a_reference=len(med_ge))


def table(rows):
    """Canonical TSV bytes (FIELDS, sorted by accession, one row per accession: the highest version)."""
    by = {}
    for r in rows:
        k = r["accession"]
        if k not in by or int(r["version"] or 0) > int(by[k]["version"] or 0):
            by[k] = r
    out = io.StringIO()
    w = csv.writer(out, delimiter="\t", lineterminator="\n")
    w.writerow(FIELDS)
    for k in sorted(by):
        w.writerow([str(by[k].get(f, "")).replace("\t", " ").replace("\n", " ") for f in FIELDS])
    return out.getvalue().encode(), len(by)


def build(st, cfg, run_dir):
    from ci.stages import fetch
    from ci.taxonomy import Taxonomy
    s = settings(cfg)
    tax = Taxonomy(st.repo(cfg["taxdump"]))
    cache = {}

    def genus_of(sp):
        if sp not in cache:
            try:
                cache[sp] = tax.at_rank(int(sp), "genus")
            except Exception:  # noqa: BLE001 - a species the taxonomy does not know has no genus here
                cache[sp] = 0
        return cache[sp]
    cas = os.path.join(cfg["data_root"], "cas")
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    lock, files = dict(source="NCBI GenBank assembly summaries", thresholds=dict((k, s[k]) for k in ("max_contigs", "length_band", "genus_band", "quality_reasons")),
                       catalogues={}), []
    for g in s["groups"]:
        url = SUMMARY % g
        raw, raw_sha, nbytes, date = download(url, cas)
        rows, ctx = judge(rows_of(raw), s, genus_of)
        data, n = table(rows)
        path, sha = fetch.store(cas, data, ".tsv")
        p = os.path.join(folder, "%s_scaffold.tsv" % g)
        shutil.copyfile(path, p)
        kept = {}
        for r in rows:                                          # the row the table keeps per accession
            if r["accession"] not in kept or int(r["version"] or 0) > int(kept[r["accession"]]["version"] or 0):
                kept[r["accession"]] = r
        verdicts = {}
        for r in kept.values():
            k = "length outside the band" if r["quality"].startswith("length ") else r["quality"]
            verdicts[k] = verdicts.get(k, 0) + 1
        lock["catalogues"]["%s_scaffold" % g] = dict(query=url, query_sha256=sha256_json(dict(url=url, thresholds=lock["thresholds"])), rows=n, records=n,
                                                     announced=n, older_versions_dropped=0, version_ties=0, sha256=sha, bytes=len(data),
                                                     fetched=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                     summary=dict(url=url, server_date=date, bytes=nbytes, sha256=raw_sha), verdicts=verdicts, context=ctx)
        files.append(p)
        print("rel_w13_scaffold: %s: %d scaffold assemblies, %d pass" % (g, n, verdicts.get("pass", 0)), flush=True)
    lock["context"] = {}
    lp = os.path.join(folder, "lock.yaml")
    with open(lp, "w") as f:
        f.write("# Scaffold-level snapshot of stage rel_w13_scaffold (metadata only; no sequence was downloaded).\n")
        yaml.safe_dump(lock, f, sort_keys=True)
    files.append(lp)
    man = dict(format="rel-w13-scaffold", version=1, note=__doc__.strip().splitlines()[0], lock=lock,
               files=[dict(path=os.path.relpath(p, run_dir), sha256=sha256_file(p, use_sidecar=False), bytes=os.path.getsize(p)) for p in files])
    mp = os.path.join(folder, "manifest.json")
    write_json(mp, man)
    st.output("scaffold_manifest", mp)
    st.extra.update(verdicts={k: v["verdicts"] for k, v in lock["catalogues"].items()})
    return man


def read(run, same_run=False):
    """({catalogue: [rows]}, lock, provenance) of a passed run of this stage (as w13_catalog.read)."""
    return w13_catalog.read(run, same_run=same_run, stage="rel_w13_scaffold", folder=OUT, output="scaffold_manifest")


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "manifest.json")
    with open(p) as f:
        man = json.load(f)
    problems = []
    for e in man.get("files") or []:
        full = os.path.join(run_dir, e["path"])
        if not os.path.exists(full) or sha256_file(full, use_sidecar=False) != e["sha256"]:
            problems.append(dict(file=e["path"], error="missing or not the file the manifest names"))
    for cid, c in (man.get("lock") or {}).get("catalogues", {}).items():
        with open(os.path.join(run_dir, OUT, cid + ".tsv")) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if len(rows) != c["records"] or len({r["accession"] for r in rows}) != len(rows) or sum(c["verdicts"].values()) != len(rows):
            problems.append(dict(catalogue=cid, error="another count, an accession twice, or verdicts that do not add up"))
        if any(not (r.get("tax_id") or "").isdigit() or not r.get("quality") for r in rows):
            problems.append(dict(catalogue=cid, error="rows without a taxid or a verdict"))
    require(GateResult("G4_rel_w13_scaffold", "fail" if problems else "pass", dict(problems=problems[:50])), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "manifest.json")) as f:
        man = json.load(f)
    cats = man["lock"]["catalogues"]
    return dict(json={k: v["verdicts"] for k, v in cats.items()},
                md=["Scaffold-level snapshot (NCBI GenBank assembly summaries, metadata only): " + "; ".join(
                    "%s %s rows, %s pass" % (k, "{:,}".format(v["records"]), "{:,}".format(v["verdicts"].get("pass", 0))) for k, v in cats.items())])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w13_scaffold") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
