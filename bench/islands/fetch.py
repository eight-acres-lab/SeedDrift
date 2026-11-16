#!/usr/bin/env python
"""Fetch the genomic-island reference set and the genome records of the test chromosomes (protocol: README.md).

  tables    Sup_tables.tar.gz from the IslandViewer download page: SupTable1 = islands, SupTable2 = backbone
  selection one chromosome per genus outside the order of the development genomes, by label counts only
  genomes   GenBank records with sequence, by accession.version, from NCBI E-utilities

Everything is downloaded to --dir (on the machine this runs on). The selection and the checksums are written
to bench/islands/results/ so that they can be committed before any genome is run.

usage:  python bench/islands/fetch.py [--dir $SEEDDRIFT_DATA/evalsets/islandpick]
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import os
import sys
import tarfile
import time
import urllib.request

TABLES = "https://www.pathogenomics.sfu.ca/islandviewer/download/datasets/Sup_tables.tar.gz"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=%s&rettype=gbwithparts&retmode=text"
DEV_GENERA = {"Escherichia", "Salmonella", "Shigella", "Yersinia", "Legionella"}   # same order as a development genome
HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def get(url, dest, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "python-urllib"}), timeout=600) as r, open(dest + ".part", "wb") as fo:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fo.write(b)
            os.replace(dest + ".part", dest)
            return True
        except Exception as e:               # noqa: BLE001
            print("  retry %d for %s: %s" % (k + 1, os.path.basename(dest), e), file=sys.stderr)
            time.sleep(5 * (k + 1))
    return False


def read_table(path):
    txt = open(path, newline="").read().replace("\r\n", "\n").replace("\r", "\n")
    return list(csv.DictReader(io.StringIO(txt), delimiter="\t"))


def ranked(pos, neg):
    """Per genus outside the development order: chromosomes in the order of the selection rule
    (largest min(islands, backbone regions), then most regions, then smallest accession)."""
    g = {}
    for rows, k in ((pos, "islands"), (neg, "backbone")):
        for r in rows:
            x = g.setdefault(r["Accession"], dict(accession=r["Accession"], organism=r["Organism"].strip(), islands=0, backbone=0))
            x[k] += 1
    by = collections.defaultdict(list)
    for x in g.values():
        by[x["organism"].split()[0]].append(x)
    return {genus: sorted(v, key=lambda x: (-min(x["islands"], x["backbone"]), -(x["islands"] + x["backbone"]), x["accession"]))
            for genus, v in sorted(by.items()) if genus not in DEV_GENERA}


def good_record(path, acc):
    """The file is a complete GenBank record of exactly this accession.version with annotated coding sequences."""
    with open(path, errors="replace") as fh:
        head = fh.read(20000)
        fh.seek(max(0, os.path.getsize(path) - 200))
        tail = fh.read()
    n_cds = sum(1 for line in open(path, errors="replace") if line.startswith("     CDS  "))
    return head.startswith("LOCUS") and ("VERSION     %s" % acc) in head and tail.rstrip().endswith("//") and n_cds >= 100, n_cds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(os.environ.get("SEEDDRIFT_DATA", "."), "evalsets", "islandpick"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--dev", action="store_true", help="instead of the test selection: every chromosome of the development genera, into "
                    "<dir>/dev_genomes and results/dev_genomes.tsv (for developing methods; never part of the test)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.dir, "genomes"), exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    tgz = os.path.join(a.dir, "Sup_tables.tar.gz")
    if not os.path.exists(tgz) and not get(TABLES, tgz):
        sys.exit("could not download " + TABLES)
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            if os.path.basename(m.name) in ("SupTable1.txt", "SupTable2.txt", "SupTable3.txt"):
                m.name = os.path.basename(m.name)
                tf.extract(m, a.dir)
    pos, neg = read_table(os.path.join(a.dir, "SupTable1.txt")), read_table(os.path.join(a.dir, "SupTable2.txt"))
    if a.dev:
        os.makedirs(os.path.join(a.dir, "dev_genomes"), exist_ok=True)
        g = {}
        for rows, k in ((pos, "islands"), (neg, "backbone")):
            for r in rows:
                x = g.setdefault(r["Accession"], dict(accession=r["Accession"], organism=r["Organism"].strip(), islands=0, backbone=0))
                x[k] += 1
        with open(os.path.join(a.out, "dev_genomes.tsv"), "w") as fo:
            fo.write("genus\taccession\torganism\tislands\tbackbone\tcds\tsha256\n")
            for x in sorted(g.values(), key=lambda x: (x["organism"], x["accession"])):
                genus = x["organism"].split()[0]
                if genus not in DEV_GENERA:
                    continue
                dest = os.path.join(a.dir, "dev_genomes", x["accession"] + ".gb")
                ok = os.path.exists(dest) and good_record(dest, x["accession"])[0]
                if not ok and get(EFETCH % x["accession"], dest):
                    ok = good_record(dest, x["accession"])[0]
                    time.sleep(1)
                if not ok:
                    print("%-14s %s  not usable" % (genus, x["accession"]), flush=True)
                    continue
                n_cds = good_record(dest, x["accession"])[1]
                fo.write("%s\t%s\t%s\t%d\t%d\t%d\t%s\n" % (genus, x["accession"], x["organism"], x["islands"], x["backbone"], n_cds, sha256(dest)))
                print("%-14s %s  %d CDS" % (genus, x["accession"], n_cds), flush=True)
        return
    prov = dict(date=datetime.date.today().isoformat(), tables_url=TABLES, tables_sha256=sha256(tgz),
                n_island_regions=len(pos), n_backbone_regions=len(neg), genome_source=EFETCH % "<accession.version>", genomes={}, replaced=[])
    chosen = []
    for genus, cands in ranked(pos, neg).items():
        for rank, x in enumerate(cands):
            dest = os.path.join(a.dir, "genomes", x["accession"] + ".gb")
            ok = os.path.exists(dest) and good_record(dest, x["accession"])[0]
            if not ok and get(EFETCH % x["accession"], dest):
                ok = good_record(dest, x["accession"])[0]
                time.sleep(1)
            if ok:
                n_cds = good_record(dest, x["accession"])[1]
                chosen.append(dict(genus=genus, **x, cds=n_cds, rank_in_genus=rank + 1))
                prov["genomes"][x["accession"]] = dict(sha256=sha256(dest), bytes=os.path.getsize(dest), cds=n_cds)
                print("%-18s %s  %d CDS%s" % (genus, x["accession"], n_cds, "" if rank == 0 else "  (choice %d of the genus)" % (rank + 1)), flush=True)
                break
            prov["replaced"].append(dict(genus=genus, accession=x["accession"], reason="record not available as a complete annotated GenBank file of this version"))
            print("%-18s %s  not usable, trying the next of the genus" % (genus, x["accession"]), flush=True)
        else:
            print("%-18s no usable record" % genus, flush=True)
    with open(os.path.join(a.out, "genomes.tsv"), "w") as fo:
        fo.write("genus\taccession\torganism\tislands\tbackbone\tcds\trank_in_genus\n")
        for x in chosen:
            fo.write("%(genus)s\t%(accession)s\t%(organism)s\t%(islands)d\t%(backbone)d\t%(cds)d\t%(rank_in_genus)d\n" % x)
    json.dump(prov, open(os.path.join(a.out, "provenance.json"), "w"), indent=1)
    print("%d genomes; selection and checksums in %s" % (len(chosen), a.out))


if __name__ == "__main__":
    main()
