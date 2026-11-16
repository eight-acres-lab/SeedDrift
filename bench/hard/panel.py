#!/usr/bin/env python
"""Genus panels: for every benchmark chromosome, annotated complete genomes of its own genus and of sibling genera.

The panel is the label source of the hard benchmark (labels.py in this folder reads it). No method under test
ever sees a panel genome: the genome's own genus is hidden from every method.

  candidates   INSDC assemblies (ENA) of level "complete genome" or "chromosome", full representation, in the
               family of the test chromosome
  left out     the assemblies behind the reference proteomes of the search library; the strains of the published
               island set (their taxa); assemblies without annotated coding sequences; assemblies of which the
               mirror holds no record. A download that merely fails stops the build of that panel: it is rerun,
               so that the day's network does not decide which genomes certify the labels
  chosen       in ascending order of SHA-256("<assembly accession>|panel-v2"):
                 same     up to 5 assemblies of the species of the test chromosome
                 other    up to 5 assemblies of other species of the genus, one per species
                 sibling  4 assemblies of other genera of the family, one per genus (from the order if the
                          family has fewer than 4 other genera)
  files        <dir>/assemblies/<assembly>.embl.gz  (all replicons of the assembly, EMBL flat file format)
               <dir>/<accession>/panel.tsv          (one row per chosen assembly: role, species, replicons, CDS)
               <dir>/panel_log.json                 (candidate counts and what was skipped, per chromosome)

Records come from ENA (the mirror that is fast from the server); --source ncbi takes the same INSDC records from the
NCBI mirror when the ENA record service does not answer. The annotation is the submitters'.

usage (repo root, on the machine with the data store; needs the seeddrift package):
  python bench/hard/panel.py --genomes bench/islands/results/genomes.tsv --genbank DIR/genomes --dir DIR/panel
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

PORTAL = "https://www.ebi.ac.uk/ena/portal/api/search"
BROWSER = "https://www.ebi.ac.uk/ena/browser/api"
UNIPROT = "https://rest.uniprot.org/proteomes/stream?query=%s&fields=upid,organism_id,genome_assembly&format=tsv"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=nuccore&retmode=json&id=%s"
NCBI_REPORT = "https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/%s/sequence_reports"
NCBI_RECORD = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=%s&rettype=gbwithparts&retmode=text"
QUOTA = {"same": 5, "other": 5, "sibling": 4}
SALT = "panel-v2"
MIN_CDS = 300


class NoRecord(Exception):
    """The mirror answers that it holds no such record (HTTP 400, 404 or 410): a property of the assembly, not of the day."""


def fetch(url, tries=4, pause=3.0):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib", "Accept-Encoding": "gzip"})
            with urllib.request.urlopen(req, timeout=300) as r:
                b = r.read()
            while b[:2] == b"\x1f\x8b":             # some servers compress an already compressed stream
                b = gzip.decompress(b)
            return b
        except urllib.error.HTTPError as e:
            if e.code in (400, 404, 410):
                raise NoRecord(url[:120])
            err = e
            time.sleep(pause * (k + 1))
        except Exception as e:                       # noqa: BLE001
            err = e
            time.sleep(pause * (k + 1))
    raise RuntimeError("%s: %s" % (url[:120], err))


class NotAnnotated(Exception):
    pass


def fetch_record(acc, tries=3, pause=5.0, probe=400000, source="ena"):
    """Flat file of one sequence (EMBL format from ENA, GenBank format from NCBI: the same INSDC record), read as a
    stream. The feature table precedes the sequence, so a record whose first `probe` bytes hold no coding sequence is
    given up at once instead of being downloaded whole."""
    for k in range(tries):
        try:
            url = "%s/embl/%s" % (BROWSER, acc) if source == "ena" else NCBI_RECORD % acc
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib", "Accept-Encoding": "gzip"})
            buf, z, checked = bytearray(), None, False
            with urllib.request.urlopen(req, timeout=300) as r:
                first = True
                while True:
                    b = r.read(1 << 17)
                    if not b:
                        break
                    if first:
                        z = zlib.decompressobj(16 + zlib.MAX_WBITS) if b[:2] == b"\x1f\x8b" else None
                        first = False
                    buf += z.decompress(b) if z else b
                    if probe and not checked and len(buf) >= probe:
                        checked = True
                        if b"FT   CDS  " not in buf and b"     CDS  " not in buf:
                            raise NotAnnotated(acc)
            return bytes(buf).decode(errors="replace")
        except NotAnnotated:
            raise
        except urllib.error.HTTPError as e:
            if e.code in (400, 404, 410):
                raise NoRecord(acc)
            err = e
            time.sleep(pause * (k + 1))
        except Exception as e:                       # noqa: BLE001
            err = e
            time.sleep(pause * (k + 1))
    raise RuntimeError("%s: %s" % (acc, err))


def order_key(acc):
    return hashlib.sha256(("%s|%s" % (acc, SALT)).encode()).hexdigest()


def candidates(taxid):
    """Complete assemblies under a taxon: rows with accession.version, tax_id, name, strain, size."""
    q = 'tax_tree(%d) AND (assembly_level="complete genome" OR assembly_level="chromosome") AND genome_representation="full"' % taxid
    url = "%s?result=assembly&query=%s&fields=accession,version,scientific_name,tax_id,strain,base_count,assembly_level&limit=0&format=tsv" % (
        PORTAL, urllib.parse.quote(q))
    rows = list(csv.DictReader(io.StringIO(fetch(url).decode()), delimiter="\t"))
    for r in rows:
        r["assembly"] = "%s.%s" % (r["accession"], r["version"])
    return rows


def replicons(assembly):
    """[(sequence accession, type)] of an assembly, from its ENA record."""
    xml = fetch("%s/xml/%s" % (BROWSER, assembly)).decode(errors="replace")
    return re.findall(r'<CHROMOSOME accession="([^"]+)">\s*(?:<NAME>[^<]*</NAME>\s*)?<TYPE>([^<]*)</TYPE>', xml)


def replicons_ncbi(assembly):
    """The same list from the NCBI mirror of the INSDC assembly (used when the ENA record service does not answer)."""
    j = json.loads(fetch(NCBI_REPORT % assembly))
    return [(r["genbank_accession"], "Chromosome" if r.get("assigned_molecule_location_type") == "Chromosome" else "Plasmid")
            for r in j.get("reports", []) if r.get("role") == "assembled-molecule" and r.get("genbank_accession")]


def assembly_text(assembly, source):
    """(replicons, flat-file text of all replicons, source) of one assembly from one mirror. Raises NoRecord when the
    mirror does not hold it, NotAnnotated when its chromosome record carries no coding sequence."""
    reps = replicons(assembly) if source == "ena" else replicons_ncbi(assembly)
    if not reps:
        raise NoRecord(assembly)
    reps = sorted(reps, key=lambda x: x[1] != "Chromosome")              # a chromosome first: it shows at once whether the assembly is annotated
    text = ""
    for i, (r, _) in enumerate(reps):
        text += fetch_record(r, probe=400000 if i == 0 else 0, source=source)
        if source == "ncbi":
            time.sleep(0.4)
    return reps, text, source


def n_cds(text):
    """(coding sequences, translated coding sequences) of EMBL or GenBank flat file text."""
    return sum(1 for line in text.split("\n") if line.startswith("FT   CDS  ") or line.startswith("     CDS  ")), text.count('/translation="')


def library_assemblies(cache):
    """Assembly accessions (without version) behind the reference proteomes of the library."""
    if not os.path.exists(cache):
        from seeddrift import config
        ups = [r["proteome"] for r in csv.DictReader(open(os.path.join(config.REFERENCE, "proteomes.tsv")), delimiter="\t")]
        out = {}
        for i in range(0, len(ups), 100):
            q = urllib.parse.quote(" OR ".join("upid:" + u for u in ups[i:i + 100]))
            for r in csv.DictReader(io.StringIO(fetch(UNIPROT % q).decode()), delimiter="\t"):
                keys = list(r)
                out[r[keys[0]]] = (r[next(k for k in keys if "assembly" in k.lower())] or "").strip()
            time.sleep(0.3)
        json.dump(out, open(cache, "w"))
    return {v.split(".")[0].replace("GCF_", "GCA_") for v in json.load(open(cache)).values() if v}


def island_set_taxa(tables, cache):
    """Taxa of the chromosomes of the published island set (by their nucleotide accessions)."""
    if not os.path.exists(cache):
        accs = sorted({r["Accession"].strip() for t in tables for r in csv.DictReader(
            io.StringIO(open(t, newline="").read().replace("\r\n", "\n").replace("\r", "\n")), delimiter="\t")})
        out = {}
        for i in range(0, len(accs), 40):
            res = json.loads(fetch(ESUMMARY % ",".join(accs[i:i + 40])))["result"]
            for uid in res.get("uids", []):
                out[res[uid]["accessionversion"]] = int(res[uid]["taxid"])
            time.sleep(1.0)
        json.dump(out, open(cache, "w"), indent=1)
    return json.load(open(cache))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genomes", nargs="+", required=True, help="tables with the columns genus and accession (test and development chromosomes)")
    ap.add_argument("--genbank", nargs="+", required=True, help="folders with <accession>.gb (searched in order)")
    ap.add_argument("--tables", nargs="+", required=True, help="SupTable1.txt SupTable2.txt of the island set (strains to leave out)")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--only", nargs="*", default=[], help="only these accessions")
    ap.add_argument("--source", default="ena", choices=["ena", "ncbi"], help="mirror for the records of the chosen assemblies (the candidate list always comes from ENA)")
    a = ap.parse_args()
    from seeddrift.pipeline import get_taxonomy
    tax = get_taxonomy()
    store = os.path.join(a.dir, "assemblies")
    os.makedirs(store, exist_ok=True)
    bad_path = os.path.join(a.dir, "not_usable.json")          # assemblies found without annotation or without a replicon list
    bad = json.load(open(bad_path)) if os.path.exists(bad_path) else {}
    lib = library_assemblies(os.path.join(a.dir, "library_assemblies.json"))
    isl = island_set_taxa(a.tables, os.path.join(a.dir, "island_set_taxa.json")) if a.tables else {}
    isl_taxa = set(isl.values())
    log_path = os.path.join(a.dir, "panel_log.json")
    log = json.load(open(log_path)) if os.path.exists(log_path) else {}
    rows = [r for g in a.genomes for r in csv.DictReader(open(g), delimiter="\t") if not a.only or r["accession"] in a.only]
    fam_cache = {}
    for g in rows:
        acc = g["accession"]
        gb = next(os.path.join(d, acc + ".gb") for d in a.genbank if os.path.exists(os.path.join(d, acc + ".gb")))
        t = int(re.search(r'/db_xref="taxon:(\d+)"', open(gb, errors="replace").read(300000)).group(1))
        species, genus = tax.at_rank(t, "species"), tax.at_rank(t, "genus")
        family, order = tax.at_rank(t, "family"), tax.at_rank(t, "order")
        out = os.path.join(a.dir, acc)
        os.makedirs(out, exist_ok=True)
        if os.path.exists(os.path.join(out, "panel.tsv")):
            print("%s: panel exists" % acc, flush=True)
            continue
        top = family or order or genus
        if top not in fam_cache:
            fam_cache[top] = candidates(top)
        cand = fam_cache[top]
        info = dict(taxid=t, species=species, genus=genus, family=family, order=order, candidates=len(cand), skipped={})
        pool = {"same": [], "other": [], "sibling": []}
        for c in cand:
            ct = int(c["tax_id"])
            if c["accession"] in lib:
                info["skipped"]["library proteome"] = info["skipped"].get("library proteome", 0) + 1
                continue
            if ct in isl_taxa or ct == t:
                info["skipped"]["island-set strain"] = info["skipped"].get("island-set strain", 0) + 1
                continue
            cs, cg = tax.at_rank(ct, "species"), tax.at_rank(ct, "genus")
            if not cs or not cg:
                info["skipped"]["no species or genus"] = info["skipped"].get("no species or genus", 0) + 1
                continue
            role = "same" if cs == species else "other" if cg == genus else "sibling"
            pool[role].append(dict(c, species=cs, genus=cg, role=role))
        if len({c["genus"] for c in pool["sibling"]}) < QUOTA["sibling"] and order and order != top:      # fill sibling genera from the order
            if order not in fam_cache:
                fam_cache[order] = candidates(order)
            have = {c["assembly"] for c in pool["sibling"]}
            for c in fam_cache[order]:
                ct = int(c["tax_id"])
                cs, cg = tax.at_rank(ct, "species"), tax.at_rank(ct, "genus")
                if c["assembly"] in have or c["accession"] in lib or ct in isl_taxa or not cs or not cg or cg == genus:
                    continue
                pool["sibling"].append(dict(c, species=cs, genus=cg, role="sibling", from_order=1))
        info["pool"] = {k: len(v) for k, v in pool.items()}
        info["pool_species_other"] = len({c["species"] for c in pool["other"]})
        chosen = []
        for role in ("same", "other", "sibling"):
            taken, used = 0, set()
            ranked = sorted(pool[role], key=lambda c: (c.get("from_order", 0), order_key(c["assembly"])))    # family genera before order genera
            for c in ranked:
                if taken >= QUOTA[role]:
                    break
                unit = c["species"] if role == "other" else c["genus"] if role == "sibling" else c["assembly"]
                if unit in used:
                    continue
                if c["assembly"] in bad:
                    info["skipped"][bad[c["assembly"]]] = info["skipped"].get(bad[c["assembly"]], 0) + 1
                    continue
                meta = os.path.join(store, c["assembly"] + ".json")
                dest = os.path.join(store, c["assembly"] + (".embl.gz" if a.source == "ena" else ".gb.gz"))
                have = [x for x in (os.path.join(store, c["assembly"] + e) for e in (".embl.gz", ".gb.gz")) if os.path.exists(x)]
                if have and os.path.exists(meta):
                    m = json.load(open(meta))
                    reps, tr = [tuple(x) for x in m["replicons"]], m["cds"]
                else:
                    got = None
                    for src in (a.source, "ncbi" if a.source == "ena" else "ena"):     # an assembly is left out only if neither mirror holds it
                        try:
                            got = assembly_text(c["assembly"], src)
                            break
                        except NoRecord:
                            continue
                        except NotAnnotated:
                            got = "not annotated"
                            break
                        except Exception as e:       # noqa: BLE001 - a failed download must not change which assemblies the rule picks
                            sys.exit("%s: download of %s from %s failed (%s); no panel written for this chromosome, rerun" % (acc, c["assembly"], src, str(e)[:80]))
                    if got is None or got == "not annotated":
                        why = "no record" if got is None else "not annotated"
                        info["skipped"][why] = info["skipped"].get(why, 0) + 1
                        bad[c["assembly"]] = why
                        json.dump(bad, open(bad_path, "w"), indent=1)
                        continue
                    reps, text, src = got
                    dest = os.path.join(store, c["assembly"] + (".embl.gz" if src == "ena" else ".gb.gz"))
                    cds, tr = n_cds(text)
                    if tr < MIN_CDS or tr < 0.8 * cds:
                        info["skipped"]["not annotated"] = info["skipped"].get("not annotated", 0) + 1
                        bad[c["assembly"]] = "not annotated"
                        json.dump(bad, open(bad_path, "w"), indent=1)
                        continue
                    with gzip.open(dest + ".part", "wt") as fo:
                        fo.write(text)
                    os.replace(dest + ".part", dest)
                    json.dump(dict(replicons=reps, cds=tr), open(meta, "w"))
                    time.sleep(0.2)
                chosen.append(dict(assembly=c["assembly"], role=role, tax_id=c["tax_id"], species=c["species"], genus=c["genus"],
                                   name=c["scientific_name"], strain=c["strain"], bases=c["base_count"],
                                   replicons=";".join("%s:%s" % (r, k) for r, k in reps), cds=tr, from_order=c.get("from_order", 0)))
                used.add(unit); taken += 1
        with open(os.path.join(out, "panel.tsv"), "w") as fo:
            cols = ["assembly", "role", "tax_id", "species", "genus", "name", "strain", "bases", "replicons", "cds", "from_order"]
            fo.write("\t".join(cols) + "\n")
            for c in chosen:
                fo.write("\t".join(str(c[k]) for k in cols) + "\n")
        info["chosen"] = {k: sum(1 for c in chosen if c["role"] == k) for k in QUOTA}
        log[acc] = info
        json.dump(log, open(log_path, "w"), indent=1)
        print("%s %s: candidates %d, pool %s (other species %d), chosen %s, skipped %s" % (
            g["genus"], acc, len(cand), info["pool"], info["pool_species_other"], info["chosen"], info["skipped"]), flush=True)


if __name__ == "__main__":
    main()
