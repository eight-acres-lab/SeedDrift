"""Spike-in benchmark data: a real recipient proteome plus foreign proteins of known origin.

Donors are UniProt reference proteomes that are NOT in the SeedDrift reference library (held-out
species: their genus or family is usually represented, as for real donors). Spiked proteins are
mutated to several divergence levels (fraction of sites replaced by residues drawn from the
background composition), which makes them harder to place, though random substitutions are
harsher than real evolution (no conserved sites).

Truth labels:
  spike    foreign, with its donor class (mammal / eukaryote / prokaryote / virus) and group
  core     recipient ribosomal proteins and RNA polymerase subunits: expected native
  other    the rest of the recipient: unlabeled (real genomes carry real transfers)

usage (on the machine with the data):
  python bench/spikein.py --recipient Bacteria/UP000000625_83333 --out bench/work/spike_ecoli
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import random
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seeddrift import config  # noqa: E402
from seeddrift.donors import _readme_rows, select_proteomes, taxonomy  # noqa: E402
from seeddrift.single import ARCHAEA, BACTERIA, EUKARYOTA, MAMMALIA, VIRUSES, home_group  # noqa: E402

RP = "https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes"
BG = dict(zip("ARNDCQEGHILKMFPSTWYV", [8.25, 5.53, 4.06, 5.45, 1.37, 3.93, 6.75, 7.07, 2.27, 5.96, 9.66, 5.84,
                                        2.42, 3.86, 4.70, 6.56, 5.34, 1.08, 2.92, 6.87]))   # UniProtKB composition
CORE = re.compile(r"\b(30S|50S|40S|60S) ribosomal protein\b|\bribosomal protein [LS]\d|DNA-directed RNA polymerase subunit",
                  re.I)
CORE_EXCLUDE = re.compile(r"methyltransferase|acetyltransferase|modification|kinase|maturation|assembly|"
                          r"biogenesis|accessory|binding protein|like|factor", re.I)


def fetch_fasta(url, cache):
    if not os.path.exists(cache):
        with urllib.request.urlopen(url, timeout=600) as r, open(cache + ".part", "wb") as fh:
            fh.write(r.read())
        os.replace(cache + ".part", cache)
    seqs, name, buf = [], None, []
    for line in io.TextIOWrapper(gzip.open(cache), errors="replace"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name:
                seqs.append((name, "".join(buf)))
            name, buf = line[1:], []
        else:
            buf.append(line.strip())
    if name:
        seqs.append((name, "".join(buf)))
    return seqs


def mutate(seq, d, rnd):
    if d <= 0:
        return seq
    aa, w = zip(*BG.items())
    s = list(seq)
    for i in rnd.sample(range(1, len(s)), int(round(d * (len(s) - 1)))):    # keep the start residue
        s[i] = rnd.choices(aa, w)[0]
    return "".join(s)


def donor_class(tax, t, home_t):
    lin = set(tax.lineage(t))
    if home_t in lin:
        return None                                    # inside the recipient's home group: not foreign
    if MAMMALIA in lin:
        return "mammal"
    if EUKARYOTA in lin:
        return "eukaryote"
    if BACTERIA in lin or ARCHAEA in lin:
        return "prokaryote"
    if VIRUSES in lin:
        return "virus"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipient", required=True, help="Regnum/UPxxxx_taxid, e.g. Bacteria/UP000000625_83333")
    ap.add_argument("--out", required=True)
    ap.add_argument("--library", default=os.path.join(config.REFERENCE, "proteomes.tsv"))
    ap.add_argument("--readme", default=os.path.join(config.REFERENCE, "README"))
    ap.add_argument("--donors-per-class", type=int, default=4)
    ap.add_argument("--per-donor-level", type=int, default=10, help="proteins per donor and divergence level")
    ap.add_argument("--divergence", default="0,0.3,0.5,0.7")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, "cache"), exist_ok=True)
    tax = taxonomy(config.REFERENCE)
    regnum, rid = a.recipient.split("/")
    rtax = int(rid.split("_")[1])
    home_t, home_name = home_group(tax, tax.at_rank(rtax, "genus") or rtax)
    rows = _readme_rows(a.readme)
    in_lib = {r["proteome"] for r in csv.DictReader(open(a.library), delimiter="\t")} if os.path.exists(a.library) else set()
    for pol in ("enterobacterales", "broad"):          # held out from both library versions (ablation)
        in_lib |= {r["id"] for r in select_proteomes(rows, tax, pol)}
    rself = set(tax.lineage(tax.at_rank(rtax, "genus") or rtax))
    pools = {}
    for r in rows:
        if r["id"] in in_lib or r["n"] < 50 or r["taxid"] in rself:
            continue
        c = donor_class(tax, r["taxid"], home_t)
        if c == "virus" and r["n"] < 40:
            continue
        if c:
            pools.setdefault(c, []).append(r)
    levels = [float(x) for x in a.divergence.split(",")]
    truth, prots = [], []
    for name, seq in fetch_fasta("%s/%s/%s.fasta.gz" % (RP, regnum, rid), os.path.join(a.out, "cache", rid + ".fasta.gz")):
        acc = name.split()[0].split("|")[1] if "|" in name.split()[0] else name.split()[0]
        desc = name.split(" ", 1)[1] if " " in name else ""
        label = "core" if CORE.search(desc) and not CORE_EXCLUDE.search(desc.split(" OS=")[0]) else "other"
        prots.append((acc, seq))
        truth.append(dict(gene_id=acc, label=label, true_class="native" if label == "core" else "", divergence="",
                          donor_taxid="", donor_group="", donor_name=desc.split(" OS=")[0][:60]))
    print("recipient %s: %d proteins, %d core; home group %s" % (rid, len(prots), sum(t["label"] == "core" for t in truth),
                                                                  home_name), file=sys.stderr)
    for c in ("mammal", "eukaryote", "prokaryote", "virus"):
        pool = pools.get(c, [])
        if not pool:
            continue
        # spread donors over groups: one per distinct donor group first
        rnd.shuffle(pool)
        chosen, seen = [], set()
        for r in pool:
            g = tax.group(r["taxid"])
            if g not in seen or len(seen) >= a.donors_per_class:
                chosen.append(r)
                seen.add(g)
            if len(chosen) >= a.donors_per_class:
                break
        for r in chosen:
            fn = "%s_%d" % (r["id"], r["taxid"])
            try:
                seqs = fetch_fasta("%s/%s/%s/%s.fasta.gz" % (RP, r["regnum"].capitalize(), r["id"], fn),
                                   os.path.join(a.out, "cache", fn + ".fasta.gz"))
            except Exception as e:                    # noqa: BLE001
                print("skip %s: %s" % (fn, e), file=sys.stderr)
                continue
            seqs = [(n, s) for n, s in seqs if 80 <= len(s) <= 1500]
            pick = rnd.sample(seqs, min(len(seqs), a.per_donor_level * len(levels)))
            for k, (n, s) in enumerate(pick):
                d = levels[k % len(levels)]
                gid = "SPIKE_%s_%d_%s_%d" % (c, int(d * 100), r["id"], k)
                prots.append((gid, mutate(s, d, rnd)))
                truth.append(dict(gene_id=gid, label="spike", true_class=c, divergence=d, donor_taxid=r["taxid"],
                                  donor_group=tax.group(r["taxid"]), donor_name=r["species"][:60]))
            print("  %-10s %-34s %s (%d proteins spiked)" % (c, tax.group(r["taxid"]), r["species"][:50], len(pick)),
                  file=sys.stderr)
    with open(os.path.join(a.out, "proteins.faa"), "w") as fh:
        for gid, s in prots:
            fh.write(">%s\n%s\n" % (gid, s))
    with open(os.path.join(a.out, "truth.tsv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(truth[0]), delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(truth)
    print("%d proteins (%d spiked) -> %s" % (len(prots), sum(t["label"] == "spike" for t in truth), a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
