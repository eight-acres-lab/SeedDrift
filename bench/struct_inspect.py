"""Top Foldseek hits (with Swiss-Prot protein names and taxon groups) for selected query genes."""
import csv, gzip, re, sys
sys.path.insert(0, __import__("os").environ.get("SEEDDRIFT_HOME", "."))
from seeddrift.donors import taxonomy
from seeddrift.structure import _acc
B, ref, sprot, genes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
tax = taxonomy(ref)
info = {}
for line in gzip.open(sprot, "rt"):
    if line[0] == ">":
        acc = line.split("|")[1]; m = re.search(r" OX=(\d+)", line)
        name = re.sub(r" OS=.*", "", line.split(" ", 1)[1]).strip()
        info[acc] = (name, int(m.group(1)) if m else 0)
selfst = {r["gene_id"]: int(r["self_taxid"]) for r in csv.DictReader(open(B + "queries.tsv"), delimiter="\t")}
hits = {}
for line in open(B + "hits.m8"):
    f = line.rstrip("\n").split("\t")
    if f[0] in genes: hits.setdefault(f[0], []).append(f)
for g in genes:
    print("==", g)
    k = 0
    for f in sorted(hits.get(g, []), key=lambda f: -float(f[3])):
        a = _acc(f[1]); nm, t = info.get(a, ("?", 0))
        if t and selfst[g] in tax.lineage(t): continue
        if float(f[4]) < 0.5 or float(f[2]) > 1e-3 or float(f[8]) < 0.4: continue
        print("  %-10s %-34s %-55s TM=%s prob=%s fid=%s qcov=%s" % (a, tax.group(t)[:34], (nm[:40] + " | " + tax.name.get(t, "")[:12]), f[5], f[4], f[7], f[8]))
        k += 1
        if k == 6: break
