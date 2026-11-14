"""Summary of the Foldseek step (run on the GPU box): structural donor groups by pLDDT, overlap with the
sequence-profile search, and every mammal structural match with its rank."""
import csv, sys, collections
B = sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/hgtscan/struct/run/"
sd = list(csv.DictReader(open(B + "structure_donors.tsv"), delimiter="\t"))
prof = {r["gene_id"]: r for r in csv.DictReader(open(B + "profile/donors_by_gene.tsv"), delimiter="\t")}
def pb(v):
    if v in ("", "NA"): return "NA"
    v = float(v); return ">=70" if v >= 70 else ("50-70" if v >= 50 else "<50")
print("events", len(sd))
for p in (">=70", "50-70", "<50", "NA"):
    rs = [r for r in sd if pb(r["mean_plddt"]) == p]
    m = [r for r in rs if r["top_group"] != "no structural match"]
    print(" pLDDT %-5s n=%4d  structural match %4d (%.0f%%)" % (p, len(rs), len(m), 100.0 * len(m) / max(1, len(rs))))
conf = [r for r in sd if pb(r["mean_plddt"]) == ">=70" and r["top_group"] != "no structural match"]
print("\nconfident (pLDDT>=70) matches by group:", collections.Counter(r["top_group"] for r in conf).most_common())
allm = [r for r in sd if r["top_group"] != "no structural match"]
print("all matches by group:", collections.Counter(r["top_group"] for r in allm).most_common())
seq_hit = lambda g: prof.get(g, {}).get("top_group", "no outside homolog") != "no outside homolog"
both = sum(1 for r in allm if seq_hit(r["gene_id"]))
print("\nstructure match %d: also profile hit %d, structure-only %d; profile-only %d"
      % (len(allm), both, len(allm) - both, sum(1 for r in sd if seq_hit(r["gene_id"]) and r["top_group"] == "no structural match")))
agree = [r for r in allm if seq_hit(r["gene_id"])]
print("group agreement (structure vs profile) where both: %d/%d"
      % (sum(1 for r in agree if r["top_group"] == prof[r["gene_id"]]["top_group"]), len(agree)))
tm = sorted(float(r["best_tm"]) for r in allm if r["best_tm"])
if tm: print("best TM quantiles 25/50/75: %.2f %.2f %.2f" % (tm[len(tm)//4], tm[len(tm)//2], tm[3*len(tm)//4]))
print("\n-- top structure-only matches (pLDDT>=70, by TM) --")
so = sorted((r for r in conf if not seq_hit(r["gene_id"])), key=lambda r: -float(r["best_tm"] or 0))
for r in so[:20]:
    print("%-40s len=%-5s pLDDT=%-5s TM=%s prob=%s fid=%s %s / %s (%s %s)" % (r["gene_id"], r["length"], r["mean_plddt"], r["best_tm"],
          r["best_prob"], r["best_fident"], r["best_hit"], r["best_hit_taxon"], r["top_group"], r["top_group_share"]))
print("\n-- mammal structural matches --")
for r in sd:
    if r["mammal_hit"]:
        print("%-40s len=%-5s pLDDT=%-5s top=%s(%s) best=%s %s TM=%s | mammal: %s" % (r["gene_id"], r["length"], r["mean_plddt"],
              r["top_group"], r["top_group_share"], r["best_hit_taxon"], r["best_hit_group"], r["best_tm"], r["mammal_hit"]))

# baseline: taxonomic make-up of the target library (Swiss-Prot is heavy in model organisms)
if len(sys.argv) > 3:
    import gzip, re
    sys.path.insert(0, __import__("os").environ.get("SEEDDRIFT_HOME", "."))
    from seeddrift.donors import taxonomy
    tax = taxonomy(sys.argv[2])
    c = collections.Counter()
    for line in gzip.open(sys.argv[3], "rt"):
        if line[0] == ">":
            m = re.search(r" OX=(\d+)", line)
            if m: c[tax.group(int(m.group(1)))] += 1
    n = sum(c.values())
    print("\nSwiss-Prot make-up:", ", ".join("%s %.1f%%" % (g, 100.0 * k / n) for g, k in c.most_common()))
