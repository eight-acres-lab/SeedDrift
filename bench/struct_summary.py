import csv, sys, collections
sys.path.insert(0, __import__("os").environ.get("SEEDDRIFT_HOME", "."))
B = "/hy-tmp/hgtscan/struct/run/"
pred = {r["gene_id"]: r for r in csv.DictReader(open(B + "predict.tsv"), delimiter="\t")}
prof = {r["gene_id"]: r for r in csv.DictReader(open(B + "profile/donors_by_gene.tsv"), delimiter="\t")}
q = list(csv.DictReader(open(B + "queries.tsv"), delimiter="\t"))
def pl(g):
    v = pred.get(g, {}).get("mean_plddt", "NA")
    return None if v in ("NA", "") else float(v)
skipped = [g for g in pred if pl(g) is None]
print("predicted", len(pred) - len(skipped), "skipped", len(skipped), [pred[g]["length"] for g in skipped])
def lb(n):
    n = int(n); return "<100" if n < 100 else ("100-299" if n < 300 else ">=300")
def pb(v):
    return "NA" if v is None else (">=70" if v >= 70 else ("50-70" if v >= 50 else "<50"))
tab = collections.Counter(); tot = collections.Counter()
for r in q:
    v = pl(r["gene_id"]); tab[(lb(r["length"]), pb(v))] += 1; tot[pb(v)] += 1
print("pLDDT overall", dict(tot))
for L in ("<100", "100-299", ">=300"):
    n = sum(tab[(L, p)] for p in (">=70", "50-70", "<50", "NA"))
    print(" len %-7s n=%4d  >=70 %3d  50-70 %3d  <50 %3d  NA %d" % (L, n, tab[(L, ">=70")], tab[(L, "50-70")], tab[(L, "<50")], tab[(L, "NA")]))
vals = sorted(v for v in (pl(g) for g in pred) if v is not None)
print("median pLDDT %.1f" % vals[len(vals)//2])
# profile search outcome x pLDDT
hit = [r for r in prof.values() if r["top_group"] != "no outside homolog"]
print("profile: outside homolog for", len(hit), "of", len(prof))
c = collections.Counter(pb(pl(r["gene_id"])) for r in hit); print(" pLDDT of those", dict(c))
c = collections.Counter(r["top_group"] for r in hit); print(" groups", c.most_common())
# P_HGT x outcome
ph = collections.Counter((float(r["P_HGT"]) >= 0.9, r["top_group"] != "no outside homolog") for r in prof.values())
print(" P>=0.9 with hit %d/%d ; P<0.9 with hit %d/%d" % (ph[(True, True)], ph[(True, True)] + ph[(True, False)], ph[(False, True)], ph[(False, True)] + ph[(False, False)]))
print("\n-- mammal / vertebrate hits --")
for r in prof.values():
    if r["mammal_hit"] or r["top_group"] in ("Vertebrata, other", "Metazoa, other"):
        print("%s len=%s pLDDT=%s P=%s top=%s(%s) best=%s %s %s%% bits_out=%s AI=%s | mammal: %s"
              % (r["gene_id"], pred.get(r["gene_id"], {}).get("length"), pl(r["gene_id"]), r["P_HGT"], r["top_group"], r["top_group_share"],
                 r["best_hit_taxon"], r["best_hit_group"], r["best_hit_pident"], r["best_out_bits"], r["AI"], r["mammal_hit"]))
print("\n-- strongest non-Enterobacterales hits --")
rs = sorted((r for r in hit if r["top_group"] != "Enterobacterales, other genera"), key=lambda r: -float(r["best_out_bits"] or 0))
for r in rs[:15]:
    print("%s len=%s pLDDT=%s top=%s best=%s %s%% bits=%s" % (r["gene_id"], pred.get(r["gene_id"], {}).get("length"), pl(r["gene_id"]), r["top_group"], r["best_hit_taxon"], r["best_hit_pident"], r["best_out_bits"]))
