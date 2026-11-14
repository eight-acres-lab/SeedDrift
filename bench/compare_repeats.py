"""What the tandem-repeat and coiled-coil checks changed, relative to the Pfam-only filter."""
import csv, collections, sys
sys.path.insert(0, __import__("os").environ.get("SEEDDRIFT_HOME", "."))
from seeddrift.structure import _overlap, _acc
B = "/hy-tmp/hgtscan/struct/run/"
rd = lambda p: {r["gene_id"]: r for r in csv.DictReader(open(B + p), delimiter="\t")}
s0, s1 = rd("structure_donors_pfam.tsv"), rd("structure_donors.tsv")
p0, p1 = rd("profile_nr_pfam/donors_by_gene.tsv"), rd("profile_nr/donors_by_gene.tsv")
q = {}
for r in csv.DictReader(open(B + "pfam.tsv"), delimiter="\t"):
    q.setdefault(r["gene_id"], collections.defaultdict(list))[r["pfam_id"] if r["pfam_id"] in ("tandem_repeat", "coiled_coil") else "pfam"].append((int(r["start"]), int(r["end"])))
t = {}
for r in csv.DictReader(open(B + "repeat_targets.tsv"), delimiter="\t"):
    t.setdefault(r["target"], collections.defaultdict(list))[r["pfam_id"] if r["pfam_id"] in ("tandem_repeat", "coiled_coil") else "pfam"].append((int(r["start"]), int(r["end"])))
def why(g, hits, qi, ti, key):
    c = collections.Counter()
    for f in hits.get(g, [])[:10]:
        for side, reg, a, b in (("q", q.get(g, {}), int(f[qi]), int(f[qi + 1])), ("t", t.get(key(f[1]), {}), int(f[ti]), int(f[ti + 1]))):
            for k, iv in reg.items():
                if _overlap(iv, a, b) >= 0.5: c[side + ":" + k] += 1
    return ",".join("%s %d" % kv for kv in c.most_common())
def load(path):
    h = collections.defaultdict(list)
    for line in open(path):
        f = line.rstrip("\n").split("\t"); h[f[0]].append(f)
    for v in h.values(): v.sort(key=lambda f: -float(f[3] if len(f) > 13 else f[5]))
    return h
hs = load(B + "hits.m8"); hp = load(B + "profile/hits.m8")
for v in hp.values(): v.sort(key=lambda f: -float(f[5]))
print("== structure: changed calls")
for g in s0:
    if s0[g]["top_group"] != s1[g]["top_group"]:
        print("%-38s %-28s -> %-28s pLDDT=%-5s | %s | %s" % (g, s0[g]["top_group"], s1[g]["top_group"], s1[g]["mean_plddt"],
              why(g, hs, 10, 12, _acc), s0[g]["best_hit_name"][:40]))
print("== profile: changed calls")
for g in p0:
    if p0[g]["top_group"] != p1[g]["top_group"]:
        print("%-38s %-28s -> %-28s pLDDT=%-5s | %s | best was %s %s%%" % (g, p0[g]["top_group"], p1[g]["top_group"], s1[g]["mean_plddt"],
              why(g, hp, 8, 10, lambda x: x.rpartition("|")[0]), p0[g]["best_hit_taxon"][:30], p0[g]["best_hit_pident"]))
