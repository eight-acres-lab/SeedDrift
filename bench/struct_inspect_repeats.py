import csv, sys
sys.path.insert(0, __import__("os").environ.get("SEEDDRIFT_HOME", "."))
from seeddrift.structure import _overlap, _acc, _load_pfam
B = "/hy-tmp/hgtscan/struct/run/"
treg = {}
for r in csv.DictReader(open(B + "pfam_targets.tsv"), delimiter="\t"):
    treg.setdefault(r["target"], []).append((int(r["start"]), int(r["end"])))
tdom = {}
for r in csv.DictReader(open(B + "pfam_targets.tsv"), delimiter="\t"):
    tdom.setdefault(r["target"], set()).add(r["pfam_id"])
qreg, _ = _load_pfam(B)
rs = list(csv.DictReader(open(B + "structure_donors.tsv"), delimiter="\t"))
euk = ("Mammalia", "Vertebrata, other", "Metazoa, other", "Fungi", "Viridiplantae", "Eukaryota, other")
hits = {}
for line in open(B + "hits.m8"):
    f = line.rstrip("\n").split("\t"); hits.setdefault(f[0], []).append(f)
for r in rs:
    if r["top_group"] in euk or r["top_group_all"] in euk:
        g = r["gene_id"]
        print("== %s len=%s pfam=%s | all=%s -> %s nrep=%s keep=%s best=%s" % (g, r["length"], r["pfam"][:50], r["top_group_all"], r["top_group"], r["n_repeat_hits"], r["n_struct_hits"], r["best_hit_name"][:45]))
        for f in sorted(hits[g], key=lambda f: -float(f[3]))[:4]:
            a = _acc(f[1]); q = _overlap(qreg.get(g), int(f[10]), int(f[11])); t = _overlap(treg.get(a), int(f[12]), int(f[13]))
            print("    %-10s TM=%.2f q%s-%s t%s-%s qrep=%.2f trep=%.2f tdom=%s" % (a, float(f[5]), f[10], f[11], f[12], f[13], q, t, ",".join(sorted(tdom.get(a, [])))[:50]))
