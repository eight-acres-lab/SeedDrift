"""Agreement of the search engines on the same queries (part A results of benchmark_gpu.py).

DIAMOND --sensitive is the reference. For each engine: queries with any hit, queries with an
outside-Enterobacterales hit covering >= 40% of the query, overlap of the top-50 target lists,
and agreement of the donor group of the best outside hit.
usage: python bench/search_agreement.py /hy-tmp/hgtscan
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from seeddrift.donors import _fasta_gz, _load_hits, taxonomy  # noqa: E402

root = sys.argv[1]
d = os.path.join(root, "bench/gpu/search")
tax = taxonomy(os.path.join(root, "ref"))
names = [n for n, _ in _fasta_gz(os.path.join(d, "q1k.faa"))]
engines = ["diamond_sensitive", "mmseqs_cpu_s7.5", "mmseqs_gpu"]
hits = {e: _load_hits(os.path.join(d, e + ".m8")) for e in engines}


def best_outside(hl):
    for bits, ev, acc, t, pid, qcov in hl:
        if qcov >= 0.4 and 91347 not in tax.lineage(t):
            return tax.group(t), bits
    return None, None


res = {}
for e in engines:
    h = hits[e]
    outs = {q: best_outside(h.get(q, [])) for q in names}
    res[e] = {"queries_with_hits": sum(1 for q in names if h.get(q)),
              "queries_with_outside_hit": sum(1 for q in names if outs[q][0]),
              "median_hits": sorted(len(h.get(q, [])) for q in names)[len(names) // 2]}
    if e != "diamond_sensitive":
        base = hits["diamond_sensitive"]
        jac, same, both = [], 0, 0
        for q in names:
            a = set(x[2] for x in base.get(q, [])[:50])
            b = set(x[2] for x in h.get(q, [])[:50])
            if a or b:
                jac.append(len(a & b) / float(len(a | b)))
            ga = best_outside(base.get(q, []))[0]
            gb = outs[q][0]
            if ga and gb:
                both += 1
                same += ga == gb
        res[e]["top50_jaccard_vs_diamond"] = round(sum(jac) / len(jac), 3)
        res[e]["outside_group_agreement_vs_diamond"] = round(same / float(both), 3) if both else None
        res[e]["outside_group_compared"] = both
with open(os.path.join(root, "bench/gpu/part_A_agreement.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))
