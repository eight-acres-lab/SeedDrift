"""Sample query proteins from the reference library for self-supervised placement training.
Proteomes are drawn per taxonomic group; proteins are a reservoir sample per proteome.
Writes queries_a.faa (a third of the proteomes), queries_b.faa and proteomes_sampled.tsv to $SEEDDRIFT_DATA/placement.

usage:  SEEDDRIFT_DATA=... python research/placement/sample.py 44 100"""
import collections, csv, os, random, sys
R = os.path.join(os.environ["SEEDDRIFT_DATA"], "reference")
OUT = os.path.join(os.environ["SEEDDRIFT_DATA"], "placement")
PER_GROUP, PER_PROTEOME, SEED = int(sys.argv[1]), int(sys.argv[2]), 20261115   # 44 proteomes per group, 100 proteins each, in the run reported
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(SEED)
rows = list(csv.DictReader(open(R + "/proteomes.tsv"), delimiter="\t"))
by = collections.defaultdict(list)
for r in rows:
    if int(r["n_proteins"]) >= 300:                      # enough genes for a genome-level context
        by[r["group"]].append(r)
chosen = {}
for g, rs in sorted(by.items()):
    rnd.shuffle(rs)
    for r in rs[:PER_GROUP]:
        chosen[int(r["taxid"])] = r
print("groups", {g: (len(rs), min(len(rs), PER_GROUP)) for g, rs in sorted(by.items())}, file=sys.stderr)
print("proteomes chosen", len(chosen), file=sys.stderr)
faa = R + "/reference.faa"
res, seen = collections.defaultdict(list), collections.Counter()
name, seq, tx = None, [], None
def flush():
    if name is None or tx not in chosen:
        return
    s = "".join(seq)
    if not 80 <= len(s) <= 1500:
        return
    seen[tx] += 1
    k = seen[tx]
    if len(res[tx]) < PER_PROTEOME:
        res[tx].append((name, s))
    else:
        j = rnd.randrange(k)
        if j < PER_PROTEOME:
            res[tx][j] = (name, s)
with open(faa) as fh:
    for line in fh:
        if line.startswith(">"):
            flush()
            name = line[1:].split()[0]
            t = name.rpartition("|")[2]
            tx = int(t) if t.isdigit() else None
            seq = []
        elif tx in chosen:
            seq.append(line.strip())
flush()
order = sorted(res)
rnd.shuffle(order)
half = len(order) // 3
for part, txs in (("a", order[:half]), ("b", order[half:])):
    with open("%s/queries_%s.faa" % (OUT, part), "w") as fo:
        for t in txs:
            for n, s in res[t]:
                fo.write(">%s\n%s\n" % (n, s))
with open(OUT + "/proteomes_sampled.tsv", "w") as fo:
    fo.write("taxid\tgroup\tproteome\tn_sampled\tpart\tspecies\n")
    for part, txs in (("a", order[:half]), ("b", order[half:])):
        for t in txs:
            r = chosen[t]
            fo.write("%d\t%s\t%s\t%d\t%s\t%s\n" % (t, r["group"], r["proteome"], len(res[t]), part, r["species"]))
print("queries", sum(len(v) for v in res.values()), "proteomes with queries", len(res), file=sys.stderr)
