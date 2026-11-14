"""Calibrate the tandem-repeat z and COILS P cut-offs: shuffled vs real E. coli K-12 proteins, known artifacts."""
import random, sys, time
sys.path.insert(0, __import__("os").environ.get("SEEDDRIFT_HOME", "."))
from seeddrift.donors import _fasta_gz
from seeddrift.structure import tandem_regions, coil_regions, _blosum
B = _blosum()
k12 = {n.split()[0]: s.rstrip("*") for n, s in _fasta_gz("/hy-tmp/hgtscan/real/enterobacterales/proteomes/Ecoli_K12.faa")}
q = {n.split()[0]: s for n, s in _fasta_gz("/hy-tmp/hgtscan/struct/run/queries.faa")}
random.seed(7)
shuf = {k: "".join(random.sample(v, len(v))) for k, v in k12.items()}
pos = {"tandem ORF 96_5": "Efergusonii__ENSB_96_5gxym4VHlvZI", "coiled coil iA_fi8": "Csakazakii__ENSB_iA_fi8lO2q0aRFQ",
       "LRR IpaH 7Buw": "Ecoli_O157__ENSB_7BuwLNUO8VsV_Oh", "pentapeptide AAC77036": "Ecoli_K12__AAC77036",
       "ankyrin AAC76987": "Ecoli_K12__AAC76987", "Sel1 Sa5G": "Ecoli_CFT073__ENSB_Sa5GdE_owKzVcl4",
       "kinase YopO-like": "Yenterocolitica__ENSB_-6EkLVVzoXkUaS2"}
def frac(reg, L):
    return sum(e - s + 1 for s, e in reg) / float(L)
print("K-12 proteins", len(k12), "queries", len(q))
for z in (5.0, 6.0, 7.0):
    t0 = time.time()
    real = {k: tandem_regions(v, B, z=z) for k, v in k12.items()}
    sh = {k: tandem_regions(v, B, z=z) for k, v in shuf.items()}
    qq = {k: tandem_regions(v, B, z=z) for k, v in q.items()}
    print("z=%.0f: K-12 with tandem repeat %d (%.1f%%), shuffled %d (%.2f%%), queries %d (%.1f%%); %.0f s"
          % (z, sum(1 for v in real.values() if v), 100.0 * sum(1 for v in real.values() if v) / len(k12),
             sum(1 for v in sh.values() if v), 100.0 * sum(1 for v in sh.values() if v) / len(k12),
             sum(1 for v in qq.values() if v), 100.0 * sum(1 for v in qq.values() if v) / len(q), time.time() - t0))
    print("   positives:", ", ".join("%s %.2f" % (lab, frac(qq.get(g, []), len(q[g]))) for lab, g in pos.items()))
    if z == 6.0:
        top = sorted(((frac(v, len(k12[k])), k) for k, v in real.items() if v), reverse=True)[:12]
        print("   K-12 most repetitive:", ", ".join("%s %.2f" % (k, f) for f, k in top))
COILS = "/hy-tmp/hgtscan/mamba/envs/cc/bin/ncoils"
for p in (0.5, 0.9):
    real, sh, qq = coil_regions(k12, COILS, min_p=p), coil_regions(shuf, COILS, min_p=p), coil_regions(q, COILS, min_p=p)
    print("COILS P>=%.1f: K-12 %d (%.1f%%), shuffled %d (%.2f%%), queries %d (%.1f%%)" % (p, len(real), 100.0 * len(real) / len(k12),
          len(sh), 100.0 * len(sh) / len(k12), len(qq), 100.0 * len(qq) / len(q)))
    print("   positives:", ", ".join("%s %.2f" % (lab, frac(qq.get(g, []), len(q[g]))) for lab, g in pos.items()))
    if p == 0.9:
        top = sorted(((frac(v, len(k12[k])), k) for k, v in real.items()), reverse=True)[:12]
        print("   K-12 most coiled:", ", ".join("%s %.2f" % (k, f) for f, k in top))
