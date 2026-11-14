"""Repeat-like regions of proteins: hits on them say little about where a gene came from.

Three sources, all reported as 1-based inclusive intervals:
  - Pfam-A families of type Repeat / Coiled-coil / Disordered, or in a clan made mostly of Repeat
    families (LRR, TPR, ankyrin, beta-propellers ...)   -> pfam_repeat_regions (pyhmmer)
  - tandem repeats: BLOSUM62 self-comparison, off-diagonal windows TANDEM_Z SDs above the expectation
    for the sequence's own composition                    -> tandem_regions (numpy, no dependency)
  - coiled coils: COILS / ncoils (MTIDK, window 28, -w)  -> coil_regions

A hit is repeat-driven when >= MAX_REPEAT_OVERLAP of its aligned segment (query or target side) lies on
such regions and it is < KEEP_IDENTITY identical; a gene is "repeat-like" when repeat-driven hits carry
>= REPEAT_DOMINANCE of the bit score of its top outside hits.
"""
from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

REPEAT_TYPES = ("Repeat", "Coiled-coil", "Disordered")     # Pfam TP field
MAX_REPEAT_OVERLAP = 0.5     # a hit is repeat-driven when this much of its aligned segment is repeat-like
REPEAT_DOMINANCE = 0.5       # an event is "repeat-like" when such hits carry this share of the top-10 bits
KEEP_IDENTITY = 0.40         # repeat-driven hits this identical are kept: real homologs, not profile drift
TANDEM_Z = 6.0               # tandem-repeat window score: SDs above the composition-based expectation
COIL_MIN_P = 0.9             # COILS probability for a coiled-coil residue (window 28, -w)


def _s(x):
    return x.decode() if isinstance(x, bytes) else (x or "")


def _pfam_meta(dat):
    """Pfam-A.hmm.dat(.gz) -> {accession without version: (id, type, clan, repeat_like)}."""
    meta, cur = {}, {}
    op = gzip.open if dat.endswith(".gz") else open
    with op(dat, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("#=GF"):
                f = line.rstrip("\n").split(None, 2)
                if len(f) == 3:
                    cur[f[1]] = f[2].strip()
            elif line.startswith("//"):
                if "AC" in cur:
                    meta[cur["AC"].split(".")[0]] = [cur.get("ID", ""), cur.get("TP", ""), cur.get("CL", "")]
                cur = {}
    clan = defaultdict(Counter)
    for i, tp, cl in meta.values():
        if cl:
            clan[cl][tp == "Repeat"] += 1
    rep_clan = set(cl for cl, c in clan.items() if c[True] >= c[False])
    return {k: (i, tp, cl, tp in REPEAT_TYPES or cl in rep_clan) for k, (i, tp, cl) in meta.items()}



AA = "ARNDCQEGHILKMFPSTWYV"


def _blosum():
    """BLOSUM62 as a 21x21 array (index 20: any other residue, scored -1)."""
    import numpy as np
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    B = -np.ones((21, 21))
    B[:20, :20] = [[m[a][b] for b in AA] for a in AA]
    return B


def tandem_regions(seq, B=None, z=TANDEM_Z, dmin=3, dmax=200, min_len=20):
    """Internal tandem repeats: windows along an off-diagonal (offset d) of the BLOSUM62 self-comparison
    whose summed score exceeds the expectation for the sequence's own composition by z standard
    deviations. The window spans about one repeat unit (12-40 residues). Returns 1-based intervals
    covering both copies; compositional bias alone does not trigger it."""
    import numpy as np
    B = _blosum() if B is None else B
    idx = {c: i for i, c in enumerate(AA)}
    x = np.array([idx.get(c, 20) for c in seq.upper()], dtype=np.int64)
    L = len(x)
    if L < 2 * dmin + 12:
        return []
    p = np.bincount(x, minlength=21)[:20].astype(float)
    p /= max(1.0, p.sum())
    mu = p @ B[:20, :20] @ p
    sd = np.sqrt(max(1e-9, p @ (B[:20, :20] ** 2) @ p - mu * mu))
    cov = np.zeros(L + 1)
    for d in range(dmin, min(dmax, L // 2) + 1):
        w = min(max(d, 12), 40)
        sc = B[x[:-d], x[d:]]
        if len(sc) < w:
            continue
        c = np.cumsum(np.concatenate([[0.0], sc]))
        hit = np.nonzero(c[w:] - c[:-w] >= w * mu + z * sd * np.sqrt(w))[0]
        if len(hit):
            t = w // 4                                  # mark the window core: windows that only
            for off in (0, d):                          # straddle a repeat edge do not leak far
                np.add.at(cov, hit + off + t, 1)
                np.add.at(cov, np.minimum(hit + off + w - t, L), -1)
    on = np.cumsum(cov[:L]) > 0
    out, i = [], 0
    while i < L:
        if on[i]:
            j = i
            while j + 1 < L and on[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i + 1, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def coil_regions(seqs, ncoils, win=28, min_p=COIL_MIN_P, min_len=21):
    """Coiled coils with COILS (ncoils; MTIDK matrix, a/d weighted like the others: -w).
    seqs: {id: sequence}; returns {id: 1-based intervals}. Empty if ncoils is not available."""
    exe = shutil.which(ncoils) if ncoils else None
    if not exe:
        log("ncoils not found (%s): coiled coils are not checked" % ncoils)
        return {}
    env = dict(os.environ)
    if "COILSDIR" not in env:                          # bioconda keeps new.mat in <env>/share/coils
        env["COILSDIR"] = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(exe))), "share", "coils")
    fa = "".join(">%s\n%s\n" % (k, v) for k, v in seqs.items())
    r = subprocess.run([exe, "-f", "-w", "-win", str(win), "-min_P", str(min_p)], input=fa, env=env,
                       capture_output=True, text=True, check=True)
    out, name, buf = {}, None, []

    def flush():
        if name is None:
            return
        m = "".join(buf)
        iv, i = [], 0
        while i < len(m):
            if m[i] == "x":
                j = i
                while j + 1 < len(m) and m[j + 1] == "x":
                    j += 1
                if j - i + 1 >= min_len:
                    iv.append((i + 1, j + 1))
                i = j + 1
            else:
                i += 1
        if iv:
            out[name] = iv
    for line in r.stdout.splitlines():
        if line.startswith(">"):
            flush()
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    flush()
    return out


def _hmm_path(pfam_dir):
    hmm = os.path.join(pfam_dir, "Pfam-A.hmm")
    if not os.path.exists(hmm):
        with gzip.open(hmm + ".gz", "rb") as fi, open(hmm + ".part", "wb") as fo:
            shutil.copyfileobj(fi, fo, 1 << 24)
        os.replace(hmm + ".part", hmm)
    return hmm


def _overlap(regions, a, b):
    """Fraction of the query segment a..b (1-based, inclusive) that lies on any of the regions."""
    if not regions or b < a:
        return 0.0
    cov = set()
    for s, e in regions:
        cov.update(range(max(s, a), min(e, b) + 1))
    return len(cov) / float(b - a + 1)



def log(msg):
    print("[repeats] " + msg, file=sys.stderr, flush=True)


_HMM_CACHE = {}


def pfam_repeat_regions(seqs, pfam_dir, threads=8, all_domains=False):
    """{id: [(start, end, pfam_id, repeat_like)]} from pyhmmer hmmsearch (gathering cut-offs).
    Only repeat-like families are searched unless all_domains is set."""
    import pyhmmer

    key = (os.path.abspath(pfam_dir), all_domains)
    if key not in _HMM_CACHE:                          # reading Pfam-A takes ~1 min: keep it in memory
        meta = _pfam_meta(os.path.join(pfam_dir, "Pfam-A.hmm.dat.gz"))
        with pyhmmer.plan7.HMMFile(_hmm_path(pfam_dir)) as hf:
            hmms = [h for h in hf if all_domains or meta.get(_s(h.accession).split(".")[0], (0, 0, 0, False))[3]]
        _HMM_CACHE[key] = (meta, hmms)
    meta, hmms = _HMM_CACHE[key]
    out = defaultdict(list)
    if not seqs:
        return out
    with tempfile.NamedTemporaryFile("w", suffix=".faa", delete=False) as fo:
        for k, v in seqs.items():
            fo.write(">%s\n%s\n" % (k, v.replace("*", "")))
        tmp = fo.name
    try:
        abc = pyhmmer.easel.Alphabet.amino()
        with pyhmmer.easel.SequenceFile(tmp, digital=True, alphabet=abc) as sf:
            block = sf.read_block()
        for top in pyhmmer.hmmsearch(hmms, block, cpus=threads, bit_cutoffs="gathering"):
            q = getattr(top, "query", None)
            acc = _s(getattr(q, "accession", None) if q is not None else getattr(top, "query_accession", None))
            pid, _, _, rep = meta.get(acc.split(".")[0], (_s(getattr(q, "name", None)), "", "", False))
            for hit in top:
                if hit.included:
                    for d in hit.domains:
                        if d.included:
                            out[_s(hit.name)].append((d.env_from, d.env_to, pid, rep))
    finally:
        os.remove(tmp)
    return out


def annotate(seqs, pfam_dir=None, ncoils="ncoils", threads=8, tandem=True):
    """All repeat-like regions: {id: [(start, end, kind)]}, kind = Pfam family, tandem_repeat or coiled_coil."""
    out = defaultdict(list)
    if pfam_dir:
        for k, v in pfam_repeat_regions(seqs, pfam_dir, threads).items():
            out[k].extend((a, b, n) for a, b, n, rep in v if rep)
    if tandem:
        B = _blosum()
        for k, v in seqs.items():
            out[k].extend((a, b, "tandem_repeat") for a, b in tandem_regions(v, B))
    for k, v in coil_regions(seqs, ncoils).items():
        out[k].extend((a, b, "coiled_coil") for a, b in v)
    return out


def is_repeat_hit(qreg, treg, qa, qb, ta, tb, identity):
    """identity as a fraction; coordinates 1-based inclusive (None when unknown)."""
    if identity >= KEEP_IDENTITY:
        return False
    if qa is not None and _overlap(qreg, qa, qb) >= MAX_REPEAT_OVERLAP:
        return True
    return ta is not None and _overlap(treg, ta, tb) >= MAX_REPEAT_OVERLAP
