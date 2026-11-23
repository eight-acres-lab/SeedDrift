"""The readouts of package W18 on ruler tables, their controls and their statistics (protocol w18-v0), from the
per-cell bits of the passes of w18_passes.py. numpy only: no torch, no ruler generator. The readouts' one
implementation; W11 reads through it. Every result is "construct only".

Cells every readout and control leaves out: truth OTHER, the first codon of a block, and the last codon of a block that
reaches its gene's end. Every score is oriented so that higher means more foreign. Per table, from one arm:
  source_shift     1/2 sum_t |pT(t) - pC(t)| over the tiers t (of the rows' distance to A) present, where pT is the
                   share of tier t in the positive costs dT(t) = mean bits of the target's scattered cells with tier t
                   removed minus with every row, pC the same over the context cells; 0 with one tier; undefined (nan)
                   when either block relies on no tier (positive costs below UNDEFINED bits per codon)
  gain             the two-block gain [sum over the first half (W - H1) + sum over the second half (W - H2)] / (2 cells)
  gain_norel       the same from W0, H10, H20 (every other row removed); gain_rel = gain - gain_norel
  *_aa             the same from the bits of the amino acid alone (the bits a codon's amino acid costs): the lineage
                   reading, free of the donor's codon usage
  abstain          (T0 - T) below ABSTAIN bits per codon (a table without other rows abstains): reported, not scored
  composition      -log2 p(codon | amino acid) of the arm without relatives under pass N, over the cells kept (and an
                   amino acid of two codons or more), per cell; composition_rel divides each cell by log2 of the codons
Controls (no model): best_similarity, rows_on_target, covered_share, coverage_shift (the source-shift formula on the
share of cells the rows of each tier cover on the target and on the context), host_codon and host_codon_rel (the
composition readout with p(codon | amino acid) from A's codon counts over its other genes, plus 0.5 per codon).
Exploratory, outside the pre-registered protocol (the programme lead, 2026-11-23; never to change a conclusion or choose
anything): host_aa (-log2 p(amino acid) from A's amino-acid counts over its other genes, plus 0.5 per amino acid, over the
same cells) and the composition of the target itself, x_gc (its share of G and C) and x_aa_distance (the total
variation between its amino-acid composition and A's), which the strata of Triples.exploratory read.

Statistics (stat_rows): the primary statistic is the paired win rate inside a triple, P(implant > own) + 1/2 P(tie), per
group of A, then the plain mean over the groups of a block that are readable; the pooled AUC per group (every implant
table against every control table of the group and stratum) is secondary; implant against original is the position
control. Intervals: a two-level bootstrap, A's split units inside each group and D's split units, the same replicates
for every score (differences paired).
"""
from __future__ import annotations

import warnings

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.evalh import score as SC
from seeddrift.train.rel.readouts import w18_passes as P

VERSION = 2
UNDEFINED = 0.02
ABSTAIN = 0.05
MIN_COLUMNS = 10
PSEUDO = 0.5
ENTROPY_BINS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, np.inf)
UNCALIBRATED = 0.1
IMPLANT, ORIGINAL, OWN = 0, 1, 2
COMPARISONS = (("own", OWN), ("original", ORIGINAL))         # original: the position control
BOOT, BOOT_SEED = 2000, 20261124
MIN_GROUP, MIN_GROUPS, MIN_TRIPLES, MIN_UNITS = 10, 3, 30, 8
NEAREST_BINS = ((0.0, 0.3, "<0.3"), (0.3, 0.5, "0.3-0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 1.01, ">=0.7"))
FILL_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 99, "8-15"))
RATIO_BINS = ((0.0, 0.8, "<0.8"), (0.8, 1.25, "0.8-1.25"), (1.25, np.inf, ">=1.25"))
T0_, T1_ = P.T0_, P.T1_
BLOCK_OF, OFFSET_OF = P.BLOCK_OF, P.OFFSET_OF
CLASS = {"archaea": "prokaryote", "bacteria": "prokaryote", "mobile.plasmid": "prokaryote"}
DIRECTIONS = ("prokaryote <-> prokaryote", "prokaryote -> eukaryote", "eukaryote -> prokaryote", "eukaryote <-> eukaryote", "virus -> host",
              "host -> virus", "virus <-> virus")


def class_of(group):
    g = str(group)
    return CLASS.get(g) or ("eukaryote" if g.startswith("eukaryota.") else ("virus" if g.startswith("viruses.") else "other"))


def direction(donor_group, recipient_group):
    d, r = class_of(donor_group), class_of(recipient_group)
    if d == "virus" and r == "virus":
        return DIRECTIONS[6]
    if d == "virus":
        return DIRECTIONS[4]
    if r == "virus":
        return DIRECTIONS[5]
    return {("prokaryote", "prokaryote"): DIRECTIONS[0], ("prokaryote", "eukaryote"): DIRECTIONS[1], ("eukaryote", "prokaryote"): DIRECTIONS[2],
            ("eukaryote", "eukaryote"): DIRECTIONS[3]}.get((d, r), "other")


def _classes():
    cls = SC.amino_acid_classes()                       # 1..20 amino acids, 21 stop, 22 other; the genetic code, reporting only
    deg = np.bincount(cls[1:65], minlength=23)
    deg[22] = 1
    return cls, deg


def kept(cells, meta, n0):
    """(C,) the cells the readouts count: truth not OTHER, not the first codon of its block, not the last codon of a block
    that reaches its gene's end. n0: (N, B) row 0's segment length of every block."""
    tab = cells["table"].astype(np.int64)
    slot = cells["slot"].astype(np.int64)
    b, off = BLOCK_OF[slot], OFFSET_OF[slot]
    last = meta["block_ends"][tab, b] & (off == np.asarray(n0)[tab, b] - 1)
    return (cells["truth"] != api.OTHER) & (off > 0) & ~last


def pass_sums(cells, n_tables, keep, value):
    """{pass name: (sum [N], cells [N])} of `value` over the kept cells."""
    out = {}
    pid, tab = cells["pass_id"].astype(np.int64), cells["table"].astype(np.int64)
    names = [str(x) for x in cells["pass_names"]]
    for k in np.unique(pid[keep]):
        m = keep & (pid == k)
        s, c = np.zeros(n_tables), np.zeros(n_tables)
        np.add.at(s, tab[m], value[m])
        np.add.at(c, tab[m], 1)
        out[names[k]] = (s, c)
    return out


def _mean(sums, name, n):
    s, c = sums.get(name, (np.zeros(n), np.zeros(n)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def _half(cells, n_tables, keep, value, target_len, name, half):
    names = [str(x) for x in cells["pass_names"]]
    if name not in names:
        return np.zeros(n_tables), np.zeros(n_tables)
    m = keep & (cells["pass_id"].astype(np.int64) == names.index(name))
    tab = cells["table"][m].astype(np.int64)
    off = cells["slot"][m].astype(np.int64) - T0_
    sec = off >= (np.asarray(target_len, dtype=np.int64)[tab] // 2)
    sel = sec if half else ~sec
    s, c = np.zeros(n_tables), np.zeros(n_tables)
    np.add.at(s, tab[sel], value[m][sel])
    np.add.at(c, tab[sel], 1)
    return s, c


def _gain(cells, n, keep, value, target_len, prefix, w, h1, h2):
    w1, c1 = _half(cells, n, keep, value, target_len, prefix + w, 0)
    w2, c2 = _half(cells, n, keep, value, target_len, prefix + w, 1)
    a1, d1 = _half(cells, n, keep, value, target_len, prefix + h1, 0)
    a2, d2 = _half(cells, n, keep, value, target_len, prefix + h2, 1)
    ok = (c1 > 0) & (c2 > 0) & (c1 == d1) & (c2 == d2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ok, ((w1 - a1) + (w2 - a2)) / (2 * np.maximum(c1 + c2, 1)), np.nan)


def _shift(t_all, c_all, removed_t, removed_c, tiers):
    n = len(t_all)
    shift = np.full(n, np.nan)
    for t in range(n):
        present = sorted({int(x) for x in tiers[t][1:] if x >= 0})
        if not present or not np.isfinite(t_all[t]) or not np.isfinite(c_all[t]):
            continue
        dT = np.array([removed_t[k][t] - t_all[t] for k in present])
        dC = np.array([removed_c[k][t] - c_all[t] for k in present])
        if not (np.isfinite(dT).all() and np.isfinite(dC).all()):
            continue
        pT, pC = np.maximum(dT, 0), np.maximum(dC, 0)
        if pT.sum() < UNDEFINED or pC.sum() < UNDEFINED:
            continue
        shift[t] = 0.0 if len(present) == 1 else 0.5 * np.abs(pT / pT.sum() - pC / pC.sum()).sum()
    return shift


def readouts(cells, meta, n0, prefix=""):
    """{score: [N]} from the cells of one arm with relatives (prefix "P:" reads its presence passes): source_shift,
    source_shift_top (undefined at the top), gain, gain_norel, gain_rel, the _aa version of each, and abstain."""
    n = len(meta["triple"])
    keep = kept(cells, meta, n0)
    out = {}
    for suffix, value in (("", cells["bits"].astype(np.float64)), ("_aa", cells["aa_bits"].astype(np.float64))):
        sums = pass_sums(cells, n, keep, value)
        t_all, c_all = _mean(sums, prefix + "T", n), _mean(sums, prefix + "C", n)
        rt = {k: _mean(sums, "%sT-t%d" % (prefix, k), n) for k in range(P.MAX_TIERS)}
        rc = {k: _mean(sums, "%sC-t%d" % (prefix, k), n) for k in range(P.MAX_TIERS)}
        shift = _shift(t_all, c_all, rt, rc, meta["row_tier"])
        gain = _gain(cells, n, keep, value, meta["target_len"], prefix, "W", "H1", "H2")
        out["source_shift" + suffix] = shift
        out["source_shift_top" + suffix] = np.where(np.isnan(shift), np.inf, shift)
        out["gain" + suffix] = gain
        if not prefix:
            norel = _gain(cells, n, keep, value, meta["target_len"], "", "W0", "H10", "H20")
            out["gain_norel" + suffix] = norel
            out["gain_rel" + suffix] = gain - norel
        if not prefix and not suffix:
            t0 = _mean(sums, "T0", n)
            ab = np.where(np.isfinite(t0), (t0 - t_all) < ABSTAIN, True).astype(np.float64)
            ab[~np.isfinite(t_all)] = np.nan
            out["abstain"] = ab
    return out


def composition(cells, meta, n0):
    """{composition, composition_rel: [N]} from the cells of pass N of an arm without relatives."""
    n = len(meta["triple"])
    cls, deg = _classes()
    names = [str(x) for x in cells["pass_names"]]
    out = {"composition": np.full(n, np.nan), "composition_rel": np.full(n, np.nan)}
    if "N" not in names:
        return out
    truth = cells["truth"].astype(np.int64)
    m = kept(cells, meta, n0) & (cells["pass_id"].astype(np.int64) == names.index("N")) & (deg[cls[np.clip(truth, 0, 65)]] >= 2) & (truth <= 64)
    tab = cells["table"][m].astype(np.int64)
    c = (cells["bits"][m] - cells["aa_bits"][m]).astype(np.float64)
    rel = c / np.log2(np.maximum(deg[cls[truth[m]]], 2))
    for name, v in (("composition", c), ("composition_rel", rel)):
        s, k = np.zeros(n), np.zeros(n)
        np.add.at(s, tab, v)
        np.add.at(k, tab, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[name] = np.where(k > 0, s / np.maximum(k, 1), np.nan)
    return out


def calibration(cells, meta, n0, names=("W", "H1", "H2")):
    """Reliability by bits per variant: {variant: dict(bins, error, cells)} and whether the own controls are uncalibrated."""
    pn = [str(x) for x in cells["pass_names"]]
    m = kept(cells, meta, n0) & np.isin(cells["pass_id"].astype(np.int64), [pn.index(x) for x in names if x in pn])
    var = meta["variant"][cells["table"][m].astype(np.int64)]
    H, b = cells["entropy"][m].astype(np.float64), cells["bits"][m].astype(np.float64)
    which = np.digitize(H, ENTROPY_BINS[1:-1])
    out = {}
    for v, vname in enumerate(("implant", "original", "own")):
        rows, err, tot = [], 0.0, 0
        for k in range(len(ENTROPY_BINS) - 1):
            sel = (var == v) & (which == k)
            c = int(sel.sum())
            rows.append(dict(bin="%g-%g" % (ENTROPY_BINS[k], ENTROPY_BINS[k + 1]), cells=c, entropy=float(H[sel].mean()) if c else None,
                             bits=float(b[sel].mean()) if c else None))
            if c:
                err += c * abs(float(b[sel].mean()) - float(H[sel].mean()))
                tot += c
        out[vname] = dict(bins=rows, error=err / tot if tot else None, cells=tot)
    e = out["own"]["error"]
    out["uncalibrated"] = bool(e is not None and e > UNCALIBRATED)
    return out


def controls(tables, meta):
    """{best_similarity, rows_on_target, covered_share, coverage_shift, host_codon, host_codon_rel: [N]} (tables: tok, n,
    col, kind of every table [N, R, ...])."""
    from seeddrift.train import source_model as S            # the genetic code, for the classical controls only
    code = np.asarray(S.CODE)
    cls, deg = _classes()
    tok, n, col, kind = (np.asarray(tables[k]) for k in ("tok", "n", "col", "kind"))
    N, R = tok.shape[:2]
    tb = P.TARGET_BLOCK
    placed = (col >= 0) & (OFFSET_OF[None, None, :] < n[:, :, BLOCK_OF])   # (N, R, L) a codon placed in row 0's frame
    on_t = placed[:, :, T0_:T1_].sum(axis=2)
    on_c = placed[:, :, :T0_].sum(axis=2)
    best, cshift = np.zeros(N), np.full(N, np.nan)
    for t in range(N):
        own = tok[t, 0, T0_:T1_].astype(np.int64)
        tiers = meta["row_tier"][t]
        ids = {}
        for q in range(1, R):
            if kind[t, q, tb] != api.KIND_PRESENT:
                continue
            c = col[t, q, T0_:T0_ + int(n[t, q, tb])].astype(np.int64)
            k = tok[t, q, T0_:T0_ + int(n[t, q, tb])].astype(np.int64)
            m = c >= 0
            if m.sum() < MIN_COLUMNS:
                continue
            ids[q] = float((code[k[m]] == code[own[c[m]]]).mean())
        if ids:
            near = min(int(tiers[q]) for q in ids)
            inside = [v for q, v in ids.items() if tiers[q] == near]
            outside = [v for q, v in ids.items() if tiers[q] != near]
            best[t] = (max(outside) if outside else 0.0) - (max(inside) if inside else 0.0)
        present = sorted({int(x) for x in tiers[1:] if x >= 0})
        if present:
            cT = np.array([on_t[t, 1:][tiers[1:] == k].sum() for k in present], dtype=np.float64)
            cC = np.array([on_c[t, 1:][tiers[1:] == k].sum() for k in present], dtype=np.float64)
            if cT.sum() > 0 and cC.sum() > 0:
                cshift[t] = 0.0 if len(present) == 1 else 0.5 * np.abs(cT / cT.sum() - cC / cC.sum()).sum()
    on = meta["on_target"].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = np.where(on > 0, meta["covered"] / np.maximum(on * meta["target_len"], 1), 0.0)
    host, host_rel = np.full(N, np.nan), np.full(N, np.nan)
    for t in range(N):
        L = int(meta["target_len"][t])
        truth = tok[t, 0, T0_:T0_ + L].astype(np.int64)
        pos = np.arange(L)
        keep = (truth >= 1) & (truth <= 64) & (pos > 0) & ~((pos == L - 1) & bool(meta["target_ends"][t])) & (deg[cls[np.clip(truth, 0, 65)]] >= 2)
        if not keep.any():
            continue
        cnt = meta["host_counts"][t].astype(np.float64) + PSEUDO            # tokens 1..64
        tot = np.zeros(23)
        np.add.at(tot, cls[1:65], cnt)
        c = -np.log2(cnt[truth[keep] - 1] / tot[cls[truth[keep]]])
        host[t] = c.mean()
        host_rel[t] = (c / np.log2(deg[cls[truth[keep]]])).mean()
    host_aa, gc, aad = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
    for t in range(N):
        L = int(meta["target_len"][t])
        truth = tok[t, 0, T0_:T0_ + L].astype(np.int64)
        pos = np.arange(L)
        keep = (truth >= 1) & (truth <= 64) & (pos > 0) & ~((pos == L - 1) & bool(meta["target_ends"][t]))
        codons = truth[(truth >= 1) & (truth <= 64)] - 1
        if len(codons):
            bases = np.stack([codons // 16, (codons // 4) % 4, codons % 4])      # A C G T = 0 1 2 3
            gc[t] = float(((bases == 1) | (bases == 2)).mean())
        if not keep.any():
            continue
        cnt = meta["host_counts"][t].astype(np.float64)
        per_aa = np.zeros(23)
        np.add.at(per_aa, cls[1:65], cnt)
        p_aa = (per_aa[1:22] + PSEUDO) / (per_aa[1:22] + PSEUDO).sum()           # 20 amino acids and the stops
        host_aa[t] = float((-np.log2(p_aa[cls[truth[keep]] - 1])).mean())
        own = np.bincount(cls[truth[keep]] - 1, minlength=21)[:21].astype(np.float64)
        aad[t] = 0.5 * float(np.abs(own / own.sum() - per_aa[1:22] / max(per_aa[1:22].sum(), 1)).sum())
    return {"best_similarity": best, "rows_on_target": on, "covered_share": cov, "coverage_shift": cshift, "host_codon": host, "host_codon_rel": host_rel,
            "host_aa": host_aa, "x_gc": gc, "x_aa_distance": aad}


# ---------------------------------------------------------------------------------------------------------------------
class Triples:
    """The triples of a ruler set: the table of each variant, the group and units, and their triple-level properties.
    group: (N,) the group index of every table; identity: (N, R) the stored identity; donor_group: (N,) the group name of
    every table's donor; group_names: the names of the group indices."""

    def __init__(self, meta, group, identity, donor_group, group_names):
        trip, var = meta["triple"].astype(np.int64), meta["variant"].astype(np.int64)
        self.ids = np.unique(trip)
        pos = {int(s): k for k, s in enumerate(self.ids)}
        self.table = np.full((len(self.ids), 3), -1, dtype=np.int64)
        for t, (s, v) in enumerate(zip(trip, var)):
            self.table[pos[int(s)], v] = t
        if (self.table < 0).any():
            raise ValueError("a triple without its three tables")
        imp = self.table[:, IMPLANT]
        self.group = np.asarray(group)[imp].astype(np.int64)
        self.unit_a, self.unit_d = meta["unit_a"][imp].astype(np.int64), meta["unit_d"][imp].astype(np.int64)
        self.scenario, self.tier, self.rows = meta["scenario"][imp], meta["donor_tier"][imp], meta["on_target"][imp]
        self.phase, self.context = meta["phase"][imp], meta["context_homolog"][imp]
        self.group_names = list(group_names)
        self.block = np.array([SC.block_of(group_names[g]) for g in self.group])
        dg = np.asarray(donor_group)[imp]
        self.direction = np.array([direction(d, group_names[g]) for d, g in zip(dg, self.group)])
        self.donor_block_same = np.array([SC.block_of(d) for d in dg]) == self.block
        self.nearest = SC.nearest_identity(np.asarray(identity)[imp])
        ri = meta["row_tier"][imp]
        self.tiers_present = np.array([len({int(x) for x in r[1:] if x >= 0}) for r in ri])
        cov = meta["covered"].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.coverage_ratio = np.where(cov[self.table[:, OWN]] > 0, cov[imp] / np.maximum(cov[self.table[:, OWN]], 1), np.nan)
        g_imp, g_orig, g_own = (meta["row_gene"][self.table[:, v]] for v in (IMPLANT, ORIGINAL, OWN))
        exists = g_imp >= 0
        shared = exists & ((g_imp == g_orig) | (g_imp == g_own))
        n_sh, n_ex = shared.sum(axis=1), exists.sum(axis=1)
        self.shared = np.where(n_ex == 0, "no rows", np.where(n_sh == 0, "none", np.where(n_sh == n_ex, "all", "partial")))

    def strata(self):
        """{(kind, name): bool [triples]} of the triple-level strata of the protocol."""
        out = {("all", "all"): np.ones(len(self.ids), dtype=bool)}
        for t in sorted(set(self.tier.tolist())):
            out[("donor_tier", "tier=%d" % t)] = self.tier == t
        for lo, hi, name in NEAREST_BINS:
            out[("nearest_of_implant", name)] = (self.nearest >= lo) & (self.nearest < hi)
        out[("nearest_of_implant", "none")] = ~np.isfinite(self.nearest)
        for lo, hi, name in FILL_BINS:
            out[("rows", name)] = (self.rows >= lo) & (self.rows <= hi)
        for lo, hi, name in RATIO_BINS:
            out[("coverage_ratio", name)] = (self.coverage_ratio >= lo) & (self.coverage_ratio < hi)
        for name in ("none", "partial", "all"):
            out[("shared_homolog", name)] = self.shared == name
        out[("donor_block", "same")] = self.donor_block_same
        out[("donor_block", "other")] = ~self.donor_block_same
        out[("phase", "n=256")] = self.phase == 1
        out[("phase", "matched")] = self.phase == 2
        for d in sorted(set(self.direction.tolist())):
            out[("direction", d)] = self.direction == d
        return out

    def exploratory(self, gc, aa_distance):
        """{(kind, q): bool [triples]}, exploratory (outside the protocol): quartiles per block of the implant's G+C share
        (x_implant_gc), of the implant's minus the own control's (x_gc_minus_own) and of the implant's amino-acid distance
        to A (x_aa_distance)."""
        gc = np.asarray(gc, dtype=np.float64)
        values = {"x_implant_gc": gc[self.table[:, IMPLANT]], "x_gc_minus_own": gc[self.table[:, IMPLANT]] - gc[self.table[:, OWN]],
                  "x_aa_distance": np.asarray(aa_distance, dtype=np.float64)[self.table[:, IMPLANT]]}
        return {k: v for name, x in values.items() for k, v in self._quartiles(name, x).items()}

    def _quartiles(self, kind, x):
        out = {(kind, "q%d" % (q + 1)): np.zeros(len(self.ids), dtype=bool) for q in range(4)}
        for block in SC.BLOCKS:
            sel = (self.block == block) & np.isfinite(x)
            if sel.sum() < 4:
                continue
            which = np.searchsorted(np.quantile(x[sel], [0.25, 0.5, 0.75]), x, side="right")
            for q in range(4):
                out[(kind, "q%d" % (q + 1))] |= sel & (which == q)
        return out

    def host_quartiles(self, host):
        """{(host_codon_quartile, q): bool [triples]}: quartiles, per block, of the host codon model's score of the
        implant table (implant and own compared inside the quartile of the implant)."""
        h = np.asarray(host, dtype=np.float64)[self.table[:, IMPLANT]]
        out = {}
        for q in range(4):
            out[("host_codon_quartile", "q%d" % (q + 1))] = np.zeros(len(self.ids), dtype=bool)
        for block in SC.BLOCKS:
            sel = (self.block == block) & np.isfinite(h)
            if sel.sum() < 4:
                continue
            edges = np.quantile(h[sel], [0.25, 0.5, 0.75])
            which = np.searchsorted(edges, h, side="right")
            for q in range(4):
                out[("host_codon_quartile", "q%d" % (q + 1))] |= sel & (which == q)
        return out


class Bootstrap:
    """Replicate weights on split units: A's units inside each group (multinomial over the group's units) times D's units
    (multinomial over all donor units). weights(sel): (reps, triples of sel)."""

    def __init__(self, tr, reps=BOOT, seed=BOOT_SEED):
        self.reps, self.tr = int(reps), tr
        rng = np.random.default_rng([int(seed), 1])
        self.wa = np.zeros((self.reps, int(tr.unit_a.max()) + 1))
        for g in sorted(set(tr.group.tolist())):
            units = sorted(set(tr.unit_a[tr.group == g].tolist()))
            self.wa[:, units] = rng.multinomial(len(units), np.full(len(units), 1.0 / len(units)), size=self.reps)
        dun = sorted(set(tr.unit_d.tolist()))
        self.wd = np.zeros((self.reps, int(tr.unit_d.max()) + 1))
        self.wd[:, dun] = np.random.default_rng([int(seed), 2]).multinomial(len(dun), np.full(len(dun), 1.0 / len(dun)), size=self.reps)

    def weights(self, sel):
        return self.wa[:, self.tr.unit_a[sel]] * self.wd[:, self.tr.unit_d[sel]]


def _num(x):
    return float(x) if x is not None and np.isfinite(x) else None


def auc_matrix(sp, sn):
    sp, sn = np.asarray(sp, dtype=np.float64)[:, None], np.asarray(sn, dtype=np.float64)[None, :]
    return (sp > sn) + 0.5 * (sp == sn)


def auc(sp, sn):
    sp, sn = np.asarray(sp, dtype=np.float64), np.asarray(sn, dtype=np.float64)
    return float(auc_matrix(sp, sn).mean()) if len(sp) and len(sn) else np.nan


def group_stat(stat, s_imp, s_ctl, w):
    """(point, replicates [B]) of one group: stat "win" (the paired win rate over the triples) or "pooled" (the AUC of
    every implant against every control of the group); w (B, triples) the replicate weights of the triples."""
    s_imp, s_ctl = np.asarray(s_imp, dtype=np.float64), np.asarray(s_ctl, dtype=np.float64)
    if stat == "win":
        win = (s_imp > s_ctl) + 0.5 * (s_imp == s_ctl)
        den = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(win.mean()), np.where(den > 0, (w @ win) / np.where(den > 0, den, 1), np.nan)
    K = auc_matrix(s_imp, s_ctl)
    num = np.einsum("bi,ij,bj->b", w, K, w)
    den = w.sum(axis=1) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(K.mean()), np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def _agg(items):
    """(value, lo, hi) of a block from its groups' (point, replicates)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        reps = np.nanmean(np.vstack([x[1] for x in items]), axis=0)
    fin = np.isfinite(reps).any()
    return float(np.mean([x[0] for x in items])), (_num(np.nanpercentile(reps, 2.5)) if fin else None), (_num(np.nanpercentile(reps, 97.5)) if fin else None), reps


def stat_rows(scores, tr, population, strata, boot, label="", reference=None, stats=("win", "pooled"), expected_too_few=(), tag=None):
    """Result rows of the scores ({name: [N tables]}) on the triples of `population` (bool [triples]): per statistic,
    comparison, block and stratum the mean over the readable groups with its interval, counts and readability; with
    reference ({control: [N]}), vs_<control>: both values on the triples where both are defined and their paired
    difference over the same replicates. tag: fields every row carries (scenario, rule 9, ...)."""
    rows = []
    for sname, s in scores.items():
        s = np.asarray(s, dtype=np.float64)
        for stat in stats:
            for cname, cv in COMPARISONS:
                si, sc = s[tr.table[:, IMPLANT]], s[tr.table[:, cv]]
                ok = population & ~np.isnan(si) & ~np.isnan(sc)
                for (skind, stratum), mask in strata.items():
                    for block in SC.BLOCKS:
                        sel_b = ok & mask & (tr.block == block)
                        items = []
                        for g in sorted(set(tr.group[sel_b].tolist())):
                            sel = np.flatnonzero(sel_b & (tr.group == g))
                            items.append((g, sel) + group_stat(stat, si[sel], sc[sel], boot.weights(sel)))
                        enter = [x for x in items if len(x[1]) >= MIN_GROUP]
                        n_tr = sum(len(x[1]) for x in enter)
                        units = len({int(u) for x in enter for u in tr.unit_a[x[1]]})
                        row = dict(tag or {}, score=sname, label=label, statistic=stat, comparison=cname, block=block, stratum_kind=skind, stratum=stratum,
                                   groups=len(enter), groups_with_triples=len(items), triples=n_tr, units=units,
                                   readable=bool(len(enter) >= MIN_GROUPS and n_tr >= MIN_TRIPLES and units >= MIN_UNITS),
                                   expected_too_few=bool(skind == "donor_tier" and stratum in expected_too_few), value=None, lo=None, hi=None)
                        if enter:
                            row["value"], row["lo"], row["hi"], _ = _agg([(x[2], x[3]) for x in enter])
                        for rname, r in (reference or {}).items():
                            if not enter:
                                break
                            r = np.asarray(r, dtype=np.float64)
                            ri, rc = r[tr.table[:, IMPLANT]], r[tr.table[:, cv]]
                            a_items, r_items = [], []
                            for x in enter:
                                sel = x[1][~np.isnan(ri[x[1]]) & ~np.isnan(rc[x[1]])]
                                if len(sel):
                                    w = boot.weights(sel)
                                    a_items.append(group_stat(stat, si[sel], sc[sel], w))
                                    r_items.append(group_stat(stat, ri[sel], rc[sel], w))
                            if not a_items:
                                continue
                            va, _, _, ra = _agg(a_items)
                            vr, _, _, rr = _agg(r_items)
                            d = ra - rr
                            fin = np.isfinite(d).any()
                            row["vs_" + rname] = dict(value=va, reference=vr, diff=va - vr, lo=_num(np.nanpercentile(d, 2.5)) if fin else None,
                                                      hi=_num(np.nanpercentile(d, 97.5)) if fin else None)
                        rows.append(row)
    return rows


def defined_share_rows(shift, tr, population, boot, label="", tag=None):
    """The defined share of the source shift, implant minus own, per block: per group the mean over its triples of
    defined(implant) - defined(own), then the mean over the groups with MIN_GROUP triples (interval from the same
    replicates)."""
    d = np.isfinite(np.asarray(shift))
    diff = d[tr.table[:, IMPLANT]].astype(np.float64) - d[tr.table[:, OWN]].astype(np.float64)
    rows = []
    for block in SC.BLOCKS:
        pb = population & (tr.block == block)
        items = []
        for g in sorted(set(tr.group[pb].tolist())):
            sel = np.flatnonzero(pb & (tr.group == g))
            if len(sel) < MIN_GROUP:
                continue
            w = boot.weights(sel)
            den = w.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                items.append((float(diff[sel].mean()), np.where(den > 0, (w @ diff[sel]) / np.where(den > 0, den, 1), np.nan), len(sel)))
        row = dict(tag or {}, score="source_shift_defined", label=label, statistic="implant_minus_own", block=block, groups=len(items),
                   triples=sum(x[2] for x in items), implant_defined=float(d[tr.table[pb, IMPLANT]].mean()) if pb.any() else None,
                   own_defined=float(d[tr.table[pb, OWN]].mean()) if pb.any() else None, value=None, lo=None, hi=None)
        if items:
            row["value"], row["lo"], row["hi"], _ = _agg([(x[0], x[1]) for x in items])
        rows.append(row)
    return rows


def coverage_table(tr, population, min_triples=MIN_GROUP):
    """Rule 7's table, direction x recipient group: the triples of the population in every cell and its mark. A ruler
    marks a cell "construct only" when it holds at least min_triples triples, "not evaluated" otherwise; never
    "evaluated" (every cell without "evaluated" reads "unverified")."""
    out = []
    for dname in DIRECTIONS:
        for g, gname in enumerate(tr.group_names):
            n = int((population & (tr.direction == dname) & (tr.group == g)).sum())
            out.append(dict(direction=dname, recipient=gname, triples=n, mark="construct only" if n >= min_triples else "not evaluated"))
    return out
