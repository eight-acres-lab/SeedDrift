"""Census of the tables of version 0 (relational line, package W2): what the stored data gives a table, counted while
the tables are built. Numpy only. Nothing here is read by a model: the counts are for sampling decisions and reports.

Per group of the examined genome (and summed over the groups, which hold equally many tables):
  relatives   tables by their number of other rows
  candidates  tables by the number of allowed genomes that have a homolog row starting inside the target block
  cells       cells of the other rows by kind and by the role of the block (only rows and blocks that exist)
  nearest     blocks by the share of identical codons of their nearest other row (the highest share of a block)
  identity    cells with codons by their share of identical codons
  random      the share of cells that would hold codons if the other rows were drawn among the allowed genomes without
              looking at which of them have a homolog row (the expectation, counted from stage pairs: per block the
              allowed genomes with a homolog row starting inside it, over all allowed genomes but the examined one)
  codons      codons stored in the examined row and in the other rows; codons of the other rows that have a column
              and, of those, the ones identical to the examined row's codon
Identity is the contract's: identical codons over the codons of a row that have a column in the examined row.
"""
from __future__ import annotations

import numpy as np

ROLES = ("relation", "left", "right", "target")
KINDS = ("none", "present", "absent", "not_searched", "truncated", "self_copy")        # index = KIND_* of the contract
# identity bins: [0, 0.05), ..., [0.90, 0.95), [0.95, 0.99), [0.99, 1), exactly 1
EDGES = np.array([i / 20 for i in range(20)] + [0.99, 1.0])
BIN_LABELS = ["%.2f-%.2f" % (EDGES[i], EDGES[i + 1]) for i in range(len(EDGES) - 1)] + ["1.00"]
N_BINS = len(EDGES)
MAX_CAND = 65                                                     # candidates are counted 0..64, the last bin is "more"
MISC = ("cut_cells", "cut_codons", "checked", "agree", "rows_without_codons", "duplicate_rows")


def bin_of(x):
    """Bin index of identity shares (array)."""
    return np.clip(np.searchsorted(EDGES, np.asarray(x, dtype=np.float64), side="right") - 1, 0, N_BINS - 1)


class Census:
    def __init__(self, n_groups, rows):
        g, self.rows = int(n_groups), int(rows)
        self.a = dict(tables=np.zeros(g, np.int64), relatives=np.zeros((g, self.rows), np.int64), candidates=np.zeros((g, MAX_CAND + 1), np.int64),
                      cells=np.zeros((g, 4, len(KINDS)), np.int64), blocks=np.zeros((g, 4), np.int64), nearest=np.zeros((g, 4, N_BINS + 1), np.int64),
                      identity=np.zeros((g, 4, N_BINS), np.int64), random=np.zeros((g, 4), np.float64), codons=np.zeros((g, 4, 2), np.int64),
                      aligned=np.zeros((g, 4, 2), np.int64), misc=np.zeros((g, len(MISC)), np.int64))

    def add(self, group, role_of_block, kinds, ident, n, same, with_col, random_share, candidates, misc):
        """One table. role_of_block [B]; kinds, n, same, with_col [R, B] (row 0 first); ident [R, B] (nan = no codons);
        random_share [B] (nan = no such block); candidates: allowed genomes with a row on the target; misc: {name: count}."""
        a, g = self.a, int(group)
        exists = kinds[0] > 0                                       # blocks of the examined row
        rows = kinds[1:].max(1) > 0 if kinds.shape[0] > 1 else np.zeros(0, bool)
        a["tables"][g] += 1
        a["relatives"][g, int(rows.sum())] += 1
        a["candidates"][g, min(int(candidates), MAX_CAND)] += 1
        for r in range(4):
            m = exists & (role_of_block == r)
            if not m.any():
                continue
            a["blocks"][g, r] += int(m.sum())
            a["codons"][g, r, 0] += int(n[0, m].sum())
            a["random"][g, r] += float(np.nansum(random_share[m]))
            if rows.any():
                k = kinds[1:][rows][:, m]
                a["cells"][g, r] += np.bincount(k.ravel(), minlength=len(KINDS))[:len(KINDS)]
                a["codons"][g, r, 1] += int(n[1:][rows][:, m].sum())
                a["aligned"][g, r, 0] += int(with_col[1:][rows][:, m].sum())
                a["aligned"][g, r, 1] += int(same[1:][rows][:, m].sum())
                idn = ident[1:][rows][:, m]
                ok = ~np.isnan(idn)
                if ok.any():
                    a["identity"][g, r] += np.bincount(bin_of(idn[ok]), minlength=N_BINS)
                best = np.where(ok.any(0), np.where(ok, idn, -1.0).max(0), np.nan)
            else:
                best = np.full(int(m.sum()), np.nan)
            has = ~np.isnan(best)
            if has.any():
                a["nearest"][g, r, :N_BINS] += np.bincount(bin_of(best[has]), minlength=N_BINS)
            a["nearest"][g, r, N_BINS] += int((~has).sum())
        for name, v in misc.items():
            a["misc"][g, MISC.index(name)] += int(v)

    def merge(self, other):
        for k in self.a:
            self.a[k] += other.a[k]
        return self

    def to_arrays(self):
        return {k: v.copy() for k, v in self.a.items()}

    @classmethod
    def from_arrays(cls, arrays, rows):
        c = cls(len(arrays["tables"]), rows)
        c.a = {k: np.array(v) for k, v in arrays.items()}
        return c


def _share(num, den):
    return float(num) / float(den) if den else None


def _quantiles(hist, qs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Quantiles of an identity histogram, as the lower edge of the bin the quantile falls in."""
    total = int(hist.sum())
    if not total:
        return {("q%02d" % round(100 * q)): None for q in qs}
    cum = np.cumsum(hist)
    return {("q%02d" % round(100 * q)): float(EDGES[int(np.searchsorted(cum, q * total, side="left"))]) for q in qs}


def _nearest(hist):
    """Summary of a nearest-row histogram (N_BINS identity bins and the count of blocks without any row with codons)."""
    with_row, none = hist[:N_BINS], int(hist[N_BINS])
    n = int(with_row.sum())
    ge = lambda x: int(with_row[int(np.searchsorted(EDGES, x)):].sum())          # x is an edge
    return dict(blocks=n + none, without_any_row=none, share_without_any_row=_share(none, n + none), quantiles=_quantiles(with_row),
                share_below_0_50=_share(n - ge(0.5), n), share_from_0_50=_share(ge(0.5), n), share_from_0_90=_share(ge(0.9), n),
                share_from_0_95=_share(ge(0.95), n), share_from_0_99=_share(ge(0.99), n), share_identical=_share(int(with_row[-1]), n),
                histogram={BIN_LABELS[i]: int(with_row[i]) for i in range(N_BINS)})


def _one(a, g, rows):
    """The summary of one group (g: an index) or of all groups (g: None)."""
    pick = (lambda v: v.sum(0)) if g is None else (lambda v: v[g])
    tables = int(pick(a["tables"]))
    rel = pick(a["relatives"])
    cand = pick(a["candidates"])
    cells, blocks, random, codons, aligned = pick(a["cells"]), pick(a["blocks"]), pick(a["random"]), pick(a["codons"]), pick(a["aligned"])
    present = 1                                                     # KIND_PRESENT
    out = dict(tables=tables, other_rows=dict(mean=_share(int((rel * np.arange(rows)).sum()), tables), share_of_tables_without=_share(int(rel[0]), tables),
                                              share_of_tables_full=_share(int(rel[rows - 1]), tables), histogram=[int(x) for x in rel]),
               candidates_on_target=dict(mean=_share(int((cand * np.arange(len(cand))).sum()), tables), share_of_tables_without=_share(int(cand[0]), tables),
                                         share_with_fewer_than_rows=_share(int(cand[:rows - 1].sum()), tables)))
    fill, kinds, rnd, near, cod = {}, {}, {}, {}, {}
    other = [0, 1, 2]                                               # every role but the target
    views = [(ROLES[r], [r]) for r in range(4)] + [("not_target", other), ("all", [0, 1, 2, 3])]
    for name, rs in views:
        c = cells[rs].sum(0)
        # cells of rows that exist; and against the full number of row slots (rows - 1 per table), where a missing row is empty
        fill[name] = dict(cells=int(c.sum()), with_codons=int(c[present]), share=_share(int(c[present]), int(c.sum())),
                          share_of_row_slots=_share(int(c[present]), int(blocks[rs].sum()) * (rows - 1)))
        kinds[name] = {KINDS[k]: int(c[k]) for k in range(1, len(KINDS)) if c[k]}
        rnd[name] = dict(blocks=int(blocks[rs].sum()), share=_share(float(random[rs].sum()), int(blocks[rs].sum())))
        near[name] = _nearest(pick(a["nearest"])[rs].sum(0))
        al = aligned[rs].sum(0)
        cod[name] = dict(examined_row=int(codons[rs, 0].sum()), other_rows=int(codons[rs, 1].sum()), other_rows_with_column=int(al[0]),
                         identical_to_examined=int(al[1]), share_identical=_share(int(al[1]), int(al[0])))
    misc = {MISC[i]: int(pick(a["misc"])[i]) for i in range(len(MISC))}
    out.update(filled=fill, kinds=kinds, random_rows_filled=rnd, nearest_row_identity=near, codons=cod, checks=dict(
        cells_cut_at_block_length=misc["cut_cells"], aligned_codons_lost_to_the_cut=misc["cut_codons"],
        codons_coding_their_aligned_residue=_share(misc["agree"], misc["checked"]), aligned_codons_checked=misc["checked"],
        candidate_rows_without_stored_codons=misc["rows_without_codons"], duplicate_pair_rows=misc["duplicate_rows"]))
    return out


def summarise(census, groups):
    """{"all": ..., "per_group": {group: ...}} of one table set."""
    a = census.a
    return dict(all=_one(a, None, census.rows), per_group={name: _one(a, g, census.rows) for g, name in enumerate(groups) if a["tables"][g]})


def corpus_counts(stats, groups):
    """Codons per group and split from the per-genome counts ({id: dict(group, split, genes, codons, long_genes)})."""
    out = {}
    for s in stats.values():
        d = out.setdefault(s["split"], {}).setdefault(s["group"], dict(genomes=0, genes=0, codons=0, genes_longer_than_target_block=0))
        d["genomes"] += 1
        d["genes"] += int(s["genes"])
        d["codons"] += int(s["codons"])
        d["genes_longer_than_target_block"] += int(s["long_genes"])
    for split, by in out.items():
        by["all"] = {k: sum(by[g][k] for g in groups if g in by) for k in ("genomes", "genes", "codons", "genes_longer_than_target_block")}
    return out


def _pct(x):
    return "n/a" if x is None else "%.1f%%" % (100 * x)


def page(doc):
    """One page of text from the census document (the JSON written beside it)."""
    L = ["# Census of the tables, version 0", "", doc["note"], "",
         "Source run: `%s`. Coverage: %s" % (doc["source_run"], doc["coverage"]), "",
         "## 1. Codons per group (stored genes; a gene is cut at %d codons in the source arrays)" % doc["max_codons"], "",
         "| group | train genomes | train codons | held-out genomes | held-out codons |", "|---|---|---|---|---|"]
    tr, ho = doc["corpus"].get("train", {}), doc["corpus"].get("heldout", {})
    for g in doc["groups"] + ["all"]:
        a, b = tr.get(g, {}), ho.get(g, {})
        L.append("| %s | %s | %s | %s | %s |" % (g, "{:,}".format(a.get("genomes", 0)), "{:,}".format(a.get("codons", 0)), "{:,}".format(b.get("genomes", 0)),
                                               "{:,}".format(b.get("codons", 0))))
    for name in ("train", "heldout"):
        if name not in doc["sets"]:
            continue
        s = doc["sets"][name]
        L += ["", "## Table set `%s`: %s tables, %d rows stored" % (name, "{:,}".format(s["tables"]), doc["rows"]), "",
              "| group | tables | other rows (mean of %d) | tables without other rows | cells with codons, blocks other than the target | "
              "the same against all row slots | if rows were drawn at random | nearest row on the target: median identity | from 0.90 | from 0.99 |" % (doc["rows"] - 1),
              "|---|---|---|---|---|---|---|---|---|---|"]
        for g in doc["groups"] + ["all"]:
            x = s["census"]["all"] if g == "all" else s["census"]["per_group"].get(g)
            if not x:
                L.append("| %s | 0 | | | | | | | | |" % g)
                continue
            nr = x["nearest_row_identity"]["target"]
            med = nr["quantiles"]["q50"]
            L.append("| %s | %s | %.2f | %s | %s | %s | %s | %s | %s | %s |" % (
                g, "{:,}".format(x["tables"]), x["other_rows"]["mean"] or 0.0, _pct(x["other_rows"]["share_of_tables_without"]),
                _pct(x["filled"]["not_target"]["share"]), _pct(x["filled"]["not_target"]["share_of_row_slots"]),
                _pct(x["random_rows_filled"]["not_target"]["share"]), "n/a" if med is None else "%.2f" % med, _pct(nr["share_from_0_90"]),
                _pct(nr["share_from_0_99"])))
        a = s["census"]["all"]
        L += ["", "All groups together: other rows hold codons in %s of their cells on the relation and neighbour blocks (%s on the relation blocks, %s on "
              "the left and %s on the right neighbour); against all %d row slots of a table the share is %s. Rows drawn at random among the training genomes, "
              "without looking at which of them have a homolog row, would hold codons in %s of those cells and in %s of the target cells." % (
                  _pct(a["filled"]["not_target"]["share"]), _pct(a["filled"]["relation"]["share"]), _pct(a["filled"]["left"]["share"]),
                  _pct(a["filled"]["right"]["share"]), doc["rows"] - 1, _pct(a["filled"]["not_target"]["share_of_row_slots"]),
                  _pct(a["random_rows_filled"]["not_target"]["share"]), _pct(a["random_rows_filled"]["target"]["share"])),
              "", "Nearest other row per block (share of identical codons; quantiles are lower bin edges): target q10 / q50 / q90 = %s / %s / %s, %s of the "
              "target blocks have no other row at all; relation blocks q10 / q50 / q90 = %s / %s / %s, %s without any row with codons." % (
                  a["nearest_row_identity"]["target"]["quantiles"]["q10"], a["nearest_row_identity"]["target"]["quantiles"]["q50"],
                  a["nearest_row_identity"]["target"]["quantiles"]["q90"], _pct(a["nearest_row_identity"]["target"]["share_without_any_row"]),
                  a["nearest_row_identity"]["relation"]["quantiles"]["q10"], a["nearest_row_identity"]["relation"]["quantiles"]["q50"],
                  a["nearest_row_identity"]["relation"]["quantiles"]["q90"], _pct(a["nearest_row_identity"]["relation"]["share_without_any_row"])),
              "", "Checks: %s of %s aligned codons code the residue the alignment shows; %s cells were cut at the block length (%s aligned codons lost); "
              "%s candidate rows had no stored codon under the target block; %s duplicate homolog rows." % (
                  _pct(a["checks"]["codons_coding_their_aligned_residue"]), "{:,}".format(a["checks"]["aligned_codons_checked"]),
                  "{:,}".format(a["checks"]["cells_cut_at_block_length"]), "{:,}".format(a["checks"]["aligned_codons_lost_to_the_cut"]),
                  "{:,}".format(a["checks"]["candidate_rows_without_stored_codons"]), "{:,}".format(a["checks"]["duplicate_pair_rows"]))]
    L += ["", "## What these numbers are not", ""] + ["- " + x for x in doc["caveats"]] + [""]
    return "\n".join(L)
