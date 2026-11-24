"""From per-cell bits to result rows: the strata, the averaging and the row writer every arm and every baseline share
(the one implementation; the trainer's held-out evaluation writes its rows through here as well).

- bits per codon of a set of cells = the sum of -log2 p(residue) over the cells / the number of cells;
- per group first, then the plain mean of the groups that have cells (12 groups in a real table set, equal weight);
- strata of a table (column `stratum`): the identity of the nearest row at the target locus, the largest stored
  identity of the rows 1.. that cover at least MIN_COVERAGE of row 0's stored target block, in ten equal bins
  (nearest_identity=0.0-0.1 ... 0.9-1.0); nearest_identity=short_fragments_only when other rows have cells there but
  none covers that much; nearest_identity=none when no other row has a cell there; the same stratum under every
  condition. A row's identity is the share of identical codons over the codons it has aligned to row 0 (the stored
  placement), its coverage those codons over row 0's stored codons of the target block (row_coverage). RELATIVES:
  whether some other row has a cell there, of any coverage (the headline is read on the tables that have one).
  OTHER_ROWS: how many other rows the table has (0, 1, two or more); the tables with two or more are the ones the
  re-paired condition scores, so the whole gene can be read on the same tables. Reporting keys only, never given to a
  model.
- version 5 (the programme lead's ruling of 2026-11-24): the coverage rule and the stratum of short fragments. Under
  version 4 a fragment of a remote homologue of one to three codons at the end of the target block (a local alignment
  starts on a match, so its identity is often 1/1) won the largest identity over every long row and filled the top
  bin with tables that have no close relative. RELATIVES, OTHER_ROWS, RELATIVE_KINDS and the self-copy rule are
  version 4's; so are the headline and what relatives are worth over all tables.
- blocks of groups: the eukaryote groups (their genomes hold the first 1,500 genes only) and the prokaryotes,
  plasmids and viruses, read apart (block_of).
- rows (contract api.RESULT_ROW): n_items = the tables, target_cells = the scored cells; today tier is -1 and
  baseline and bits_minus_baseline stay empty (the report puts models and baselines side by side).
- parts of the bits of a codon (reporting only, the genetic code never reaches a model): the bits of its amino acid
  (stop and "other" each a class of their own) and the bits of the codon given the amino acid; they add up to the
  bits of the codon. Rows carry them under stratum PARTS[0] and PARTS[1].
"""
from __future__ import annotations

import csv

import numpy as np

from seeddrift.train.rel import api

VERSION = 5
ALL = "all"
TIER = -1                               # distance tiers come after the census
N_FINE = 10
MIN_COVERAGE = 0.5                      # a row counts for the nearest identity from half of row 0's stored target block
FINE_PREFIX = "nearest_identity="
FINE = tuple("%s%.1f-%.1f" % (FINE_PREFIX, k / 10.0, (k + 1) / 10.0) for k in range(N_FINE)) + \
    (FINE_PREFIX + "short_fragments_only", FINE_PREFIX + "none")
FINE_SHORT, FINE_NONE = N_FINE, N_FINE + 1    # the indices of those two strata in FINE
OTHER_ROWS = ("other_rows=0", "other_rows=1", "other_rows>=2")
RELATIVES = ("relatives_on_target=yes", "relatives_on_target=no")
RELATIVE_KINDS = ("relatives=other_genomes", "relatives=self_copies_only", "relatives=none")    # round 1's three strata
BLOCKS = ("eukaryotes", "prokaryotes, plasmids, viruses")


def block_of(group):
    """The block of a group name: the eukaryote groups (eukaryota.*) and all others."""
    return BLOCKS[0] if str(group).startswith("eukaryota.") else BLOCKS[1]
PARTS = ("part=amino_acid", "part=codon_given_amino_acid")
_CLASSES = []


def amino_acid_classes():
    """(66,) the class of every token for the parts of the bits: the amino acid of a codon (1..20, the standard code of
    the training arrays), stop (21), other (22); tokens outside 1..65 are 0."""
    if not _CLASSES:
        from seeddrift.train import source_model as S                       # the genetic code, for reporting only
        c = np.zeros(66, dtype=np.int64)
        c[1:65] = np.asarray(S.CODE)[1:65]
        c[api.OTHER] = 22
        _CLASSES.append(c)
    return _CLASSES[0]


def amino_acid_bits(P, truth):
    """(C,) -log2 of the probability a distribution P (C, 65) over tokens 1..65 gives to the amino acid (class) of the
    true token of every cell. The bits of the codon minus these are the bits of the codon given the amino acid."""
    cls = amino_acid_classes()[1:66]
    onehot = np.zeros((cls.shape[0], 23))
    onehot[np.arange(cls.shape[0]), cls] = 1.0
    per_class = np.asarray(P, dtype=np.float64) @ onehot
    return -np.log2(per_class[np.arange(truth.shape[0]), cls[np.asarray(truth) - 1]])


def self_copy_rows(kind):
    """(N, R) bool: the rows that are copies in the examined genome itself (KIND_SELF_COPY on some block; tables of
    version 1). Row 0 is never one."""
    k = np.asarray(kind)
    out = (k == api.KIND_SELF_COPY).any(axis=2)
    out[:, 0] = False
    return out


def row_coverage(n, col):
    """(N, R) float: the coverage of every row at the target locus, the codons of its target-block segment that have
    a column in row 0 (col >= 0: the codons its stored identity is taken over) divided by row 0's stored codons of the
    target block (n of row 0); 1 for row 0, nan where row 0 has no codon there. n (N, R, B) and col (N, R, ROW_LEN) of
    the stored tables (the frame row of the target block is row 0)."""
    n, col = np.asarray(n), np.asarray(col)
    _, s, e = api.block_slices()[-1]
    ct = col[:, :, s:e]
    aligned = ((ct >= 0) & (np.arange(e - s)[None, None, :] < n[:, :, -1, None])).sum(axis=2).astype(np.float64)
    own = n[:, 0, -1].astype(np.float64)[:, None]
    return np.where(own > 0, aligned / np.where(own > 0, own, 1.0), np.nan)


def _rows(identity, kind=None, coverage=None):
    """(identity of rows 1.., the rows with a cell at the target locus, the rows that count for the nearest identity);
    with kind the self-copy rows are left out of both, with coverage the rows below MIN_COVERAGE out of the latter."""
    ident = np.asarray(identity, dtype=np.float64)[:, 1:]
    has = np.isfinite(ident)
    if kind is not None:
        has &= ~self_copy_rows(kind)[:, 1:]
    counts = has.copy()
    if coverage is not None:
        counts &= np.nan_to_num(np.asarray(coverage, dtype=np.float64)[:, 1:], nan=0.0) >= MIN_COVERAGE - 1e-9
    return ident, has, counts


def nearest_identity(identity, kind=None, coverage=None):
    """(N,) the largest identity of rows 1.. at the target locus; nan when no row counts. With kind (N, R, B) the
    self-copy rows are left out: the nearest other-genome row (round 1's strata). With coverage (N, R; row_coverage)
    only the rows covering at least MIN_COVERAGE of row 0's stored target block count (version 5); without it every row
    with a cell does (version 4's reading, for the census of the tables and W18's implants until their protocol says
    otherwise)."""
    ident, _, counts = _rows(identity, kind, coverage)
    if ident.shape[1] == 0:
        return np.full(ident.shape[0], np.nan)
    best = np.where(counts, ident, -1.0).max(axis=1)
    return np.where(counts.any(axis=1), best, np.nan)


def nearest_coverage(identity, coverage, kind=None):
    """(N,) the coverage of the nearest row of every table (of the counting rows with the largest identity, the one
    covering most); for a table of short fragments only, the largest coverage of its rows with a cell; nan without
    other rows. The column beside the strata of the nearest identity in the reports."""
    ident, has, counts = _rows(identity, kind, coverage)
    cov = np.nan_to_num(np.asarray(coverage, dtype=np.float64)[:, 1:], nan=0.0)
    if ident.shape[1] == 0:
        return np.full(ident.shape[0], np.nan)
    near = np.where(counts, ident, -1.0).max(axis=1)
    top = counts & (np.abs(ident - near[:, None]) <= 1e-9)
    out = np.where(top, cov, -1.0).max(axis=1)
    short = np.where(has, cov, -1.0).max(axis=1)
    return np.where(counts.any(axis=1), out, np.where(has.any(axis=1), short, np.nan))


def fine_stratum(identity, kind=None, coverage=None):
    """(N,) index into FINE for every table, from the stored (N, R) identity of every row with row 0 (with kind: of the
    other-genome rows only; with coverage: of the rows covering at least MIN_COVERAGE, FINE_SHORT where rows have cells
    but none covers that much). FINE_NONE where no other row has a cell."""
    _, has, _ = _rows(identity, kind)
    near = nearest_identity(identity, kind, coverage)
    k = np.minimum(np.floor(np.nan_to_num(near, nan=0.0) * N_FINE + 1e-9).astype(np.int64), N_FINE - 1)
    return np.where(np.isnan(near), np.where(has.any(axis=1), FINE_SHORT, FINE_NONE), np.maximum(k, 0))


def coverage_by_stratum(fine, near_cov):
    """{FINE label: median coverage of the nearest row} over the tables of every stratum of the nearest identity that
    has tables (fine: fine_stratum; near_cov: nearest_coverage); None for the stratum without other rows."""
    fine, near_cov = np.asarray(fine), np.asarray(near_cov, dtype=np.float64)
    out = {}
    for k, lab in enumerate(FINE):
        m = (fine == k) & np.isfinite(near_cov)
        if (fine == k).any():
            out[lab] = float(np.median(near_cov[m])) if m.any() else None
    return out


def relative_kinds(kind):
    """(N,) index into RELATIVE_KINDS: the table has a row of another genome (a row 1.. that exists and is no self
    copy), else self-copy rows only, else no other row."""
    k = np.asarray(kind)
    exists = (k != api.KIND_NONE).any(axis=2)
    exists[:, 0] = False
    copy = self_copy_rows(k)
    return np.where((exists & ~copy).any(axis=1), 0, np.where(copy.any(axis=1), 1, 2)).astype(np.int64)


def strata_of(identity, other_rows=None, kind=None, coverage=None):
    """[(labels, index of every table)]: the bins of the nearest identity (FINE, version 5: coverage (N, R) of every
    row, row_coverage of the stored tables, is required) and whether some other row has a cell at the target locus
    (RELATIVES, of any coverage); given the number of other rows of every table (conditions.present_rows(kind)[:,
    1:].sum(axis=1)), OTHER_ROWS; given the kinds (N, R, B) of the tables, the nearest identity is that of the nearest
    other-genome row (self copies left out) and RELATIVE_KINDS is added: other-genome rows, self copies only, none
    (round 1's protocol)."""
    if coverage is None:
        raise ValueError("the strata of version 5 need the coverage of every row (score.row_coverage(n, col) of the stored tables)")
    out = [(FINE, fine_stratum(identity, kind, coverage)), (RELATIVES, np.isnan(nearest_identity(identity, kind)).astype(np.int64))]
    if other_rows is not None:
        out.append((OTHER_ROWS, np.minimum(np.asarray(other_rows).astype(np.int64), 2)))
    if kind is not None:
        out.append((RELATIVE_KINDS, relative_kinds(kind)))
    return out


def aggregate(bits, cell_table, group, strata, n_groups=None, group_names=None):
    """bits (C,) of the scored cells; cell_table (C,) the table of every cell (an index into group and the strata);
    group (T,) the group index of every table; strata: [(labels, (T,) index into labels)] (strata_of).
    Returns rows dict(group, stratum, bits, target_cells, n_items, n_groups): every group x stratum that has cells and
    the margins; group "all" = the mean over the groups that have cells in that stratum, n_items its tables with
    cells. A group is named group_names[index] when given, else the index."""
    bits = np.asarray(bits, dtype=np.float64)
    cell_table = np.asarray(cell_table).astype(np.int64)
    group = np.asarray(group).astype(np.int64)
    G = int(n_groups) if n_groups else (int(group.max()) + 1 if group.size else 0)
    has = np.zeros((group.shape[0],), dtype=bool)
    has[cell_table] = True
    name = lambda g: str(group_names[g]) if group_names is not None and g in group_names else str(g)      # noqa: E731
    rows = []

    def emit(gname, sname, tot, cnt, tab):
        """tot, cnt, tab: per group. One row for the mean of the groups that have cells."""
        ok = cnt > 0
        if ok.any():
            rows.append(dict(group=gname, stratum=sname, bits=float((tot[ok] / cnt[ok]).mean()), target_cells=int(cnt.sum()),
                             n_items=int(tab.sum()), n_groups=int(ok.sum())))

    def table(key, size):
        total = np.bincount(key[cell_table], weights=bits, minlength=size)
        return total, np.bincount(key[cell_table], minlength=size), np.bincount(key[has], minlength=size)

    tot, cnt, tab = table(group, G)
    emit(ALL, ALL, tot, cnt, tab)
    for g in range(G):
        emit(name(g), ALL, tot[g:g + 1], cnt[g:g + 1], tab[g:g + 1])
    for labels, index in strata:
        S = len(labels)
        tot, cnt, tab = (x.reshape(G, S) for x in table(group * S + np.asarray(index).astype(np.int64), G * S))
        for k in range(S):
            emit(ALL, labels[k], tot[:, k], cnt[:, k], tab[:, k])
            for g in range(G):
                emit(name(g), labels[k], tot[g:g + 1, k], cnt[g:g + 1, k], tab[g:g + 1, k])
    return rows


def result_row(**fields):
    """A row of the result table with exactly the columns of the contract; columns not given are empty."""
    unknown = sorted(set(fields) - set(api.RESULT_ROW))
    if unknown:
        raise KeyError("not columns of the result table: %s" % unknown)
    return {k: fields.get(k, "") for k in api.RESULT_ROW}


def _cell(v):
    if isinstance(v, (float, np.floating)):
        return "%.6f" % v
    return str(v)


def write_rows(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(api.RESULT_ROW)
        for r in rows:
            w.writerow([_cell(r[k]) for k in api.RESULT_ROW])


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))
