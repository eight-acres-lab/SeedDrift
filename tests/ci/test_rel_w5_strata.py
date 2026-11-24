"""The evaluator's strata of score version 5 (seeddrift/train/rel/evalh/score.py, package W5; the programme lead's
ruling of 2026-11-24): a row counts for the nearest identity only when it covers at least half of row 0's stored target
block; tables whose other rows are all shorter fall in a stratum of their own, not in the one without relatives; the
relatives strata, the self-copy rule and the headline are version 4's. Numpy only."""
import numpy as np
import pytest

from seeddrift.train.rel import api as A
from seeddrift.train.rel.evalh import score as SC

_, S, E = A.block_slices()[-1]                     # the target block
L = E - S


def _tables(rows):
    """n (N, R, B), col (N, R, ROW_LEN), identity (N, R), kind (N, R, B) from, per table, row 0's stored target length
    and the other rows as (target length, the columns of row 0 their codons are aligned to (-1: none), identity, kind)."""
    R = 1 + max(len(r[1]) for r in rows)
    N = len(rows)
    n = np.zeros((N, R, A.N_BLOCKS), np.int16)
    col = np.full((N, R, A.ROW_LEN), -1, np.int16)
    identity = np.full((N, R), np.nan)
    kind = np.zeros((N, R, A.N_BLOCKS), np.uint8)
    for i, (own, others) in enumerate(rows):
        n[i, 0, -1] = own
        col[i, 0, S:S + own] = np.arange(own)
        identity[i, 0] = 1.0
        kind[i, 0] = A.KIND_PRESENT
        for r, (m, cols, ident, k) in enumerate(others, start=1):
            n[i, r, -1] = m
            col[i, r, S:S + m] = cols
            col[i, r, S + m:E] = 7                    # past the segment: ignored, whatever it holds
            identity[i, r] = ident
            kind[i, r] = k
    return n, col, identity, kind


P = A.KIND_PRESENT
ROWS = [
    (L, [(200, np.arange(200), 0.3, P), (1, [L - 1], 1.0, P)]),                    # 0: a long row and a one-codon fragment at the block's end
    (L, [(3, [L - 3, L - 2, L - 1], 1.0, P), (2, [0, 1], 0.5, P)]),                 # 1: short fragments only
    (L, []),                                                                        # 2: no other row
    (100, [(60, np.where(np.arange(60) % 6 == 0, -1, np.arange(60)), 0.8, P)]),      # 3: 50 of row 0's 100 codons aligned, exactly half
    (L, [(L, np.arange(L), 0.99, A.KIND_SELF_COPY), (1, [L - 1], 1.0, P), (180, np.arange(180), 0.45, P)]),   # 4: a self copy beside
    (100, [(49, np.arange(49), 0.9, P)]),                                           # 5: 49 of 100: below half
]


def test_a_tail_fragment_is_never_the_nearest_row():
    """A fragment of one codon at the end of the target block, identical to row 0 (identity 1/1), is not the nearest
    row: the long row is, and the table falls in the bin of its identity. Under version 4's reading (no coverage) the
    fragment won the top bin."""
    n, col, identity, kind = _tables(ROWS)
    cov = SC.row_coverage(n, col)
    assert cov[0, 1] == pytest.approx(200 / L) and cov[0, 2] == pytest.approx(1 / L) and cov[0, 0] == 1.0
    assert SC.nearest_identity(identity)[0] == 1.0                                   # version 4: the fragment wins
    assert SC.nearest_identity(identity, kind, cov)[0] == pytest.approx(0.3)
    fine = SC.fine_stratum(identity, kind, cov)
    assert SC.FINE[fine[0]] == "nearest_identity=0.3-0.4" and SC.FINE[SC.fine_stratum(identity)[0]] == "nearest_identity=0.9-1.0"
    assert SC.nearest_coverage(identity, cov, kind)[0] == pytest.approx(200 / L)
    one = _tables([(L, [(1, [L - 1], 1.0, P)])])                                     # the fragment alone: short fragments only
    assert SC.fine_stratum(one[2], one[3], SC.row_coverage(one[0], one[1])).tolist() == [SC.FINE_SHORT]


def test_short_fragments_only_and_none_are_two_strata():
    n, col, identity, kind = _tables(ROWS)
    cov = SC.row_coverage(n, col)
    fine = SC.fine_stratum(identity, kind, cov)
    assert [SC.FINE[k] for k in fine] == ["nearest_identity=0.3-0.4", "nearest_identity=short_fragments_only", "nearest_identity=none",
                                          "nearest_identity=0.8-0.9", "nearest_identity=0.4-0.5", "nearest_identity=short_fragments_only"]
    assert SC.FINE[-1] == SC.FINE[SC.FINE_NONE] == "nearest_identity=none" and len(SC.FINE) == SC.N_FINE + 2
    assert cov[3, 1] == 0.5 and cov[5, 1] == pytest.approx(0.49)                     # half counts; the unaligned codons do not
    strata = {labels[0]: [labels[k] for k in idx] for labels, idx in SC.strata_of(identity, [2, 2, 0, 1, 3, 1], kind, coverage=cov)}
    assert strata[SC.RELATIVES[0]] == ["relatives_on_target=yes", "relatives_on_target=yes", "relatives_on_target=no",     # relatives of any
                                       "relatives_on_target=yes", "relatives_on_target=yes", "relatives_on_target=yes"]   # coverage: version 4's
    near = SC.nearest_coverage(identity, cov, kind)
    assert near[1] == pytest.approx(3 / L) and np.isnan(near[2]) and near[4] == pytest.approx(180 / L)                   # the self copy left out
    med = SC.coverage_by_stratum(fine, near)
    assert med["nearest_identity=short_fragments_only"] == pytest.approx(np.median([3 / L, 0.49])) and med["nearest_identity=none"] is None
    assert "nearest_identity=0.0-0.1" not in med


def test_the_strata_need_the_coverage():
    n, col, identity, kind = _tables(ROWS)
    with pytest.raises(ValueError):
        SC.strata_of(identity, None, kind)
    assert SC.VERSION == 5 and SC.MIN_COVERAGE == 0.5


def test_ties_take_the_row_that_covers_most():
    rows = [(L, [(40, np.arange(40), 0.6, P), (L, np.arange(L), 0.6, P), (130, np.arange(130), 0.6, P)])]
    n, col, identity, kind = _tables(rows)
    cov = SC.row_coverage(n, col)
    assert SC.nearest_coverage(identity, cov, kind)[0] == 1.0 and SC.nearest_identity(identity, kind, cov)[0] == pytest.approx(0.6)
