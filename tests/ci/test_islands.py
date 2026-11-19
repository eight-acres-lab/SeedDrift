"""Helpers of the island stage of pipeline selfsup (ci/stages/islands.py): sealed or non-development chromosomes are
refused, gene names follow the benchmark's, the comparators point the right way, and the contrasts over units."""
import os

import numpy as np

from ci.stages import islands as I
from seeddrift.train import source_model as S

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _St:
    def repo(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(ROOT, rel)


def test_development_chromosomes_accepted():
    accs = I.accessions(_St(), {})
    assert len(accs) == 24 and all(a.startswith("NC_") for _, a in accs)


def test_sealed_or_unlisted_chromosome_fails(tmp_path):
    sealed = open(os.path.join(ROOT, "bench", "hard", "results", "sealed.tsv")).read().split("\n")[1].split("\t")[0]
    for acc in (sealed, "NC_999999.1"):
        p = tmp_path / ("g_%s.tsv" % acc)
        p.write_text("genus\taccession\nX\t%s\n" % acc)
        try:
            I.accessions(_St(), {"genomes": str(p)})
        except SystemExit as err:
            assert acc in str(err)
            continue
        raise AssertionError("chromosome %s was accepted" % acc)


def test_clean_names_match_the_pipeline():
    seqs = {"a b": "MKV" * 20, "a_b": "MKV" * 20, "short": "MKV", "x|y": "MK*V" * 10}
    out, ren = I.clean_names(seqs)
    assert ren == {"a b": "a_b", "a_b": "a_b_", "x|y": "x_y"} and out["x_y"] == "MKV" * 10
    try:
        from seeddrift.pipeline import clean_proteins
    except Exception:                                                   # noqa: BLE001 - optional cross-check
        return
    assert clean_proteins(seqs) == (out, ren)


def _genome(tmp_path):
    """Ten genes with the same codon usage and one (gene 3) with another, fewer hits and shorter."""
    rng = np.random.default_rng(0)
    base, other = rng.dirichlet(np.ones(64)), rng.dirichlet(np.ones(64))
    cod, coff, hoff = [], [0], [0]
    for i in range(10):
        L = 60 if i == 3 else 200
        c = rng.choice(64, size=L, p=other if i == 3 else base) + 1
        cod.append(c.astype(np.uint8))
        coff.append(coff[-1] + L)
        hoff.append(hoff[-1] + (0 if i == 3 else 3))
    n_h = hoff[-1]
    p = tmp_path / "g.npz"
    np.savez(p, codons=np.concatenate(cod), codon_off=np.array(coff), hit_off=np.array(hoff), hit_prot=np.zeros(n_h, np.int32),
             hit_qs=np.zeros(n_h, np.int32), track=np.ones(n_h, np.uint8), track_off=np.arange(n_h + 1), replicon=np.zeros(10, np.int32),
             start=np.arange(10) * 1000, end=np.arange(10) * 1000 + 600, strand=np.ones(10, np.int8), replicon_len=np.array([10000]))
    return S.Genome(dict(id="g", taxid=1, ranks=[0] * 6, group="bacteria"), str(p))


def test_comparators_point_to_the_odd_gene(tmp_path):
    c = I.comparators(_genome(tmp_path))
    for k in ("cmp_codon", "cmp_hits", "cmp_short"):
        assert int(np.argmax(c[k])) == 3, k
    assert all(np.isfinite(c[k]).all() for k in I.COMPARATORS)


def test_contrasts_over_units():
    per = {"a1": dict(_unit="U1", m=dict(r1=0.5, auroc=0.8), c=dict(r1=0.2, auroc=0.7)),
           "a2": dict(_unit="U1", m=dict(r1=0.3, auroc=0.8), c=dict(r1=0.2, auroc=0.7)),
           "b1": dict(_unit="U2", m=dict(r1=0.4, auroc=0.9), c=dict(r1=0.1, auroc=0.6))}
    rows = {(r["metric"], r["a"], r["b"]): r for r in I.contrasts(per, ["U1", "U2"], ["m"], ["c"])}
    r = rows[("r1", "m", "c")]
    assert r["n_units"] == 2 and np.allclose(r["per_unit"], [0.2, 0.3]) and np.isclose(r["mean"], 0.25)
    assert r["p_signflip"] == 0.5                                        # 2 of the 4 sign patterns are as extreme
