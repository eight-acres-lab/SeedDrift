"""The passes (seeddrift/train/rel/readouts/w18_passes.py) and readouts (w18_readouts.py) of package W18: what each pass
hides and scores, a run through the stored renderer with a toy model, and the readouts and AUCs on cells whose values are
known. numpy only (python tests/ci/test_rel_w18_readouts.py runs them without pytest)."""
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_rel_w18_implant as IT  # noqa: E402
from seeddrift.train.rel import api  # noqa: E402
from seeddrift.train.rel.data import tables as T  # noqa: E402
from seeddrift.train.rel.evalh import score as SC  # noqa: E402
from seeddrift.train.rel.readouts import w18_passes as P  # noqa: E402
from seeddrift.train.rel.readouts import w18_readouts as R  # noqa: E402

CLS = SC.amino_acid_classes()[1:66]


def toy_score(items):
    """A normalised toy model: 0.6 x the frequency of the other rows' visible tokens in the cell's column (uniform when
    none) + 0.4 x uniform over tokens 1..65."""
    out = []
    for grid, truth in items:
        bits, aab, ent = [], [], []
        for r, s in zip(*np.nonzero(truth)):
            col = grid[1:, s]
            vis = col[(col >= 1) & (col <= 65)].astype(np.int64)
            p = np.full(65, 0.4 / 65) + (0.6 * np.bincount(vis - 1, minlength=65) / len(vis) if len(vis) else 0.6 / 65)
            t = int(truth[r, s])
            pc = np.bincount(CLS, weights=p, minlength=23)
            bits.append(-np.log2(p[t - 1]))
            aab.append(-np.log2(pc[CLS[t - 1]]))
            ent.append(-(p * np.log2(p)).sum())
        out.append((np.array(bits), np.array(aab), np.array(ent)))
    return out


def ruler():
    out, meta = IT.built()[:2]
    return out, meta


def source(out, meta):
    for t in range(len(out["tok"])):
        yield t, {k: out[k][t] for k in P.FIELDS}, meta["row_tier"][t]


def test_passes_hide_and_score_what_the_protocol_says():
    out, meta = ruler()
    tslots = np.zeros(api.ROW_LEN, dtype=bool)
    tslots[P.T0_:P.T1_] = True
    for t, tab, tier in source(out, meta):
        ps = {p.name: p for p in P.plan(tab, tier, t, relatives=True, presence=True)}
        res = P.residue(tab)
        others = P.present(tab)
        others[0] = False
        assert {"T", "C", "W"} <= set(ps)
        for name, p in ps.items():
            assert p.keep[0] and not p.target[1:].any() and not (p.target & p.visible).any()
            if name.startswith(P.PRESENCE):
                assert not p.visible[1:].any() and (p.target == ps[name[2:]].target).all()
                continue
            assert (p.visible[1:] == res[1:]).all()                    # other rows are visible as stored
            if "-t" in name:
                k = int(name.split("-t")[1])
                assert (p.keep[1:] == (others[1:] & (tier[1:] != k))).all() and ((tier[1:] == k) & others[1:]).any()
                assert (p.target == ps[name.split("-t")[0]].target).all()
        assert (ps["W"].target[0] == (res[0] & tslots)).all()
        assert not (ps["T"].target[0] & ~tslots).any() and not (ps["C"].target[0] & tslots).any()
        if "H1" in ps:
            assert not (ps["H1"].target & ps["H2"].target).any() and ((ps["H1"].target | ps["H2"].target) == ps["W"].target).all()
        if others.any():
            assert ps["T0"].keep.sum() == 1 and (ps["T0"].target == ps["T"].target).all()
        assert [p.name for p in P.plan(tab, tier, t, relatives=False)] == ["N"]
        again = P.plan(tab, tier, t, relatives=True, presence=True)
        assert all((a.target == ps[a.name].target).all() for a in again)  # the plan is the same every time


def test_a_run_through_the_stored_renderer_scores_the_plan():
    out, meta = ruler()
    cells = P.run(source(out, meta), T.render, toy_score, relatives=True, presence=True, max_rows=20)
    names = list(cells["pass_names"])
    for t, tab, tier in source(out, meta):
        for p in P.plan(tab, tier, t, relatives=True, presence=True):
            m = (cells["table"] == t) & (cells["pass_id"] == names.index(p.name))
            assert m.sum() == p.target[0].sum()
            assert (np.sort(cells["slot"][m]) == np.flatnonzero(p.target[0])).all()
            assert (cells["truth"][m] == tab["tok"][0][cells["slot"][m]]).all()
    assert cells["digest"] == P.plan_digest(source(out, meta), True, True)[0]
    assert cells["digest"] != P.plan_digest(source(out, meta), True, False)[0]
    assert np.isfinite(cells["bits"]).all() and (cells["bits"] >= cells["aa_bits"] - 1e-9).all()
    d = tempfile.mkdtemp()
    head = P.save_cells(os.path.join(d, "c.npz"), cells, arm="toy", weights="final", relatives=True, presence=True, layout="stored",
                        render_version=T.RENDER_VERSION, tableset_sha256="0" * 64, run="r", commit="0" * 40)
    back, h2 = P.load_cells(os.path.join(d, "c.npz"))
    assert h2 == head and all(np.array_equal(back[k], cells[k]) for k in ("table", "pass_id", "slot", "bits"))
    sc = R.readouts(cells, meta)
    assert np.isfinite(sc["gain"]).sum() > 0 and set(np.unique(sc["abstain"][np.isfinite(sc["abstain"])])) <= {0.0, 1.0}


def _cells(rows):
    """Per-cell arrays from [(table, pass, slot offset in the target or a context slot, truth, bits, aa_bits)]."""
    names = list(P.PASS_NAMES)
    out = dict(table=[], pass_id=[], slot=[], truth=[], bits=[], aa_bits=[], entropy=[])
    for t, name, slot, truth, bits, aab in rows:
        for k, v in zip(("table", "pass_id", "slot", "truth", "bits", "aa_bits", "entropy"), (t, names.index(name), slot, truth, bits, aab, bits)):
            out[k].append(v)
    out = {k: np.array(v) for k, v in out.items()}
    out["pass_names"] = np.array(names)
    return out


def _meta(n=1, R_=4, tiers=(-1, 0, 1, -1), length=10, ends=True):
    return dict(triple=np.arange(n), variant=np.zeros(n, np.int8), row_tier=np.tile(np.array(tiers, np.int8), (n, 1)),
                target_len=np.full(n, length, np.int16), target_ends=np.full(n, ends), on_target=np.full(n, 2, np.int8))


def test_source_shift_gain_and_abstention_on_known_cells():
    T0 = P.T0_
    rows = []
    for k in range(4):                                                 # four scattered target cells, five context cells
        rows += [(0, "T", T0 + k, 5, 1.0, 0.5), (0, "T-t0", T0 + k, 5, 3.0, 0.5), (0, "T-t1", T0 + k, 5, 1.0, 0.5), (0, "T0", T0 + k, 5, 4.0, 0.5)]
    for k in range(5):
        rows += [(0, "C", k, 5, 1.0, 0.5), (0, "C-t0", k, 5, 1.0, 0.5), (0, "C-t1", k, 5, 3.0, 0.5)]
    for k in range(10):
        rows += [(0, "W", T0 + k, 5, 4.0, 1.0), (0, "H1" if k < 5 else "H2", T0 + k, 5, 2.0, 1.0)]
    sc = R.readouts(_cells(rows), _meta())
    assert np.isclose(sc["source_shift"][0], 1.0)                      # the target leans on tier 0, the context on tier 1
    assert np.isclose(sc["gain"][0], 1.0) and sc["abstain"][0] == 0.0     # M = 40, J = 30 in either order, over 10 codons
    flat = [r if r[1] != "T-t0" else (r[0], r[1], r[2], r[3], 1.0, r[5]) for r in rows]
    flat = [r if r[1] != "T0" else (r[0], r[1], r[2], r[3], 1.01, r[5]) for r in flat]
    sc = R.readouts(_cells(flat), _meta())
    assert np.isnan(sc["source_shift"][0]) and np.isinf(sc["source_shift_top"][0]) and sc["abstain"][0] == 1.0
    both = [r if r[1] != "C-t0" else (r[0], r[1], r[2], r[3], 3.0, r[5]) for r in rows]
    one = R.readouts(_cells(both), _meta(tiers=(-1, 0, 0, -1)))
    assert one["source_shift"][0] == 0.0                               # one tier present: no shape to compare
    assert np.isnan(R.readouts(_cells(rows), _meta(tiers=(-1, 0, 0, -1)))["source_shift"][0])   # the context leans on no tier


def test_composition_keeps_degenerate_codons_only():
    T0 = P.T0_
    met, trp, ala = 15, 59, 40                                         # ATG, TGG, GCT
    rows = [(0, "N", T0 + 0, met, 3.0, 3.0), (0, "N", T0 + 1, trp, 3.0, 3.0), (0, "N", T0 + 2, 65, 9.0, 1.0), (0, "N", T0 + 3, ala, 5.0, 3.0),
            (0, "N", T0 + 4, ala, 4.0, 3.0), (0, "N", T0 + 9, ala, 9.0, 1.0)]       # the last codon of the gene
    c = R.composition(_cells(rows), _meta())
    assert np.isclose(c["composition"][0], 1.5) and np.isclose(c["composition_rel"][0], 1.5 / 2.0)    # Ala has 4 codons
    c = R.composition(_cells(rows), _meta(ends=False))
    assert np.isclose(c["composition"][0], (2 + 1 + 8) / 3.0)


def test_host_codon_and_best_similarity_controls():
    out, meta = ruler()
    ctl = R.controls(out, meta)
    assert (ctl["rows_on_target"] == meta["on_target"]).all() and np.isfinite(ctl["host_codon"]).all()
    assert (ctl["covered_share"] > 0).all() and (ctl["covered_share"] <= 1).all()
    assert (np.abs(ctl["best_similarity"]) <= 1).all()
    t = 0
    L = int(meta["target_len"][t])
    tok = out["tok"][t, 0, P.T0_:P.T0_ + L].astype(np.int64)
    cnt = meta["host_counts"][t] + 0.5
    cls, deg = R._classes()
    keep = R.kept_codons(tok, np.arange(L), L, meta["target_ends"][t], cls, deg)
    tot = np.bincount(cls[1:65], weights=cnt, minlength=23)
    assert np.isclose(ctl["host_codon"][t], (-np.log2(cnt[tok[keep] - 1] / tot[cls[tok[keep]]])).mean())


def test_auc_bootstrap_and_block_rows():
    assert R.auc([3, 2], [1, 2]) == 0.875
    names = sorted(["eukaryota.fungi", "eukaryota.mammals", "eukaryota.protists", "bacteria"])
    n_tr = 40
    meta = dict(triple=np.repeat(np.arange(n_tr), 3), variant=np.tile(np.array([0, 1, 2], np.int8), n_tr), genus_a=np.repeat(np.arange(n_tr) % 10, 3),
                donor_tier=np.repeat(np.arange(n_tr) % 2, 3), on_target=np.full(3 * n_tr, 3), row_tier=np.zeros((3 * n_tr, 4), np.int8),
                target_len=np.full(3 * n_tr, 10), target_ends=np.ones(3 * n_tr, bool))
    group = np.repeat(np.arange(n_tr) % 4, 3)
    meta["genus_a"] = meta["genus_a"] + 100 * group
    score = np.where(meta["variant"] == 0, 1.0, 0.0)
    boot = R.Bootstrap(group, meta["genus_a"], reps=50)
    a, reps = boot.group_auc(0, score[(group == 0) & (meta["variant"] == 0)], meta["genus_a"][(group == 0) & (meta["variant"] == 0)],
                             score[(group == 0) & (meta["variant"] == 2)], meta["genus_a"][(group == 0) & (meta["variant"] == 2)])
    assert a == 1.0 and np.allclose(reps, 1.0)
    boot.weights[0] = np.ones_like(boot.weights[0])
    noisy = score + np.random.default_rng(1).normal(0, 1, len(score))
    sp, sn = (group == 0) & (meta["variant"] == 0), (group == 0) & (meta["variant"] == 2)
    a, reps = boot.group_auc(0, noisy[sp], meta["genus_a"][sp], noisy[sn], meta["genus_a"][sn])
    assert np.allclose(reps, a) and np.isclose(a, R.auc(noisy[sp], noisy[sn]))   # unit weights give the point estimate
    ident = np.full((3 * n_tr, 4), np.nan)
    strata = R.stratum_masks(meta, ident, np.ones(3 * n_tr, bool))
    rows = R.auc_rows(dict(s=score), meta, group, names, {k: v for k, v in strata.items() if k[0] in ("all", "donor_tier")}, R.Bootstrap(group, meta["genus_a"], reps=50),
                      reference=dict(zero=np.zeros(len(score))))
    euk = [r for r in rows if r["block"] == "eukaryotes" and r["stratum"] == "all" and r["comparison"] == "own"][0]
    assert euk["auc"] == 1.0 and euk["groups"] == 3 and euk["readable"] and np.isclose(euk["vs_zero"]["diff"], 0.5)
    pro = [r for r in rows if r["block"] != "eukaryotes" and r["stratum"] == "all"][0]
    assert pro["groups"] == 1 and not pro["readable"]                  # one group of the block: too few to read
    tier = [r for r in rows if r["block"] == "eukaryotes" and r["stratum"] == "tier=0" and r["comparison"] == "own"][0]
    assert tier["implants"] == 10 and tier["groups"] == 1 and not tier["readable"]       # one eukaryote group holds tier 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
