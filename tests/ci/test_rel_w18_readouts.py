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
    b = IT.built()
    return b["out"], b["meta"]


def source(out, meta):
    for t in range(len(out["tok"])):
        yield t, {k: out[k][t] for k in P.FIELDS}, meta["row_tier"][t], int(meta["parent"][t])


def test_passes_hide_and_score_what_the_protocol_says():
    out, meta = ruler()
    tslots = np.zeros(api.ROW_LEN, dtype=bool)
    tslots[P.T0_:P.T1_] = True
    for t, tab, tier, mid in source(out, meta):
        ps = {p.name: p for p in P.plan(tab, tier, t, relatives=True, presence=True, mask_id=mid)}
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
            for a, b in (("W0", "W"), ("H10", "H1"), ("H20", "H2")):
                assert ps[a].keep.sum() == 1 and (ps[a].target == ps[b].target).all() and P.PRESENCE + a not in ps
        assert [p.name for p in P.plan(tab, tier, t, relatives=False)] == ["N"]
        again = P.plan(tab, tier, t, relatives=True, presence=True, mask_id=mid)
        assert all((a.target == ps[a.name].target).all() for a in again)  # the plan is the same every time


def test_a_run_through_the_stored_renderer_scores_the_plan():
    out, meta = ruler()
    cells = P.run(source(out, meta), T.render, toy_score, relatives=True, presence=True, max_rows=20)
    names = list(cells["pass_names"])
    for t, tab, tier, mid in source(out, meta):
        for p in P.plan(tab, tier, t, relatives=True, presence=True, mask_id=mid):
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
    sc = R.readouts(cells, meta, out["n"][:, 0, :])
    assert np.isfinite(sc["gain"]).sum() > 0 and np.isfinite(sc["gain_rel_aa"]).sum() > 0
    assert set(np.unique(sc["abstain"][np.isfinite(sc["abstain"])])) <= {0.0, 1.0}
    first = np.isin(cells["slot"], [s for _, s, _ in api.block_slices()])
    assert first.any() and not R.kept(cells, meta, out["n"][:, 0, :])[first].any()      # a block's first codon never counts


def _cells(rows):
    """Per-cell arrays from [(table, pass, slot, truth, bits, aa_bits)]."""
    names = list(P.PASS_NAMES)
    out = dict(table=[], pass_id=[], slot=[], truth=[], bits=[], aa_bits=[], entropy=[])
    for t, name, slot, truth, bits, aab in rows:
        for k, v in zip(("table", "pass_id", "slot", "truth", "bits", "aa_bits", "entropy"), (t, names.index(name), slot, truth, bits, aab, bits)):
            out[k].append(v)
    out = {k: np.array(v) for k, v in out.items()}
    out["pass_names"] = np.array(names)
    return out


def _meta(n=1, tiers=(-1, 0, 1, -1), length=12, ends=True):
    be = np.zeros((n, api.N_BLOCKS), bool)
    be[:, -1] = ends
    n0 = np.full((n, api.N_BLOCKS), 32)
    n0[:, -1] = length
    return dict(triple=np.arange(n), variant=np.zeros(n, np.int8), row_tier=np.tile(np.array(tiers, np.int8), (n, 1)),
                target_len=np.full(n, length, np.int16), target_ends=np.full(n, ends), block_ends=be, on_target=np.full(n, 2, np.int8)), n0


def test_source_shift_gain_and_abstention_on_known_cells():
    T0 = P.T0_
    rows = []
    for k in range(1, 5):                                              # four scattered target cells, five context cells (first codons skipped)
        rows += [(0, "T", T0 + k, 5, 1.0, 0.5), (0, "T-t0", T0 + k, 5, 3.0, 0.5), (0, "T-t1", T0 + k, 5, 1.0, 0.5), (0, "T0", T0 + k, 5, 4.0, 0.5)]
    for k in range(1, 6):
        rows += [(0, "C", k, 5, 1.0, 0.5), (0, "C-t0", k, 5, 1.0, 0.5), (0, "C-t1", k, 5, 3.0, 0.5)]
    for k in range(1, 11):                                             # target offsets 1..10 of 12 (0 and 11 never count)
        h = "H1" if k < 6 else "H2"
        rows += [(0, "W", T0 + k, 5, 4.0, 1.0), (0, h, T0 + k, 5, 2.0, 1.0), (0, "W0", T0 + k, 5, 5.0, 2.0), (0, h + "0", T0 + k, 5, 4.5, 2.0)]
    rows += [(0, "W", T0 + 11, 5, 40.0, 1.0), (0, "H2", T0 + 11, 5, 0.0, 1.0), (0, "W", T0, 65, 40.0, 1.0), (0, "H1", T0, 65, 0.0, 1.0)]
    meta, n0 = _meta()
    sc = R.readouts(_cells(rows), meta, n0)
    assert np.isclose(sc["source_shift"][0], 1.0)                      # the target leans on tier 0, the context on tier 1
    assert np.isclose(sc["gain"][0], 1.0) and sc["abstain"][0] == 0.0  # M = 40, J = 30 in either order, over 10 codons
    assert np.isclose(sc["gain_aa"][0], 0.0) and np.isclose(sc["gain_norel"][0], 0.25) and np.isclose(sc["gain_rel"][0], 0.75)
    assert np.isclose(sc["source_shift_aa"][0], 0.0) or np.isnan(sc["source_shift_aa"][0])   # the amino-acid bits do not move
    flat = [r if r[1] != "T-t0" else (r[0], r[1], r[2], r[3], 1.0, r[5]) for r in rows]
    flat = [r if r[1] != "T0" else (r[0], r[1], r[2], r[3], 1.01, r[5]) for r in flat]
    sc = R.readouts(_cells(flat), meta, n0)
    assert np.isnan(sc["source_shift"][0]) and np.isinf(sc["source_shift_top"][0]) and sc["abstain"][0] == 1.0
    both = [r if r[1] != "C-t0" else (r[0], r[1], r[2], r[3], 3.0, r[5]) for r in rows]
    m1, n1 = _meta(tiers=(-1, 0, 0, -1))
    assert R.readouts(_cells(both), m1, n1)["source_shift"][0] == 0.0  # one tier present: no shape to compare
    assert np.isnan(R.readouts(_cells(rows), m1, n1)["source_shift"][0])     # the context leans on no tier


def test_composition_keeps_degenerate_codons_inside_the_gene():
    T0 = P.T0_
    met, trp, ala = 15, 59, 40                                         # ATG, TGG, GCT
    rows = [(0, "N", T0 + 0, ala, 9.0, 1.0), (0, "N", T0 + 1, met, 3.0, 3.0), (0, "N", T0 + 2, trp, 3.0, 3.0), (0, "N", T0 + 3, 65, 9.0, 1.0),
            (0, "N", T0 + 4, ala, 5.0, 3.0), (0, "N", T0 + 5, ala, 4.0, 3.0), (0, "N", T0 + 11, ala, 9.0, 1.0)]   # first and last codon of the gene
    meta, n0 = _meta()
    c = R.composition(_cells(rows), meta, n0)
    assert np.isclose(c["composition"][0], 1.5) and np.isclose(c["composition_rel"][0], 1.5 / 2.0)    # Ala has 4 codons
    meta, n0 = _meta(ends=False)
    assert np.isclose(R.composition(_cells(rows), meta, n0)["composition"][0], (2 + 1 + 8) / 3.0)


def test_controls():
    out, meta = ruler()
    ctl = R.controls(out, meta)
    assert (ctl["rows_on_target"] == meta["on_target"]).all() and np.isfinite(ctl["host_codon"]).all()
    rows = meta["on_target"] > 0
    assert (ctl["covered_share"][rows] > 0).all() and (ctl["covered_share"] <= 1).all() and (np.abs(ctl["best_similarity"]) <= 1).all()
    one = np.array([len({int(x) for x in r[1:] if x >= 0}) for r in meta["row_tier"]]) == 1
    assert (ctl["coverage_shift"][one & rows] == 0).all() and np.isnan(ctl["coverage_shift"][~rows]).all()
    t = 0
    L = int(meta["target_len"][t])
    tok = out["tok"][t, 0, P.T0_:P.T0_ + L].astype(np.int64)
    cnt = meta["host_counts"][t] + 0.5
    cls, deg = R._classes()
    pos = np.arange(L)
    keep = (tok <= 64) & (pos > 0) & ~((pos == L - 1) & bool(meta["target_ends"][t])) & (deg[cls[tok]] >= 2)
    tot = np.bincount(cls[1:65], weights=cnt, minlength=23)
    assert np.isclose(ctl["host_codon"][t], (-np.log2(cnt[tok[keep] - 1] / tot[cls[tok[keep]]])).mean())


def _triples(n_tr=40, groups=("bacteria", "eukaryota.fungi", "eukaryota.mammals", "eukaryota.protists")):
    names = sorted(groups)
    N = 3 * n_tr
    g_of = np.arange(n_tr) % len(names)
    meta = dict(triple=np.repeat(np.arange(n_tr), 3), variant=np.tile(np.array([0, 1, 2], np.int8), n_tr), unit_a=np.repeat(np.arange(n_tr) % 13, 3),
                unit_d=np.repeat(np.arange(n_tr) % 7, 3), scenario=np.zeros(N, np.int8), donor_tier=np.repeat(np.arange(n_tr) % 2 + 3, 3),
                on_target=np.full(N, 3, np.int8), phase=np.ones(N, np.int8), context_homolog=np.zeros(N, np.int8), row_tier=np.tile(np.array([-1, 3, 4, -1], np.int8), (N, 1)),
                covered=np.full(N, 30), row_gene=np.tile(np.array([-1, 1, 2, -1]), (N, 1)))
    meta["unit_a"] = meta["unit_a"] + 100 * np.repeat(g_of, 3)
    return meta, np.repeat(g_of, 3), names


def test_paired_win_rate_bootstrap_and_differences():
    meta, group, names = _triples()
    tr = R.Triples(meta, group, np.full((len(group), 4), np.nan), ["bacteria"] * len(group), names)
    boot = R.Bootstrap(tr, reps=50)
    score = np.where(meta["variant"] == 0, 1.0, 0.0)
    pop = np.ones(len(tr.ids), bool)
    strata = {k: v for k, v in tr.strata().items() if k[0] in ("all", "donor_tier")}
    rows = R.stat_rows(dict(s=score), tr, pop, strata, boot, label="x", reference=dict(zero=np.zeros(len(score))))
    euk = [r for r in rows if r["block"] == "eukaryotes" and r["stratum"] == "all" and r["comparison"] == "own" and r["statistic"] == "win"][0]
    assert euk["value"] == 1.0 and euk["lo"] == 1.0 and euk["groups"] == 3 and euk["triples"] == 30 and euk["readable"]
    assert np.isclose(euk["vs_zero"]["diff"], 0.5) and euk["vs_zero"]["reference"] == 0.5
    pooled = [r for r in rows if r["block"] == "eukaryotes" and r["stratum"] == "all" and r["statistic"] == "pooled"][0]
    assert pooled["value"] == 1.0
    pro = [r for r in rows if r["block"] != "eukaryotes" and r["stratum"] == "all"][0]
    assert pro["groups"] == 1 and not pro["readable"]                  # one group of the block: too few to read
    tier = [r for r in rows if r["block"] == "eukaryotes" and r["stratum"] == "tier=3" and r["statistic"] == "win"][0]
    assert tier["triples"] < 30 and not tier["readable"]
    rng = np.random.default_rng(3)
    noisy = score + rng.normal(0, 1, len(score))
    sel = np.flatnonzero(tr.group == 1)
    a, reps = R.group_stat("win", noisy[tr.table[sel, 0]], noisy[tr.table[sel, 2]], np.ones((4, len(sel))))
    assert np.allclose(reps, a)                                        # unit weights give the point estimate
    a, reps = R.group_stat("pooled", noisy[tr.table[sel, 0]], noisy[tr.table[sel, 2]], np.ones((4, len(sel))))
    assert np.allclose(reps, a) and np.isclose(a, R.auc(noisy[tr.table[sel, 0]], noisy[tr.table[sel, 2]]))
    d = R.defined_share_rows(np.where(meta["variant"] == 0, np.nan, 1.0), tr, pop, boot)
    assert [x["value"] for x in d if x["block"] == "eukaryotes"] == [-1.0]
    cov = R.coverage_table(tr, pop)
    assert {(c["direction"], c["recipient"]): c["mark"] for c in cov}[("prokaryote <-> prokaryote", "bacteria")] == "construct only"
    assert {(c["direction"], c["recipient"]): c["triples"] for c in cov}[("prokaryote -> eukaryote", "eukaryota.fungi")] == 10


def test_directions_follow_the_charter_table():
    assert R.direction("bacteria", "eukaryota.fungi") == "prokaryote -> eukaryote"
    assert R.direction("mobile.plasmid", "archaea") == "prokaryote <-> prokaryote"
    assert R.direction("viruses.phage", "bacteria") == "virus -> host" and R.direction("bacteria", "viruses.phage") == "host -> virus"
    assert R.direction("viruses.giant", "viruses.eukaryotic") == "virus <-> virus"
    assert R.direction("eukaryota.plants_algae", "eukaryota.invertebrates") == "eukaryote <-> eukaryote"


def test_a_table_whose_row_0_has_no_context_codon_has_no_C_passes():
    """A genome of very few genes (i and k only, on two replicons) gives row 0 nothing outside the target block: no pass
    can score context cells, so the C passes are not made and the source shift is undefined (ruler run 20261122T141051,
    table 2403, stopped here)."""
    out, meta = ruler()
    t = 0
    tab = {k: np.array(out[k][t]) for k in P.FIELDS}
    ctx = P.CONTEXT_SLOT
    tab["tok"][0, ctx], tab["col"][0, ctx] = api.PAD, -1
    tab["n"][0, :P.TARGET_BLOCK] = 0
    tab["kind"][0, :P.TARGET_BLOCK] = api.KIND_NONE
    tab["frame_len"][:P.TARGET_BLOCK] = 0
    for k in ("tok", "col"):                                           # the other rows have no context frame either
        tab[k][1:, ctx] = api.PAD if k == "tok" else -1
    tab["n"][1:, :P.TARGET_BLOCK] = 0
    tab["kind"][1:, :P.TARGET_BLOCK] = np.where(tab["kind"][1:, :P.TARGET_BLOCK] == api.KIND_NONE, api.KIND_NONE, api.KIND_NOT_SEARCHED)
    names = [p.name for p in P.plan(tab, meta["row_tier"][t], t, relatives=True, presence=True, mask_id=0)]
    assert "T" in names and "W" in names and not any(x == "C" or x.startswith(("C-t", "P:C")) for x in names)
    cells = P.run([(t, tab, meta["row_tier"][t], 0)], T.render, toy_score, relatives=True, presence=True)
    one = {k: v[t:t + 1] for k, v in meta.items()}
    one["triple"] = np.zeros(1, np.int64)
    cells["table"] = cells["table"] * 0
    sc = R.readouts(cells, one, tab["n"][None, 0, :])
    assert np.isnan(sc["source_shift"][0]) and np.isfinite(sc["gain"][0])
    assert P.plan_digest([(t, tab, meta["row_tier"][t], 0)], True, True)[0] == cells["digest"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
