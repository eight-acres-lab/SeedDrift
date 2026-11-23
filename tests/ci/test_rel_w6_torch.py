"""The trainer of the relational line end to end on small random tables (package W6): preflight with its divergence
guard, a few steps with the warm-up and the moving average, the held-out evaluation at the nodes of tables seen and at
the end, the result rows and the digests of the cells scored. torch only: skipped where torch is not installed (the CI
environment), run in the model interpreter by the stage rel_w6_selftest (the trainer's selftest mode loads this file
by its path) or by hand, `python tests/ci/test_rel_w6_torch.py`."""
import csv
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

import test_rel_w6_data as D  # noqa: E402

from seeddrift.train.rel import api as A  # noqa: E402
from seeddrift.train.rel.data import tables as T  # noqa: E402
from seeddrift.train.rel.evalh import conditions as cond  # noqa: E402
from seeddrift.train.rel.evalh import score as SC  # noqa: E402

if HAVE_TORCH:
    from seeddrift.train.rel.trainer import loop  # noqa: E402
else:
    import pytest
    pytestmark = pytest.mark.skip(reason="torch is not installed here; the model interpreter runs these tests")


def _config(root, kind=A.TABLE_SYNTHETIC, **over):
    tables = D.write_tableset(root, [("train/s0000", D.random_tables(24, rows=4, seed=1)), ("heldout/s0000", D.random_tables(12, rows=4, seed=2, split=1))],
                              kind=kind)
    cfg = dict(name="tiny", author="w6", model="axial", size=dict(layers=1, width=256), objective="mlm", lr=1e-3, seed=1, wall_s=900,
               train_tables=tables, train_root=root, heldout_tables=tables, heldout_root=root, out=os.path.join(root, "out"), probe=True,
               rows=[2, 4], per_step=2, micro=1, max_steps=6, warmup_steps=3, eval_first=4, eval_per_group=1, preflight_steps=8,
               preflight_warmup=3, preflight_guard="reduce", device="cpu", threads=2, run="run-x", commit="abc", group_names={"0": "bacteria"})
    cfg.update(over)
    return cfg


def _digest(cfg, root, condition):
    held = T.open_split(root, cfg["heldout_tables"], 1)
    data = held.load(T.MODEL_FIELDS)
    d = cond.ItemsDigest(condition)
    d.add(cond.materialise(data, np.arange(len(held)), condition))
    return d.hexdigest()


def test_tiny_fit():
    root = tempfile.mkdtemp()
    cfg = _config(root)
    pre = loop.preflight(cfg)
    assert pre["ok"] and pre["micro"] == 1 and pre["params"] > 1e6 and pre["cells_per_s"] > 0 and set(pre["eval_bits"]) == set(cond.CONDITIONS)
    assert pre["guard"]["steps"] == 8 and pre["lr"] in (1e-3, 3e-4) and pre["lr_reduced"] == (pre["lr"] != 1e-3)
    out = loop.fit(cfg)
    assert out["status"] == "ok" and out["steps"] == 6 and out["ended_by"] == "max_steps" and out["tables_seen"] == 12 and out["cells_seen"] > 0
    assert [e["tables_seen"] for e in out["evals"]] == [0, 4, 8] and out["weights_read"] == "ema" and out["eval_final"] == 12
    assert abs(out["evals"][0]["bits"]["scattered"] - math.log2(65)) < 0.3                         # an untrained model is near uniform
    rows = SC.read_rows(os.path.join(cfg["out"], "results.tsv"))
    assert set(rows[0]) == set(A.RESULT_ROW)
    assert {r["weights"] for r in rows} == {"ema@tables=0", "ema@tables=4", "ema@tables=8", "ema@final"}
    final = [r for r in rows if r["weights"] == "ema@final"]
    assert {r["condition"] for r in final} == set(cond.CONDITIONS) and {r["item_half"] for r in final} == {"all"}
    assert {r["item_half"] for r in rows if r["weights"] != "ema@final"} == {"subset"}
    assert all(r["layout"] == A.LAYOUT_STORED and r["table_kind"] == A.TABLE_SYNTHETIC and r["variant"] == "tiny" and r["run"] == "run-x" for r in rows)
    top = [r for r in final if r["group"] == "all" and r["stratum"] == "all"]
    assert len(top) == 5 and all(0 < float(r["bits"]) < 8 and int(r["cells_seen"]) == out["cells_seen"] for r in top)
    assert {r["group"] for r in final} >= {"all", "bacteria", "11"} and {r["stratum"] for r in final} >= {"all", "other_rows>=2"} | set(SC.PARTS)
    for c in cond.CONDITIONS:                                     # the two parts add up to the bits of the codon
        whole, aa, rest = (float(next(r["bits"] for r in final if r["condition"] == c and r["group"] == "all" and r["stratum"] == x))
                           for x in ("all",) + tuple(SC.PARTS))
        assert abs(aa + rest - whole) < 1e-4 and 0 < aa < whole, c
    for c in cond.CONDITIONS:                                     # the cells scored are the evaluator's, table for table
        assert out["final_items"][c] == _digest(cfg, root, c), c
    with open(os.path.join(cfg["out"], "steps.csv")) as f:
        steps = list(csv.DictReader(f))
    lr = [float(s["lr"]) for s in steps]
    assert len(steps) == 6 and lr[0] < lr[1] < lr[2] == lr[3] == lr[5] == 1e-3 and all(float(s["loss_bits"]) > 0 for s in steps)
    assert int(steps[-1]["tables_seen"]) == 12 and int(steps[-1]["cells_seen"]) == out["cells_seen"] and int(steps[-1]["target_cells"]) == out["target_cells"]
    assert not os.path.exists(os.path.join(cfg["out"], "final.pt"))                                # probe mode keeps no checkpoint
    with open(os.path.join(cfg["out"], "evals.jsonl")) as f:
        ev = [json.loads(line) for line in f]
    assert [e.get("final", False) for e in ev] == [False, False, False, True] and ev[-1]["items"] == out["final_items"]
    try:                                                                                         # a second training in the same folder
        loop.fit(cfg)
    except SystemExit:
        pass
    else:
        raise AssertionError("a training started in a folder that holds one")


def test_preflight_guard_stops_by_default():
    root = tempfile.mkdtemp()
    try:
        loop.preflight(_config(root, lr=10.0, preflight_guard="stop"))
    except SystemExit as err:
        assert "did not train" in str(err)
    else:
        raise AssertionError("a preflight whose loss did not fall passed")
    with open(os.path.join(root, "out", "preflight.json")) as f:
        assert json.load(f)["ok"] is False


def test_anyorder_fill_in_trains():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, masks=["anyorder"], max_steps=3))
    assert out["status"] == "ok" and out["steps"] == 3 and all(math.isfinite(v) for v in out["final"].values() if v is not None)


def test_hierarchical_head_trains():
    root = tempfile.mkdtemp()
    cfg = _config(root, size=dict(layers=1, width=256, head="hierarchical"), max_steps=3)
    pre = loop.preflight(cfg)
    assert pre["ok"] and pre["params"] == loop.registry.load("model", "axial").count_parameters(1, 256, head="hierarchical")
    out = loop.fit(cfg)
    assert out["status"] == "ok" and out["steps"] == 3 and all(0 < v < 12 for v in out["final"].values() if v is not None)
    final = [r for r in SC.read_rows(os.path.join(cfg["out"], "results.tsv")) if r["weights"] == "ema@final" and r["group"] == "all"]
    for c in cond.CONDITIONS:                                     # the parts of the bits still add up to the bits of the codon
        whole, aa, rest = (float(next(r["bits"] for r in final if r["condition"] == c and r["stratum"] == x)) for x in ("all",) + tuple(SC.PARTS))
        assert abs(aa + rest - whole) < 1e-4, c


def test_leakfree_layout_trains():
    """Leak-free scattered and span samples, stored whole genes: the model is built with the ext vector (one more
    width of parameters), trains, and its checkpoint records the size it was built with."""
    root = tempfile.mkdtemp()
    cfg = _config(root, layout=A.LAYOUT_LEAKFREE, whole_gene_layout=A.LAYOUT_STORED, max_steps=3, eval_first=4)
    axial = loop.registry.load("model", "axial")
    pre = loop.preflight(cfg)
    assert pre["ok"] and pre["params"] == axial.count_parameters(1, 256) + 256
    out = loop.fit(cfg)
    assert out["status"] == "ok" and out["steps"] == 3
    rows = SC.read_rows(os.path.join(cfg["out"], "results.tsv"))         # probe mode: the leak-free conditions rendered online
    lay = {}
    for r in rows:
        if r["weights"] == "ema@final":
            lay.setdefault(r["condition"], set()).add(r["layout"])
    assert lay == {"scattered": {"leakfree-online"}, "whole_gene": {"leakfree-online", "stored"}, "whole_gene+repaired": {"leakfree-online", "stored"},
                   "scattered+no_relatives": {"stored"}, "whole_gene+no_relatives": {"stored"}}           # the whole gene in both layouts
    assert {"whole_gene@stored", "whole_gene+repaired@stored"} <= set(out["final"])
    assert all(set(e["bits"]) == set(cond.CONDITIONS) for e in out["evals"])                                    # the nodes: the main readings
    for c in cond.CONDITIONS:                                     # the cells scored stay the evaluator's whatever the layout
        assert out["final_items"][c] == _digest(cfg, root, c), c
    assert out["final_items"]["whole_gene@stored"] == out["final_items"]["whole_gene"]
    real_root = tempfile.mkdtemp()                                # outside probe mode a leak-free arm needs the strict offline rendering
    try:
        loop.fit(_config(real_root, kind=A.TABLE_REAL, probe=False, layout=A.LAYOUT_LEAKFREE, whole_gene_layout=A.LAYOUT_STORED, max_steps=1,
                         eval_first=4))
    except SystemExit as err:
        assert "offline" in str(err)
    else:
        raise AssertionError("a leak-free arm was scored without the strict offline rendering")
    run = loop.Run(dict(cfg, out=os.path.join(root, "other")))
    assert run.size == dict(layers=1, width=256, ext=True) and run.model.ext is not None
    assert loop.layouts_of(run.cfg) == {"scattered": "leakfree", "span": "leakfree", "whole_gene": "stored"}
    assert sorted({run.sampler.layouts[e["scheme"]] for s in range(8) for e in run.sampler.draw(s)}) == ["leakfree", "stored"]
    try:
        loop.Run(dict(cfg, layout="unaligned", out=os.path.join(root, "x")))
    except SystemExit as err:
        assert "layouts" in str(err)
    else:
        raise AssertionError("a training in the unaligned layout was accepted")


def test_moving_average():
    root = tempfile.mkdtemp()
    run = loop.Run(_config(root))
    w0 = [p.detach().clone() for p in run.ema_weights]
    run.step(run.sampler.draw(0), 1e-3)
    run.update_ema(1)
    d = 2.0 / 11.0
    for e, a, w in zip(run.ema_weights, w0, run.weights):
        assert torch.allclose(e, d * a + (1 - d) * w.detach(), atol=1e-6)
    assert any(not torch.equal(e, w) for e, w in zip(run.ema_weights, run.weights))
    assert all(not p.requires_grad for p in run.ema_weights)


def test_tiny_fit_without_relatives():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, relatives=False, max_steps=2, per_step=4, micro=4, ema=False))
    assert out["status"] == "ok" and out["conditions"] == [c for c in cond.CONDITIONS if "no_relatives" in c] and out["cells_seen"] == 2 * 4 * A.ROW_LEN
    rows = SC.read_rows(os.path.join(root, "out", "results.tsv"))
    assert {r["condition"] for r in rows} == {"scattered+no_relatives", "whole_gene+no_relatives"} and {r["weights"] for r in rows} == {
        "raw@tables=0", "raw@tables=4", "raw@tables=8", "raw@final"}
    assert {r["budget"] for r in rows if r["weights"] == "raw@tables=4"} <= {"tables=4;wall_s=%d" % s for s in range(0, 900)}
    assert "final_best" not in out                                                              # probe mode keeps no best weights


def test_the_best_node_of_an_arm_without_relatives_is_read_on_every_table():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, kind=A.TABLE_REAL, probe=False, relatives=False, max_steps=4, per_step=4, micro=4, eval_first=4))
    rows = SC.read_rows(os.path.join(root, "out", "results.tsv"))
    if out["best"]["tables_seen"] < out["tables_seen"]:
        best = [r for r in rows if r["weights"] == "ema@best"]
        assert out["final_best"]["tables_seen"] == out["best"]["tables_seen"] and {r["item_half"] for r in best} == {"all"}
        assert {r["budget"].split(";")[0] for r in best} == {"tables=%d" % out["best"]["tables_seen"]}
        assert out["final_best"]["items"] == out["final_items"]                                  # the same cells as the end
    else:
        assert "final_best" not in out and not any(r["weights"] == "ema@best" for r in rows)


def test_saved_weights_evaluated_again():
    """An arm's final.pt evaluated again gives the bits and the digests of the training's own final evaluation; best.pt
    gives rows ema@best."""
    root = tempfile.mkdtemp()
    cfg = _config(root, kind=A.TABLE_REAL, probe=False, relatives=False, max_steps=4, per_step=4, micro=4, eval_first=4)
    out = loop.fit(cfg)
    folder = os.path.join(root, "again")
    best = out["best"]
    again = loop.evaluate_checkpoints(dict(cfg, out=folder, final=os.path.join(root, "out", "final.pt"), best=os.path.join(root, "out", "best.pt"),
                                           final_tables=out["tables_seen"], final_cells=out["cells_seen"], final_wall_s=out["wall_s"],
                                           best_tables=best["tables_seen"], best_cells=best["cells_seen"], best_wall_s=best["wall_s"]))
    assert again["status"] == "ok" and again["final_items"] == out["final_items"] and again["tables_seen"] == out["tables_seen"]
    assert all(abs(again["final"][c] - out["final"][c]) < 1e-5 for c in out["final"]) and "final_best" in again
    rows = SC.read_rows(os.path.join(folder, "results.tsv"))
    assert {r["weights"] for r in rows} == {"ema@final", "ema@best"} and {r["item_half"] for r in rows} == {"all"}
    assert {r["budget"].split(";")[0] for r in rows if r["weights"] == "ema@final"} == {"tables=%d" % out["tables_seen"]}


def test_saved_weights_read_on_a_ruler_set_and_never_trained_on():
    """An evaluation of saved weights reads a ruler set (package W18) as its held-out set; a training never does."""
    root = tempfile.mkdtemp()
    cfg = _config(root, kind=A.TABLE_REAL, probe=False, max_steps=3, eval_first=4)
    out = loop.fit(cfg)
    rroot = tempfile.mkdtemp()
    ruler = D.write_tableset(rroot, [("heldout/s0000", D.random_tables(12, rows=4, seed=3, split=1))], kind="ruler", name="w6-ruler")
    again = loop.evaluate_checkpoints(dict(cfg, heldout_tables=ruler, heldout_root=rroot, out=os.path.join(root, "again"),
                                           final=os.path.join(root, "out", "final.pt"), best=None, final_tables=out["tables_seen"],
                                           final_cells=out["cells_seen"], final_wall_s=out["wall_s"]))
    rows = SC.read_rows(os.path.join(root, "again", "results.tsv"))
    assert again["status"] == "ok" and {r["table_kind"] for r in rows} == {"ruler"} and {r["table_set"] for r in rows} == {"w6-ruler"}
    try:
        loop.fit(dict(cfg, heldout_tables=ruler, heldout_root=rroot, out=os.path.join(root, "out2")))
    except SystemExit as err:
        assert "kind" in str(err)
    else:
        raise AssertionError("a training took a ruler set")


def test_score_cells_gives_the_bits_amino_acid_bits_and_entropy_of_every_cell():
    run = loop.Run(_config(tempfile.mkdtemp()))
    ex = run.sampler.draw(0) + run.sampler.draw(1)
    got = loop.E.score_cells(run.model, ex, run.device, max_rows=5)
    n = [int((e["truth"] > 0).sum()) for e in ex]
    assert got["example"].tolist() == [k for k, m in enumerate(n) for _ in range(m)]
    from seeddrift.train.rel.data import sampler as S
    b = loop.E.to_device(S.collate(ex), run.device)
    with torch.no_grad():
        lp = run.model.eval().logp(b["grid"], b["sizes"], b["row"], b["col"]).double()
    bits = (-lp.gather(1, (b["token"] - A.TARGET_MIN)[:, None])[:, 0] / math.log(2)).numpy()
    ent = (-(lp.exp() * lp).sum(1) / math.log(2)).numpy()
    assert np.allclose(got["bits"], bits, atol=1e-5) and np.allclose(got["entropy"], ent, atol=1e-5)
    assert (got["aa_bits"] <= got["bits"] + 1e-9).all() and (got["aa_bits"] >= 0).all() and (got["entropy"] <= math.log2(65) + 1e-6).all()


def run_ruler(scorer, tables, info, out_dir, render=None, presence=False, max_rows=32, progress=None, scenarios=None, presence_scenarios=None,
              adapt_v0=False):
    """A stand-in of a readout module (package W18's run_ruler): the whole target of row 0 hidden on every table of the
    set, scored, written to ruler_cells.npz."""
    from seeddrift.train.rel.data.masks import whole_gene
    render = render or T.render
    items = []
    for i in range(len(tables)):
        tab = T.take_rows(tables.table(i), T.present_rows(tables.table(i)))
        vis, tgt = whole_gene.plan(dict(n=tab["n"], kind=tab["kind"]), np.random.default_rng(0))
        r = render(tab, vis, tgt)
        items.append(dict(grid=r[0], truth=r[1]) if len(r) == 2 else dict(grid=r[0], truth=r[1], ext=r[2]))
    got = scorer(items)
    np.savez(os.path.join(out_dir, "ruler_cells.npz"), **got)
    return dict(cells=int(len(got["bits"])), tables=len(items), weights=info["weights"], presence=bool(presence), kind=tables.kind,
                scenarios=str(scenarios), presence_scenarios=str(presence_scenarios), adapt_v0=bool(adapt_v0))


def test_a_readout_reads_a_ruler_set_with_final_or_untrained_weights():
    root = tempfile.mkdtemp()
    cfg = _config(root, kind=A.TABLE_REAL, probe=False, max_steps=3, eval_first=4)
    out = loop.fit(cfg)
    rroot = tempfile.mkdtemp()
    ruler = D.write_tableset(rroot, [("heldout/s0000", D.random_tables(6, rows=4, seed=3, split=1))], kind="ruler", name="w6-ruler")
    base = dict(cfg, heldout_tables=ruler, heldout_root=rroot, readout=dict(module="test_rel_w6_torch", presence=True, max_rows=8))
    a = loop.evaluate_checkpoints(dict(base, out=os.path.join(root, "r-final"), final=os.path.join(root, "out", "final.pt"),
                                       final_tables=out["tables_seen"], final_cells=out["cells_seen"], final_wall_s=out["wall_s"]))
    b = loop.evaluate_checkpoints(dict(base, out=os.path.join(root, "r-untrained"), final="untrained", readout=dict(base["readout"], in_place_only=True,
                                                                                                                       adapt_v0=True)))
    assert a["status"] == b["status"] == "ok" and a["weights"] == "final" and b["weights"] == "untrained" and a["ruler"]["kind"] == "ruler"
    assert a["ruler"]["tables"] == 6 and a["ruler"]["cells"] > 0 and a["tables_seen"] == out["tables_seen"] and b["tables_seen"] == 0
    assert b["ruler"]["scenarios"] == "[0]" and a["ruler"]["scenarios"] == "None" and a["ruler"]["presence_scenarios"] == "None"
    assert b["ruler"]["adapt_v0"] is True and a["ruler"]["adapt_v0"] is False
    za, zb = (np.load(os.path.join(root, d, "ruler_cells.npz")) for d in ("r-final", "r-untrained"))
    assert za["bits"].shape == zb["bits"].shape and not np.allclose(za["bits"], zb["bits"])            # trained weights read differently
    assert os.path.exists(os.path.join(root, "r-untrained", "untrained.pt")) and not os.path.exists(os.path.join(root, "r-final", "results.tsv"))
    assert abs(zb["entropy"].mean() - math.log2(65)) < 0.3                                             # an untrained model is near uniform


def test_extra_and_only_conditions():
    E = loop.E
    assert E.conditions_of(True) == list(cond.CONDITIONS) and E.conditions_of(True, [cond.SUFFIX])[-1] == cond.SUFFIX
    assert E.conditions_of(True, only=[cond.SUFFIX]) == [cond.SUFFIX] and E.conditions_of(False, [cond.SUFFIX]) == E.conditions_of(False)
    try:
        E.conditions_of(True, ["sideways"])
    except ValueError:
        pass
    else:
        raise AssertionError("a condition that is not the evaluator's was accepted")
    root = tempfile.mkdtemp()                                     # suffix scored beside today's five, its cells the evaluator's
    cfg = _config(root, extra_conditions=[cond.SUFFIX], max_steps=2)
    out = loop.fit(cfg)
    assert cond.SUFFIX in out["final"] and out["final_items"][cond.SUFFIX] == _digest(cfg, root, cond.SUFFIX)
    rows = SC.read_rows(os.path.join(cfg["out"], "results.tsv"))
    assert {r["condition"] for r in rows if r["weights"] == "ema@final"} == set(cond.CONDITIONS) | {cond.SUFFIX}
    assert {r["condition"] for r in rows if r["weights"].startswith("ema@tables=")} == set(cond.CONDITIONS)       # references at the end only
    root = tempfile.mkdtemp()
    cfg = _config(root, extra_conditions=[cond.SUFFIX], references_at_nodes=True, max_steps=2)
    loop.fit(cfg)
    rows = SC.read_rows(os.path.join(cfg["out"], "results.tsv"))
    assert {r["condition"] for r in rows if r["weights"].startswith("ema@tables=")} == set(cond.CONDITIONS) | {cond.SUFFIX}


def test_v1_tables_train_with_the_frame_layout_the_genome_bias_and_the_end_controls():
    """Tables of version 1: every mask leak-free, the whole gene on its frame relative through the pair index, the
    same-genome bias; at the end the two controls of the renderer under scattered and whole_gene; the cells scored are
    the evaluator's whatever the layout and control."""
    root = tempfile.mkdtemp()
    train, held = D.v1_tables(24, rows=8, seed=11), D.v1_tables(12, rows=8, seed=12, split=1)
    tables = D.write_tableset(root, [("train/s0000", train), ("heldout/s0000", held)])
    ix = D.frame_index_of(held, D.frame_index_of(train))
    cfg = _config(root, train_tables=tables, heldout_tables=tables, rows=[2, 8], layout=A.LAYOUT_LEAKFREE, whole_gene_layout=A.LAYOUT_LEAKFREE,
                  pair_index=ix, size=dict(layers=1, width=256, genome_bias=True), end_controls=["shuffle", "presence"], max_steps=3, eval_first=4)
    axial = loop.registry.load("model", "axial")
    pre = loop.preflight(cfg)
    assert pre["ok"] and pre["params"] == axial.count_parameters(1, 256) + 256 + 1 * 2       # the ext vector, one bias per column head
    out = loop.fit(cfg)
    assert out["status"] == "ok" and {"scattered@shuffle", "whole_gene@presence"} <= set(out["final"])
    rows = SC.read_rows(os.path.join(cfg["out"], "results.tsv"))
    lay = {}
    for r in rows:
        if r["weights"] == "ema@final":
            lay.setdefault(r["condition"], set()).add(r["layout"])
    full = {"leakfree-online", "leakfree-online+shuffle", "leakfree-online+presence"}
    assert lay == {"scattered": full, "whole_gene": full, "whole_gene+repaired": {"leakfree-online"}, "scattered+no_relatives": {"stored"},
                   "whole_gene+no_relatives": {"stored"}}
    assert all(set(e["bits"]) == set(cond.CONDITIONS) for e in out["evals"])                     # the controls at the end only
    for c in cond.CONDITIONS:
        assert out["final_items"][c] == _digest(cfg, root, c), c
    assert out["final_items"]["scattered@shuffle"] == out["final_items"]["scattered"]
    assert out["final"]["scattered@presence"] != out["final"]["scattered"]
    kept = tempfile.mkdtemp()                                    # keep_node_weights: the moving average of every node, outside probes
    kt = D.write_tableset(kept, [("train/s0000", train), ("heldout/s0000", held)], kind=A.TABLE_REAL)
    got = loop.fit(dict(cfg, train_tables=kt, heldout_tables=kt, train_root=kept, heldout_root=kept, probe=False, keep_node_weights=True, eval_online=True,
                        out=os.path.join(kept, "out")))
    assert got["node_weights"] == [4] and os.path.exists(os.path.join(kept, "out", "ema_tables4.pt"))
    assert "model" not in torch.load(os.path.join(kept, "out", "ema_tables4.pt"), map_location="cpu", weights_only=True)
    real = tempfile.mkdtemp()                                    # outside probe mode the whole gene needs the pair index
    rt = D.write_tableset(real, [("train/s0000", train), ("heldout/s0000", held)], kind=A.TABLE_REAL)
    try:
        loop.Run(dict(cfg, train_tables=rt, heldout_tables=rt, train_root=real, heldout_root=real, pair_index=None, probe=False,
                      out=os.path.join(real, "x")))
    except SystemExit as err:
        assert "pair index" in str(err) or "pair_index" in str(err)
    else:
        raise AssertionError("an arm laid the whole gene on its frame relative without the pair index")


def _offline(root, held):
    """A strict offline rendering of the held-out tables as package W3 writes it (one part per mask, layout leakfree):
    scattered and suffix the stored tables, whole_gene every table laid on its frame relative by align.frame_layout."""
    from seeddrift.train.rel.tables import align as AL
    framed = {f: np.array(v, copy=True) for f, v in held.items()}
    for i in range(len(held["tok"])):
        tab = {f: held[f][i] for f in ("tok", "n", "col", "kind", "frame_len", "frame_row")}
        present = T.present_rows(tab)
        col, frame_row, frame_len = AL.frame_layout(T.take_rows(tab, present))
        frame_row = np.where(frame_row >= 0, present[np.clip(frame_row, 0, None)], frame_row)    # back to the stored rows' numbers
        framed["col"][i][present], framed["frame_row"][i], framed["frame_len"][i] = col, frame_row, frame_len
    path = D.write_tableset(root, [("scattered/s0000", held), ("suffix/s0000", held), ("whole_gene/s0000", framed)], kind=A.TABLE_REAL,
                            name="w3-offline", sub="offline")
    with open(path) as f:
        doc = json.load(f)
    doc.update(layout=A.LAYOUT_LEAKFREE, parts={p: dict(folder="offline/" + p) for p in ("scattered", "suffix", "whole_gene")})
    with open(path, "w") as f:
        json.dump(doc, f)
    return path


def test_round1_whole_gene_by_the_strict_aligner_nodes_online_end_offline():
    """Round 1 (protocol round1): the whole gene laid by W3's strict aligner after the rows are drawn (no pair index);
    the nodes read four conditions rendered online (rows leakfree-online), the end the strict offline rendering (rows
    leakfree), the controls keyed <condition>@<control>; nodes also every nodes_every tables, the moving average of
    each kept as ema_tables<n>.pt; the mask schemes' shares of cells and loss; the examples drawn by a background
    process are those drawn in the training's process."""
    root = tempfile.mkdtemp()
    train, held = D.v1_tables(24, rows=8, seed=11), D.v1_tables(12, rows=8, seed=12, split=1)
    tables = D.write_tableset(root, [("train/s0000", train), ("heldout/s0000", held)], kind=A.TABLE_REAL, sub="v1")   # beside _config's own set
    off = _offline(root, held)
    four = [cond.SCATTERED, cond.WHOLE_GENE, cond.SCATTERED + "+" + cond.NO_RELATIVES, cond.WHOLE_GENE + "+" + cond.NO_RELATIVES]
    cfg = _config(root, train_tables=tables, heldout_tables=tables, probe=False, rows=[2, 8], layout=A.LAYOUT_LEAKFREE, whole_gene_layout="frame",
                  eval_online=True, heldout_offline_tables=off, heldout_offline_root=root, only_conditions=four, end_controls=["shuffle", "presence"],
                  keep_node_weights=True, nodes_every=4, nodes_every_from=8, max_steps=6, eval_first=4)
    pre = loop.preflight(cfg)
    assert pre["ok"] and pre["sampler"]["frame_tables"] > 0 and pre["sampler"]["frame_s_per_table"] > 0 and pre["tables_per_s"] > 0
    out = loop.fit(cfg)
    assert out["status"] == "ok" and out["conditions"] == four
    assert [e["tables_seen"] for e in out["evals"]] == [0, 4, 8, 12] and out["node_weights"] == [4, 8, 12]   # doubling, and every 4 from 8
    assert all(os.path.exists(os.path.join(cfg["out"], "ema_tables%d.pt" % t)) for t in (4, 8, 12))
    rows = SC.read_rows(os.path.join(cfg["out"], "results.tsv"))
    lay = {}
    for r in rows:
        lay.setdefault((r["weights"] == "ema@final", r["condition"]), set()).add(r["layout"])
    assert lay[(False, cond.SCATTERED)] == lay[(False, cond.WHOLE_GENE)] == {"leakfree-online"}
    assert lay[(True, cond.SCATTERED)] == lay[(True, cond.WHOLE_GENE)] == {"leakfree", "leakfree+shuffle", "leakfree+presence"}
    assert lay[(True, four[2])] == lay[(True, four[3])] == lay[(False, four[2])] == {"stored"}
    assert {"scattered@shuffle", "scattered@presence", "whole_gene@shuffle", "whole_gene@presence"} <= set(out["final"])
    for c in four:                                               # the cells scored are the evaluator's, online and offline
        assert out["final_items"][c] == _digest(cfg, root, c), c
    sch = out["schemes"]
    assert set(sch) == {"scattered", "span", "whole_gene"} and sum(x["tables"] for x in sch.values()) == 12
    assert abs(sum(x["cell_share"] for x in sch.values()) - 1) < 1e-3 and abs(sum(x["loss_share"] for x in sch.values()) - 1) < 1e-3
    with open(os.path.join(cfg["out"], "steps.csv")) as f:
        steps = list(csv.DictReader(f))
    assert {"cells_scattered", "cells_whole_gene", "data_wait_s"} <= set(steps[0])
    assert sum(int(r["cells_whole_gene"] or 0) for r in steps) == sch["whole_gene"]["target_cells"]
    run = loop.Run(dict(cfg, out=os.path.join(root, "again")))     # the whole gene online by the strict aligner = W3's offline part
    data, ids = run.held_data()[0], np.arange(len(run.held))
    ts = run.offline["parts"]["whole_gene"]
    offline = ts.load(0, ts.tables, loop.E.PLAN_FIELDS, verify=False)
    b_on = loop.E.score(run.reader(), data, ids, cond.WHOLE_GENE, run.device, layout=A.LAYOUT_LEAKFREE, frame=dict(strict=True, cache=None))[0]
    b_off = loop.E.score(run.reader(), offline, ids, cond.WHOLE_GENE, run.device, layout=A.LAYOUT_LEAKFREE)[0]
    assert len(b_on) and np.allclose(b_on, b_off)
    w1 = loop.fit(dict(cfg, data_workers=1, keep_node_weights=False, out=os.path.join(root, "workers")))   # drawn ahead in a background process
    with open(os.path.join(root, "workers", "steps.csv")) as f:
        again = list(csv.DictReader(f))
    assert [r["loss_bits"] for r in again] == [r["loss_bits"] for r in steps] and w1["data_workers"] == 1
    assert w1["final"] == out["final"]


def test_split_and_gradient_checks():
    root = tempfile.mkdtemp()
    try:
        loop.Run(_config(root, train_split=1))
    except SystemExit as err:
        assert "split" in str(err)
    else:
        raise AssertionError("a training on split 1 was accepted")
    run = loop.Run(_config(tempfile.mkdtemp()))
    with torch.no_grad():
        next(run.model.parameters()).fill_(float("inf"))
    before = [p.detach().clone() for p in run.model.parameters()]
    try:
        run.step(run.sampler.draw(0), 1e-3)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("a step with a gradient that is not finite was taken")
    assert all(torch.equal(a, b) or not torch.isfinite(a).all() for a, b in zip(before, run.model.parameters()))


def test_real_tables_only_outside_probe_mode():
    root = tempfile.mkdtemp()
    try:
        loop.fit(_config(root, probe=False))
    except SystemExit as err:
        assert "probe" in str(err)
    else:
        raise AssertionError("a training outside probe mode took synthetic tables")


def test_budget_ends_the_run_and_checkpoints_are_kept():
    root = tempfile.mkdtemp()
    out = loop.fit(_config(root, kind=A.TABLE_REAL, probe=False, max_steps=None, budget_cells=30000))
    assert out["ended_by"] == "budget" and out["cells_seen"] >= 30000 and out["status"] == "ok"
    assert os.path.exists(os.path.join(root, "out", "final.pt")) and os.path.exists(os.path.join(root, "out", "best.pt")) and out["best"] is not None
    ck = torch.load(os.path.join(root, "out", "final.pt"), map_location="cpu", weights_only=True)
    assert ck["cells_seen"] == out["cells_seen"] and any(k.startswith("blocks.0.row.") for k in ck["model"]) and set(ck["ema"]) == set(ck["model"])
    best = torch.load(os.path.join(root, "out", "best.pt"), map_location="cpu", weights_only=True)
    assert "model" not in best and best["tables_seen"] == out["best"]["tables_seen"]
    z = np.load(os.path.join(root, "out", "heldout_final.npz"))
    assert len(z["bits_whole_gene"]) == len(z["table_whole_gene"]) > 0 and set(z["table_scattered"].tolist()) <= set(range(12))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
