"""Stage rel_w5_sources (package W5) in the CI environment (no torch): its settings, the bits gained from relatives read
from result rows, the sums of the readout's cells by stratum, the report and the gate. The readout itself runs in the
model interpreter (tests/ci/test_rel_w5_source_torch.py, test_the_sources_readout_on_saved_weights)."""
import json
import os

import numpy as np
import pytest
import yaml

from conftest import ROOT
from ci.gates import GateFailure
from ci.stages.common import Stage
from ci.stages.rel import kinds, w5_sources as W
from seeddrift.train.rel.evalh import score as SC

HEADER = ["weights", "group", "condition", "layout", "stratum", "bits"]


def _rows(path, table):
    with open(path, "w") as f:
        f.write("\t".join(HEADER) + "\n")
        for (cond, layout, stratum), bits in table.items():
            f.write("\t".join(["ema@final", "all", cond, layout, stratum, "%.6f" % bits]) + "\n")
        f.write("\t".join(["ema@tables=1000", "all", "scattered", "leakfree", "all", "9.0"]) + "\n")


def test_settings_and_registry():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "ci", "profiles", "rel-dev-w5-sources-i1", "config.yaml")))
    s = W.settings(cfg)
    assert [a["name"] for a in s["arms"]] == ["S1", "S2"] and s["reference"]["folder"].startswith("w6-ref")
    assert s["conditions"] == {"scattered": "leakfree"} and s["per_group"] == 50
    assert s["arms"][0]["fit"].endswith(os.path.join("data", "rel_w6", "fit", "w5-src30m-s1-s1"))
    d = kinds.stage("w5_sources")
    assert d["kind"] == "rel_traineval" and d["module"] == "ci.stages.rel.w5_sources" and d["output"] == os.path.join(W.OUT, "report.json")
    for bad in (dict(arms=[]), dict(cfg["rel_w5_sources"], conditions={"scattered": "sideways"}),
                dict(cfg["rel_w5_sources"], arms=[dict(name="x", run="r")]), dict(cfg["rel_w5_sources"], extra=1)):
        with pytest.raises(SystemExit):
            W.settings(dict(rel_w5_sources=bad))


def test_gains_from_result_rows(tmp_path):
    table = {("scattered", "leakfree", "all"): 4.4, ("scattered+no_relatives", "stored", "all"): 5.6,
             ("scattered", "leakfree", "nearest_identity=0.2-0.3"): 4.2, ("scattered+no_relatives", "stored", "nearest_identity=0.2-0.3"): 5.4,
             ("scattered", "leakfree", "nearest_identity=0.9-1.0"): 1.0}                       # no partner without relatives
    p = tmp_path / "results.tsv"
    _rows(p, table)
    rows = W.final_rows(str(tmp_path))
    assert rows[("scattered", "leakfree", "all")] == 4.4 and len(rows) == len(table)          # the final rows only
    g = W.gains(rows)
    assert g == pytest.approx({"all": 1.2, "nearest_identity=0.2-0.3": 1.2})


def test_summaries_by_stratum():
    identity = np.array([[1.0, 0.25, np.nan], [1.0, np.nan, np.nan], [1.0, 0.95, 0.4]])
    group = np.array(["bacteria", "eukaryota.fungi", "bacteria"])
    cells = dict(table=np.array([0, 0, 1, 2, 2]), bits=np.array([4.0, 5.0, 6.0, 1.0, 2.0]), none=np.array([0.2, 0.4, 1.0, 0.0, 0.1]),
                 entropy=np.array([1.0, 1.5, 0.0, 0.5, 0.7]), top=np.array([0.5, 0.3, 0.0, 0.9, 0.8]),
                 level=np.array([[0.2, 0.3, 0.5]] * 4 + [[0.0, 0.0, 1.0]]), top_identity=np.array([0.25, 0.25, np.nan, 0.95, 0.95]),
                 top_is_nearest=np.array([True, True, False, True, True]))
    s = W.summarise(cells, dict(identity=identity, coverage=np.ones(identity.shape)), group)
    assert s["all"]["cells"] == 5 and s["all"]["tables"] == 3 and s["all"]["none"] == pytest.approx(0.34) and s["all"]["lam"] == pytest.approx(0.66)
    assert s["nearest_identity=0.2-0.3"]["cells"] == 2 and s["nearest_identity=0.9-1.0"]["top_identity"] == pytest.approx(0.95)
    assert s["nearest_identity=none"]["top_identity"] is None and s["relatives_on_target=no"]["none"] == 1.0
    assert s[SC.BLOCKS[0]]["tables"] == 1 and s[SC.BLOCKS[1]]["tables"] == 2
    assert s["all"]["level"] == pytest.approx([0.16, 0.24, 0.6]) and s["all"]["coverage"] == 1.0 and s["nearest_identity=none"]["coverage"] is None


def test_report_and_gate(tmp_path):
    import rel_fixtures as RF
    run_dir, cfg = RF.rel_run(tmp_path, kind="rel_traineval", stages=["w5_sources"], rel_w5_sources=dict(
        arms=[dict(name="S1", run="/r/a", folder="x")], reference=dict(name="ref", run="/r/b", folder="y"), conditions={"scattered": "leakfree"}))
    folder = os.path.join(run_dir, W.OUT)
    os.makedirs(folder)
    stats = dict(bits=4.6, lam=0.7, none=0.3, entropy=1.2, top=0.5, top_identity=0.4, top_is_nearest=0.6, level=[0.2, 0.3, 0.5], cells=10, tables=2, coverage=0.8)
    report = dict(gpu_hidden=True, per_group=50, conditions={"scattered": "leakfree"}, reference=dict(name="ref", gain={"all": 1.1}),
                  arms=dict(S1=dict(arm="s1", tables_seen=14000, gain={"all": 0.9}, share_of_reference_gain={"all": 0.9 / 1.1},
                                    conditions=dict(scattered=dict(layout="leakfree", tables=2, cells=10, same_cells_as_final=True, strata={"all": stats})))))
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(report, f)
    with open(os.path.join(folder, "report.md"), "w") as f:
        f.write("\n".join(W.markdown(report)))
    with Stage(run_dir, "gates_rel_w5_sources", requires=("pre",)) as st:
        W.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w5_sources.json")))["status"] == "pass"
    assert "| all | 2 | 0.80 | 10 | 4.600 |" in "\n".join(W.markdown(report)) and W.summary(run_dir)["json"]["S1"]["scattered"]["lam"] == 0.7
    report["arms"]["S1"]["conditions"]["scattered"]["aligned_with_final"] = False                  # the readout's cells in another order
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(report, f)
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w5_sources", requires=("pre",)) as st:
            W.gate(st, cfg, run_dir)
    report["arms"]["S1"]["conditions"]["scattered"]["aligned_with_final"] = True
    report["arms"]["S1"]["conditions"]["scattered"]["same_cells_as_final"] = False                 # other cells than the final evaluation's
    with open(os.path.join(folder, "report.json"), "w") as f:
        json.dump(report, f)
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w5_sources", requires=("pre",)) as st:
            W.gate(st, cfg, run_dir)


def test_gains_cell_by_cell(tmp_path):
    """The per-cell files of the final evaluations: the gain of the arm and of the reference arm on the same cells,
    per bin of the nearest identity; the readout's weights by band of the reference arm's gain."""
    identity = np.array([[1.0, 0.25], [1.0, 0.95], [1.0, np.nan]])
    table = np.array([0, 0, 1, 2])

    def write(folder, with_rel, without):
        os.makedirs(folder)
        np.savez(os.path.join(folder, "heldout_final.npz"), bits_scattered=np.array(with_rel), table_scattered=table,
                 bits_scattered__no_relatives=np.array(without), table_scattered__no_relatives=table)
    write(tmp_path / "arm", [4.0, 5.0, 1.0, 6.0], [5.0, 5.5, 4.0, 6.0])               # gains 1.0, 0.5, 3.0, 0.0
    write(tmp_path / "ref", [3.0, 4.0, 0.5, 6.0], [5.0, 5.5, 4.0, 6.1])               # gains 2.0, 1.5, 3.5, 0.1
    arm, ref = W.final_cells(str(tmp_path / "arm")), W.final_cells(str(tmp_path / "ref"))
    assert set(arm) == {"scattered", "scattered+no_relatives"} and np.array_equal(arm["scattered"][1], table)
    got = W.gains_by_cell(arm, ref, dict(identity=identity, coverage=np.ones(identity.shape)))["scattered"]
    assert got["all"]["cells"] == 4 and got["all"]["arm"] == pytest.approx(1.125) and got["all"]["reference"] == pytest.approx(1.775)
    assert got["nearest_identity=0.2-0.3"]["ratio"] == pytest.approx(0.75 / 1.75) and got["nearest_identity=0.9-1.0"]["arm_at_least"] == 0.0
    assert got["nearest_identity=none"]["cells"] == 1
    read = dict(none=np.array([0.1, 0.3, 0.0, 1.0]), entropy=np.array([1.0, 1.0, 0.5, 0.0]))
    bands = W.by_reference_gain(read, np.array([2.0, 1.5, 3.5, 0.1]), np.array([1.0, 0.5, 3.0, 0.0]))
    assert bands["reference gain -inf to 0.25"]["none"] == 1.0 and bands["reference gain 1.0 to 3.0"]["cells"] == 2
    assert bands["reference gain 3.0 to inf"]["lam"] == 1.0
    bad = dict(arm, **{"scattered+no_relatives": (arm["scattered+no_relatives"][0], table[::-1])})
    with pytest.raises(SystemExit):                                                    # the two conditions on other cells
        W.cell_gain(bad, "scattered", "scattered+no_relatives")
