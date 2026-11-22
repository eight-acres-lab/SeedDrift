"""Stage rel_w13_census (ci/stages/rel/w13_census.py): the close relatives of the corpus genomes in a selection's
candidate table, on a synthetic selection run whose relatives are known, over the synthetic source run of the table
stages (tests/ci/test_rel_w2_stage.py)."""
import csv
import json
import os

import pytest

import rel_fixtures as RF
import test_rel_w2_stage as TS
from ci.gates import GateFailure
from ci.hashing import sha256_file
from ci.stages.common import Stage
from ci.stages.rel import w13_census as WC

COLS = ["id", "group", "catalogue", "taxid", "host_taxid", "species_taxid", "order", "genus", "accession", "units", "proteins_central", "proteins_level",
        "out", "selected", "pick"]


def make_selection(tmp, metas):
    """A passed selection run: every corpus genome selected; relatives of the first open genome of every group: two of its
    species (one eligible as not needed, one as another genome of the species), one excluded by lineage, one of its genus;
    and a relative of a test genome. Returns (run, corpus spec, the true counts {anchor: (same species, same genus)})."""
    run = tmp / "select-run"
    (run / "data" / "select").mkdir(parents=True)
    rows, truth, seen = [], {}, set()
    for m in metas:
        rows.append(dict(id=m["id"], group=m["group"], catalogue="c", taxid=m["taxid"], host_taxid=0, species_taxid=m["ranks"][5], order=m["ranks"][2],
                         genus=m["ranks"][4], accession="", units=1, proteins_central=900, proteins_level="", out="", selected=1, pick=1))
    for m in metas:
        first = m["split"] in ("train", "heldout") and m["group"] not in seen and m["id"] != TS.LEFT_OUT
        if m["split"] == "test" or first:
            seen.add(m["group"]) if first else None
            base = dict(group=m["group"], catalogue="c", taxid=m["taxid"], host_taxid=0, order=m["ranks"][2], genus=m["ranks"][4], accession="", units=1,
                        proteins_central=3000, proteins_level="", pick="")
            rows += [dict(base, id=m["id"] + "_sp1", species_taxid=m["ranks"][5], out="not needed for the target (pick 9)", selected=0),
                     dict(base, id=m["id"] + "_sp2", species_taxid=m["ranks"][5], out="another genome of the species (%s) preferred" % m["id"], selected=0),
                     dict(base, id=m["id"] + "_sp3", species_taxid=m["ranks"][5], out="excluded: sealed genus X", selected=0),
                     dict(base, id=m["id"] + "_ge1", species_taxid=m["ranks"][5] + 5000, out="not needed for the target (pick 4)", selected=0)]
            if first:
                truth[m["id"]] = (2, 1)
    for name, rs in (("candidates.tsv", rows), ("selected.tsv", [r for r in rows if r["selected"] == 1])):
        with open(run / "data" / "select" / name, "w") as f:
            w = csv.DictWriter(f, COLS, delimiter="\t")
            w.writeheader()
            w.writerows(rs)
    (run / "record.json").write_text(json.dumps(dict(run_id="sel", status="passed", pipeline="select", git=dict(sha="2" * 40))))
    (run / "config.resolved.yaml").write_text("catalog: {mode: lock, lock: configs/select/v1.lock.yaml}\nselect_spec: configs/select/medium.yaml\n")
    spec = tmp / "corpus.yaml"
    spec.write_text("# Corpus x, emitted from the selection of run sel (\n# selected.tsv sha256 %s). Edit the selection.\nname: x\n"
                    % sha256_file(str(run / "data" / "select" / "selected.tsv"), use_sidecar=False))
    return str(run), str(spec), truth


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("w13")
    src = str(tmp / "ci-root" / "runs" / "20261120T000000-src")
    os.makedirs(src)
    metas = TS.make_source(src)
    sel, spec, truth = make_selection(tmp, metas)
    run_dir, cfg = RF.rel_run(tmp, kind="rel_corpus", stages=("w13_census",), run_id="20261124T000000-w13", corpus_spec=spec,
                              rel_w13_census=dict(select_from=sel, source=src, expect=TS.SIZES, per_anchor=[1, 3], propose=3))
    with pytest.MonkeyPatch.context() as mp:
        TS.patch(mp, tmp)
        with Stage(run_dir, "rel_w13_census") as st:
            rep = WC.build(st, cfg, run_dir)
    return dict(tmp=tmp, src=src, metas=metas, run_dir=run_dir, cfg=cfg, rep=rep, truth=truth, sel=sel)


def test_relatives_are_counted_after_every_exclusion(built):
    rep, truth, metas = built["rep"], built["truth"], built["metas"]
    split = {m["id"]: m["split"] for m in metas}
    for sp in ("train", "heldout"):
        want = [a for a in truth if split[a] == sp]
        d = rep["counts"][sp]["all"]
        assert d["new_same_species"]["anchors_with_one"] == len(want) and d["new_same_species"]["candidates"] == 2 * len(want)
        assert d["new_same_genus"]["candidates"] == len(want) and d["new_same_species"]["anchors_with_three"] == 0
        assert d["proposals"]["1"]["genomes"] == len(want) and d["proposals"]["3"]["genomes"] == 3 * len(want)
    assert sum(rep["counts"][sp]["all"]["anchors"] for sp in ("train", "heldout")) == 36
    assert set(rep["counts"]["train"]) - {"all"} == {m["group"] for m in metas if m["split"] == "train" and m["id"] != TS.LEFT_OUT}
    euk = [a for a in truth if a.startswith("eukaryota") and split[a] == "train"]
    g = [m["group"] for m in metas if m["id"] == euk[0]][0]
    assert rep["counts"]["train"][g]["proposals"]["3"]["proteins"] == 3 * 1500 * sum(1 for a in euk if [m["group"] for m in metas if m["id"] == a][0] == g)
    text = open(os.path.join(built["run_dir"], WC.OUT, "report.json")).read() + open(os.path.join(built["run_dir"], WC.OUT, "report.md")).read()
    assert not any(x in text for x in TS.SEALED) and "nothing was downloaded" in text
    assert rep["candidates"]["by_group_and_reason"]["bacteria | excluded by lineage"] >= 1


def test_gate_passes_and_fails_when_the_selection_is_not_the_corpus(built, monkeypatch, tmp_path):
    TS.patch(monkeypatch, built["tmp"])
    run_dir, cfg = built["run_dir"], built["cfg"]
    with Stage(run_dir, "gates_rel_w13_census") as st:
        WC.gate(st, cfg, run_dir)
    assert json.load(open(os.path.join(run_dir, "gates", "G4_rel_w13_census.json")))["status"] == "pass"
    other = tmp_path / "other.yaml"
    other.write_text("# selected.tsv sha256 %s\n" % ("0" * 64))
    with pytest.raises(GateFailure):
        with Stage(run_dir, "gates_rel_w13_census") as st:
            WC.gate(st, dict(cfg, corpus_spec=str(other)), run_dir)
    with pytest.raises(SystemExit, match="not the one the corpus spec"):
        with Stage(run_dir, "rel_w13_census") as st:
            WC.build(st, dict(cfg, corpus_spec=str(other)), run_dir)


def test_reason_kinds_and_settings():
    assert WC.reason_kind("") == "selected" and WC.reason_kind("not needed for the target (pick 3)") == "eligible"
    assert WC.reason_kind("excluded: sealed genus Vibrio") == "excluded by lineage" and WC.reason_kind("host unknown (no host record)") == "excluded by a host rule"
    assert WC.reason_kind("a host taxid does not resolve") == "excluded by a host rule" and WC.reason_kind("hosts in more than one class") == "excluded by a host rule"
    assert WC.reason_kind("group conflict: virus 1 (x) is in group a by its taxon but its host 2 is a b") == "group conflict"
    for bad in (dict(source="x"), dict(select_from="x", source="y", colour=1), dict(select_from="x", source="y", per_anchor=[1], propose=3)):
        with pytest.raises(SystemExit):
            WC.settings(dict(rel_w13_census=bad))
