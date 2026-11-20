"""The register of the island reference set for benchmark v2 (ci/stages/register.py) and gate G3_register."""
import csv
import io
import json
import os
import sys
import tarfile

import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from ci.hashing import sha256_file  # noqa: E402
from ci.stages import register as RG  # noqa: E402


def _tables(rows_islands, rows_backbone):
    """An archive like Sup_tables.tar.gz: SupTable1 and SupTable2 with Accession, Organism and coordinates."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, rows in (("SupTable1.txt", rows_islands), ("SupTable2.txt", rows_backbone)):
            text = "GI_ID\tAccession\tOrganism\tStart\tEnd\r\n" + "".join("%d\t%s\t%s\t%d\t%d\r\n" % (i, a, o, 100 * i, 100 * i + 50) for i, (a, o) in enumerate(rows))
            data = text.encode()
            info = tarfile.TarInfo("Sup_tables/" + name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_reference_accessions_reads_names_only(tmp_path):
    rows = [("NC_1.1", "Alpha beta 1"), ("NC_1.1", "Alpha beta 1"), ("NC_2.1", "Gamma delta 2")]
    p = tmp_path / "t.tar.gz"
    p.write_bytes(_tables(rows, [("NC_2.1", "Gamma delta 2"), ("NC_3.1", "Eps zeta")]))
    got, shas = RG.reference_accessions(str(p), sha256_file(str(p), use_sidecar=False))
    assert got == {"NC_1.1": "Alpha beta 1", "NC_2.1": "Gamma delta 2", "NC_3.1": "Eps zeta"} and set(shas) == set(RG.TABLES)
    # invariant to row multiplicity and coordinates (no row is counted)
    q = tmp_path / "u.tar.gz"
    q.write_bytes(_tables(rows[:1] + rows[2:], [("NC_3.1", "Eps zeta"), ("NC_2.1", "Gamma delta 2")] * 3))
    assert RG.reference_accessions(str(q), sha256_file(str(q), use_sidecar=False))[0] == got
    with pytest.raises(SystemExit, match="not the archive"):
        RG.reference_accessions(str(p), "0" * 64)


def test_path_levels_and_name_scan(tmp_path):
    assert RG.path_level("genomes/NC_1.1.gb") == "L3" and RG.path_level("panel/NC_1.1/panel.tsv") == "L3"
    assert RG.path_level("dev24_v/genus/scores_NC_1.1.tsv") == "L4" and RG.path_level("runs/x/data/islands/raw/v/NC_1.1.npz") == "L4"
    assert RG.path_level("runs/x/data/islands/eval/NC_1.1.json") == "L5"
    (tmp_path / "a" / "NC_000123.2").mkdir(parents=True)
    (tmp_path / "a" / "NC_0001234.1.txt").write_text("")            # another accession (a longer number): no match
    (tmp_path / "a" / "x_NC_000123.tsv").write_text("")               # without its version
    found, walked = RG.scan_names([str(tmp_path / "a")], ["NC_000123.2"])
    assert walked == 3 and sorted(os.path.basename(p) for p, _ in found["NC_000123.2"]) == ["NC_000123.2", "x_NC_000123.tsv"]


def test_parts_and_levels():
    org = {"D1": "Escherichia coli", "D2": "Shigella x", "S1": "Bacillus y", "NC_010167.1": "Brucella suis", "R1": "Bacillus z",
           "R2": "Brucella abortus", "NC_009504.1": "Brucella ovis", "U1": "Yersinia q"}
    accs = sorted(org)
    part = RG.parts(accs, org, dev={"D1", "D2"}, sealed={"S1", "NC_010167.1"}, dev_genera={"Escherichia", "Shigella", "Yersinia"})
    assert part == {"D1": "development", "D2": "development", "S1": "sealed_v1", "NC_010167.1": "sealed_secondary", "R1": "rest", "R2": "rest",
                    "NC_009504.1": "rest", "U1": "development_unlisted"}
    declared = [dict(selector="all", level="L0", source="a"), dict(selector="rest", level="L1", source="b"),
                dict(selector="NC_009504.1", level="L2", source="c")]
    lv = RG.levels(accs, part, declared)
    assert lv["R1"] == ("L1", ["b"]) and lv["NC_009504.1"] == ("L2", ["c"]) and lv["D1"] == ("L0", ["a"])
    with pytest.raises(SystemExit, match="selector"):
        RG.levels(accs, part, [dict(selector="NC_404.1", level="L2", source="x")])


def _setup(tmp_path):
    """A store, a taxonomy and a run whose island tables hold the repository's 24 development and 18 sealed chromosomes
    with 62 invented others: 61 in sealed_v1 genera and NC_009504.1 (Brucella)."""
    with open(os.path.join(ROOT, "bench/islands/results/dev_genomes.tsv")) as f:
        dev = [(r["accession"], r["organism"]) for r in csv.DictReader(f, delimiter="\t")]
    with open(os.path.join(ROOT, "bench/islands/results/genomes.tsv")) as f:
        sel = [(r["accession"], r["organism"]) for r in csv.DictReader(f, delimiter="\t")]
    genera = sorted({o.split()[0] for a, o in sel if a != RG.SECONDARY})
    rest = [("NC_9%05d.1" % i, "%s sp. %d" % (genera[i % len(genera)], i)) for i in range(61)] + [("NC_009504.1", "Brucella ovis ATCC 25840")]
    allrows = dev + sel + rest
    store = tmp_path / "store"
    (store / "panel").mkdir(parents=True)
    tar = _tables(allrows[:50], allrows[30:])
    (store / "Sup_tables.tar.gz").write_bytes(tar)
    names = sorted({o.split()[0] for _, o in allrows})
    gid = {g: 1000 + i for i, g in enumerate(names)}
    taxid = {a: 50000 + i for i, (a, _) in enumerate(allrows)}
    (store / "panel" / "island_set_taxa.json").write_text(json.dumps(taxid))
    tax = tmp_path / "tax"
    tax.mkdir()
    nodes = ["1\t|\t1\t|\tno rank\t|"] + ["%d\t|\t1\t|\tgenus\t|" % v for v in gid.values()] + \
            ["%d\t|\t%d\t|\tspecies\t|" % (taxid[a], gid[o.split()[0]]) for a, o in allrows]
    (tax / "nodes.dmp").write_text("\n".join(nodes) + "\n")
    (tax / "names.dmp").write_text("\n".join("%d\t|\t%s\t|\t\t|\tscientific name\t|" % (v, g) for g, v in gid.items()) + "\n")
    run = tmp_path / "run"
    (run / "gates").mkdir(parents=True)
    (run / "gates" / "pre.ok").write_text("{}")
    cfg = dict(pipeline="register", taxdump=str(tax), register=dict(store=str(store), expect=104))
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    return run, cfg, store


def test_register_stage_and_gate(tmp_path, monkeypatch):
    from ci.stages import gates as GA
    run, cfg, store = _setup(tmp_path)
    real = RG.reference_accessions
    monkeypatch.setattr(RG, "reference_accessions", lambda path, sha: real(path, sha256_file(path, use_sidecar=False)))   # the fixture's archive
    monkeypatch.setattr(sys, "argv", ["x", str(run)])
    RG.main()
    rep = json.loads((run / "data" / "register" / "report.json").read_text())
    assert rep["parts"] == {"development": 24, "rest": 62, "sealed_secondary": 1, "sealed_v1": 17}
    assert rep["status"]["candidate"] == 61 and rep["status"]["demoted"] == 1
    with open(run / "data" / "register" / "reference_set.tsv") as f:
        rows = {r["accession"]: r for r in csv.DictReader(f, delimiter="\t")}
    assert rows["NC_009504.1"]["status"] == "demoted" and rows["NC_009504.1"]["level"] == "L2"
    assert rows["NC_900000.1"]["level"] == "L1" and rows["NC_900000.1"]["status"] == "candidate" and rows["NC_900000.1"]["genus"]
    st = type("St", (), dict(repo=staticmethod(lambda rel: rel if os.path.isabs(rel) else os.path.join(ROOT, rel))))
    GA.phase_register(st, cfg, str(run))
    assert json.loads((run / "gates" / "G3_register.json").read_text())["status"] == "pass"
    # a candidate edited into the list fails the gate
    cp = run / "data" / "register" / "candidates.tsv"
    cp.write_text(cp.read_text() + "NC_008253.1\tEscherichia coli 536\tEscherichia\tEnterobacteriaceae\n")
    with pytest.raises(Exception):
        GA.phase_register(st, cfg, str(run))
    g = json.loads((run / "gates" / "G3_register.json").read_text())
    assert g["status"] == "fail" and any("disagree" in p["error"] or "differs" in p["error"] for p in g["evidence"]["problems"])
    assert rep["scan"]["paths_walked"] > 0 and rep["scan"]["rest_findings"] == {}
    # the names-only scan: a development chromosome's files are expected; a rest chromosome's record stops the stage
    (store / "dev_genomes").mkdir()
    (store / "dev_genomes" / "NC_008253.1.gb").write_text("LOCUS")
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg))
    RG.main()
    (store / "genomes").mkdir()
    (store / "genomes" / "NC_900005.1.gb").write_text("LOCUS")
    with pytest.raises(SystemExit, match="exposures the scan found"):
        RG.main()
    (store / "genomes" / "NC_900005.1.gb").unlink()
    (store / "dev24_v" / "genus").mkdir(parents=True)
    (store / "dev24_v" / "genus" / "scores_NC_009504.tsv").write_text("x")        # L4 above NC_009504.1's declared L2
    with pytest.raises(SystemExit, match="NC_009504.1"):
        RG.main()
    (store / "dev24_v" / "genus" / "scores_NC_009504.tsv").unlink()
    # a table without 104 chromosomes stops the stage
    cfg2 = dict(cfg, register=dict(cfg["register"], expect=103))
    (run / "config.resolved.yaml").write_text(yaml.safe_dump(cfg2))
    with pytest.raises(SystemExit, match="expected 103"):
        RG.main()
