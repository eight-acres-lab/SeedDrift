import copy

from ci.gates import g4_report

COMMIT = "db0e24b90550a4aae7db62cbf7fba428da687074"


def _report(released_status="computed"):
    rows = []
    for m in g4_report.METHODS:
        r = dict(method=m, auroc=0.7, recall_at_lowest_fpr=0.3, status="computed", genes_sha256="g1")
        if m == "released_v0.1.0":
            r["status"] = released_status
            if released_status == "computed":
                r["provenance"] = dict(tag="v0.1.0", commit=COMMIT)
            else:
                r["auroc"] = r["recall_at_lowest_fpr"] = None
        rows.append(r)
    return dict(scenarios=[dict(name="s1", genes_sha256="g1", rows=rows)])


def test_complete_report_passes():
    assert g4_report.check(_report(), "smoke", (), COMMIT).status == "pass"


def test_tiny_may_skip_the_released_screen_only_when_declared():
    rep = _report("skipped: profile ci-tiny skips the released screen")
    assert g4_report.check(rep, "tiny", ("released_v0.1.0",), COMMIT).status == "pass"
    assert g4_report.check(rep, "tiny", (), COMMIT).status == "fail"


def test_released_skipped_in_smoke_fails():
    rep = _report("skipped: no")
    assert g4_report.check(rep, "smoke", ("released_v0.1.0",), COMMIT).status == "fail"


def test_missing_row_or_other_gene_set_fails():
    rep = _report()
    rep["scenarios"][0]["rows"] = rep["scenarios"][0]["rows"][:-1]
    assert g4_report.check(rep, "smoke", (), COMMIT).status == "fail"
    rep = _report()
    rep["scenarios"][0]["rows"][2]["genes_sha256"] = "g2"
    assert g4_report.check(rep, "smoke", (), COMMIT).status == "fail"


def test_released_not_from_the_tag_commit_fails():
    rep = _report()
    assert g4_report.check(rep, "smoke", (), "0" * 40).status == "fail"
    assert g4_report.check(rep, "smoke", (), "").status == "fail"


def test_model_not_scored_or_empty_report_fails():
    rep = _report()
    rep["scenarios"][0]["rows"][0].update(status="not_applicable (no score)", auroc=None, recall_at_lowest_fpr=None)
    assert g4_report.check(rep, "smoke", (), COMMIT).status == "fail"
    assert g4_report.check(dict(scenarios=[]), "smoke", (), COMMIT).status == "fail"
    rep = copy.deepcopy(_report())
    rep["scenarios"][0]["rows"][1]["auroc"] = None                     # computed without metrics
    assert g4_report.check(rep, "smoke", (), COMMIT).status == "fail"
