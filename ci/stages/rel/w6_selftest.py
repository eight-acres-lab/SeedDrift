"""Stage rel_w6_selftest (package W6; pipeline rel_traineval): the self-tests of the torch code, in the model
interpreter with the GPUs hidden. `python -m ci.stages.rel.w6_selftest RUN_DIR`.

No torch code runs in the CI environment or on GitHub, so this is where it is tested: every test_ function of the
modules listed (profile block rel_w6_selftest: modules, dotted names or paths of files in the repository; threads;
wall_s) is called by the trainer's selftest mode. Outputs under data/rel_w6/selftest/: selftest.json (every test with
passed or the end of its traceback), selftest.log, report.json (per module: tests, passed, the names of the failed).
The stage writes its report whatever the tests say; gate G4_rel_w6_selftest fails when a test failed, a module did not
load, no test ran, or a GPU was visible.
"""
from __future__ import annotations

import json
import os
import subprocess

from ci import ROOT
from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common, w6_tools

OUT = os.path.join("data", "rel_w6", "selftest")
MODULES = ["seeddrift.train.rel.models.axial", "tests/ci/test_rel_w6_data.py", "tests/ci/test_rel_w6_torch.py"]


def settings(cfg) -> dict:
    sc = cfg.get("rel_w6_selftest") or {}
    unknown = set(sc) - {"modules", "threads", "wall_s"}
    if unknown:
        raise SystemExit("rel_w6_selftest: unknown keys %s" % sorted(unknown))
    out = dict(modules=list(sc.get("modules") or MODULES), threads=int(sc.get("threads", 2)), wall_s=int(sc.get("wall_s", 900)))
    for m in out["modules"]:
        if not isinstance(m, str) or m.startswith("/") or ".." in m:
            raise SystemExit("rel_w6_selftest: a module is a dotted name or a path inside the repository: %r" % (m,))
    return out


def build(st, cfg, run_dir):
    s = settings(cfg)
    py, lock, have = w6_tools.interpreter(st, cfg)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    result, conf, log = os.path.join(folder, "selftest.json"), os.path.join(folder, "selftest.conf.json"), os.path.join(folder, "selftest.log")
    write_json(conf, dict(modules=[os.path.join(ROOT, m) if m.endswith(".py") else m for m in s["modules"]], out=result, threads=s["threads"]))
    proc, lg = w6_tools.start(py, "selftest", conf, log, w6_tools.environment(s["threads"], hide_gpu=True))
    try:
        code = proc.wait(timeout=s["wall_s"])
    except subprocess.TimeoutExpired:
        code = "timeout after %d s" % s["wall_s"]
    w6_tools.stop(proc, lg)
    report = dict(exit=code, python=py, python_lock=lock, model_env_sha256=sha256_json(have), modules={}, gpu_hidden=None, torch=None, failed=None)
    if os.path.exists(result):
        with open(result) as f:
            got = json.load(f)
        report.update(gpu_hidden=got.get("gpu_hidden"), torch=got.get("torch"), failed=got.get("failed"))
        for name in s["modules"]:
            tests = (got.get("modules") or {}).get(os.path.join(ROOT, name) if name.endswith(".py") else name)
            if tests is None:
                continue
            report["modules"][name] = dict(tests=len(tests), passed=sum(1 for t in tests.values() if t["passed"]),
                                           failed={k: t.get("error", "")[-600:] for k, t in tests.items() if not t["passed"]})
        st.output("selftest", result)
    else:
        report["error"] = "the self-test wrote no result: %s" % w6_tools.tail(log)
    p = os.path.join(folder, "report.json")
    write_json(p, report)
    st.output("report", p)
    st.extra.update(selftest={k: "%d of %d" % (v["passed"], v["tests"]) for k, v in report["modules"].items()}, selftest_exit=code)


def problems(report, wanted) -> list:
    out = []
    if report.get("error"):
        out.append(dict(error=report["error"][-800:]))
    if report.get("gpu_hidden") is not True:
        out.append(dict(error="the GPUs were not hidden from the self-test"))
    for m in wanted:
        r = (report.get("modules") or {}).get(m)
        if r is None:
            out.append(dict(module=m, error="not run"))
        elif not r["tests"] or r["passed"] != r["tests"]:
            out.append(dict(module=m, error="%d of %d tests passed" % (r["passed"], r["tests"]), failed=r["failed"]))
    if report.get("exit") != 0 and not out:
        out.append(dict(error="the self-test ended with %r" % (report.get("exit"),)))
    return out


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    with open(p) as f:
        report = json.load(f)
    bad = problems(report, settings(cfg)["modules"])
    require(GateResult("G4_rel_w6_selftest", "fail" if bad else "pass", dict(problems=bad[:20], modules={k: [v["passed"], v["tests"]] for k, v in report["modules"].items()},
                                                                         torch=report.get("torch"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    with open(os.path.join(run_dir, OUT, "report.json")) as f:
        report = json.load(f)
    mods = report["modules"]
    return dict(json=dict(torch=report.get("torch"), gpu_hidden=report.get("gpu_hidden"), modules={k: dict(tests=v["tests"], passed=v["passed"], failed=sorted(v["failed"])) for k, v in mods.items()}),
                md=["torch %s in the model interpreter, GPUs hidden: %s" % (report.get("torch"), report.get("gpu_hidden")), "", "| module | tests | passed | failed |", "|---|---|---|---|"]
                + ["| %s | %d | %d | %s |" % (k, v["tests"], v["passed"], ", ".join(sorted(v["failed"])) or "-") for k, v in mods.items()])


def main():
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w6_selftest") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
