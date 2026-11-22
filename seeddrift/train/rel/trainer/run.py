"""Entry of the relational line's trainer, run with the model interpreter from the root of the worktree:

  python -m seeddrift.train.rel.trainer.run selftest  CONFIG_JSON   every test_ function of the modules listed
  python -m seeddrift.train.rel.trainer.run preflight CONFIG_JSON   a variant before its training (loop.preflight)
  python -m seeddrift.train.rel.trainer.run fit       CONFIG_JSON   the training with its held-out evaluation (loop.fit)

selftest: CONFIG_JSON holds modules (dotted names, or paths of files ending in .py), out (the report it writes) and
threads. The GPUs are hidden by whoever starts it (CUDA_VISIBLE_DEVICES empty); a visible GPU fails the self-test.
The report lists every test with passed or the end of its traceback; the exit code is 1 when a test failed.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import time
import traceback


def load_module(name):
    if name.endswith(".py"):
        spec = importlib.util.spec_from_file_location("selftest_" + os.path.basename(name)[:-3], name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(name)


def selftest(cfg):
    import torch
    torch.set_num_threads(int(cfg.get("threads", 2)))
    report = dict(torch=torch.__version__, python=sys.version.split()[0], gpu_hidden=not torch.cuda.is_available(), modules={})
    failed = 0 if report["gpu_hidden"] else 1
    for name in cfg["modules"]:
        tests = {}
        try:
            mod = load_module(name)
            fns = sorted(k for k, v in vars(mod).items() if k.startswith("test_") and callable(v))
            if not fns:
                raise AssertionError("no test_ function in %s" % name)
        except BaseException:  # noqa: BLE001 - a module that does not load is a failed self-test, not a crash
            tests["import"] = dict(passed=False, error=traceback.format_exc()[-1500:])
            fns = []
        for fn in fns:
            t = time.time()
            try:
                getattr(mod, fn)()
                tests[fn] = dict(passed=True, seconds=round(time.time() - t, 2))
            except BaseException:  # noqa: BLE001
                tests[fn] = dict(passed=False, seconds=round(time.time() - t, 2), error=traceback.format_exc()[-1500:])
        failed += sum(not t["passed"] for t in tests.values())
        report["modules"][name] = tests
        print("%s: %d of %d passed" % (name, sum(t["passed"] for t in tests.values()), len(tests)), flush=True)
    report["failed"] = failed
    with open(cfg["out"] + ".part", "w") as f:
        json.dump(report, f, indent=1, sort_keys=True)
    os.replace(cfg["out"] + ".part", cfg["out"])
    return 1 if failed else 0


def main(argv):
    if len(argv) != 3 or argv[1] not in ("selftest", "preflight", "fit"):
        raise SystemExit(__doc__)
    with open(argv[2]) as f:
        cfg = json.load(f)
    if argv[1] == "selftest":
        return selftest(cfg)
    from seeddrift.train.rel.trainer import loop
    out = loop.preflight(cfg) if argv[1] == "preflight" else loop.fit(cfg)
    print(json.dumps({k: out[k] for k in sorted(out) if k not in ("evals", "memory")}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
