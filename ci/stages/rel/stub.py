"""A placeholder stage of the relational-model line: `python -m ci.stages.rel.stub RUN_DIR STAGE`.

It fails, with a stage report, so that a profile listing a stage whose package has not written it yet stops there
instead of passing. The stub rules in ci/rules/rel_<package>.smk call it; configs/rel/kinds.yaml marks those stages
`stub: true`.
"""
from __future__ import annotations

import sys

from ci.stages.common import Stage


def main():
    run_dir, stage = sys.argv[1], sys.argv[2]
    with Stage(run_dir, "rel_" + stage, requires=("pre",)) as st:
        st.extra["stub"] = True
        raise SystemExit("stage %s is a stub: its package has not written it yet (configs/rel/kinds.yaml)" % stage)


if __name__ == "__main__":
    main()
