"""Stage fit (pipeline select): the budget against the data disk as it is now (data/select/fit.json).

Free space changes from run to run, so this output is environmental: `python -m ci compare` does not require it to be
identical. For each case of the budget (low, central, high): total, free space, the reserve of gate G7_resources, the
margin, and the largest uniform fraction of every group that would fit (information for the user, never applied: the
charter scales all groups together and forbids training below half of any floor).
"""
from __future__ import annotations

import json
import os
import shutil

from ci.stages.common import Stage, main_args, write_json

CASES = ("low", "central", "high")


def main():
    run_dir = main_args()
    with Stage(run_dir, "fit") as st:
        cfg = st.config
        with open(os.path.join(run_dir, "data", "select", "budget.json")) as f:
            b = json.load(f)
        free = shutil.disk_usage(cfg["data_root"]).free / 1e9
        reserve = float(cfg.get("disk_reserve_gb", 30))
        out = dict(data_root_free_gb=round(free, 1), reserve_gb=reserve, cases={},
                   groups_short_now=sorted(g for g, x in b["groups"].items() if x.get("floor") and x["units"] < x["floor"]))
        for c in CASES:
            d = b["disk"][c]
            total, scalable = d["total_gb"], d["scalable_gb"]
            room = free - reserve - (total - scalable)
            frac = max(0.0, min(1.0, room / scalable)) if scalable > 0 else 1.0
            # at that scale, the smallest share of its floor kept by a group that meets its floor now (groups already short
            # are listed apart: their shortfall is not a disk question)
            floor_part = min((frac * x["units"] / x["floor"] for x in b["groups"].values() if x.get("floor") and x["units"] >= x["floor"]), default=frac)
            out["cases"][c] = dict(total_gb=total, margin_gb=round(free - reserve - total, 1), verdict="fits" if total <= free - reserve else "does not fit",
                                   uniform_fraction_that_fits=round(frac, 3), smallest_fraction_of_a_floor=round(floor_part, 3),
                                   below_half_of_floors=floor_part < 0.5, search_hours=b["search_hours"][c])
        p = st.path("select", "fit.json")
        write_json(p, out)
        st.output("fit", p)
        st.extra.update(**{c: out["cases"][c]["verdict"] for c in CASES})


if __name__ == "__main__":
    main()
