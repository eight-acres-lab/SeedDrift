"""Run the classical baselines over a held-out table set under the evaluation conditions, and write result rows.

fit_all takes training tables, evaluate takes held-out tables; neither reads anything but the arrays it is given.
"""
from __future__ import annotations

import time

import numpy as np

from seeddrift.train.rel import api
from seeddrift.train.rel.baselines import classical as cl
from seeddrift.train.rel.baselines import fit as fitting
from seeddrift.train.rel.baselines import switch
from seeddrift.train.rel.evalh import conditions as cond
from seeddrift.train.rel.evalh import score

VERSION = 1
AUTHOR = "w5"
REFERENCE = "own_freq"                      # bits_minus_baseline of every row is against it: what the relatives buy
MASKS = (cond.SCATTERED, cond.WHOLE_GENE)


def fit_all(tables, table_ids, seed, base=4, chain_tables=1500, with_chain=True, log=None):
    """{mask: numbers of every baseline}, from training tables. The chain is fitted on the first chain_tables of them."""
    out = {}
    table_ids = np.asarray(table_ids)
    for mask in MASKS:
        t0 = time.time()
        p = fitting.fit(tables, table_ids, mask, seed, base)
        if with_chain:
            k = min(int(chain_tables), table_ids.shape[0])
            p["chain"] = switch.fit(fitting.take(tables, 0, k), table_ids[:k], mask, seed, p, base)
        p["fit"]["seconds"] = round(time.time() - t0, 1)
        out[mask] = p
        if log:
            log("fitted %s: alpha %s beta %s eps %s (%d cells, %.0f s)" % (mask, p["alpha"], p["beta"], ["%.4f" % e for e in p["eps"]], p["fit"]["cells"], p["fit"]["seconds"]))
    return out


def names(with_chain=True):
    return cl.NAMES + ((switch.NAME,) if with_chain else ())


def n_params(name):
    return switch.N_PARAMS if name == switch.NAME else fitting.N_PARAMS[name]


def evaluate(tables, table_ids, params, conditions=cond.CONDITIONS, seed=cond.SEED, base=4, with_chain=True, log=None):
    """Per condition: dict(items_sha256, cell_table (C,) index of the table of every scored cell, bits {name: (C,)},
    seconds {name}, tables)."""
    table_ids = np.asarray(table_ids)
    out = {}
    for condition in conditions:
        mask, rows_mode = cond.parse(condition)
        p = params[mask]
        digest = cond.ItemsDigest(condition, seed)
        which = [n for n in names(with_chain) if rows_mode != cond.NO_RELATIVES or n not in cl.USES_RELATIVES + (switch.NAME,)]
        bits = {n: [] for n in which}
        seconds = {n: 0.0 for n in which}
        cell_table = []
        for a, _, m, bt, truth in fitting.batches(tables, table_ids, condition, seed, base):
            digest.add(m)
            cell_table.append(a + bt.cells[0])
            for n in which:
                t0 = time.time()
                P = switch.predict(bt, p) if n == switch.NAME else cl.predict(n, bt, p)
                cl.check_normalised(P)
                bits[n].append(cl.bits_of(P, truth))
                seconds[n] += time.time() - t0
        out[condition] = dict(
            items_sha256=digest.hexdigest(), tables=int(table_ids.shape[0]), seconds=seconds,
            cell_table=np.concatenate(cell_table) if cell_table else np.zeros(0, dtype=np.int64),
            bits={n: (np.concatenate(v) if v else np.zeros(0)) for n, v in bits.items()},
        )
        if log:
            log("%s: %d cells of %d tables" % (condition, out[condition]["cell_table"].shape[0], table_ids.shape[0]))
    return out


def result_rows(results, group, identity, params, head, n_groups=12):
    """Rows of the result table (the columns of the contract). head: run, commit, seed, table_set, table_kind and
    versions. One row per baseline x condition x group x stratum that has cells."""
    stratum = score.identity_stratum(identity)
    rows = []
    for condition, res in results.items():
        mask = cond.parse(condition)[0]
        agg = {n: score.aggregate(b, res["cell_table"], group, stratum, n_groups) for n, b in res["bits"].items()}
        ref = {(r["group"], r["stratum"]): r["bits"] for r in agg.get(REFERENCE, [])}
        for n, rws in agg.items():
            cells = int(res["cell_table"].shape[0])
            for r in rws:
                key = (r["group"], r["stratum"])
                rows.append(score.result_row(
                    run=head.get("run", ""), commit=head.get("commit", ""), variant=n, author=AUTHOR,
                    module="baselines." + ("switch" if n == switch.NAME else "classical"), params=n_params(n),
                    seed=head.get("seed", ""), table_set=head.get("table_set", ""), table_kind=head.get("table_kind", ""),
                    items_sha256=res["items_sha256"], item_half="", layout=api.LAYOUT_STORED, budget=0, weights="",
                    cells_seen=int(params[mask]["fit"]["cells"]), target_cells=r["target_cells"],
                    cells_per_s=round(cells / max(res["seconds"][n], 1e-9), 1), peak_mb="", condition=condition,
                    group=r["group"], tier=score.ALL, stratum=r["stratum"], passes=1, bits=r["bits"], baseline=REFERENCE,
                    bits_minus_baseline=(r["bits"] - ref[key]) if key in ref else "", n_items=r["n_items"], status="ok",
                    versions=head.get("versions", ""),
                ))
    return rows


def summary(rows):
    """{condition: {baseline: bits}} over all groups, and {condition: {stratum: {baseline: bits}}}."""
    main, strata = {}, {}
    for r in rows:
        if r["group"] != score.ALL:
            continue
        if r["stratum"] == score.ALL:
            main.setdefault(r["condition"], {})[r["variant"]] = r["bits"]
        else:
            strata.setdefault(r["condition"], {}).setdefault(r["stratum"], {})[r["variant"]] = r["bits"]
    return main, strata


def markdown(rows, title="Classical baselines: bits per codon", groups=None):
    """The tables a reader wants first: baselines x conditions, baselines x identity of the nearest row, and
    baselines x group (groups: the names the group indices of the rows stand for)."""
    main, strata = summary(rows)
    cells = {}
    for r in rows:
        if r["group"] == score.ALL and r["variant"] == REFERENCE:
            cells[(r["condition"], r["stratum"])] = (r["target_cells"], r["n_items"])
    order = [n for n in names(True) if any(n in v for v in main.values())]
    conds = [c for c in cond.CONDITIONS if c in main]
    out = ["# " + title, "", "Mean over the groups of the per-group bits per codon; row 0, target block, stored layout (an upper", "bound where the whole gene is hidden).", ""]
    out += ["| baseline | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for n in order:
        out.append("| %s | " % n + " | ".join("%.3f" % main[c][n] if n in main[c] else "" for c in conds) + " |")
    out += ["", "| scored | " + " | ".join("%d cells, %d tables" % cells.get((c, score.ALL), (0, 0)) for c in conds) + " |", ""]
    for c in conds:
        if c not in strata or cond.parse(c)[1] == cond.NO_RELATIVES:      # without relatives: the own_freq column of the base condition
            continue
        mine = [n for n in order if n in main[c]]
        out += ["## %s, by the identity of the nearest row at the target locus" % c, "", "| nearest identity | cells | tables | " + " | ".join(mine) + " |", "|---|---|---|" + "---|" * len(mine)]
        for s in score.STRATA:
            if s not in strata[c]:
                continue
            nc, nt = cells.get((c, s), (0, 0))
            out.append("| %s | %d | %d | " % (s.split("=")[1], nc, nt) + " | ".join("%.3f" % strata[c][s][n] if n in strata[c][s] else "" for n in mine) + " |")
        out.append("")
    per_group = {}
    for r in rows:
        if r["group"] != score.ALL and r["stratum"] == score.ALL:
            per_group.setdefault(r["condition"], {}).setdefault(r["group"], {})[r["variant"]] = (r["bits"], r["target_cells"])
    for c in conds:
        if c not in per_group or cond.parse(c)[1] == cond.NO_RELATIVES:
            continue
        mine = [n for n in order if n in main[c]]
        out += ["## %s, by group" % c, "", "| group | cells | " + " | ".join(mine) + " |", "|---|---|" + "---|" * len(mine)]
        for g in sorted(per_group[c], key=int):
            name = groups[int(g)] if groups and int(g) < len(groups) else g
            row = per_group[c][g]
            out.append("| %s | %d | " % (name, row[REFERENCE][1] if REFERENCE in row else 0) + " | ".join("%.3f" % row[n][0] if n in row else "" for n in mine) + " |")
        out.append("")
    return "\n".join(out) + "\n"
