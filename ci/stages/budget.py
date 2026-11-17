"""Stage budget (pipeline select): what the selected corpus would cost, with ranges and the basis of every number.

Rates measured by the CI (data/select/rates.json): from passed runs named in the profile (`calibration: {runs: [run
directories]}`), per group: stored hit bytes per protein (nearest targets plus the best target of every reference
genus, with CIGAR), corpus bytes per protein (genes.tsv, proteins.faa, cds.fna) and the protein file alone (the query
copy of the search); low, central, high = minimum, median, maximum over the run's genomes of the group, with n and
the run ids. A profile without calibration runs (select-tiny) names a committed rates file instead.
Assumptions (configs/select/assumptions.yaml, each with its basis): protein-count factors, stored download bytes per
protein, search time (fixed cost plus a per-protein rate), DIAMOND temporary space, live copies of a run.
Outputs: data/select/{rates.json, budget.json, report.md}, all a function of the selection, the rates and the
assumptions, so two runs give the same bytes. The comparison with the free disk is stage fit (data/select/fit.json).
"""
from __future__ import annotations

import csv
import json
import os

import yaml

from ci.hashing import sha256_file
from ci.kingdom import DEFAULT_MAP, KingdomMap, NoGroup
from ci.stages.common import Stage, main_args, write_json
from ci.taxonomy import Taxonomy, UnknownTaxid

CASES = ("low", "central", "high")
GB = 1e9


def _q(values):
    v = sorted(values)
    return [v[0], v[len(v) // 2], v[-1]] if v else None


def measure(run_dirs, tax, kmap):
    """Per-group rates from passed runs."""
    per = {}
    runs = []
    for d in run_dirs:
        with open(os.path.join(d, "record.json")) as f:
            rec = json.load(f)
        if rec.get("status") != "passed":
            raise SystemExit("calibration run %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
        with open(os.path.join(d, "data", "corpus", "manifest.json")) as f:
            corpus = {e["id"]: e for e in json.load(f)["entries"]}
        with open(os.path.join(d, "data", "hits", "manifest.json")) as f:
            hits = {e["id"]: e for e in json.load(f)["entries"]}
        search = [s for s in rec["stages"] if s["name"] == "search"]
        runs.append(dict(run_id=rec["run_id"], record_sha256=sha256_file(os.path.join(d, "record.json"), use_sidecar=False),
                         search_wall_s=search[0]["wall_s"] if search else None, proteins=sum(int(e.get("proteins", 0)) for e in corpus.values())))
        for gid, e in sorted(corpus.items()):
            n = int(e.get("proteins", 0))
            if not n or gid not in hits:
                continue
            try:
                g = kmap.group(tax, int(e["taxid"]), e.get("source_kind", ""))
            except (NoGroup, UnknownTaxid, KeyError, ValueError):
                continue
            hb = sum(f["bytes"] for f in hits[gid]["files"])
            cb = sum(f["bytes"] for f in e.get("files", []))
            pb = sum(f["bytes"] for f in e.get("files", []) if f["path"].endswith("proteins.faa"))
            p = per.setdefault(g, dict(hits=[], corpus=[], proteins_faa=[], genomes=[]))
            p["hits"].append(hb / n)
            p["corpus"].append(cb / n)
            p["proteins_faa"].append(pb / n)
            p["genomes"].append(gid)
    out = {}
    for g, p in sorted(per.items()):
        out[g] = dict(n=len(p["genomes"]), genomes=sorted(set(p["genomes"])), hits_bytes_per_protein=[round(x, 1) for x in _q(p["hits"])],
                      corpus_bytes_per_protein=[round(x, 1) for x in _q(p["corpus"])], proteins_faa_bytes_per_protein=[round(x, 1) for x in _q(p["proteins_faa"])])
    return dict(basis="measured: passed CI runs %s" % ", ".join(r["run_id"] for r in runs), runs=runs, groups=out)


def _fallback(rates, g, field, assumptions):
    """A group the calibration runs did not have takes the widest range over the measured groups of its major class."""
    gr = rates["groups"].get(g)
    if gr:
        return gr[field], "measured n=%d" % gr["n"]
    pool = [v[field] for k, v in rates["groups"].items() if k.split(".")[0] == g.split(".")[0]] or [v[field] for v in rates["groups"].values()]
    if not pool:
        return assumptions["unmeasured_" + field], "assumed"
    return [min(x[0] for x in pool), sorted(x[1] for x in pool)[len(pool) // 2], max(x[2] for x in pool)], "not measured; range of related groups"


def main():
    run_dir = main_args()
    with Stage(run_dir, "budget") as st:
        cfg = st.config
        with open(st.repo(cfg["select_spec"])) as f:
            spec = yaml.safe_load(f)
        with open(st.repo(spec["assumptions"])) as f:
            A = yaml.safe_load(f)
        cal = cfg.get("calibration") or {}
        tax = Taxonomy(st.repo(cfg["taxdump"]))
        kmap = KingdomMap(st.repo(cfg.get("kingdom_map", DEFAULT_MAP)))
        if cal.get("runs"):
            rates = measure([st.repo(d) for d in cal["runs"]], tax, kmap)
        else:
            with open(st.repo(cal["file"])) as f:
                rates = yaml.safe_load(f)
            rates["basis"] = "committed rates file %s (%s)" % (cal["file"], rates.get("basis", ""))
        rp = st.path("select", "rates.json")
        write_json(rp, rates)
        with open(os.path.join(run_dir, "data", "select", "selected.tsv")) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        with open(os.path.join(run_dir, "data", "select", "by_group.json")) as f:
            bg = json.load(f)
        groups = {}
        tot = {k: {c: 0.0 for c in CASES} for k in ("proteins", "downloads", "corpus", "queries", "hits", "search_s")}
        S = A["search"]
        for g in kmap.names:
            rs = [r for r in rows if r["group"] == g]
            prot = {c: sum(int(r["proteins_" + c]) for r in rs) for c in CASES}
            hb, hbasis = _fallback(rates, g, "hits_bytes_per_protein", A)
            cb, cbasis = _fallback(rates, g, "corpus_bytes_per_protein", A)
            qb, _ = _fallback(rates, g, "proteins_faa_bytes_per_protein", A)
            dl = {c: 0.0 for c in CASES}
            for r in rs:
                f = A["stored_download_bytes_per_protein"][r["source_kind"]]
                for i, c in enumerate(CASES):
                    dl[c] += int(r["proteins_" + c]) * f[i]
            line = dict(genomes=len(rs), units=sum(int(r["units"]) for r in rs), floor=bg["groups"][g]["floor"], target=bg["groups"][g]["target"], proteins=prot,
                        downloads_gb={c: round(dl[c] / GB, 3) for c in CASES},
                        corpus_gb={c: round(prot[c] * cb[i] / GB, 3) for i, c in enumerate(CASES)},
                        hits_gb={c: round(prot[c] * hb[i] / GB, 3) for i, c in enumerate(CASES)},
                        search_h={c: round(prot[c] * S["per_protein_ms_at_12_threads"][i] / 1000 / 3600, 2) for i, c in enumerate(CASES)},
                        basis=dict(hits=hbasis, corpus=cbasis, proteins=sorted({r["proteins_level"] for r in rs})))
            groups[g] = line
            for i, c in enumerate(CASES):
                tot["proteins"][c] += prot[c]
                tot["downloads"][c] += dl[c]
                tot["corpus"][c] += prot[c] * cb[i]
                tot["queries"][c] += prot[c] * qb[i]
                tot["hits"][c] += prot[c] * hb[i]
                tot["search_s"][c] += prot[c] * S["per_protein_ms_at_12_threads"][i] / 1000
        runs = A["live_run_copies"]
        disk = {}
        for i, c in enumerate(CASES):
            fixed = tot["downloads"][c] + tot["queries"][c] + A["diamond_tmp_gb"][i] * GB
            per_run = tot["corpus"][c] + tot["hits"][c]
            disk[c] = dict(downloads_gb=round(tot["downloads"][c] / GB, 2), corpus_gb=round(tot["corpus"][c] / GB, 2), queries_gb=round(tot["queries"][c] / GB, 2),
                           diamond_tmp_gb=A["diamond_tmp_gb"][i], hits_gb=round(tot["hits"][c] / GB, 2), live_run_copies=runs[i],
                           total_gb=round((fixed + per_run * runs[i]) / GB, 2), scalable_gb=round((tot["downloads"][c] + tot["queries"][c] + per_run * runs[i]) / GB, 2))
        search = {c: round((S["fixed_s"][i] + tot["search_s"][c]) / 3600, 1) for i, c in enumerate(CASES)}
        # the protein-share rules at the unfavourable ends of the ranges (a class low, the others high, and the reverse)
        maj = {}
        for g, x in groups.items():
            m = "prokaryotes" if g in ("bacteria", "archaea") else ("eukaryotes" if g.startswith("eukaryota.") else "viral_mobile")
            for c in CASES:
                maj.setdefault(m, {c2: 0 for c2 in CASES})[c] += x["proteins"][c]
        worst = {}
        for m, v in sorted(maj.items()):
            lo_rest = sum(o["low"] for k, o in maj.items() if k != m)
            hi_rest = sum(o["high"] for k, o in maj.items() if k != m)
            worst[m] = dict(lowest=round(v["low"] / ((v["low"] + hi_rest) or 1), 4), highest=round(v["high"] / ((v["high"] + lo_rest) or 1), 4))
        budget = dict(spec=spec["name"], groups=groups, proteins={c: int(tot["proteins"][c]) for c in CASES}, disk=disk, search_hours=search,
                      search_threads=12, shares=bg["shares"], shares_range=worst, repair=bg["repair"], rates_basis=rates["basis"], assumptions=A)
        bp = st.path("select", "budget.json")
        write_json(bp, budget)
        rep = st.path("select", "report.md")
        with open(rep, "w") as f:
            f.write(report(spec, bg, budget))
        for name, p in (("rates", rp), ("budget", bp), ("report", rep)):
            st.output(name, p)


def _r(d, fmt="%.0f"):
    return "%s [%s–%s]" % (fmt % d["central"], fmt % d["low"], fmt % d["high"])


def report(spec, bg, b):
    L = ["# Corpus selection %s: groups, coverage and cost" % spec["name"], "",
         "Planning numbers, not results: protein counts are catalogue annotations or estimates (the corpus stage counts its own genes and gate G1 "
         "decides again on those), byte and time rates are measured on few genomes or assumed (basis in every row). Ranges are [low–high].", "",
         "| group | floor | target | selected | eligible | proteins | hits GB | search h | proteins basis |", "|---|---|---|---|---|---|---|---|---|"]
    for g, x in b["groups"].items():
        d = bg["groups"][g]
        L.append("| %s | %s | %s | %d | %d | %s | %s | %s | %s |" % (g, d["floor"] if d["floor"] is not None else "—", d["target"], d["selected_units"], d["eligible_units"],
                                                       _r(x["proteins"]), _r(x["hits_gb"], "%.1f"), _r(x["search_h"], "%.1f"), "; ".join(x["basis"]["proteins"])))
    t = b["disk"]
    L += ["", "**Totals**: %s proteins; disk %s GB (hits %s GB, corpus files %s GB, downloads %s GB, query copy %s GB, DIAMOND temporary %s GB; "
          "%s live copies of a run); search %s hours at 12 threads." % (
              _r(b["proteins"]), _r({c: t[c]["total_gb"] for c in CASES}), _r({c: t[c]["hits_gb"] for c in CASES}), _r({c: t[c]["corpus_gb"] for c in CASES}),
              _r({c: t[c]["downloads_gb"] for c in CASES}), _r({c: t[c]["queries_gb"] for c in CASES}), _r({c: t[c]["diamond_tmp_gb"] for c in CASES}),
              _r({c: t[c]["live_run_copies"] for c in CASES}), _r(b["search_hours"], "%.0f")), "",
          "**Protein shares** (central estimate [lowest–highest the ranges allow]; the rules: no class above 0.60, eukaryotes at "
          "least 0.20, viruses and mobile elements at least 0.05; gate G1_select decides on the central estimate): "
          + "; ".join("%s %.3f [%.3f–%.3f]" % (k, v, b["shares_range"][k]["lowest"], b["shares_range"][k]["highest"]) for k, v in b["shares"]["central"].items()), ""]
    if b["repair"]:
        L += ["**Share repair**: %s" % json.dumps(b["repair"]), ""]
    if bg["inputs"].get("event_catalogue") == "absent":
        L += ["**Test-recipient exclusions are not applied yet**: the public confirmed-event catalogue (bench/catalogue/events.tsv) does not exist "
              "(gate G5 is waived until 2027-01-16). The selection must be run again once its recipients are excluded, before any search or training of the "
              "affected groups.", ""]
    L += ["## Coverage and reasons", "", "| group | selected phyla/orders/families/genera | eligible phyla/orders/families/genera | cap passed | hosts unknown | main reasons out |",
          "|---|---|---|---|---|---|"]
    for g, d in bg["groups"].items():
        cs, ce = d["coverage_selected"], d["coverage_eligible"]
        outs = ", ".join("%s %d" % kv for kv in sorted(d["out"].items(), key=lambda kv: -kv[1])[:4])
        L.append("| %s | %d/%d/%d/%d | %d/%d/%d/%d | %s | %s | %s |" % (g, cs["phylum"], cs["order"], cs["family"], cs["genus"], ce["phylum"], ce["order"], ce["family"],
                                                                 ce["genus"], "yes" if d["cap_passed"] else "no", d["host_unknown"], outs))
    if bg.get("ungrouped"):
        L += ["", "Rejected before a group could be given: " + "; ".join("%s: %s" % (p, ", ".join("%s %d" % kv for kv in v.items())) for p, v in bg["ungrouped"].items()) + "."]
    L += ["", "## Catalogue snapshots and what their filters leave out", "",
          "| catalogue | kind | query | records | wider queries (records) | sha256 |", "|---|---|---|---|---|---|"]
    for cid, c in bg["catalogues"].items():
        q = c.get("query") or {}
        body = q.get(c["kind"]) if isinstance(q.get(c["kind"]), dict) else {}
        qs = ", ".join("%s=%s" % kv for kv in sorted(((body or {}).get("params") or {}).items())) or "—"
        wider = "; ".join("%s: %d" % (lab, t["total"]) for lab, t in sorted((c.get("totals") or {}).items())) or "—"
        L.append("| %s | %s | %s | %d | %s | %s |" % (cid, c["kind"], qs, c["records"], wider, c["sha256"][:16]))
    L += ["", "Widening a catalogue (GenBank-only viruses, draft or uncultured prokaryotes, eukaryotes outside Ensembl) is a decision for the user."]
    L += ["", "Rates: %s. Selection inputs: %s." % (b["rates_basis"], ", ".join("%s %s" % (k, str(v)[:12]) for k, v in sorted(bg["inputs"].items()))), ""]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
