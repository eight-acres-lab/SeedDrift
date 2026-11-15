"""Run the SeedDrift benchmark against a running SeedDrift API and write a report.

  datasets  bench/benchmark.json: spike-in genomes (bench/spikein.py) and a contaminated assembly
            (bench/contamination.py), each with a truth table
  configs   full pipeline and ablations (repeat filter, deep search, phylogeny off; AI-only baseline)
  metrics   recall by donor class and divergence, donor-class and donor-group accuracy, false positives
            on core genes, background call rate, AUPRC (spikes vs core genes, score = Alien Index),
            confidence tiers, contamination contigs, time per stage; 95% bootstrap intervals

usage:  python bench/run_benchmark.py --api http://127.0.0.1:8600 --out bench/results [--only spike_ecoli]
        python bench/run_benchmark.py --label broad --reeval      # metrics again from the stored jobs
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request
import uuid

FOREIGN = ("mammal", "prokaryote", "eukaryote", "virus")
CONFIGS = {
    "full": dict(),
    "no_repeat_filter": dict(repeat_filter="false"),
    "no_deep_search": dict(deep="false"),
    "no_phylogeny": dict(trees="false"),
    "baseline_ai_only": dict(repeat_filter="false", deep="false", trees="false"),
}


# -- API ---------------------------------------------------------------------------------------------
def post_job(api, files, fields):
    b = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n" % (b, k, v)).encode())
    for k, path in files.items():
        body.write(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                    "Content-Type: application/octet-stream\r\n\r\n" % (b, k, os.path.basename(path))).encode())
        body.write(open(path, "rb").read())
        body.write(b"\r\n")
    body.write(("--%s--\r\n" % b).encode())
    req = urllib.request.Request(api + "/api/jobs", data=body.getvalue(), method="POST",
                                 headers={"Content-Type": "multipart/form-data; boundary=" + b})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["id"]


def wait(api, jid, poll=20):
    while True:
        with urllib.request.urlopen("%s/api/jobs/%s" % (api, jid), timeout=60) as r:
            s = json.load(r)
        if s["status"] in ("done", "failed"):
            return s
        time.sleep(poll)


def genes(api, jid):
    with urllib.request.urlopen("%s/api/jobs/%s/files/genes.tsv" % (api, jid), timeout=600) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode()), delimiter="\t"))


# -- metrics -----------------------------------------------------------------------------------------
def boot(xs, f, n=1000, seed=1):
    """95% bootstrap interval of f over the list xs."""
    if not xs:
        return (None, None)
    rnd = random.Random(seed)
    vals = sorted(f([xs[rnd.randrange(len(xs))] for _ in xs]) for _ in range(n))
    return (round(vals[int(0.025 * n)], 3), round(vals[int(0.975 * n) - 1], 3))


def frac(xs):
    return sum(xs) / len(xs) if xs else 0.0


def auprc(pos, neg):
    """pos / neg: lists of scores (higher = more foreign); average precision. Genes with the same score are
    taken together: the uncalled genes all share the lowest score, and counting the uncalled spikes ahead of
    the uncalled natives would credit recall that no threshold can reach."""
    items = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg], key=lambda x: -x[0])
    tp = fp = i = 0
    ap, last_r = 0.0, 0.0
    while i < len(items):
        j = i
        while j < len(items) and items[j][0] == items[i][0]:
            tp += items[j][1]
            fp += 1 - items[j][1]
            j += 1
        r = tp / len(pos)
        ap += (tp / (tp + fp)) * (r - last_r)
        last_r, i = r, j
    return round(ap, 3)


def score(r):
    return float(r["AI"]) if r.get("category") in FOREIGN and r.get("AI") not in ("", None) else -1e9


def eval_spikein(pred, truth):
    by = {r["gene_id"]: r for r in pred}
    spikes = [t for t in truth if t["label"] == "spike" and t["gene_id"] in by]
    core = [t for t in truth if t["label"] == "core" and t["gene_id"] in by]
    other = [t for t in truth if t["label"] == "other" and t["gene_id"] in by]
    out = dict(n_spikes=len(spikes), n_core=len(core), n_other=len(other), classes={})
    for c in FOREIGN:
        ts = [t for t in spikes if t["true_class"] == c]
        if not ts:
            continue
        row = {}
        for d in sorted(set(t["divergence"] for t in ts), key=float):
            tt = [t for t in ts if t["divergence"] == d]
            hit = [by[t["gene_id"]]["category"] == c for t in tt]
            anyf = [by[t["gene_id"]]["category"] in FOREIGN for t in tt]
            grp = [by[t["gene_id"]]["donor_group"] == t["donor_group"] for t in tt if by[t["gene_id"]]["category"] == c]
            conf = [by[t["gene_id"]]["confidence"] in ("high", "medium") for t in tt if by[t["gene_id"]]["category"] == c]
            cats = {}
            for t in tt:
                k = by[t["gene_id"]]["category"]
                cats[k] = cats.get(k, 0) + 1
            row[d] = dict(n=len(tt), recall=round(frac(hit), 3), recall_ci=boot(hit, frac), any_foreign=round(frac(anyf), 3),
                          group_accuracy=round(frac(grp), 3) if grp else None,
                          high_or_medium=round(frac(conf), 3) if conf else None, calls=cats)
        out["classes"][c] = row
    fp = [by[t["gene_id"]]["category"] in FOREIGN for t in core]
    fph = [by[t["gene_id"]]["confidence"] == "high" for t in core]
    bg = [by[t["gene_id"]]["category"] in FOREIGN for t in other]
    bgh = [by[t["gene_id"]]["confidence"] == "high" for t in other]
    out.update(core_fp=round(frac(fp), 3), core_fp_ci=boot(fp, frac), core_fp_high=round(frac(fph), 3),
               background_foreign=round(frac(bg), 4), background_high=round(frac(bgh), 4),
               auprc_vs_core=auprc([score(by[t["gene_id"]]) for t in spikes], [score(by[t["gene_id"]]) for t in core]),
               auprc_vs_all=auprc([score(by[t["gene_id"]]) for t in spikes],
                                  [score(by[t["gene_id"]]) for t in core + other]))
    hi = [t for t in spikes if by[t["gene_id"]]["confidence"] == "high"]
    hi_bg = sum(bgh) + sum(fph)
    out["high_confidence_precision_vs_all"] = round(len(hi) / (len(hi) + hi_bg), 3) if hi or hi_bg else None
    out["high_confidence_recall"] = round(len(hi) / len(spikes), 3) if spikes else None
    return out


def match_transplant(pred, truth):
    """Predicted genes <-> inserted / core intervals (same contig, >= 50% of the true gene covered).
    Returns (pred rows incl. 'missed' placeholders, per-gene truth rows)."""
    by_contig = {}
    for r in pred:
        if r.get("contig"):
            by_contig.setdefault(r["contig"], []).append(r)
    labeled, used, rows = [], set(), list(pred)
    for k, t in enumerate(truth):
        a, b = int(t["start"]), int(t["end"])
        best, ov = None, 0
        for r in by_contig.get(t["contig"], []):
            o = min(b, int(r["end"])) - max(a, int(r["start"])) + 1
            if o > ov:
                best, ov = r, o
        if best is not None and ov >= 0.5 * (b - a + 1) and best["gene_id"] not in used:
            used.add(best["gene_id"])
            labeled.append(dict(t, gene_id=best["gene_id"]))
        elif t["label"] == "spike":
            gid = "MISSED_%d" % k
            rows.append(dict(gene_id=gid, category="missed", confidence="", donor_group="", AI="", comp_z="", island=""))
            labeled.append(dict(t, gene_id=gid))
    for r in pred:
        if r["gene_id"] not in used:
            labeled.append(dict(gene_id=r["gene_id"], label="other", true_class="", divergence="", donor_group=""))
    return rows, labeled


def eval_transplant(pred, truth):
    rows, labeled = match_transplant(pred, truth)
    out = eval_spikein(rows, labeled)
    by = {r["gene_id"]: r for r in rows}
    spikes = [t for t in labeled if t["label"] == "spike"]
    out["gene_calling_recall"] = round(frac([by[t["gene_id"]]["category"] != "missed" for t in spikes]), 3)
    comp = {}
    for d in sorted(set(t["divergence"] for t in spikes), key=float):
        z = [abs(float(by[t["gene_id"]]["comp_z"])) >= 2 for t in spikes if t["divergence"] == d and by[t["gene_id"]].get("comp_z")]
        comp[d] = round(frac(z), 3) if z else None
    other = [by[t["gene_id"]] for t in labeled if t["label"] == "other" and by[t["gene_id"]].get("comp_z")]
    out["composition_flag_by_amelioration"] = comp
    out["composition_flag_native"] = round(frac([abs(float(r["comp_z"])) >= 2 for r in other]), 3) if other else None
    isl = [int(by[t["gene_id"]].get("island") or 0) >= 2 for t in spikes
           if int(t.get("island_size") or 1) >= 2 and by[t["gene_id"]]["category"] in FOREIGN]
    out["island_recall"] = round(frac(isl), 3) if isl else None
    return out


def eval_contam(pred, summary, truth):
    bad = set(summary.get("contaminant_contigs", []))
    lab = {t["contig"]: t["label"] for t in truth}
    contam = [c for c, l in lab.items() if l == "contamination"]
    euk = [c for c, l in lab.items() if l != "contamination"]
    on = [r for r in pred if r.get("contig", "").startswith("contam_") and r["category"] in FOREIGN]
    euk_cand = [r for r in pred if r.get("contig") in euk and r["category"] in FOREIGN]
    return dict(contam_contigs=len(contam), flagged=sum(c in bad for c in contam), false_flags=sum(c in bad for c in euk),
                candidates_on_contam=len(on), high_on_contam=sum(r["confidence"] == "high" for r in on),
                candidates_on_eukaryote=len(euk_cand),
                eukaryote_candidates_by_conf={k: sum(r["confidence"] == k for r in euk_cand) for k in ("high", "medium", "low")})


# -- report ------------------------------------------------------------------------------------------
def report(res, path):
    L = ["# SeedDrift benchmark results", "",
         "Generated by `bench/run_benchmark.py`. Raw numbers: `bench/results/results.json`.", ""]
    for ds, runs in res.items():
        L += ["## %s" % ds, ""]
        full = runs.get("full") or next(iter(runs.values()), {})
        m = full.get("metrics", {})
        if "classes" in m:
            L += ["Spikes: %d; core genes (expected native): %d; other recipient genes (unlabeled): %d." %
                  (m["n_spikes"], m["n_core"], m["n_other"]), "",
                  "**Full pipeline: recall by donor class and divergence** (divergence: fraction of sites mutated for "
                  "protein spike-ins, amelioration level for DNA transplants; recall = called with the right class; "
                  "95% bootstrap interval; any = called foreign of any class; group = right donor group among the hits)", "",
                  "| class | divergence | n | recall | 95% CI | any foreign | group accuracy | high/medium confidence |",
                  "|---|---|---|---|---|---|---|---|"]
            for c, rows in m["classes"].items():
                for d, x in rows.items():
                    L.append("| %s | %d%% | %d | %.2f | %s–%s | %.2f | %s | %s |" % (
                        c, round(float(d) * 100), x["n"], x["recall"], x["recall_ci"][0], x["recall_ci"][1], x["any_foreign"],
                        "%.2f" % x["group_accuracy"] if x["group_accuracy"] is not None else "–",
                        "%.2f" % x["high_or_medium"] if x["high_or_medium"] is not None else "–"))
            if "gene_calling_recall" in m:
                L += ["", "DNA level: inserted genes found by gene calling %.2f; composition flag (|z| ≥ 2) by amelioration "
                      "%s, on native genes %s; inserted clusters recovered as islands %s." % (
                          m["gene_calling_recall"], m["composition_flag_by_amelioration"], m["composition_flag_native"],
                          m["island_recall"])]
            L += [""]
            L += ["**Configurations** (false positives: core genes called foreign; background: other recipient genes "
                  "called foreign, an upper bound since real genomes carry real transfers)", "",
                  "| configuration | mean recall (0% / 30% / 50% / 70%) | core FP | core FP high | background | background high | "
                  "AUPRC vs core | AUPRC vs all | high-conf precision vs all | high-conf recall | time |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for cfg, run in runs.items():
                mm = run.get("metrics", {})
                if "classes" not in mm:
                    L.append("| %s | failed: %s |" % (cfg, run.get("error", "")))
                    continue
                per = {}
                for rows in mm["classes"].values():
                    for d, x in rows.items():
                        per.setdefault(d, []).append(x["recall"])
                rec = " / ".join("%.2f" % (sum(v) / len(v)) for d, v in sorted(per.items(), key=lambda kv: float(kv[0])))
                L.append("| %s | %s | %.3f | %.3f | %.4f | %.4f | %.3f | %.3f | %s | %s | %s |" % (
                    cfg, rec, mm["core_fp"], mm["core_fp_high"], mm["background_foreign"], mm["background_high"],
                    mm["auprc_vs_core"], mm["auprc_vs_all"], mm["high_confidence_precision_vs_all"],
                    mm["high_confidence_recall"], fmt_t(run.get("seconds"))))
            L += [""]
        elif m:
            L += ["Contaminant contigs flagged: %d / %d; eukaryote contigs wrongly flagged: %d; candidate genes on "
                  "contaminant contigs: %d (high confidence: %d); candidates on eukaryote contigs: %d %s." %
                  (m["flagged"], m["contam_contigs"], m["false_flags"], m["candidates_on_contam"], m["high_on_contam"],
                   m["candidates_on_eukaryote"], m["eukaryote_candidates_by_conf"]), ""]
        t = full.get("timings")
        if t:
            L += ["Time per stage (full): " + ", ".join("%s %s" % (k, fmt_t(v)) for k, v in t.items()), ""]
    open(path, "w").write("\n".join(L) + "\n")


def fmt_t(s):
    if s is None:
        return "–"
    return "%.0f s" % s if s < 120 else "%.1f min" % (s / 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8600")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "benchmark.json"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--only", default="", help="comma-separated dataset names")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--label", default="", help="suffix for the result keys, e.g. the reference library version")
    ap.add_argument("--reeval", action="store_true",
                    help="recompute the metrics of the stored runs from their jobs (which must still exist); runs nothing")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rpath = os.path.join(a.out, "results.json")
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for ds in json.load(open(a.config)):
        if a.only and ds["name"] not in a.only.split(","):
            continue
        d = os.path.join(root, ds["dir"])
        truth = list(csv.DictReader(open(os.path.join(d, "truth.tsv")), delimiter="\t"))
        cfgs = a.configs.split(",") if ds["kind"] in ("spikein", "transplant") else ["full"]
        for cfg in cfgs:
            key = cfg + ("@" + a.label if a.label else "")
            print("[bench] %s / %s" % (ds["name"], cfg), file=sys.stderr, flush=True)
            if a.reeval:
                jid = res.get(ds["name"], {}).get(key, {}).get("job")
                if not jid:
                    continue
            else:
                fields = dict(name="bench %s %s" % (ds["name"], cfg), self_taxon=ds.get("self_taxon", ""),
                              domain=ds.get("domain", "auto"), structure="false", **CONFIGS[cfg])
                jid = post_job(a.api, {"file": os.path.join(d, ds["input"])}, fields)
            st = wait(a.api, jid)
            run = dict(job=jid, status=st["status"], error=st.get("error"))
            if st["status"] == "done":
                pred = genes(a.api, jid)
                s = st.get("summary", {})
                run.update(seconds=s.get("seconds"), timings=s.get("timings"), summary_categories=s.get("categories"),
                           metrics=(eval_spikein(pred, truth) if ds["kind"] == "spikein" else
                                    eval_transplant(pred, truth) if ds["kind"] == "transplant" else
                                    eval_contam(pred, s, truth)))
            res.setdefault(ds["name"], {})[key] = run
            json.dump(res, open(rpath, "w"), indent=1)
            report(res, os.path.join(root, "docs", "benchmark-results.md"))
    print("[bench] report: docs/benchmark-results.md", file=sys.stderr)


if __name__ == "__main__":
    main()
