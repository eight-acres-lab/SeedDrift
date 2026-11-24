"""Stage rel_w13_build (package W13; pipeline rel_corpus): the corpus with the first batch of close relatives, built by
the corpus stages of the CI as they are (fetch, corpus, reference, eval_inputs, search, train_data; its rule asks for
their outputs), and checked (task breakdown W13 step 2). `python -m ci.stages.rel.w13_build RUN_DIR`.

The run's corpus spec must be the corpus.yaml of the passed pick run (rel_w13_build.pick_from, stage rel_w13_pick or rel_w13_xpick),
byte for byte. Checks on this run's training-data index:
  pinned     every genome of the pinned index (rel_w13_build.pinned_from: the run whose splits the line works on; its
             index sha256 rel_w13_build.pinned_index_sha256) is here with the same split; nothing else of the old corpus
             is new or gone
  relatives  every relative of the pick is here (or listed by stage fetch as unavailable), with its anchor's split, and
             no other genome is new
  no leak    no new training genome shares a genus or, without a genus, a species with a held-out or a test genome of
             its group; no new held-out genome shares one with a test genome; no new genome carries a cluster
  sealed     the new genomes pass the sealed check and the exclusions by lineage
It reports the genes of the new genomes per group and split and the disk the run takes. Test genomes are compared by
their split only; none is named in an output. Outputs under data/rel_w13/build/: report.json (written last) and
report.md. Gate G4_rel_w13_build.

rel_w13_build.upto chooses how far the run goes (its rule asks for that stage's output), so the build can run in
parts of a few hours; files fetched before are taken from the content store, hit files of a passed part per genome:
  fetch   stage fetch only: the downloads (sources, bytes, unavailable) of the relatives
  search  up to the search: the relatives' hit files (the search stage searches only genomes no run of
          search.reuse_from has); with the corpus specs of parts 1..i of n (`python -m ci.stages.rel.w13_build part
          I N ...`) the search of the new genomes takes n runs, each reusing the earlier ones per genome; with part i
          alone (`... --only`) the n runs take nothing from each other and the run of the whole spec takes them all
  all     up to train_data, with the checks above (the default)
Sources the fetch stage could not get (a 404, or a server error or no connection through every try with the profile's
fetch.retry_s and fetch.server_errors_unavailable) are listed; more than max_unavailable (default 5%) of the relatives
in the spec fails the stage. Their places are refilled by a new pick with rel_w13_pick.avoid_from.
Outside `all` the corpus spec may be the pick's restricted to some of its relatives (parts 1..i); in `all` it is the
pick's corpus.yaml byte for byte.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from ci.gates import GateResult, require
from ci.hashing import sha256_file, sha256_json
from ci.stages.common import main_args, write_json
from ci.stages.rel import common
from ci.stages.rel import w13_pick

OUT = os.path.join("data", "rel_w13", "build")
KEYS = {"pick_from", "pinned_from", "pinned_index_sha256", "upto", "max_unavailable", "flag_start_stop_share"}
UPTO = ("fetch", "search", "all")


def settings(cfg) -> dict:
    c = cfg.get("rel_w13_build") or {}
    unknown = set(c) - KEYS
    upto = str(c.get("upto", "all"))
    need = {"pick_from"} | ({"pinned_from", "pinned_index_sha256"} if upto == "all" else set())
    if unknown or need - set(c) or upto not in UPTO:
        raise SystemExit("rel_w13_build: needs %s, upto one of %s; unknown keys %s" % (sorted(need), UPTO, sorted(unknown)))
    return dict({k: str(c[k]) for k in KEYS if k in c}, upto=upto, max_unavailable=float(c.get("max_unavailable", 0.05)),
                flag_start_stop_share=float(c.get("flag_start_stop_share", 0.2)))


def estimate(x):
    """Proteins of a relative, for splitting the search: eukaryotes capped as in the corpus, plasmids small (the pick's
    figure for a plasmid relative is its host assembly's)."""
    p = int(x.get("proteins") or 0)
    return min(p, 1500) if x["group"].startswith("eukaryota.") else (min(p, 200) if x["group"] == "mobile.plasmid" else p)


def parts(rel, n):
    """The relatives in n parts of about equal proteins (greedy, the largest first, into the lightest part; fixed order):
    a list of n sets of ids."""
    bins, out = [0] * n, [set() for _ in range(n)]
    for x in sorted(rel, key=lambda x: (-estimate(x), x["id"])):
        b = min(range(n), key=lambda k: (bins[k], k))
        bins[b] += estimate(x)
        out[b].add(x["id"])
    return out


def part_spec(text, rel, i, n, only=False):
    """The corpus spec of parts 1..i of n: the pick's corpus.yaml (text) without the relatives of the later parts, so
    run i searches part i and takes the earlier parts' hit files per genome; with `only`, part i alone (the corpus's
    earlier genomes and that part's relatives: the runs of the parts take nothing from each other, and the run of the
    whole spec takes every part's hit files)."""
    import yaml
    spec = yaml.safe_load(text)
    ids = {x["id"] for x in rel}
    keep = parts(rel, n)[i - 1] if only else set().union(*parts(rel, n)[:i])
    head = [x for x in text[:6000].splitlines() if x.startswith("#")]
    spec = dict(spec, name=("%s-part%dof%d-only" if only else "%s-part%dof%d") % (spec["name"], i, n),
                sources=[x for x in spec["sources"] if x["id"] not in ids or x["id"] in keep])
    what = ("part %d of %d of the corpus below alone (its earlier genomes and that part's relatives)" % (i, n) if only else
            "parts 1-%d of %d of the corpus below (the relatives of the later parts are left out)" % (i, n))
    return ("# The %s, for the search of its new genomes in %d runs (stage rel_w13_build, upto search;\n"
            "# parts(): about equal proteins). Made by `python -m ci.stages.rel.w13_build part%s`.\n"
            % (what, n, " --only" if only else "") + "\n".join(head) + "\n" + yaml.safe_dump(spec, sort_keys=False, width=160))


def _json(p):
    with open(p) as f:
        return json.load(f)


def pick_run(run):
    """(relatives, corpus spec sha256) of a passed run of stage rel_w13_pick, or of stage rel_w13_xpick (the supplement
    of eukaryotic viruses and plants and algae: its new genomes are the relatives here, without anchors)."""
    rec = _json(os.path.join(run, "record.json"))
    if rec.get("status") != "passed":
        raise SystemExit("rel_w13_build: the pick run %s is %s, not passed" % (rec.get("run_id"), rec.get("status")))
    from ci.stages.rel import w2_v0 as W
    from ci.stages.rel import w13_xpick
    rp = os.path.join(run, w13_pick.OUT, "report.json")
    if os.path.exists(rp):
        W._output(run, "rel_w13_pick", "pick_report", rp)
    else:
        rp = os.path.join(run, w13_xpick.OUT, "report.json")
        W._output(run, "rel_w13_xpick", "xpick_report", rp)
    rep = _json(rp)
    for name, e in rep["files"].items():
        if sha256_file(os.path.join(run, e["path"]), use_sidecar=False) != e["sha256"]:
            raise SystemExit("rel_w13_build: the %s file of the pick run is not the one its report names" % name)
    rel = _json(os.path.join(run, rep["files"]["relatives"]["path"]))["relatives"]
    return rel, os.path.join(run, rep["files"]["corpus"]["path"]), dict(run_id=rec.get("run_id") or os.path.basename(run), report_sha256=sha256_file(rp, use_sidecar=False),
                                                                     corpus_sha256=rep["files"]["corpus"]["sha256"])


def spec_scope(mine_path, pick_path, rel, upto):
    """The relatives of this run's corpus spec: all of them (`all`: the pick's spec byte for byte), or, outside `all`,
    those of a spec made of the pick's sources (every old source and some relatives, each equal to the pick's)."""
    import yaml
    if upto == "all" or sha256_file(mine_path, use_sidecar=False) == sha256_file(pick_path, use_sidecar=False):
        if sha256_file(mine_path, use_sidecar=False) != sha256_file(pick_path, use_sidecar=False):
            raise SystemExit("rel_w13_build: the corpus spec is not the pick run's corpus.yaml")
        return list(rel)
    with open(mine_path) as f:
        mine = yaml.safe_load(f)
    with open(pick_path) as f:
        pick = yaml.safe_load(f)
    by = {x["id"]: x for x in pick["sources"]}
    ids = {x["id"] for x in rel}
    other = [x["id"] for x in mine["sources"] if by.get(x["id"]) != x]
    old_missing = [i for i in by if i not in ids and i not in {x["id"] for x in mine["sources"]}]
    if other or old_missing:
        raise SystemExit("rel_w13_build: the corpus spec is not the pick's restricted to some relatives (other sources %s, missing %d old ones)" % (other[:3], len(old_missing)))
    have = {x["id"] for x in mine["sources"]}
    return [x for x in rel if x["id"] in have]


def check(old, new, rel, unavailable):
    """Problems of the new index against the pinned one and the pick, and the counts. old, new: index genome lists."""
    problems = []
    o = {g["id"]: g for g in old if g.get("source", "corpus") == "corpus"}
    n = {g["id"]: g for g in new if g.get("source", "corpus") == "corpus"}
    moved = sorted(i for i in o if i in n and n[i]["split"] != o[i]["split"])
    gone = sorted(i for i in o if i not in n)
    if moved:
        problems.append(dict(check="pinned", error="%d genomes of the pinned index changed their split" % len(moved),
                             examples=[i for i in moved if o[i]["split"] != "test" and n[i]["split"] != "test"][:5]))
    if gone:
        problems.append(dict(check="pinned", error="%d genomes of the pinned index are not here" % len(gone),
                             examples=[i for i in gone if o[i]["split"] != "test"][:5]))
    r = {x["id"]: x for x in rel}
    new_ids = sorted(set(n) - set(o))
    other = [i for i in new_ids if i not in r]
    if other:
        problems.append(dict(check="relatives", error="new genomes that are not relatives of the pick", examples=other[:5]))
    missing = sorted(i for i in r if i not in n and i not in unavailable)
    if missing:
        problems.append(dict(check="relatives", error="relatives neither built nor listed as unavailable", examples=missing[:5]))
    wrong = sorted(i for i in r if i in n and n[i]["split"] != r[i]["split"])
    if wrong:
        problems.append(dict(check="relatives", error="relatives outside their anchor's split", examples=wrong[:5]))
    clustered = sorted(i for i in new_ids if n[i].get("cluster"))
    if clustered:
        problems.append(dict(check="no leak", error="new genomes with a near-relative cluster", examples=clustered[:5]))

    def unit(g):
        return ("g", g["ranks"][4]) if g["ranks"][4] else (("s", g["ranks"][5]) if g["ranks"][5] else ("id", g["id"]))
    held = {}
    for g in n.values():
        held.setdefault((g["group"], g["split"]), set()).add(unit(g))
    leaks = []
    for i in new_ids:
        g = n[i]
        above = ("heldout", "test") if g["split"] == "train" else (("test",) if g["split"] == "heldout" else ())
        if g["split"] == "test" or any(unit(g) in held.get((g["group"], sp), set()) for sp in above):
            leaks.append(i)
    if leaks:
        problems.append(dict(check="no leak", error="%d new genomes in a held-out or test unit, or in the test split" % len(leaks), examples=leaks[:5]))
    counts = {}
    for i in new_ids:
        g = n[i]
        d = counts.setdefault(g["split"], {}).setdefault(g["group"], dict(genomes=0, genes=0, with_hits=0))
        d["genomes"] += 1
        d["genes"] += int(g.get("genes") or 0)
        d["with_hits"] += 1 if int(g.get("hits") or 0) else 0
    return problems, counts, dict(pinned=len(o), kept=len(o) - len(gone), new=len(new_ids), relatives=len(r), unavailable=sorted(set(unavailable) & set(r)))


def disk(run_dir, data_root):
    out = {}
    for name in ("fetch", "corpus", "hits", "train_data", "reference", "eval_inputs"):
        p = os.path.join(run_dir, "data", name)
        if os.path.isdir(p):
            try:
                kb = int(subprocess.run(["du", "-sk", p], capture_output=True, text=True, timeout=600).stdout.split()[0])
            except Exception:  # noqa: BLE001 - du is a report, not a check
                kb = -1
            out[name] = round(kb / 1048576.0, 2)
    u = shutil.disk_usage(data_root)
    return dict(run_gb=out, free_gb=round(u.free / 1e9, 1), total_gb=round(u.total / 1e9, 1))


def cds_report(run_dir, scope, flag_share):
    """The coding sequences the corpus stage left out of the new genomes read from NCBI Datasets files (their corpus
    manifest entries' cds_quality): totals by reason, and per genome those left out for want of an ATG start or a stop
    codon; a genome that lost more than flag_share of its genes that way is flagged (its annotation is doubtful; it is
    kept and named, the programme lead, 2026-11-24)."""
    p = os.path.join(run_dir, "data", "corpus", "manifest.json")
    if not os.path.exists(p):
        return None
    ids = {x["id"] for x in scope}
    ents = [e for e in _json(p)["entries"] if e["id"] in ids and e.get("cds_quality")]
    if not ents:
        return None
    totals, per = {}, []
    for e in ents:
        q = e["cds_quality"]
        for k, v in q.items():
            if isinstance(v, int):
                totals[k] = totals.get(k, 0) + v
        per.append(dict(id=e["id"], genes=q["genes"], kept=q["kept"], left_out_start_stop=q.get("left_out_start_stop", 0),
                        share=q.get("left_out_start_stop_share", 0.0), gene_calling=e.get("gene_calling")))
    flagged = sorted((x for x in per if x["share"] > flag_share), key=lambda x: (-x["share"], x["id"]))
    return dict(genomes=len(per), totals=totals, flag_share=flag_share, flagged=flagged,
                rules=(_json(p).get("meta") or {}).get("cds_rules"), share_quantiles={str(qq): round(float(sorted(x["share"] for x in per)[min(len(per) - 1, int(qq * len(per)))]), 4)
                                                                                    for qq in (0.5, 0.9, 0.99)})


def fetch_report(run_dir, scope):
    fm = _json(os.path.join(run_dir, "data", "fetch", "manifest.json"))
    unavailable = {m["id"] for m in ((fm.get("meta") or {}).get("unavailable") or [])}
    got = {e["id"]: e for e in fm["entries"]}
    by = {}
    for x in scope:
        e = got.get(x["id"])
        d = by.setdefault(x["group"], dict(relatives=0, fetched=0, unavailable=0, bytes=0))
        d["relatives"] += 1
        if e:
            d["fetched"] += 1
            d["bytes"] += sum(int(f.get("bytes") or 0) for f in e.get("files") or [])
        elif x["id"] in unavailable:
            d["unavailable"] += 1
    missing = sorted(x["id"] for x in scope if x["id"] not in got and x["id"] not in unavailable)
    return by, unavailable, missing


def search_report(run_dir, scope):
    hm = _json(os.path.join(run_dir, "data", "hits", "manifest.json"))
    have = {e["id"] for e in hm["entries"]}
    rp = os.path.join(run_dir, "stages", "search.json")
    srep = _json(rp) if os.path.exists(rp) else {}
    keep = ("reused_per_genome", "searched", "search_wall_s", "reused_from", "genomes")
    return dict({k: srep.get(k) for k in keep if k in srep}, seconds=srep.get("seconds")), sorted(x["id"] for x in scope if x["id"] not in have)


def build(st, cfg, run_dir):
    s = settings(cfg)
    rel, pick_spec, prov = pick_run(s["pick_from"])
    mine = sha256_file(st.repo(cfg["corpus_spec"]), use_sidecar=False)
    scope = spec_scope(st.repo(cfg["corpus_spec"]), pick_spec, rel, s["upto"])
    by_fetch, unavailable, not_fetched = fetch_report(run_dir, scope)
    problems = [dict(check="fetch", error="relatives neither fetched nor listed as unavailable", examples=not_fetched[:5])] if not_fetched else []
    gone = sorted(set(unavailable) & {x["id"] for x in scope})
    if scope and len(gone) > s["max_unavailable"] * len(scope):          # an empty pass is no pass (the programme lead, 2026-11-23)
        problems.append(dict(check="fetch", error="%d of %d relatives unavailable, more than %.0f%%" % (len(gone), len(scope), 100 * s["max_unavailable"]),
                             examples=gone[:5]))
    search, counts, totals = None, {}, dict(relatives=len(scope), unavailable=sorted(set(unavailable) & {x["id"] for x in scope}))
    if s["upto"] in ("search", "all"):
        search, no_hits = search_report(run_dir, [x for x in scope if x["id"] not in unavailable])
        if no_hits:
            problems.append(dict(check="search", error="relatives without a hit file", examples=no_hits[:5]))
    by = {x["id"]: x for x in scope}
    cds = cds_report(run_dir, [x for x in scope if x["id"] not in unavailable], s["flag_start_stop_share"]) if s["upto"] in ("search", "all") else None
    if s["upto"] == "all":
        pp = os.path.join(s["pinned_from"], "data", "train_data", "index.json")
        if sha256_file(pp, use_sidecar=False) != s["pinned_index_sha256"]:
            raise SystemExit("rel_w13_build: the pinned index of %s is not the one the profile names" % s["pinned_from"])
        old = _json(pp)["genomes"]
        new = _json(os.path.join(run_dir, "data", "train_data", "index.json"))["genomes"]
        more, counts, totals = check(old, new, scope, unavailable)
        problems += more
        built = [g for g in new if g["id"] in by]
        ents = [dict(id=g["id"], taxid=int(g["taxid"]), split=g["split"]) for g in built]
    else:
        ents = [dict(id=x["id"], taxid=int(x["taxid"]), split=x["split"]) for x in scope if x["id"] not in unavailable]
    from ci.taxonomy import Taxonomy
    ev = common.sealed_check(cfg, [dict(id=e["id"], taxid=e["taxid"], accession=by[e["id"]]["accession"], paired_accession="",
                                        **({"host_taxid": by[e["id"]]["host_taxid"]} if by[e["id"]]["host_taxid"] else {})) for e in ents],
                             splits={e["id"]: e["split"] for e in ents}, tax=Taxonomy(st.repo(cfg["taxdump"]))) if ents else dict(note="no relative")
    st.extra["sealed_check"] = ev
    rep = dict(format="rel-w13-build", version=1, note=__doc__.strip().splitlines()[0], upto=s["upto"], corpus_spec=cfg["corpus_spec"], corpus_spec_sha256=mine,
               pick=prov, pinned=dict(run=os.path.basename(os.path.normpath(s["pinned_from"])), index_sha256=s["pinned_index_sha256"]) if s["upto"] == "all" else None,
               totals=totals, counts=counts, fetch=by_fetch, search=search, problems=problems, disk=disk(run_dir, cfg["data_root"]), sealed_check=ev, cds=cds)
    folder = os.path.join(run_dir, OUT)
    os.makedirs(folder, exist_ok=True)
    page = os.path.join(folder, "report.md")
    with open(page, "w") as f:
        f.write(page_of(rep))
    st.output("build_page", page)
    p = os.path.join(folder, "report.json")
    write_json(p, rep)
    st.output("build_report", p)
    st.extra.update(totals=totals, problems=len(problems), upto=s["upto"])
    if problems:
        raise SystemExit("rel_w13_build: %s" % json.dumps(problems)[:1500])
    return rep


def page_of(rep):
    t = rep["totals"]
    L = ["# The corpus with the first batch of close relatives (package W13, step 2; this run up to %s)" % rep["upto"], "", rep["note"], "",
         "Corpus spec %s (sha256 %s), the pick of run `%s`: %d relatives in this spec, %d unavailable in the archive." % (
             rep["corpus_spec"], rep["corpus_spec_sha256"][:12], rep["pick"]["run_id"], t["relatives"], len(t["unavailable"])), ""]
    L += ["## Downloads of the relatives", "", "| group | relatives | fetched | unavailable | MB |", "|---|---|---|---|---|"]
    for g, d in sorted(rep["fetch"].items()):
        L.append("| %s | %d | %d | %d | %.1f |" % (g, d["relatives"], d["fetched"], d["unavailable"], d["bytes"] / 1e6))
    L.append("")
    if rep.get("search"):
        L += ["Search: %s" % json.dumps(rep["search"]), ""]
    if rep.get("pinned"):
        L += ["Pinned splits of run `%s`: %d genomes kept with their split; %d new genomes." % (rep["pinned"]["run"], t["kept"], t["new"]), ""]
    for sp, by in sorted(rep["counts"].items()):
        L += ["## New genomes of the %s split" % sp, "", "| group | genomes | genes | with reference hits |", "|---|---|---|---|"]
        for g, d in sorted(by.items()):
            L.append("| %s | %d | %s | %d |" % (g, d["genomes"], "{:,}".format(d["genes"]), d["with_hits"]))
        L.append("")
    c = rep.get("cds")
    if c:
        t_ = c["totals"]
        L += ["## Coding sequences of the NCBI Datasets files", "", c.get("rules") or "", "",
              "%d genomes: %s coding sequences, %s genes, %s kept (at most the cap). Left out: %s." % (
                  c["genomes"], "{:,}".format(t_.get("records", 0)), "{:,}".format(t_.get("genes", 0)), "{:,}".format(t_.get("kept", 0)),
                  "; ".join("%s %s" % (k, "{:,}".format(t_.get(k, 0))) for k in ("pseudogene", "organelle", "another_genetic_code", "other_isoforms",
                                                                                   "length_not_multiple_of_3", "stop_codon_inside", "fewer_than_30_codons",
                                                                                   "no_atg_start", "no_stop_codon", "beyond_max_genes"))),
              "Share of a genome's genes left out for want of an ATG start or a stop codon: median %s, 90%% %s, 99%% %s." % (
                  c["share_quantiles"]["0.5"], c["share_quantiles"]["0.9"], c["share_quantiles"]["0.99"]),
              "Genomes above %.0f%% (their annotation is doubtful; kept, named here): %d." % (100 * c["flag_share"], len(c["flagged"])), ""]
        if c["flagged"]:
            L += ["| genome | genes | left out (start, stop) | share | kept |", "|---|---|---|---|---|"]
            L += ["| %s | %s | %s | %.1f%% | %s |" % (x["id"], "{:,}".format(x["genes"]), "{:,}".format(x["left_out_start_stop"]), 100 * x["share"], "{:,}".format(x["kept"]))
                  for x in c["flagged"]]
            L.append("")
    d = rep["disk"]
    L += ["Disk: this run %s GB (%s); %s GB free of %s GB." % (round(sum(v for v in d["run_gb"].values() if v > 0), 1),
                                                         ", ".join("%s %s" % (k, v) for k, v in d["run_gb"].items()), d["free_gb"], d["total_gb"]), ""]
    L += ["Checks: " + ("all passed." if not rep["problems"] else "; ".join("%s: %s" % (x["check"], x["error"]) for x in rep["problems"])), ""]
    return "\n".join(L)


def gate(st, cfg, run_dir):
    p = os.path.join(run_dir, OUT, "report.json")
    rep = _json(p)
    problems = list(rep.get("problems") or [])
    if sha256_file(st.repo(cfg["corpus_spec"]), use_sidecar=False) != rep.get("corpus_spec_sha256"):
        problems.append(dict(error="the corpus spec is not the one the report names"))
    if not (rep.get("sealed_check") or {}).get("ids") and (rep.get("totals") or {}).get("relatives"):
        problems.append(dict(error="no sealed check of the new genomes"))
    if rep.get("upto") != settings(cfg)["upto"]:
        problems.append(dict(error="the report is of another part of the build"))
    require(GateResult("G4_rel_w13_build", "fail" if problems else "pass", dict(problems=problems[:50], totals=rep.get("totals"))), run_dir)
    return sha256_json([sha256_file(p, use_sidecar=False)])


def summary(run_dir):
    rep = _json(os.path.join(run_dir, OUT, "report.json"))
    return dict(json=dict(totals=rep["totals"], counts=rep["counts"], disk=rep["disk"]), md=page_of(rep).splitlines()[2:40])


def main():
    if len(sys.argv) >= 6 and sys.argv[1] == "part":                 # python -m ci.stages.rel.w13_build part I N CORPUS_YAML RELATIVES_JSON [--only]
        with open(sys.argv[4]) as f:
            text = f.read()
        sys.stdout.write(part_spec(text, _json(sys.argv[5])["relatives"], int(sys.argv[2]), int(sys.argv[3]), only="--only" in sys.argv[6:]))
        return
    run_dir = main_args()
    with common.RelStage(run_dir, "rel_w13_build") as st:
        build(st, st.config, run_dir)


if __name__ == "__main__":
    main()
