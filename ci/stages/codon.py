"""Stage codon (pipeline codon, the truth-labels line): the codon-usage study on the development island chromosomes
(seeddrift/train/codon_study.py; 2026-11-20, the user: "我赞同你的去探究一下为啥密码子这条路有信号"). No corpus, search or
training of its own: the island arrays, comparators and evaluator output of a passed run of pipeline selfsup
(codon.islands_from) and the training-data arrays of passed runs (codon.corpora), each file checked against the sha256
its source stage or manifest reported. The panels draw on the train and held-out splits only (never the sealed part).

  python -m ci.stages.codon RUN_DIR study    the columns, in the model interpreter (GPU; train.python and its lock); no
                                             label file is opened
  python -m ci.stages.codon RUN_DIR report   the shared protocol (ci/stages/protocol.py) on the columns and on the island
                                             run's comparators, paired bootstrap intervals, the diagnostics

Configuration (`codon:`): islands_from (a run directory), corpora ({name: run directory}), panels, curves, manifolds,
relax_grid, trim, block_bp (see seeddrift/train/codon_study.py), baseline (the reference column of the paired intervals,
default cd_cu_<first panel>), expect ({column: AUROC}: gate G4_codon requires each within 0.001, for reproductions),
boot (draws, default 1000), block (native genes per resampling cluster, default 10). The interpreter: train.python,
train.python_lock, train.device, train.gpu_lease.
"""
from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from ci import manifest as M
from ci.hashing import sha256_file
from ci.stages.common import Stage, main_args, write_json

KEYS = ("islands_from", "corpora", "panels", "curves", "manifolds", "relax_grid", "trim", "block_bp", "baseline", "expect", "boot", "block",
        "relax_batch", "manifold_batch")


def stage_output(run, stage, name, path):
    """The sha256 of `path`, which must be output `name` of stage `stage` of `run`, and that stage must have passed."""
    p = os.path.join(run, "stages", stage + ".json")
    if not os.path.exists(p):
        raise SystemExit("codon: %s has no stage %s" % (run, stage))
    with open(p) as f:
        rep = json.load(f)
    if rep.get("status") != "passed":
        raise SystemExit("codon: stage %s of %s did not pass (%s)" % (stage, run, rep.get("status")))
    out = {o["name"]: o["sha256"] for o in rep.get("outputs") or []}
    sha = sha256_file(path, use_sidecar=False)
    if out.get(name) != sha:
        raise SystemExit("codon: %s is not the output %s of stage %s of %s" % (path, name, stage, run))
    return sha


def gate_passed(run, gate):
    p = os.path.join(run, "gates", gate + ".json")
    if not os.path.exists(p):
        raise SystemExit("codon: %s has no gate %s" % (run, gate))
    with open(p) as f:
        g = json.load(f)
    if g.get("status") not in ("pass", "waived"):
        raise SystemExit("codon: gate %s of %s did not pass (%s)" % (gate, run, g.get("status")))
    return g


def check_files(run, manifest_path, paths):
    """Every path (relative to run) must be a file of the manifest with its sha256."""
    man = M.load(manifest_path)
    want = {f["path"]: f["sha256"] for e in man["entries"] for f in e.get("files", [])}
    bad = [p for p in paths if want.get(p) is None or sha256_file(os.path.join(run, p), use_sidecar=False) != want[p]]
    if bad:
        raise SystemExit("codon: files of %s differ from manifest %s: %s" % (run, manifest_path, bad[:5]))


def sources(cfg):
    """The checked inputs: the island run's index, evaluator output, comparators and arrays; each corpus's index and the
    arrays of its train and held-out genomes. Returns the provenance and the paths."""
    cc = cfg["codon"]
    isl = cc["islands_from"]
    prov = dict(islands_from=isl, corpora={})
    ip = os.path.join(isl, "data", "islands", "index.json")
    prov["islands_index"] = stage_output(isl, "islands_data", "islands_index", ip)
    gate_passed(isl, "G3_islands")
    prov["islands_eval"] = stage_output(isl, "islands_report", "islands_eval", os.path.join(isl, "data", "islands", "eval", "dev.json"))
    prov["islands_scores"] = stage_output(isl, "islands_score", "islands_scores", os.path.join(isl, "data", "islands", "scores.json"))
    with open(ip) as f:
        idx = json.load(f)
    mp = os.path.join(isl, "data", "islands", "manifest.json")
    with open(mp) as f:
        msha = json.load(f)["sha256"]
    check_files(isl, os.path.join(isl, "manifests", msha + ".json"), [g["path"] for g in idx["genomes"]] + [os.path.join(g["folder"], "genes.tsv") for g in idx["genomes"]])
    prov["comparators"] = {g["id"]: sha256_file(os.path.join(isl, "data", "islands", "scores", "comparators", "scores_%s.tsv" % g["id"]), use_sidecar=False)
                           for g in idx["genomes"]}
    for name, run in sorted(cc["corpora"].items()):
        tp = os.path.join(run, "data", "train_data", "index.json")
        sha = stage_output(run, "train_data", "train_data_index", tp)
        with open(tp) as f:
            tix = json.load(f)
        used = [os.path.relpath(os.path.join(run, "data", "train_data", m["id"] + ".npz"), run) for m in tix["genomes"] if m["split"] in ("train", "heldout")]
        check_files(run, os.path.join(run, "manifests", tix["sha256"] + ".json"), used)
        prov["corpora"][name] = dict(run=run, train_data_index=sha, genomes=len(used))
    return prov, idx


def do_study(run_dir):
    with Stage(run_dir, "codon_study", requires=("pre",)) as st:
        cfg = st.config
        cc = cfg["codon"]
        unknown = set(cc) - set(KEYS)
        if unknown:
            raise SystemExit("codon: unknown keys %s" % sorted(unknown))
        from ci import gpu_lease
        from ci.stages.train import model_env, selfsup_env
        tc = cfg.get("train") or {}
        py = tc.get("python", "/usr/local/bin/python3")
        model_env(st, py, tc.get("python_lock", "ci/envs/model-torch.pip.txt"))
        prov, idx = sources(cfg)
        st.extra["sources"] = prov
        conf = {k: v for k, v in cc.items() if k in ("panels", "curves", "manifolds", "relax_grid", "trim", "block_bp", "relax_batch", "manifold_batch")}
        conf.update(run_dir=run_dir, out=os.path.join("data", "codon", "raw"), islands_run=cc["islands_from"],
                    islands_index=os.path.join(cc["islands_from"], "data", "islands", "index.json"), device=tc.get("device", "auto"),
                    corpora={k: dict(index=os.path.join(v, "data", "train_data", "index.json")) for k, v in cc["corpora"].items()})
        cp = st.path("codon", "study-conf.json")
        write_json(cp, conf)
        env = selfsup_env(tc)
        with gpu_lease.hold("codon-" + os.path.basename(run_dir), enabled=bool(tc.get("gpu_lease", True))) as lease:
            st.extra["gpu_lease"] = lease
            with open(st.path("codon", "study.log"), "w") as lg:
                pr = subprocess.Popen([py, st.repo("seeddrift/train/codon_study.py"), cp], stdout=lg, stderr=subprocess.STDOUT, env=env, cwd=st.repo("."))
                lease.add(pr.pid)
                rc = pr.wait()
        if rc != 0:
            raise SystemExit("codon_study.py failed (exit %d); see %s" % (rc, st.path("codon", "study.log")))
        raw = os.path.join(run_dir, "data", "codon", "raw")
        with open(os.path.join(raw, "study.json")) as f:
            study = json.load(f)
        from ci.stages.islands import write_scores
        from ci.stages.search import gene_ids
        d = os.path.join(run_dir, "data", "codon", "scores")
        os.makedirs(d, exist_ok=True)
        for g in idx["genomes"]:
            names = gene_ids(os.path.join(cc["islands_from"], g["folder"]))
            z = np.load(os.path.join(raw, g["id"] + ".npz"))
            if len(z["gene"]) != len(names) or not np.array_equal(z["gene"], np.arange(len(names))):
                raise SystemExit("codon: the columns of %s do not cover its genes in order" % g["id"])
            cols = {k[4:]: z[k] for k in sorted(z.files) if k.startswith("col_")}
            write_scores(os.path.join(d, "scores_%s.tsv" % g["id"]), names, cols)
        out = os.path.join(run_dir, "data", "codon", "study.json")
        write_json(out, dict(study, sources=prov, chromosomes=[g["id"] for g in idx["genomes"]]))
        st.extra.update(columns=len(study.get("columns") or []), seconds=study.get("seconds"), device=study.get("device"))
        st.output("codon_study", out)


def default_pairs(columns, baseline, comps):
    """The paired intervals the report computes: every column and comparator against the baseline; host variants
    against cd_cun of their panel; Markov-1 and relaxation against cd_cun; the parts against cd_cu; a hidden panel
    against its base and against its random controls."""
    pairs = [(c, baseline) for c in columns + comps if c != baseline]
    pairs += [(c, None) for c in [baseline] + comps]
    for c in columns:
        kind, _, p = c[3:].partition("_")
        if kind in ("hin", "trim", "block", "mix3", "m1", "relax") and "cd_cun_" + p in columns:
            pairs.append((c, "cd_cun_" + p))
        if kind in ("aa", "syn", "cun", "lse", "std") and "cd_cu_" + p in columns:
            pairs.append((c, "cd_cu_" + p))
    return pairs


def panel_contrasts(columns, specs):
    """Hidden panels: (hidden, base) and (hidden, each random control of the same base and rank)."""
    out = []
    for s in specs:
        if not s.get("hide") or s.get("control") or not s.get("base"):
            continue
        h, b = "cd_cu_" + s["name"], "cd_cu_" + s["base"]
        if h in columns and b in columns:
            out.append((h, b))
        for r in specs:
            if r.get("control") == "random" and r.get("hide") == s["hide"] and r.get("base") == s["base"] and "cd_cu_" + r["name"] in columns:
                out.append((h, "cd_cu_" + r["name"]))
                out.append(("cd_cu_" + r["name"], b))
    return out


def winners(table, raw, study, corpora_meta, specs):
    """For panels with a winner array: the share of island and of native genes whose best panel genome shares the
    chromosome's order, class or phylum, or is of another group (labels used here, after scoring)."""
    out = {}
    for s in specs:
        name = s["name"]
        if name.startswith("rc_"):
            continue
        cnt = {1: {}, 0: {}}
        for acc, t in table.items():
            z = np.load(os.path.join(raw, acc + ".npz"))
            key = "diag_win_" + name
            if key not in z.files:
                continue
            mem = study["panels"][name]["per_chromosome"].get(acc) or study["panels"][name]["base"]
            own = t["ranks"]
            for y in (1, 0):
                for w in z[key][t["y"] == y]:
                    if w < 0:
                        continue
                    m = corpora_meta(mem[int(w)])
                    r = m["ranks"]
                    kind = ("order" if own[2] and int(r[2]) == int(own[2]) else "class" if own[1] and int(r[1]) == int(own[1])
                            else "phylum" if own[0] and int(r[0]) == int(own[0]) else m["group"])
                    cnt[y][kind] = cnt[y].get(kind, 0) + 1
        if cnt[1] or cnt[0]:
            out[name] = {("island" if y else "native"): {k: round(v / max(sum(c.values()), 1), 4) for k, v in sorted(c.items())} for y, c in cnt.items()}
    return out


def relax_rates(table, raw, specs):
    """Mean relaxation rates at the best panel genome (l_a amino-acid usage, l_s synonymous choice) for island genes,
    native genes, and the island genes in the top decile of their chromosome's cd_relax."""
    out = {}
    for s in specs:
        name = s["name"]
        acc_v = {"island": [[], []], "native": [[], []], "island_top": [[], []]}
        for acc, t in table.items():
            z = np.load(os.path.join(raw, acc + ".npz"))
            if "diag_relax_la_" + name not in z.files:
                continue
            la, ls = z["diag_relax_la_" + name], z["diag_relax_ls_" + name]
            sc = np.asarray(t["cols"]["cd_relax_" + name], float)
            thr = np.nanquantile(sc, 0.9) if np.isfinite(sc).any() else np.inf
            for key, m in (("island", t["y"] == 1), ("native", t["y"] == 0), ("island_top", (t["y"] == 1) & (sc >= thr))):
                acc_v[key][0] += list(la[m])
                acc_v[key][1] += list(ls[m])
        if acc_v["island"][0]:
            out[name] = {k: dict(l_aa=float(np.mean(v[0])) if v[0] else None, l_syn=float(np.mean(v[1])) if v[1] else None, genes=len(v[0]))
                         for k, v in acc_v.items()}
    return out


def do_report(run_dir):
    with Stage(run_dir, "codon_report", requires=("pre",)) as st:
        cfg = st.config
        cc = cfg["codon"]
        from ci.stages import protocol as PR
        from ci.stages.islands import COMPARATORS, admitted, island_table
        isl = cc["islands_from"]
        store = (cfg.get("islands") or {}).get("store")
        with open(os.path.join(run_dir, "data", "codon", "study.json")) as f:
            study = json.load(f)
        if not store:
            with open(os.path.join(isl, "config.resolved.yaml")) as f:
                import yaml
                store = yaml.safe_load(f)["islands"]["store"]
        # the inputs the report reads again are the ones the study checked (their sha256 in study.json)
        src = study["sources"]
        dev_json = os.path.join(isl, "data", "islands", "eval", "dev.json")
        if sha256_file(dev_json, use_sidecar=False) != src["islands_eval"]:
            raise SystemExit("codon: %s changed since the study checked it" % dev_json)
        for acc, sha in src["comparators"].items():
            cp = os.path.join(isl, "data", "islands", "scores", "comparators", "scores_%s.tsv" % acc)
            if sha256_file(cp, use_sidecar=False) != sha:
                raise SystemExit("codon: %s changed since the study read it" % cp)
        with open(dev_json) as f:
            res = json.load(f)
        per = admitted(res)
        if not per:
            raise SystemExit("codon: no chromosome of %s is admitted at gene level with gated labels" % isl)
        dirs = [os.path.join(run_dir, "data", "codon", "scores"), os.path.join(isl, "data", "islands", "scores", "comparators")]
        table, units, e5 = island_table(dirs, store, per, res["units"])
        with open(os.path.join(isl, "data", "islands", "index.json")) as f:
            ranks = {g["id"]: g["ranks"] for g in json.load(f)["genomes"]}
        for acc, t in table.items():
            t["ranks"] = ranks[acc]
        columns = [c for c in study["columns"] if all(c in t["cols"] for t in table.values())]
        comps = [c for c in list(COMPARATORS) + e5 if all(c in t["cols"] for t in table.values())]
        specs = [dict(s) for s in cc.get("panels") or []]
        from seeddrift.train.codon_study import expand_curve
        specs += expand_curve(cc.get("curves"))
        baseline = cc.get("baseline") or "cd_cu_" + specs[0]["name"]
        main_cols = [c for c in columns if not c.startswith("cd_cu_rc_")]
        ev = PR.evaluate(table, units, main_cols + comps, comps)
        curve = {}                                                     # the panel-size curves: AUROC per size, over seeds
        for c in columns:
            if c.startswith("cd_cu_rc_"):
                name, _, rest = c[len("cd_cu_rc_"):].rpartition("_")
                by = {}
                for t in table.values():
                    v = np.asarray(t["cols"][c], float)
                    a = PR.auroc(v[t["y"] == 1], v[t["y"] == 0])
                    if a is not None:
                        by.setdefault(t["unit"], []).append(a)
                size = rest if rest == "all" else rest.split("s")[0]
                curve.setdefault(name, {}).setdefault(size, []).append(float(np.mean([np.mean(v) for v in by.values()])))
        curve = {n: {k: dict(mean=float(np.mean(v)), sd=float(np.std(v)), draws=len(v)) for k, v in sorted(c.items(), key=lambda kv: (kv[0] == "all", int(kv[0]) if kv[0] != "all" else 0))}
                 for n, c in curve.items()}
        pairs = default_pairs(main_cols, baseline, comps) + panel_contrasts(columns, specs)
        seen, uniq = set(), []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        intervals = PR.paired_delta(table, units, uniq, n_boot=int(cc.get("boot", 1000)), seed=int(cfg.get("seed", 1)), block=int(cc.get("block", 10)))
        names = main_cols + [c for c in ("cmp_codon", "cmp_gc3", "cmp_kmer", "cmp_hits", "e5_codon", "e5_kmer", "e5_n_rows") if c in comps]
        spear = PR.spearman_matrix(table, names)
        corp = {}
        for name, run in cc["corpora"].items():
            with open(os.path.join(run, "data", "train_data", "index.json")) as f:
                for m in json.load(f)["genomes"]:
                    corp.setdefault(m["id"], m)
        raw = os.path.join(run_dir, "data", "codon", "raw")
        sizes = {}                                                     # panel sizes: the base, and per chromosome when hidden
        for s in specs:
            sp = study["panels"][s["name"]]
            per_c = {acc: len(sp["per_chromosome"][acc]) for acc in sorted(table) if acc in (sp.get("per_chromosome") or {})}
            sizes[s["name"]] = dict(base=len(sp["base"]), **({"per_chromosome": per_c, "empty_for": sorted(a for a, n in per_c.items() if n == 0)} if per_c else {}))
        out = st.path("codon", "report.json")
        write_json(out, dict(note=__doc__.strip().splitlines()[0], baseline=baseline, units=units, chromosomes=sorted(table), scores=ev,
                             intervals=intervals, panel_size_curve=curve, panel_sizes=sizes, spearman=spear, winners=winners(table, raw, study, corp.__getitem__, specs),
                             relaxation=relax_rates(table, raw, specs), manifolds={k: {kk: vv for kk, vv in v.items() if kk != "members"} for k, v in study["manifolds"].items()},
                             expect=cc.get("expect") or {}, not_implemented=list(PR.NOT_IMPLEMENTED), decide=PR.DECIDE))
        st.output("codon_report", out)


def main():
    run_dir = main_args()
    import sys
    {"study": do_study, "report": do_report}[sys.argv[2]](run_dir)


if __name__ == "__main__":
    main()
