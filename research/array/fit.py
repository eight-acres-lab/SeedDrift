#!/usr/bin/env python
"""Train one arm of the array model and test it: held-out genomes, then the benchmark genomes.

Training signal: is the query a gene of the same genome as the other genes shown? Positives of that question
are real genes of the genome; negatives are real genes of another reference genome. No transfer labels, no
taxonomy in the input. Taxonomy is used once, outside the model, to keep whole classes out of training
(the split of research/placement).

Written per run: model.pt, result.json, trials_validation.npz and trials_unseen.npz (one line per trial, with
the model's output and two fixed rules computed on the same rows), scores_<genome>.tsv for benchmark genomes.

usage (system python with torch; repo root):
  python research/array/fit.py --data DIR --names a b --profiles DIR/profiles.npz --library LIB --out DIR/run \
      --mode num --pairs 1 [--comp 1] [--blind 1] [--seed 0] [--bench spike_ecoli=JOB.npz ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy("file_system")     # many small tensors from the workers: avoid running out of descriptors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(1, os.path.join(HERE, "..", "placement"))
from model import BANDS, ArrayNet, Cfg, Donors, Genome, Store, Train, Trials  # noqa: E402

KEEP = ("label", "kind", "k_g", "k_t", "g", "x", "donor", "d_gt", "n_c", "ptr", "ptr_rule", "n_present", "rule_ai", "rule_ctx")


def auroc(score, y):
    score, y = np.asarray(score, dtype=np.float64), np.asarray(y).astype(bool)
    if len(y) == 0 or y.all() or not y.any():
        return float("nan")
    order = np.argsort(score, kind="stable")
    s = score[order]
    rank = np.empty(len(s))
    i = 0
    while i < len(s):                                   # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        rank[i:j + 1] = (i + j) / 2.0 + 1
        i = j + 1
    r = np.empty(len(s)); r[order] = rank
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at(score, y, fpr):
    if len(y) == 0 or y.all() or not y.any():
        return float("nan")
    neg = np.sort(score[~y])[::-1]
    thr = neg[min(len(neg) - 1, int(fpr * len(neg)))]
    return float((score[y] > thr).mean())


def splits(st, library):
    from placenet import Library
    from train import split_of
    lib = Library(library)
    s = split_of(lib)
    return {g: int(s[st.own[g]]) for g in range(len(st.genomes)) if st.own[g] >= 0}


def run(model, loader, dev):
    model.eval()
    logits, ptr_ok, keep = [], [], {}
    with torch.no_grad():
        for b in loader:
            logit, ptr = model(b["msa"].to(dev), b["present"].to(dev), b["site"].to(dev) if "site" in b else None)
            logits.append(logit.float().cpu().numpy())
            if "ptr" in b:
                ptr_ok.append((ptr.float().masked_fill(~b["present"][:, 0].to(dev), -1e9).argmax(1).cpu() == b["ptr"]).numpy())
            for k in b:
                if k not in ("msa", "present", "site"):
                    keep.setdefault(k, []).append(b[k].numpy())
    res = {k: np.concatenate(v) for k, v in keep.items()}
    res["logit"] = np.concatenate(logits)
    if ptr_ok:
        res["ptr_ok"] = np.concatenate(ptr_ok)
    return res


def band(d):
    return np.where(d < BANDS[0][1], 0, np.where(d < BANDS[1][1], 1, 2))


def trial_stats(r, score="logit"):
    """Native versus foreign queries. Strata use the measured distance between the genome and the donor of a
    trial pair (the native trial of a pair carries the same donor and settings as its foreign partner)."""
    y, s = r["label"] > 0.5, r[score]
    out = {"n": int(len(y)), "genomes": int(len(set(r["g"].tolist()))), "auroc": auroc(s, y), "recall_at_5": recall_at(s, y, 0.05)}
    b = band(r["d_gt"])
    for k, name in enumerate(("donor_near", "donor_mid", "donor_far")):
        m = b == k
        out[name] = {"n": int(m.sum()), "auroc": auroc(s[m], y[m]), "recall_at_5": recall_at(s[m], y[m], 0.05)}
        for kg in (0, 4, 16):
            for kt in (0, 4, 16):
                mm = m & (r["k_g"] == kg) & (r["k_t"] == kt)
                out[name]["hide_%d_%d" % (kg, kt)] = auroc(s[mm], y[mm])
    m = r["n_c"] == 0
    out["clean_context"] = auroc(s[m], y[m]); out["mixed_context"] = auroc(s[~m], y[~m])
    m = r["n_present"] > 0
    out["query_has_rows"] = {"share_native": float(m[~y].mean()), "share_foreign": float(m[y].mean()), "auroc": auroc(s[m], y[m])}
    per = [auroc(s[r["g"] == g], y[r["g"] == g]) for g in sorted(set(r["g"].tolist()))]
    out["auroc_per_genome_median"] = float(np.nanmedian(per))
    if score == "logit" and "ptr_ok" in r:
        m = (r["ptr"] >= 0) & y
        out["pointer"] = {"n": int(m.sum()), "model": float(r["ptr_ok"][m].mean()) if m.any() else float("nan"),
                          "most_similar_row": float((r["ptr_rule"][m] == r["ptr"][m]).mean()) if m.any() else float("nan"),
                          "chance": float((1.0 / np.maximum(1, r["n_present"][m])).mean()) if m.any() else float("nan")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True); ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--profiles", required=True); ap.add_argument("--library", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="num", choices=("none", "num", "msa", "plm")); ap.add_argument("--pairs", type=int, default=1)
    ap.add_argument("--comp", type=int, default=0); ap.add_argument("--blind", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--per-genome", type=int, default=54)
    ap.add_argument("--bench", nargs="*", default=[], help="name=job.npz (the job file gives the genome's own genus)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--rescore", type=int, default=0, help="score the saved model of --out on a fresh draw of unseen-class trials "
                    "(this seed), write trials_fresh.npz, and stop")
    a = ap.parse_args()
    ck = None
    if a.rescore:                                             # the settings of the saved run, not of this command line
        ck = torch.load(os.path.join(a.out, "model.pt"), weights_only=False)
        a.mode, a.pairs, a.comp = ck["cfg"]["mode"], int(ck["cfg"]["pairs"]), int(ck["cfg"]["comp"])
        a.blind, a.bench = json.load(open(os.path.join(a.out, "result.json")))["args"]["blind"], []
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bench = dict(x.split("=", 1) for x in a.bench)
    st = Store(a.data, a.names + list(bench), a.profiles, sites=a.mode == "plm")
    cfg = Cfg(blind=bool(a.blind))
    sp = splits(st, a.library)
    enough = {g for g, genes in st.genes_of.items() if len(genes) > cfg.M + 1}
    grp = {k: sorted(g for g, s in sp.items() if s == k and g in enough) for k in range(4)}
    print("genomes: train %d, validation %d, unseen classes %d, benchmark lineages %d (kept out)" % tuple(len(grp[k]) for k in range(4)), flush=True)
    # trials: validation genomes with validation donors (used to choose the checkpoint); unseen classes with donors
    # from unseen classes only (nothing in a test trial was trained on or used for selection)
    val = Trials(st, cfg, grp[1], Donors(st, grp[1]), a.per_genome, 101)
    test = Trials(st, cfg, grp[2], Donors(st, grp[2]), a.per_genome, 202)
    kw = dict(batch_size=a.batch, num_workers=a.workers, persistent_workers=False)
    if ck is not None:
        model = ArrayNet(**ck["cfg"]).to(dev)
        model.load_state_dict(ck["state"])
        r = run(model, torch.utils.data.DataLoader(Trials(st, cfg, grp[2], Donors(st, grp[2]), a.per_genome, a.rescore), **kw), dev)
        np.savez(os.path.join(a.out, "trials_fresh.npz"), **r)
        res = json.load(open(os.path.join(a.out, "result.json")))
        res["fresh"], res["fresh_seed"] = trial_stats(r), a.rescore
        res["fresh_rules"] = {k: trial_stats(r, k) for k in ("rule_ai", "rule_ctx")}
        json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)
        v = res["fresh"]
        print("fresh trials (seed %d): auroc %.4f; near %.4f, mid %.4f, far %.4f" % (
            a.rescore, v["auroc"], v["donor_near"]["auroc"], v["donor_mid"]["auroc"], v["donor_far"]["auroc"]), flush=True)
        return
    model = ArrayNet(a.mode, bool(a.pairs), bool(a.comp), a.d, a.layers).to(dev)
    print("parameters %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    log = []
    if not a.untrained:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
        data = Train(st, cfg, grp[0], Donors(st, grp[0]), a.steps * a.batch, a.seed)
        best, t0, step, run_loss = -1.0, time.time(), 0, []
        model.train()
        for b in torch.utils.data.DataLoader(data, **kw):
            logit, ptr = model(b["msa"].to(dev), b["present"].to(dev), b["site"].to(dev) if "site" in b else None)
            y, tgt = b["label"].to(dev), b["ptr"].to(dev)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y)
            has = tgt >= 0
            if has.any():
                pl = ptr.float().masked_fill(~b["present"][:, 0].to(dev), -1e9)
                loss = loss + 0.3 * F.cross_entropy(pl[has], tgt[has])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            run_loss.append(float(loss)); step += 1
            if step % 200 == 0:
                print("step %d loss %.4f  %.1f steps/s" % (step, np.mean(run_loss[-200:]), step / (time.time() - t0)), flush=True)
            if step % a.eval_every == 0 or step == a.steps:
                r = trial_stats(run(model, torch.utils.data.DataLoader(val, **kw), dev))
                crit = 0.5 * (r["auroc"] + r["donor_near"]["auroc"])          # the near donors are where arms can differ
                log.append({"step": step, "loss": float(np.mean(run_loss[-200:])), "criterion": crit, "val": r})
                print("  validation: auroc %.4f, near donors %.4f (n %d), mid %.4f, far %.4f" % (
                    r["auroc"], r["donor_near"]["auroc"], r["donor_near"]["n"], r["donor_mid"]["auroc"], r["donor_far"]["auroc"]), flush=True)
                if crit > best:
                    best = crit
                    torch.save({"cfg": model.cfg, "state": model.state_dict(), "step": step}, os.path.join(a.out, "model.pt"))
                model.train()
        model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), weights_only=False)["state"])
    res = {"args": vars(a), "log": log}
    for name, ds in (("validation", val), ("unseen", test)):
        r = run(model, torch.utils.data.DataLoader(ds, **kw), dev)
        np.savez(os.path.join(a.out, "trials_%s.npz" % name), **r)
        res[name] = trial_stats(r)
        res[name + "_rules"] = {k: trial_stats(r, k) for k in ("rule_ai", "rule_ctx")}
    r = res["unseen"]
    print("unseen classes: auroc %.4f, recall at 5%% %.3f; near %.4f (n %d), mid %.4f, far %.4f; pointer %.3f (most similar row %.3f)" % (
        r["auroc"], r["recall_at_5"], r["donor_near"]["auroc"], r["donor_near"]["n"], r["donor_mid"]["auroc"], r["donor_far"]["auroc"],
        r["pointer"]["model"], r["pointer"]["most_similar_row"]), flush=True)
    for k, v in res["unseen_rules"].items():
        print("  %s on the same trials: auroc %.4f; near %.4f, mid %.4f, far %.4f" % (
            k, v["auroc"], v["donor_near"]["auroc"], v["donor_mid"]["auroc"], v["donor_far"]["auroc"]), flush=True)
    lib = np.load(a.library)
    for name, job in bench.items():
        g = st.genomes.index(name)
        genus = int(np.load(job)["genome_ranks"][5])
        if not genus:
            sys.exit("%s: the job gives no genus, so nothing of the genome's own lineage would be hidden" % name)
        extra = lib["ranks"][:, 5] == genus
        r = run(model, torch.utils.data.DataLoader(Genome(st, cfg, g, extra, a.repeats, 7), **kw), dev)
        score, n_seen = {}, {}
        for i, x in enumerate(r["x"]):
            score.setdefault(int(x), []).append((float(r["logit"][i]), float(r["rule_ai"][i]), float(r["rule_ctx"][i])))
            n_seen[int(x)] = n_seen.get(int(x), 0) + int(r["n_present"][i])
        n_out = 0
        with open(os.path.join(a.out, "scores_%s.tsv" % name), "w") as fh:
            fh.write("gene_id\tarray\trule_ai\trule_ctx\n")
            for x, v in score.items():
                if n_seen[x] == 0:                    # no member of the gene in any row shown: no evidence, ranks last like everywhere else
                    n_out += 1
                    continue
                fh.write("%s\t%.4f\t%.4f\t%.4f\n" % ((st.gene_id[x],) + tuple(np.mean(v, axis=0))))
        print("scored %s: %d genes, %d more without any visible member (rank last); own genus hidden: %d proteomes" % (
            name, len(score) - n_out, n_out, int(extra.sum())), flush=True)
    if dev == "cuda":
        res["gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 1)
        print("peak GPU memory %.1f GB" % res["gpu_gb"], flush=True)
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
