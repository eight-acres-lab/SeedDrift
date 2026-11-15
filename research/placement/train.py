#!/usr/bin/env python
"""Train the placement model by self-supervision on reference proteins whose lineage is known.

For every sampled reference protein and every hold-out level (hide its own genus, family, order, class or
phylum) the selected hits are the input and the true relation of the query to each hit is the target.
Splits and the validity rule are fixed in README.md. Held-out lineages are held out as QUERIES; their
proteomes can still appear among the hits of training queries from other lineages.

usage (needs torch):
  python research/placement/train.py --library DIR/library.npz --data DIR/train_a.npz [DIR/train_b.npz] --out DIR/run_full
  ... --no-pair      attention without the taxonomic relation between hits
  ... --no-attn      every hit judged on its own features only
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from placenet import COLS, LEVELS, N_OUT, ROOT, Library, PlaceNet, Runner, make_example, pack  # noqa: E402

LIST_LIMIT = 1000                        # --max-target-seqs of the search that produced the data
# lineages of the three benchmark recipients (domain, phylum, class, order, family, genus): never trained on
BENCH = {"Escherichia": ([2, 1224, 1236, 91347, 543, 561], 4), "Legionella": ([2, 1224, 1236, 118969, 444, 445], 4),
         "Entamoeba": ([2759, 2605435, 0, 2682482, 33084, 5758], 2)}   # (lineage, hold out everything sharing this depth)


def bucket(x):
    return (int(x) * 2654435761 % 2 ** 32) % 10


def split_of(lib):
    """Per proteome: 0 train, 1 validation (unseen genomes of seen classes), 2 test (unseen classes),
    3 lineages of the benchmark recipients."""
    r = lib.ranks
    key = np.where(r[:, 2] != 0, r[:, 2], np.where(r[:, 1] != 0, r[:, 1], r[:, 0]))
    s = np.where(np.array([bucket(k) == 0 for k in key]), 2, np.where(np.array([bucket(t) == 1 for t in lib.taxid]), 1, 0))
    for lin, depth in BENCH.values():
        s[lib.relation_to(lin) >= depth] = 3
    return s


def queries(paths, lib):
    """Yield (name, proteome index, proteome index of each hit, hit columns, length) for every query with hits."""
    for path in paths:
        z = np.load(path)
        off, hp_all, cols = z["off"], z["h_p"], {k: z["h_" + k] for k in COLS}
        q_tax, q_id, q_len = z["q_tax"], z["q_id"], z["q_len"]      # read once: an .npz member is re-read on every access
        for q in range(len(q_tax)):
            qp = lib.pidx.get(int(q_tax[q]))
            a, b = off[q], off[q + 1]
            if qp is not None and b > a:
                yield str(q_id[q]), qp, hp_all[a:b], {k: v[a:b] for k, v in cols.items()}, float(q_len[q])


def stored_limit(paths, default):
    """--max-target-seqs recorded in the data files (older files have none: the default is used)."""
    lim = {int(z["limit"]) if "limit" in z.files else default for z in map(np.load, paths)}
    if len(lim) != 1:
        sys.exit("data files were searched with different list limits: %s" % sorted(lim))
    return lim.pop()


def build(paths, lib, list_limit):
    ex, Q, L, dropped = [], [], [], np.zeros(len(LEVELS), dtype=int)
    for _, qp, hp, d, qlen in queries(paths, lib):
        rel, prev = lib.S[qp], -1
        for li, m in enumerate(LEVELS):
            hidden = rel >= m
            n_hid = int(hidden.sum())
            if n_hid == prev:
                continue                                   # hiding one more rank removed nothing: same example
            prev = n_hid
            e = make_example(lib, hp, d, qlen, hidden, list_limit)
            if e is None:
                dropped[li] += 1
                continue
            ex.append(e); Q.append(qp); L.append(m)
    print("examples %d; not usable (no hit left, or stored list exhausted), by level %s: %s" % (len(ex), LEVELS, dropped.tolist()), file=sys.stderr)
    X, P, NID = pack(ex, lib)
    return X, P, NID, np.array(Q), np.array(L)


def node_target(y, pad, nid):
    """Where the query truly attaches: its ancestor at the deepest rank it shares with any token (ROOT if none).
    Returns the node index and that depth."""
    ymax, j = torch.where(pad, torch.full_like(y, -1), y).max(1)
    slot = nid[torch.arange(len(y), device=y.device), j, (ymax - 1).clamp(min=0)]
    return torch.where(ymax > 0, slot, torch.full_like(slot, ROOT)), ymax


def metrics(prob, y, pad, level, node, nid):
    """Per-hit accuracy and error, how well the closest remaining relative is recognised, and whether the
    single attachment node is right."""
    target, depth = node_target(y, pad, nid)
    rank = torch.zeros(node.shape[0], node.shape[1] + 2, dtype=torch.long, device=node.device)   # depth of each candidate node
    rank.scatter_(1, nid.reshape(len(nid), -1), torch.arange(1, 7, device=node.device).repeat(nid.shape[1]).expand(len(nid), -1))
    rank[:, ROOT] = 0
    best = node.argmax(1)
    node_depth = rank.gather(1, best[:, None])[:, 0]
    pred = prob.argmax(-1)
    ev = (prob * torch.arange(N_OUT, device=prob.device)).sum(-1)
    ok = ~pad
    out = {}
    for name, sel in [("all", torch.ones_like(level, dtype=torch.bool))] + [("hide>=%d" % m, level == m) for m in LEVELS]:
        if not sel.any():
            continue
        o = ok & sel[:, None]
        near_true = torch.where(ok, y, torch.zeros_like(y)).max(1).values[sel]
        near_pred = torch.where(ok, pred, torch.zeros_like(pred)).max(1).values[sel]
        near_ev = torch.where(ok, ev, torch.zeros_like(ev)).max(1).values[sel]
        per_depth = {int(k): round(float((pred == y)[o & (y == k)].float().mean()), 4) for k in range(N_OUT) if (o & (y == k)).any()}
        out[name] = dict(n=int(sel.sum()), hit_acc=round(float((pred == y)[o].float().mean()), 4),
                         hit_acc_macro=round(float(np.mean(list(per_depth.values()))), 4), hit_recall_by_depth=per_depth,
                         hit_mae=round(float((ev - y).abs()[o].mean()), 4),
                         nearest_acc=round(float((near_pred == near_true).float().mean()), 4),
                         nearest_within1=round(float(((near_pred - near_true).abs() <= 1).float().mean()), 4),
                         nearest_mae=round(float((near_ev - near_true).abs().mean()), 4),
                         node_acc=round(float((best == target)[sel].float().mean()), 4),
                         node_depth_acc=round(float((node_depth == depth)[sel].float().mean()), 4),
                         node_depth_within1=round(float(((node_depth - depth).abs() <= 1)[sel].float().mean()), 4))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--no-pair", action="store_true")
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--domain", action="store_true", help="also give the model the domain of each hit (absolute taxon identity; off by default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="random split by example, for testing the code on a toy file")
    ap.add_argument("--list-limit", type=int, default=LIST_LIMIT, help="--max-target-seqs of the search that produced the data")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    lib = Library(a.library)
    t0 = time.time()
    X, P, NID, Q, L = build(a.data, lib, stored_limit(a.data, a.list_limit))
    print("built in %.0f s" % (time.time() - t0), file=sys.stderr)
    split = split_of(lib)[Q]
    if a.smoke:
        split = np.random.randint(0, 4, len(Q))
    names = ("train", "val", "test_unseen_class", "test_bench_lineage")
    for s, name in enumerate(names):
        print("%s: %d examples, %d proteomes" % (name, (split == s).sum(), len(set(Q[split == s]))), file=sys.stderr)
    run = Runner(lib, dev)
    Xt, Pt, Nt = torch.from_numpy(X).to(dev), torch.from_numpy(P).to(dev), torch.from_numpy(NID).to(dev)
    Qt, Lt = torch.from_numpy(Q).to(dev), torch.from_numpy(L).to(dev)

    def batch(ix):
        x, dom, rel, pad, nid, pc = run.inputs(Xt[ix], Pt[ix], Nt[ix])
        return x, dom, rel, pad, nid, run.S[Qt[ix][:, None], pc].long().clamp(max=N_OUT - 1)

    model = PlaceNet(d=a.d, layers=a.layers, use_attn=not a.no_attn, use_pair=not a.no_pair, use_domain=a.domain).to(dev)
    print("parameters: %d" % sum(p.numel() for p in model.parameters()), file=sys.stderr)
    tr = torch.from_numpy(np.flatnonzero(split == 0)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * ((len(tr) + a.batch - 1) // a.batch), pct_start=0.05)

    @torch.no_grad()
    def evaluate(s):
        model.eval()
        ix = torch.from_numpy(np.flatnonzero(split == s)).to(dev)
        if len(ix) == 0:
            return {}
        pr, ys, pads, nodes, nids = [], [], [], [], []
        for i in range(0, len(ix), 2048):
            x, dom, rel, pad, nid, y = batch(ix[i:i + 2048])
            hit, node = model(x, dom, rel, pad, nid)
            pr.append(hit.softmax(-1)); ys.append(y); pads.append(pad); nodes.append(node); nids.append(nid)
        return metrics(torch.cat(pr), torch.cat(ys), torch.cat(pads), Lt[ix], torch.cat(nodes), torch.cat(nids))

    best, log = -1, []
    for ep in range(a.epochs):
        model.train()
        perm = tr[torch.randperm(len(tr), device=dev)]
        tot, n = 0.0, 0
        for i in range(0, len(perm), a.batch):
            x, dom, rel, pad, nid, y = batch(perm[i:i + a.batch])
            hit, node = model(x, dom, rel, pad, nid)
            loss = F.cross_entropy(hit[~pad], y[~pad]) + F.cross_entropy(node, node_target(y, pad, nid)[0])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss) * len(x); n += len(x)
        v = evaluate(1).get("all", {})
        log.append(dict(epoch=ep + 1, loss=round(tot / n, 4), val=v))
        print("epoch %d loss %.4f val %s (%.0f s)" % (ep + 1, tot / n, {k: x for k, x in v.items() if k != "hit_recall_by_depth"}, time.time() - t0),
              file=sys.stderr, flush=True)
        if v.get("node_acc", 0) > best:                    # per-hit accuracy is dominated by the many distant hits
            best = v.get("node_acc", 0)
            torch.save(dict(cfg=model.cfg, state=model.state_dict()), os.path.join(a.out, "model.pt"))
    model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev, weights_only=False)["state"])
    res = dict(args=vars(a), n_examples=int(len(X)), log=log, **{name: evaluate(s) for s, name in enumerate(names) if s > 0})
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    for name in names[1:]:
        for k, v in res[name].items():
            print("%-19s %-9s %s" % (name, k, v))


if __name__ == "__main__":
    main()
