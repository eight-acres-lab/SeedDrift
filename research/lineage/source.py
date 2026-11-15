#!/usr/bin/env python
"""Source representation: does a protein carry the signature of the genome it comes from?

Every protein is described by latent states of a pretrained protein language model (embed.py). A small head
is trained with one signal only: proteins of the same genome are pulled together; proteins of other genomes
are pushed away, including homologs of the same family from other genomes (hard negatives, homologs.py).
No alignment score, no taxonomy and no transfer label enters training. Taxonomy is used afterwards, to hold
out whole classes for testing and to report results by how distant the other genome is.

Tests on held-out lineages (genomes never seen in training):
  verify   a protein against the centroid of its own genome (leave-one-out) versus proteins of other genomes
  swap     the native protein versus an ortholog from another genome put in its place (same family, other
           source): the family-controlled test of whether a foreign gene would stand out
Baselines: the same head trained on amino-acid composition alone; the raw latent states without training.

usage (needs torch):
  python research/lineage/source.py --library L.npz --queries emb_queries.npz --homologs emb_homologs.npz \
      --pairs homologs.tsv --fasta queries_a.faa queries_b.faa homologs.faa --out RUN [--bench name=emb_bench_x.npz:x.faa ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "placement"))
sys.path.insert(0, HERE)
from placenet import Library  # noqa: E402
from train import split_of  # noqa: E402
from embed import read_fasta  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
DEPTH = ["other domain", "same domain", "same phylum", "same class", "same order", "same family", "same genus"]


def composition(seqs):
    """Amino-acid frequencies and log length: the classical alignment-free signature, as a baseline."""
    out = np.zeros((len(seqs), 21), dtype=np.float32)
    ix = {a: i for i, a in enumerate(AA)}
    for k, s in enumerate(seqs):
        for ch in s:
            j = ix.get(ch)
            if j is not None:
                out[k, j] += 1
        n = max(1.0, out[k, :20].sum())
        out[k, :20] /= n
        out[k, 20] = np.log(max(1, len(s))) / 7.0
    return out


class Head(nn.Module):
    def __init__(self, d_in, d=512, out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, out))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon(z, label, tau):
    """Supervised contrastive loss: items with the same label are positives; items without a positive in the
    batch are negatives only."""
    sim = z @ z.t() / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    pos = (label[:, None] == label[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    n = pos.sum(1)
    has = n > 0
    return -(logp.masked_fill(~pos, 0.0).sum(1)[has] / n[has]).mean()


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None
    x = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    rank = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    return round(float((rank[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))), 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--homologs", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--fasta", nargs="+", required=True)
    ap.add_argument("--bench", nargs="*", default=[], help="name=embeddings.npz:proteins.faa of a genome to score without any search")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--genomes", type=int, default=48)
    ap.add_argument("--per-genome", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rnd = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(a.out, exist_ok=True)
    lib = Library(a.library)
    split_p = split_of(lib)                                  # per library proteome: 0 train, 1 val, 2 unseen class, 3 benchmark lineage
    zq, zh = np.load(a.queries), np.load(a.homologs)
    ids = np.concatenate([zq["ids"], zh["ids"]])
    emb = np.concatenate([zq["emb"], zh["emb"]]).reshape(len(ids), -1).astype(np.float32)
    nq = len(zq["ids"])
    genome = np.array([int(str(i).rpartition("|")[2]) for i in ids])
    pidx = np.array([lib.pidx.get(int(t), -1) for t in genome])
    seq = dict(zip(*read_fasta(a.fasta)))
    comp = composition([seq.get(str(i), "") for i in ids])
    row = {str(i): k for k, i in enumerate(ids)}
    hom = {}                                                 # query row -> [(homolog row, rank)]
    for r in csv.DictReader(open(a.pairs), delimiter="\t"):
        if r["query"] in row and r["homolog"] in row:
            hom.setdefault(row[r["query"]], []).append((row[r["homolog"]], int(r["rank"])))
    sp = np.where(pidx >= 0, split_p[np.clip(pidx, 0, None)], -1)
    q_rows = np.arange(nq)
    by_genome = {}
    for i in q_rows:
        by_genome.setdefault(genome[i], []).append(i)
    train_g = [g for g, v in by_genome.items() if sp[v[0]] == 0 and len(v) >= a.per_genome]
    print("proteins %d (queries %d, homologs %d); genomes with queries: train %d, val %d, unseen class %d, benchmark lineage %d" % (
        len(ids), nq, len(ids) - nq, len(train_g), sum(sp[v[0]] == 1 for v in by_genome.values()),
        sum(sp[v[0]] == 2 for v in by_genome.values()), sum(sp[v[0]] == 3 for v in by_genome.values())), file=sys.stderr)

    feats = {"plm": emb, "composition": comp}
    Z, log = {}, {}
    for name, X in feats.items():
        tr_rows = np.array([i for g in train_g for i in by_genome[g]])
        mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0) + 1e-6
        Xn = torch.from_numpy((X - mu) / sd).to(dev)
        Z["raw_" + name] = F.normalize(Xn, dim=-1).cpu().numpy()
        head = Head(X.shape[1]).to(dev)
        opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=a.steps, pct_start=0.05)
        lab = torch.from_numpy(genome).to(dev)
        t0, hist = time.time(), []
        for step in range(a.steps):
            gs = rnd.choice(len(train_g), a.genomes, replace=False)
            qs = np.concatenate([rnd.choice(by_genome[train_g[g]], a.per_genome, replace=False) for g in gs])
            # hard negatives: homologs of these queries from genomes that are not held out for testing
            hs = [h for q in qs for h, _ in hom.get(int(q), []) if sp[h] in (0, 1)]
            ix = torch.from_numpy(np.concatenate([qs, np.array(hs, dtype=np.int64)])).to(dev)
            loss = supcon(head(Xn[ix]), lab[ix], a.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            hist.append(float(loss))
            if (step + 1) % 500 == 0:
                print("%s step %d loss %.3f (%.0f s)" % (name, step + 1, np.mean(hist[-500:]), time.time() - t0), file=sys.stderr, flush=True)
        head.eval()
        with torch.no_grad():
            Z[name] = torch.cat([head(Xn[i:i + 8192]) for i in range(0, len(Xn), 8192)]).cpu().numpy()
        log[name] = dict(final_loss=float(np.mean(hist[-200:])), d_in=int(X.shape[1]))
        torch.save(dict(state=head.state_dict(), mu=mu, sd=sd, d_in=int(X.shape[1])), os.path.join(a.out, "head_%s.pt" % name))

    res = dict(args=vars(a), train=log, tests={})
    for s, sname in ((2, "unseen_class"), (3, "bench_lineage")):
        gs = [g for g, v in by_genome.items() if sp[v[0]] == s and len(v) >= 20]
        out = dict(n_genomes=len(gs), verify={}, swap={})
        for name, z in Z.items():
            cent = {g: z[by_genome[g]].sum(0) for g in gs}
            # verify: own proteins (leave-one-out centroid) against proteins of the other held-out genomes
            pos, neg, negd = [], [], []
            for g in gs:
                rows = np.array(by_genome[g])
                loo = cent[g][None, :] - z[rows]
                pos.append((z[rows] * F.normalize(torch.from_numpy(loo), dim=-1).numpy()).sum(1))
                c = cent[g] / np.linalg.norm(cent[g])
                for g2 in gs:
                    if g2 != g:
                        r2 = np.array(by_genome[g2])
                        neg.append(z[r2] @ c)
                        negd.append(np.full(len(r2), lib.S[pidx[rows[0]], pidx[r2[0]]]))
            pos, neg, negd = np.concatenate(pos), np.concatenate(neg), np.concatenate(negd)
            out["verify"][name] = dict(all=auroc(pos, neg), **{DEPTH[d]: auroc(pos, neg[negd == d]) for d in range(7) if (negd == d).sum() >= 200})
            # swap: the native protein against its homolog from another genome, both scored against the native's genome
            nat, forn, frank, fdepth, pair = [], [], [], [], []
            for g in gs:
                for q in by_genome[g]:
                    if int(q) not in hom:
                        continue
                    c = cent[g] - z[q]
                    c = c / np.linalg.norm(c)
                    sn = float(z[q] @ c)
                    nat.append(sn)
                    for h, rank in hom[int(q)]:
                        sf = float(z[h] @ c)
                        forn.append(sf); frank.append(rank); pair.append(sn > sf)
                        fdepth.append(lib.S[pidx[q], pidx[h]] if pidx[h] >= 0 else -1)
            nat, forn, frank, fdepth, pair = map(np.array, (nat, forn, frank, fdepth, pair))
            sw = dict(n_native=int(len(nat)), n_foreign=int(len(forn)), all=auroc(nat, forn), native_higher=round(float(pair.mean()), 4))
            sw["by_rank"] = {int(r): dict(n=int((frank == r).sum()), auroc=auroc(nat, forn[frank == r]), native_higher=round(float(pair[frank == r].mean()), 4)) for r in sorted(set(frank))}
            sw["by_depth"] = {DEPTH[d]: dict(n=int((fdepth == d).sum()), auroc=auroc(nat, forn[fdepth == d]), native_higher=round(float(pair[fdepth == d].mean()), 4))
                              for d in range(7) if (fdepth == d).sum() >= 100}
            out["swap"][name] = sw
        res["tests"][sname] = out
        print("\n## %s: %d genomes" % (sname, len(gs)))
        names = list(Z)
        print("| test | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        keys = ["all"] + [k for k in DEPTH if k in out["verify"][names[0]]]
        for k in keys:
            print("| verify, others from %s | %s |" % (k, " | ".join("%.3f" % out["verify"][n][k] if out["verify"][n].get(k) is not None else "–" for n in names)))
        for k, v in out["swap"][names[0]]["by_depth"].items():
            print("| swap, ortholog from %s (n=%d) | %s |" % (k, v["n"], " | ".join("%.3f / %.2f" % (out["swap"][n]["by_depth"][k]["auroc"], out["swap"][n]["by_depth"][k]["native_higher"]) for n in names)))
        for k, v in out["swap"][names[0]]["by_rank"].items():
            print("| swap, homolog of rank %d (n=%d) | %s |" % (k, v["n"], " | ".join("%.3f / %.2f" % (out["swap"][n]["by_rank"][k]["auroc"], out["swap"][n]["by_rank"][k]["native_higher"]) for n in names)))
        print("verify: AUROC. swap: AUROC / share of pairs in which the native protein is closer to its genome than the homolog.")
    json.dump(res, open(os.path.join(a.out, "tests.json"), "w"), indent=1)

    # genomes scored without any search: distance of every gene from the centre of its own genome
    for spec in a.bench:
        name, rest = spec.split("=", 1)
        epath, fpath = rest.split(":", 1)
        zb = np.load(epath)
        gids, gseqs = read_fasta([fpath])
        sq = dict(zip(gids, gseqs))
        Xb = {"plm": zb["emb"].reshape(len(zb["ids"]), -1).astype(np.float32), "composition": composition([sq.get(str(i), "") for i in zb["ids"]])}
        cols = {}
        for fname in feats:
            ck = torch.load(os.path.join(a.out, "head_%s.pt" % fname), map_location=dev, weights_only=False)
            head = Head(ck["d_in"]).to(dev)
            head.load_state_dict(ck["state"]); head.eval()
            xn = torch.from_numpy((Xb[fname] - ck["mu"]) / ck["sd"]).to(dev)
            with torch.no_grad():
                variants = {fname: head(xn).cpu().numpy(), "raw_" + fname: F.normalize(xn, dim=-1).cpu().numpy()}
            for vname, z in variants.items():
                c = z.mean(0)
                for _ in range(3):                           # centre of the bulk: drop the tenth of the genes farthest from it and recompute
                    s = z @ (c / np.linalg.norm(c))
                    c = z[s >= np.quantile(s, 0.1)].mean(0)
                cols[vname] = -(z @ (c / np.linalg.norm(c)))
        with open(os.path.join(a.out, "scores_%s.tsv" % name), "w") as fo:
            fo.write("gene_id\t" + "\t".join(cols) + "\n")
            for k, gid in enumerate(zb["ids"]):
                fo.write("%s\t%s\n" % (gid, "\t".join("%.5f" % cols[c][k] for c in cols)))
        print("scored %s: %d genes" % (name, len(zb["ids"])), file=sys.stderr)


if __name__ == "__main__":
    main()
