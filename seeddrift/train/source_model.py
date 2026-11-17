"""Source model v0 (strategy: model-pilot): does a gene come from the generating process of its genome?

Run by the train stage of the CI with the CUDA Python of the server (ci/envs/model-torch.pip.txt):

  python source_model.py fit   RUN_DIR CONFIG_JSON    train, log, checkpoint (resumable), then score
  python source_model.py score RUN_DIR CONFIG_JSON    score the evaluation genomes and the constructed development set

Reads data/train_data (stage train_data) and data/reference/library.npz; writes data/model/{model.npz, model.json,
eval_scores.npz, dev_eval.npz} and data/train/logs (seeddrift.train.logger). Inputs are raw data only: codons of the
gene and its neighbours, and its hits as residue-by-residue alignment tracks with the identity of the hit's proteome.
No identity, coverage, score or lineage path enters the network; lineage is used only to hide the genome's own clade
(and an inserted gene's donor clade) from the hits, at a random depth during training and at genus depth in evaluation.

Training signal (no HGT labels): in training genomes, the centre gene of a window is, half of the time, replaced by a
gene of another training genome (its codons and its own hits), the donor drawn across distance strata (same genus to
another kingdom, and mobile elements); the model predicts the replacement. Natives carry the soft label 0.05 because
training genomes contain unlabelled transfers. The constructed development set uses held-out recipients and held-out
donors only. This file must never import an evaluation generator (gate G2 static).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
from seeddrift.train.logger import Diverged, TrainLogger, save_npz  # noqa: E402

RANKS = ("phylum", "class", "order", "family", "genus", "species")
DEPTHS = ("species", "genus", "family", "order")
STRATA = ("same genus", "same family", "same order", "same phylum", "other phylum", "other kingdom", "mobile element")
GAP = 22
WIN = 8                                                  # neighbours on each side
# standard genetic code over codon tokens 1..64 (ACGT order) -> amino-acid tokens 1..20 (ACDEFGHIKLMNPQRSTVWY), stop 21
_AA = "ACDEFGHIKLMNPQRSTVWY"
_CODE = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"


def _code_table():
    t = np.full(66, 21, dtype=np.int64)
    for c in range(64):
        a = _CODE[c]
        t[c + 1] = _AA.index(a) + 1 if a in _AA else 21
    return t


CODE = _code_table()
KINGDOM = {"bacteria": "prok", "archaea": "prok", "mobile.plasmid": "mobile", "viruses.phage": "mobile"}


class Genome:
    def __init__(self, meta, path):
        self.meta = meta
        z = np.load(path)
        self.codons, self.coff = z["codons"], z["codon_off"]
        self.hoff, self.hprot, self.hqs = z["hit_off"], z["hit_prot"], z["hit_qs"]
        self.track, self.toff = z["track"], z["track_off"]
        rep, start = z["replicon"], z["start"]
        self.n = len(self.coff) - 1
        order = np.lexsort((start, rep))                 # genome order within replicons
        self.pos = np.empty(self.n, dtype=np.int64)
        self.pos[order] = np.arange(self.n)
        self.order, self.rep, self.start, self.end, self.strand = order, rep, start, z["end"], z["strand"]
        self.rep_len = z["replicon_len"]
        self.ranks = np.array(meta["ranks"], dtype=np.int64)

    def neighbours(self, i):
        p = self.pos[i]
        out = []
        for d in range(-WIN, WIN + 1):
            q = p + d
            j = self.order[q] if 0 <= q < self.n else -1
            out.append(j if j >= 0 and self.rep[j] == self.rep[i] else -1)
        return out


def kingdom(group):
    if group.startswith("eukaryota."):
        return "euk"
    if group in ("viruses.eukaryotic", "viruses.giant"):
        return "euk_virus"
    return KINGDOM.get(group, group)


def stratum(r, d):
    """Distance stratum of donor genome d for recipient genome r."""
    if d.meta["group"] in ("mobile.plasmid", "viruses.phage") and r.meta["group"] not in ("mobile.plasmid", "viruses.phage"):
        return 6
    if kingdom(r.meta["group"]) != kingdom(d.meta["group"]):
        return 5
    for k, col in ((0, 4), (1, 3), (2, 2), (3, 0)):      # genus, family, order, phylum
        if r.ranks[col] and r.ranks[col] == d.ranks[col]:
            return k
    return 4


class Data:
    def __init__(self, run_dir, cfg):
        with open(os.path.join(run_dir, "data", "train_data", "index.json")) as f:
            self.index = json.load(f)
        lib = np.load(os.path.join(run_dir, self.index["library"]))
        self.lib_ranks = lib["ranks"].astype(np.int64)                  # rows: proteomes; columns: RANKS
        self.n_lib = len(self.lib_ranks)
        self.g = {m["id"]: Genome(m, os.path.join(run_dir, "data", "train_data", m["id"] + ".npz")) for m in self.index["genomes"]}
        self.by_split = {}
        for m in self.index["genomes"]:
            if self.g[m["id"]].n:
                self.by_split.setdefault(m["split"], []).append(m["id"])
        self.maxc = int(cfg.get("max_codons", 512))
        self.kc, self.kn, self.maxt = int(cfg.get("hits_centre", 16)), int(cfg.get("hits_neighbour", 4)), int(cfg.get("max_track", 512))

    # ---- one gene of one genome as arrays; `hide` = list of (column, value) whose hits are removed
    def gene(self, gen, i, hide, k):
        c = gen.codons[gen.coff[i]:gen.coff[i + 1]][: self.maxc].astype(np.int64)
        q = CODE[c]                                                    # query amino acids (for the alignment pairs)
        tracks, prots = [], []
        for h in range(gen.hoff[i], gen.hoff[i + 1]):
            if len(tracks) >= k:
                break
            pr = gen.hprot[h]
            if any(self.lib_ranks[pr, col] == val for col, val in hide):
                continue
            t = gen.track[gen.toff[h]:gen.toff[h + 1]].astype(np.int64)
            qs = gen.hqs[h]
            qq = q[qs:qs + len(t)] if qs < len(q) else np.zeros(0, np.int64)
            m = min(len(t), len(qq), self.maxt)
            tracks.append((qq[:m] * 23 + t[:m]))                      # pair token: query residue x target residue
            prots.append(pr)
        return c, tracks, prots

    def hide_for(self, gen, depth):
        col = RANKS.index(depth)
        for cc in range(col, len(RANKS)):                              # the clade at `depth`, or the nearest defined rank below
            if gen.ranks[cc]:
                return [(cc, int(gen.ranks[cc]))]
        return []

    def window(self, gen, i, donor=None, depth="genus"):
        """The 17 slots of gene i of `gen`; the centre replaced by donor = (genome, gene) if given."""
        hide = self.hide_for(gen, depth)
        slots = []
        for s, j in enumerate(gen.neighbours(i)):
            if s == WIN and donor is not None:
                dg, dj = donor
                c, tr, pr = self.gene(dg, dj, hide + self.hide_for(dg, depth), self.kc)
                strand = 1
            elif j < 0:
                slots.append(None)
                continue
            else:
                c, tr, pr = self.gene(gen, j, hide, self.kc if s == WIN else self.kn)
                strand = int(gen.strand[j] * gen.strand[i])
            jj = i if j < 0 else j
            dist = 0.0 if s == WIN else math.log1p(abs(int(gen.start[jj]) - int(gen.start[i])))
            edge = math.log1p(min(int(gen.start[i]), int(gen.rep_len[gen.rep[i]]) - int(gen.end[i])) if s == WIN else 0)
            slots.append(dict(c=c, tracks=tr, prots=pr, feat=[strand, dist / 15.0, edge / 15.0, math.log1p(int(gen.rep_len[gen.rep[i]])) / 20.0, (s - WIN) / WIN]))
        return slots

    def source(self, gen, rng, exclude, n=32):
        idx = [j for j in rng.choice(gen.n, size=min(n + 1, gen.n), replace=False) if j != exclude][:n]
        return [gen.codons[gen.coff[j]:gen.coff[j + 1]][: self.maxc].astype(np.int64) for j in idx]


def collate(batch, torch):
    """batch: list of (slots, source codon lists, label) -> padded tensors."""
    B, S = len(batch), 2 * WIN + 1
    maxc = max([len(sl["c"]) for slots, _, _ in batch for sl in slots if sl] + [len(c) for _, src, _ in batch for c in src] + [1])
    codons = torch.zeros(B, S, maxc, dtype=torch.long)
    present = torch.zeros(B, S, dtype=torch.bool)
    feats = torch.zeros(B, S, 5)
    nh = max([len(sl["tracks"]) for slots, _, _ in batch for sl in slots if sl] + [1])
    maxt = max([len(t) for slots, _, _ in batch for sl in slots if sl for t in sl["tracks"]] + [1])
    tracks = torch.zeros(B, S, nh, maxt, dtype=torch.long)
    prots = torch.zeros(B, S, nh, dtype=torch.long)
    hmask = torch.zeros(B, S, nh, dtype=torch.bool)
    ns = max(len(src) for _, src, _ in batch)
    src = torch.zeros(B, ns, maxc, dtype=torch.long)
    labels = torch.zeros(B)
    for b, (slots, srcs, y) in enumerate(batch):
        labels[b] = y
        for s, sl in enumerate(slots):
            if not sl:
                continue
            present[b, s] = True
            codons[b, s, :len(sl["c"])] = torch.from_numpy(sl["c"])
            feats[b, s] = torch.tensor(sl["feat"], dtype=torch.float32)
            for h, (t, p) in enumerate(zip(sl["tracks"], sl["prots"])):
                tracks[b, s, h, :len(t)] = torch.from_numpy(t) + 1      # 0 = padding
                prots[b, s, h] = int(p) + 1
                hmask[b, s, h] = True
        for k, c in enumerate(srcs):
            src[b, k, :len(c)] = torch.from_numpy(c)
    return dict(codons=codons, present=present, feats=feats, tracks=tracks, prots=prots, hmask=hmask, src=src, labels=labels)


def build(torch, n_lib, d=128):
    nn = torch.nn

    class Pool(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w = nn.Linear(dim, 1)

        def forward(self, x, mask):                                  # x [N, L, D], mask [N, L]
            a = self.w(x).squeeze(-1).masked_fill(~mask, -1e4)
            a = torch.softmax(a, -1) * mask.any(-1, keepdim=True)
            return (a.unsqueeze(-1) * x).sum(1)

    class Conv(nn.Module):
        def __init__(self, vocab, dim):
            super().__init__()
            self.emb = nn.Embedding(vocab, 64, padding_idx=0)
            self.c1 = nn.Conv1d(64, dim, 9, padding=4)
            self.c2 = nn.Conv1d(dim, dim, 9, padding=8, dilation=2)
            self.pool = Pool(dim)

        def forward(self, tok):                                      # tok [N, L]
            mask = tok > 0
            x = self.emb(tok).transpose(1, 2)
            x = torch.nn.functional.gelu(self.c1(x))
            x = torch.nn.functional.gelu(self.c2(x) + x).transpose(1, 2)
            return self.pool(x, mask)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.codon = Conv(66, d)
            self.hit = Conv(23 * 23 + 1, d)
            self.prot = nn.Embedding(n_lib + 1, 32, padding_idx=0)
            self.hit_proj = nn.Linear(d + 32, d)
            self.hit_set = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.1), 1)
            self.hit_pool = Pool(d)
            self.no_hit = nn.Parameter(torch.zeros(d))
            self.slot = nn.Linear(2 * d + 5, d)
            self.ctx = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, batch_first=True, dropout=0.1), 2)
            self.src_proj = nn.Linear(d, d)
            self.head = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))

        def forward(self, x):
            B, S, L = x["codons"].shape
            pres = x["present"].view(-1)                                 # encode only the slots that hold a gene
            cenc = self.codon(x["codons"].view(B * S, L)[pres])
            cv = torch.zeros(B * S, cenc.shape[-1], device=cenc.device, dtype=cenc.dtype)
            cv[pres] = cenc
            cv = cv.view(B, S, -1)
            _, _, H, T = x["tracks"].shape
            hm = x["hmask"].view(B * S, H)
            flat = hm.view(-1)                                           # encode only the real hit tracks, packed
            enc = self.hit(x["tracks"].view(B * S * H, T)[flat])
            tv = torch.zeros(B * S * H, enc.shape[-1], device=enc.device, dtype=enc.dtype)
            tv[flat] = enc
            tv = self.hit_proj(torch.cat([tv.view(B * S, H, -1), self.prot(x["prots"].view(B * S, H))], -1))
            any_hit = hm.any(-1)
            # a gene without hits masks nothing (a fully masked row gives NaN); its output is replaced by no_hit below
            tv = self.hit_set(tv, src_key_padding_mask=~hm & any_hit.unsqueeze(-1))
            hv = self.hit_pool(tv, hm)
            hv = torch.where(any_hit.unsqueeze(-1), hv, self.no_hit.expand_as(hv)).view(B, S, -1)
            slots = self.slot(torch.cat([cv, hv, x["feats"]], -1))
            Bs, Ns, Ls = x["src"].shape
            sv = self.codon(x["src"].view(Bs * Ns, Ls)).view(Bs, Ns, -1)
            smask = (x["src"] > 0).any(-1)
            source = (sv * smask.unsqueeze(-1)).sum(1) / smask.sum(1, keepdim=True).clamp(min=1)
            seq = torch.cat([self.src_proj(source).unsqueeze(1), slots], 1)
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=seq.device), ~x["present"]], 1)
            out = self.ctx(seq, src_key_padding_mask=pad)
            centre_in, centre_out = slots[:, WIN], out[:, 1 + WIN]
            return self.head(torch.cat([centre_out, centre_in, source, centre_in - source], -1)).squeeze(-1)

    return Model()


def sample(data, rng, genomes, depth=None):
    """One training example: (slots, source, label, stratum, recipient group)."""
    groups = sorted({data.g[g].meta["group"] for g in genomes})
    grp = groups[rng.integers(len(groups))]
    pool = [g for g in genomes if data.g[g].meta["group"] == grp]
    gen = data.g[pool[rng.integers(len(pool))]]
    i = int(rng.integers(gen.n))
    depth = depth or DEPTHS[rng.integers(len(DEPTHS))]
    if rng.random() < 0.5:
        others = [data.g[g] for g in genomes if g != gen.meta["id"]]
        by = {}
        for o in others:
            by.setdefault(stratum(gen, o), []).append(o)
        ks = sorted(by)
        k = ks[rng.integers(len(ks))]
        dg = by[k][rng.integers(len(by[k]))]
        dj = int(rng.integers(dg.n))
        return data.window(gen, i, (dg, dj), depth), data.source(gen, rng, i), 1.0, k, grp
    return data.window(gen, i, None, depth), data.source(gen, rng, i), 0.0, -1, grp


def dev_set(data, seed, n_per=120):
    """Constructed development examples: held-out recipients, held-out donors, genus hidden; fixed by the seed."""
    rng = np.random.default_rng(seed)
    ho = sorted(data.by_split.get("heldout", []))
    out = []
    for gid in ho:
        gen = data.g[gid]
        others = [data.g[g] for g in ho if g != gid]
        if not others:
            continue
        by = {}
        for o in others:
            by.setdefault(stratum(gen, o), []).append(o)
        for _ in range(n_per):
            i = int(rng.integers(gen.n))
            if rng.random() < 0.5:
                ks = sorted(by)
                k = ks[rng.integers(len(ks))]
                dg = by[k][rng.integers(len(by[k]))]
                out.append((gid, i, dg.meta["id"], int(rng.integers(dg.n)), k))
            else:
                out.append((gid, i, "", -1, -1))
    return out


def predict(model, data, torch, dev, examples, rng_seed=0, batch=64):
    rng = np.random.default_rng(rng_seed)
    scores = []
    model.eval()
    with torch.no_grad():
        for b0 in range(0, len(examples), batch):
            items = []
            for gid, i, did, dj, hidden in examples[b0:b0 + batch]:
                gen = data.g[gid]
                donor = (data.g[did], dj) if did else None
                slots = data.window(gen, i, donor, "genus")
                if hidden:                                           # extra hidden genera (donors of an evaluation scenario)
                    for sl in slots:
                        if sl:
                            keep = [k for k, p in enumerate(sl["prots"]) if data.lib_ranks[p, RANKS.index("genus")] not in hidden]
                            sl["tracks"] = [sl["tracks"][k] for k in keep]
                            sl["prots"] = [sl["prots"][k] for k in keep]
                items.append((slots, data.source(gen, rng, i), 0.0))
            x = {k: v.to(dev) for k, v in collate(items, torch).items()}
            scores += torch.sigmoid(model(x)).float().cpu().tolist()
    model.train()
    return np.array(scores, dtype=np.float64)


def auroc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(allv):                                          # ties at their mean rank
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit(run_dir, cfg):
    import torch
    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Data(run_dir, cfg)
    train_ids = sorted(data.by_split.get("train", []))
    model = build(torch, data.n_lib).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=0.01)
    steps, batch = int(cfg["steps"]), int(cfg["batch"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    folder = os.path.join(run_dir, "data", "train", "logs")
    resume = bool(cfg.get("resume")) and os.path.exists(os.path.join(folder, "last.npz"))
    log = TrainLogger(folder, commit=cfg.get("commit", ""), dataset_ids=[data.index["dataset_id"]], resume=resume)
    start = 0
    if resume:
        last = log.last()
        state = {k[2:]: torch.from_numpy(v) for k, v in last.items() if k.startswith("m:")}
        model.load_state_dict(state)
        start = int(last["step"])
    rng = np.random.default_rng([seed, start])                     # data order from the step reached: a resume continues
    dev_ex = [(g, i, d, j, None) for g, i, d, j, _ in dev_set(data, seed)]
    dev_y = np.array([1 if d else 0 for _, _, d, _, _ in dev_ex])
    for _ in range(start):
        sched.step()
    t0 = time.time()
    for step in range(start + 1, steps + 1):
        items, groups = [], []
        for _ in range(batch):
            slots, src, y, k, grp = sample(data, rng, train_ids)
            items.append((slots, src, 0.05 if y == 0 else 1.0))
            groups.append(grp)
        x = {k: v.to(dev) for k, v in collate(items, torch).items()}
        logit = model(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, x["labels"])
        opt.zero_grad()
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        sched.step()
        log.count(groups)
        try:
            log.step(step, float(loss), lr=float(sched.get_last_lr()[0]), grad_norm=gn, examples_per_s=batch * (step - start) / max(time.time() - t0, 1e-9))
        except Diverged:
            print("diverged at step %d" % step, file=sys.stderr)
            sys.exit(3)
        if step % int(cfg.get("eval_every", 500)) == 0 or step == steps:
            a = auroc(dev_y, predict(model, data, torch, dev, dev_ex)) if dev_ex else None
            if a is not None:                                          # undefined without both classes (tiny fixtures)
                log.evaluate(step, dev_auroc=a)
            print("step %d loss %.4f dev AUROC %s" % (step, float(loss), "n/a" if a is None else "%.4f" % a), file=sys.stderr, flush=True)
        if step % int(cfg.get("ckpt_every", 1000)) == 0 or step == steps:
            log.checkpoint(step, {"m:" + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}, final=step == steps)
    log.save_seen("")
    model_dir = os.path.join(run_dir, "data", "model")
    os.makedirs(model_dir, exist_ok=True)
    save_npz(os.path.join(model_dir, "model.npz"), **{"m:" + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    score(run_dir, cfg, model=model, data=data)


def score(run_dir, cfg, model=None, data=None):
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data or Data(run_dir, cfg)
    if model is None:
        z = np.load(os.path.join(run_dir, "data", "model", "model.npz"))
        model = build(torch, data.n_lib).to(dev)
        model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files})
    seed = int(cfg["seed"])
    model_dir = os.path.join(run_dir, "data", "model")
    # evaluation genomes: every gene, own genus hidden, and the scenario's donor genera hidden too (as for the comparators)
    hidden = cfg.get("hidden_genera") or {}
    ev, keys = [], []
    for gid in sorted(data.by_split.get("eval", [])):
        h = set(int(x) for x in hidden.get(gid, []))
        for i in range(data.g[gid].n):
            ev.append((gid, i, "", -1, h))
            keys.append(gid)
    s_eval = predict(model, data, torch, dev, ev) if ev else np.zeros(0)
    save_npz(os.path.join(model_dir, "eval_scores.npz"), genome=np.array(keys, dtype="S"), gene=np.array([e[1] for e in ev], dtype=np.int64), score=s_eval)
    dex = dev_set(data, seed)
    s_dev = predict(model, data, torch, dev, [(g, i, d, j, None) for g, i, d, j, _ in dex]) if dex else np.zeros(0)
    save_npz(os.path.join(model_dir, "dev_eval.npz"), recipient=np.array([e[0] for e in dex], dtype="S"), gene=np.array([e[1] for e in dex], dtype=np.int64),
             donor=np.array([e[2] for e in dex], dtype="S"), donor_gene=np.array([e[3] for e in dex], dtype=np.int64),
             stratum=np.array([e[4] for e in dex], dtype=np.int64), score=s_dev)


def main():
    mode, run_dir, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(cfg_path) as f:
        cfg = json.load(f)
    if mode == "fit":
        fit(run_dir, cfg)
    elif mode == "score":
        score(run_dir, cfg)
    else:
        raise SystemExit("mode is fit or score")


if __name__ == "__main__":
    main()
