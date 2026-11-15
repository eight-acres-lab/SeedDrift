#!/usr/bin/env python
"""Per-residue states of a pretrained protein language model for the genes of an array data set.

For every position of every gene: the model's log-probabilities of the 20 amino acids at that position (what the
site tolerates) and a low-dimensional projection of its last hidden state (what kind of site it is). Stored in
the same order as the gene codes of the index, so the array model can read them as context of each alignment
column. Members are not embedded: the language model is asked about the gene, the members are judged against it.

usage (needs torch and transformers; the model folder is a local copy of an ESM-2 checkpoint):
  python research/array/sites.py --model MODELS/esm2_t33_650M_UR50D --index DIR/a.index.npz --fasta queries_a.faa --out DIR/a.site.npy \
      [--pca DIR/site_pca.npz]     (the projection is fitted on the first data set and reused for the others)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

AA = "ACDEFGHIKLMNPQRSTVWY"
MAX_LEN = 1022                           # residues the model reads; positions beyond stay zero
N_PC = 28
DIM = 20 + N_PC


def read_fasta(paths):
    out, name, buf = {}, None, []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                if line.startswith(">"):
                    if name is not None:
                        out[name] = "".join(buf)
                    name, buf = line[1:].split()[0], []
                else:
                    buf.append(line.strip())
        if name is not None:
            out[name] = "".join(buf)
            name, buf = None, []
    return out


def batches(seqs, order, tokens, pairs):
    i = 0
    while i < len(order):
        L = len(seqs[order[i]]) + 2
        n = max(1, min(256, tokens // L, int(pairs // (L * L))))
        yield order[i:i + n]
        i += n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--index", required=True); ap.add_argument("--fasta", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--pca", required=True)
    ap.add_argument("--tokens", type=int, default=40000); ap.add_argument("--pairs", type=float, default=2.0e7)
    a = ap.parse_args()
    from transformers import AutoTokenizer, EsmForMaskedLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = EsmForMaskedLM.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    aa_ids = torch.tensor(tok.convert_tokens_to_ids(list(AA)), device="cuda")
    z = np.load(a.index)
    fa = read_fasta(a.fasta)
    ids, start, L = [str(x) for x in z["gene_id"]], z["code_start"], z["gene_L"]
    seqs = [fa[g].upper().replace("*", "")[:MAX_LEN] for g in ids]
    order = np.argsort([-len(s) for s in seqs], kind="stable")

    def forward(ix):
        enc = tok([seqs[j] for j in ix], return_tensors="pt", padding=True).to("cuda")
        o = model(**enc, output_hidden_states=True)
        lp = torch.log_softmax(o.logits.float()[:, :, aa_ids], -1)
        return lp[:, 1:], o.hidden_states[-1].float()[:, 1:]              # drop the start token

    with torch.no_grad():
        if os.path.exists(a.pca):
            p = np.load(a.pca)
            mean, comp = torch.tensor(p["mean"], device="cuda"), torch.tensor(p["comp"], device="cuda")
        else:                                                              # fit the projection on a random sample of proteins
            rng = np.random.default_rng(0)
            sample = rng.choice(len(seqs), min(3000, len(seqs)), replace=False)
            sample = sample[np.argsort([-len(seqs[j]) for j in sample], kind="stable")]
            hs = []
            for ix in batches(seqs, sample, a.tokens, a.pairs):
                _, h = forward(ix)
                for k, j in enumerate(ix):
                    n = len(seqs[j])
                    hs.append(h[k, :n][torch.randperm(n, device="cuda")[:40]])
            H = torch.cat(hs)
            mean = H.mean(0)
            _, S, V = torch.pca_lowrank(H - mean, q=N_PC, center=False, niter=4)
            comp = V / (S / (len(H) - 1) ** 0.5)                           # unit variance per component
            np.savez(a.pca, mean=mean.cpu().numpy(), comp=comp.cpu().numpy())
            print("projection fitted on %d residues" % len(H), file=sys.stderr, flush=True)
        out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float16, shape=(int(start[-1]), DIM))
        t0, done = time.time(), 0
        for ix in batches(seqs, order, a.tokens, a.pairs):
            lp, h = forward(ix)
            pc = (h - mean) @ comp
            f = torch.cat([lp / 4.0, pc.clamp(-6, 6) / 2.0], -1).cpu().numpy().astype(np.float16)
            for k, j in enumerate(ix):
                n = min(len(seqs[j]), int(L[j]))
                out[start[j]:start[j] + n] = f[k, :n]
            done += len(ix)
            if done % 5000 < len(ix):
                print("%d / %d genes, %.0f per second" % (done, len(seqs), done / (time.time() - t0)), file=sys.stderr, flush=True)
        out.flush()
    print("%d genes in %.0f s -> %s" % (len(seqs), time.time() - t0, a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
