#!/usr/bin/env python
"""Latent states of a pretrained protein language model for every protein of a FASTA file.

For each protein: the mean over residues of the hidden states of a few layers (early, middle, last). Nothing
else is computed from the sequence here: no alignment, no database search.

usage (needs torch and transformers; the model folder is a local copy of an ESM-2 checkpoint):
  python research/lineage/embed.py --model MODELS/esm2_t33_650M_UR50D --fasta a.faa [b.faa ...] --out emb.npz
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch

LAYERS = (6, 20, 33)
MAX_LEN = 1022                           # residues kept per protein (the model's limit); longer proteins are cut


def read_fasta(paths):
    ids, seqs = [], []
    for p in paths:
        name, buf = None, []
        with open(p) as fh:
            for line in fh:
                if line.startswith(">"):
                    if name is not None:
                        ids.append(name); seqs.append("".join(buf))
                    name, buf = line[1:].split()[0], []
                else:
                    buf.append(line.strip())
        if name is not None:
            ids.append(name); seqs.append("".join(buf))
    return ids, seqs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fasta", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=60000, help="residues per batch")
    ap.add_argument("--pairs", type=float, default=2.5e7, help="residue pairs per batch (bounds the attention maps)")
    a = ap.parse_args()
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained(a.model)
    model = EsmModel.from_pretrained(a.model, add_pooling_layer=False, torch_dtype=torch.float16).cuda().eval()
    ids, seqs = read_fasta(a.fasta)
    seqs = [s.upper().replace("*", "")[:MAX_LEN] for s in seqs]
    order = np.argsort([-len(s) for s in seqs])
    out = np.zeros((len(seqs), len(LAYERS), model.config.hidden_size), dtype=np.float16)
    t0, done, i = time.time(), 0, 0
    with torch.no_grad():
        while i < len(order):
            L = len(seqs[order[i]]) + 2
            n = max(1, min(256, a.tokens // L, int(a.pairs // (L * L))))   # attention memory grows with the square of the length
            ix = order[i:i + n]
            enc = tok([seqs[j] for j in ix], return_tensors="pt", padding=True).to("cuda")
            hs = model(**enc, output_hidden_states=True).hidden_states
            keep = (enc["attention_mask"].bool() & (enc["input_ids"] != tok.cls_token_id) & (enc["input_ids"] != tok.eos_token_id)).unsqueeze(-1)
            for k, layer in enumerate(LAYERS):
                h = hs[layer].float()
                out[ix, k] = ((h * keep).sum(1) / keep.sum(1).clamp(min=1)).cpu().numpy().astype(np.float16)
            i += n; done += n
            if done % 5000 < n:
                print("%d / %d proteins, %.0f per second" % (done, len(seqs), done / (time.time() - t0)), file=sys.stderr, flush=True)
    np.savez(a.out, ids=np.array(ids), length=np.array([len(s) for s in seqs], dtype=np.int32), emb=out, layers=np.array(LAYERS))
    print("%d proteins in %.0f s -> %s" % (len(seqs), time.time() - t0, a.out), file=sys.stderr)


if __name__ == "__main__":
    main()
