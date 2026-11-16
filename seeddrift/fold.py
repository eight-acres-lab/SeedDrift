"""ESMFold structure prediction (HuggingFace `facebook/esmfold_v1`, local copy). Used in-process by the
ESMFold service (services/esmfold); needs torch with CUDA and transformers."""
from __future__ import annotations

import os
import re
import sys
import threading
import time

# Proteins of very different lengths leave the caching allocator with large blocks it cannot reuse: the memory the
# process holds then climbs to tens of GB although a single fold needs a few. Expandable segments let blocks grow and
# shrink in place; it must be set before torch touches the GPU. The cache is also released after every protein.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log(msg):
    print("[esmfold] " + msg, file=sys.stderr, flush=True)


class Folder:
    def __init__(self, model_dir, chunk=64, device="cuda"):
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
        t0 = time.time()
        self.torch, self.device = torch, device
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = EsmForProteinFolding.from_pretrained(model_dir).to(device).eval()
        self.model.esm = self.model.esm.half()      # the 3B language model in fp16; the trunk stays fp32
        self.model.trunk.set_chunk_size(chunk)
        self.lock = threading.Lock()
        log("model loaded from %s in %.0f s" % (model_dir, time.time() - t0))

    def fold(self, seqs, max_len=1000):
        """{id: sequence} -> {id: {plddt, pdb, seconds}} ({id: {skipped: reason}} for long ones)."""
        out = {}
        with self.lock:
            try:
                for name, seq in sorted(seqs.items(), key=lambda kv: len(kv[1])):
                    if len(seq) > max_len:
                        out[name] = dict(skipped="longer than %d aa" % max_len)
                        continue
                    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWY]", "X", seq.upper())
                    t1 = time.time()
                    ids = self.tok([seq], return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
                    with self.torch.no_grad():
                        o = self.model(ids)
                    pl = o["plddt"][0, :, 1].float()     # CA atom
                    mean = float(pl.mean()) * (100.0 if float(pl.max()) <= 1.0 else 1.0)
                    out[name] = dict(plddt=round(mean, 1), pdb=self.model.output_to_pdb(o)[0],
                                     seconds=round(time.time() - t1, 2))
                    del o, ids, pl
                    self.release()
            finally:                                      # also after a failed fold (out of memory on a long protein)
                self.release()
        return out

    def release(self):
        """Give cached GPU blocks back to the driver, so that the process holds what the model needs and no more."""
        if self.device != "cpu" and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def memory(self):
        """GiB of GPU memory in live tensors and held by this process's allocator."""
        if self.device == "cpu" or not self.torch.cuda.is_available():
            return dict(allocated_gb=0.0, reserved_gb=0.0)
        return dict(allocated_gb=round(self.torch.cuda.memory_allocated() / 2 ** 30, 2), reserved_gb=round(self.torch.cuda.memory_reserved() / 2 ** 30, 2))
