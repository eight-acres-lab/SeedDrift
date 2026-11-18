"""The numpy collate of the source model gives the same tensors as the element-by-element torch version it replaced
(2026-11-18). Needs torch: skipped where it is not installed (the CI environment); run on the server with the CUDA
Python (python3 tests/ci/test_collate.py)."""
import numpy as np

try:
    import pytest
except ImportError:                                                    # the server's CUDA Python
    pytest = None


def _reference(batch, torch, WIN):
    """The element-by-element version (seeddrift/train/source_model.py before 2026-11-18), kept as the reference."""
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
    ns = max([len(src) for _, src, _ in batch] + [1])
    src = torch.zeros(B, ns, maxc, dtype=torch.long)
    labels = torch.zeros(B)
    gcomp = torch.zeros(B, 66)
    ccomp = torch.zeros(B, 66)
    for b, (slots, srcs, y) in enumerate(batch):
        labels[b] = y
        if slots[WIN] and "gcomp" in slots[WIN]:
            gcomp[b] = torch.from_numpy(slots[WIN]["gcomp"])
            ccomp[b] = torch.from_numpy(slots[WIN]["ccomp"])
        for s, sl in enumerate(slots):
            if not sl:
                continue
            present[b, s] = True
            codons[b, s, :len(sl["c"])] = torch.from_numpy(sl["c"])
            feats[b, s] = torch.tensor(sl["feat"], dtype=torch.float32)
            for h, (t, p) in enumerate(zip(sl["tracks"], sl["prots"])):
                tracks[b, s, h, :len(t)] = torch.from_numpy(t) + 1
                prots[b, s, h] = int(p) + 1
                hmask[b, s, h] = True
        for k, c in enumerate(srcs):
            src[b, k, :len(c)] = torch.from_numpy(c)
    return dict(codons=codons, present=present, feats=feats, tracks=tracks, prots=prots, hmask=hmask, src=src, labels=labels, gcomp=gcomp, ccomp=ccomp)


def _batch(rng, S, n, comp):
    out = []
    for b in range(n):
        slots = []
        for s in range(2 * S.WIN + 1):
            if s != S.WIN and rng.random() < 0.2:
                slots.append(None)                                   # past a replicon end
                continue
            nt = int(rng.integers(0, 5)) if rng.random() < 0.8 else 0  # some genes without hits
            sl = dict(c=rng.integers(1, 66, int(rng.integers(1, 40))).astype(np.int64),
                      tracks=[(rng.integers(1, 22, int(rng.integers(0, 30))) * 23 + rng.integers(1, 23)).astype(np.int64) for _ in range(nt)],
                      prots=[int(rng.integers(0, 50)) for _ in range(nt)],
                      feat=[float(rng.choice([-1, 1])), 0.0, float(rng.random()), float(rng.random()), (s - S.WIN) / S.WIN])
            if s == S.WIN and comp:
                sl["gcomp"] = S.frequencies(rng.integers(0, 9, 66))
                sl["ccomp"] = S.frequencies(rng.integers(0, 9, 66))
            slots.append(sl)
        srcs = [rng.integers(1, 66, int(rng.integers(1, 50))).astype(np.int64) for _ in range(int(rng.integers(0, 6)))]
        out.append((slots, srcs, 0.05 if b % 2 else 1.0))
    return out


def test_numpy_collate_equals_the_element_by_element_version():
    torch = pytest.importorskip("torch") if pytest else __import__("torch")
    from seeddrift.train import source_model as S
    rng = np.random.default_rng(0)
    for k in range(30):
        batch = _batch(rng, S, int(rng.integers(1, 9)), comp=bool(k % 2))
        new, ref = S.collate(batch, torch), _reference(batch, torch, S.WIN)
        assert set(new) == set(ref)
        for key in ref:
            assert new[key].dtype == ref[key].dtype and new[key].shape == ref[key].shape, key
            assert torch.equal(new[key], ref[key]), key


if __name__ == "__main__":                                             # the server's CUDA Python has no pytest
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    test_numpy_collate_equals_the_element_by_element_version()
    print("collate: equal")
