"""The data store ($SEEDDRIFT_DATA): what is needed, where it comes from, how to get and index it.

  seeddrift data status            what is present
  seeddrift data fetch [names]     download from the original sources onto THIS machine
  seeddrift data build [names]     build indices (DIAMOND, Foldseek, sequence index, Pfam)
  seeddrift data link NAME --from PATH   use files that already exist elsewhere (symlinks)

Datasets are described in data/datasets.json (sources, licenses, attribution: see REFERENCES.md).
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys

from . import config


def log(msg):
    print("[data] " + msg, file=sys.stderr, flush=True)


def datasets():
    return json.load(open(os.path.join(config.HOME, "data", "datasets.json")))


def _dir(d):
    return config.path(d["dir"])


def present(d):
    base = _dir(d)
    ok = all(os.path.exists(os.path.join(base, f)) for f in d["files"])
    if not ok and d.get("alt_files"):
        ok = all(os.path.exists(os.path.join(base, f)) for f in d["alt_files"])
    built = all(os.path.exists(os.path.join(base, f)) for f in d.get("built", []))
    return ok, built


def _size(p):
    if os.path.isfile(p):
        return os.path.getsize(p)
    tot = 0
    for root, _, files in os.walk(p, followlinks=True):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


def status():
    print("data store: %s" % config.DATA)
    print("%-22s %-20s %-9s %-9s %8s  %s" % ("dataset", "kind", "present", "indexed", "GB", "license"))
    for d in datasets():
        ok, built = present(d)
        gb = _size(_dir(d)) / 1e9 if os.path.exists(_dir(d)) else 0
        print("%-22s %-20s %-9s %-9s %8.1f  %s" % (d["name"], d["kind"], "yes" if ok else "no",
              ("yes" if built else "no") if d.get("built") else "-", gb, d["license"]))


def _select(names):
    ds = datasets()
    if not names:
        return ds
    known = {d["name"]: d for d in ds}
    bad = [n for n in names if n not in known]
    if bad:
        sys.exit("unknown dataset(s): %s (known: %s)" % (", ".join(bad), ", ".join(known)))
    return [known[n] for n in names]


def fetch(names, threads=None):
    from .donors import _fetch_parallel, reference, taxonomy
    for d in _select(names):
        ok, _ = present(d)
        if ok:
            log("%s: present" % d["name"])
            continue
        base = _dir(d)
        os.makedirs(base, exist_ok=True)
        log("%s: fetching %.1f GB from %s" % (d["name"], d["size_gb"], d["source"]))
        if d["fetch"] == "url":
            for fname, url in d["urls"].items():
                dest = os.path.join(base, fname)
                if not os.path.exists(dest):
                    _fetch_parallel(url, dest, parts=d.get("parts", 4))
        elif d["fetch"] == "reference":
            reference(config.REFERENCE, threads or config.THREADS)
        elif d["fetch"] == "taxonomy":
            taxonomy(config.REFERENCE)
        elif d["fetch"] == "huggingface":
            _fetch_hf(d["repo"], base, d.get("revision"))
        log("%s: done" % d["name"])


def _fetch_hf(repo, dest, revision=None):
    """HF hub download; honours HF_ENDPOINT (e.g. https://hf-mirror.com) and HF_TOKEN."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)
        from huggingface_hub import snapshot_download
    snapshot_download(repo_id=repo, revision=revision, local_dir=dest,
                      allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors", "*.model"])


def build(names, threads=None):
    threads = threads or config.THREADS
    for d in _select(names):
        ok, built = present(d)
        if not ok:
            log("%s: not present (fetch or link it first)" % d["name"])
            continue
        base = _dir(d)
        for step in d.get("build", []):
            if step == "seqindex":
                from .tools import SeqIndex
                SeqIndex(os.path.join(base, "reference.faa"))
            elif step == "diamond" and not os.path.exists(os.path.join(base, "ref.dmnd")):
                log("DIAMOND index")
                subprocess.run([config.DIAMOND, "makedb", "--in", os.path.join(base, "reference.faa"), "-d",
                                os.path.join(base, "ref.dmnd"), "--threads", str(threads)], check=True)
            elif step == "foldseek" and not os.path.exists(os.path.join(base, "targetDB.dbtype")):
                log("Foldseek database (reads the tar directly)")
                subprocess.run([config.FOLDSEEK, "createdb", os.path.join(base, "swissprot_pdb_v6.tar"),
                                os.path.join(base, "targetDB"), "--threads", str(threads)], check=True)
            elif step == "pfam" and not os.path.exists(os.path.join(base, "Pfam-A.hmm")):
                log("decompressing Pfam-A.hmm")
                with gzip.open(os.path.join(base, "Pfam-A.hmm.gz"), "rb") as fi, \
                        open(os.path.join(base, "Pfam-A.hmm.part"), "wb") as fo:
                    shutil.copyfileobj(fi, fo, 1 << 24)
                os.replace(os.path.join(base, "Pfam-A.hmm.part"), os.path.join(base, "Pfam-A.hmm"))
        log("%s: indexed" % d["name"])


def link(name, src):
    """Symlink existing files into the store: SRC is a folder holding the dataset's files (and
    possibly its indices), or, for single-file datasets, the file itself."""
    d = _select([name])[0]
    base = _dir(d)
    os.makedirs(os.path.dirname(base) if d["kind"] == "model" else base, exist_ok=True)
    src = os.path.abspath(os.path.expanduser(src))
    if d["kind"] == "model":                          # the whole folder
        if os.path.lexists(base):
            os.remove(base) if os.path.islink(base) else sys.exit("%s exists" % base)
        os.symlink(src, base)
        log("%s -> %s" % (base, src))
        return
    if os.path.isfile(src):
        names = {d["files"][0]: src}
    else:
        want = set(d["files"]) | set(d.get("alt_files", [])) | set(d.get("built", []))
        stems = [os.path.splitext(f)[0] for f in d.get("built", [])]
        names = {}
        for f in os.listdir(src):
            if f in want or any(f.startswith(s) for s in stems) or f.startswith(tuple(d["files"])):
                names[f] = os.path.join(src, f)
    if not names:
        sys.exit("nothing to link in %s" % src)
    for f, p in sorted(names.items()):
        dest = os.path.join(base, f)
        if os.path.lexists(dest):
            continue
        os.symlink(p, dest)
        log("%s -> %s" % (dest, p))


def main(a):
    if a.action == "status":
        status()
    elif a.action == "fetch":
        fetch(a.items, a.threads)
    elif a.action == "build":
        build(a.items, a.threads)
    elif a.action == "link":
        if len(a.items) != 1 or not a.src:
            sys.exit("usage: seeddrift data link NAME --from PATH")
        link(a.items[0], a.src)
