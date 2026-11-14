"""Dependency tools bound to the data store: DIAMOND, Foldseek, Pfam (pyhmmer), COILS, sequence lookup.

`LocalTools` runs them in this process. The tools service (services/tools) exposes the same methods
over HTTP and `clients.ToolsClient` calls it, so the SeedDrift pipeline does not care where the tools
and the ~70 GB of data live.
"""
from __future__ import annotations

import gzip
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

from . import config
from .repeats import annotate

DIAMOND_FIELDS = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qstart", "qend", "sstart",
                  "send", "qlen", "slen"]
FOLDSEEK_FIELDS = "query,target,evalue,bits,prob,alntmscore,lddt,fident,qcov,tcov,qstart,qend,tstart,tend"


def log(msg):
    print("[tools] " + msg, file=sys.stderr, flush=True)


def _which(exe):
    return shutil.which(exe) or (exe if os.path.isfile(exe) and os.access(exe, os.X_OK) else None)


class SeqIndex:
    """Random access to a large FASTA (reference.faa, headers 'acc|taxid'): sorted accessions + byte
    offsets, built once and stored next to the file (.sdx.npz)."""

    def __init__(self, fasta):
        import numpy as np
        self.fasta, idx = fasta, fasta + ".sdx.npz"
        if not os.path.exists(idx) or os.path.getmtime(idx) < os.path.getmtime(fasta):
            log("indexing %s (once)" % fasta)
            accs, offs, pos = [], [], 0
            with open(fasta, "rb") as fh:
                for line in fh:
                    if line[:1] == b">":
                        accs.append(line[1:].split(None, 1)[0].split(b"|", 1)[0])
                        offs.append(pos)
                    pos += len(line)
            a = np.array(accs, dtype="S%d" % max(len(x) for x in accs))
            o = np.array(offs, dtype=np.int64)
            order = np.argsort(a, kind="stable")
            np.savez(idx + ".part.npz", acc=a[order], off=o[order])
            os.replace(idx + ".part.npz", idx)
        z = np.load(idx)
        self.acc, self.off = z["acc"], z["off"]

    def get(self, ids):
        import numpy as np
        out = {}
        want = np.array([i.encode() for i in ids], dtype=self.acc.dtype)
        pos = np.searchsorted(self.acc, want)
        with open(self.fasta, "rb") as fh:
            for i, p in zip(ids, pos):
                if p < len(self.acc) and self.acc[p] == i.encode():
                    fh.seek(int(self.off[p]))
                    fh.readline()
                    seq = []
                    for line in fh:
                        if line[:1] == b">":
                            break
                        seq.append(line.strip())
                    out[i] = b"".join(seq).decode()
        return out


class SprotTable:
    """Swiss-Prot accession -> (taxid, protein name, sequence), loaded once (~570k entries)."""

    def __init__(self, path):
        self.taxid, self.name, self.seq = {}, {}, {}
        acc, buf = None, []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line[0] == ">":
                    if acc:
                        self.seq[acc] = "".join(buf)
                    acc, buf = line.split("|")[1], []
                    m = re.search(r" OX=(\d+)", line)
                    self.taxid[acc] = int(m.group(1)) if m else 0
                    self.name[acc] = re.sub(r" OS=.*", "", line.rstrip("\n").split(" ", 1)[-1])
                else:
                    buf.append(line.strip())
        if acc:
            self.seq[acc] = "".join(buf)


def af_acc(target):
    m = re.match(r"AF-([A-Z0-9]+)-F\d+", target)
    return m.group(1) if m else target.split(".")[0]


class LocalTools:
    def __init__(self, threads=None):
        self.threads = threads or config.THREADS
        self._lock = threading.Lock()
        self._refidx = self._sprot = None

    # -- status -------------------------------------------------------------------------------------
    def health(self):
        have = lambda p: os.path.exists(p)  # noqa: E731
        return dict(
            diamond=bool(_which(config.DIAMOND)), foldseek=bool(_which(config.FOLDSEEK)),
            ncoils=bool(_which(config.NCOILS)), pyhmmer=_importable("pyhmmer"),
            reference=have(config.DIAMOND_DB), taxonomy=have(os.path.join(config.REFERENCE, "taxdump")),
            sprot=have(config.SPROT), afdb=have(os.path.join(config.AFDB, "targetDB.dbtype")),
            pfam=have(os.path.join(config.PFAM, "Pfam-A.hmm.dat.gz")), data=config.DATA)

    # -- sequence search ----------------------------------------------------------------------------
    def diamond(self, fasta, max_target_seqs=100, evalue=1e-5, sensitivity="sensitive"):
        """Proteins (FASTA text) vs the reference library. Returns hit dicts."""
        tmp = tempfile.mkdtemp(prefix="sd-dmnd-")
        try:
            q, o = os.path.join(tmp, "q.faa"), os.path.join(tmp, "hits.tsv")
            with open(q, "w") as fh:
                fh.write(fasta)
            cmd = [config.DIAMOND, "blastp", "-q", q, "-d", config.DIAMOND_DB, "-o", o, "--" + sensitivity,
                   "--max-target-seqs", str(max_target_seqs), "-e", str(evalue), "--threads", str(self.threads),
                   "--outfmt", "6"] + DIAMOND_FIELDS
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            hits = []
            for line in open(o):
                f = line.rstrip("\n").split("\t")
                acc, _, tx = f[1].rpartition("|")
                qs, qe, ql = int(f[6]), int(f[7]), int(f[10])
                hits.append(dict(query=f[0], target=acc or f[1], taxid=int(tx) if tx.isdigit() else 0,
                                 pident=float(f[2]), alnlen=int(f[3]), evalue=float(f[4]), bits=float(f[5]),
                                 qstart=qs, qend=qe, tstart=int(f[8]), tend=int(f[9]), qlen=ql, tlen=int(f[11]),
                                 qcov=(qe - qs + 1) / float(ql)))
            return hits
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def sequences(self, ids, source="reference"):
        with self._lock:
            if source == "sprot":
                self._sprot = self._sprot or SprotTable(config.SPROT)
            else:
                self._refidx = self._refidx or SeqIndex(config.REFERENCE_FAA)
        if source == "sprot":
            return {i: self._sprot.seq[i] for i in ids if i in self._sprot.seq}
        return self._refidx.get(list(ids))

    def repeats(self, seqs):
        """Pfam repeat-like domains and coiled coils (tandem repeats are computed by the pipeline)."""
        pf = config.PFAM if os.path.exists(os.path.join(config.PFAM, "Pfam-A.hmm.dat.gz")) else None
        reg = annotate(seqs, pf, config.NCOILS, self.threads, tandem=False)
        return {k: [list(x) for x in v] for k, v in reg.items()}

    # -- structure search ---------------------------------------------------------------------------
    def foldseek(self, pdbs):
        """{id: PDB text} vs AlphaFold DB Swiss-Prot. Returns hit dicts with taxid and protein name."""
        with self._lock:
            self._sprot = self._sprot or SprotTable(config.SPROT)
        tdb = os.path.join(config.AFDB, "targetDB")
        tmp = tempfile.mkdtemp(prefix="sd-fs-")
        try:
            pdir = os.path.join(tmp, "pdb")
            os.makedirs(pdir)
            for k, v in pdbs.items():
                with open(os.path.join(pdir, re.sub(r"[^\w.-]", "_", k) + ".pdb"), "w") as fh:
                    fh.write(v)
            q, res, t = os.path.join(tmp, "q"), os.path.join(tmp, "aln"), os.path.join(tmp, "t")
            fs = config.FOLDSEEK
            subprocess.run([fs, "createdb", pdir, q, "--threads", str(self.threads)], check=True,
                           stdout=subprocess.DEVNULL)
            cmd = [fs, "search", q, tdb, res, t, "-a", "-e", "1e-2", "--max-seqs", "2000",
                   "--threads", str(self.threads)]
            target = tdb
            try:
                if not config.FOLDSEEK_GPU:
                    raise RuntimeError
                if not os.path.exists(tdb + "_pad.dbtype"):
                    subprocess.run([fs, "makepaddedseqdb", tdb, tdb + "_pad"], check=True, stdout=subprocess.DEVNULL)
                subprocess.run(cmd[:3] + [tdb + "_pad"] + cmd[4:] + ["--gpu", "1"], check=True,
                               stdout=subprocess.DEVNULL)
                target = tdb + "_pad"
            except (RuntimeError, subprocess.CalledProcessError):
                shutil.rmtree(t, ignore_errors=True)
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            o = os.path.join(tmp, "hits.tsv")
            subprocess.run([fs, "convertalis", q, target, res, o, "--format-output", FOLDSEEK_FIELDS,
                            "--threads", str(self.threads)], check=True, stdout=subprocess.DEVNULL)
            hits = []
            for line in open(o):
                f = line.rstrip("\n").split("\t")
                a = af_acc(f[1])
                hits.append(dict(query=f[0], target=a, taxid=self._sprot.taxid.get(a, 0),
                                 name=self._sprot.name.get(a, ""), evalue=float(f[2]), bits=float(f[3]),
                                 prob=float(f[4]), tm=float(f[5]), lddt=float(f[6]), fident=float(f[7]),
                                 qcov=float(f[8]), qstart=int(f[10]), qend=int(f[11]), tstart=int(f[12]),
                                 tend=int(f[13])))
            return hits
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _importable(mod):
    try:
        __import__(mod)
        return True
    except ImportError:
        return False
