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


def _db_stamp(path):
    try:
        st = os.stat(path)
        return ":%d:%d" % (st.st_size, int(st.st_mtime))
    except OSError:
        return ""


def _block_args():
    """DIAMOND -b/-c: larger blocks mean fewer passes over the ~15 G-letter reference (much faster for a
    single genome) but more memory (roughly 6 x b GB, more with -c 1)."""
    v = config.DIAMOND_BLOCK
    if v and v != "auto":
        b, _, c = v.partition(",")
        return ["-b", b] + (["-c", c] if c else [])
    avail = _memory_gb()
    if avail > 400:
        return ["-b", "8", "-c", "1"]
    if avail > 160:
        return ["-b", "4", "-c", "1"]
    if avail > 64:
        return ["-b", "4", "-c", "2"]
    return []


def _memory_gb():
    """Memory this process can use: MemAvailable, capped by a container (cgroup v1/v2) limit."""
    avail = 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                avail = int(line.split()[1]) / 1e6
    except OSError:
        try:
            avail = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9 / 2
        except (ValueError, OSError, AttributeError):
            return 0.0
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(f).read().strip()
            if v.isdigit() and int(v) < 1 << 60:
                avail = min(avail, int(v) / 1e9)
        except OSError:
            pass
    return avail


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
            mafft=bool(_which(config.MAFFT)), fasttree=bool(_which(config.FASTTREE)),
            metaeuk=bool(_which(config.METAEUK)) and have(config.METAEUK_DB + ".dbtype"),
            mmseqs_gpu=bool(config.MMSEQS_GPU and _which(config.MMSEQS_GPU)) and have(config.MMSEQS_DB + ".dbtype"),
            reference=have(config.DIAMOND_DB), taxonomy=have(os.path.join(config.REFERENCE, "taxdump")),
            sprot=have(config.SPROT), afdb=have(os.path.join(config.AFDB, "targetDB.dbtype")),
            pfam=have(os.path.join(config.PFAM, "Pfam-A.hmm.dat.gz")), data=config.DATA)

    # -- sequence search ----------------------------------------------------------------------------
    def diamond(self, fasta, max_target_seqs=100, evalue=1e-5, sensitivity="sensitive", masking=None, db=None):
        """Proteins (FASTA text) vs the reference library. Returns hit dicts. masking=0 switches off
        DIAMOND's repeat masking (tantan), which otherwise hides repeat-rich proteins entirely."""
        tmp = tempfile.mkdtemp(prefix="sd-dmnd-")
        try:
            q, o = os.path.join(tmp, "q.faa"), os.path.join(tmp, "hits.tsv")
            with open(q, "w") as fh:
                fh.write(fasta)
            cmd = [config.DIAMOND, "blastp", "-q", q, "-d", db or config.DIAMOND_DB, "-o", o, "--" + sensitivity] + \
                (_block_args() if not db else []) + [
                   "--max-target-seqs", str(max_target_seqs), "-e", str(evalue), "--threads", str(self.threads),
                   "--outfmt", "6"] + DIAMOND_FIELDS + (["--masking", str(masking)] if masking is not None else [])
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

    def home_search(self, fasta, taxids, max_target_seqs=25, evalue=1e-5):
        """Search only the reference proteomes of the given taxa (the genome's home group). With ~40 M
        references a gene's top 100 hits can all lie outside a sparsely sampled home group, so the
        home-group relatives are looked up directly. The subset database is built once per taxon set."""
        import hashlib
        tx = sorted(set(int(t) for t in taxids))
        key = hashlib.sha1((",".join(map(str, tx)) + _db_stamp(config.REFERENCE_FAA)).encode()).hexdigest()[:16]
        d = config.path("cache", "homedb")
        os.makedirs(d, exist_ok=True)
        db = os.path.join(d, key + ".dmnd")
        with self._lock:
            if not os.path.exists(db):
                want, fa, n = set(str(t) for t in tx), os.path.join(d, key + ".faa"), 0
                with open(config.REFERENCE_FAA) as fi, open(fa, "w") as fo:
                    keep = False
                    for line in fi:
                        if line[0] == ">":
                            keep = line.rstrip("\n").rsplit("|", 1)[-1] in want
                            n += keep
                        if keep:
                            fo.write(line)
                log("home-group database: %d taxa, %d proteins" % (len(tx), n))
                subprocess.run([config.DIAMOND, "makedb", "--in", fa, "-d", db + ".part", "--threads", str(self.threads)],
                               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.replace(db + ".part.dmnd", db)
                os.remove(fa)
        return self.diamond(fasta, max_target_seqs, evalue, "sensitive", db=db)

    def deep_search(self, fasta, max_target_seqs=100, evalue=1e-5):
        """Most sensitive search for genes without outside hits, repeat masking off. MMseqs2-GPU (exhaustive
        ungapped prefilter on the GPU: seconds per query) when configured, else DIAMOND --ultra-sensitive."""
        if config.MMSEQS_GPU and os.path.exists(config.MMSEQS_DB + ".dbtype") and _which(config.MMSEQS_GPU):
            try:
                return self._mmseqs_gpu(fasta, max_target_seqs, evalue)
            except subprocess.CalledProcessError as e:
                log("MMseqs2-GPU failed (%s); DIAMOND --ultra-sensitive instead" % e)
        return self.diamond(fasta, max_target_seqs, evalue, "ultra-sensitive", masking=0)

    def _mmseqs_gpu(self, fasta, max_target_seqs, evalue):
        mm = config.MMSEQS_GPU
        tmp = tempfile.mkdtemp(prefix="sd-mmgpu-")
        try:
            fa, q, res, o = (os.path.join(tmp, x) for x in ("q.faa", "q", "res", "hits.tsv"))
            with open(fa, "w") as fh:
                fh.write(fasta)
            quiet = dict(check=True, stdout=subprocess.DEVNULL)
            subprocess.run([mm, "createdb", fa, q], **quiet)
            subprocess.run([mm, "search", q, config.MMSEQS_DB, res, os.path.join(tmp, "t"), "--gpu", "1", "--mask", "0",
                            "-e", str(evalue), "--max-seqs", str(max_target_seqs * 4), "--threads", str(self.threads)], **quiet)
            subprocess.run([mm, "convertalis", q, config.MMSEQS_DB, res, o, "--format-output",
                            "query,target,pident,alnlen,evalue,bits,qstart,qend,tstart,tend,qlen,tlen",
                            "--threads", str(self.threads)], **quiet)
            hits = []
            for line in open(o):
                f = line.rstrip("\n").split("\t")
                acc, _, tx = f[1].rpartition("|")
                qs, qe, ql = int(f[6]), int(f[7]), int(f[10])
                pid = float(f[2])
                hits.append(dict(query=f[0], target=acc or f[1], taxid=int(tx) if tx.isdigit() else 0,
                                 pident=pid * 100 if pid <= 1.0 else pid, alnlen=int(f[3]), evalue=float(f[4]),
                                 bits=float(f[5]), qstart=qs, qend=qe, tstart=int(f[8]), tend=int(f[9]), qlen=ql,
                                 tlen=int(f[11]), qcov=(qe - qs + 1) / float(ql)))
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

    # -- phylogenies ----------------------------------------------------------------------------------
    def trees(self, jobs):
        """{job: {label: protein}} -> {job: newick}: MAFFT (--auto) + FastTree (-lg -gamma), in parallel."""
        from concurrent.futures import ThreadPoolExecutor

        def one(item):
            name, seqs = item
            tmp = tempfile.mkdtemp(prefix="sd-tree-")
            try:
                fa, aln = os.path.join(tmp, "in.faa"), os.path.join(tmp, "aln.faa")
                with open(fa, "w") as fh:
                    for k, v in seqs.items():
                        fh.write(">%s\n%s\n" % (k, v))
                with open(aln, "w") as fh:
                    subprocess.run([config.MAFFT, "--auto", "--quiet", "--thread", "1", fa], stdout=fh, check=True)
                r = subprocess.run([config.FASTTREE, "-lg", "-gamma", "-quiet", "-nopr", aln],
                                   capture_output=True, text=True, check=True)
                return name, r.stdout.strip()
            except subprocess.CalledProcessError:
                return name, ""
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        with ThreadPoolExecutor(self.threads) as ex:
            return dict(ex.map(one, jobs.items()))

    # -- eukaryotic gene calling ----------------------------------------------------------------------
    def metaeuk(self, fasta):
        """Contigs (FASTA text) -> predicted proteins (FASTA text, MetaEuk headers), homology to Swiss-Prot."""
        db = config.METAEUK_DB
        if not os.path.exists(db + ".dbtype"):
            raise RuntimeError("MetaEuk reference database missing: seeddrift data build uniprot-swissprot")
        tmp = tempfile.mkdtemp(prefix="sd-metaeuk-")
        try:
            fa = os.path.join(tmp, "contigs.fna")
            with open(fa, "w") as fh:
                fh.write(fasta)
            out = os.path.join(tmp, "pred")
            subprocess.run([config.METAEUK, "easy-predict", fa, db, out, os.path.join(tmp, "t"),
                            "--threads", str(self.threads)], check=True, stdout=subprocess.DEVNULL)
            return open(out + ".fas").read()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

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
