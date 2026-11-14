"""Where things live. Everything is driven by environment variables so that the same code runs on a
laptop, a rented GPU box or in containers (see deploy/seeddrift.env.example)."""
from __future__ import annotations

import os

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def env(name, default=None):
    v = os.environ.get(name)
    return v if v not in (None, "") else default


DATA = os.path.abspath(os.path.expanduser(env("SEEDDRIFT_DATA", os.path.join(HOME, "data", "store"))))
THREADS = int(env("SEEDDRIFT_THREADS", str(min(16, os.cpu_count() or 1))))

# dependency services (HTTP); empty = run the tools in this process
TOOLS_URL = env("SEEDDRIFT_TOOLS_URL", "")
ESMFOLD_URL = env("SEEDDRIFT_ESMFOLD_URL", "")

# third-party binaries (on PATH by default)
DIAMOND = env("SEEDDRIFT_DIAMOND", "diamond")
FOLDSEEK = env("SEEDDRIFT_FOLDSEEK", "foldseek")
NCOILS = env("SEEDDRIFT_NCOILS", "ncoils")
MAFFT = env("SEEDDRIFT_MAFFT", "mafft")
FASTTREE = env("SEEDDRIFT_FASTTREE", "FastTree")
METAEUK = env("SEEDDRIFT_METAEUK", "metaeuk")
MMSEQS_GPU = env("SEEDDRIFT_MMSEQS_GPU", "")     # MMseqs2 linux-gpu build: GPU deep search (optional)
DIAMOND_BLOCK = env("SEEDDRIFT_DIAMOND_BLOCK", "auto")   # "auto" (by free memory) or e.g. "8,1" (-b, -c)
FOLDSEEK_GPU = env("SEEDDRIFT_FOLDSEEK_GPU", "0") == "1"


def path(*parts):
    return os.path.join(DATA, *parts)


# data layout (data/datasets.json describes how each item is obtained)
REFERENCE = path("reference")            # reference.faa (acc|taxid), ref.dmnd, taxdump/
REFERENCE_FAA = path("reference", "reference.faa")
DIAMOND_DB = path("reference", "ref.dmnd")
MMSEQS_DB = path("reference", "refDB_pad")       # padded MMseqs2 database for MMseqs2-GPU
SPROT = path("sprot", "uniprot_sprot.fasta.gz")
METAEUK_DB = path("sprot", "metaeukDB")          # MetaEuk/MMseqs2 database of Swiss-Prot
AFDB = path("afdb")                      # swissprot_pdb_v6.tar, targetDB*
AFDB_TAR = path("afdb", "swissprot_pdb_v6.tar")
PFAM = path("pfam")                      # Pfam-A.hmm.gz, Pfam-A.hmm.dat.gz
ESMFOLD_MODEL = path("models", "esmfold_v1")
JOBS = env("SEEDDRIFT_JOBS", path("jobs"))
