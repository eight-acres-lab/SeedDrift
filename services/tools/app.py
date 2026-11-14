"""SeedDrift tools service: third-party search tools bound to the data store.

  GET  /health                       which tools and datasets are available
  POST /diamond    {fasta, max_target_seqs, evalue, sensitivity}  -> {hits}
  POST /sequences  {ids, source: reference|sprot}                 -> {sequences}
  POST /repeats    {sequences}  Pfam repeat-like domains + coiled coils -> {regions}
  POST /foldseek   {pdbs}       vs AlphaFold DB Swiss-Prot              -> {hits}
  POST /trees      {jobs: {name: {label: protein}}}  MAFFT + FastTree      -> {trees}
  POST /metaeuk    {fasta}      eukaryotic gene calling (MetaEuk vs Swiss-Prot) -> {proteins}

Run:  uvicorn services.tools.app:app --host 127.0.0.1 --port 8601   (from the repository root)
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict, List, Optional  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from seeddrift.tools import LocalTools  # noqa: E402

app = FastAPI(title="SeedDrift tools", version="0.1")
T = LocalTools()


class DiamondReq(BaseModel):
    fasta: str
    max_target_seqs: int = 100
    evalue: float = 1e-5
    sensitivity: str = "sensitive"
    masking: Optional[int] = None


class SeqReq(BaseModel):
    ids: List[str]
    source: str = "reference"


class RepeatReq(BaseModel):
    sequences: Dict[str, str]


class TreesReq(BaseModel):
    jobs: Dict[str, Dict[str, str]]


class MetaeukReq(BaseModel):
    fasta: str


class FoldseekReq(BaseModel):
    pdbs: Dict[str, str]


@app.get("/health")
def health():
    return T.health()


@app.post("/diamond")
def diamond(r: DiamondReq):
    return {"hits": T.diamond(r.fasta, r.max_target_seqs, r.evalue, r.sensitivity, r.masking)}


@app.post("/sequences")
def sequences(r: SeqReq):
    return {"sequences": T.sequences(r.ids, r.source)}


@app.post("/repeats")
def repeats(r: RepeatReq):
    return {"regions": T.repeats(r.sequences)}


@app.post("/foldseek")
def foldseek(r: FoldseekReq):
    return {"hits": T.foldseek(r.pdbs)}


@app.post("/trees")
def trees(r: TreesReq):
    return {"trees": T.trees(r.jobs)}


@app.post("/metaeuk")
def metaeuk(r: MetaeukReq):
    return {"proteins": T.metaeuk(r.fasta)}
