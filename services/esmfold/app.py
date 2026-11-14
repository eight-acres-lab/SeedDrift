"""SeedDrift ESMFold service: protein structure prediction on the GPU.

  GET  /health                         model status and device
  POST /fold   {sequences, max_len}    -> {structures: {id: {plddt, pdb, seconds} | {skipped}}}

The model (HuggingFace facebook/esmfold_v1, see REFERENCES.md) is read from
$SEEDDRIFT_DATA/models/esmfold_v1 and loaded in the background at start-up.

Run:  uvicorn services.esmfold.app:app --host 127.0.0.1 --port 8602   (from the repository root)
"""
from __future__ import annotations

import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Dict  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from seeddrift import config  # noqa: E402

app = FastAPI(title="SeedDrift ESMFold", version="0.1")
STATE = {"folder": None, "error": None, "loading": False}


def _load():
    STATE["loading"] = True
    try:
        from seeddrift.fold import Folder
        STATE["folder"] = Folder(config.ESMFOLD_MODEL, chunk=int(config.env("SEEDDRIFT_ESMFOLD_CHUNK", "64")))
    except Exception as e:                      # noqa: BLE001 - reported through /health
        STATE["error"] = "%s: %s" % (type(e).__name__, e)
    finally:
        STATE["loading"] = False


@app.on_event("startup")
def startup():
    threading.Thread(target=_load, daemon=True).start()


class FoldReq(BaseModel):
    sequences: Dict[str, str]
    max_len: int = 1000


@app.get("/health")
def health():
    import torch
    return dict(ready=STATE["folder"] is not None, loading=STATE["loading"], error=STATE["error"],
                model=config.ESMFOLD_MODEL, cuda=torch.cuda.is_available(),
                device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")


@app.post("/fold")
def fold(r: FoldReq):
    if STATE["folder"] is None:
        raise HTTPException(503, STATE["error"] or "model still loading")
    return {"structures": STATE["folder"].fold(r.sequences, r.max_len)}
