"""SeedDrift API: our own algorithm as a job service (the web page talks to this).

  GET  /api/health                  this service + the dependency services
  GET  /api/demos                   demo genomes (data/demos.json)
  POST /api/jobs                    multipart: file (+ file2, e.g. the FASTA of a GFF3) | url | demo, name,
                                    self_taxon, domain, home_rank, deep, trees, structure
  GET  /api/jobs                    all jobs, newest first
  GET  /api/jobs/{id}               status, progress, summary
  GET  /api/jobs/{id}/genes         per-gene table (JSON); ?category=mammal,prokaryote
  GET  /api/jobs/{id}/files/{name}  genes.tsv, proteins.faa, hits.tsv, summary.json, structures/<gene>.pdb,
                                    trees/<gene>.nwk
  /                                 the web page (web/), when it is served from here

Jobs run one at a time in a background thread; each job is a folder under $SEEDDRIFT_JOBS.
Run:  uvicorn services.api.app:app --host 127.0.0.1 --port 8600   (from the repository root)
"""
from __future__ import annotations

import csv
import json
import os
import queue
import random
import string
import sys
import threading
import time
import traceback
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from typing import Optional  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from seeddrift import __version__, clients, config, pipeline  # noqa: E402

app = FastAPI(title="SeedDrift", version=__version__)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
os.makedirs(config.JOBS, exist_ok=True)
Q = queue.Queue()
FILES = ("genes.tsv", "proteins.faa", "hits.tsv", "summary.json")
SUBDIRS = {"structures": ".pdb", "trees": ".nwk"}
MAX_UPLOAD = int(config.env("SEEDDRIFT_MAX_UPLOAD_MB", "200")) * 1024 * 1024


def _demos():
    p = os.path.join(ROOT, "data", "demos.json")
    return json.load(open(p)) if os.path.exists(p) else []


def _jdir(jid):
    if not jid or any(c not in string.ascii_letters + string.digits + "-" for c in jid):
        raise HTTPException(400, "bad job id")
    d = os.path.join(config.JOBS, jid)
    if not os.path.isdir(d):
        raise HTTPException(404, "no such job")
    return d


def _status(d):
    p = os.path.join(d, "status.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _write_status(d, **kw):
    s = _status(d)
    s.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = os.path.join(d, "status.json.part")
    with open(tmp, "w") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(d, "status.json"))
    return s


def _fetch(url, limit=MAX_UPLOAD):
    req = urllib.request.Request(url, headers={"User-Agent": "seeddrift/" + __version__})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = r.read(limit + 1)
    if len(data) > limit:
        raise ValueError("input larger than %d MB" % (limit >> 20))
    return data


def _worker():
    while True:
        jid = Q.get()
        d = os.path.join(config.JOBS, jid)
        try:
            st = _status(d)
            _write_status(d, status="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))
            inp = os.path.join(d, "input.bin")
            if not os.path.exists(inp):
                _write_status(d, stage="input", message="服务器下载输入：%s" % st["url"], progress=0.01)
                with open(inp + ".part", "wb") as fh:
                    fh.write(_fetch(st["url"]))
                os.replace(inp + ".part", inp)
            params = st["params"]
            folder = clients.folder() if params.get("structure") else None
            files = [open(inp, "rb").read()]
            if os.path.exists(os.path.join(d, "input2.bin")):
                files.append(open(os.path.join(d, "input2.bin"), "rb").read())

            def progress(stage, message, frac):
                _write_status(d, stage=stage, message=message, progress=round(frac, 3))
            summary = pipeline.run(d, files, params, clients.tools(), folder, progress)
            _write_status(d, status="done", summary=summary, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        except Exception as e:                  # noqa: BLE001 - the job records the error
            _write_status(d, status="failed", error="%s: %s" % (type(e).__name__, e),
                          trace=traceback.format_exc()[-4000:])
        finally:
            Q.task_done()


@app.on_event("startup")
def startup():
    for jid in sorted(os.listdir(config.JOBS)):     # resume jobs interrupted by a restart
        s = _status(os.path.join(config.JOBS, jid))
        if s.get("status") in ("queued", "running"):
            Q.put(jid)
    threading.Thread(target=_worker, daemon=True).start()


@app.get("/api/health")
def health():
    out = dict(api=__version__, data=config.DATA, jobs_queued=Q.qsize())
    for name, url, make in (("tools", config.TOOLS_URL, clients.ToolsClient),
                            ("esmfold", config.ESMFOLD_URL, clients.FoldClient)):
        if not url:
            out[name] = dict(mode="in-process")
            continue
        try:
            out[name] = dict(make(url).health(), url=url)
        except Exception as e:                  # noqa: BLE001
            out[name] = dict(url=url, error=str(e))
    return out


@app.get("/api/demos")
def demos():
    return _demos()


@app.post("/api/jobs")
async def create_job(file: Optional[UploadFile] = File(None), file2: Optional[UploadFile] = File(None),
                     url: str = Form(""), demo: str = Form(""), name: str = Form(""), self_taxon: str = Form(""),
                     domain: str = Form("auto"), home_rank: str = Form("auto"), deep: str = Form("true"),
                     trees: str = Form("true"), structure: str = Form("true"), max_structures: int = Form(400),
                     repeat_filter: str = Form("true")):
    if demo:
        dm = next((x for x in _demos() if x["id"] == demo), None)
        if not dm:
            raise HTTPException(400, "unknown demo")
        url, name, self_taxon = dm["url"], name or dm["name"], self_taxon or dm.get("self_taxon", "")
    if not file and not url:
        raise HTTPException(400, "give a file, a URL or a demo")
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + "".join(random.choice("abcdefghjkmnpqrstuvwxyz23456789")
                                                          for _ in range(4))
    d = os.path.join(config.JOBS, jid)
    os.makedirs(d)
    src = url
    if file:
        data = await file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file larger than %d MB" % (MAX_UPLOAD >> 20))
        with open(os.path.join(d, "input.bin"), "wb") as fh:
            fh.write(data)
        src = file.filename
        if file2:
            data2 = await file2.read(MAX_UPLOAD + 1)
            if len(data2) > MAX_UPLOAD:
                raise HTTPException(413, "file larger than %d MB" % (MAX_UPLOAD >> 20))
            with open(os.path.join(d, "input2.bin"), "wb") as fh:
                fh.write(data2)
            src += " + " + file2.filename
    _write_status(d, id=jid, name=name or src, source=src, url=url if not file else "", status="queued",
                  stage="queued", message="排队中", progress=0.0, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                  params=dict(self_taxon=self_taxon, domain=domain if domain in ("auto", "prokaryote", "eukaryote")
                              else "auto", home_rank=home_rank, deep=_flag(deep), trees=_flag(trees),
                              structure=_flag(structure), max_structures=max_structures,
                              repeat_filter=_flag(repeat_filter)))
    Q.put(jid)
    return {"id": jid}


def _flag(v):
    return str(v).lower() in ("1", "true", "yes", "on")


@app.get("/api/jobs")
def list_jobs():
    out = []
    for jid in sorted(os.listdir(config.JOBS), reverse=True):
        s = _status(os.path.join(config.JOBS, jid))
        if s:
            s.pop("trace", None)
            out.append(s)
    return out


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    return _status(_jdir(jid))


@app.get("/api/jobs/{jid}/genes")
def genes(jid: str, category: str = ""):
    p = os.path.join(_jdir(jid), "genes.tsv")
    if not os.path.exists(p):
        raise HTTPException(404, "not ready")
    want = set(c for c in category.split(",") if c)
    rows = list(csv.DictReader(open(p), delimiter="\t"))
    return [r for r in rows if not want or r["category"] in want]


@app.get("/api/jobs/{jid}/files/{name:path}")
def files(jid: str, name: str):
    d = _jdir(jid)
    sub, _, base = name.partition("/")
    ok = name in FILES or (sub in SUBDIRS and base.endswith(SUBDIRS[sub]) and "/" not in base and ".." not in base)
    p = os.path.join(d, name)
    if not ok or not os.path.exists(p):
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=os.path.basename(p))


WEB = config.env("SEEDDRIFT_WEB", os.path.join(ROOT, "web"))
if os.path.isdir(WEB):
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
