"""The run record: runs/<run_id>/record.json, written atomically on every change and never deleted.

Stages report into runs/<run_id>/stages/<rule>.json (see stage_report); the runner merges them into the record. A
heartbeat thread refreshes record["heartbeat"] every 60 s; `ci status` calls a running record whose heartbeat is older
than 10 minutes "interrupted".
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import threading
import time

from ci import ROOT
from ci.hashing import canonical_json

SCHEMA = os.path.join(ROOT, "ci", "schemas", "run_record.schema.json")
STATUSES = ("running", "passed", "failed", "gate_failed", "interrupted", "diverged")
STALE_S = 600


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: str, obj) -> None:
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def validate(record: dict) -> None:
    import jsonschema
    with open(SCHEMA) as f:
        jsonschema.validate(record, json.load(f))


class Record:
    def __init__(self, run_dir: str):
        self.dir = run_dir
        self.path = os.path.join(run_dir, "record.json")
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.data = None

    @classmethod
    def create(cls, run_dir: str, **fields) -> "Record":
        os.makedirs(os.path.join(run_dir, "stages"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "gates"), exist_ok=True)
        r = cls(run_dir)
        if os.path.exists(r.path):
            raise FileExistsError("record exists: %s (a run id is used once)" % r.path)
        r.data = dict(fields, status="running", started=now(), heartbeat=now(), ended=None, stages=[], gates={}, metrics={},
                      artifacts=[], waivers_used=[], deviations=[])
        r.save()
        return r

    @classmethod
    def open(cls, run_dir: str) -> "Record":
        r = cls(run_dir)
        with open(r.path) as f:
            r.data = json.load(f)
        return r

    def save(self) -> None:
        with self._lock:
            atomic_write_json(self.path, self.data)

    def update(self, **fields) -> None:
        with self._lock:
            self.data.update(fields)
        self.save()

    def beat(self) -> None:
        with self._lock:
            self.data["heartbeat"] = now()
        self.save()

    def start_heartbeat(self, every_s: int = 60) -> None:
        def loop():
            while not self._stop.wait(every_s):
                self.beat()
        threading.Thread(target=loop, daemon=True).start()

    def merge_stages(self) -> None:
        folder = os.path.join(self.dir, "stages")
        rows = []
        for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if name.endswith(".json"):
                with open(os.path.join(folder, name)) as f:
                    rows.append(json.load(f))
        gates = {}
        gdir = os.path.join(self.dir, "gates")
        for name in sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []:
            if name.endswith(".json"):
                with open(os.path.join(gdir, name)) as f:
                    g = json.load(f)
                gates[g["gate"]] = g["status"]
        with self._lock:
            self.data["stages"] = rows
            self.data["gates"] = gates
        self.save()

    def finish(self, status: str) -> None:
        if status not in STATUSES:
            raise ValueError(status)
        self._stop.set()
        self.merge_stages()
        self.update(status=status, ended=now())


def stage_report(run_dir: str, rule: str, key: str, started: float, outputs, exit_code: int = 0, **extra) -> None:
    """Called by every stage script at its end: timing, outputs (manifest sha256 each) and resource peaks."""
    import resource
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    row = dict(name=rule, key=key, status="passed" if exit_code == 0 else "failed", started=_dt.datetime.fromtimestamp(started, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               ended=now(), wall_s=round(time.time() - started, 2), exit_code=exit_code, peak_rss_mb=round(peak_kb / 1024, 1), outputs=outputs, **extra)
    folder = os.path.join(run_dir, "stages")
    os.makedirs(folder, exist_ok=True)
    atomic_write_json(os.path.join(folder, rule + ".json"), row)


def status_of(run_dir: str) -> str:
    """The record's status, with a stale heartbeat reported as interrupted."""
    with open(os.path.join(run_dir, "record.json")) as f:
        d = json.load(f)
    if d["status"] != "running":
        return d["status"]
    beat = _dt.datetime.strptime(d["heartbeat"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
    if (_dt.datetime.now(_dt.timezone.utc) - beat).total_seconds() > STALE_S:
        return "interrupted"
    return "running"


def summary_bytes(record: dict) -> bytes:
    return canonical_json(record)
