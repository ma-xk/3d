"""Background jobs for the web UI.

Meshes are processed in a separate worker process so the server stays
responsive while MeshLab's C++ filters run, and so a crash or out-of-memory in
one job cannot take the UI down. The worker is kept alive between jobs; it
caches the last loaded mesh so "clean" does not re-read the file.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

_CACHE: dict[str, object] = {}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _load_cached(input_path: str, display_name: str):
    from ..engine import load_mesh

    stamp = f"{input_path}:{os.path.getmtime(input_path)}"
    if stamp not in _CACHE:
        _CACHE.clear()
        loaded = load_mesh(input_path)
        loaded.name = Path(display_name).stem or loaded.name
        _CACHE[stamp] = loaded
    return _CACHE[stamp]


def worker_warmup() -> bool:
    from ..engine import mlab

    mlab._pymeshlab()
    return True


def worker_inspect(job_dir: str, input_path: str, display_name: str) -> dict:
    from ..engine import encode_preview, inspect_loaded

    loaded = _load_cached(input_path, display_name)
    info, snap = inspect_loaded(loaded, None, self_intersections=True)
    Path(job_dir, "preview-before.bin").write_bytes(
        encode_preview(snap.vertices, snap.faces, snap.rim, snap.non_manifold))
    return info


def worker_clean(job_dir: str, input_path: str, display_name: str, options: dict, output_name: str) -> dict:
    from ..engine import CleanOptions, encode_preview, run_pipeline

    job = Path(job_dir)
    progress_file = job / "progress.json"

    def progress(message: str, fraction: float) -> None:
        _write_json(progress_file, {"message": message, "fraction": fraction, "time": time.time()})

    progress("Starting", 0.01)
    loaded = _load_cached(input_path, display_name)
    output = job / output_name
    result = run_pipeline(input_path, output, CleanOptions.from_dict(options), progress=progress, loaded=loaded)
    progress("Preparing the preview", 0.99)
    for which, snap in (("before", result.before_mesh), ("after", result.after_mesh)):
        (job / f"preview-{which}.bin").write_bytes(
            encode_preview(snap.vertices, snap.faces, snap.rim, snap.non_manifold))
    data = result.to_dict()
    data["input"]["path"] = display_name
    data["output"]["path"] = output_name
    _write_json(job / "report.json", data)
    return data


@dataclass
class Job:
    id: str
    dir: Path
    filename: str
    input_path: Path
    state: str = "inspecting"  # inspecting | ready | cleaning | done | error
    inspect: dict | None = None
    result: dict | None = None
    error: str | None = None
    options: dict | None = None
    created: float = field(default_factory=time.time)
    output_name: str = ""

    def public(self) -> dict:
        progress = None
        if self.state == "cleaning":
            try:
                progress = json.loads((self.dir / "progress.json").read_text())
            except (OSError, ValueError):
                progress = {"message": "Starting", "fraction": 0.0}
        return {
            "id": self.id, "state": self.state, "filename": self.filename, "inspect": self.inspect,
            "result": self.result, "error": self.error, "progress": progress, "options": self.options,
            "has_after": (self.dir / "preview-after.bin").exists() and self.state == "done",
        }


class JobManager:
    def __init__(self, root: Path, executor: str = "process", keep: int = 6) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.kind = executor
        self.keep = keep
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.executor = self._make_executor()
        if executor == "process":
            self.executor.submit(worker_warmup)

    def _make_executor(self) -> Executor:
        if self.kind == "thread":
            return ThreadPoolExecutor(max_workers=1)
        return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))

    def _submit(self, fn, *args) -> Future:
        try:
            return self.executor.submit(fn, *args)
        except (BrokenProcessPool, RuntimeError):
            self.executor = self._make_executor()
            return self.executor.submit(fn, *args)

    def create(self, filename: str, suffix: str) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True)
        job = Job(id=job_id, dir=job_dir, filename=filename, input_path=job_dir / f"input{suffix}")
        stem = Path(filename).stem or "scan"
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)[:80] or "scan"
        job.output_name = f"{safe}-clean.3mf"
        with self.lock:
            self.jobs[job_id] = job
            self._prune()
        return job

    def _prune(self) -> None:
        if len(self.jobs) <= self.keep:
            return
        for job in sorted(self.jobs.values(), key=lambda j: j.created)[: len(self.jobs) - self.keep]:
            if job.state in ("inspecting", "cleaning"):
                continue
            shutil.rmtree(job.dir, ignore_errors=True)
            self.jobs.pop(job.id, None)

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def start_inspect(self, job: Job) -> None:
        job.state = "inspecting"
        future = self._submit(worker_inspect, str(job.dir), str(job.input_path), job.filename)
        future.add_done_callback(lambda fut: self._finish(job, fut, "ready", "inspect"))

    def start_clean(self, job: Job, options: dict) -> None:
        job.state = "cleaning"
        job.options = options
        job.result = None
        job.error = None
        for name in ("preview-after.bin", "report.json", job.output_name, "progress.json"):
            (job.dir / name).unlink(missing_ok=True)
        future = self._submit(worker_clean, str(job.dir), str(job.input_path), job.filename, options,
                              job.output_name)
        future.add_done_callback(lambda fut: self._finish(job, fut, "done", "result"))

    def _finish(self, job: Job, future: Future, state: str, attr: str) -> None:
        try:
            setattr(job, attr, future.result())
            job.state = state
        except BrokenProcessPool:
            job.error = "The worker process stopped unexpectedly (possibly out of memory)."
            job.state = "error"
            self.executor = self._make_executor()
        except Exception as exc:  # shown to the user
            job.error = str(exc) or exc.__class__.__name__
            job.state = "error"

    def delete(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs.pop(job_id, None)
        if job:
            shutil.rmtree(job.dir, ignore_errors=True)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(self.root, ignore_errors=True)
