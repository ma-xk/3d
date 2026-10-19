"""Local browser UI. Binds to 127.0.0.1 by default: files never leave this computer."""

from __future__ import annotations

import socket
import tempfile
import threading
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..engine.meshio import SUPPORTED_EXTENSIONS
from ..engine.pipeline import CleanOptions
from ..engine.units import UNITS
from .jobs import JobManager

STATIC_DIR = Path(__file__).parent / "static"
CLIENT_HEADER = "x-scan-to-3mf"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")


def create_app(work_dir: Path | None = None, executor: str = "process", restrict_hosts: bool = True) -> FastAPI:
    root = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="scan-to-3mf-"))
    state: dict = {"slicer": None, "slicer_checked": False}

    def detect_slicer() -> None:
        from ..engine import find_slicer

        s = find_slicer()
        state["slicer"] = {"label": s.label, "kind": s.kind, "can_slice": s.can_slice} if s else None
        state["slicer_checked"] = True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.jobs = JobManager(root, executor=executor)
        threading.Thread(target=detect_slicer, daemon=True).start()
        yield
        app.state.jobs.shutdown()

    app = FastAPI(title="scan-to-3mf", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if restrict_hosts:
            host = request.headers.get("host", "")
            name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
            if name not in LOOPBACK_HOSTS:
                return JSONResponse({"detail": "This app only answers on localhost."}, status_code=400)
        if request.method not in ("GET", "HEAD") and request.headers.get(CLIENT_HEADER) != "1":
            return JSONResponse({"detail": "Missing client header."}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    def job_or_404(request: Request, job_id: str):
        job = request.app.state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job (it may have been cleared).")
        return job

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html", media_type="text/html")

    @app.get("/api/info")
    async def info():
        return {
            "version": __version__,
            "slicer": state["slicer"],
            "slicer_checked": state["slicer_checked"],
            "units": {k: {"name": v[0], "factor": v[1], "label": v[2]} for k, v in UNITS.items()},
            "defaults": asdict(CleanOptions()),
            "extensions": list(SUPPORTED_EXTENSIONS),
        }

    @app.post("/api/upload")
    async def upload(request: Request, name: str):
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(400, f"Unsupported file type {suffix or '(none)'}. Use STL, OBJ, PLY, or 3MF.")
        jobs: JobManager = request.app.state.jobs
        job = jobs.create(Path(name).name, suffix)
        size = 0
        with open(job.input_path, "wb") as fh:
            async for chunk in request.stream():
                fh.write(chunk)
                size += len(chunk)
        if size == 0:
            jobs.delete(job.id)
            raise HTTPException(400, "The file is empty.")
        jobs.start_inspect(job)
        return job.public()

    @app.get("/api/jobs/{job_id}")
    async def job_status(request: Request, job_id: str):
        return job_or_404(request, job_id).public()

    @app.post("/api/jobs/{job_id}/clean")
    async def clean(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        if job.state in ("inspecting", "cleaning"):
            raise HTTPException(409, "The file is still being processed.")
        if job.state == "error" and job.inspect is None:
            raise HTTPException(409, job.error or "The file could not be read.")
        body = await request.json()
        try:
            options = CleanOptions.from_dict(body or {})
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        request.app.state.jobs.start_clean(job, asdict(options))
        return job.public()

    @app.get("/api/jobs/{job_id}/preview/{which}")
    async def preview(request: Request, job_id: str, which: str):
        job = job_or_404(request, job_id)
        if which not in ("before", "after"):
            raise HTTPException(404, "Unknown preview.")
        path = job.dir / f"preview-{which}.bin"
        if not path.exists():
            raise HTTPException(404, "Preview not ready.")
        return Response(path.read_bytes(), media_type="application/octet-stream")

    @app.get("/api/jobs/{job_id}/download")
    async def download(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        path = job.dir / job.output_name
        if job.state != "done" or not path.exists():
            raise HTTPException(404, "No 3MF yet.")
        return FileResponse(path, media_type="model/3mf", filename=job.output_name)

    @app.get("/api/jobs/{job_id}/report")
    async def report(request: Request, job_id: str):
        job = job_or_404(request, job_id)
        path = job.dir / "report.json"
        if not path.exists():
            raise HTTPException(404, "No report yet.")
        return FileResponse(path, media_type="application/json",
                            filename=job.output_name.replace(".3mf", "-report.json"))

    @app.delete("/api/jobs/{job_id}")
    async def delete(request: Request, job_id: str):
        request.app.state.jobs.delete(job_id)
        return {"deleted": job_id}

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _free_port(host: str, start: int) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise OSError(f"No free port found from {start} to {start + 49}")


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> int:
    import uvicorn

    port = _free_port(host, port)
    loopback = host in LOOPBACK_HOSTS
    url = f"http://{'localhost' if loopback else host}:{port}/"
    if not loopback:
        print(f"Warning: listening on {host}. Other computers on the network can reach this app.")
    print(f"scan-to-3mf {__version__} is running at {url}")
    print("Files are processed on this computer only. Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(restrict_hosts=loopback), host=host, port=port, log_level="warning")
    return 0
