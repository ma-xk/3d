"""Headless slice test with PrusaSlicer, plus a size check with OrcaSlicer or
Bambu Studio when PrusaSlicer is not installed.

The slice test answers one question: will a real slicer turn this file into
toolpaths? It uses a bed large enough for the model, so "too big for my
printer" is not reported as a failure. OrcaSlicer and Bambu Studio can read a
3MF from the command line (``--info``) but refuse to slice without a matching
set of printer, process, and filament presets, so with them the app only
checks the size and that the mesh is closed.
"""

from __future__ import annotations

import math
import os
import platform
import plistlib
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

ENV_VAR = "SCAN_TO_3MF_SLICER"
LAYER_HEIGHT = 0.3
SLICE_TIMEOUT_S = 600.0

KINDS = ("prusaslicer", "orcaslicer", "bambustudio")  # order of preference
LABELS = {"prusaslicer": "PrusaSlicer", "orcaslicer": "OrcaSlicer", "bambustudio": "Bambu Studio"}
_PATH_NAMES = {
    "prusaslicer": ("prusa-slicer", "PrusaSlicer", "prusaslicer", "prusa-slicer-console"),
    "orcaslicer": ("orca-slicer", "OrcaSlicer", "orcaslicer"),
    "bambustudio": ("bambu-studio", "BambuStudio", "bambustudio"),
}
_APP_BUNDLES = {
    "prusaslicer": ("PrusaSlicer*.app", "Original Prusa Drivers/PrusaSlicer*.app"),
    "orcaslicer": ("OrcaSlicer*.app",),
    "bambustudio": ("BambuStudio*.app", "Bambu Studio*.app"),
}
_WINDOWS_APPS = {
    "prusaslicer": (r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",),
    "orcaslicer": (r"C:\Program Files\OrcaSlicer\orca-slicer.exe",),
    "bambustudio": (r"C:\Program Files\Bambu Studio\bambu-studio.exe",),
}
_FLATPAK_ID = "com.prusa3d.PrusaSlicer"
_VERSION = re.compile(r"(\d+(?:\.\d+)+(?:-(?:alpha|beta|rc)\.?\d*)?)")


@dataclass
class Slicer:
    kind: str  # "prusaslicer" | "orcaslicer" | "bambustudio"
    argv: list[str]  # command prefix
    version: str | None = None
    flatpak: bool = False

    @property
    def label(self) -> str:
        name = LABELS.get(self.kind, self.kind)
        text = f"{name} {self.version}" if self.version else name
        return text + (" (Flatpak)" if self.flatpak else "")

    @property
    def can_slice(self) -> bool:
        return self.kind == "prusaslicer"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(label=self.label, can_slice=self.can_slice)
        return d


@dataclass
class SliceResult:
    ok: bool | None  # None: the slice itself was not run
    slicer: str
    command: str
    seconds: float
    message: str
    info: dict = field(default_factory=dict)
    layers: int | None = None
    expected_layers: int | None = None
    print_time: str | None = None
    filament_mm: float | None = None
    gcode_bytes: int | None = None
    warnings: list[str] = field(default_factory=list)
    log_tail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def clean_version(text: str | None) -> str | None:
    """'2.7.2+UNKNOWN' -> '2.7.2'; '3.0.0-alpha12' stays; None if no version."""
    if not text:
        return None
    m = _VERSION.search(str(text))
    return m.group(1) if m else None


def _kind_for(path: str) -> str:
    name = os.path.basename(path).lower().replace(" ", "")
    if "orca" in name:
        return "orcaslicer"
    if "bambu" in name:
        return "bambustudio"
    return "prusaslicer"


def _version_from_help(argv: list[str]) -> str | None:
    try:
        out = subprocess.run(argv + ["--help"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or "") + (out.stderr or "")
    m = re.search(r"(PrusaSlicer|OrcaSlicer|BambuStudio|Bambu Studio)[- ]v?(\S+)", text)
    return clean_version(m.group(2)) if m else None


def bundle_executable(app: Path) -> tuple[Path, str | None] | None:
    """Executable and version of a macOS .app bundle, read from its Info.plist."""
    exe_name = version = None
    info = app / "Contents" / "Info.plist"
    if info.is_file():
        try:
            with open(info, "rb") as fh:
                plist = plistlib.load(fh)
            exe_name = plist.get("CFBundleExecutable")
            version = clean_version(plist.get("CFBundleShortVersionString") or plist.get("CFBundleVersion"))
        except Exception:
            pass
    macos = app / "Contents" / "MacOS"
    candidates = [macos / exe_name] if exe_name else []
    if macos.is_dir():
        candidates += sorted(macos.iterdir())
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand, version
    return None


def default_app_dirs() -> list[Path]:
    if platform.system() != "Darwin":
        return []
    return [Path("/Applications"), Path.home() / "Applications"]


def find_slicer(explicit: str | None = None, app_dirs: Iterable[Path] | None = None,
                which: Callable[[str], str | None] = shutil.which, use_flatpak: bool = True) -> Slicer | None:
    """Find a slicer: an explicit path or $SCAN_TO_3MF_SLICER first, then macOS app
    bundles, the PATH, and standard Windows locations, preferring PrusaSlicer."""
    for cand in (explicit, os.environ.get(ENV_VAR)):
        if not cand:
            continue
        path = Path(os.path.expanduser(cand))
        if path.suffix == ".app" and path.is_dir():
            found = bundle_executable(path)
            if found:
                return Slicer(_kind_for(path.name), [str(found[0])], found[1])
        if path.is_file() and os.access(path, os.X_OK):
            argv = [str(path)]
            return Slicer(_kind_for(str(path)), argv, _version_from_help(argv))
    if explicit:
        return None

    found: dict[str, Slicer] = {}
    dirs = default_app_dirs() if app_dirs is None else [Path(d) for d in app_dirs]
    for kind in KINDS:
        for folder in dirs:
            for pattern in _APP_BUNDLES[kind]:
                for app in sorted(folder.glob(pattern)):
                    hit = bundle_executable(app)
                    if hit and kind not in found:
                        found[kind] = Slicer(kind, [str(hit[0])], hit[1])
        if kind not in found:
            for name in _PATH_NAMES[kind]:
                path = which(name)
                if path:
                    found[kind] = Slicer(kind, [path], _version_from_help([path]))
                    break
        if kind not in found and platform.system() == "Windows":
            for path in _WINDOWS_APPS[kind]:
                if os.path.isfile(path):
                    found[kind] = Slicer(kind, [path], _version_from_help([path]))
                    break
        if kind in found:
            return found[kind]

    if use_flatpak and platform.system() == "Linux" and which("flatpak"):
        try:
            info = subprocess.run(["flatpak", "info", _FLATPAK_ID], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if info.returncode == 0:
            m = re.search(r"Version:\s*(\S+)", info.stdout)
            argv = ["flatpak", "run", "--filesystem=/tmp", "--filesystem=home", _FLATPAK_ID]
            return Slicer("prusaslicer", argv, clean_version(m.group(1)) if m else None, flatpak=True)
    return None


def _parse_info(text: str) -> dict:
    info = {}
    for line in text.splitlines():
        m = re.match(r"\s*([a-z_]+)\s*=\s*(.+?)\s*$", line)
        if not m:
            continue
        key, value = m.groups()
        try:
            info[key] = value if key == "manifold" else float(value)
        except ValueError:
            info[key] = value
    return info


def _parse_gcode(path: Path) -> tuple[int, str | None, float | None]:
    layers = 0
    print_time = None
    filament = None
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(";LAYER_CHANGE"):
                layers += 1
            elif line.startswith("; estimated printing time (normal mode)"):
                print_time = line.split("=", 1)[-1].strip()
            elif line.startswith("; filament used [mm]"):
                try:
                    filament = float(line.split("=", 1)[1].split(",")[0])
                except ValueError:
                    pass
    return layers, print_time, filament


def _tail(text: str, lines: int = 25) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _info(slicer: Slicer, model_path: Path, timeout: float) -> tuple[dict, str | None]:
    try:
        with tempfile.TemporaryDirectory(prefix="scan-to-3mf-info-") as scratch:  # some slicers write result.json to cwd
            res = subprocess.run(slicer.argv + ["--info", str(model_path)], capture_output=True, text=True,
                                 timeout=timeout, cwd=scratch)
    except subprocess.TimeoutExpired:
        return {}, "timed out reading the file"
    except OSError as exc:
        return {}, f"could not run: {exc}"
    info = _parse_info(res.stdout)
    return info, None if info else "printed no model information"


def slice_check(slicer: Slicer, model_path: str | Path, extents_mm: list[float] | None = None,
                timeout: float = SLICE_TIMEOUT_S, workdir: str | Path | None = None) -> SliceResult:
    model_path = Path(model_path).resolve()
    t0 = time.perf_counter()
    info, problem = _info(slicer, model_path, timeout)
    if not slicer.can_slice:
        message = (f"{slicer.label} read the file (size and closed-mesh check below), but its command line "
                   "needs printer presets to slice, so the slice itself was not run. Install PrusaSlicer for "
                   "a full slice test.")
        if problem:
            message = f"{slicer.label} {problem}; no slice test was run."
        return SliceResult(None, slicer.label, " ".join(slicer.argv + ["--info"]), time.perf_counter() - t0,
                           message, info=info)
    if problem and "timed out" in problem:
        return SliceResult(False, slicer.label, "--info", time.perf_counter() - t0, f"Slicer {problem}")

    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="scan-to-3mf-slice-"))
    tmp.mkdir(parents=True, exist_ok=True)
    gcode = tmp / (model_path.stem + ".gcode")
    size = max(extents_mm or [200.0])
    bed = float(max(250, math.ceil(size * 1.5 + 50)))
    cmd = slicer.argv + [
        "--export-gcode", "--loglevel", "2",
        "--layer-height", f"{LAYER_HEIGHT:g}", "--first-layer-height", f"{LAYER_HEIGHT:g}",
        "--bed-shape", f"0x0,{bed:g}x0,{bed:g}x{bed:g},0x{bed:g}",
        "--max-print-height", f"{bed:g}",
        "--center", f"{bed / 2:g},{bed / 2:g}",
        "--output", str(gcode), str(model_path),
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp)
    except subprocess.TimeoutExpired:
        return SliceResult(False, slicer.label, " ".join(cmd), time.perf_counter() - t0,
                           f"Slicing did not finish within {timeout:.0f} s", info=info)
    log = (res.stdout or "") + (res.stderr or "")
    warnings = [line.strip() for line in log.splitlines()
                if ("warning" in line.lower() or "error" in line.lower()) and "opengl" not in line.lower()][:10]
    seconds = time.perf_counter() - t0
    if res.returncode != 0 or not gcode.exists() or gcode.stat().st_size == 0:
        return SliceResult(False, slicer.label, " ".join(cmd), seconds, f"Slicer failed (exit code {res.returncode})",
                           info=info, warnings=warnings, log_tail=_tail(log))
    layers, print_time, filament = _parse_gcode(gcode)
    ok = layers > 0
    message = f"Sliced into {layers:,} layers" if ok else "Slicer produced G-code with no layers"
    expected = None
    if ok and extents_mm:
        expected = max(1, round(extents_mm[2] / LAYER_HEIGHT))
        if layers < expected * 0.98 - 2:
            ok = False
            message = (f"Sliced into only {layers:,} of about {expected:,} layers at {LAYER_HEIGHT:g} mm: "
                       "part of the model was lost, usually because of open holes")
    result = SliceResult(ok, slicer.label, " ".join(cmd), seconds, message, info=info, layers=layers,
                         expected_layers=expected, print_time=print_time, filament_mm=filament,
                         gcode_bytes=gcode.stat().st_size, warnings=warnings, log_tail=_tail(log))
    if not workdir:
        shutil.rmtree(tmp, ignore_errors=True)
    return result
