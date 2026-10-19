"""Load STL, OBJ, PLY, and 3MF into plain numpy arrays (geometry only)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .threemf import ThreeMFError, read_3mf

SUPPORTED_EXTENSIONS = (".stl", ".obj", ".ply", ".3mf")


class MeshLoadError(Exception):
    pass


@dataclass
class LoadedMesh:
    vertices: np.ndarray  # (n, 3) float64, in file units
    faces: np.ndarray  # (m, 3) int64
    fmt: str  # "stl" | "obj" | "ply" | "3mf"
    name: str
    file_unit: str | None = None  # declared by the file (3MF only)
    soup: bool = False  # format stores every triangle's corners separately (STL)
    file_triangles: int = 0
    notes: list[str] = field(default_factory=list)


def detect_format(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise MeshLoadError(
            f"Unsupported file type {ext or '(none)'}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return ext[1:]


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Positions and faces only. Texture and normal indices are ignored, so UV
    seams do not split the surface into separate pieces."""
    v_rows: list[list[str]] = []
    polys: dict[int, list[list[int]]] = defaultdict(list)
    has_extras = False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith(("v ", "v\t")):
                parts = line.split()
                if len(parts) < 4:
                    raise MeshLoadError(f"Bad OBJ vertex line: {line.strip()!r}")
                v_rows.append(parts[1:4])
            elif line.startswith(("f ", "f\t")):
                n_v = len(v_rows)
                idx = []
                for tok in line.split()[1:]:
                    head = tok.split("/", 1)[0]
                    if not head:
                        continue
                    i = int(head)
                    idx.append(i - 1 if i > 0 else n_v + i)
                if len(idx) >= 3:
                    polys[len(idx)].append(idx)
            elif line.startswith(("vt", "vn", "usemtl", "mtllib")):
                has_extras = True
    if not v_rows:
        raise MeshLoadError("The OBJ file has no vertices")
    vertices = np.array(v_rows, dtype=np.float64)
    tris = []
    split = 0
    for n, rows in polys.items():
        arr = np.array(rows, dtype=np.int64)
        if n > 3:
            split += len(arr)
        for k in range(1, n - 1):
            tris.append(np.stack([arr[:, 0], arr[:, k], arr[:, k + 1]], axis=1))
    faces = np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    notes = []
    if split:
        notes.append(f"Split {split:,} OBJ polygons with more than three corners into triangles")
    if has_extras:
        notes.append("Ignored OBJ texture coordinates, normals, and materials (output is geometry only)")
    return vertices, faces, notes


def _load_with_trimesh(path: Path, fmt: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import trimesh

    try:
        loaded = trimesh.load(str(path), file_type=fmt, process=False, force="mesh", skip_materials=True)
    except Exception as exc:
        raise MeshLoadError(f"Could not read the {fmt.upper()} file: {exc}") from exc
    faces = getattr(loaded, "faces", None)
    if faces is None or len(faces) == 0:
        raise MeshLoadError(
            f"The {fmt.upper()} file has no triangles. If it is a point cloud, mesh it first "
            "(for example in the scanner software, MeshLab, or CloudCompare)."
        )
    notes = []
    kind = getattr(getattr(loaded, "visual", None), "kind", None)
    if kind in ("vertex", "face", "texture"):
        notes.append("Ignored color/texture data (output is geometry only)")
    return np.asarray(loaded.vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), notes


def load_mesh(path: str | Path) -> LoadedMesh:
    path = Path(path)
    fmt = detect_format(path)
    if not path.is_file():
        raise MeshLoadError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise MeshLoadError("The file is empty")
    file_unit = None
    soup = False
    if fmt == "3mf":
        try:
            content = read_3mf(path)
        except ThreeMFError as exc:
            raise MeshLoadError(str(exc)) from exc
        vertices, faces, notes = content.vertices, content.faces, list(content.notes)
        file_unit = content.unit
        if content.instances > 1:
            notes.append(f"Merged {content.instances} objects from the 3MF build into one mesh")
    elif fmt == "obj":
        vertices, faces, notes = _load_obj(path)
    else:
        vertices, faces, notes = _load_with_trimesh(path, fmt)
        soup = fmt == "stl"

    if len(faces) == 0:
        raise MeshLoadError("The file has no triangles")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise MeshLoadError("The file has triangles that point at vertices that do not exist")
    finite = np.all(np.isfinite(vertices), axis=1)
    if not finite.all():
        bad = ~np.all(finite[faces], axis=1)
        faces = faces[~bad]
        notes.append(f"Dropped {int(bad.sum()):,} triangles with invalid (NaN/infinite) coordinates")
        if len(faces) == 0:
            raise MeshLoadError("All triangles have invalid coordinates")
    return LoadedMesh(
        vertices=vertices, faces=faces, fmt=fmt, name=path.stem, file_unit=file_unit,
        soup=soup, file_triangles=int(len(faces)), notes=notes,
    )
