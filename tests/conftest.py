from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def scan_dirs() -> list[Path]:
    dirs = [Path(p) for p in os.environ.get("SCAN_TO_3MF_SCANS", "").split(os.pathsep) if p]
    return dirs + [ROOT / "data" / "scans"]


def find_scan(name: str) -> Path | None:
    for folder in scan_dirs():
        for dirpath, _, files in os.walk(folder, followlinks=True):
            if name in files:
                return Path(dirpath) / name
    return None


def box(size=(20.0, 30.0, 10.0), center=(0.0, 0.0, 5.0)) -> tuple[np.ndarray, np.ndarray]:
    m = trimesh.creation.box(extents=size)
    m.apply_translation(center)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def sphere(radius=10.0, subdivisions=3, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    m = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    m.apply_translation(center)
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


def write_stl(path: Path, v: np.ndarray, f: np.ndarray, ascii: bool = False) -> Path:
    trimesh.Trimesh(v, f, process=False).export(path, file_type="stl_ascii" if ascii else "stl")
    return path


def write_ply(path: Path, v: np.ndarray, f: np.ndarray, ascii: bool = False) -> Path:
    data = trimesh.exchange.ply.export_ply(trimesh.Trimesh(v, f, process=False), encoding="ascii" if ascii else "binary")
    path.write_bytes(data)
    return path


def write_obj(path: Path, v: np.ndarray, f: np.ndarray) -> Path:
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in f]
    path.write_text("\n".join(lines) + "\n")
    return path


def remove_faces_around(v: np.ndarray, f: np.ndarray, point, radius: float) -> np.ndarray:
    """Drop faces whose centroid is within ``radius`` of ``point`` (makes a hole)."""
    centres = v[f].mean(axis=1)
    return f[np.linalg.norm(centres - np.asarray(point), axis=1) > radius]


@pytest.fixture
def no_slicer(monkeypatch):
    from scan_to_3mf.engine import pipeline

    monkeypatch.setattr(pipeline, "find_slicer", lambda *a, **k: None)
