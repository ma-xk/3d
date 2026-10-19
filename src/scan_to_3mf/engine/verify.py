"""Read every written 3MF back and check it against what we meant to write."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .threemf import read_3mf_lib3mf, read_3mf_unit_attribute, read_3mf_xml


@dataclass
class Check:
    key: str
    label: str
    ok: bool | None  # None means skipped
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt_xyz(v) -> str:
    return " × ".join(f"{x:.2f}" for x in v) + " mm"


def _compare(label: str, key: str, content, vertices: np.ndarray, faces: np.ndarray) -> Check:
    if content.unit != "mm":
        return Check(key, label, False, f"read back as {content.unit}, expected millimeters")
    if content.vertices.shape != vertices.shape or content.faces.shape != faces.shape:
        return Check(key, label, False,
                     f"read back {len(content.faces):,} triangles / {len(content.vertices):,} vertices, "
                     f"expected {len(faces):,} / {len(vertices):,}")
    if not np.array_equal(content.faces, faces):
        return Check(key, label, False, "triangle indices differ from what was written")
    expected = vertices.astype(np.float32).astype(np.float64)
    err = float(np.abs(content.vertices - expected).max()) if len(expected) else 0.0
    tol = 1e-5 + 1e-7 * float(np.abs(expected).max() if len(expected) else 0.0)
    if err > tol:
        return Check(key, label, False, f"coordinates differ by up to {err:.2e} mm")
    return Check(key, label, True, f"{len(faces):,} triangles, unit millimeter, max coordinate error {err:.1e} mm")


def verify_3mf(path: str | Path, vertices: np.ndarray, faces: np.ndarray, reference_extents: list[float],
               *, size_tolerance: float = 0.005) -> list[Check]:
    """``reference_extents`` is the size (mm) the kept input geometry had after
    applying the unit, before any cleaning. The output must match it."""
    checks: list[Check] = []

    unit_attr = read_3mf_unit_attribute(path)
    checks.append(Check("unit", "3MF declares millimeters", unit_attr == "millimeter",
                        f'<model unit="{unit_attr}">' if unit_attr else "no unit attribute (spec default is mm)"))

    try:
        checks.append(_compare("Reads back with lib3mf (3MF reference library)", "lib3mf",
                               read_3mf_lib3mf(path), vertices, faces))
    except Exception as exc:  # lib3mf missing or rejected the file
        checks.append(Check("lib3mf", "Reads back with lib3mf (3MF reference library)", False, str(exc)))

    xml = None
    try:
        xml = read_3mf_xml(path)
        checks.append(_compare("Reads back with an independent XML parser", "xml", xml, vertices, faces))
    except Exception as exc:
        checks.append(Check("xml", "Reads back with an independent XML parser", False, str(exc)))

    if xml is not None and len(xml.faces):
        used = xml.vertices[np.unique(xml.faces)]
        ext = used.max(axis=0) - used.min(axis=0)
        ref = np.asarray(reference_extents, dtype=np.float64)
        diff = np.abs(ext - ref)
        allowed = np.maximum(size_tolerance * ref, 0.05)
        ok = bool(np.all(diff <= allowed))
        detail = f"file {_fmt_xyz(ext)} vs input {_fmt_xyz(ref)}"
        if not ok:
            ratio = float(np.max(ext) / max(np.max(ref), 1e-12))
            detail += f" (size ratio {ratio:.4g})"
        checks.append(Check("size", "Size matches the input after units", ok, detail))

    try:
        import manifold3d

        mesh = manifold3d.Mesh(vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
                               tri_verts=np.ascontiguousarray(faces, dtype=np.uint32))
        solid = manifold3d.Manifold(mesh)
        status = solid.status()
        if status == manifold3d.Error.NoError and solid.volume() > 0:
            checks.append(Check("manifold", "Closed, outward-facing solid (manifold3d)", True,
                                f"volume {solid.volume() / 1000:.2f} cm³, genus {solid.genus()}"))
        elif status == manifold3d.Error.NoError:
            checks.append(Check("manifold", "Closed, outward-facing solid (manifold3d)", False,
                                "closed but inside-out (negative volume)"))
        else:
            checks.append(Check("manifold", "Closed, outward-facing solid (manifold3d)", False,
                                f"not a closed manifold ({str(status).split('.')[-1]})"))
    except ImportError:
        checks.append(Check("manifold", "Closed, outward-facing solid (manifold3d)", None, "manifold3d not installed"))
    return checks
