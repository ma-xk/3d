"""Thin helpers around pymeshlab (MeshLab's C++ mesh filters)."""

from __future__ import annotations

import numpy as np

REQUIRED_FILTERS = (
    "meshing_close_holes",
    "meshing_re_orient_faces_coherently",
    "meshing_repair_non_manifold_edges",
    "meshing_repair_non_manifold_vertices",
    "meshing_decimation_quadric_edge_collapse",
    "compute_selection_by_condition_per_face",
    "compute_selection_by_self_intersections_per_face",
)


class MeshLabUnavailable(RuntimeError):
    pass


_checked = False


def _pymeshlab():
    global _checked
    import pymeshlab

    if not _checked:
        available = set(pymeshlab.filter_list())
        missing = [name for name in REQUIRED_FILTERS if name not in available]
        if missing:
            raise MeshLabUnavailable(
                "pymeshlab loaded without its meshing plugin (missing: "
                + ", ".join(missing)
                + "). On Linux install the OpenGL runtime, e.g. `sudo apt install libopengl0`."
            )
        _checked = True
    return pymeshlab


def meshset(vertices: np.ndarray, faces: np.ndarray, face_quality: np.ndarray | None = None):
    pm = _pymeshlab()
    kwargs = {
        "vertex_matrix": np.ascontiguousarray(vertices, dtype=np.float64),
        "face_matrix": np.ascontiguousarray(faces, dtype=np.int32),
    }
    if face_quality is not None:
        kwargs["f_scalar_array"] = np.ascontiguousarray(face_quality, dtype=np.float64)
    ms = pm.MeshSet()
    ms.set_verbosity(False)
    ms.add_mesh(pm.Mesh(**kwargs))
    return ms


def arrays(ms) -> tuple[np.ndarray, np.ndarray]:
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float64, copy=True), m.face_matrix().astype(np.int64, copy=True)


def count_self_intersections(vertices: np.ndarray, faces: np.ndarray) -> int:
    ms = meshset(vertices, faces)
    ms.compute_selection_by_self_intersections_per_face()
    return int(ms.current_mesh().selected_face_number())
