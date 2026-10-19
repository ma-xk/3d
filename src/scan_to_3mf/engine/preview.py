"""Compact binary mesh format for the browser preview.

Layout (little endian): b"S3MP", u32 version, u32 vertex count, u32 triangle
count, u32 rim-edge count, u32 non-manifold-edge count, then float32
positions, uint32 indices, float32 rim segments, float32 non-manifold segments.
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"S3MP"
VERSION = 1
MAX_PREVIEW_TRIANGLES = 1_500_000


def encode_preview(vertices: np.ndarray, faces: np.ndarray, rim: np.ndarray | None = None,
                   non_manifold: np.ndarray | None = None, max_triangles: int = MAX_PREVIEW_TRIANGLES) -> bytes:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) > max_triangles:
        from . import mlab

        ms = mlab.meshset(v, f)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(max_triangles), preservetopology=False,
                                                    autoclean=True)
        v, f = mlab.arrays(ms)
    used = np.unique(f)
    remap = np.full(len(v), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    v = v[used]
    f = remap[f]
    rim = np.zeros((0, 2, 3)) if rim is None else np.asarray(rim)
    non_manifold = np.zeros((0, 2, 3)) if non_manifold is None else np.asarray(non_manifold)
    header = MAGIC + struct.pack("<5I", VERSION, len(v), len(f), len(rim), len(non_manifold))
    return b"".join([
        header,
        np.ascontiguousarray(v, dtype="<f4").tobytes(),
        np.ascontiguousarray(f, dtype="<u4").tobytes(),
        np.ascontiguousarray(rim, dtype="<f4").tobytes(),
        np.ascontiguousarray(non_manifold, dtype="<f4").tobytes(),
    ])


def decode_preview(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data[:4] != MAGIC:
        raise ValueError("not a preview blob")
    _, nv, nf, nr, nn = struct.unpack("<5I", data[4:24])
    off = 24
    v = np.frombuffer(data, "<f4", nv * 3, off).reshape(-1, 3)
    off += nv * 12
    f = np.frombuffer(data, "<u4", nf * 3, off).reshape(-1, 3)
    off += nf * 12
    r = np.frombuffer(data, "<f4", nr * 6, off).reshape(-1, 2, 3)
    off += nr * 24
    n = np.frombuffer(data, "<f4", nn * 6, off).reshape(-1, 2, 3)
    return v, f, r, n
