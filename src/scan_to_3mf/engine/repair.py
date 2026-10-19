"""Cleaning steps.

Every step takes numpy arrays (vertices in mm, int64 faces) and returns new
arrays plus plain numbers describing what it changed. None of them scales the
model. Surface is only invented by hole filling, which is capped by size.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

from . import analysis as an
from . import mlab


def unique_rows(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact duplicate rows. Returns (index of first copy per unique row, inverse map)."""
    v = vertices
    order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
    vs = v[order]
    new_group = np.r_[True, np.any(vs[1:] != vs[:-1], axis=1)]
    group_id = np.cumsum(new_group) - 1
    inverse = np.empty(len(v), dtype=np.int64)
    inverse[order] = group_id
    first = order[new_group]
    return first, inverse


def compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Drop vertices no face uses. Returns (vertices, faces, removed_count)."""
    if len(faces) == 0:
        return vertices[:0], faces, len(vertices)
    used = np.unique(faces)
    if len(used) == len(vertices):
        return vertices, faces, 0
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces], int(len(vertices) - len(used))


def merge_identical(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    first, inverse = unique_rows(vertices)
    if len(first) == len(vertices):
        return vertices, faces, 0
    return vertices[first], inverse[faces], int(len(vertices) - len(first))


def weld_boundary(vertices: np.ndarray, faces: np.ndarray, tolerance_mm: float) -> tuple[np.ndarray, int]:
    """Close cracks: merge open-rim vertices that sit within ``tolerance_mm`` of
    another open-rim vertex. Interior vertices are never merged this way, because
    merging two touching sheets would pinch a watertight surface."""
    topo = an.edge_topology(faces, len(vertices))
    bh = topo.boundary_halfedges
    if len(bh) == 0:
        return faces, 0
    bverts = np.unique(np.r_[topo.he_from[bh], topo.he_to[bh]])
    pairs = cKDTree(vertices[bverts]).query_pairs(max(tolerance_mm, 0.0), output_type="ndarray")
    if len(pairs) == 0:
        return faces, 0
    n = len(bverts)
    g = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)
    rep = np.full(n_comp, n, dtype=np.int64)
    np.minimum.at(rep, labels, np.arange(n))
    remap = np.arange(len(vertices), dtype=np.int64)
    remap[bverts] = bverts[rep[labels]]
    return remap[faces], int(n - n_comp)


def collapse_short_edges(vertices: np.ndarray, faces: np.ndarray, tolerance_mm: float, max_rounds: int = 10
                         ) -> tuple[np.ndarray, int]:
    """Collapse edges no longer than ``tolerance_mm`` (zero-length "needle" edges).

    A collapse is only done when it keeps the surface manifold (the link
    condition: the two end vertices share exactly the neighbours opposite the
    edge). The kept vertex does not move, so the shape changes by at most the
    tolerance. The one or two faces on the edge disappear; no hole is created.
    """
    from scipy.sparse import csr_matrix

    f = faces
    total = 0
    for _ in range(max_rounds):
        topo = an.edge_topology(f, len(vertices))
        rep_h = topo.order[topo.starts]
        a_all, b_all = topo.he_from[rep_h], topo.he_to[rep_h]
        lengths = np.linalg.norm(vertices[a_all] - vertices[b_all], axis=1)
        cand = np.flatnonzero((lengths <= tolerance_mm) & (topo.counts <= 2))
        if len(cand) == 0:
            break
        cand = cand[np.argsort(lengths[cand], kind="stable")]
        n = len(vertices)
        adj = csr_matrix((np.ones(2 * len(a_all), dtype=np.int8), (np.r_[a_all, b_all], np.r_[b_all, a_all])), shape=(n, n))
        indptr, indices = adj.indptr, adj.indices
        is_rim = np.zeros(n, dtype=bool)
        bh = topo.boundary_halfedges
        is_rim[topo.he_from[bh]] = True
        is_rim[topo.he_to[bh]] = True
        remap = np.arange(n, dtype=np.int64)
        touched: set[int] = set()
        merged = 0
        for g in cand:
            a, b = int(a_all[g]), int(b_all[g])
            if a in touched or b in touched:
                continue
            hs = topo.order[topo.starts[g]: topo.starts[g] + topo.counts[g]]
            opposite = {int(f[h // 3, (h % 3 + 2) % 3]) for h in hs}
            ring_a = set(indices[indptr[a]:indptr[a + 1]].tolist())
            ring_b = set(indices[indptr[b]:indptr[b + 1]].tolist())
            if (ring_a & ring_b) != opposite:
                continue
            if topo.counts[g] == 2 and is_rim[a] and is_rim[b]:
                continue
            remap[b] = a
            touched.update(ring_a)
            touched.update(ring_b)
            touched.update((a, b))
            merged += 1
        if merged == 0:
            break
        f = remap[f]
        f = f[~an.degenerate_index_mask(f)]
        total += merged
    return f, total


def drop_degenerate(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove faces that use the same vertex twice (zero area after welding)."""
    mask = an.degenerate_index_mask(faces)
    return faces[~mask], int(mask.sum())


def fix_zero_area(vertices: np.ndarray, faces: np.ndarray, max_rounds: int = 8) -> tuple[np.ndarray, dict]:
    """Remove zero-area slivers whose three corners are distinct but collinear.

    Deleting such a face would open a crack, so each one is removed by flipping
    its long edge: the neighbour across that edge is split at the sliver's middle
    vertex. Slivers that cannot be flipped safely are left and reported.
    """
    fixed = 0
    f = faces.copy()
    for _ in range(max_rounds):
        zero = np.flatnonzero(an.zero_area_mask(vertices, f) & ~an.degenerate_index_mask(f))
        if len(zero) == 0:
            break
        topo = an.edge_topology(f, len(vertices))
        n_edges = len(topo.starts)
        he_group = np.empty(len(topo.order), dtype=np.int64)
        he_group[topo.order] = np.repeat(np.arange(n_edges), topo.counts)
        touched: set[int] = set()
        remove: list[int] = []
        add: list[list[int]] = []
        existing_edges = None
        for fi in zero:
            tri = vertices[f[fi]]
            lengths = [np.linalg.norm(tri[(k + 1) % 3] - tri[k]) for k in range(3)]
            k = int(np.argmax(lengths))  # long edge runs from corner k to k+1
            h = 3 * fi + k
            g = he_group[h]
            if topo.counts[g] != 2:
                continue
            s = topo.starts[g]
            other = topo.order[s] if topo.order[s] != h else topo.order[s + 1]
            gi = int(other // 3)
            if gi in touched or fi in touched or topo.he_from[other] == topo.he_from[h]:
                continue
            p, q, r = f[fi, k], f[fi, (k + 1) % 3], f[fi, (k + 2) % 3]
            s_vert = [x for x in f[gi] if x != p and x != q]
            if len(s_vert) != 1:
                continue
            s_vert = s_vert[0]
            if existing_edges is None:
                lo = np.minimum(topo.he_from, topo.he_to)
                hi = np.maximum(topo.he_from, topo.he_to)
                existing_edges = set((lo * len(vertices) + hi).tolist())
            key = min(r, s_vert) * len(vertices) + max(r, s_vert)
            if key in existing_edges:
                continue
            new = [[r, p, s_vert], [r, s_vert, q]]
            if np.any(an.zero_area_mask(vertices, np.array(new))):
                continue
            touched.update((fi, gi))
            remove += [fi, gi]
            add += new
            existing_edges.add(key)
        if not remove:
            break
        keep = np.ones(len(f), dtype=bool)
        keep[remove] = False
        f = np.vstack([f[keep], np.array(add, dtype=np.int64)])
        fixed += len(remove) // 2
    remaining = int((an.zero_area_mask(vertices, f) & ~an.degenerate_index_mask(f)).sum())
    return f, {"fixed": fixed, "remaining": remaining}


def drop_duplicate_faces(faces: np.ndarray) -> tuple[np.ndarray, int]:
    mask = an.duplicate_face_mask(faces)
    return faces[~mask], int(mask.sum())


@dataclass
class ShellRemoval:
    removed: list[an.Shell] = field(default_factory=list)
    kept: int = 0
    threshold_triangles: int = 0

    @property
    def removed_triangles(self) -> int:
        return sum(s.triangles for s in self.removed)


def remove_small_shells(vertices: np.ndarray, faces: np.ndarray, min_fraction: float
                        ) -> tuple[np.ndarray, ShellRemoval]:
    """Drop shells with fewer triangles than ``min_fraction`` x the largest shell."""
    n, labels = an.face_components(faces, len(vertices))
    result = ShellRemoval(kept=n)
    if n <= 1 or min_fraction <= 0:
        return faces, result
    counts = np.bincount(labels, minlength=n)
    threshold = int(np.ceil(min_fraction * counts.max()))
    small = counts < threshold
    result.threshold_triangles = threshold
    if not small.any():
        return faces, result
    shells = an.describe_shells(vertices, faces, labels, n)
    result.removed = sorted((shells[i] for i in np.flatnonzero(small)), key=lambda s: -s.triangles)
    result.kept = int((~small).sum())
    return faces[~small[labels]], result


def repair_non_manifold(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Remove the smallest faces on edges shared by 3+ faces, then split pinched vertices."""
    ms = mlab.meshset(vertices, faces)
    n_faces, n_verts = len(faces), len(vertices)
    ms.meshing_repair_non_manifold_edges(method="Remove Faces")
    after_edges = ms.current_mesh().face_number()
    ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
    v, f = mlab.arrays(ms)
    return v, f, {"faces_removed": int(n_faces - after_edges), "vertices_added": int(len(v) - n_verts)}


def _count_flips(before: np.ndarray, after: np.ndarray) -> int:
    if before.shape != after.shape:
        return -1
    same_set = np.all(np.sort(before, axis=1) == np.sort(after, axis=1), axis=1)
    if not same_set.all():
        return -1
    cyc = [after, after[:, [1, 2, 0]], after[:, [2, 0, 1]]]
    same_orientation = np.zeros(len(before), dtype=bool)
    for c in cyc:
        same_orientation |= np.all(before == c, axis=1)
    return int((~same_orientation).sum())


def orient_outward(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, dict]:
    """Make winding consistent inside each shell, then flip shells that face inward."""
    ms = mlab.meshset(vertices, faces)
    try:
        ms.meshing_re_orient_faces_coherently()
        v2, f2 = mlab.arrays(ms)
    except Exception:  # MeshLab needs a manifold mesh; keep the winding as is
        v2, f2 = vertices, faces.copy()
    if len(v2) != len(vertices) or len(f2) != len(faces):
        v2, f2 = vertices, faces.copy()
    flipped_faces = _count_flips(faces, f2)
    n, labels = an.face_components(f2, len(vertices))
    vols = an.shell_signed_volumes(vertices, f2, labels, n)
    inward = vols < 0
    if inward.any():
        rows = inward[labels]
        f2[rows] = f2[rows][:, ::-1]
    total_flipped = _count_flips(faces, f2)
    return f2, {
        "coherence_flips": flipped_faces,
        "shells_turned_outward": int(inward.sum()),
        "faces_flipped": total_flipped,
    }


@dataclass
class HoleFill:
    candidates: list[an.Hole] = field(default_factory=list)  # holes we tried to fill
    skipped: list[an.Hole] = field(default_factory=list)  # holes above the size cap
    faces_added: int = 0
    unfilled_candidates: int = 0
    fan_filled: int = 0
    leftover_holes: int = 0  # holes MeshLab's guarded pass left open
    leftover_method: str | None = None
    leftover_crossings: int | None = None


PLANAR_CAP_MIN_EDGES = 64


def trace_rim_loops(he_from: np.ndarray, he_to: np.ndarray) -> list[np.ndarray]:
    """Split rim half-edges into simple loops (no vertex repeated within a loop).
    Where two holes touch at a vertex, they come out as separate loops."""
    outgoing: dict[int, list[int]] = {}
    for i, a in enumerate(he_from.tolist()):
        outgoing.setdefault(a, []).append(i)
    used = np.zeros(len(he_from), dtype=bool)
    loops: list[np.ndarray] = []
    for start in range(len(he_from)):
        if used[start]:
            continue
        used[start] = True
        path = [start]
        position = {int(he_from[start]): 0}
        current = int(he_to[start])
        while True:
            if current in position:
                k = position[current]
                loops.append(np.array(path[k:], dtype=np.int64))
                for e in path[k:]:
                    position.pop(int(he_from[e]), None)
                path = path[:k]
                if not path:
                    break
                current = int(he_to[path[-1]])
                continue
            nxt = next((e for e in outgoing.get(current, ()) if not used[e]), None)
            if nxt is None:
                break
            used[nxt] = True
            position[current] = len(path)
            path.append(nxt)
            current = int(he_to[nxt])
    return loops


@dataclass
class _Projection:
    tris: np.ndarray  # (n - 2, 3) indices into the ring
    xy: np.ndarray  # (n, 2) ring positions in the projection plane
    h: np.ndarray  # (n,) ring heights above the plane
    u: np.ndarray
    w: np.ndarray
    n: np.ndarray


def _triangulate_projected(pts: np.ndarray, normal: np.ndarray) -> _Projection | None:
    import manifold3d

    n = normal / (np.linalg.norm(normal) or 1.0)
    u = np.cross(n, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(n, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    xy = np.stack([pts @ u, pts @ w], axis=1)
    x, y = xy[:, 0], xy[:, 1]
    signed = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    if signed < 0:
        xy[:, 1] *= -1.0
        w = -w
    area = abs(signed)
    if area <= 0:
        return None
    tris = np.asarray(manifold3d.triangulate([np.ascontiguousarray(xy)]), dtype=np.int64).reshape(-1, 3)
    if len(tris) != len(pts) - 2:
        return None
    p = xy[tris]
    tri_area = 0.5 * np.abs((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                            - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1]))
    if abs(tri_area.sum() - area) > 1e-6 * area + 1e-9:
        return None  # overlapping triangles: the loop folds over in this projection
    return _Projection(tris, xy, pts @ n, u, w, n)


def _inside_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test, vectorized in chunks."""
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    dy = np.where(y2 == y1, 1e-300, y2 - y1)
    inside = np.zeros(len(points), dtype=bool)
    step = max(1, 20_000_000 // max(len(polygon), 1))
    for s in range(0, len(points), step):
        px = points[s:s + step, 0][:, None]
        py = points[s:s + step, 1][:, None]
        crosses = ((y1 > py) != (y2 > py)) & (px < (x2 - x1) * (py - y1) / dy + x1)
        inside[s:s + step] = (np.count_nonzero(crosses, axis=1) % 2) == 1
    return inside


def _lattice_cap(xy: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Triangulate the polygon ``xy`` (counter-clockwise) with its own vertices
    plus a hexagonal lattice of interior points, using a Delaunay triangulation
    clipped to the polygon. Returns (points, triangles, missing rim edges)."""
    from scipy.spatial import Delaunay, QhullError

    n = len(xy)
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    dy = spacing * np.sqrt(3.0) / 2.0
    rows = []
    for i, yv in enumerate(np.arange(lo[1] + dy / 2, hi[1], dy)):
        xs = np.arange(lo[0] + spacing * (0.25 if i % 2 else 0.75), hi[0], spacing)
        rows.append(np.column_stack([xs, np.full(len(xs), yv)]))
    grid = np.vstack(rows) if rows else np.zeros((0, 2))
    if len(grid):
        grid = grid[_inside_polygon(grid, xy)]
    if len(grid):
        seg = np.roll(xy, -1, axis=0) - xy
        steps = np.maximum(1, np.ceil(np.linalg.norm(seg, axis=1) / (0.25 * spacing))).astype(int)
        start = np.repeat(np.arange(n), steps)
        frac = np.arange(int(steps.sum())) - np.repeat(np.cumsum(steps) - steps, steps)
        dense = xy[start] + seg[start] * (frac / np.repeat(steps, steps))[:, None]
        dist, _ = cKDTree(dense).query(grid)
        grid = grid[dist >= 0.6 * spacing]
    pts = np.vstack([xy, grid])
    try:
        tri = Delaunay(pts).simplices.astype(np.int64)
    except (QhullError, ValueError):
        return None
    p = pts[tri]
    cross = (p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])
    tri = np.where((cross < 0)[:, None], tri[:, [0, 2, 1]], tri)
    tri = tri[np.abs(cross) > 0]
    tri = tri[_inside_polygon(pts[tri].mean(axis=1), xy)]
    e = tri[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    have = set((e[:, 0] * len(pts) + e[:, 1]).tolist())
    rim = np.arange(n)
    missing = sum(1 for a, b in zip(rim.tolist(), np.roll(rim, -1).tolist(), strict=True) if a * len(pts) + b not in have)
    return pts, tri, missing


def _membrane_heights(points: np.ndarray, tris: np.ndarray, n_fixed: int) -> np.ndarray:
    """Give the cap's interior vertices the heights of a membrane stretched over
    the rim (discrete Laplace equation). Heights stay within the rim's range and
    x/y positions do not move, so the cap still cannot fold."""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve

    n = len(points)
    if n == n_fixed:
        return points
    e = np.sort(np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1).reshape(-1, 2), axis=1)
    e = np.unique(e, axis=0)
    adj = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    free = np.arange(n_fixed, n)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    lap = (diags(degree[free]) - adj[free][:, free]).tocsc()
    rhs = adj[free][:, :n_fixed] @ points[:n_fixed, 2]
    out = points.copy()
    out[free, 2] = spsolve(lap, rhs)
    return out


@dataclass
class Cap:
    ring: np.ndarray  # rim vertex ids, in cap order
    new_vertices: np.ndarray  # (k, 3) interior vertices added by refinement
    faces: np.ndarray  # (m, 3) indices: < len(ring) into ring, otherwise into new_vertices


def planar_cap(vertices: np.ndarray, loop_vertices: np.ndarray, extra_normals=(), existing_edges=None,
               max_edge_mm: float | None = None) -> Cap | None:
    """Close a rim loop with a cap that is triangulated in 2D and lifted back.

    Tries the rim's best-fit plane, then ``extra_normals``, then the X/Y/Z axes.
    A projection is usable when the loop is a simple polygon in it. Among usable
    projections the one whose cap edges duplicate the fewest existing mesh edges
    wins; a few duplicates (at most 1 %) are accepted because the non-manifold
    repair that follows removes them.

    The cap is meshed with the rim's vertices plus a lattice of interior points
    (Delaunay, clipped to the rim) whose heights form a membrane over the rim.
    Over its projection plane it is a height field, so it cannot fold over
    itself. Rim edges the lattice mesh misses leave tiny gaps that the normal
    small-hole fill closes. Returns None if no projection works."""
    ring = loop_vertices[::-1]  # fill triangles must run against the rim edges
    centre = vertices[ring].mean(axis=0)
    pts = vertices[ring] - centre
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    candidates = [vt[2], *extra_normals, np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]),
                  np.array([1.0, 0.0, 0.0])]
    n_v = len(vertices)
    rim_keys = np.sort(np.minimum(ring, np.roll(ring, -1)) * n_v + np.maximum(ring, np.roll(ring, -1)))
    best, best_conflicts = None, None
    for normal in candidates:
        if np.linalg.norm(normal) == 0:
            continue
        proj = _triangulate_projected(pts, np.asarray(normal, dtype=np.float64))
        if proj is None:
            continue
        conflicts = 0
        if existing_edges is not None:
            cap = ring[proj.tris]
            e = np.stack([cap[:, [0, 1]], cap[:, [1, 2]], cap[:, [2, 0]]], axis=1).reshape(-1, 2)
            keys = np.unique(np.minimum(e[:, 0], e[:, 1]) * n_v + np.maximum(e[:, 0], e[:, 1]))
            chords = keys[~np.isin(keys, rim_keys, assume_unique=True)]
            idx = np.searchsorted(existing_edges, chords)
            idx[idx >= len(existing_edges)] = 0
            conflicts = int(np.sum(existing_edges[idx] == chords))
            if conflicts > max(3, 0.01 * len(chords)):
                continue
        if best is None or conflicts < best_conflicts:
            best, best_conflicts = proj, conflicts
        if conflicts == 0:
            break
    if best is None:
        return None
    rim_len = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
    x, y = best.xy[:, 0], best.xy[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
    spacing = max_edge_mm or max(4.0 * float(np.median(rim_len)), float(np.sqrt(area / 8000.0)))
    lattice = _lattice_cap(best.xy, spacing)
    if lattice is not None and lattice[2] <= max(3, 0.05 * len(ring)):
        pts2, tris, _ = lattice
        local = np.column_stack([pts2, np.r_[best.h, np.zeros(len(pts2) - len(ring))]])
        local = _membrane_heights(local, tris, len(ring))
    else:
        local, tris = np.column_stack([best.xy, best.h]), best.tris
    extra = local[len(ring):]
    new_vertices = centre + extra[:, [0]] * best.u + extra[:, [1]] * best.w + extra[:, [2]] * best.n
    return Cap(ring=ring, new_vertices=new_vertices, faces=tris)


def _planar_caps(vertices: np.ndarray, faces: np.ndarray, wanted, min_edges: int = PLANAR_CAP_MIN_EDGES
                 ) -> tuple[np.ndarray, np.ndarray, int]:
    topo = an.edge_topology(faces, len(vertices))
    bh = topo.boundary_halfedges
    if len(bh) == 0:
        return vertices, faces, 0
    a, b = topo.he_from[bh], topo.he_to[bh]
    rep = topo.order[topo.starts]
    n_v = len(vertices)
    existing = np.sort(np.minimum(topo.he_from[rep], topo.he_to[rep]) * n_v + np.maximum(topo.he_from[rep], topo.he_to[rep]))
    face_normals = None
    new_vertices, new_faces = [], []
    next_id = len(vertices)
    for loop in trace_rim_loops(a, b):
        if len(loop) < min_edges:
            continue
        loop_vertices = a[loop]
        pts = vertices[loop_vertices]
        lengths = np.linalg.norm(pts - np.roll(pts, -1, axis=0), axis=1)
        hole = an.Hole(edges=len(loop), perimeter_mm=float(lengths.sum()),
                       span_mm=float(pdist(pts).max()) if len(pts) <= 3000 else float(np.linalg.norm(pts.max(0) - pts.min(0))),
                       center=((pts.max(0) + pts.min(0)) / 2).tolist())
        if not wanted(hole):
            continue
        if face_normals is None:
            tri = vertices[faces]
            face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        rim_normal = face_normals[bh[loop] // 3].sum(axis=0)
        cap = planar_cap(vertices, loop_vertices, extra_normals=(rim_normal,), existing_edges=existing)
        if cap is None:
            continue
        index = np.r_[cap.ring, next_id + np.arange(len(cap.new_vertices))]
        new_faces.append(index[cap.faces])
        new_vertices.append(cap.new_vertices)
        next_id += len(cap.new_vertices)
    if not new_faces:
        return vertices, faces, 0
    return np.vstack([vertices] + new_vertices), np.vstack([faces] + new_faces), len(new_faces)


def _open_wanted(vertices: np.ndarray, faces: np.ndarray, wanted) -> int:
    topo = an.edge_topology(faces, len(vertices))
    bh, labels, n = an.boundary_loops(vertices, faces, topo)
    if n == 0:
        return 0
    return sum(1 for h in an.describe_holes(vertices, topo, bh, labels, int(n)) if wanted(h))


def _meshlab_close(vertices: np.ndarray, faces: np.ndarray, wanted, guarded: bool) -> tuple[np.ndarray, np.ndarray]:
    topo = an.edge_topology(faces, len(vertices))
    bh, labels, n = an.boundary_loops(vertices, faces, topo)
    if n == 0:
        return vertices, faces
    current = an.describe_holes(vertices, topo, bh, labels, int(n))
    todo = [i for i, h in enumerate(current) if wanted(h)]
    if not todo:
        return vertices, faces
    quality = np.zeros(len(faces))
    quality[bh[np.isin(labels, todo)] // 3] = 1.0
    ms = mlab.meshset(vertices, faces, face_quality=quality)
    ms.compute_selection_by_condition_per_face(condselect="fq > 0.5")
    try:
        ms.meshing_close_holes(maxholesize=int(max(current[i].edges for i in todo)) + 1, selected=True,
                               newfaceselected=False, selfintersection=guarded, refinehole=False)
    except Exception:  # MeshLab refuses non-manifold input; the caller repairs and retries
        return vertices, faces
    return mlab.arrays(ms)


def fill_holes(vertices: np.ndarray, faces: np.ndarray, max_span_mm: float | None, fill_all: bool,
               budget_s: float | None = None) -> tuple[np.ndarray, np.ndarray, HoleFill]:
    """Close holes whose rim spans at most ``max_span_mm`` (all holes if ``fill_all``).

    MeshLab's hole closing runs first with its self-intersection guard. It never
    creates crossings, but on a large opening it can stop part-way and leave a
    polygon of long edges. That remainder is closed three ways (MeshLab without
    the guard, a centroid fan, and a planar cap on the remainder's best-fit
    plane) and the version whose triangles cross the fewest others is kept. Each
    crossing count runs under a time budget, so this step cannot hang.
    """
    result = HoleFill()
    topo = an.edge_topology(faces, len(vertices))
    bh, labels, n = an.boundary_loops(vertices, faces, topo)
    if n == 0:
        return vertices, faces, result
    holes = an.describe_holes(vertices, topo, bh, labels, int(n))
    chosen = [i for i, h in enumerate(holes) if fill_all or (max_span_mm is not None and h.span_mm <= max_span_mm)]
    result.candidates = [holes[i] for i in chosen]
    result.skipped = [h for i, h in enumerate(holes) if i not in set(chosen)]
    if not chosen:
        return vertices, faces, result

    def wanted(h: an.Hole) -> bool:
        return fill_all or h.span_mm <= (max_span_mm or 0)

    budget = an.SELF_INTERSECTION_BUDGET_S if budget_s is None else budget_s
    v, f = _meshlab_close(vertices, faces, wanted, guarded=True)
    result.leftover_holes = _open_wanted(v, f, wanted)
    if result.leftover_holes:
        options = []
        attempts = (
            ("MeshLab without its guard", lambda: _meshlab_close(v, f, wanted, guarded=False)),
            ("a centroid fan", lambda: _fan_fill(v, f, wanted)[:2]),
            ("a planar cap", lambda: _planar_caps(v, f, wanted, min_edges=3)[:2]),
        )
        for rank, (name, attempt) in enumerate(attempts):
            cv, cf = attempt()
            if len(cf) == len(f) or _open_wanted(cv, cf, wanted):
                continue
            crossings = an.self_intersecting_faces(cv, cf, budget)
            options.append((np.inf if crossings is None else crossings, rank, name, cv, cf))
        if options:
            crossings, _, name, v, f = min(options, key=lambda o: (o[0], o[1]))
            result.leftover_method = name
            result.leftover_crossings = None if crossings == np.inf else int(crossings)

    v, f, result.fan_filled = _fan_fill(v, f, wanted)
    result.unfilled_candidates = _open_wanted(v, f, wanted)
    result.faces_added = int(len(f) - len(faces))
    return v, f, result


def _fan_fill(vertices: np.ndarray, faces: np.ndarray, wanted) -> tuple[np.ndarray, np.ndarray, int]:
    """Last resort for holes MeshLab would not close: a fan of triangles around the
    rim's centroid. Only used on simple loops (each rim vertex used once)."""
    topo = an.edge_topology(faces, len(vertices))
    bh, labels, n = an.boundary_loops(vertices, faces, topo)
    if n == 0:
        return vertices, faces, 0
    holes = an.describe_holes(vertices, topo, bh, labels, int(n))
    a, b = topo.he_from[bh], topo.he_to[bh]
    order = np.argsort(labels, kind="stable")
    splits = np.r_[np.flatnonzero(np.r_[True, labels[order][1:] != labels[order][:-1]]), len(order)]
    new_v, new_f = [], []
    for k in range(len(splits) - 1):
        sel = order[splits[k]:splits[k + 1]]
        loop = int(labels[sel[0]])
        if not wanted(holes[loop]):
            continue
        ai, bi = a[sel], b[sel]
        if len(np.unique(ai)) != len(ai) or len(np.unique(bi)) != len(bi) or set(ai.tolist()) != set(bi.tolist()):
            continue
        centre = len(vertices) + len(new_v)
        new_v.append(vertices[ai].mean(axis=0))
        new_f.append(np.stack([bi, ai, np.full(len(ai), centre, dtype=np.int64)], axis=1))
    if not new_v:
        return vertices, faces, 0
    return np.vstack([vertices, np.array(new_v)]), np.vstack([faces] + new_f), len(new_v)


def decimate(vertices: np.ndarray, faces: np.ndarray, target_faces: int) -> tuple[np.ndarray, np.ndarray]:
    ms = mlab.meshset(vertices, faces)
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces), qualitythr=0.3, preserveboundary=True, boundaryweight=1.0,
        preservenormal=True, preservetopology=True, optimalplacement=True, planarquadric=True,
        autoclean=True,
    )
    return mlab.arrays(ms)


def place_on_bed(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, float]:
    z_min = float(vertices[np.unique(faces), 2].min())
    out = vertices.copy()
    out[:, 2] -= z_min
    return out, -z_min
