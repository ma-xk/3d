"""Mesh diagnostics: the numbers in the before/after report.

All functions take plain numpy arrays (vertices in mm, int faces) and are
vectorized so they stay fast on million-triangle scans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist

EXACT_SPAN_MAX_LOOPS = 5000
EXACT_SPAN_MAX_VERTS = 3000


@dataclass
class Hole:
    edges: int
    perimeter_mm: float
    span_mm: float  # largest distance between two points of the hole's rim
    center: list[float]
    span_exact: bool = True


@dataclass
class Shell:
    triangles: int
    extents: list[float]
    center: list[float]


@dataclass
class Metrics:
    triangles: int
    vertices: int
    shells: int
    holes: int
    boundary_edges: int
    non_manifold_edges: int
    non_manifold_vertices: int
    degenerate_faces: int
    duplicate_faces: int
    inconsistent_edges: int
    unreferenced_vertices: int
    watertight: bool
    orientation: str  # outward | inward | mixed | inconsistent | none
    orientation_estimated: bool
    volume_mm3: float | None
    area_mm2: float
    bbox_min: list[float]
    bbox_max: list[float]
    extents: list[float]
    self_intersecting_faces: int | None = None
    self_intersections_checked: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Analysis:
    metrics: Metrics
    holes: list[Hole] = field(default_factory=list)
    shells: list[Shell] = field(default_factory=list)
    boundary_segments: np.ndarray | None = None  # (k, 2, 3) rim edges, for the preview
    non_manifold_segments: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Building blocks (also used by the repair steps)
# ---------------------------------------------------------------------------


def degenerate_index_mask(faces: np.ndarray) -> np.ndarray:
    f = faces
    return (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])


def face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    if len(faces) == 0:
        return np.zeros(0)
    tri = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)


def zero_area_mask(vertices: np.ndarray, faces: np.ndarray, areas: np.ndarray | None = None,
                   diag: float | None = None) -> np.ndarray:
    """Slivers: triangles whose height over their longest edge is below 1e-7 of
    the model size (collinear or coincident corners). Tiny but well-shaped
    triangles are not flagged."""
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    if areas is None:
        areas = face_areas(vertices, faces)
    if diag is None:
        used = vertices[np.unique(faces)]
        diag = float(np.linalg.norm(used.max(axis=0) - used.min(axis=0))) or 1.0
    tri = vertices[faces]
    longest = np.max(np.stack([
        np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
        np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
        np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1),
    ]), axis=0)
    height = np.divide(2.0 * areas, longest, out=np.zeros_like(areas), where=longest > 0)
    return height <= 1e-7 * diag


def duplicate_face_mask(faces: np.ndarray) -> np.ndarray:
    """True for every face that repeats an earlier face (same three vertices, any order)."""
    if len(faces) == 0:
        return np.zeros(0, dtype=bool)
    s = np.sort(faces, axis=1)
    order = np.lexsort((s[:, 2], s[:, 1], s[:, 0]))
    ss = s[order]
    same_as_prev = np.r_[False, np.all(ss[1:] == ss[:-1], axis=1)]
    mask = np.zeros(len(faces), dtype=bool)
    mask[order[same_as_prev]] = True
    return mask


@dataclass
class EdgeTopology:
    """Half-edge grouping. Half-edge ``h`` belongs to face ``h // 3`` and runs
    from corner ``h % 3`` to the next corner."""

    he_from: np.ndarray
    he_to: np.ndarray
    order: np.ndarray  # half-edges sorted by undirected edge key
    starts: np.ndarray  # start of each undirected edge group in ``order``
    counts: np.ndarray  # faces sharing each undirected edge

    @property
    def boundary_halfedges(self) -> np.ndarray:
        return self.order[self.starts[self.counts == 1]]

    @property
    def non_manifold_edges(self) -> int:
        return int((self.counts > 2).sum())

    def non_manifold_edge_vertices(self) -> np.ndarray:
        h = self.order[self.starts[self.counts > 2]]
        return np.stack([self.he_from[h], self.he_to[h]], axis=1)

    def inconsistent_edges(self) -> int:
        two = self.starts[self.counts == 2]
        h1, h2 = self.order[two], self.order[two + 1]
        return int((self.he_from[h1] == self.he_from[h2]).sum())


def edge_topology(faces: np.ndarray, n_vertices: int) -> EdgeTopology:
    f = np.asarray(faces, dtype=np.int64)
    he_from = f.ravel()
    he_to = f[:, [1, 2, 0]].ravel()
    lo = np.minimum(he_from, he_to)
    hi = np.maximum(he_from, he_to)
    key = lo * np.int64(max(n_vertices, 1)) + hi
    order = np.argsort(key, kind="stable")
    ks = key[order]
    if len(ks):
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        counts = np.diff(np.r_[starts, len(ks)])
    else:
        starts = counts = np.zeros(0, dtype=np.int64)
    return EdgeTopology(he_from, he_to, order, starts, counts)


def non_manifold_vertices(faces: np.ndarray, n_vertices: int, topo: EdgeTopology | None = None) -> np.ndarray:
    """Vertices where the surface pinches: the faces around the vertex form more
    than one fan (for example two cones touching at their tips).

    Face corners are joined across every shared edge; a vertex whose corners fall
    into more than one group is non-manifold. Corners on edges shared by more than
    two faces are joined too, so those are reported as non-manifold *edges* only.
    """
    nf = len(faces)
    if nf == 0:
        return np.zeros(0, dtype=np.int64)
    topo = topo or edge_topology(faces, n_vertices)
    shared = topo.counts >= 2
    # pair each half-edge in a group with the next one in the same group
    first = topo.starts[shared]
    sizes = topo.counts[shared]
    idx = np.repeat(first, sizes - 1) + (np.arange(int((sizes - 1).sum())) - np.repeat(np.cumsum(sizes - 1) - (sizes - 1), sizes - 1))
    h1 = topo.order[idx]
    h2 = topo.order[idx + 1]
    c1_start = h1
    c1_end = (h1 // 3) * 3 + (h1 % 3 + 1) % 3
    c2_start = h2
    c2_end = (h2 // 3) * 3 + (h2 % 3 + 1) % 3
    same_dir = topo.he_from[h1] == topo.he_from[h2]
    a_pair = np.where(same_dir, c2_start, c2_end)  # corner of face 2 at h1's start vertex
    b_pair = np.where(same_dir, c2_end, c2_start)
    rows = np.r_[c1_start, c1_end]
    cols = np.r_[a_pair, b_pair]
    n = 3 * nf
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    corner_vertex = np.asarray(faces, dtype=np.int64).ravel()
    pair_keys = np.unique(corner_vertex * np.int64(n) + labels.astype(np.int64))
    verts, groups = np.unique(pair_keys // np.int64(n), return_counts=True)
    return verts[groups > 1]


def boundary_loops(vertices: np.ndarray, faces: np.ndarray, topo: EdgeTopology | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group rim (boundary) edges into holes.

    Returns (halfedges, loop_label_per_halfedge, n_loops). A hole is a connected
    set of rim edges, so two holes that touch at one vertex count as one.
    """
    topo = topo or edge_topology(faces, len(vertices))
    bh = topo.boundary_halfedges
    if len(bh) == 0:
        return bh, np.zeros(0, dtype=np.int64), np.int64(0)
    a, b = topo.he_from[bh], topo.he_to[bh]
    verts, inv = np.unique(np.r_[a, b], return_inverse=True)
    ia, ib = inv[: len(a)], inv[len(a):]
    g = coo_matrix((np.ones(len(ia), dtype=np.int8), (ia, ib)), shape=(len(verts), len(verts)))
    n_loops, vlabels = connected_components(g, directed=False)
    return bh, vlabels[ia].astype(np.int64), np.int64(n_loops)


def describe_holes(vertices: np.ndarray, topo: EdgeTopology, bh: np.ndarray, labels: np.ndarray, n_loops: int
                   ) -> list[Hole]:
    if n_loops == 0:
        return []
    a, b = topo.he_from[bh], topo.he_to[bh]
    lengths = np.linalg.norm(vertices[a] - vertices[b], axis=1)
    perimeter = np.bincount(labels, weights=lengths, minlength=n_loops)
    edges = np.bincount(labels, minlength=n_loops)
    order = np.argsort(labels, kind="stable")
    lab_sorted = labels[order]
    splits = np.flatnonzero(np.r_[True, lab_sorted[1:] != lab_sorted[:-1]])
    exact = n_loops <= EXACT_SPAN_MAX_LOOPS
    holes = []
    for i, start in enumerate(splits):
        end = splits[i + 1] if i + 1 < len(splits) else len(order)
        loop = int(lab_sorted[start])
        vids = np.unique(a[order[start:end]])
        pts = vertices[vids]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        if exact and len(pts) <= EXACT_SPAN_MAX_VERTS:
            span = float(pdist(pts).max()) if len(pts) > 1 else 0.0
            is_exact = True
        else:
            span = float(np.linalg.norm(hi - lo))
            is_exact = False
        holes.append(Hole(
            edges=int(edges[loop]), perimeter_mm=float(perimeter[loop]), span_mm=span,
            center=((lo + hi) / 2).round(3).tolist(), span_exact=is_exact,
        ))
    return holes


def face_components(faces: np.ndarray, n_vertices: int) -> tuple[int, np.ndarray]:
    """Connected pieces ("shells"). Faces are connected if they share a vertex.
    Returns (n_shells, shell label per face) with labels 0..n_shells-1."""
    if len(faces) == 0:
        return 0, np.zeros(0, dtype=np.int64)
    f = np.asarray(faces, dtype=np.int64)
    rows = np.r_[f[:, 0], f[:, 1]]
    cols = np.r_[f[:, 1], f[:, 2]]
    g = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_vertices, n_vertices))
    _, vlabels = connected_components(g, directed=False)
    flabels = vlabels[f[:, 0]]
    uniq, compact = np.unique(flabels, return_inverse=True)
    return len(uniq), compact.astype(np.int64)


def describe_shells(vertices: np.ndarray, faces: np.ndarray, flabels: np.ndarray, n_shells: int) -> list[Shell]:
    counts = np.bincount(flabels, minlength=n_shells)
    corner_labels = np.repeat(flabels, 3)
    pts = vertices[np.asarray(faces).ravel()]
    lo = np.full((n_shells, 3), np.inf)
    hi = np.full((n_shells, 3), -np.inf)
    for k in range(3):
        np.minimum.at(lo[:, k], corner_labels, pts[:, k])
        np.maximum.at(hi[:, k], corner_labels, pts[:, k])
    return [
        Shell(triangles=int(counts[i]), extents=(hi[i] - lo[i]).round(4).tolist(),
              center=((hi[i] + lo[i]) / 2).round(3).tolist())
        for i in range(n_shells)
    ]


def shell_signed_volumes(vertices: np.ndarray, faces: np.ndarray, flabels: np.ndarray, n_shells: int) -> np.ndarray:
    """Signed volume per shell, measured from each shell's own centroid so the
    sign is meaningful even for shells that still have small holes."""
    if len(faces) == 0:
        return np.zeros(0)
    tri = vertices[faces]
    centroid_sum = np.zeros((n_shells, 3))
    np.add.at(centroid_sum, flabels, tri.mean(axis=1))
    centroid = centroid_sum / np.maximum(np.bincount(flabels, minlength=n_shells), 1)[:, None]
    rel = tri - centroid[flabels][:, None, :]
    vol = np.einsum("ij,ij->i", rel[:, 0], np.cross(rel[:, 1], rel[:, 2])) / 6.0
    return np.bincount(flabels, weights=vol, minlength=n_shells)


SELF_INTERSECTION_BUDGET_S = 45.0
_INLINE_MAX_TRIANGLES = 150_000


def self_intersecting_faces(vertices: np.ndarray, faces: np.ndarray,
                            budget_s: float = SELF_INTERSECTION_BUDGET_S) -> int | None:
    """Count triangles that cross other triangles, or None if the count did not
    finish within ``budget_s`` seconds.

    MeshLab's grid-based test is fast on typical scans but can take very long on
    long, thin triangles. Small, evenly sized meshes are checked in-process;
    everything else runs in a child process that is killed at the budget.
    """
    from . import mlab

    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    if len(f) == 0:
        return 0
    if len(f) <= _INLINE_MAX_TRIANGLES:
        tri = v[f]
        edges = np.linalg.norm(tri - tri[:, [1, 2, 0]], axis=2)
        median = float(np.median(edges)) or 1.0
        if float(edges.max()) <= 40.0 * median:
            return mlab.count_self_intersections(v, f)
    from .budget import run_with_budget

    status, value = run_with_budget(mlab.count_self_intersections, (v, f), budget_s)
    return int(value) if status == "ok" else None


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def analyze(vertices: np.ndarray, faces: np.ndarray, *, details: bool = True,
            self_intersections: bool = False, budget_s: float = SELF_INTERSECTION_BUDGET_S) -> Analysis:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    nv, nf = len(v), len(f)
    if nf == 0:
        zero = [0.0, 0.0, 0.0]
        return Analysis(Metrics(0, nv, 0, 0, 0, 0, 0, 0, 0, 0, nv, False, "none", False,
                                None, 0.0, zero, zero, zero))

    deg_idx = degenerate_index_mask(f)
    areas = face_areas(v, f)
    deg_area = zero_area_mask(v, f, areas) & ~deg_idx
    good = f[~deg_idx]

    topo = edge_topology(good, nv)
    bh, loop_labels, n_loops = boundary_loops(v, good, topo)
    nm_edges = topo.non_manifold_edges
    nm_verts = non_manifold_vertices(good, nv, topo)
    inconsistent = topo.inconsistent_edges()
    watertight = len(bh) == 0 and nm_edges == 0 and len(good) > 0

    n_shells, flabels = face_components(good, nv)
    signed = shell_signed_volumes(v, good, flabels, n_shells)
    if inconsistent:
        orientation = "inconsistent"
    elif np.all(signed > 0):
        orientation = "outward"
    elif np.all(signed < 0):
        orientation = "inward"
    else:
        orientation = "mixed"

    volume = None
    if watertight and inconsistent == 0:
        tri = v[good]
        volume = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)

    used = np.unique(f)
    pts = v[used]
    lo, hi = pts.min(axis=0), pts.max(axis=0)

    metrics = Metrics(
        triangles=nf,
        vertices=int(len(used)),
        shells=int(n_shells),
        holes=int(n_loops),
        boundary_edges=int(len(bh)),
        non_manifold_edges=int(nm_edges),
        non_manifold_vertices=int(len(nm_verts)),
        degenerate_faces=int(deg_idx.sum() + deg_area.sum()),
        duplicate_faces=int(duplicate_face_mask(f).sum()),
        inconsistent_edges=int(inconsistent),
        unreferenced_vertices=int(nv - len(used)),
        watertight=bool(watertight),
        orientation=orientation,
        orientation_estimated=not watertight,
        volume_mm3=volume,
        area_mm2=float(areas.sum()),
        bbox_min=lo.tolist(),
        bbox_max=hi.tolist(),
        extents=(hi - lo).tolist(),
    )
    if self_intersections:
        metrics.self_intersecting_faces = self_intersecting_faces(v, f, budget_s)
        metrics.self_intersections_checked = metrics.self_intersecting_faces is not None

    result = Analysis(metrics)
    if details:
        result.holes = describe_holes(v, topo, bh, loop_labels, int(n_loops))
        result.shells = describe_shells(v, good, flabels, n_shells)
        result.boundary_segments = np.stack([v[topo.he_from[bh]], v[topo.he_to[bh]]], axis=1)
        nmv = topo.non_manifold_edge_vertices()
        result.non_manifold_segments = np.stack([v[nmv[:, 0]], v[nmv[:, 1]]], axis=1) if len(nmv) else np.zeros((0, 2, 3))
    return result
