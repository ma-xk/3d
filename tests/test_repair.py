import numpy as np
import pytest
from conftest import box, remove_faces_around, sphere

from scan_to_3mf.engine import analysis as an
from scan_to_3mf.engine import repair


def metrics(v, f):
    return an.analyze(v, f).metrics


def split_edge(v, f, at_first_vertex: bool):
    """Split the edge shared by faces 0 and its neighbour. With ``at_first_vertex``
    the new vertex sits exactly on a corner (a zero-length needle edge);
    otherwise at the midpoint, closed with a zero-area cap triangle."""
    a, b, c = f[0]
    other = [i for i in range(1, len(f)) if {a, b} <= set(f[i])][0]
    d = [x for x in f[other] if x not in (a, b)][0]
    m = len(v)
    v2 = np.vstack([v, v[a] if at_first_vertex else (v[a] + v[b]) / 2])
    keep = [i for i in range(len(f)) if i not in (0, other)]
    if at_first_vertex:
        new = [[a, m, c], [m, b, c], [b, m, d], [m, a, d]]
    else:
        new = [[a, m, c], [m, b, c], [a, b, m], [b, a, d]]
    return v2, np.vstack([f[keep], new])


def test_needle_edge_collapses_without_opening_the_surface():
    v, f = sphere(10, 2)
    v2, f2 = split_edge(v, f, at_first_vertex=True)
    assert metrics(v2, f2).degenerate_faces == 2
    f3, collapsed = repair.collapse_short_edges(v2, f2, 1e-6)
    v3, f3, _ = repair.compact(v2, f3)
    assert collapsed == 1
    m = metrics(v3, f3)
    assert m.watertight and m.degenerate_faces == 0 and len(f3) == len(f)


def test_zero_area_cap_is_removed_by_an_edge_flip():
    v, f = sphere(10, 2)
    v2, f2 = split_edge(v, f, at_first_vertex=False)
    assert metrics(v2, f2).watertight and metrics(v2, f2).degenerate_faces == 1
    f3, stats = repair.fix_zero_area(v2, f2)
    assert stats == {"fixed": 1, "remaining": 0}
    m = metrics(v2, f3)
    assert m.watertight and m.degenerate_faces == 0


def test_crack_is_welded_but_touching_sheets_are_not():
    v, f = box()
    # duplicate the vertices of the top face: the top becomes a separate sheet with a crack around it
    top = np.flatnonzero(v[:, 2] > 9.9)
    dup = {int(i): len(v) + k for k, i in enumerate(top)}
    v2 = np.vstack([v, v[top]])
    f2 = f.copy()
    for i in np.flatnonzero(v[f][:, :, 2].min(axis=1) > 9.9):
        f2[i] = [dup.get(int(x), x) for x in f2[i]]
    assert metrics(v2, f2).holes == 2
    f3, welded = repair.weld_boundary(v2, f2, 0.001)
    assert welded == 4
    v3, f3, _ = repair.compact(v2, f3)
    assert metrics(v3, f3).watertight


def test_duplicate_and_degenerate_faces_are_dropped():
    v, f = box()
    f2 = np.vstack([f, f[:3], [[0, 0, 1]]])
    f3, n_deg = repair.drop_degenerate(v, f2)
    f4, n_dup = repair.drop_duplicate_faces(f3)
    assert (n_deg, n_dup) == (1, 3) and len(f4) == 12


def test_small_floating_shell_is_removed_and_reported():
    v1, f1 = sphere(20, 3)
    v2, f2 = box(size=(0.2, 0.2, 0.2), center=(60, 0, 0))
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    f3, removal = repair.remove_small_shells(v, f, 0.01)
    assert len(removal.removed) == 1 and removal.removed[0].triangles == 12
    assert removal.removed[0].extents == pytest.approx([0.2, 0.2, 0.2])
    assert len(f3) == len(f1)


def test_non_manifold_fin_is_repaired():
    v, f = box()
    v2 = np.vstack([v, [[5.0, 5.0, 40.0]]])
    f2 = np.vstack([f, [[f[0, 0], f[0, 1], len(v)]]])
    v3, f3, stats = repair.repair_non_manifold(v2, f2)
    assert stats["faces_removed"] >= 1
    assert metrics(v3, f3).non_manifold_edges == 0


def test_orientation_is_made_consistent_and_outward():
    v, f = sphere(10, 2)
    inverted = f[:, ::-1].copy()
    inverted[5] = inverted[5, ::-1]  # one face disagrees with its neighbours
    f2, stats = repair.orient_outward(v, inverted)
    assert stats["shells_turned_outward"] == 1
    m = metrics(v, f2)
    assert m.orientation == "outward" and m.inconsistent_edges == 0


def test_small_holes_are_filled_and_large_ones_only_on_request():
    v, f = sphere(20, 4)
    f = remove_faces_around(v, f, (0, 0, 20), 2.5)  # about 5 mm across
    f = remove_faces_around(v, f, (20, 0, 0), 9.0)  # about 18 mm across
    assert metrics(v, f).holes == 2
    v2, f2, fill = repair.fill_holes(v, f, 10.0, fill_all=False)
    assert len(fill.candidates) == 1 and len(fill.skipped) == 1 and metrics(v2, f2).holes == 1
    v3, f3, fill = repair.fill_holes(v, f, 10.0, fill_all=True)
    m = metrics(v3, f3)
    assert m.watertight and m.holes == 0 and fill.unfilled_candidates == 0
    assert m.extents == pytest.approx(metrics(v, f).extents, abs=1e-6)


def test_large_open_side_is_filled_when_asked():
    v, f = box(size=(40, 30, 20))
    side = np.flatnonzero(v[f][:, :, 1].max(axis=1) < -14.9)  # the y = -15 face
    f = np.delete(f, side, axis=0)
    v2, f2, fill = repair.fill_holes(v, f, 10.0, fill_all=True)
    m = metrics(v2, f2)
    assert m.watertight and m.orientation in ("outward", "inward")
    assert m.self_intersecting_faces in (None, 0)


def test_rim_loops_split_where_two_holes_touch():
    # two separate squares' rims sharing one vertex: 0-1-2-3 and 0-4-5-6
    he_from = np.array([0, 1, 2, 3, 0, 4, 5, 6])
    he_to = np.array([1, 2, 3, 0, 4, 5, 6, 0])
    loops = repair.trace_rim_loops(he_from, he_to)
    assert sorted(len(loop) for loop in loops) == [4, 4]


def test_planar_cap_closes_a_square_opening():
    v, f = box(size=(10, 10, 10))
    top = np.flatnonzero(v[f][:, :, 2].min(axis=1) > 9.9)
    f = np.delete(f, top, axis=0)
    v2, f2, capped = repair._planar_caps(v, f, lambda h: True, min_edges=3)
    assert capped == 1
    m = metrics(v2, f2)
    assert m.watertight and m.orientation == "outward"


def test_decimation_hits_the_target_and_stays_closed():
    v, f = sphere(20, 4)
    v2, f2 = repair.decimate(v, f, 1000)
    assert 900 <= len(f2) <= 1100
    assert metrics(v2, f2).watertight


def test_place_on_bed_moves_only_z():
    v, f = box(center=(3, 4, 50))
    v2, dz = repair.place_on_bed(v, f)
    assert dz == pytest.approx(-45.0)
    assert v2[:, 2].min() == 0 and np.allclose(v2[:, :2], v[:, :2])
