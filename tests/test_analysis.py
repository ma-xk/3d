import numpy as np
import pytest
from conftest import box, remove_faces_around, sphere

from scan_to_3mf.engine import analysis as an


def test_closed_box():
    v, f = box()
    m = an.analyze(v, f, self_intersections=True).metrics
    assert m.watertight and m.holes == 0 and m.boundary_edges == 0 and m.shells == 1
    assert m.orientation == "outward" and not m.orientation_estimated
    assert m.volume_mm3 == pytest.approx(6000.0)
    assert m.extents == pytest.approx([20.0, 30.0, 10.0])
    assert m.self_intersecting_faces == 0 and m.self_intersections_checked


def test_open_box_has_one_hole_with_measured_span():
    v, f = box()
    top = np.flatnonzero(v[f][:, :, 2].min(axis=1) > 9.9)  # the two triangles of the top face
    a = an.analyze(v, np.delete(f, top, axis=0))
    assert a.metrics.holes == 1 and a.metrics.boundary_edges == 4 and not a.metrics.watertight
    assert a.holes[0].edges == 4
    assert a.holes[0].span_mm == pytest.approx(np.hypot(20, 30))


def test_inverted_and_inconsistent_winding():
    v, f = box()
    assert an.analyze(v, f[:, ::-1]).metrics.orientation == "inward"
    g = f.copy()
    g[0] = g[0, ::-1]
    m = an.analyze(v, g).metrics
    assert m.inconsistent_edges == 3 and m.orientation == "inconsistent"


def test_non_manifold_edge_from_a_fin():
    v, f = box()
    a, b = f[0, 0], f[0, 1]
    v2 = np.vstack([v, [[50.0, 50.0, 50.0]]])
    f2 = np.vstack([f, [[a, b, len(v)]]])
    m = an.analyze(v2, f2).metrics
    assert m.non_manifold_edges == 1 and not m.watertight


def test_pinched_vertex_between_two_cones():
    v1, f1 = box(center=(0, 0, 5))
    v2, f2 = box(center=(20, 30, 15))  # touches the first box at one corner (10, 15, 10)
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    from scan_to_3mf.engine.repair import merge_identical

    v, f, merged = merge_identical(v, f)
    assert merged == 1
    m = an.analyze(v, f).metrics
    assert m.non_manifold_vertices == 1 and m.non_manifold_edges == 0 and m.shells == 1


def test_shells_duplicates_and_degenerate_faces():
    v1, f1 = box()
    v2, f2 = box(center=(100, 0, 5))
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1), f1[:1], [[0, 0, 1]]])
    m = an.analyze(v, f).metrics
    assert m.shells == 2 and m.duplicate_faces == 1 and m.degenerate_faces == 1


def test_zero_area_sliver_is_degenerate():
    v = np.array([[0, 0, 0], [2, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    assert an.zero_area_mask(v, np.array([[0, 1, 2], [0, 1, 3]])).tolist() == [True, False]


def test_self_intersections_and_time_budget(monkeypatch):
    v1, f1 = sphere(10)
    v2, f2 = sphere(10, center=(5, 0, 0))
    v = np.vstack([v1, v2])
    f = np.vstack([f1, f2 + len(v1)])
    count = an.self_intersecting_faces(v, f)
    assert count and count > 0
    monkeypatch.setattr(an, "_INLINE_MAX_TRIANGLES", 0)  # force the child-process path
    assert an.self_intersecting_faces(v, f, budget_s=60) == count
    assert an.self_intersecting_faces(v, f, budget_s=0.001) is None  # budget exceeded: reported as unknown


def test_hole_in_sphere_and_rim_segments():
    v, f = sphere(20)
    g = remove_faces_around(v, f, (0, 0, 20), 3.0)
    a = an.analyze(v, g)
    assert a.metrics.holes == 1
    assert len(a.boundary_segments) == a.metrics.boundary_edges
    assert 3.0 < a.holes[0].span_mm < 12.0
