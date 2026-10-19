"""Runs on real scans when they are present (see scripts/download_scans.py).
Set SCAN_TO_3MF_SCANS to extra folders to search. Million-triangle scans are
marked ``large``; run them with ``pytest -m large``."""

import pytest
from conftest import find_scan

from scan_to_3mf.engine import CleanOptions, run_pipeline

pytestmark = pytest.mark.scans


def run(name, tmp_path, **options):
    path = find_scan(name)
    if path is None:
        pytest.skip(f"{name} not downloaded")
    return run_pipeline(path, tmp_path / "out.3mf", CleanOptions(**options))


def assert_verified(r):
    assert not [c for c in r.verification if c["ok"] is False]
    assert r.printable


def test_stanford_bunny(tmp_path):
    r = run("bun_zipper.ply", tmp_path, unit="m", fill_large_holes=True)
    assert r.before["holes"] == 5 and r.after["holes"] == 0
    assert r.after["extents"] == pytest.approx([155.7, 154.33, 120.67], abs=0.01)
    assert_verified(r)


def test_stanford_bunny_default_leaves_large_holes_open(tmp_path):
    r = run("bun_zipper.ply", tmp_path, unit="m", slicer_check=False)
    assert r.after["holes"] == 5 and not r.printable


@pytest.mark.large
def test_pop3plus_trilobite(tmp_path):
    r = run("pop3plus-trilobite-fossil-revopoint.stl", tmp_path, fill_large_holes=True)
    assert r.after["shells"] == 1 and r.after["holes"] == 0
    assert r.after["extents"] == pytest.approx([82.28, 49.62, 57.78], abs=0.01)
    assert_verified(r)


@pytest.mark.large
def test_pop3plus_giganotosaurus_is_not_damaged(tmp_path):
    r = run("pop3plus-giganotosaurus-julien-mairy.stl", tmp_path)
    assert r.after["holes"] == 0 and r.after["shells"] == 1
    assert r.after["extents"] == pytest.approx(r.before["extents"], abs=0.01)
    assert abs(r.after["area_mm2"] - r.before["area_mm2"]) < 0.001 * r.before["area_mm2"]
    assert r.after["triangles"] >= 0.999 * r.before["triangles"]
    assert_verified(r)


@pytest.mark.large
def test_stanford_happy_buddha(tmp_path):
    r = run("happy_vrip.ply", tmp_path, unit="m")
    assert r.after["watertight"] and r.after["extents"] == pytest.approx(r.before["extents"], abs=0.01)
    assert_verified(r)
