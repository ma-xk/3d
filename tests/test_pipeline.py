import numpy as np
import pytest
from conftest import box, remove_faces_around, sphere, write_obj, write_stl

from scan_to_3mf.engine import CleanOptions, read_3mf, run_pipeline
from scan_to_3mf.engine.threemf import read_3mf_metadata, write_3mf_xml


def broken_scan(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A 40 mm sphere with the defects scans have: a small hole, a floating
    speck, a flipped triangle, duplicate triangles, and a degenerate triangle."""
    v, f = sphere(20, 4)
    f = remove_faces_around(v, f, (0, 0, 20), 2.5)
    f = f.copy()
    f[10] = f[10, ::-1]
    sv, sf = box(size=(0.3, 0.3, 0.3), center=(40, 0, 0))
    v = np.vstack([v, sv])
    f = np.vstack([f, sf + len(v) - len(sv), f[20:22], [[0, 0, 1]]])
    return v * scale, f


def open_box():
    v, f = box(size=(60, 40, 30), center=(0, 0, 15))
    side = np.flatnonzero(v[f][:, :, 1].max(axis=1) < -19.9)
    return v, np.delete(f, side, axis=0)


def check_ok(result):
    failed = [c for c in result.verification if c["ok"] is False]
    assert not failed, failed
    assert result.after["watertight"] and result.after["orientation"] == "outward"


def test_broken_scan_is_repaired_and_verified(tmp_path, no_slicer):
    src = write_stl(tmp_path / "scan.stl", *broken_scan())
    out = tmp_path / "scan.3mf"
    r = run_pipeline(src, out, CleanOptions())
    check_ok(r)
    assert r.printable and r.before["holes"] >= 1 and r.after["holes"] == 0
    assert r.before["shells"] == 2 and r.after["shells"] == 1
    assert len(r.removed_shells) == 1 and r.removed_shells[0]["triangles"] == 12
    assert r.after["extents"] == pytest.approx([40.0, 40.0, 40.0], rel=0.01)
    assert r.slicer["ok"] is None and "skipped" in r.slicer["message"]
    assert not any("slice" in issue.lower() for issue in r.remaining)
    content = read_3mf(out)
    assert content.unit == "mm" and len(content.faces) == r.after["triangles"]


def test_inch_file_is_scaled_only_by_the_unit(tmp_path, no_slicer):
    src = write_obj(tmp_path / "scan.obj", *broken_scan(scale=1 / 25.4))
    r = run_pipeline(src, tmp_path / "out.3mf", CleanOptions(unit="in"))
    check_ok(r)
    assert r.units["factor_to_mm"] == pytest.approx(25.4)
    assert r.after["extents"] == pytest.approx([40.0, 40.0, 40.0], rel=0.01)


def test_3mf_input_uses_its_declared_unit(tmp_path, no_slicer):
    v, f = sphere(1.0, 3)  # 2 units across
    src = tmp_path / "in.3mf"
    write_3mf_xml(src, v, f)
    src.write_bytes(src.read_bytes())  # written in mm; rewrite the unit below
    import zipfile

    with zipfile.ZipFile(src) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    parts["3D/3dmodel.model"] = parts["3D/3dmodel.model"].replace(b'unit="millimeter"', b'unit="inch"')
    with zipfile.ZipFile(src, "w") as z:
        for name, data in parts.items():
            z.writestr(name, data)
    r = run_pipeline(src, tmp_path / "out.3mf", CleanOptions(unit="mm"))
    assert r.units["source"] == "file" and r.units["unit"] == "in"
    assert any("Ignored the chosen unit" in n for n in r.notes)
    assert r.after["extents"] == pytest.approx([50.8] * 3, rel=0.01)
    r2 = run_pipeline(src, tmp_path / "out2.3mf", CleanOptions(unit="mm", override_file_unit=True))
    assert r2.after["extents"] == pytest.approx([2.0] * 3, rel=0.01)


def test_large_hole_is_reported_not_filled_by_default(tmp_path, no_slicer):
    src = write_stl(tmp_path / "open.stl", *open_box())
    r = run_pipeline(src, tmp_path / "open.3mf", CleanOptions())
    assert not r.printable and r.after["holes"] == 1
    assert any("hole" in issue for issue in r.remaining)
    assert r.before_holes[0]["status"] == "open"
    r2 = run_pipeline(src, tmp_path / "filled.3mf", CleanOptions(fill_large_holes=True))
    check_ok(r2)
    assert r2.printable and r2.before_holes[0]["status"] == "closed"
    assert r2.after["extents"] == pytest.approx([60.0, 40.0, 30.0], abs=1e-3)


def test_decimation_place_on_bed_and_metadata(tmp_path, no_slicer):
    v, f = sphere(20, 5)  # 20,480 triangles
    src = write_stl(tmp_path / "ball.stl", v, f)
    out = tmp_path / "ball.3mf"
    opts = CleanOptions(decimate_to=2000, place_on_bed=True,
                        metadata={"Designer": "Someone", "LicenseTerms": "CC BY 4.0"})
    r = run_pipeline(src, out, opts)
    check_ok(r)
    assert 1800 <= r.after["triangles"] <= 2200
    assert r.after["bbox_min"][2] == pytest.approx(0.0, abs=1e-6)
    meta = read_3mf_metadata(out)
    assert meta["Designer"] == "Someone" and meta["LicenseTerms"] == "CC BY 4.0" and meta["Title"] == "ball"


@pytest.mark.parametrize("bad", [{"writer": "nope"}, {"decimate_to": 10}, {"min_shell_fraction": 2},
                                 {"metadata": {"Author": "x"}}, {"unit": "furlong"}])
def test_invalid_options_are_rejected(bad):
    with pytest.raises(ValueError):
        CleanOptions.from_dict(bad)


def test_options_from_form_values():
    o = CleanOptions.from_dict({"unit": "cm", "fill_large_holes": "true", "decimate_to": "0", "max_hole_mm": "5"})
    assert o.unit == "cm" and o.fill_large_holes is True and o.decimate_to is None and o.max_hole_mm == 5.0


def test_report_is_json_serialisable(tmp_path, no_slicer):
    import json

    src = write_stl(tmp_path / "scan.stl", *broken_scan())
    r = run_pipeline(src, tmp_path / "scan.3mf", CleanOptions())
    text = json.dumps(r.to_dict())
    assert '"verification"' in text and '"timings"' in text
