import plistlib
import stat

import pytest
from conftest import box

from scan_to_3mf.engine import slicer as sl
from scan_to_3mf.engine.threemf import write_3mf


def fake_app(folder, name, exe, version):
    app = folder / f"{name}.app"
    (app / "Contents" / "MacOS").mkdir(parents=True)
    with open(app / "Contents" / "Info.plist", "wb") as fh:
        plistlib.dump({"CFBundleExecutable": exe, "CFBundleShortVersionString": version}, fh)
    script = app / "Contents" / "MacOS" / exe
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return app


def fake_cli(path, help_text, info_text="size_x = 20\nsize_y = 30\nsize_z = 10\nmanifold = yes\n"):
    path.write_text(f"#!/bin/sh\nif [ \"$1\" = \"--help\" ]; then echo '{help_text}'; exit 0; fi\n"
                    f"if [ \"$1\" = \"--info\" ]; then printf '{info_text}'; exit 0; fi\nexit 3\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.mark.parametrize("raw,clean", [("2.7.2+UNKNOWN", "2.7.2"), ("2.9.6", "2.9.6"),
                                       ("3.0.0-alpha12", "3.0.0-alpha12"), ("01.10.02.76", "01.10.02.76"),
                                       ("", None), (None, None)])
def test_clean_version(raw, clean):
    assert sl.clean_version(raw) == clean


def test_mac_bundles_prefer_prusaslicer(tmp_path):
    fake_app(tmp_path, "OrcaSlicer", "OrcaSlicer", "2.4.2")
    fake_app(tmp_path, "BambuStudio", "BambuStudio", "02.02.01.60")
    fake_app(tmp_path, "PrusaSlicer", "PrusaSlicer", "2.9.6+arm64")
    s = sl.find_slicer(app_dirs=[tmp_path], which=lambda name: None, use_flatpak=False)
    assert s.kind == "prusaslicer" and s.label == "PrusaSlicer 2.9.6" and s.can_slice
    assert s.argv[0].endswith("PrusaSlicer.app/Contents/MacOS/PrusaSlicer")


def test_mac_orca_and_bambu_are_found_when_alone(tmp_path):
    fake_app(tmp_path, "BambuStudio", "BambuStudio", "02.02.01.60")
    s = sl.find_slicer(app_dirs=[tmp_path], which=lambda name: None, use_flatpak=False)
    assert s.kind == "bambustudio" and s.label == "Bambu Studio 02.02.01.60" and not s.can_slice
    fake_app(tmp_path, "OrcaSlicer", "OrcaSlicer", "2.4.2")
    s = sl.find_slicer(app_dirs=[tmp_path], which=lambda name: None, use_flatpak=False)
    assert s.kind == "orcaslicer"


def test_no_slicer_found(tmp_path, monkeypatch):
    monkeypatch.delenv(sl.ENV_VAR, raising=False)
    assert sl.find_slicer(app_dirs=[tmp_path], which=lambda name: None, use_flatpak=False) is None


def test_path_slicer_gets_a_clean_version(tmp_path, monkeypatch):
    monkeypatch.delenv(sl.ENV_VAR, raising=False)
    exe = fake_cli(tmp_path / "prusa-slicer", "PrusaSlicer-2.7.2+UNKNOWN based on Slic3r (with GUI support)")
    s = sl.find_slicer(app_dirs=[], which=lambda name: str(exe) if name == "prusa-slicer" else None, use_flatpak=False)
    assert s.label == "PrusaSlicer 2.7.2"
    monkeypatch.setenv(sl.ENV_VAR, str(exe))
    assert sl.find_slicer(app_dirs=[], which=lambda name: None, use_flatpak=False).label == "PrusaSlicer 2.7.2"


def test_size_check_only_slicer_reports_not_run(tmp_path):
    exe = fake_cli(tmp_path / "orca-slicer", "OrcaSlicer-2.4.2:")
    s = sl.find_slicer(str(exe))
    assert s.kind == "orcaslicer" and s.label == "OrcaSlicer 2.4.2"
    v, f = box()
    model = tmp_path / "box.3mf"
    write_3mf(model, v, f)
    r = sl.slice_check(s, model, [20, 30, 10])
    assert r.ok is None and "not run" in r.message and r.info["size_y"] == 30 and r.info["manifold"] == "yes"


@pytest.mark.slicer
def test_real_slicer_slices_a_box(tmp_path):
    s = sl.find_slicer()
    if s is None or not s.can_slice:
        pytest.skip("PrusaSlicer is not installed")
    v, f = box()
    model = tmp_path / "box.3mf"
    write_3mf(model, v, f)
    r = sl.slice_check(s, model, [20, 30, 10])
    assert r.ok and r.layers == pytest.approx(10 / sl.LAYER_HEIGHT, abs=2)
    assert r.info["size_x"] == pytest.approx(20, abs=0.01) and r.info["manifold"] == "yes"
