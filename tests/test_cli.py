import json

from conftest import box, write_stl
from test_pipeline import broken_scan, open_box

from scan_to_3mf.cli import main


def test_clean_json_and_exit_code(tmp_path, capsys):
    src = write_stl(tmp_path / "scan.stl", *broken_scan())
    out = tmp_path / "out.3mf"
    code = main(["clean", str(src), "-o", str(out), "--no-slice", "--json", "--meta", "Designer=Me"])
    assert code == 0 and out.exists()
    report = json.loads(capsys.readouterr().out)
    assert report["printable"] and report["after"]["watertight"]


def test_bare_path_means_clean_and_default_output_name(tmp_path, capsys):
    src = write_stl(tmp_path / "part.stl", *box())
    assert main([str(src), "--no-slice", "-q"]) == 0
    assert (tmp_path / "part-clean.3mf").exists()
    assert "Verdict:" in capsys.readouterr().out


def test_not_printable_exit_code_and_report_file(tmp_path):
    src = write_stl(tmp_path / "open.stl", *open_box())
    report = tmp_path / "r.json"
    assert main(["clean", str(src), "--no-slice", "-q", "--report", str(report)]) == 2
    assert json.loads(report.read_text())["after"]["holes"] == 1


def test_errors_exit_1(tmp_path, capsys):
    assert main(["clean", str(tmp_path / "missing.stl"), "--no-slice"]) == 1
    assert main(["clean", str(write_stl(tmp_path / "a.stl", *box())), "--meta", "nonsense", "--no-slice"]) == 1
    assert main(["slicer", "--slicer", str(tmp_path / "no-such-slicer")]) == 1
    assert "error" in capsys.readouterr().err


def test_inspect(tmp_path, capsys):
    src = write_stl(tmp_path / "scan.stl", *broken_scan())
    assert main(["inspect", str(src), "--json", "--no-self-intersections"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["format"] == "stl" and info["metrics"]["holes"] >= 1
