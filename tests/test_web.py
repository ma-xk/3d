import time

import pytest
from conftest import write_stl
from fastapi.testclient import TestClient
from test_pipeline import broken_scan

from scan_to_3mf.engine import decode_preview, read_3mf
from scan_to_3mf.web.server import create_app

H = {"X-Scan-To-3MF": "1"}


@pytest.fixture
def client(tmp_path, no_slicer):
    app = create_app(work_dir=tmp_path / "jobs", executor="thread")
    with TestClient(app, base_url="http://localhost") as c:
        yield c


def wait(client, job_id, states, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["state"] in states:
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_page_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200 and "scan-to-3mf" in r.text
    assert "default-src 'self'" in r.headers["content-security-policy"]
    info = client.get("/api/info").json()
    assert "mm" in info["units"] and info["defaults"]["max_hole_mm"] == 10.0


def test_requests_from_other_sites_are_refused(client, tmp_path):
    assert client.get("/api/info", headers={"host": "evil.example"}).status_code == 400
    assert client.post("/api/upload?name=a.stl", content=b"x").status_code == 403
    assert client.post("/api/upload?name=a.step", content=b"x", headers=H).status_code == 400


def test_upload_clean_preview_and_download(client, tmp_path):
    src = write_stl(tmp_path / "scan.stl", *broken_scan())
    job = client.post("/api/upload?name=my%20scan.stl", content=src.read_bytes(), headers=H).json()
    job = wait(client, job["id"], {"ready", "error"})
    assert job["state"] == "ready" and job["inspect"]["metrics"]["holes"] >= 1
    before = client.get(f"/api/jobs/{job['id']}/preview/before")
    assert before.status_code == 200 and len(decode_preview(before.content)[1]) > 0

    r = client.post(f"/api/jobs/{job['id']}/clean", json={"unit": "mm", "fill_large_holes": False}, headers=H)
    assert r.status_code == 200
    job = wait(client, job["id"], {"done", "error"})
    assert job["state"] == "done", job["error"]
    assert job["result"]["printable"] and job["result"]["input"]["path"] == "my scan.stl"

    after = client.get(f"/api/jobs/{job['id']}/preview/after")
    assert after.status_code == 200 and len(decode_preview(after.content)[2]) == 0  # no open edges left
    dl = client.get(f"/api/jobs/{job['id']}/download")
    assert dl.status_code == 200 and 'filename="my_scan-clean.3mf"' in dl.headers["content-disposition"]
    out = tmp_path / "dl.3mf"
    out.write_bytes(dl.content)
    assert read_3mf(out).unit == "mm"
    assert client.get(f"/api/jobs/{job['id']}/report").json()["after"]["watertight"]
    assert client.delete(f"/api/jobs/{job['id']}", headers=H).status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404


def test_bad_file_reports_an_error(client):
    job = client.post("/api/upload?name=bad.stl", content=b"not an stl file at all", headers=H).json()
    job = wait(client, job["id"], {"ready", "error"})
    assert job["state"] == "error" and job["error"]
