"""Render still views of a cleaned 3MF: front, filled side, side, and top.

Surface that is not in the original scan (hole fills and repairs) is drawn in
amber, so the invented parts are visible. Uses headless Chrome and the
three.js copy that ships with the app.

    uv run --group e2e python scripts/render_views.py OUT.3mf --original SCAN.obj \
        --unit cm --title "Thermostat shell" --credit "Revopoint 3D, CC BY-NC-ND 4.0" \
        --prefix media/thermostat-angle
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import shutil
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from scan_to_3mf.engine import load_mesh, read_3mf, to_mm  # noqa: E402

STATIC = ROOT / "src" / "scan_to_3mf" / "web" / "static" / "vendor" / "three"

PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#eef2f7;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
#c{position:absolute;inset:0}
.cap{position:absolute;left:28px;top:22px;color:#1b2330}
.cap h1{margin:0;font-size:30px}.cap h2{margin:4px 0 0;font-size:22px;font-weight:500;color:#394556}
.foot{position:absolute;left:28px;bottom:20px;font-size:18px;color:#4a5566}
.legend{position:absolute;right:28px;top:26px;font-size:18px;color:#394556;display:flex;gap:18px}
.legend i{display:inline-block;width:16px;height:16px;border-radius:4px;margin-right:6px;vertical-align:-2px}
</style></head><body><div id="c"></div>
<div class="cap"><h1 id="t"></h1><h2 id="v"></h2></div><div class="legend" id="lg"></div><div class="foot" id="f"></div>
<script type="module">
import * as THREE from './three.module.js';
const meta = await (await fetch('meta.json')).json();
const buf = await (await fetch('mesh.bin')).arrayBuffer();
const pos = new Float32Array(buf, 0, meta.nv * 3);
const idx = new Uint32Array(buf, meta.nv * 12, meta.nf * 3);
const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geom.setIndex(new THREE.BufferAttribute(idx, 1));
geom.computeVertexNormals();
geom.addGroup(0, (meta.nf - meta.nfill) * 3, 0);
if (meta.nfill) geom.addGroup((meta.nf - meta.nfill) * 3, meta.nfill * 3, 1);
const mats = [new THREE.MeshStandardMaterial({color: 0xb7c3d3, roughness: 0.62, metalness: 0.04}),
              new THREE.MeshStandardMaterial({color: 0xf0a23a, roughness: 0.6, metalness: 0.0})];
const mesh = new THREE.Mesh(geom, mats);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeef2f7);
scene.add(mesh);
scene.add(new THREE.HemisphereLight(0xffffff, 0x6b7686, 1.25));
const key = new THREE.DirectionalLight(0xffffff, 1.9); scene.add(key); scene.add(key.target);
const fill = new THREE.DirectionalLight(0xffffff, 0.55); fill.position.set(0, 0, 1); scene.add(fill);
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(1);
renderer.setSize(window.innerWidth, window.innerHeight);
document.getElementById('c').appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(28, window.innerWidth / window.innerHeight, 0.01, 1e7);
document.getElementById('t').textContent = meta.title;
document.getElementById('f').textContent = meta.footer;
if (meta.nfill) document.getElementById('lg').innerHTML = '<span><i style="background:#b7c3d3"></i>scanned surface</span><span><i style="background:#f0a23a"></i>added by cleaning (fills and repairs)</span>';
geom.computeBoundingSphere();
const r = geom.boundingSphere.radius, c = geom.boundingSphere.center;
window.setView = (dir, up, label) => {
  const d = new THREE.Vector3(...dir).normalize();
  const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 1.12;
  camera.position.copy(c).addScaledVector(d, dist);
  camera.up.set(...up);
  camera.near = dist / 100; camera.far = dist * 10;
  camera.lookAt(c);
  camera.updateProjectionMatrix();
  key.position.copy(camera.position).add(new THREE.Vector3(...up).multiplyScalar(dist * 0.35));
  key.target.position.copy(c);
  document.getElementById('v').textContent = label;
  renderer.render(scene, camera);
  return new Promise((res) => requestAnimationFrame(() => requestAnimationFrame(() => res(true))));
};
window.ready = true;
</script></body></html>"""


def added_faces(out_v: np.ndarray, out_f: np.ndarray, in_v: np.ndarray, in_f: np.ndarray, tol: float = 1e-3) -> np.ndarray:
    """Output triangles whose three corners do not form an input triangle.

    Output vertices are matched to the nearest input vertex within ``tol`` mm
    (the 3MF stores float32, so positions differ by rounding); input vertices
    that share a position are merged first."""
    from scipy.spatial import cKDTree

    from scan_to_3mf.engine.repair import unique_rows

    first, inverse = unique_rows(in_v)
    canon_v, canon_f = in_v[first], inverse[in_f]
    dist, nearest = cKDTree(canon_v).query(out_v, distance_upper_bound=tol)
    out_ids = np.where(np.isfinite(dist), nearest, -1 - np.arange(len(out_v)))
    key = lambda f: np.ascontiguousarray(np.sort(f, axis=1)).view([("", np.int64)] * 3).ravel()  # noqa: E731
    return ~np.isin(key(out_ids[out_f].astype(np.int64)), key(canon_f.astype(np.int64)))


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n else v


def choose_views(vertices: np.ndarray, faces: np.ndarray, added: np.ndarray, note: str = ""
                 ) -> list[tuple[str, list, list, str]]:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # length = 2 x area
    area = np.linalg.norm(normals, axis=1)
    fill_dir = None
    if added.any() and area[added].sum() >= 0.0005 * area.sum():
        # look from the direction that shows the most added surface (back faces don't count)
        candidates = [unit(np.array([x, y, z], dtype=float)) for x in (-1, 0, 1) for y in (-1, 0, 1)
                      for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
        scores = [float(np.maximum(normals[added] @ d, 0).sum()) for d in candidates]
        fill_dir = candidates[int(np.argmax(scores))]
    big_fill = added.any() and area[added].sum() > 0.05 * area.sum()
    front = -unit(normals[added].sum(axis=0)) if big_fill else np.array([0.0, -1.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    if abs(front @ up) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    side = unit(np.cross(up, front))
    tilt = 0.28
    views = [("front", unit(front + 0.4 * up + 0.45 * side), up, "Front (scanned side)")]
    if fill_dir is not None:
        views.append(("filled", unit(fill_dir + 0.12 * up), up,
                      "Filled side: amber is surface the cleaner added" + (f" ({note})" if note else "")))
    else:
        views.append(("filled", unit(-front + tilt * up - 0.18 * side), up,
                      f"Back: nothing needed filling (the cleaner changed {int(added.sum()):,} triangles)"))
    views.append(("side", unit(side + tilt * up), up, "Side"))
    top_up = unit(front - (front @ np.array([0, 0, 1.0])) * np.array([0, 0, 1.0])) if abs(front[2]) < 0.99 else np.array([0, 1.0, 0])
    views.append(("top", np.array([0.0, 0.0, 1.0]), -top_up, "Top"))
    return [(name, list(map(float, d)), list(map(float, u)), label) for name, d, u, label in views]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cleaned")
    ap.add_argument("--original", required=True)
    ap.add_argument("--unit", default="mm")
    ap.add_argument("--title", required=True)
    ap.add_argument("--credit", default="")
    ap.add_argument("--prefix", required=True, help="output path prefix; writes <prefix>-front.png etc.")
    ap.add_argument("--size", default="1600x1200")
    ap.add_argument("--note", default="", help="extra caption for the filled-side view")
    args = ap.parse_args()

    out = read_3mf(args.cleaned)
    v, f = out.vertices, out.faces
    src = load_mesh(args.original)
    added = added_faces(v, f, src.vertices * to_mm(args.unit), src.faces)
    order = np.r_[np.flatnonzero(~added), np.flatnonzero(added)]
    f_sorted = f[order]
    ext = v[np.unique(f)].max(axis=0) - v[np.unique(f)].min(axis=0)
    footer = (f"{' × '.join(f'{x:.1f}' for x in ext)} mm · {len(f):,} triangles · "
              f"{int(added.sum()):,} added by cleaning" + (f" · scan: {args.credit}" if args.credit else ""))
    views = choose_views(v, f, added, args.note)

    tmp = Path(tempfile.mkdtemp(prefix="render-views-"))
    for name in ("three.module.js", "three.core.js"):
        shutil.copy(STATIC / name, tmp / name)
    (tmp / "index.html").write_text(PAGE)
    (tmp / "meta.json").write_text(json.dumps({"nv": len(v), "nf": len(f), "nfill": int(added.sum()),
                                               "title": args.title, "footer": footer}))
    (tmp / "mesh.bin").write_bytes(np.ascontiguousarray(v, "<f4").tobytes() + np.ascontiguousarray(f_sorted, "<u4").tobytes())

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    width, height = (int(x) for x in args.size.split("x"))

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=shutil.which("google-chrome") or None, headless=True,
                                    args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                          "--ignore-gpu-blocklist"])
        page = browser.new_page(viewport={"width": width, "height": height})
        page.on("console", lambda m: print("console:", m.text) if m.type == "error" else None)
        page.goto(f"http://127.0.0.1:{server.server_address[1]}/index.html")
        page.wait_for_function("window.ready === true", timeout=300_000)
        for name, direction, up, label in views:
            page.evaluate("([d, u, l]) => window.setView(d, u, l)", [direction, up, label])
            path = f"{args.prefix}-{name}.png"
            page.screenshot(path=path)
            print("wrote", path)
        browser.close()
    server.shutdown()
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"added triangles: {int(added.sum()):,} of {len(f):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
