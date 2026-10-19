# scan-to-3mf

Turn a 3D-scan triangle mesh (STL, OBJ, PLY, or 3MF) into a cleaned, verified, printable 3MF in millimeters.
Drop a scan into the local browser app, check its size and what is wrong with it, and download a 3MF.
Everything runs on your computer; files never leave it.

The cleaning engine (`src/scan_to_3mf/engine`) has no web code in it, so the same engine can later sit behind a
hosted website. The browser UI and the command line are thin layers over it.

## Start it (macOS)

You need [uv](https://docs.astral.sh/uv/), which also installs the right Python for you:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once
cd scan-to-3mf
uv run scan-to-3mf                                  # opens http://localhost:8765 in your browser
```

Or double-click **`Start scan-to-3mf.command`** in Finder. It offers to install uv if it is missing, then starts the
app and opens your browser. Close the Terminal window to stop it. The first start downloads about 150 MB of Python
packages; later starts take a few seconds.

If macOS refuses to open the launcher because it came from the internet, right-click it and choose **Open**, or run
`xattr -d com.apple.quarantine "Start scan-to-3mf.command"` once.

For the slice test, install [PrusaSlicer](https://www.prusa3d.com/page/prusaslicer_424/) in `/Applications`.
Without it the app still cleans and verifies the file; it just skips the slice test. OrcaSlicer or Bambu Studio are
used, if PrusaSlicer is missing, to double-check the size and that the mesh is closed.

## Use

1. Drop a scan on the page (or choose a file).
2. Pick the unit the file's numbers are in. STL, OBJ, and PLY have no units; the app shows the resulting size in mm
   and inches and warns when it looks wrong (a 0.16 mm bunny is a meters file). 3MF files declare their unit.
3. Choose cleaning options and click **Clean and build 3MF**.
4. Read the before/after report, look at the before/after 3D preview (red lines are open edges, orange is the inside
   seen through holes), and download the 3MF.

Command line, same pipeline:

```bash
uv run scan-to-3mf clean scan.stl --unit mm                 # writes scan-clean.3mf next to the input
uv run scan-to-3mf clean scan.obj --unit cm --fill-large-holes --place-on-bed --report report.json
uv run scan-to-3mf inspect scan.ply --unit m                # report only, writes nothing
uv run scan-to-3mf slicer                                   # which slicer the slice test would use
```

Exit codes: 0 printable, 2 written but not printable (for example a large hole left open), 1 error.

## What cleaning does

Always:

- Welds duplicate vertices: STL corners are shared, open-edge vertices within 0.001 mm are joined (closes cracks),
  and zero-length edges are collapsed only where that keeps the surface manifold. Touching sheets are never merged.
- Drops degenerate and duplicate triangles; zero-area slivers are removed by an edge flip so no hole opens.
- Removes tiny floating shells (under 1% of the main shell's triangles by default) and lists each one.
- Repairs non-manifold edges (removes the smallest triangles on them) and splits pinched vertices.
- Makes normals consistent and turns inside-out shells outward.
- Fills holes up to 10 mm across by default. Larger holes are reported, not filled, unless you turn on
  **fill larger holes**: that invents surface where the scanner saw nothing.
- Never changes the scale except through the unit.

Optional: decimate to a target triangle count, and place the model on the bed (lowest point at z = 0).

Every output is verified: read back with lib3mf and with an independent parser (unit must be millimeter, triangles
and coordinates must match), size compared with the input, closed-solid check with manifold3d, and a headless slice
with the PrusaSlicer command line (size, closed mesh, and layer count). The report says plainly what changed and what
is still wrong.

## Limits

- Hole filling invents geometry. Large fills are flat-ish caps, not a reconstruction of the missing surface.
- Self-intersections are counted, not repaired. Big fills can add some; slicers usually merge them.
- Not checked: wall thickness, overhangs and supports, fit on your printer's bed, accuracy against the real object.
- One object per 3MF, geometry only (no color, textures, or print settings). No automatic orientation.
- Point clouds and scanner project files are not supported; export a mesh from the scanner software first.

## Development

```bash
uv run pytest                  # fast suite; real-scan tests run when the scans are present
uv run pytest -m large         # million-triangle scans (slow)
uv run ruff check src tests scripts
```

Third-party scans are never committed. Put them in `data/scans/` (ignored by git) or point `SCAN_TO_3MF_SCANS` at
a folder of scans.

three.js (MIT) is vendored in `src/scan_to_3mf/web/static/vendor/three` so the app works offline.
