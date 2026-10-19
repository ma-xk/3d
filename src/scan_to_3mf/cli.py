"""Command line interface: the same pipeline as the browser UI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__

EXIT_OK, EXIT_ERROR, EXIT_NOT_PRINTABLE = 0, 1, 2

ROWS = [
    ("Triangles", "triangles", "int"),
    ("Vertices", "vertices", "int"),
    ("Shells", "shells", "int"),
    ("Holes", "holes", "int"),
    ("Boundary (open) edges", "boundary_edges", "int"),
    ("Non-manifold edges", "non_manifold_edges", "int"),
    ("Non-manifold vertices", "non_manifold_vertices", "int"),
    ("Degenerate triangles", "degenerate_faces", "int"),
    ("Duplicate triangles", "duplicate_faces", "int"),
    ("Self-intersecting triangles", "self_intersecting_faces", "int"),
    ("Normals", "orientation", "orientation"),
    ("Watertight", "watertight", "bool"),
    ("Volume", "volume_mm3", "volume"),
    ("Bounding box (mm)", "extents", "bbox"),
]


def _cell(metrics: dict | None, key: str, kind: str) -> str:
    if metrics is None:
        return ""
    value = metrics.get(key)
    if value is None:
        return "—"
    if kind == "int":
        return f"{value:,}"
    if kind == "bool":
        return "yes" if value else "no"
    if kind == "volume":
        return f"{value / 1000:,.2f} cm³"
    if kind == "bbox":
        return " × ".join(f"{x:.2f}" for x in value)
    if kind == "orientation":
        return value + (" (est.)" if metrics.get("orientation_estimated") and value == "outward" else "")
    return str(value)


def _table(before: dict, after: dict | None) -> str:
    width = max(len(r[0]) for r in ROWS) + 2
    cols = [(label, _cell(before, key, kind), _cell(after, key, kind)) for label, key, kind in ROWS]
    w1 = max(len("Before"), *(len(c[1]) for c in cols)) + 3
    lines = [f"{'':{width}}{'Before':<{w1}}{'After' if after else ''}"]
    for label, b, a in cols:
        lines.append(f"{label:<{width}}{b:<{w1}}{a}")
    return "\n".join(lines)


def _mark(ok) -> str:
    return {True: "✓", False: "✗"}.get(ok, "–")


def print_report(d: dict, out=None) -> None:
    out = out or sys.stdout
    inp = d["input"]
    print(f"scan-to-3mf {d.get('version', __version__)} — {Path(inp['path']).name} "
          f"({inp['format'].upper()}, {inp['file_triangles']:,} triangles)", file=out)
    print(file=out)
    print(_table(d["before"], d.get("after")), file=out)
    print("\nWhat changed:", file=out)
    for s in d["steps"]:
        print(f"  • {s['title']}: {s['summary']}", file=out)
    if d["removed_shells"]:
        print("\nRemoved shells (largest first):", file=out)
        for s in d["removed_shells"][:10]:
            print(f"  • {s['triangles']:,} triangles, {' × '.join(f'{x:.2f}' for x in s['extents'])} mm, "
                  f"at ({', '.join(f'{x:.1f}' for x in s['center'])})", file=out)
    print("\nStill wrong:" if d["remaining"] else "\nStill wrong: nothing found.", file=out)
    for issue in d["remaining"]:
        print(f"  • {issue}", file=out)
    if d["verification"]:
        print("\nVerification of the written 3MF:", file=out)
        for c in d["verification"]:
            print(f"  {_mark(c['ok'])} {c['label']} — {c['detail']}", file=out)
    s = d.get("slicer")
    if s:
        extra = f" (est. print time {s['print_time']})" if s.get("print_time") else ""
        print(f"\nSlice test: {_mark(s.get('ok'))} {s.get('slicer') or ''} {s['message']}{extra}".replace("  ", " "),
              file=out)
    print("\nNot checked: " + " ".join(d["not_checked"]), file=out)
    if d.get("notes"):
        print("Notes: " + " ".join(d["notes"]), file=out)
    t = d["timings"]
    mem = f", peak memory {t['peak_memory_mb']:.0f} MB" if "peak_memory_mb" in t else ""
    if d.get("output"):
        o = d["output"]
        print(f"\nWrote {o['path']} ({o['bytes'] / 1e6:.1f} MB) in {t['total']:.1f} s{mem}.", file=out)
    print(f"Verdict: {d['verdict']}", file=out)


def _add_clean_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("input", help="STL, OBJ, PLY, or 3MF scan mesh")
    p.add_argument("-o", "--output", help="output .3mf (default: <input>-clean.3mf next to the input)")
    p.add_argument("--unit", help="unit of STL/OBJ/PLY coordinates: mm (default), cm, m, in, ft, um")
    p.add_argument("--override-file-unit", action="store_true", help="apply --unit even to a 3MF that declares its own")
    p.add_argument("--max-hole-mm", type=float, default=10.0, help="fill holes up to this size across (default 10)")
    p.add_argument("--fill-large-holes", action="store_true", help="fill every hole, whatever its size (invents surface)")
    p.add_argument("--no-fill-holes", action="store_true", help="do not fill any holes")
    p.add_argument("--min-shell-fraction", type=float, default=0.01,
                   help="remove shells with fewer triangles than this fraction of the largest shell (default 0.01)")
    p.add_argument("--keep-all-shells", action="store_true", help="do not remove floating shells")
    p.add_argument("--decimate", type=int, metavar="TRIANGLES", help="reduce to about this many triangles")
    p.add_argument("--place-on-bed", action="store_true", help="move the model so its lowest point is at z = 0")
    p.add_argument("--weld-tolerance-mm", type=float, default=0.001, help="vertex weld distance (default 0.001 mm)")
    p.add_argument("--no-self-intersections", action="store_true", help="skip the self-intersection count (faster)")
    p.add_argument("--no-slice", action="store_true", help="skip the PrusaSlicer/OrcaSlicer slice test")
    p.add_argument("--slicer", help="path to a PrusaSlicer or OrcaSlicer executable")
    p.add_argument("--writer", choices=["auto", "lib3mf", "xml"], default="auto", help=argparse.SUPPRESS)
    p.add_argument("--meta", action="append", default=[], metavar="NAME=VALUE",
                   help="3MF metadata, repeatable. Names: Title, Designer, Description, Copyright, LicenseTerms")
    p.add_argument("--report", help="also write the full report as JSON to this file")
    p.add_argument("--json", action="store_true", help="print the report as JSON instead of text")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress output")


def _options(args):
    from .engine import CleanOptions

    metadata = {}
    for item in args.meta:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--meta expects NAME=VALUE, got {item!r}")
        metadata[key.strip()] = value.strip()
    return CleanOptions(
        metadata=metadata or None,
        unit=args.unit, override_file_unit=args.override_file_unit, weld_tolerance_mm=args.weld_tolerance_mm,
        remove_small_shells=not args.keep_all_shells, min_shell_fraction=args.min_shell_fraction,
        fill_holes=not args.no_fill_holes, max_hole_mm=args.max_hole_mm, fill_large_holes=args.fill_large_holes,
        decimate_to=args.decimate, place_on_bed=args.place_on_bed,
        check_self_intersections=not args.no_self_intersections, slicer_check=not args.no_slice,
        slicer_path=args.slicer, writer=args.writer,
    )


def cmd_clean(args) -> int:
    from .engine import MeshLoadError, run_pipeline

    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_name(src.stem + "-clean.3mf")
    if out.resolve() == src.resolve():
        print("error: output would overwrite the input", file=sys.stderr)
        return EXIT_ERROR

    def progress(message: str, fraction: float) -> None:
        if not args.quiet and not args.json:
            print(f"[{fraction:4.0%}] {message}", file=sys.stderr, flush=True)

    try:
        result = run_pipeline(src, out, _options(args), progress=progress)
    except (MeshLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    d = result.to_dict()
    if args.report:
        Path(args.report).write_text(json.dumps(d, indent=2))
    if args.json:
        print(json.dumps(d, indent=2))
    else:
        print_report(d)
    return EXIT_OK if result.printable else EXIT_NOT_PRINTABLE


def cmd_inspect(args) -> int:
    from .engine import MeshLoadError, inspect_loaded, load_mesh

    try:
        info, _ = inspect_loaded(load_mesh(args.input), args.unit, self_intersections=not args.no_self_intersections)
    except (MeshLoadError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if args.json:
        print(json.dumps(info, indent=2))
        return EXIT_OK
    print(f"{info['name']} ({info['format'].upper()}), unit: {info['unit']} ({info['unit_source']})")
    print(_table(info["metrics"], None))
    if info["holes"]:
        spans = ", ".join(f"{h['span_mm']:.2f}" for h in info["holes"][:10])
        print(f"\nLargest holes (mm across): {spans}")
    for line in info["notes"] + info["unit_hints"]:
        print(f"• {line}")
    return EXIT_OK


def cmd_serve(args) -> int:
    from .web.server import serve

    return serve(host=args.host, port=args.port, open_browser=not args.no_browser)


def cmd_slicer(args) -> int:
    from .engine import find_slicer

    s = find_slicer(args.slicer)
    if s is None:
        print("No PrusaSlicer or OrcaSlicer found. Install PrusaSlicer, or set SCAN_TO_3MF_SLICER to its executable.")
        return EXIT_ERROR
    print(f"{s.label}: {' '.join(s.argv)}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scan-to-3mf", description="Turn a 3D-scan mesh into a cleaned, printable 3MF.")
    parser.add_argument("--version", action="version", version=f"scan-to-3mf {__version__}")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("clean", help="clean a mesh and write a verified 3MF (default)")
    _add_clean_args(p)
    p.set_defaults(func=cmd_clean)
    p = sub.add_parser("inspect", help="report problems without writing anything")
    p.add_argument("input")
    p.add_argument("--unit")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-self-intersections", action="store_true")
    p.set_defaults(func=cmd_inspect)
    p = sub.add_parser("serve", help="start the local browser UI")
    p.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1, this computer only)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    p.set_defaults(func=cmd_serve)
    p = sub.add_parser("slicer", help="show which slicer the slice test would use")
    p.add_argument("--slicer")
    p.set_defaults(func=cmd_slicer)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"clean", "inspect", "serve", "slicer", "-h", "--help", "--version"}
    if argv and argv[0] not in commands:
        argv = ["clean"] + argv
    if not argv:
        argv = ["serve"]
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
