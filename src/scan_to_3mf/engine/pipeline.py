"""Scan mesh in, cleaned and verified millimeter 3MF out.

The pipeline never scales the model except by the chosen/declared unit. Every
step records what it changed so the report can say it plainly.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from . import analysis as an
from . import repair
from .meshio import LoadedMesh, load_mesh
from .slicer import find_slicer, slice_check
from .threemf import write_3mf
from .units import UNITS, label, normalize_unit, to_mm
from .verify import verify_3mf

ProgressFn = Callable[[str, float], None]

APP_NAME = "scan-to-3mf"
REPORT_LIST_LIMIT = 50
NOT_CHECKED = [
    "Wall thickness: thin parts may be too thin to print.",
    "Overhangs and supports.",
    "Whether the model fits your printer's bed.",
    "Accuracy against the real object. Cleaning does not make a scan more accurate.",
]


def _version() -> str:
    from .. import __version__

    return __version__


@dataclass
class CleanOptions:
    unit: str | None = None  # unit of unitless files (STL/OBJ/PLY); default millimeters
    override_file_unit: bool = False  # use `unit` even if the 3MF declares one
    weld_tolerance_mm: float = 0.001
    remove_small_shells: bool = True
    min_shell_fraction: float = 0.01  # of the largest shell's triangle count
    fill_holes: bool = True
    max_hole_mm: float = 10.0  # holes whose rim spans at most this are filled
    fill_large_holes: bool = False  # fill every hole, whatever its size
    decimate_to: int | None = None
    place_on_bed: bool = False
    check_self_intersections: bool = True
    slicer_check: bool = True
    slicer_path: str | None = None
    writer: str = "auto"  # "auto" | "lib3mf" | "xml"
    metadata: dict | None = None  # 3MF metadata, e.g. {"Designer": ..., "LicenseTerms": ...}

    def normalized(self) -> CleanOptions:
        o = CleanOptions(**asdict(self))
        if o.metadata:
            from .threemf import WELL_KNOWN_METADATA

            bad = [k for k in o.metadata if k not in WELL_KNOWN_METADATA]
            if bad:
                raise ValueError(f"unsupported 3MF metadata name(s): {', '.join(bad)}")
            o.metadata = {str(k): str(v) for k, v in o.metadata.items() if v not in (None, "")}
        if o.unit is not None and str(o.unit).strip() != "":
            o.unit = normalize_unit(o.unit)
        else:
            o.unit = None
        if not 0 <= o.weld_tolerance_mm <= 1:
            raise ValueError("weld tolerance must be between 0 and 1 mm")
        if not 0 <= o.min_shell_fraction < 1:
            raise ValueError("minimum shell fraction must be between 0 and 1")
        if o.max_hole_mm is not None and o.max_hole_mm < 0:
            raise ValueError("maximum hole size must be positive")
        if o.decimate_to is not None:
            o.decimate_to = int(o.decimate_to)
            if o.decimate_to <= 0:
                o.decimate_to = None
            elif o.decimate_to < 100:
                raise ValueError("decimation target must be at least 100 triangles")
        if o.writer not in ("auto", "lib3mf", "xml"):
            raise ValueError("writer must be auto, lib3mf, or xml")
        return o

    @classmethod
    def from_dict(cls, data: dict) -> CleanOptions:
        known = {f.name: f for f in fields(cls)}
        kwargs = {}
        for key, value in (data or {}).items():
            if key not in known or value is None or value == "":
                continue
            default = getattr(cls(), key)
            if isinstance(default, bool):
                kwargs[key] = value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
            elif isinstance(default, float):
                kwargs[key] = float(value)
            elif key == "decimate_to":
                kwargs[key] = int(float(value))
            elif key == "metadata":
                kwargs[key] = dict(value)
            else:
                kwargs[key] = value
        return cls(**kwargs).normalized()


@dataclass
class Step:
    key: str
    title: str
    summary: str
    changed: bool
    seconds: float
    details: dict = field(default_factory=dict)


@dataclass
class MeshSnapshot:
    vertices: np.ndarray
    faces: np.ndarray
    rim: np.ndarray | None
    non_manifold: np.ndarray | None


@dataclass
class PipelineResult:
    printable: bool
    verdict: str
    input: dict
    options: dict
    units: dict
    before: dict
    after: dict | None
    before_holes: list[dict]
    after_holes: list[dict]
    removed_shells: list[dict]
    steps: list[dict]
    remaining: list[str]
    not_checked: list[str]
    verification: list[dict]
    slicer: dict | None
    output: dict | None
    timings: dict
    notes: list[str]
    version: str = ""
    before_mesh: MeshSnapshot | None = None
    after_mesh: MeshSnapshot | None = None

    def to_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("before_mesh", "after_mesh")}
        return _plain(d)


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def peak_memory_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def _n(x: int) -> str:
    return f"{x:,}"


def _size(ext) -> str:
    return " × ".join(f"{x:.4g}" if x < 1 else f"{x:.1f}" for x in ext) + " mm"


def _plural(n: int, word: str, plural: str | None = None) -> str:
    return f"{_n(n)} {word if n == 1 else (plural or word + 's')}"


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


def resolve_unit(loaded: LoadedMesh, options: CleanOptions) -> tuple[str, str, list[str]]:
    notes = []
    if loaded.file_unit and not options.override_file_unit:
        if options.unit and options.unit != loaded.file_unit:
            notes.append(f"Ignored the chosen unit ({label(options.unit)}) because the 3MF declares "
                         f"{label(loaded.file_unit)}. Use the override option to force it.")
        return loaded.file_unit, "file", notes
    if options.unit:
        return options.unit, "chosen", notes
    return "mm", "default", notes


def unit_hints(raw_extents, unit: str) -> list[str]:
    size = float(max(raw_extents)) * to_mm(unit)
    if 3 <= size <= 2000:
        return []
    alternatives = []
    for key in ("mm", "cm", "m", "in"):
        if key == unit:
            continue
        s = float(max(raw_extents)) * UNITS[key][1]
        if 10 <= s <= 600:
            alternatives.append(f"{label(key)} would make it {s:.0f} mm")
    what = "small" if size < 3 else "large"
    msg = f"The model is {size:.3g} mm across at the longest side, which is unusually {what}."
    if alternatives:
        msg += " Check the unit: " + "; ".join(alternatives) + "."
    return [msg]


def _unit_summary(loaded: LoadedMesh, unit: str, source: str, extents_mm) -> str:
    size = _size(extents_mm)
    factor = to_mm(unit)
    scale = "" if factor == 1 else f", so coordinates were multiplied by {factor:g}"
    if source == "file":
        return f"The 3MF declares {label(unit)}{scale}. Size: {size}."
    if source == "chosen":
        return f"{loaded.fmt.upper()} files have no units. You chose {label(unit)}{scale}. Size: {size}."
    return f"{loaded.fmt.upper()} files have no units. Assumed millimeters (the default). Size: {size}."


# ---------------------------------------------------------------------------
# Inspection only (used by the UI right after a file is dropped)
# ---------------------------------------------------------------------------


def inspect_loaded(loaded: LoadedMesh, unit: str | None = None, self_intersections: bool = True
                   ) -> tuple[dict, MeshSnapshot]:
    opts = CleanOptions(unit=unit).normalized()
    unit_key, source, notes = resolve_unit(loaded, opts)
    v = loaded.vertices * to_mm(unit_key)
    f = loaded.faces
    if loaded.soup:
        v, f, _ = repair.merge_identical(v, f)
    a = an.analyze(v, f, self_intersections=self_intersections)
    used = np.unique(loaded.faces)
    raw = loaded.vertices[used]
    raw_extents = (raw.max(axis=0) - raw.min(axis=0)).tolist()
    info = {
        "name": loaded.name,
        "format": loaded.fmt,
        "file_unit": loaded.file_unit,
        "unit": unit_key,
        "unit_source": source,
        "raw_extents": raw_extents,
        "file_triangles": loaded.file_triangles,
        "metrics": a.metrics.to_dict(),
        "holes": [asdict(h) for h in sorted(a.holes, key=lambda h: -h.span_mm)[:REPORT_LIST_LIMIT]],
        "notes": loaded.notes + notes,
        "unit_hints": unit_hints(raw_extents, unit_key),
    }
    return _plain(info), MeshSnapshot(v, f, a.boundary_segments, a.non_manifold_segments)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(path: str | Path, output_path: str | Path | None = None, options: CleanOptions | None = None,
                 progress: ProgressFn | None = None, loaded: LoadedMesh | None = None) -> PipelineResult:
    options = (options or CleanOptions()).normalized()
    report = progress or (lambda message, fraction: None)
    t_start = time.perf_counter()
    steps: list[Step] = []
    timings: dict[str, float] = {}
    notes: list[str] = []

    def timed(key: str, t0: float) -> float:
        now = time.perf_counter()
        timings[key] = round(timings.get(key, 0.0) + now - t0, 3)
        return now

    # -- read ------------------------------------------------------------------
    report("Reading the file", 0.02)
    t = time.perf_counter()
    if loaded is None:
        loaded = load_mesh(path)
    notes += loaded.notes
    t = timed("read", t)

    unit, unit_source, unit_notes = resolve_unit(loaded, options)
    notes += unit_notes
    factor = to_mm(unit)
    v = loaded.vertices * factor
    f = loaded.faces.copy()
    identical = 0
    if loaded.soup:
        v, f, identical = repair.merge_identical(v, f)

    report("Measuring the input", 0.08)
    before = an.analyze(v, f, self_intersections=options.check_self_intersections)
    before_snapshot = MeshSnapshot(v, f, before.boundary_segments, before.non_manifold_segments)
    t = timed("analyze_before", t)
    raw_used = loaded.vertices[np.unique(loaded.faces)]
    raw_extents = (raw_used.max(axis=0) - raw_used.min(axis=0)).tolist()
    hints = unit_hints(raw_extents, unit)
    steps.append(Step("units", "Units", _unit_summary(loaded, unit, unit_source, before.metrics.extents),
                      factor != 1, 0.0, {"unit": unit, "source": unit_source, "factor_to_mm": factor}))

    # -- weld --------------------------------------------------------------------
    report("Welding duplicate vertices", 0.2)
    tol = options.weld_tolerance_mm
    f, rim_welds = repair.weld_boundary(v, f, tol)
    f, collapsed = repair.collapse_short_edges(v, f, tol)
    parts = []
    if identical:
        parts.append(f"Shared {_n(identical)} duplicate vertex copies (STL stores every triangle's corners separately).")
    if rim_welds:
        parts.append(f"Closed cracks by welding {_plural(rim_welds, 'open-edge vertex', 'open-edge vertices')} "
                     f"within {tol:g} mm.")
    if collapsed:
        parts.append(f"Collapsed {_plural(collapsed, 'zero-length edge')} (≤ {tol:g} mm).")
    steps.append(Step("weld", "Weld duplicate vertices", " ".join(parts) or "No duplicate vertices.",
                      bool(identical or rim_welds or collapsed), 0.0,
                      {"identical": identical, "rim_welds": rim_welds, "collapsed_edges": collapsed,
                       "tolerance_mm": tol}))
    t = timed("weld", t)

    # -- degenerate / duplicate faces -------------------------------------------
    report("Removing degenerate and duplicate triangles", 0.28)
    f, n_degenerate = repair.drop_degenerate(v, f)
    f, zero = repair.fix_zero_area(v, f)
    f, n_duplicate = repair.drop_duplicate_faces(f)
    v, f, n_unused = repair.compact(v, f)
    parts = []
    if n_degenerate:
        parts.append(f"Removed {_plural(n_degenerate, 'degenerate triangle')} (a corner used twice).")
    if zero["fixed"]:
        parts.append(f"Removed {_plural(zero['fixed'], 'zero-area sliver')} by flipping an edge (no hole created).")
    if n_duplicate:
        parts.append(f"Removed {_plural(n_duplicate, 'duplicate triangle')}.")
    if zero["remaining"]:
        parts.append(f"Left {_plural(zero['remaining'], 'zero-area sliver')} in place: removing them would open "
                     "the surface, and slicers ignore them.")
    if n_unused:
        parts.append(f"Dropped {_plural(n_unused, 'unused vertex', 'unused vertices')}.")
    steps.append(Step("faces", "Degenerate and duplicate triangles",
                      " ".join(parts) or "No degenerate or duplicate triangles.",
                      bool(n_degenerate or zero["fixed"] or n_duplicate or n_unused), 0.0,
                      {"degenerate": n_degenerate, "slivers_fixed": zero["fixed"],
                       "slivers_left": zero["remaining"], "duplicates": n_duplicate, "unused_vertices": n_unused}))
    t = timed("faces", t)

    # -- floating shells ---------------------------------------------------------
    report("Removing tiny floating shells", 0.34)
    removed_shells: list[an.Shell] = []
    threshold = 0
    if options.remove_small_shells:
        f, removal = repair.remove_small_shells(v, f, options.min_shell_fraction)
        removed_shells += removal.removed
        threshold = removal.threshold_triangles
        v, f, _ = repair.compact(v, f)
        kept = removal.kept
    else:
        kept = an.face_components(f, len(v))[0]
    reference_used = v[np.unique(f)]
    reference_extents = (reference_used.max(axis=0) - reference_used.min(axis=0)).tolist()
    t = timed("shells", t)

    # -- non-manifold, orientation, holes (repeat until stable) -------------------
    nm_faces_removed = nm_vertices_added = nm_edges_found = nm_vertices_found = 0
    coherence_flips = shells_outward = faces_flipped = 0
    first_fill: repair.HoleFill | None = None
    holes_filled = faces_added = fan_filled = 0
    leftovers: list[repair.HoleFill] = []
    fragments: list[an.Shell] = []
    fill_enabled = options.fill_holes or options.fill_large_holes
    cap = options.max_hole_mm if options.fill_holes else None
    for rnd in range(3):
        report("Repairing non-manifold edges and vertices", 0.42 + 0.02 * rnd)
        topo = an.edge_topology(f, len(v))
        n_nm_e = topo.non_manifold_edges
        n_nm_v = len(an.non_manifold_vertices(f, len(v), topo))
        if rnd == 0:
            nm_edges_found, nm_vertices_found = n_nm_e, n_nm_v
        if n_nm_e or n_nm_v:
            v, f, st = repair.repair_non_manifold(v, f)
            nm_faces_removed += st["faces_removed"]
            nm_vertices_added += st["vertices_added"]
            f, _ = repair.drop_degenerate(v, f)
            f, _ = repair.drop_duplicate_faces(f)
            v, f, _ = repair.compact(v, f)
            if options.remove_small_shells:
                f, extra = repair.remove_small_shells(v, f, options.min_shell_fraction)
                fragments += extra.removed
                v, f, _ = repair.compact(v, f)
        t = timed("non_manifold", t)

        report("Making normals consistent and outward", 0.5)
        f, st = repair.orient_outward(v, f)
        coherence_flips += max(st["coherence_flips"], 0)
        shells_outward += st["shells_turned_outward"]
        faces_flipped += max(st["faces_flipped"], 0)
        t = timed("orientation", t)

        if fill_enabled:
            report("Filling holes", 0.58)
            v, f, hf = repair.fill_holes(v, f, cap, options.fill_large_holes)
            if first_fill is None:
                first_fill = hf
            holes_filled += len(hf.candidates) - hf.unfilled_candidates
            faces_added += hf.faces_added
            fan_filled += hf.fan_filled
            if hf.leftover_method:
                leftovers.append(hf)
            t = timed("holes", t)

        topo = an.edge_topology(f, len(v))
        if topo.non_manifold_edges == 0 and len(an.non_manifold_vertices(f, len(v), topo)) == 0:
            if fill_enabled and hf.faces_added:
                f, st = repair.orient_outward(v, f)
                shells_outward += st["shells_turned_outward"]
            break

    if removed_shells or fragments:
        total = sum(s.triangles for s in removed_shells)
        largest = max((max(s.extents) for s in removed_shells), default=0.0)
        summary = (f"Removed {_plural(len(removed_shells), 'floating shell')} with fewer than "
                   f"{_n(threshold)} triangles ({options.min_shell_fraction:.1%} of the main shell): "
                   f"{_n(total)} triangles in total, the largest {largest:.3g} mm across.") if removed_shells else ""
        if fragments:
            summary += (" " if summary else "") + (f"Removed {_plural(len(fragments), 'small fragment')} left "
                                                    "behind by the non-manifold repair.")
        steps.append(Step("shells", "Tiny floating shells", summary, True, timings.get("shells", 0.0),
                          {"threshold_triangles": threshold, "removed": len(removed_shells),
                           "fragments": len(fragments), "kept": kept}))
    elif options.remove_small_shells:
        steps.append(Step("shells", "Tiny floating shells",
                          f"{_plural(kept, 'shell')}; none below {options.min_shell_fraction:.1%} of the main "
                          "shell, nothing removed.", False, timings.get("shells", 0.0), {"kept": kept}))
    else:
        steps.append(Step("shells", "Tiny floating shells", "Shell removal is off.", False, 0.0, {}))

    if nm_edges_found or nm_vertices_found or nm_faces_removed or nm_vertices_added:
        parts = []
        if nm_edges_found:
            parts.append(f"Found {_plural(nm_edges_found, 'non-manifold edge')} (shared by 3+ triangles); "
                         f"removed {_plural(nm_faces_removed, 'triangle')} to fix them.")
        if nm_vertices_found or nm_vertices_added:
            parts.append(f"Split {_plural(max(nm_vertices_added, nm_vertices_found), 'pinched vertex', 'pinched vertices')}"
                         " so each surface sheet has its own copy.")
        steps.append(Step("non_manifold", "Non-manifold edges and vertices", " ".join(parts), True,
                          timings.get("non_manifold", 0.0),
                          {"edges": nm_edges_found, "vertices": nm_vertices_found,
                           "faces_removed": nm_faces_removed, "vertices_added": nm_vertices_added}))
    else:
        steps.append(Step("non_manifold", "Non-manifold edges and vertices", "None found.", False,
                          timings.get("non_manifold", 0.0), {}))

    if coherence_flips or shells_outward:
        parts = []
        if coherence_flips:
            parts.append(f"Flipped {_plural(coherence_flips, 'triangle')} so neighbouring triangles agree.")
        if shells_outward:
            parts.append(f"Turned {_plural(shells_outward, 'inside-out shell')} outward.")
        steps.append(Step("orientation", "Normals", " ".join(parts), True, timings.get("orientation", 0.0),
                          {"coherence_flips": coherence_flips, "shells_turned_outward": shells_outward,
                           "faces_flipped": faces_flipped}))
    else:
        steps.append(Step("orientation", "Normals", "Already consistent and facing outward.", False,
                          timings.get("orientation", 0.0), {}))

    skipped = first_fill.skipped if first_fill else []
    if not fill_enabled:
        steps.append(Step("holes", "Holes", f"Hole filling is off; {_plural(before.metrics.holes, 'hole')} left as found.",
                          False, 0.0, {}))
    else:
        parts = []
        if holes_filled:
            what = "hole" if options.fill_large_holes else f"hole up to {options.max_hole_mm:g} mm across"
            parts.append(f"Filled {_plural(holes_filled, what, what.replace('hole', 'holes', 1))} "
                         f"(added {_plural(faces_added, 'triangle')}).")
            for hf in leftovers:
                crossing = ("" if hf.leftover_crossings is None
                            else f", which crossed the fewest other triangles ({hf.leftover_crossings:,})")
                parts.append(f"MeshLab's careful fill stopped short on {_plural(hf.leftover_holes, 'hole')}; the rest "
                             f"was closed with {hf.leftover_method}{crossing}.")
            if options.fill_large_holes and any(h.span_mm > (options.max_hole_mm or 0) for h in (first_fill.candidates if first_fill else [])):
                parts.append("Large fills are invented surface: the scanner saw nothing there.")
            if fan_filled:
                parts.append(f"{_plural(fan_filled, 'hole')} needed the simple fan fallback.")
        if skipped:
            spans = ", ".join(f"{h.span_mm:.1f}" for h in sorted(skipped, key=lambda h: -h.span_mm)[:6])
            parts.append(f"Left {_plural(len(skipped), 'larger hole')} open ({spans} mm across) instead of "
                         "inventing surface. Turn on “fill larger holes” to close them.")
        if not parts:
            parts.append("No holes." if before.metrics.holes == 0 else "No holes left after welding.")
        steps.append(Step("holes", "Holes", " ".join(parts), bool(holes_filled), timings.get("holes", 0.0),
                          {"filled": holes_filled, "faces_added": faces_added, "left_open": len(skipped),
                           "cap_mm": options.max_hole_mm, "fill_large": options.fill_large_holes}))

    # -- decimation ----------------------------------------------------------------
    if options.decimate_to:
        report("Decimating", 0.66)
        n0 = len(f)
        if n0 > options.decimate_to:
            v, f = repair.decimate(v, f, options.decimate_to)
            v, f, _ = repair.compact(v, f)
            f, _ = repair.drop_degenerate(v, f)
            f, _ = repair.drop_duplicate_faces(f)
            topo = an.edge_topology(f, len(v))
            if topo.non_manifold_edges or len(an.non_manifold_vertices(f, len(v), topo)):
                v, f, _ = repair.repair_non_manifold(v, f)
                v, f, _ = repair.compact(v, f)
            f, _ = repair.orient_outward(v, f)
            if fill_enabled:
                v, f, _ = repair.fill_holes(v, f, cap, options.fill_large_holes)
                f, _ = repair.orient_outward(v, f)
            steps.append(Step("decimate", "Decimation", f"Reduced {_n(n0)} → {_n(len(f))} triangles "
                              "(quadric edge collapse, topology preserved).", True, 0.0,
                              {"before": n0, "after": len(f), "target": options.decimate_to}))
        else:
            steps.append(Step("decimate", "Decimation", f"Skipped: {_n(n0)} triangles is already at or below the "
                              f"target of {_n(options.decimate_to)}.", False, 0.0, {}))
        t = timed("decimate", t)

    # -- place on bed --------------------------------------------------------------------
    dz = 0.0
    if options.place_on_bed:
        v, dz = repair.place_on_bed(v, f)
        steps.append(Step("place_on_bed", "Place on bed",
                          f"Moved the model {dz:+.3f} mm in Z so its lowest point is at z = 0.", abs(dz) > 0,
                          0.0, {"dz_mm": dz}))

    report("Measuring the result", 0.74)
    after = an.analyze(v, f, self_intersections=options.check_self_intersections)
    after_snapshot = MeshSnapshot(v, f, after.boundary_segments, after.non_manifold_segments)
    t = timed("analyze_after", t)
    open_centres = np.array([h.center for h in after.holes]).reshape(-1, 3) - np.array([0.0, 0.0, dz])

    def hole_status(h: an.Hole) -> dict:
        d = asdict(h)
        near = len(open_centres) and np.linalg.norm(open_centres - np.array(h.center), axis=1).min() <= max(0.5, 0.25 * h.span_mm)
        d["status"] = "open" if near else "closed"
        return d

    # -- write + verify + slice ----------------------------------------------------------
    verification: list[dict] = []
    slicer_result = None
    output = None
    if output_path is not None:
        report("Writing the 3MF", 0.8)
        output_path = Path(output_path)
        metadata = {"Title": loaded.name, "CreationDate": time.strftime("%Y-%m-%d")}
        metadata.update(options.metadata or {})
        writer = write_3mf(output_path, v, f, name=metadata["Title"], application=f"{APP_NAME} {_version()}",
                           writer=options.writer, metadata=metadata)
        output = {"path": str(output_path), "bytes": output_path.stat().st_size, "writer": writer,
                  "unit": "millimeter", "triangles": int(len(f)), "vertices": int(len(v))}
        steps.append(Step("export", "Write 3MF",
                          f"Wrote a geometry-only 3MF in millimeters with {'lib3mf' if writer == 'lib3mf' else 'the built-in writer'} "
                          f"(one object, {_n(len(f))} triangles, {output['bytes'] / 1e6:.1f} MB).", True, 0.0,
                          dict(output)))
        t = timed("write", t)

        report("Reading the 3MF back to verify it", 0.84)
        checks = verify_3mf(output_path, v, f, reference_extents,
                            size_tolerance=0.01 if options.decimate_to else 0.005)
        verification = [c.to_dict() for c in checks]
        t = timed("verify", t)

        if options.slicer_check:
            report("Slicing with a real slicer", 0.9)
            slicer = find_slicer(options.slicer_path)
            if slicer is None:
                slicer_result = {"ok": None, "slicer": None,
                                 "message": "No slicer found (PrusaSlicer, OrcaSlicer, or Bambu Studio), so the slice "
                                            "test was skipped."}
            else:
                res = slice_check(slicer, output_path, after.metrics.extents)
                slicer_result = res.to_dict()
                info = res.info
                if info.get("size_x") is not None:
                    got = [info.get("size_x"), info.get("size_y"), info.get("size_z")]
                    diff = max(abs(a - b) for a, b in zip(got, after.metrics.extents, strict=True))
                    ok = diff <= max(0.01 * max(after.metrics.extents), 0.05)
                    verification.append({"key": "slicer_size", "label": "Slicer reads the same size in mm",
                                         "ok": ok, "detail": f"{slicer.label}: {_size(got)}"})
                if "manifold" in info:
                    closed = str(info["manifold"]).strip().lower() == "yes"
                    verification.append({"key": "slicer_manifold", "label": "Slicer sees a closed mesh",
                                         "ok": closed, "detail": f"{slicer.label} reports manifold = {info['manifold']}"})
            t = timed("slice", t)

    timings["total"] = round(time.perf_counter() - t_start, 3)
    peak = peak_memory_mb()
    if peak is not None:
        timings["peak_memory_mb"] = round(peak, 1)

    remaining = _remaining_issues(after, hints, verification, slicer_result)
    failed_checks = [c for c in verification if c.get("ok") is False]
    m = after.metrics
    printable = bool(m.watertight and m.non_manifold_vertices == 0 and m.orientation == "outward"
                     and not failed_checks and (slicer_result is None or slicer_result.get("ok") is not False))
    if output is None:
        verdict = "Analysed only (no file written)."
    elif printable and slicer_result and slicer_result.get("ok"):
        verdict = "Ready to print: watertight, verified, and sliced successfully."
    elif printable:
        verdict = "Watertight and verified. No slice test was run (see verification)."
    else:
        verdict = "Written, but not printable yet. See what is still wrong."

    return PipelineResult(
        printable=printable,
        verdict=verdict,
        input={"name": loaded.name, "format": loaded.fmt, "path": str(path), "file_unit": loaded.file_unit,
               "file_triangles": loaded.file_triangles, "raw_extents": raw_extents},
        options=asdict(options),
        units={"unit": unit, "label": label(unit), "source": unit_source, "factor_to_mm": factor, "hints": hints},
        before=before.metrics.to_dict(),
        after=after.metrics.to_dict(),
        before_holes=[hole_status(h) for h in sorted(before.holes, key=lambda h: -h.span_mm)[:REPORT_LIST_LIMIT]],
        after_holes=[asdict(h) for h in sorted(after.holes, key=lambda h: -h.span_mm)[:REPORT_LIST_LIMIT]],
        removed_shells=[asdict(s) for s in (removed_shells + fragments)[:REPORT_LIST_LIMIT]],
        steps=[asdict(s) for s in steps],
        remaining=remaining,
        not_checked=list(NOT_CHECKED),
        verification=verification,
        slicer=slicer_result,
        output=output,
        timings=timings,
        notes=notes,
        version=_version(),
        before_mesh=before_snapshot,
        after_mesh=after_snapshot,
    )


def _remaining_issues(after: an.Analysis, hints: list[str], verification: list[dict], slicer: dict | None) -> list[str]:
    m = after.metrics
    issues: list[str] = []
    if m.holes:
        spans = ", ".join(f"{h.span_mm:.1f}" for h in sorted(after.holes, key=lambda h: -h.span_mm)[:6])
        issues.append(f"{_plural(m.holes, 'hole')} still open ({spans} mm across). The mesh is not watertight, so "
                      "slicers may refuse it or print gaps. Turn on “fill larger holes” to close them.")
    if m.non_manifold_edges or m.non_manifold_vertices:
        issues.append(f"{_plural(m.non_manifold_edges, 'non-manifold edge')} and "
                      f"{_plural(m.non_manifold_vertices, 'non-manifold vertex', 'non-manifold vertices')} remain.")
    if m.orientation in ("inward", "mixed", "inconsistent"):
        issues.append(f"Normals are {m.orientation}; the slicer may treat the inside as outside.")
    if m.degenerate_faces:
        issues.append(f"{_plural(m.degenerate_faces, 'zero-area sliver')} remain. They do not affect slicing.")
    if m.self_intersecting_faces:
        issues.append(f"{_plural(m.self_intersecting_faces, 'triangle')} cross other triangles (self-intersections, "
                      "not repaired in v1). Slicers usually merge the overlap, but check the slice preview.")
    if m.shells > 1:
        issues.append(f"The model has {m.shells} separate shells. They stay in one object and print as separate "
                      "pieces unless they touch.")
    issues += hints
    for c in verification:
        if c.get("ok") is False:
            issues.append(f"Check failed: {c['label']} — {c['detail']}.")
    if slicer is not None and slicer.get("ok") is False:
        issues.append(f"The slicer could not slice the file: {slicer.get('message')}.")
    return issues
