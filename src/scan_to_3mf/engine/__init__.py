"""scan-to-3mf engine: no web or CLI dependencies, so it can sit behind any front end."""

from .analysis import Analysis, Hole, Metrics, Shell, analyze
from .meshio import SUPPORTED_EXTENSIONS, LoadedMesh, MeshLoadError, load_mesh
from .pipeline import CleanOptions, MeshSnapshot, PipelineResult, inspect_loaded, run_pipeline, unit_hints
from .preview import decode_preview, encode_preview
from .slicer import Slicer, SliceResult, find_slicer, slice_check
from .threemf import read_3mf, write_3mf
from .units import DEFAULT_UNIT, UNITS, normalize_unit, to_mm
from .verify import verify_3mf

__all__ = [
    "Analysis", "CleanOptions", "DEFAULT_UNIT", "Hole", "LoadedMesh", "MeshLoadError", "MeshSnapshot", "Metrics",
    "PipelineResult", "SUPPORTED_EXTENSIONS", "Shell", "SliceResult", "Slicer", "UNITS", "analyze",
    "decode_preview", "encode_preview", "find_slicer", "inspect_loaded", "load_mesh", "normalize_unit",
    "read_3mf", "run_pipeline", "slice_check", "to_mm", "unit_hints", "verify_3mf", "write_3mf",
]
