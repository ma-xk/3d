"""3MF reading and writing.

Two independent implementations:

* lib3mf (the 3MF Consortium reference library) is the default reader and writer.
  Its Python binding copies geometry through Python lists, which is far too slow
  for million-triangle scans, so geometry is moved through numpy buffers that have
  the same memory layout as lib3mf's packed ``Position``/``Triangle`` structs.
* A small standard-library reader/writer (zipfile + expat). It is used to verify
  every output independently of lib3mf, and as a fallback if lib3mf cannot load.
"""

from __future__ import annotations

import ctypes
import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from xml.parsers import expat
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from .units import UnitError, normalize_unit

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
MODEL_REL_TYPE = "http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"
MODEL_CONTENT_TYPE = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml"

# lib3mf ModelUnit enum value -> unit key
_LIB3MF_UNITS = {0: "um", 1: "mm", 2: "cm", 3: "in", 4: "ft", 5: "m"}


class ThreeMFError(Exception):
    pass


@dataclass
class ThreeMFContent:
    vertices: np.ndarray  # (n, 3) float64, file units, build transforms applied
    faces: np.ndarray  # (m, 3) int64
    unit: str  # unit key, e.g. "mm" or "in"
    instances: int  # mesh instances merged into one
    reader: str  # "lib3mf" or "xml"
    notes: list[str] = field(default_factory=list)


def lib3mf_available() -> bool:
    try:
        _lib3mf()
        return True
    except Exception:
        return False


def _lib3mf():
    import lib3mf

    return lib3mf, lib3mf.get_wrapper()


# ---------------------------------------------------------------------------
# lib3mf
# ---------------------------------------------------------------------------


def _l3_get_geometry(lib3mf, mesh_obj) -> tuple[np.ndarray, np.ndarray]:
    nv = mesh_obj.GetVertexCount()
    nt = mesh_obj.GetTriangleCount()
    verts = np.empty((nv, 3), dtype=np.float32)
    tris = np.empty((nt, 3), dtype=np.uint32)
    try:
        wrapper, lib = mesh_obj._wrapper, mesh_obj._wrapper.lib
        needed = ctypes.c_uint64(0)
        if nv:
            wrapper.checkError(
                mesh_obj,
                lib.lib3mf_meshobject_getvertices(
                    mesh_obj._handle, ctypes.c_uint64(nv), needed,
                    verts.ctypes.data_as(ctypes.POINTER(lib3mf.Position)),
                ),
            )
        if nt:
            wrapper.checkError(
                mesh_obj,
                lib.lib3mf_meshobject_gettriangleindices(
                    mesh_obj._handle, ctypes.c_uint64(nt), needed,
                    tris.ctypes.data_as(ctypes.POINTER(lib3mf.Triangle)),
                ),
            )
    except AttributeError:  # binding internals changed: use the slow public API
        verts = np.array([p.Coordinates[:] for p in mesh_obj.GetVertices()], np.float32).reshape(-1, 3)
        tris = np.array([t.Indices[:] for t in mesh_obj.GetTriangleIndices()], np.uint32).reshape(-1, 3)
    return verts, tris


def _l3_set_geometry(lib3mf, mesh_obj, verts: np.ndarray, tris: np.ndarray) -> None:
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.uint32)
    try:
        wrapper, lib = mesh_obj._wrapper, mesh_obj._wrapper.lib
        wrapper.checkError(
            mesh_obj,
            lib.lib3mf_meshobject_setgeometry(
                mesh_obj._handle,
                ctypes.c_uint64(len(verts)), verts.ctypes.data_as(ctypes.POINTER(lib3mf.Position)),
                ctypes.c_uint64(len(tris)), tris.ctypes.data_as(ctypes.POINTER(lib3mf.Triangle)),
            ),
        )
    except AttributeError:
        positions = [lib3mf.Position((ctypes.c_float * 3)(*v)) for v in verts.tolist()]
        triangles = [lib3mf.Triangle((ctypes.c_uint32 * 3)(*t)) for t in tris.tolist()]
        mesh_obj.SetGeometry(positions, triangles)


def _l3_matrix(transform) -> np.ndarray:
    m = np.eye(4)
    for r in range(4):
        for c in range(3):
            m[r, c] = transform.Fields[r][c]
    return m


def read_3mf_lib3mf(path: str | Path) -> ThreeMFContent:
    lib3mf, wrapper = _lib3mf()
    model = wrapper.CreateModel()
    reader = model.QueryReader("3mf")
    reader.SetStrictModeActive(False)
    try:
        reader.ReadFromFile(str(path))
    except lib3mf.ELib3MFException as exc:
        raise ThreeMFError(f"lib3mf could not read the 3MF file: {exc}") from exc
    notes = []
    warnings = reader.GetWarningCount()
    if warnings:
        notes.append(f"lib3mf reported {warnings} warning(s) while reading")
    unit = _LIB3MF_UNITS.get(int(model.GetUnit()), "mm")

    parts_v: list[np.ndarray] = []
    parts_f: list[np.ndarray] = []
    offset = 0
    skipped_types: set[str] = set()

    def visit(obj, matrix: np.ndarray, depth: int) -> None:
        nonlocal offset
        if depth > 32:
            raise ThreeMFError("3MF component nesting is too deep")
        if obj.IsMeshObject():
            mesh = model.GetMeshObjectByID(obj.GetResourceID())
            otype = mesh.GetType()
            if otype not in (lib3mf.ObjectType.Model, lib3mf.ObjectType.SolidSupport):
                skipped_types.add(otype.name.lower())
                return
            v, f = _l3_get_geometry(lib3mf, mesh)
            v = v.astype(np.float64) @ matrix[:3, :3] + matrix[3, :3]
            parts_v.append(v)
            parts_f.append(f.astype(np.int64) + offset)
            offset += len(v)
        elif obj.IsComponentsObject():
            comp_obj = model.GetComponentsObjectByID(obj.GetResourceID())
            for i in range(comp_obj.GetComponentCount()):
                comp = comp_obj.GetComponent(i)
                local = _l3_matrix(comp.GetTransform()) if comp.HasTransform() else np.eye(4)
                visit(comp.GetObjectResource(), local @ matrix, depth + 1)

    items = model.GetBuildItems()
    count = 0
    while items.MoveNext():
        item = items.GetCurrent()
        matrix = _l3_matrix(item.GetObjectTransform()) if item.HasObjectTransform() else np.eye(4)
        visit(item.GetObjectResource(), matrix, 0)
        count += 1
    if skipped_types:
        notes.append(f"Skipped non-model objects ({', '.join(sorted(skipped_types))})")
    if not parts_v:
        raise ThreeMFError("The 3MF file contains no printable mesh objects")
    return ThreeMFContent(
        vertices=np.concatenate(parts_v), faces=np.concatenate(parts_f),
        unit=unit, instances=len(parts_v), reader="lib3mf", notes=notes,
    )


def write_3mf_lib3mf(path: str | Path, vertices_mm: np.ndarray, faces: np.ndarray,
                     name: str = "scan", application: str = "scan-to-3mf", metadata: dict | None = None) -> None:
    lib3mf, wrapper = _lib3mf()
    model = wrapper.CreateModel()
    model.SetUnit(lib3mf.ModelUnit.MilliMeter)
    mesh = model.AddMeshObject()
    mesh.SetName(name)
    mesh.SetType(lib3mf.ObjectType.Model)
    _l3_set_geometry(lib3mf, mesh, vertices_mm, faces)
    model.AddBuildItem(mesh, wrapper.GetIdentityTransform())
    group = model.GetMetaDataGroup()
    for key, value in _metadata(application, metadata).items():
        group.AddMetaData("", key, value, "xs:string", False)
    writer = model.QueryWriter("3mf")
    writer.WriteToFile(str(path))


# Metadata names the 3MF core specification defines for the default namespace.
WELL_KNOWN_METADATA = ("Title", "Designer", "Description", "Copyright", "LicenseTerms", "Rating",
                       "CreationDate", "ModificationDate", "Application")


def _metadata(application: str, metadata: dict | None) -> dict[str, str]:
    items = {"Application": application}
    for key, value in (metadata or {}).items():
        if key not in WELL_KNOWN_METADATA:
            raise ValueError(f"Unsupported 3MF metadata name {key!r}. Use one of: {', '.join(WELL_KNOWN_METADATA)}")
        if value not in (None, ""):
            items[key] = str(value)
    return items


# ---------------------------------------------------------------------------
# Standard-library implementation
# ---------------------------------------------------------------------------


def _parse_transform(text: str | None) -> np.ndarray:
    m = np.eye(4)
    if not text:
        return m
    vals = [float(x) for x in text.split()]
    if len(vals) != 12:
        raise ThreeMFError(f"Bad 3MF transform: {text!r}")
    m[:, :3] = np.array(vals).reshape(4, 3)
    return m


def _root_model_part(z: zipfile.ZipFile) -> str:
    try:
        rels = z.read("_rels/.rels")
    except KeyError as exc:
        raise ThreeMFError("Not a 3MF package (missing _rels/.rels)") from exc
    target = None

    def start(name, attrs):
        nonlocal target
        if name.endswith("Relationship") and attrs.get("Type") == MODEL_REL_TYPE and target is None:
            target = attrs.get("Target")

    p = expat.ParserCreate()
    p.StartElementHandler = start
    p.Parse(rels, True)
    if not target:
        raise ThreeMFError("3MF package has no 3D model part")
    return target.lstrip("/")


class _ModelParser:
    """Streaming parser for a 3MF core model part (no DOM, low memory)."""

    def __init__(self) -> None:
        self.unit = "millimeter"
        self.objects: dict[str, dict] = {}
        self.build: list[tuple[str, np.ndarray]] = []
        self._obj: dict | None = None
        self._v: list[str] = []
        self._t: list[str] = []
        self.external_refs = 0

    def start(self, name: str, attrs: dict) -> None:
        ns, _, local = name.rpartition(" ")
        if ns != CORE_NS:
            return
        if local == "vertex":
            self._v.append(attrs["x"])
            self._v.append(attrs["y"])
            self._v.append(attrs["z"])
        elif local == "triangle":
            self._t.append(attrs["v1"])
            self._t.append(attrs["v2"])
            self._t.append(attrs["v3"])
        elif local == "object":
            self._obj = {"id": attrs["id"], "type": attrs.get("type", "model"), "components": []}
            self._v, self._t = [], []
        elif local == "component" and self._obj is not None:
            if any(k.endswith(" path") for k in attrs):
                self.external_refs += 1
            self._obj["components"].append((attrs["objectid"], _parse_transform(attrs.get("transform"))))
        elif local == "item":
            if any(k.endswith(" path") for k in attrs):
                self.external_refs += 1
            self.build.append((attrs["objectid"], _parse_transform(attrs.get("transform"))))
        elif local == "model":
            self.unit = attrs.get("unit", "millimeter")

    def end(self, name: str) -> None:
        if name == CORE_NS + " object" and self._obj is not None:
            if self._v or self._t:
                self._obj["vertices"] = np.array(self._v, dtype=np.float64).reshape(-1, 3)
                self._obj["faces"] = np.array(self._t, dtype=np.int64).reshape(-1, 3)
            self.objects[self._obj["id"]] = self._obj
            self._obj, self._v, self._t = None, [], []


def read_3mf_xml(path: str | Path) -> ThreeMFContent:
    try:
        with zipfile.ZipFile(path) as z:
            part = _root_model_part(z)
            data = z.read(part)
    except zipfile.BadZipFile as exc:
        raise ThreeMFError("Not a 3MF file (it is not a ZIP package)") from exc
    except KeyError as exc:
        raise ThreeMFError(f"3MF package is missing its model part: {exc}") from exc
    mp = _ModelParser()
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.StartElementHandler = mp.start
    parser.EndElementHandler = mp.end
    try:
        parser.Parse(data, True)
    except expat.ExpatError as exc:
        raise ThreeMFError(f"3MF model XML is malformed: {exc}") from exc
    try:
        unit = normalize_unit(mp.unit)
    except UnitError as exc:
        raise ThreeMFError(f"3MF declares an unknown unit {mp.unit!r}") from exc

    parts_v, parts_f, offset = [], [], 0
    notes = []
    if mp.external_refs:
        notes.append("Ignored references to other model parts (production extension)")

    def visit(oid: str, matrix: np.ndarray, depth: int) -> None:
        nonlocal offset
        if depth > 32:
            raise ThreeMFError("3MF component nesting is too deep")
        obj = mp.objects.get(oid)
        if obj is None:
            return
        if "vertices" in obj:
            if obj["type"] not in ("model", "solidsupport"):
                return
            v = obj["vertices"] @ matrix[:3, :3] + matrix[3, :3]
            parts_v.append(v)
            parts_f.append(obj["faces"] + offset)
            offset += len(v)
        for cid, local in obj["components"]:
            visit(cid, local @ matrix, depth + 1)

    for oid, matrix in mp.build:
        visit(oid, matrix, 0)
    if not parts_v:
        raise ThreeMFError("The 3MF file contains no printable mesh objects")
    return ThreeMFContent(
        vertices=np.concatenate(parts_v), faces=np.concatenate(parts_f),
        unit=unit, instances=len(parts_v), reader="xml", notes=notes,
    )


def read_3mf_unit_attribute(path: str | Path) -> str | None:
    """Return the raw ``unit`` attribute of the root model element (None if absent)."""
    with zipfile.ZipFile(path) as z:
        data = z.read(_root_model_part(z))
    found: list[str | None] = []

    def start(name, attrs):
        if not found and name == CORE_NS + " model":
            found.append(attrs.get("unit"))
            raise StopIteration

    p = expat.ParserCreate(namespace_separator=" ")
    p.StartElementHandler = start
    try:
        p.Parse(data, True)
    except StopIteration:
        pass
    return found[0] if found else None


def _format_rows(prefix: str, arr: np.ndarray, attrs: tuple[str, str, str], fmt: str) -> str:
    template = prefix + " ".join(f'{a}="{fmt}"' for a in attrs) + "/>"
    buf = io.StringIO()
    np.savetxt(buf, arr, fmt=template)
    return buf.getvalue()


def write_3mf_xml(path: str | Path, vertices_mm: np.ndarray, faces: np.ndarray,
                  name: str = "scan", application: str = "scan-to-3mf", metadata: dict | None = None) -> None:
    verts = np.asarray(vertices_mm, dtype=np.float32).astype(np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    model = io.StringIO()
    model.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    model.write(f'<model unit="millimeter" xml:lang="en-US" xmlns="{CORE_NS}">\n')
    for key, value in _metadata(application, metadata).items():
        model.write(f" <metadata name={quoteattr(key)}>{escape(value)}</metadata>\n")
    model.write(" <resources>\n")
    model.write(f'  <object id="1" type="model" name={quoteattr(name)}>\n   <mesh>\n    <vertices>\n')
    model.write(_format_rows("     <vertex ", verts, ("x", "y", "z"), "%.6f"))
    model.write("    </vertices>\n    <triangles>\n")
    model.write(_format_rows("     <triangle ", tris, ("v1", "v2", "v3"), "%d"))
    model.write("    </triangles>\n   </mesh>\n  </object>\n </resources>\n")
    model.write(' <build>\n  <item objectid="1"/>\n </build>\n</model>\n')
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        f'<Default Extension="model" ContentType="{MODEL_CONTENT_TYPE}"/>'
        "</Types>\n"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<Relationships xmlns="{REL_NS}">'
        f'<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="{MODEL_REL_TYPE}"/>'
        "</Relationships>\n"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model.getvalue())


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def read_3mf(path: str | Path) -> ThreeMFContent:
    """Read a 3MF with lib3mf, falling back to the standard-library reader."""
    try:
        return read_3mf_lib3mf(path)
    except ThreeMFError:
        raise
    except Exception as exc:  # lib3mf missing or failed to load
        content = read_3mf_xml(path)
        content.notes.append(f"lib3mf unavailable ({exc.__class__.__name__}); used built-in reader")
        return content


def write_3mf(path: str | Path, vertices_mm: np.ndarray, faces: np.ndarray, *,
              name: str = "scan", application: str = "scan-to-3mf", writer: str = "auto",
              metadata: dict | None = None) -> str:
    """Write a geometry-only, millimeter 3MF. Returns the writer used."""
    if writer not in ("auto", "lib3mf", "xml"):
        raise ValueError(f"unknown 3MF writer {writer!r}")
    _metadata(application, metadata)
    if writer in ("auto", "lib3mf"):
        try:
            write_3mf_lib3mf(path, vertices_mm, faces, name=name, application=application, metadata=metadata)
            return "lib3mf"
        except Exception:
            if writer == "lib3mf":
                raise
    write_3mf_xml(path, vertices_mm, faces, name=name, application=application, metadata=metadata)
    return "xml"


def read_3mf_metadata(path: str | Path) -> dict[str, str]:
    """Metadata entries of the root model part (independent of lib3mf)."""
    with zipfile.ZipFile(path) as z:
        data = z.read(_root_model_part(z))
    found: dict[str, str] = {}
    current: list[str] = []
    text: list[str] = []

    def start(name, attrs):
        if name == CORE_NS + " metadata":
            current.append(attrs.get("name", ""))
            text.clear()
        elif name == CORE_NS + " resources":
            raise StopIteration

    def chars(data):
        if current:
            text.append(data)

    def end(name):
        if name == CORE_NS + " metadata" and current:
            found[current.pop()] = "".join(text)

    p = expat.ParserCreate(namespace_separator=" ")
    p.StartElementHandler, p.CharacterDataHandler, p.EndElementHandler = start, chars, end
    try:
        p.Parse(data, True)
    except StopIteration:
        pass
    return found
