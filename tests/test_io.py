import zipfile

import numpy as np
import pytest
from conftest import box, write_obj, write_ply, write_stl

from scan_to_3mf.engine import threemf
from scan_to_3mf.engine.meshio import MeshLoadError, load_mesh


@pytest.mark.parametrize("ascii", [False, True])
def test_stl_loads_as_triangle_soup(tmp_path, ascii):
    v, f = box()
    mesh = load_mesh(write_stl(tmp_path / "box.stl", v, f, ascii=ascii))
    assert mesh.fmt == "stl" and mesh.soup
    assert mesh.faces.shape == (12, 3)
    assert len(mesh.vertices) == 36  # STL repeats every corner
    assert mesh.file_unit is None


@pytest.mark.parametrize("ascii", [False, True])
def test_ply_loads_indexed(tmp_path, ascii):
    v, f = box()
    mesh = load_mesh(write_ply(tmp_path / "box.ply", v, f, ascii=ascii))
    assert not mesh.soup and len(mesh.vertices) == 8 and len(mesh.faces) == 12


def test_obj_uses_positions_only_and_triangulates_polygons(tmp_path):
    path = tmp_path / "seam.obj"
    path.write_text(
        "mtllib x.mtl\n"
        "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0.5 0.5 1\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0.5 0.5\nvn 0 0 1\n"
        "f 1/1/1 2/2/1 3/3/1 4/4/1\n"  # quad with UVs
        "f -5/4/1 -4/1/1 -1/2/1\n"  # negative indices, different UVs on shared corners
    )
    mesh = load_mesh(path)
    assert len(mesh.vertices) == 5  # UV seams do not duplicate vertices
    assert mesh.faces.tolist() == [[0, 1, 2], [0, 2, 3], [0, 1, 4]]
    assert any("polygons" in n for n in mesh.notes) and any("texture" in n for n in mesh.notes)


def test_point_cloud_is_rejected(tmp_path):
    path = tmp_path / "cloud.ply"
    path.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                    "property float z\nend_header\n0 0 0\n1 0 0\n0 1 0\n")
    with pytest.raises(MeshLoadError, match="point cloud"):
        load_mesh(path)


def test_unsupported_and_empty_files(tmp_path):
    with pytest.raises(MeshLoadError, match="Unsupported"):
        load_mesh(tmp_path / "part.step")
    empty = tmp_path / "empty.stl"
    empty.write_bytes(b"")
    with pytest.raises(MeshLoadError, match="empty"):
        load_mesh(empty)


def test_obj_with_bad_index_is_rejected(tmp_path):
    path = write_obj(tmp_path / "bad.obj", np.zeros((3, 3)), np.array([[0, 1, 5]]))
    with pytest.raises(MeshLoadError, match="do not exist"):
        load_mesh(path)


@pytest.mark.parametrize("writer", ["lib3mf", "xml"])
def test_3mf_round_trip_with_both_writers(tmp_path, writer):
    v, f = box()
    path = tmp_path / "box.3mf"
    used = threemf.write_3mf(path, v, f, writer=writer, metadata={"Designer": "Test", "LicenseTerms": "CC0"})
    assert used == writer
    assert threemf.read_3mf_unit_attribute(path) == "millimeter"
    for reader in (threemf.read_3mf_lib3mf, threemf.read_3mf_xml):
        content = reader(path)
        assert content.unit == "mm"
        assert np.array_equal(content.faces, f)
        assert np.abs(content.vertices - v).max() < 1e-5
    meta = threemf.read_3mf_metadata(path)
    assert meta["Designer"] == "Test" and meta["LicenseTerms"] == "CC0" and meta["Application"].startswith("scan-to-3mf")


def test_unknown_metadata_name_is_rejected(tmp_path):
    v, f = box()
    with pytest.raises(ValueError, match="metadata"):
        threemf.write_3mf(tmp_path / "x.3mf", v, f, metadata={"Author": "me"})


def _write_inch_3mf_with_transforms(path):
    model = """<?xml version="1.0" encoding="UTF-8"?>
<model unit="inch" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
 <resources>
  <object id="1" type="model"><mesh>
   <vertices><vertex x="0" y="0" z="0"/><vertex x="1" y="0" z="0"/><vertex x="0" y="1" z="0"/><vertex x="0" y="0" z="1"/></vertices>
   <triangles><triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="1" v3="3"/><triangle v1="0" v2="3" v3="2"/><triangle v1="1" v2="2" v3="3"/></triangles>
  </mesh></object>
  <object id="2" type="model"><components><component objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/></components></object>
 </resources>
 <build><item objectid="2" transform="2 0 0 0 2 0 0 0 2 0 5 0"/></build>
</model>"""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Target="/3D/3dmodel.model" Id="r0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)


def test_3mf_unit_components_and_build_transforms(tmp_path):
    path = tmp_path / "inch.3mf"
    _write_inch_3mf_with_transforms(path)
    a, b = threemf.read_3mf_lib3mf(path), threemf.read_3mf_xml(path)
    assert a.unit == b.unit == "in"
    assert np.allclose(a.vertices, b.vertices, atol=1e-5)
    # component moves x by 10, then the build item scales by 2 and moves y by 5
    assert a.vertices.min(axis=0).tolist() == pytest.approx([20.0, 5.0, 0.0])
    assert a.vertices.max(axis=0).tolist() == pytest.approx([22.0, 7.0, 2.0])
    mesh = load_mesh(path)
    assert mesh.file_unit == "in" and len(mesh.faces) == 4
