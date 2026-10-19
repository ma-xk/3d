import pytest

from scan_to_3mf.engine.pipeline import unit_hints
from scan_to_3mf.engine.units import UnitError, normalize_unit, threemf_name, to_mm


@pytest.mark.parametrize("text,key", [
    ("mm", "mm"), ("Millimeters", "mm"), ("centimetre", "cm"), ("M", "m"), ("inch", "in"),
    ('"', "in"), ("feet", "ft"), ("µm", "um"), ("micron", "um"),
])
def test_normalize_unit_aliases(text, key):
    assert normalize_unit(text) == key


def test_unknown_unit_is_rejected():
    with pytest.raises(UnitError):
        normalize_unit("furlong")


@pytest.mark.parametrize("unit,factor", [("mm", 1), ("cm", 10), ("m", 1000), ("in", 25.4), ("ft", 304.8), ("um", 0.001)])
def test_factors_to_mm(unit, factor):
    assert to_mm(unit) == pytest.approx(factor)


def test_3mf_unit_names():
    assert threemf_name("in") == "inch"
    assert threemf_name("um") == "micron"


def test_hint_for_a_model_in_meters_read_as_mm():
    hints = unit_hints([0.156, 0.154, 0.12], "mm")
    assert len(hints) == 1 and "meters would make it 156 mm" in hints[0]


def test_no_hint_for_plausible_size():
    assert unit_hints([82.3, 49.6, 57.8], "mm") == []


def test_hint_for_huge_model():
    assert "unusually large" in unit_hints([5000.0, 10.0, 10.0], "mm")[0]
