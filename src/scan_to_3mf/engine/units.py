"""Length units. Everything inside the engine is millimeters after loading."""

from __future__ import annotations

# key -> (3MF spelling, factor to millimeters, label)
UNITS: dict[str, tuple[str, float, str]] = {
    "mm": ("millimeter", 1.0, "millimeters"),
    "cm": ("centimeter", 10.0, "centimeters"),
    "m": ("meter", 1000.0, "meters"),
    "in": ("inch", 25.4, "inches"),
    "ft": ("foot", 304.8, "feet"),
    "um": ("micron", 0.001, "micrometers"),
}

_ALIASES = {
    "mm": "mm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "cm": "cm", "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm",
    "m": "m", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "in": "in", "inch": "in", "inches": "in", '"': "in",
    "ft": "ft", "foot": "ft", "feet": "ft", "'": "ft",
    "um": "um", "µm": "um", "μm": "um", "micron": "um", "microns": "um",
    "micrometer": "um", "micrometers": "um", "micrometre": "um", "micrometres": "um",
}

DEFAULT_UNIT = "mm"


class UnitError(ValueError):
    pass


def normalize_unit(unit: str) -> str:
    key = _ALIASES.get(str(unit).strip().lower())
    if key is None:
        raise UnitError(f"Unknown unit {unit!r}. Use one of: {', '.join(UNITS)}")
    return key


def to_mm(unit: str) -> float:
    return UNITS[normalize_unit(unit)][1]


def threemf_name(unit: str) -> str:
    return UNITS[normalize_unit(unit)][0]


def label(unit: str) -> str:
    return UNITS[normalize_unit(unit)][2]
