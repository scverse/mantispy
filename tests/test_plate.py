import pytest

from mantispy._core.plate import (
    detect_plate_format,
    normalize_well,
    row_label,
    well_name,
    well_row,
)


@pytest.mark.parametrize("bad", ["", "1A", "A", "AAA1", "A0", "A123"])
def test_normalize_well_rejects(bad):
    with pytest.raises(ValueError, match="cannot parse well name"):
        normalize_well(bad)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("A/1/0", "A01"), ("A/1", "A01"), ("B/12/3", "B12"), ("h/2/15", "H02")],
)
def test_normalize_well_accepts_row_column_field(given, expected):
    """CZI optical-pooled-screen releases write wells as row/column/field; the field is dropped."""
    assert normalize_well(given) == expected


def test_row_label_round_trips_past_z():
    """1536-well plates have rows past Z, where chr(65 + row) alone gives non-letters."""
    assert [row_label(r) for r in (0, 25, 26, 31)] == ["A", "Z", "AA", "AF"]
    assert all(well_row(well_name(r, 0)) == r for r in range(32))


def test_detect_plate_format_rejects_oversized():
    with pytest.raises(ValueError, match="no standard plate format"):
        detect_plate_format(["A01", "BZ99"])


def test_detect_plate_format_rejects_empty():
    with pytest.raises(ValueError, match="no wells given"):
        detect_plate_format([])
