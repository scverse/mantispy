import pytest

from mantispy._core.plate import (
    detect_plate_format,
    normalize_well,
    row_label,
    well_col,
    well_name,
    well_row,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A1", "A01"),
        ("a01", "A01"),
        ("A01", "A01"),
        ("  h12 ", "H12"),
        ("P24", "P24"),
        ("AF48", "AF48"),
        ("B_3", "B03"),
        ("B-3", "B03"),
    ],
)
def test_normalize_well(raw, expected):
    assert normalize_well(raw) == expected


@pytest.mark.parametrize("bad", ["", "1A", "A", "AAA1", "A0", "A123"])
def test_normalize_well_rejects(bad):
    with pytest.raises(ValueError, match="cannot parse well name"):
        normalize_well(bad)


@pytest.mark.parametrize(
    ("well", "row", "col"),
    [("A01", 0, 0), ("H12", 7, 11), ("AA01", 26, 0), ("AF48", 31, 47)],
)
def test_row_col_are_zero_based(well, row, col):
    assert (well_row(well), well_col(well)) == (row, col)


def test_row_label_round_trips_past_z():
    """1536-well plates have rows past Z, where chr(65 + row) alone gives non-letters."""
    assert [row_label(r) for r in (0, 25, 26, 31)] == ["A", "Z", "AA", "AF"]
    assert all(well_row(well_name(r, 0)) == r for r in range(32))


@pytest.mark.parametrize(
    ("wells", "expected"),
    [
        (["A01", "H12"], 96),
        (["A01", "P24"], 384),
        (["A01", "AF48"], 1536),
        (["A01", "B02"], 96),
    ],
)
def test_detect_plate_format(wells, expected):
    assert detect_plate_format(wells) == expected


def test_detect_plate_format_rejects_oversized():
    with pytest.raises(ValueError, match="no standard plate format"):
        detect_plate_format(["A01", "BZ99"])


def test_detect_plate_format_rejects_empty():
    with pytest.raises(ValueError, match="no wells given"):
        detect_plate_format([])
