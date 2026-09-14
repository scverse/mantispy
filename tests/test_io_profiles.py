from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest

import mantispy as mt

FRAME = pd.DataFrame(
    {
        "Metadata_Plate": ["P1", "P1"],
        "Metadata_Well": ["A01", "A02"],
        "ImageNumber": [1, 2],
        "Cells_AreaShape_Area": [10.0, 20.0],
        "Cells_Correlation_Correlation_DNA_RNA": [0.5, -999.0],
    }
)


def _write(path: Path, frame: pd.DataFrame, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.with_suffix(suffix)
    frame.to_parquet(file) if suffix == ".parquet" else frame.to_csv(file, index=False)
    return file


@pytest.fixture
def profiles(tmp_path: Path) -> list[Path]:
    """Three plates of one profile, the middle one with no rows, as a partly failed run leaves it."""
    return [
        _write(tmp_path / name / "profile", rows, ".csv")
        for name, rows in (("plate1", FRAME), ("plate2", FRAME.iloc[:0]), ("plate3", FRAME))
    ]


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".parquet"])
def test_numeric_columns_become_features_and_the_rest_metadata(tmp_path: Path, suffix: str) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME, suffix))

    assert adata.shape == (2, 3)
    assert adata.var_names.tolist() == [
        "ImageNumber",
        "Cells_AreaShape_Area",
        "Cells_Correlation_Correlation_DNA_RNA",
    ]
    assert adata.obs.columns.tolist() == ["Plate", "Well"]
    assert np.asarray(adata.X).dtype == np.float32


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, ["ImageNumber", "Cells_AreaShape_Area", "Cells_Correlation_Correlation_DNA_RNA"]),
        ({"metadata_columns": ("ImageNumber",)}, ["Cells_AreaShape_Area", "Cells_Correlation_Correlation_DNA_RNA"]),
        ({"metadata_prefixes": ("Cells_",)}, ["ImageNumber"]),
    ],
)
def test_which_columns_are_features(tmp_path: Path, kwargs: dict, expected: list[str]) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME, ".csv"), **kwargs)

    assert adata.var_names.tolist() == expected


@pytest.mark.parametrize("sentinels", [-999.0, (-999.0,), [-999.0, -998.0]])
def test_sentinels_become_nan(tmp_path: Path, sentinels) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME, ".csv"), sentinels=sentinels)

    missing = np.isnan(np.asarray(adata[:, "Cells_Correlation_Correlation_DNA_RNA"].X))

    assert missing.tolist() == [[False], [True]]


def test_metadata_prefixes_are_stripped_longest_first(tmp_path: Path) -> None:
    frame = FRAME.rename(columns={"Metadata_Well": "Image_Metadata_Well"})
    adata = mt.io.read_profiles(_write(tmp_path / "profile", frame, ".csv"))

    assert "Well" in adata.obs.columns


def test_features_are_annotated_with_what_their_names_encode(tmp_path: Path) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME, ".csv"))

    var = cast("pd.DataFrame", adata.var).loc["Cells_Correlation_Correlation_DNA_RNA"]
    assert (var["compartment"], var["family"]) == ("cells", "correlation")
    assert (var["channel"], var["channel_2"], var["n_channels"]) == ("dna", "rna", 2)
    assert cast("pd.DataFrame", adata.var).loc["Cells_AreaShape_Area", "n_channels"] == 0


def test_annotation_can_be_turned_off(tmp_path: Path) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME, ".csv"), annotate_features=False)

    assert adata.var.columns.empty


def test_index_columns_name_the_observations(tmp_path: Path) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME, ".csv"), index_columns=("Plate", "Well"))

    assert adata.obs_names.tolist() == ["P1:A01", "P1:A02"]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"index_columns": ("Plate",)}, ValueError, "do not identify observations uniquely"),
        ({"index_columns": ("Nope",)}, KeyError, "index columns not in metadata"),
        ({"metadata_columns": ("Nope",)}, KeyError, "metadata columns not in data"),
    ],
)
def test_bad_column_arguments_say_what_is_wrong(tmp_path: Path, kwargs: dict, error, match: str) -> None:
    with pytest.raises(error, match=match):
        mt.io.read_profiles(_write(tmp_path / "profile", FRAME, ".csv"), **kwargs)


def test_no_files_is_refused() -> None:
    with pytest.raises(ValueError, match="no profile files given"):
        mt.io.read_profiles([])


def test_a_file_with_no_rows_does_not_erase_the_features(profiles: list[Path]) -> None:
    """A header-only file reads as all-object and would drag the other files' dtypes with it through concat."""
    adata = mt.io.read_profiles(profiles)

    assert adata.shape == (4, 3)
    assert np.asarray(adata[:, "Cells_AreaShape_Area"].X).ravel().tolist() == [10.0, 20.0, 10.0, 20.0]


def test_all_files_empty_yields_no_observations(profiles: list[Path]) -> None:
    # a header-only file carries no dtype information, so there is no way to tell which columns are features
    assert mt.io.read_profiles([profiles[1]]).n_obs == 0


def test_path_columns_stay_aligned_when_a_file_is_dropped(profiles: list[Path]) -> None:
    adata = mt.io.read_profiles(profiles, path_columns={"Metadata_Source": 1})

    assert adata.obs["Source"].tolist() == ["plate1", "plate1", "plate3", "plate3"]


@pytest.mark.parametrize(
    ("on_column_mismatch", "expected"),
    [("intersect", ["ImageNumber", "Cells_AreaShape_Area"]), ("raise", None)],
)
def test_files_that_disagree_on_columns(
    tmp_path: Path, on_column_mismatch: Literal["raise", "intersect"], expected: list[str] | None
) -> None:
    paths = [
        _write(tmp_path / "plate1" / "profile", FRAME, ".csv"),
        _write(tmp_path / "plate2" / "profile", FRAME.drop(columns="Cells_Correlation_Correlation_DNA_RNA"), ".csv"),
    ]
    if expected is None:
        with pytest.raises(ValueError, match="files disagree on columns"):
            mt.io.read_profiles(paths, on_column_mismatch=on_column_mismatch)
        return
    assert mt.io.read_profiles(paths, on_column_mismatch=on_column_mismatch).var_names.tolist() == expected


def test_files_that_share_no_columns_are_refused(tmp_path: Path) -> None:
    paths = [
        _write(tmp_path / "plate1" / "profile", FRAME, ".csv"),
        _write(tmp_path / "plate2" / "profile", pd.DataFrame({"Other": [1.0]}), ".csv"),
    ]
    with pytest.raises(ValueError, match="files share no columns"):
        mt.io.read_profiles(paths, on_column_mismatch="intersect")
