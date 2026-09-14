from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from _testdata import write_export_to_spreadsheet

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
FEATURES = ["ImageNumber", "Cells_AreaShape_Area", "Cells_Correlation_Correlation_DNA_RNA"]


def _write(path: Path, frame: pd.DataFrame, suffix: str = ".csv") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file = path.with_suffix(suffix)
    frame.to_parquet(file) if suffix == ".parquet" else frame.to_csv(file, index=False)
    return file


@pytest.fixture
def profiles(tmp_path: Path) -> list[Path]:
    """Three plates of one profile, the middle one with no rows, as a partly failed run leaves it."""
    return [
        _write(tmp_path / name / "profile", rows)
        for name, rows in (("plate1", FRAME), ("plate2", FRAME.iloc[:0]), ("plate3", FRAME))
    ]


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".parquet"])
def test_numeric_columns_become_features_and_the_rest_metadata(tmp_path: Path, suffix: str) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME, suffix), index_columns=("Plate", "Well"))

    assert adata.shape == (2, 3)
    assert adata.var_names.tolist() == FEATURES
    assert adata.obs.columns.tolist() == ["Plate", "Well"]
    assert adata.obs_names.tolist() == ["P1:A01", "P1:A02"]
    assert np.asarray(adata.X).dtype == np.float32


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, FEATURES),
        ({"metadata_columns": ("ImageNumber",)}, FEATURES[1:]),
        ({"metadata_prefixes": ("Cells_",)}, ["ImageNumber"]),
    ],
)
def test_which_columns_are_features(tmp_path: Path, kwargs: dict, expected: list[str]) -> None:
    assert mt.io.read_profiles(_write(tmp_path / "profile", FRAME), **kwargs).var_names.tolist() == expected


def test_the_longest_matching_metadata_prefix_is_stripped(tmp_path: Path) -> None:
    frame = FRAME.rename(columns={"Metadata_Well": "Image_Metadata_Well"})

    assert "Well" in mt.io.read_profiles(_write(tmp_path / "profile", frame)).obs.columns


@pytest.mark.parametrize("sentinels", [-999.0, (-999.0,), [-999.0, -998.0]])
def test_sentinels_become_nan(tmp_path: Path, sentinels) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME), sentinels=sentinels)

    assert np.isnan(np.asarray(adata[:, FEATURES[2]].X)).tolist() == [[False], [True]]


def test_features_are_annotated_with_what_their_names_encode(tmp_path: Path) -> None:
    adata = mt.io.read_profiles(_write(tmp_path / "profile", FRAME))
    var = cast("pd.DataFrame", adata.var)

    assert (var.loc[FEATURES[2], "compartment"], var.loc[FEATURES[2], "family"]) == ("cells", "correlation")
    assert (var.loc[FEATURES[2], "channel"], var.loc[FEATURES[2], "channel_2"]) == ("dna", "rna")
    assert var.loc[FEATURES[2], "n_channels"] == 2
    assert var.loc["Cells_AreaShape_Area", "n_channels"] == 0
    assert mt.io.read_profiles(_write(tmp_path / "profile", FRAME), annotate_features=False).var.columns.empty


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
        mt.io.read_profiles(_write(tmp_path / "profile", FRAME), **kwargs)


def test_files_with_no_rows(profiles: list[Path]) -> None:
    """A header-only file reads as all-object and would drag the other files' dtypes with it through concat."""
    adata = mt.io.read_profiles(profiles, path_columns={"Metadata_Source": 1})

    assert adata.shape == (4, 3)
    assert np.asarray(adata[:, "Cells_AreaShape_Area"].X).ravel().tolist() == [10.0, 20.0, 10.0, 20.0]
    assert adata.obs["Source"].tolist() == ["plate1", "plate1", "plate3", "plate3"]
    # a header-only file carries no dtype information, so there is no way to tell which columns are features
    assert mt.io.read_profiles([profiles[1]]).n_obs == 0


@pytest.mark.parametrize(
    ("second", "expected", "match"),
    [
        (FRAME.drop(columns=FEATURES[2]), FEATURES[:2], None),
        (FRAME, FEATURES, None),
        (pd.DataFrame({"Other": [1.0]}), None, "files share no columns"),
    ],
    ids=["intersect", "already-agree", "nothing-shared"],
)
def test_files_that_disagree_on_columns(
    tmp_path: Path, second: pd.DataFrame, expected: list[str] | None, match: str | None
) -> None:
    paths = [_write(tmp_path / "plate1" / "profile", FRAME), _write(tmp_path / "plate2" / "profile", second)]

    if match is not None:
        with pytest.raises(ValueError, match=match):
            mt.io.read_profiles(paths, on_column_mismatch="intersect")
        return
    assert mt.io.read_profiles(paths, on_column_mismatch="intersect").var_names.tolist() == expected


@pytest.mark.parametrize(
    ("paths", "on_column_mismatch", "match"),
    [([], "raise", "no profile files given"), (["a", "b"], "raise", "files disagree on columns")],
    ids=["no-files", "mismatch-raises"],
)
def test_refused_inputs(
    tmp_path: Path, paths: list[str], on_column_mismatch: Literal["raise", "intersect"], match: str
) -> None:
    if paths:
        paths = [
            str(_write(tmp_path / "plate1" / "profile", FRAME)),
            str(_write(tmp_path / "plate2" / "profile", FRAME.drop(columns=FEATURES[2]))),
        ]

    with pytest.raises(ValueError, match=match):
        mt.io.read_profiles(paths, on_column_mismatch=on_column_mismatch)


@pytest.mark.parametrize("prefix", ["", "MyExpt_"], ids=["unprefixed", "prefixed"])
def test_an_export_to_spreadsheet_directory_joins_its_objects(tmp_path: Path, prefix: str) -> None:
    directory = write_export_to_spreadsheet(tmp_path / "run", prefix=prefix)

    adata = mt.io.read_profiles(directory)

    assert adata.n_obs == 4
    assert "Cells_AreaShape_Area" in adata.var_names
    assert "Nuclei_AreaShape_Area" in adata.var_names
    assert "Cytoplasm_AreaShape_Area" in adata.var_names
    assert not [name for name in adata.var_names if "ObjectNumber" in name or name.startswith("Parent_")]

    cells = np.asarray(adata[:, "Cells_AreaShape_Area"].X).ravel()
    nuclei = np.asarray(adata[:, "Nuclei_AreaShape_Area"].X).ravel()
    assert cells.tolist() == [100.0, 200.0, 300.0, 400.0]
    assert nuclei.tolist() == [40.0, 30.0, 20.0, 10.0]

    obs = cast("pd.DataFrame", adata.obs)
    assert obs["Plate"].tolist() == ["BR00000001"] * 4
    assert obs[["ImageNumber", "ObjectNumber"]].iloc[0].tolist() == [1, 1]
    assert adata.uns["image_table"]["ImageQuality_FocusScore_DNA"].tolist() == [0.42]
    assert not [name for name in adata.var_names if name.startswith("ImageQuality_")]


def test_export_directories_stack_like_files(tmp_path: Path) -> None:
    sites = [write_export_to_spreadsheet(tmp_path / f"site{n}", n_cells=n + 1) for n in (1, 2)]

    adata = mt.io.read_profiles(sites, path_columns={"Metadata_Site": 1})

    assert adata.n_obs == 5
    assert cast("pd.DataFrame", adata.obs)["Site"].tolist() == ["site1"] * 2 + ["site2"] * 3
    assert adata.uns["image_table"].shape[0] == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"primary_object": "Nope"}, "no Nope.csv"), ({"objects": ["Nope"]}, r"no \['Nope'\]")],
)
def test_an_export_directory_that_cannot_be_read(tmp_path: Path, kwargs: dict, match: str) -> None:
    directory = write_export_to_spreadsheet(tmp_path / "run")

    with pytest.raises(FileNotFoundError, match=match):
        mt.io.read_profiles(directory, **kwargs)
