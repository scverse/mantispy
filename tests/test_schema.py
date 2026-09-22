import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mantispy._core.features import parse_feature_names
from mantispy._core.schema import SCHEMA_VERSION, get_resolution, migrate, stamp, validate


@pytest.fixture
def adata():
    names = ["Cells_AreaShape_Area", "Cells_Intensity_MeanIntensity_DNA"]
    obj = ad.AnnData(
        X=np.ones((4, 2), dtype=np.float32),
        obs=pd.DataFrame(
            {"Metadata_Plate": ["P1"] * 4, "Metadata_Well": ["A01", "A01", "A02", "A02"]},
            index=[f"c{i}" for i in range(4)],
        ),
        var=parse_feature_names(names, channels=["DNA"]),
    )
    stamp(obj, resolution="cell")
    return obj


def test_valid_object_passes(adata):
    report = validate(adata)
    assert report.ok and bool(report)
    assert str(report) == "valid"


def _drop_obs(column):
    def mutate(a):
        a.obs = a.obs.drop(columns=column)

    return mutate


def _set_wells(values):
    def mutate(a):
        obs = a.obs.copy()
        obs["Metadata_Well"] = values
        a.obs = obs

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_obs("Metadata_Plate"), "Metadata_Plate"),
        (_drop_obs("Metadata_Well"), "Metadata_Well"),
        (lambda a: setattr(a, "var", a.var.drop(columns="feature_group")), "feature_group"),
        (lambda a: a.uns.pop("mantispy"), "schema_version"),
        (lambda a: a.uns["mantispy"].update(schema_version="99.0"), "99.0"),
        (lambda a: setattr(a, "X", a.X.astype(np.float64)), "float32"),
        (_set_wells(["A01", "A01", "nope", "A02"]), "Metadata_Well"),
    ],
    ids=["no_plate", "no_well", "no_var_column", "no_uns", "wrong_version", "float64_X", "bad_well_name"],
)
def test_each_single_mutation_is_rejected(adata, mutate, message):
    mutate(adata)
    report = validate(adata)
    assert not report.ok
    assert any(message in error for error in report.errors), report.errors


def test_perturbation_resolution_does_not_require_plate_or_well():
    """A consensus profile has no plate or well; requiring them makes it unwritable."""
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Perturbation": ["a", "b"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]),
    )
    stamp(obj, resolution="perturbation")
    assert validate(obj).ok, validate(obj).errors


def test_resolution_defaults_to_cell(adata):
    del adata.uns["mantispy"]["resolution"]
    assert get_resolution(adata) == "cell"


def test_stamp_rejects_unknown_resolution(adata):
    with pytest.raises(ValueError, match="resolution must be one of"):
        stamp(adata, resolution="nonsense")


def test_raise_on_error(adata):
    adata.obs.drop(columns="Metadata_Plate", inplace=True)
    with pytest.raises(ValueError, match="Metadata_Plate"):
        validate(adata, raise_on_error=True)


def test_warns_but_passes_when_nothing_is_a_feature():
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P"] * 2, "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=parse_feature_names(["Metadata_Plate"]),
    )
    stamp(obj, resolution="cell")
    report = validate(obj)
    assert report.ok
    assert any("is_feature" in warning for warning in report.warnings)


def test_a_well_level_object_without_a_cell_count_is_warned_about(adata):
    """Regression test for #63: the absence is reported here, not left to fail in tl.cytotoxicity."""
    assert not any("Metadata_CellCount" in warning for warning in validate(adata).warnings)

    stamp(adata, resolution="well")
    report = validate(adata)
    assert report.ok
    assert any("Metadata_CellCount" in warning for warning in report.warnings)

    adata.obs["Metadata_CellCount"] = 10.0
    assert not any("Metadata_CellCount" in warning for warning in validate(adata).warnings)


def test_schema_version_is_stamped(adata):
    assert adata.uns["mantispy"]["schema_version"] == SCHEMA_VERSION


def test_published_spec_matches_the_constants():
    """spec/schema-1.0.json is the frozen contract; it must not drift from the code."""
    import json
    import pathlib

    from mantispy._core import schema as s

    spec = json.loads((pathlib.Path(__file__).parents[1] / f"spec/schema-{s.SCHEMA_VERSION}.json").read_text())
    assert spec["schema_version"] == s.SCHEMA_VERSION
    assert spec["required_obs"] == {k: list(v) for k, v in s.REQUIRED_OBS.items()}
    assert spec["reserved_obs"] == list(s.RESERVED_OBS)
    assert spec["required_var"] == list(s.REQUIRED_VAR)
    assert spec["uns_keys"] == list(s.UNS_KEYS)
    assert spec["resolutions"] == list(s.RESOLUTIONS)
    assert spec["optional_var"] == list(s.OPTIONAL_VAR)
    assert spec["uns_results"] == list(s.UNS_RESULTS)


def test_an_object_from_the_previous_schema_migrates(adata, tmp_path):
    """An object written under the previous schema migrates to the current one."""
    import mantispy as mt

    mt.io.write(adata, tmp_path / "old.h5ad")
    stale = mt.io.read(tmp_path / "old.h5ad")
    stale.uns["mantispy"]["schema_version"] = "0.1"
    assert not validate(stale).ok
    assert "mt.io.read" in str(validate(stale))

    migrate(stale)
    assert stale.uns["mantispy"]["schema_version"] == SCHEMA_VERSION
    assert validate(stale).ok


def test_migrate_refuses_a_version_it_does_not_know(adata):
    adata.uns["mantispy"]["schema_version"] = "9.9"
    with pytest.raises(ValueError, match="newer mantispy"):
        migrate(adata)


def test_read_migrates_an_older_file_by_default(adata, tmp_path):
    import anndata as ad

    import mantispy as mt

    mt.io.write(adata, tmp_path / "old.h5ad")
    stale = ad.read_h5ad(tmp_path / "old.h5ad")
    stale.uns["mantispy"]["schema_version"] = "0.1"
    stale.write_h5ad(tmp_path / "old.h5ad")

    assert mt.io.read(tmp_path / "old.h5ad").uns["mantispy"]["schema_version"] == SCHEMA_VERSION
    with pytest.raises(ValueError, match="migrate_schema=True"):
        mt.io.read(tmp_path / "old.h5ad", migrate_schema=False)


def test_an_object_with_no_features_does_not_validate():
    adata = ad.AnnData(
        np.empty((2, 0), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
    )
    stamp(adata, resolution="well")
    report = validate(adata)
    assert not report
    assert len(report.errors) == 1
    assert "no features" in report.errors[0]


def test_stamp_supplies_the_annotation_columns_var_does_not_carry():
    """Every tool that builds a new object stamps it, so filling here is what stops the next one from
    returning something validate rejects, the way tl.feature_signature did (#103)."""
    obj = ad.AnnData(
        np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
    )
    obj.var_names = ["emb_0", "emb_1", "emb_2"]
    assert not validate(obj).ok

    stamp(obj, resolution="well")
    assert validate(obj).ok, validate(obj).errors
    assert obj.var["is_feature"].all()
    assert obj.var["feature"].isna().all()


def test_stamp_does_not_touch_an_annotation_that_is_already_there():
    obj = ad.AnnData(np.ones((2, 1), dtype=np.float32), var=parse_feature_names(["Cells_AreaShape_Area"]))
    stamp(obj)
    assert obj.var["feature"].tolist() == ["Area"]


def test_stamp_can_be_asked_to_leave_var_alone():
    """io.write passes fill_var=False, so a damaged annotation is reported rather than repaired."""
    obj = ad.AnnData(np.ones((2, 1), dtype=np.float32), var=parse_feature_names(["Cells_AreaShape_Area"]))
    del obj.var["feature"]
    stamp(obj, fill_var=False)
    assert "feature" not in obj.var
    assert not validate(obj).ok
