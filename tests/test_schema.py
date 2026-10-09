import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mantispy._core.features import parse_feature_names
from mantispy._core.schema import (
    FEATURE_KINDS,
    REQUIRED_UNS,
    RESOLUTIONS,
    SCHEMA_STATUS,
    SCHEMA_VERSION,
    ensure_object_identity,
    get_resolution,
    migrate,
    stamp,
    validate,
)


@pytest.fixture
def adata():
    names = ["Cells_AreaShape_Area", "Cells_Intensity_MeanIntensity_DNA"]
    obs = pd.DataFrame(
        {"Metadata_Plate": ["P1"] * 4, "Metadata_Well": ["A01", "A01", "A02", "A02"]},
        index=[f"c{i}" for i in range(4)],
    )
    ensure_object_identity(obs, "Cells")
    obj = ad.AnnData(X=np.ones((4, 2), dtype=np.float32), obs=obs, var=parse_feature_names(names, channels=["DNA"]))
    stamp(obj, resolution="object")
    return obj


def _aggregate(grouped_by=("Metadata_Gene",), obs=None):
    obs = obs if obs is not None else pd.DataFrame({"Metadata_Gene": ["a", "b"]}, index=["0", "1"])
    obj = ad.AnnData(
        X=np.ones((len(obs), 1), dtype=np.float32),
        obs=obs,
        var=parse_feature_names(["Cells_AreaShape_Area"]),
    )
    stamp(obj, resolution="aggregate", grouped_by=list(grouped_by))
    return obj


# --- the 2.0 core contract ------------------------------------------------


def test_valid_object_passes(adata):
    report = validate(adata)
    assert report.ok and bool(report)
    assert str(report) == "valid"


def test_schema_version_is_2_0_experimental(adata):
    assert SCHEMA_VERSION == "2.0"
    assert SCHEMA_STATUS == "experimental"
    assert adata.uns["mantispy"]["schema_version"] == "2.0"
    assert adata.uns["mantispy"]["schema_status"] == "experimental"


def test_stamp_reserves_history(adata):
    assert adata.uns["mantispy"]["history"] == []


def test_core_namespace_has_every_required_key(adata):
    for key in REQUIRED_UNS:
        assert key in adata.uns["mantispy"], key


# --- the resolution vocabulary is exactly object/well/aggregate -----------


def test_allowed_resolutions_are_object_well_aggregate():
    assert RESOLUTIONS == ("object", "well", "aggregate")


def test_object_resolution_groups_by_nothing(adata):
    assert get_resolution(adata) == "object"
    assert adata.uns["mantispy"]["grouped_by"] == []


def test_well_resolution_defaults_grouped_by_to_plate_and_well(adata):
    stamp(adata, resolution="well")
    assert adata.uns["mantispy"]["grouped_by"] == ["Metadata_Plate", "Metadata_Well"]


def test_aggregate_resolution_stores_arbitrary_grouping():
    obj = _aggregate(grouped_by=["Metadata_Gene"])
    assert get_resolution(obj) == "aggregate"
    assert obj.uns["mantispy"]["grouped_by"] == ["Metadata_Gene"]
    assert validate(obj).ok, validate(obj).errors


def test_aggregate_resolution_does_not_require_plate_or_well():
    """An aggregate profile has no plate or well; requiring them makes it unwritable."""
    assert validate(_aggregate()).ok


@pytest.mark.parametrize("old", ["cell", "perturbation"])
def test_old_resolution_values_are_rejected(adata, old):
    """The 1.0 vocabulary is not valid under schema 2.0; stamp refuses it and validate flags it."""
    with pytest.raises(ValueError, match="resolution must be one of"):
        stamp(adata, resolution=old)
    adata.uns["mantispy"]["resolution"] = old
    report = validate(adata)
    assert not report.ok
    assert any("resolution" in error for error in report.errors), report.errors


# --- resolution must be explicit (no silent default) ----------------------


def test_missing_resolution_is_rejected_by_validate(adata):
    del adata.uns["mantispy"]["resolution"]
    report = validate(adata)
    assert not report.ok
    assert any("resolution" in error for error in report.errors), report.errors


def test_get_resolution_raises_when_absent(adata):
    del adata.uns["mantispy"]["resolution"]
    with pytest.raises((KeyError, ValueError), match="resolution"):
        get_resolution(adata)


def test_get_resolution_returns_the_default_when_absent(adata):
    """Read-only consumers pass default='object' to tolerate an unstamped object instead of crashing."""
    del adata.uns["mantispy"]["resolution"]
    assert get_resolution(adata, default="object") == "object"


def test_stamp_without_resolution_on_a_fresh_object_raises():
    obj = ad.AnnData(np.ones((1, 1), dtype=np.float32), var=parse_feature_names(["Cells_AreaShape_Area"]))
    with pytest.raises(ValueError, match="resolution"):
        stamp(obj)


def test_stamp_rejects_unknown_resolution(adata):
    with pytest.raises(ValueError, match="resolution must be one of"):
        stamp(adata, resolution="nonsense")


def test_stamp_aggregate_without_grouped_by_raises():
    obj = ad.AnnData(np.ones((1, 1), dtype=np.float32), var=parse_feature_names(["Cells_AreaShape_Area"]))
    with pytest.raises(ValueError, match="grouped_by"):
        stamp(obj, resolution="aggregate")


def test_a_failed_restamp_leaves_the_object_untouched(adata):
    """Re-stamping to aggregate without a grouping must not half-rewrite a valid object-level object."""
    with pytest.raises(ValueError, match="grouped_by"):
        stamp(adata, resolution="aggregate")
    assert adata.uns["mantispy"]["resolution"] == "object"
    assert adata.uns["mantispy"]["grouped_by"] == []
    assert validate(adata).ok


# --- grouped_by validation -----------------------------------------------


def test_object_resolution_rejects_a_non_empty_grouped_by(adata):
    adata.uns["mantispy"]["grouped_by"] = ["Metadata_Plate"]
    report = validate(adata)
    assert not report.ok
    assert any("grouped_by" in error for error in report.errors), report.errors


def test_grouped_by_must_be_present(adata):
    del adata.uns["mantispy"]["grouped_by"]
    report = validate(adata)
    assert not report.ok
    assert any("grouped_by" in error for error in report.errors), report.errors


def test_grouped_by_columns_must_exist_in_obs():
    obj = _aggregate(grouped_by=["Metadata_Gene"])
    obj.uns["mantispy"]["grouped_by"] = ["Metadata_DoesNotExist"]
    report = validate(obj)
    assert not report.ok
    assert any("Metadata_DoesNotExist" in error for error in report.errors), report.errors


def test_aggregate_grouped_by_must_be_non_empty():
    obj = _aggregate()
    obj.uns["mantispy"]["grouped_by"] = []
    report = validate(obj)
    assert not report.ok
    assert any("grouped_by" in error for error in report.errors), report.errors


@pytest.mark.parametrize(
    "grouped_by",
    [
        ["Metadata_MOA"],
        ["Metadata_sgRNA"],
        ["Metadata_Compound", "Metadata_Concentration"],
    ],
)
def test_arbitrary_aggregate_groupings_are_valid(grouped_by):
    """Biological modality lives in grouped_by, not in new resolution values."""
    obs = pd.DataFrame({column: ["x", "y"] for column in grouped_by}, index=["0", "1"])
    assert validate(_aggregate(grouped_by=grouped_by, obs=obs)).ok


# --- feature_kind is required and non-null --------------------------------


def test_feature_kind_is_a_required_var_column(adata):
    assert "feature_kind" in adata.var.columns
    adata.var = adata.var.drop(columns="feature_kind")
    report = validate(adata)
    assert not report.ok
    assert any("feature_kind" in error for error in report.errors), report.errors


def test_feature_kind_vocabulary_is_enforced(adata):
    var = adata.var.copy()
    var["feature_kind"] = var["feature_kind"].astype(str)
    var.loc[var.index[0], "feature_kind"] = "bogus"
    adata.var = var
    report = validate(adata)
    assert not report.ok
    assert any("feature_kind" in error for error in report.errors), report.errors


def test_a_null_feature_kind_is_rejected(adata):
    """Every feature must declare a kind; a null is as much an error as an out-of-vocabulary value."""
    var = adata.var.copy()
    var["feature_kind"] = var["feature_kind"].astype("object")
    var.loc[:, "feature_kind"] = None
    adata.var = var
    report = validate(adata)
    assert not report.ok
    assert any("feature_kind" in error for error in report.errors), report.errors


@pytest.mark.parametrize("kind", list(FEATURE_KINDS))
def test_each_feature_kind_passes(adata, kind):
    var = adata.var.copy()
    var["feature_kind"] = var["feature_kind"].astype("object")
    var.loc[:, "feature_kind"] = kind
    adata.var = var
    assert validate(adata).ok, validate(adata).errors


def test_parsed_measurements_are_tagged_measurement():
    var = parse_feature_names(["Cells_AreaShape_Area", "Cells_Intensity_MeanIntensity_DNA"], channels=["DNA"])
    assert (var["feature_kind"] == "measurement").all()


# --- control semantics (spec 10) ------------------------------------------


def _with_controls(adata, types):
    """Attach a Metadata_Control_Type column and the boolean it implies."""
    adata.obs["Metadata_Control_Type"] = list(types)
    adata.obs["Metadata_Control"] = adata.obs["Metadata_Control_Type"] != "treatment"
    return adata


def test_a_well_formed_control_pair_passes(adata):
    _with_controls(adata, ["negcon", "poscon", "empty", "treatment"])
    assert validate(adata).ok, validate(adata).errors


def test_control_flag_without_a_type_is_rejected(adata):
    adata.obs["Metadata_Control"] = [True, False, True, False]
    report = validate(adata)
    assert not report.ok
    assert any("Metadata_Control_Type" in error for error in report.errors), report.errors


def test_a_control_type_without_the_flag_is_rejected(adata):
    adata.obs["Metadata_Control_Type"] = ["negcon", "treatment", "treatment", "treatment"]
    report = validate(adata)
    assert not report.ok
    assert any("Metadata_Control" in error for error in report.errors), report.errors


def test_an_out_of_vocabulary_control_type_is_rejected(adata):
    _with_controls(adata, ["negcon", "trt", "treatment", "treatment"])
    report = validate(adata)
    assert not report.ok
    assert any("Metadata_Control_Type" in error for error in report.errors), report.errors


def test_the_control_flag_must_agree_with_the_type(adata):
    adata.obs["Metadata_Control_Type"] = ["negcon", "treatment", "treatment", "treatment"]
    adata.obs["Metadata_Control"] = [False, False, False, False]  # negcon row wrongly flagged non-control
    report = validate(adata)
    assert not report.ok
    assert any("disagrees" in error for error in report.errors), report.errors


# --- io.stamp: explicit kind, never guess ---------------------------------


def _embedding_matrix():
    return ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=pd.DataFrame(index=["dim0", "dim1", "dim2"]),
    )


def test_io_stamp_accepts_an_explicit_uniform_feature_kind():
    import mantispy as mt

    obj = _embedding_matrix()
    mt.io.stamp(obj, resolution="well", feature_kind="embedding")
    assert (obj.var["feature_kind"] == "embedding").all()
    assert mt.io.validate(obj).ok, mt.io.validate(obj).errors


def test_io_stamp_refuses_an_object_whose_feature_kind_is_unknown():
    import mantispy as mt

    with pytest.raises(ValueError, match="feature_kind"):
        mt.io.stamp(_embedding_matrix(), resolution="well")


def test_io_stamp_rejects_an_out_of_vocabulary_feature_kind():
    import mantispy as mt

    with pytest.raises(ValueError, match="feature_kind must be one of"):
        mt.io.stamp(_embedding_matrix(), resolution="well", feature_kind="nonsense")


def test_io_stamp_preserves_a_complete_feature_kind():
    import mantispy as mt

    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]),
    )
    mt.io.stamp(obj, resolution="well")
    assert (obj.var["feature_kind"] == "measurement").all()
    assert mt.io.validate(obj).ok, mt.io.validate(obj).errors


def test_public_stamping_creates_a_valid_2_0_object():
    import mantispy as mt

    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]),
    )
    mt.io.stamp(obj, resolution="well")
    report = mt.io.validate(obj)
    assert report.ok, report.errors
    assert obj.uns["mantispy"]["schema_version"] == "2.0"
    assert obj.uns["mantispy"]["schema_status"] == "experimental"
    assert obj.uns["mantispy"]["grouped_by"] == ["Metadata_Plate", "Metadata_Well"]
    assert obj.uns["mantispy"]["history"] == []


# --- display-only fallback must not weaken validation ---------------------


def test_display_fallback_does_not_weaken_validation_of_a_stored_object():
    """pl.qc may tolerate an unstamped object, but validate must still reject one that lacks resolution."""
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]),
    )
    # No stamp: a display helper may read get_resolution(default="object"), but the object is not valid.
    assert get_resolution(obj, default="object") == "object"
    report = validate(obj)
    assert not report.ok
    assert any("resolution" in error for error in report.errors), report.errors


# --- remaining single-mutation rejections --------------------------------


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
        (lambda a: a.uns["mantispy"].pop("schema_status"), "schema_status"),
        (lambda a: setattr(a, "X", a.X.astype(np.float64)), "float32"),
        (_set_wells(["A01", "A01", "nope", "A02"]), "Metadata_Well"),
    ],
    ids=[
        "no_plate",
        "no_well",
        "no_var_column",
        "no_uns",
        "wrong_version",
        "no_status",
        "float64_X",
        "bad_well_name",
    ],
)
def test_each_single_mutation_is_rejected(adata, mutate, message):
    mutate(adata)
    report = validate(adata)
    assert not report.ok
    assert any(message in error for error in report.errors), report.errors


def test_raise_on_error(adata):
    adata.obs.drop(columns="Metadata_Plate", inplace=True)
    with pytest.raises(ValueError, match="Metadata_Plate"):
        validate(adata, raise_on_error=True)


def test_warns_but_passes_when_nothing_is_a_feature():
    obs = pd.DataFrame({"Metadata_Plate": ["P"] * 2, "Metadata_Well": ["A01", "A02"]}, index=["0", "1"])
    ensure_object_identity(obs, "Cells")
    obj = ad.AnnData(X=np.ones((2, 1), dtype=np.float32), obs=obs, var=parse_feature_names(["Metadata_Plate"]))
    stamp(obj, resolution="object")
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


# --- public spec mirror + migration --------------------------------------


def test_published_spec_matches_the_constants():
    """spec/schema-2.0.json is the published contract; it must not drift from the code."""
    import json
    import pathlib

    from mantispy._core import schema as s

    spec = json.loads((pathlib.Path(__file__).parents[1] / f"spec/schema-{s.SCHEMA_VERSION}.json").read_text())
    assert spec["schema_version"] == s.SCHEMA_VERSION
    assert spec["status"] == s.SCHEMA_STATUS
    assert spec["required_obs"] == {k: list(v) for k, v in s.REQUIRED_OBS.items()}
    assert spec["reserved_obs"] == list(s.RESERVED_OBS)
    assert spec["required_var"] == list(s.REQUIRED_VAR)
    assert spec["uns_keys"] == list(s.UNS_KEYS)
    assert spec["required_uns"] == list(s.REQUIRED_UNS)
    assert spec["feature_kinds"] == list(s.FEATURE_KINDS)
    assert spec["control_types"] == list(s.CONTROL_TYPES)
    assert spec["resolutions"] == list(s.RESOLUTIONS)
    assert spec["optional_var"] == list(s.OPTIONAL_VAR)
    assert spec["uns_results"] == list(s.UNS_RESULTS)
    assert spec["migrates_from"] == list(s.SUPPORTED_VERSIONS)
    assert spec["x_dtype"] == "float32"


def test_an_object_from_the_previous_schema_migrates(adata, tmp_path):
    import mantispy as mt

    mt.io.write(adata, tmp_path / "old.h5ad")
    stale = mt.io.read(tmp_path / "old.h5ad")
    stale.uns["mantispy"]["schema_version"] = "0.1"
    assert not validate(stale).ok
    assert "mt.io.read" in str(validate(stale))

    migrate(stale)
    assert stale.uns["mantispy"]["schema_version"] == SCHEMA_VERSION
    assert validate(stale).ok


def test_migrate_fills_the_new_core_fields_on_a_1_0_object():
    """A genuine 1.0 object lacks grouped_by, history, status and feature_kind; migrate supplies them."""
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "well"}
    migrate(obj)
    store = obj.uns["mantispy"]
    assert store["schema_version"] == SCHEMA_VERSION
    assert store["schema_status"] == SCHEMA_STATUS
    assert store["history"] == []
    assert store["grouped_by"] == ["Metadata_Plate", "Metadata_Well"]
    assert (obj.var["feature_kind"] == "measurement").all()
    assert validate(obj).ok, validate(obj).errors


def test_migrate_derives_control_type_from_a_legacy_control_flag():
    """Under 1.0 Metadata_Control==True meant negcon; migrate moves that into Metadata_Control_Type (spec 10)."""
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"], "Metadata_Control": [True, False]},
            index=["0", "1"],
        ),
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "well"}
    migrate(obj)
    assert list(obj.obs["Metadata_Control_Type"]) == ["negcon", "treatment"]
    assert list(obj.obs["Metadata_Control"].astype(bool)) == [True, False]
    assert validate(obj).ok, validate(obj).errors


@pytest.mark.parametrize(("old", "new"), [("cell", "object"), ("perturbation", "aggregate")])
def test_migrate_translates_the_old_resolution_names(old, new):
    obs = pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"])
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=obs,
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    store = {"schema_version": "1.0", "resolution": old}
    if old == "perturbation":
        store["aggregated_from"] = {"by": ["Metadata_Plate"]}
    obj.uns["mantispy"] = store
    migrate(obj)
    assert obj.uns["mantispy"]["resolution"] == new


def test_migrate_derives_object_identity_from_a_1_0_object():
    """A 1.0 cell object predates ImageID/ObjectType; migrate derives them so it validates at object resolution."""
    obs = pd.DataFrame({"Metadata_Plate": ["P1"] * 3, "Metadata_Well": ["A01", "A01", "A02"]}, index=["0", "1", "2"])
    obj = ad.AnnData(
        X=np.ones((3, 1), dtype=np.float32),
        obs=obs,
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "cell"}
    migrate(obj)
    assert {"Metadata_ImageID", "Metadata_ObjectType", "Metadata_ObjectNumber"} <= set(obj.obs.columns)
    assert not obj.obs[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().any()
    assert validate(obj).ok, validate(obj).errors


def test_migrate_raises_when_an_object_has_no_source_identity():
    """With no plate/well/site/image/source column there is nothing to mint an opaque image id from."""
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "cell"}
    with pytest.raises(ValueError, match="Metadata_ImageID"):
        migrate(obj)


def test_make_image_id_composes_the_present_columns_in_order():
    from mantispy._core.schema import make_image_id

    obs = pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"], "Metadata_Site": [1, 2]})
    assert list(make_image_id(obs)) == ["P1|A01|1", "P1|A02|2"]
    # Falls back to the source field id when plate/well/site are absent.
    bare = pd.DataFrame({"Metadata_Source": ["s", "s"], "Metadata_ImageNumber": [7, 8]})
    assert list(make_image_id(bare)) == ["s|7", "s|8"]


def test_make_image_id_rejects_a_missing_value():
    """A present-but-NaN id column must raise clearly, not stringify NaN into the identifier."""
    from mantispy._core.schema import make_image_id

    obs = pd.DataFrame({"Metadata_Plate": ["P1", None], "Metadata_Well": ["A01", "A02"]})
    with pytest.raises(ValueError, match="missing values"):
        make_image_id(obs)


def test_object_number_within_resets_per_image():
    from mantispy._core.schema import object_number_within

    numbers = object_number_within(np.array(["img1", "img1", "img2", "img1"], dtype=object))
    assert list(numbers) == [1, 2, 1, 3]


def test_migrate_renumbers_objects_when_a_coarse_image_id_would_collide():
    """A 1.0 object keeps a per-image ObjectNumber but its only identity is plate/well, so the coarse image id collapses distinct images; migrate keeps the pair unique."""
    obs = pd.DataFrame(
        {"Metadata_Plate": ["P1"] * 4, "Metadata_Well": ["A01"] * 4, "Metadata_ObjectNumber": [1, 2, 1, 2]},
        index=[str(i) for i in range(4)],
    )
    obj = ad.AnnData(
        X=np.ones((4, 1), dtype=np.float32),
        obs=obs,
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "cell"}
    migrate(obj)
    assert set(obj.obs["Metadata_ImageID"]) == {"P1|A01"}
    assert not obj.obs[["Metadata_ImageID", "Metadata_ObjectNumber"]].duplicated().any()
    assert validate(obj).ok, validate(obj).errors


def test_validate_rejects_a_duplicate_image_object_pair(adata):
    """At object resolution the (image, object) pair must identify one primary object."""
    adata.obs["Metadata_ImageID"] = "one-image"
    adata.obs["Metadata_ObjectNumber"] = [1, 1, 2, 3]
    report = validate(adata)
    assert not report.ok
    assert any("not unique" in error for error in report.errors), report.errors


def test_migrate_refuses_to_guess_feature_kind_without_cellprofiler_annotation():
    """A 1.0 embedding object has no CP annotation; migrate must raise rather than mislabel it."""
    obj = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=pd.DataFrame({"is_feature": [True, True]}, index=["dim0", "dim1"]),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "well"}
    with pytest.raises(ValueError, match="feature_kind"):
        migrate(obj)


def test_migrate_recovers_aggregate_grouping_from_provenance():
    """A 1.0 aggregate object grouped by gene migrates to a valid 2.0 object, not an empty grouping."""
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Gene": ["a", "b"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {
        "schema_version": "1.0",
        "resolution": "perturbation",
        "aggregated_from": {"by": ["Metadata_Gene"]},
    }
    migrate(obj)
    assert obj.uns["mantispy"]["resolution"] == "aggregate"
    assert obj.uns["mantispy"]["grouped_by"] == ["Metadata_Gene"]
    assert validate(obj).ok, validate(obj).errors


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


# --- regression tests for the clean-review correctness findings -----------


def test_migrate_recovers_grouping_from_array_valued_provenance():
    """aggregated_from['by'] comes back from h5ad as a numpy array; recovery must not use `arr or ...`."""
    obs = pd.DataFrame({"Metadata_Gene": ["a", "b"], "Metadata_Compound": ["x", "y"]}, index=["0", "1"])
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=obs,
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {
        "schema_version": "1.0",
        "resolution": "aggregate",
        "aggregated_from": {"by": np.array(["Metadata_Gene", "Metadata_Compound"], dtype=object)},
    }
    migrate(obj)
    assert obj.uns["mantispy"]["grouped_by"] == ["Metadata_Gene", "Metadata_Compound"]
    assert validate(obj).ok, validate(obj).errors


def _unclassifiable_1_0():
    obj = ad.AnnData(
        X=np.ones((2, 2), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=pd.DataFrame({"is_feature": [True, True]}, index=["dim0", "dim1"]),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "well"}
    return obj


def test_migrate_is_atomic_when_a_feature_cannot_be_classified():
    """A rejected migrate(copy=False) must leave the object untouched, not half-migrated."""
    obj = _unclassifiable_1_0()
    with pytest.raises(ValueError, match="feature_kind"):
        migrate(obj)
    assert obj.uns["mantispy"]["schema_version"] == "1.0"
    assert "schema_status" not in obj.uns["mantispy"]
    assert "grouped_by" not in obj.uns["mantispy"]
    assert "feature_kind" not in obj.var


def test_migrate_refuses_an_aggregate_with_no_recoverable_grouping():
    """Rather than emit a silently-invalid grouped_by=[] aggregate, migrate raises."""
    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Batch": ["b1", "b2"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]).drop(columns="feature_kind"),
    )
    obj.uns["mantispy"] = {"schema_version": "1.0", "resolution": "aggregate"}
    with pytest.raises(ValueError, match="no recoverable grouping"):
        migrate(obj)
    assert obj.uns["mantispy"]["schema_version"] == "1.0"


def test_validate_reports_a_dirty_is_feature_instead_of_raising(adata):
    """A malformed is_feature must surface as a report error, never an exception out of validate."""
    var = adata.var.copy()
    var["is_feature"] = [True, np.nan]  # NaN is not a usable boolean flag
    adata.var = var
    report = validate(adata)  # must not raise
    assert not report.ok
    assert any("is_feature" in error for error in report.errors), report.errors


def test_schema_status_stable_validates_and_unknown_is_rejected(adata):
    """An object keeps the status it was stamped with; a future 'stable' stays valid, a typo does not."""
    adata.uns["mantispy"]["schema_status"] = "stable"
    assert validate(adata).ok, validate(adata).errors
    adata.uns["mantispy"]["schema_status"] = "provisional"
    report = validate(adata)
    assert not report.ok
    assert any("schema_status" in error for error in report.errors), report.errors


def test_io_stamp_is_atomic_when_feature_kind_is_undeclared():
    """A rejected io.stamp(copy=False) must not leave the object partly stamped."""
    import mantispy as mt

    obj = ad.AnnData(
        X=np.ones((2, 3), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Plate": ["P1", "P1"], "Metadata_Well": ["A01", "A02"]}, index=["0", "1"]),
        var=pd.DataFrame(index=["d0", "d1", "d2"]),
    )
    with pytest.raises(ValueError, match="feature_kind"):
        mt.io.stamp(obj, resolution="well")
    assert "mantispy" not in obj.uns
    assert "feature_kind" not in obj.var


def test_io_stamp_aggregate_requires_grouped_by():
    import mantispy as mt

    obj = ad.AnnData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=pd.DataFrame({"Metadata_Gene": ["a", "b"]}, index=["0", "1"]),
        var=parse_feature_names(["Cells_AreaShape_Area"]),
    )
    with pytest.raises(ValueError, match="grouped_by"):
        mt.io.stamp(obj, resolution="aggregate")
    assert "mantispy" not in obj.uns
