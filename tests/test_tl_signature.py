"""Feature-family signatures: the contrast, the collapse, and the plot."""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


import mantispy as mt


@pytest.fixture
def annotated(well_profiles):
    """Well profiles whose features carry a family, a channel and a compartment."""

    def build(n_features=120, **kwargs):
        adata = well_profiles(n_features=n_features, **kwargs)
        rng = np.random.default_rng(0)
        adata.var["feature_group"] = rng.choice(["AreaShape", "Intensity", "Texture"], n_features)
        adata.var["channel"] = rng.choice(["DNA", "Tub", "none"], n_features)
        adata.var["object"] = rng.choice(["Cells", "Nuclei"], n_features)
        return adata

    return build


def test_the_rest_contrast_leaves_the_reference_out(annotated):
    """'rest' compares a perturbation with the other perturbations, not with the controls."""
    rng = np.random.default_rng(3)
    n_features = 80
    labels = ["DMSO"] * 12 + ["a"] * 6 + ["b"] * 6
    values = rng.normal(size=(len(labels), n_features))
    # Give the controls a large offset that both treatments share. Against the controls it
    # dominates; against each other it cancels.
    values[:12] += 6.0
    obs = pd.DataFrame(
        {
            "Metadata_Plate": "P0",
            "Metadata_Well": [f"A{i + 1:02d}" for i in range(len(labels))],
            "Metadata_Perturbation": labels,
            "Metadata_Control": [label == "DMSO" for label in labels],
        },
        index=[str(i) for i in range(len(labels))],
    )

    from mantispy._core.schema import stamp

    adata = ad.AnnData(X=values.astype(np.float32), obs=obs)
    adata.var_names = [f"Cells_AreaShape_F{i}" for i in range(n_features)]
    stamp(adata, resolution="well")

    mt.tl.differential_features(adata, block=None, key_added="vs_dmso")
    mt.tl.differential_features(adata, block=None, contrast="rest", key_added="vs_rest")
    against_dmso = adata.uns["mantispy"]["vs_dmso"]["difference"].abs().median()
    against_rest = adata.uns["mantispy"]["vs_rest"]["difference"].abs().median()
    assert against_dmso > 4.0, "the shared offset shows up against the controls"
    assert against_rest < 1.0, "and cancels when the perturbations are compared with each other"


def test_the_signature_collapses_to_named_families(annotated):
    adata = annotated()
    mt.tl.differential_features(adata, block=None, key_added="d")
    signature = mt.tl.feature_signature(adata, key="d")

    families = set(adata.var[["feature_group", "channel", "object"]].astype(str).apply(" | ".join, axis=1))
    assert set(signature.var_names) == families
    assert signature.n_obs == adata.uns["mantispy"]["d"]["group"].nunique()
    assert int(signature.var["n_features"].sum()) == adata.n_vars

    # Each entry is the mean statistic of its family, sign kept.
    table = adata.uns["mantispy"]["d"]
    group = signature.obs_names[0]
    family = signature.var_names[0]
    members = adata.var_names[
        adata.var[["feature_group", "channel", "object"]].astype(str).apply(" | ".join, axis=1) == family
    ]
    expected = table.loc[(table["group"] == group) & table["feature"].isin(members), "t"].mean()
    np.testing.assert_allclose(signature[group, family].X.ravel()[0], expected, rtol=1e-5)


def test_the_signature_is_an_ordinary_profile_object(annotated):
    """So scanpy clusters it and the MOA tools score it, with no special handling."""
    adata = annotated()
    mt.tl.differential_features(adata, block=None, key_added="d")
    signature = mt.tl.feature_signature(adata, key="d")
    assert signature.uns["mantispy"]["resolution"] == "perturbation"
    mt.tl.similarity(signature)
    assert signature.obsp["similarity"].shape == (signature.n_obs, signature.n_obs)


def test_the_signature_is_an_object_io_accepts(tmp_path, annotated):
    """Regression test for #103: var held only the columns that name a family, so the stamped result
    failed validation on the seven annotation columns the schema requires and could not be written."""
    adata = annotated()
    mt.tl.differential_features(adata, block=None, key_added="d")
    signature = mt.tl.feature_signature(adata, key="d")

    report = mt.io.validate(signature)
    assert report.ok, str(report)
    # A family is not a CellProfiler measurement, so the columns that do not name one stay empty.
    for column in ("feature", "scale", "angle", "gray_levels", "radial_bin", "params"):
        assert signature.var[column].isna().all(), column
    assert signature.var["is_feature"].all()
    # The columns that do name it keep their values.
    assert set(signature.var["feature_group"]) <= {"AreaShape", "Intensity", "Texture"}
    assert set(signature.var["object"]) <= {"Cells", "Nuclei"}
    assert int(signature.var["n_features"].sum()) == adata.n_vars

    path = tmp_path / "signature.h5ad"
    mt.io.write(signature, path)
    loaded = mt.io.read(path)
    assert list(loaded.var_names) == list(signature.var_names)
    assert list(loaded.var.columns) == list(signature.var.columns)
    # The empty annotation columns are category dtype for exactly this reason: an object-dtype
    # column holding only None does not survive the h5ad writer.
    for column in ("feature", "radial_bin", "params"):
        assert loaded.var[column].isna().all(), column


def test_a_by_that_is_not_the_default_still_round_trips(tmp_path, annotated):
    """`by` names whichever var columns define a family, including one the schema knows nothing about."""
    adata = annotated()
    adata.var["panel"] = np.where(adata.var["object"].to_numpy() == "Cells", "outer", "inner")
    mt.tl.differential_features(adata, block=None, key_added="d")
    signature = mt.tl.feature_signature(adata, key="d", by=("feature_group", "panel"))

    report = mt.io.validate(signature)
    assert report.ok, str(report)
    # The column the schema does not know sits alongside its own, and a schema column that names
    # no family here is left empty rather than filled with the family it was not grouped by.
    assert set(signature.var["panel"]) == {"outer", "inner"}
    assert signature.var["channel"].isna().all()
    assert signature.var["object"].isna().all()
    assert int(signature.var["n_features"].sum()) == adata.n_vars

    # validate checks that the ten names are present and never checks a dtype, so the writer is what
    # says whether a var assembled from a non-default `by` can actually be saved.
    path = tmp_path / "signature.h5ad"
    mt.io.write(signature, path)
    loaded = mt.io.read(path)
    assert list(loaded.var["panel"]) == list(signature.var["panel"])


def test_a_by_column_keeps_the_dtype_and_the_missing_values_var_held(annotated):
    """The family's name needs strings and a sentinel for a missing component; its var column does not, and
    writing them there put "none" in the schema's channel and the string "True" in its boolean is_feature."""
    adata = annotated()
    # A channel that is genuinely absent, which is what parse_feature_names leaves on an AreaShape feature.
    adata.var["channel"] = pd.Categorical(np.where(adata.var["feature_group"].to_numpy() == "AreaShape", None, "DNA"))
    adata.var["scale"] = np.where(adata.var["object"].to_numpy() == "Cells", 3.0, 5.0)
    # Supplied here like the two above: the fixture stamps, and a stamp records what an object is
    # rather than completing its annotation.
    adata.var["is_feature"] = True
    mt.tl.differential_features(adata, block=None, key_added="d")

    signature = mt.tl.feature_signature(adata, key="d")
    assert signature.var["channel"].isna().any(), "a missing channel stays missing"
    assert "none" not in set(signature.var["channel"].dropna())
    assert any(name.endswith("| none | Cells") or "| none |" in name for name in signature.var_names), (
        "the sentinel still names the family"
    )

    on_scale = mt.tl.feature_signature(adata, key="d", by=("feature_group", "scale"))
    assert on_scale.var["scale"].dtype == np.float64
    assert set(on_scale.var["scale"]) == {3.0, 5.0}

    on_flag = mt.tl.feature_signature(adata, key="d", by=("feature_group", "is_feature"))
    assert on_flag.var["is_feature"].dtype == bool, "validate's is_feature check reads a boolean"
    assert (~on_flag.var["is_feature"]).sum() == 0


def test_by_refuses_the_same_column_twice(annotated):
    adata = annotated()
    mt.tl.differential_features(adata, block=None, key_added="d")
    with pytest.raises(ValueError, match="more than once"):
        mt.tl.feature_signature(adata, key="d", by=("object", "object"))


def test_it_says_so_when_the_table_or_the_annotation_is_missing(annotated):
    adata = annotated()
    with pytest.raises(KeyError, match="differential_features"):
        mt.tl.feature_signature(adata)

    mt.tl.differential_features(adata, block=None, key_added="d")
    del adata.var["channel"]
    with pytest.raises(KeyError, match="channel"):
        mt.tl.feature_signature(adata, key="d")


def test_the_heatmap_draws_one_row_per_group(annotated):
    adata = annotated()
    adata.obs["Metadata_MOA"] = np.where(adata.obs["Metadata_Perturbation"].to_numpy() == "compound", "mechanism", None)
    mt.tl.differential_features(adata, block=None, key_added="d")
    signature = mt.tl.feature_signature(adata, key="d")
    axes = mt.pl.feature_signature(signature, groupby="Metadata_MOA")
    assert len(axes.get_yticklabels()) == 1
    assert len(axes.get_xticklabels()) == signature.n_vars


def test_an_annotation_holding_the_separator_does_not_break_the_split(annotated):
    """rohban has feature groups that contain a pipe."""
    adata = annotated()
    adata.var["feature_group"] = np.where(
        adata.var["feature_group"].to_numpy() == "Texture", "Texture|Gabor", adata.var["feature_group"]
    )
    mt.tl.differential_features(adata, block=None, key_added="d")
    signature = mt.tl.feature_signature(adata, key="d")

    # The components are read back from the annotation, so they survive intact.
    assert "Texture|Gabor" in set(signature.var["feature_group"])
    assert set(signature.var["channel"]) <= {"DNA", "Tub", "none"}
    assert int(signature.var["n_features"].sum()) == adata.n_vars


def test_the_heatmap_reads_an_infinity_as_missing():
    """Regression test for #65: one infinity overflowed the clustering distances and raised, and set the colour limits."""
    values = np.random.default_rng(0).normal(size=(6, 5))
    drawn = []
    for value in (np.inf, np.nan):
        values[0, 0] = value
        drawn.append(mt.pl.feature_signature(ad.AnnData(values.copy())).images[0])
    assert drawn[0].get_clim() == drawn[1].get_clim()
    np.testing.assert_array_equal(drawn[0].get_array(), drawn[1].get_array())


def test_a_by_column_that_is_entirely_missing_still_writes(annotated, tmp_path):
    """empty_annotation makes the text columns categorical so an entirely missing one survives an
    h5ad round trip; assigning the raw var column over it put the object dtype back, and the
    signature validated but could not be saved — the defect this whole change set out to fix."""
    adata = annotated(n_features=40)
    adata.var["channel"] = pd.Series([None] * adata.n_vars, dtype=object).values
    mt.tl.differential_features(adata, block=None, key_added="d")

    signature = mt.tl.feature_signature(adata, key="d")

    assert mt.io.validate(signature).ok
    mt.io.write(signature, tmp_path / "signature.h5ad")


def test_the_pivot_key_does_not_name_the_obs_index(annotated, tmp_path):
    """pd.Index(index, name=None) keeps the name it had — None is pandas' 'leave it alone'. var was
    cleared with rename(None); obs was not, so every signature on disk carried the pivot's key."""
    adata = annotated(n_features=40)
    mt.tl.differential_features(adata, block=None, key_added="d")

    signature = mt.tl.feature_signature(adata, key="d")
    assert signature.obs.index.name is None

    path = tmp_path / "signature.h5ad"
    mt.io.write(signature, path)
    assert mt.io.read(path).obs.index.name is None


def test_two_components_that_name_the_same_family_are_refused(annotated):
    """The separator can appear inside a component -- rohban2017's feature groups do -- so
    ('A | B', 'C') and ('A', 'B | C') join to one name. They were averaged into a single column and
    var reported whichever tuple came first, which flips when var is reordered."""
    adata = annotated(n_features=4)
    adata.var["feature_group"] = ["A | B", "A", "A | B", "A"]
    adata.var["channel"] = ["C", "B | C", "C", "B | C"]
    adata.var["object"] = ["Cells"] * 4
    mt.tl.differential_features(adata, block=None, key_added="d")

    with pytest.raises(ValueError, match="name the same family"):
        mt.tl.feature_signature(adata, key="d")


def test_a_signature_over_an_annotation_that_names_nothing_is_refused(annotated):
    """Every by column empty makes one family called "none | none | none", averaging every feature
    in the object into a single column and reporting nothing about any of them."""
    adata = annotated(n_features=20)
    for column in ("feature_group", "channel", "object"):
        adata.var[column] = pd.Categorical([None] * adata.n_vars)
    mt.tl.differential_features(adata, block=None, key_added="d")

    with pytest.raises(ValueError, match="name no family"):
        mt.tl.feature_signature(adata, key="d")


def test_a_by_column_that_is_empty_beside_a_populated_one_still_makes_families(annotated):
    """The companion direction: only some components missing is ordinary -- a geometry feature has
    no channel -- and must still give one family per populated combination."""
    adata = annotated(n_features=20)
    adata.var["channel"] = pd.Categorical([None] * adata.n_vars)
    mt.tl.differential_features(adata, block=None, key_added="d")

    signature = mt.tl.feature_signature(adata, key="d")
    assert signature.n_vars > 1
    assert all("none" in name for name in signature.var_names)


def test_the_sentinel_and_a_genuine_gap_are_one_family(annotated):
    """ "none" is this codebase's own spelling of a missing channel (pl/_features.py fills it in), so
    a var that spells it out and one that leaves it missing name the same family. Comparing the raw
    components rather than the names made that a clash and refused the object outright."""
    adata = annotated(n_features=4)
    adata.var["feature_group"] = ["Intensity"] * 4
    adata.var["channel"] = pd.Categorical(["none", None, "none", None])
    adata.var["object"] = ["Cells"] * 4
    mt.tl.differential_features(adata, block=None, key_added="d")

    signature = mt.tl.feature_signature(adata, key="d")
    assert list(signature.var_names) == ["Intensity | none | Cells"]
    assert signature.var["n_features"].tolist() == [4]
