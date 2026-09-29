"""Feature-family signatures: the contrast, the collapse, and the plot."""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

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


def test_the_signature_is_an_object_io_accepts(tmp_path, annotated):
    """Regression test for #103: var held only the columns that name a family, so the stamped result failed validation on the seven annotation columns the schema requires and could not be written."""
    adata = annotated()
    mt.tl.differential_features(adata, block=None, key_added="d")
    signature = mt.tl.feature_signature(adata, key="d")

    report = mt.io.validate(signature)
    assert report.ok, str(report)
    # A family is not a CellProfiler measurement, so the columns that do not name one stay empty.
    for column in ("feature", "scale", "angle", "gray_levels", "radial_bin", "params"):
        assert signature.var[column].isna().all(), column
    assert signature.var["is_feature"].all()
    assert set(signature.var["feature_group"]) <= {"AreaShape", "Intensity", "Texture"}
    assert set(signature.var["object"]) <= {"Cells", "Nuclei"}
    assert int(signature.var["n_features"].sum()) == adata.n_vars

    path = tmp_path / "signature.h5ad"
    mt.io.write(signature, path)
    loaded = mt.io.read(path)
    assert list(loaded.var_names) == list(signature.var_names)
    assert list(loaded.var.columns) == list(signature.var.columns)
    # The empty annotation columns are category dtype because an object-dtype column of only None does not survive the h5ad writer.
    for column in ("feature", "radial_bin", "params"):
        assert loaded.var[column].isna().all(), column


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
    _, ax = plt.subplots()
    mt.pl.feature_signature(signature, groupby="Metadata_MOA", ax=ax)
    assert len(ax.get_yticklabels()) == 1
    assert len(ax.get_xticklabels()) == signature.n_vars
    plt.close(ax.figure)


def test_the_heatmap_reads_an_infinity_as_missing():
    """Regression test for #65: one infinity overflowed the clustering distances and raised, and set the colour limits."""
    values = np.random.default_rng(0).normal(size=(6, 5))
    drawn = []
    for value in (np.inf, np.nan):
        values[0, 0] = value
        _, ax = plt.subplots()
        mt.pl.feature_signature(ad.AnnData(values.copy()), ax=ax)
        drawn.append(ax.images[0])
    assert drawn[0].get_clim() == drawn[1].get_clim()
    np.testing.assert_array_equal(drawn[0].get_array(), drawn[1].get_array())
    plt.close("all")


def test_two_components_that_name_the_same_family_are_refused(annotated):
    """The separator can appear inside a component -- rohban2017's feature groups do -- so ('A | B', 'C') and ('A', 'B | C') join to one name.
    They were averaged into a single column and var reported whichever tuple came first, which flips when var is reordered."""
    adata = annotated(n_features=4)
    adata.var["feature_group"] = ["A | B", "A", "A | B", "A"]
    adata.var["channel"] = ["C", "B | C", "C", "B | C"]
    adata.var["object"] = ["Cells"] * 4
    mt.tl.differential_features(adata, block=None, key_added="d")

    with pytest.raises(ValueError, match="name the same family"):
        mt.tl.feature_signature(adata, key="d")


def test_a_signature_over_an_annotation_that_names_nothing_is_refused(annotated):
    """Every by column empty makes one family called "none | none | none", averaging every feature in the object into a single column and reporting nothing about any of them."""
    adata = annotated(n_features=20)
    for column in ("feature_group", "channel", "object"):
        adata.var[column] = pd.Categorical([None] * adata.n_vars)
    mt.tl.differential_features(adata, block=None, key_added="d")

    with pytest.raises(ValueError, match="name no family"):
        mt.tl.feature_signature(adata, key="d")
