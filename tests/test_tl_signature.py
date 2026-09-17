"""Feature-family signatures: the contrast, the collapse, and the plot."""

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
    import anndata as ad

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
