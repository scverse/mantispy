"""Feature sets built from the parsed annotation, and enrichment over them."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.schema import stamp


@pytest.fixture
def profiles():
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=48, n_cells=10, n_features=40, n_perturbations=3, effect_size=3.0, seed=0
    )
    mt.pp.normalize(cells, by="Metadata_Plate", reference="negcon")
    return mt.tl.aggregate(cells, min_cells=0)


def test_feature_sets_builds_a_decoupler_net(profiles):
    net = mt.tl.feature_sets(profiles, by="feature_group")
    assert list(net.columns) == ["source", "target", "weight"]
    assert set(net["target"]) <= set(profiles.var_names)
    assert set(net["source"]) == set(profiles.var["feature_group"].dropna().astype(str))


def test_feature_sets_can_combine_columns(profiles):
    net = mt.tl.feature_sets(profiles, by="group_by_channel")
    assert any("|" in source for source in net["source"])
    assert net["source"].nunique() >= profiles.var["feature_group"].nunique()


def test_feature_sets_rejects_a_column_that_is_not_there(profiles):
    with pytest.raises(KeyError, match="available:"):
        mt.tl.feature_sets(profiles, by="not_a_column")


@pytest.mark.parametrize("method", ["ulm", "ora"])
def test_enrich_writes_scores_and_padj(profiles, method):
    mt.tl.enrich(profiles, by="feature_group", method=method, tmin=2)
    assert profiles.obsm[f"score_{method}"].shape[0] == profiles.n_obs
    assert f"padj_{method}" in profiles.obsm


def test_enrich_matches_calling_decoupler_directly(profiles):
    """The wrapper must not change the numbers, only where they come from."""
    import decoupler as dc

    direct = profiles.copy()
    dc.mt.ulm(direct, mt.tl.feature_sets(profiles, by="feature_group"), tmin=2)

    mt.tl.enrich(profiles, by="feature_group", method="ulm", tmin=2)
    np.testing.assert_allclose(
        np.asarray(profiles.obsm["score_ulm"], dtype=float), np.asarray(direct.obsm["score_ulm"], dtype=float)
    )


def test_rank_features_carries_the_parsed_annotation(profiles):
    mt.tl.rank_features(profiles, groupby="Metadata_Perturbation")
    table = profiles.uns["mantispy"]["rank_features"]
    assert {"group", "feature", "score", "pvalue", "qvalue", "feature_group", "channel"} <= set(table.columns)
    # The features the simulation moved should lead their perturbation's ranking.
    truth = profiles.uns["mantispy"]["truth"]["affected_features"]
    group = next(name for name, features in truth.items() if features)
    top = table[table["group"] == group].nlargest(3, "score")["feature"].tolist()
    assert set(truth[group]) & set(top)


def test_rank_sets_ranks_the_enrichment_scores(profiles):
    mt.tl.enrich(profiles, by="feature_group", method="ulm", tmin=2)
    mt.tl.rank_sets(profiles, groupby="Metadata_Perturbation")
    table = profiles.uns["mantispy"]["rank_sets"]
    assert {"group", "set", "score"} <= set(table.columns)
    assert len(table) == profiles.obs["Metadata_Perturbation"].nunique() * profiles.obsm["score_ulm"].shape[1]


def test_rank_sets_says_what_to_run_first(profiles):
    with pytest.raises(KeyError, match="mt.tl.enrich"):
        mt.tl.rank_sets(profiles, groupby="Metadata_Perturbation")


def test_round_trip(profiles, tmp_path):
    mt.tl.enrich(profiles, by="feature_group", tmin=2)
    mt.tl.rank_features(profiles, groupby="Metadata_Perturbation")
    mt.io.write(profiles, tmp_path / "enrich.h5ad")
    loaded = mt.io.read(tmp_path / "enrich.h5ad")
    assert "score_ulm" in loaded.obsm
    assert len(loaded.uns["mantispy"]["rank_features"]) > 0


def test_ora_scores_the_extreme_features_not_the_ordinary_ones():
    """decoupler keeps features whose rank exceeds n_up, so enrich sets a default n_up.
    Without it ORA runs on the bottom 95%, gives the most extreme set p = 1.0 and reports
    the depleted set as the hit."""
    names = [f"Cells_AreaShape_f{i}" for i in range(100)]
    adata = ad.AnnData(
        np.arange(100, dtype=np.float32)[None, :],
        obs=pd.DataFrame({"Metadata_Plate": ["P1"], "Metadata_Well": ["A01"]}, index=["0"]),
        var=pd.DataFrame({"feature_group": ["AreaShape"] * 100, "is_feature": True}, index=names),
    )
    stamp(adata, resolution="well")
    net = pd.DataFrame({"source": ["TOP"] * 10 + ["BOTTOM"] * 10, "target": names[90:] + names[:10], "weight": 1.0})

    mt.tl.enrich(adata, net=net, method="ora", n_bg=100, tmin=5)
    scores = adata.obsm["score_ora"]
    assert float(scores["TOP"].iloc[0]) > float(scores["BOTTOM"].iloc[0])
    assert float(adata.obsm["padj_ora"]["TOP"].iloc[0]) < 0.05

    # An explicit n_up must override that default.
    explicit = adata.copy()
    mt.tl.enrich(explicit, net=net, method="ora", n_bg=100, tmin=5, n_up=5)
    assert float(explicit.obsm["score_ora"]["TOP"].iloc[0]) != float(scores["TOP"].iloc[0])


def _unparsed(n=6):
    """An object shaped like a tl.dose_trajectory result: the ten columns present, all but one empty."""
    import anndata as ad

    from mantispy._core.features import empty_annotation

    var = empty_annotation(pd.Index([f"F{index}@0.50" for index in range(n)]))
    var["feature"] = [f"F{index}" for index in range(n)]
    adata = ad.AnnData(
        np.random.default_rng(0).random((5, n)).astype(np.float32),
        obs=pd.DataFrame(
            {"Metadata_Plate": "P0", "Metadata_Well": [f"A{index + 1:02d}" for index in range(5)]},
            index=[str(index) for index in range(5)],
        ),
        var=var,
    )
    return adata


def test_feature_sets_returns_an_empty_network_when_no_feature_is_fully_annotated():
    """Not a raise: an object may legitimately have an annotation that names no family, and the
    caller asked for the network, not for a verdict on the annotation. It crashed with
    AttributeError: 'DataFrame' object has no attribute 'str' instead."""
    network = mt.tl.feature_sets(_unparsed())
    assert list(network.columns) == ["source", "target", "weight"]
    assert network.empty


def test_feature_sets_survives_an_annotation_no_row_completes():
    """The all-empty column is not the only way in: two features can each fill a different half of
    `by`, so neither column is empty and still no row has both."""
    import anndata as ad

    from mantispy._core.features import empty_annotation

    var = empty_annotation(pd.Index(["f0", "f1"]))
    var["feature_group"] = pd.Categorical(["AreaShape", None])
    var["channel"] = pd.Categorical([None, "DNA"])
    adata = ad.AnnData(
        np.random.default_rng(0).random((4, 2)).astype(np.float32),
        obs=pd.DataFrame(
            {"Metadata_Plate": "P0", "Metadata_Well": [f"A{index + 1:02d}" for index in range(4)]},
            index=[str(index) for index in range(4)],
        ),
        var=var,
    )
    assert mt.tl.feature_sets(adata, by=("feature_group", "channel")).empty


def test_get_features_filters_a_column_that_is_legitimately_empty():
    """An AreaShape feature has no channel, and parse_feature_names correctly leaves it missing.
    Asking for a channel there matches nothing; it is not a broken annotation."""
    import anndata as ad

    from mantispy._core.features import parse_feature_names

    var = parse_feature_names(["Cells_AreaShape_Area", "Nuclei_AreaShape_Area"])
    assert var["channel"].isna().all(), "the parser leaves a geometry feature's channel missing"
    adata = ad.AnnData(
        np.random.default_rng(0).random((4, 2)).astype(np.float32),
        obs=pd.DataFrame(
            {"Metadata_Plate": "P0", "Metadata_Well": [f"A{index + 1:02d}" for index in range(4)]},
            index=[str(index) for index in range(4)],
        ),
        var=var,
    )
    assert mt.get.features(adata, channel="DNA") == []
    assert mt.get.features(adata, feature_group="AreaShape") == list(adata.var_names)
