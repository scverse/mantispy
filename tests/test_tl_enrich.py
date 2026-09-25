"""Feature sets built from the parsed annotation, and enrichment over them."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import mantispy as mt
from mantispy._core.features import empty_annotation, parse_feature_names
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


def _object(var, n_obs=5):
    """A small well-level object over a hand-made var."""
    return ad.AnnData(
        np.random.default_rng(0).random((n_obs, len(var))).astype(np.float32),
        obs=pd.DataFrame(
            {"Metadata_Plate": "P0", "Metadata_Well": [f"A{index + 1:02d}" for index in range(n_obs)]},
            index=[str(index) for index in range(n_obs)],
        ),
        var=var,
    )


def _unparsed(n=6):
    """The shape of a tl.dose_trajectory result: the ten columns present, all but one empty."""
    var = empty_annotation(pd.Index([f"F{index}@0.50" for index in range(n)]))
    var["feature"] = [f"F{index}" for index in range(n)]
    return _object(var)


def test_feature_sets_returns_an_empty_network_when_no_feature_is_fully_annotated():
    """An object may legitimately have an annotation that names no family, and the caller asked for
    the network, not for a verdict on the annotation."""
    network = mt.tl.feature_sets(_unparsed())
    assert list(network.columns) == ["source", "target", "weight"]
    assert network.empty


def test_feature_sets_survives_an_annotation_no_row_completes():
    """The all-empty column is not the only way in: two features can each fill a different half of
    `by`, so neither column is empty and still no row has both."""
    var = empty_annotation(pd.Index(["f0", "f1"]))
    var["feature_group"] = pd.Categorical(["AreaShape", None])
    var["channel"] = pd.Categorical([None, "DNA"])
    assert mt.tl.feature_sets(_object(var, n_obs=4), by=("feature_group", "channel")).empty


@pytest.fixture
def active():
    """A small object with one genuinely active feature group and a feature_sets-style net.

    The first half of the samples carry a constant added across the ACTIVE group's features,
    so a working scorer must rank ACTIVE higher there than in the untouched second half.
    The net has four groups of fifteen features each: enough sets and features for mlm to fit.
    """
    rng = np.random.default_rng(0)
    names, sources = [], []
    for group in ("ACTIVE", "G1", "G2", "G3"):
        for index in range(15):
            names.append(f"{group}_f{index}")
            sources.append(group)
    n_obs = 20
    matrix = rng.standard_normal((n_obs, len(names))).astype(np.float32)
    is_active_sample = np.zeros(n_obs, dtype=bool)
    is_active_sample[: n_obs // 2] = True
    is_active_col = np.array([source == "ACTIVE" for source in sources])
    matrix[np.ix_(is_active_sample, is_active_col)] += 5.0

    adata = ad.AnnData(
        matrix,
        obs=pd.DataFrame(
            {"state": np.where(is_active_sample, "on", "off")}, index=[str(index) for index in range(n_obs)]
        ),
        var=pd.DataFrame(index=names),
    )
    net = pd.DataFrame({"source": sources, "target": names, "weight": 1.0})
    return adata, net, is_active_sample


_SINGLE_METHODS = ("ulm", "mlm", "ora", "aucell", "gsea", "gsva", "zscore", "waggr", "viper")
#: The single methods that also write a padj frame; aucell and gsva write only a score.
_WITH_PADJ = {"ulm", "mlm", "ora", "gsea", "zscore", "waggr", "viper"}


@pytest.mark.parametrize(
    "method",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                reason="decoupler's mlm cannot fit on this small synthetic net (few features per source); "
                "mlm stays in METHODS and works on real-sized data",
            ),
        )
        if name == "mlm"
        else name
        for name in _SINGLE_METHODS
    ],
)
def test_enrich_runs_every_single_method(active, method):
    adata, net, _ = active
    mt.tl.enrich(adata, net=net, method=method, tmin=2)
    assert adata.obsm[f"score_{method}"].shape[0] == adata.n_obs
    assert (f"padj_{method}" in adata.obsm) == (method in _WITH_PADJ)


@pytest.mark.parametrize("method", ["ulm", "zscore"])
def test_active_group_scores_higher_where_it_is_active(active, method):
    adata, net, is_active_sample = active
    mt.tl.enrich(adata, net=net, method=method, tmin=2)
    scores = np.asarray(adata.obsm[f"score_{method}"]["ACTIVE"], dtype=float)
    assert scores[is_active_sample].mean() > scores[~is_active_sample].mean()


def test_consensus_writes_a_consensus_score(active):
    adata, net, is_active_sample = active
    mt.tl.enrich(adata, net=net, method="consensus", tmin=2)
    assert "score_consensus" in adata.obsm
    assert "padj_consensus" in adata.obsm
    scores = np.asarray(adata.obsm["score_consensus"]["ACTIVE"], dtype=float)
    assert scores[is_active_sample].mean() > scores[~is_active_sample].mean()


def test_consensus_uses_a_given_panel(active):
    adata, net, _ = active
    mt.tl.enrich(adata, net=net, method="consensus", methods=["ulm", "zscore"], tmin=2)
    assert "score_consensus" in adata.obsm
    assert "score_ulm" in adata.obsm and "score_zscore" in adata.obsm
    # aucell is in the default panel but not in the one asked for, so decouple must not have run it.
    assert "score_aucell" not in adata.obsm


def test_methods_only_applies_to_consensus(active):
    adata, net, _ = active
    with pytest.raises(ValueError, match="method='consensus'"):
        mt.tl.enrich(adata, net=net, method="ulm", methods=["ulm", "zscore"])


def test_consensus_rejects_a_panel_entry_that_is_not_a_single_method(active):
    adata, net, _ = active
    with pytest.raises(ValueError, match="single methods"):
        mt.tl.enrich(adata, net=net, method="consensus", methods=["ulm", "not_a_method"])


def test_get_features_filters_a_column_that_is_legitimately_empty():
    """An AreaShape feature has no channel, and parse_feature_names correctly leaves it missing.
    Asking for a channel there matches nothing; it is not a broken annotation."""
    var = parse_feature_names(["Cells_AreaShape_Area", "Nuclei_AreaShape_Area"])
    assert var["channel"].isna().all(), "the parser leaves a geometry feature's channel missing"
    adata = _object(var, n_obs=4)
    assert mt.get.features(adata, channel="DNA") == []
    assert mt.get.features(adata, feature_group="AreaShape") == list(adata.var_names)
