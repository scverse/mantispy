"""Guide -> gene aggregation against a non-targeting-control null.

The per-guide score is a one-sided p-value, so a null guide's p is uniform and a hit guide's is
small. Tested on fabricated guide tables whose null, hit and single-strong-guide genes are known,
so FDR control, power and the Stouffer/Fisher split can be checked without the network.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import stats

import mantispy as mt


def _guides(rng, *, n_hit=60, n_null=240, n_ntc=600, k=4, effect=3.0):
    """One row per guide: hit genes' guides at N(effect, 1), null genes' and NTC guides at N(0, 1)."""
    genes, sgrnas, z = [], [], []
    for g in range(n_hit):
        eff = rng.uniform(0.4, 1.0, k)
        for j, zi in enumerate(rng.normal(effect * eff, 1.0)):
            genes.append(f"hit{g}")
            sgrnas.append(f"hit{g}_{j}")
            z.append(zi)
    for g in range(n_null):
        for j, zi in enumerate(rng.normal(0, 1.0, k)):
            genes.append(f"null{g}")
            sgrnas.append(f"null{g}_{j}")
            z.append(zi)
    for i in range(n_ntc):
        genes.append("nontargeting")
        sgrnas.append(f"ntc{i}")
        z.append(rng.normal(0, 1.0))
    z = np.array(z)
    p = stats.norm.sf(z)  # one-sided: large z -> small p
    obs = pd.DataFrame({"Metadata_Gene": genes, "Metadata_sgRNA": sgrnas, "p": p})
    return AnnData(np.zeros((len(genes), 1), dtype="float32"), obs=obs)


@pytest.fixture
def adata():
    return _guides(np.random.default_rng(0))


def _call(adata, **kw):
    mt.tl.aggregate_guides(adata, score="p", guide="Metadata_sgRNA", gene="Metadata_Gene", control="nontargeting", **kw)
    return adata.uns["mantispy"]["gene_aggregation"]


def test_it_recovers_the_hit_genes_and_controls_the_false_discovery_rate(adata):
    table = _call(adata, alpha=0.05, seed=0)
    called = table[table["is_hit"]]
    is_hit_gene = called["gene"].str.startswith("hit")
    power = is_hit_gene.sum() / 60
    fdr = (~is_hit_gene).sum() / max(len(called), 1)
    assert power > 0.8
    assert fdr < 0.1  # target 0.05, allow sampling slack


def test_fisher_fires_on_a_single_strong_guide_where_stouffer_does_not():
    rng = np.random.default_rng(1)
    genes, sgrnas, z = [], [], []
    for g in range(80):  # one potent guide, three dead
        for j, zi in enumerate(np.array([4.0, 0.0, 0.0, 0.0]) + rng.normal(0, 0.2, 4)):
            genes.append(f"os{g}")
            sgrnas.append(f"os{g}_{j}")
            z.append(zi)
    for i in range(600):
        genes.append("nontargeting")
        sgrnas.append(f"ntc{i}")
        z.append(rng.normal(0, 1.0))
    obs = pd.DataFrame({"Metadata_Gene": genes, "Metadata_sgRNA": sgrnas, "p": stats.norm.sf(z)})
    stouffer = _call(AnnData(np.zeros((len(genes), 1), "float32"), obs=obs), method="stouffer")
    fisher = _call(AnnData(np.zeros((len(genes), 1), "float32"), obs=obs), method="fisher")
    assert fisher["is_hit"].sum() > stouffer["is_hit"].sum()
    assert fisher[fisher["gene"].str.startswith("os")]["is_hit"].mean() > 0.8


def test_control_guides_and_unscored_guides_are_never_hits_and_get_no_pvalue(adata):
    adata.obs.loc[adata.obs["Metadata_Gene"] == "hit0", "p"] = np.nan  # a gene with no scored guide
    _call(adata)
    obs = adata.obs
    assert not obs.loc[obs["Metadata_Gene"] == "nontargeting", "gene_aggregation"].any()
    assert obs.loc[obs["Metadata_Gene"] == "nontargeting", "gene_aggregation_qvalue"].isna().all()
    assert obs.loc[obs["Metadata_Gene"] == "hit0", "gene_aggregation_qvalue"].isna().all()
    assert not obs.loc[obs["Metadata_Gene"] == "hit0", "gene_aggregation"].any()
    table = adata.uns["mantispy"]["gene_aggregation"]
    assert "hit0" not in set(table["gene"])  # no scored guide -> absent from the table
    assert "nontargeting" not in set(table["gene"])  # controls are the null, not tested


def test_the_gene_call_is_broadcast_onto_every_guide_row(adata):
    _call(adata)
    obs = adata.obs
    for _gene, sub in obs[obs["Metadata_Gene"] != "nontargeting"].groupby("Metadata_Gene"):
        assert sub["gene_aggregation_qvalue"].nunique(dropna=False) == 1  # one gene -> one value on all its guides


def test_a_singleton_gene_is_scored_against_the_single_guide_null():
    rng = np.random.default_rng(2)
    genes = ["solo"] + ["nontargeting"] * 400
    z = np.concatenate([[5.0], rng.normal(0, 1.0, 400)])
    obs = pd.DataFrame({"Metadata_Gene": genes, "Metadata_sgRNA": [f"g{i}" for i in range(401)], "p": stats.norm.sf(z)})
    table = _call(AnnData(np.zeros((401, 1), "float32"), obs=obs))
    solo = table[table["gene"] == "solo"]
    assert len(solo) == 1
    assert int(solo["n_guides"].iloc[0]) == 1
    assert bool(solo["is_hit"].iloc[0])  # a lone guide at z=5 beats the single-NTC null


def test_bad_method_and_misplaced_weight_or_direction_are_rejected(adata):
    with pytest.raises(ValueError, match="method must be one of"):
        mt.tl.aggregate_guides(
            adata, score="p", guide="Metadata_sgRNA", gene="Metadata_Gene", control="nontargeting", method="median"
        )
    with pytest.raises(ValueError, match="weight is only supported"):
        mt.tl.aggregate_guides(
            adata,
            score="p",
            guide="Metadata_sgRNA",
            gene="Metadata_Gene",
            control="nontargeting",
            method="fisher",
            weight="p",
        )
    with pytest.raises(ValueError, match="direction is only supported"):
        mt.tl.aggregate_guides(
            adata,
            score="p",
            guide="Metadata_sgRNA",
            gene="Metadata_Gene",
            control="nontargeting",
            method="fisher",
            direction="p",
        )


def test_a_control_label_absent_from_the_data_is_an_error(adata):
    with pytest.raises(ValueError, match="no control guide is present"):
        mt.tl.aggregate_guides(adata, score="p", guide="Metadata_sgRNA", gene="Metadata_Gene", control="absent_label")


def test_a_guide_at_p_one_is_kept_not_dropped(adata):
    # p == 1.0 (the weakest guide a rank-based score can produce) must stay in its gene and the null,
    # not be silently dropped by a clip to exactly 1.0 that would send its z to -inf.
    adata.obs.loc[adata.obs["Metadata_sgRNA"] == "hit0_0", "p"] = 1.0
    adata.obs.loc[adata.obs["Metadata_sgRNA"] == "ntc0", "p"] = 1.0
    table = _call(adata)
    assert int(table.loc[table["gene"] == "hit0", "n_guides"].iloc[0]) == 4  # all four guides kept


def test_a_score_outside_the_unit_interval_is_rejected(adata):
    adata.obs["effect"] = 1.0 - adata.obs["p"]  # a reversed score where large means a stronger phenotype
    adata.obs.loc[adata.obs.index[0], "effect"] = 5.0  # push it clearly out of [0, 1]
    with pytest.raises(ValueError, match=r"must be a one-sided p-value in \[0, 1\]"):
        mt.tl.aggregate_guides(
            adata, score="effect", guide="Metadata_sgRNA", gene="Metadata_Gene", control="nontargeting"
        )


def test_weighting_a_gene_s_guides_changes_its_statistic():
    rng = np.random.default_rng(3)
    genes = ["wg"] * 4 + ["nontargeting"] * 400
    z = np.concatenate([[4.0, 0.0, 0.0, 0.0], rng.normal(0, 1.0, 400)])
    base = pd.DataFrame(
        {"Metadata_Gene": genes, "Metadata_sgRNA": [f"g{i}" for i in range(404)], "p": stats.norm.sf(z)}
    )
    equal = _call(AnnData(np.zeros((404, 1), "float32"), obs=base.copy()))
    weighted_obs = base.copy()
    weighted_obs["w"] = [10.0, 1.0, 1.0, 1.0] + [1.0] * 400  # up-weight the one strong guide
    weighted = _call(AnnData(np.zeros((404, 1), "float32"), obs=weighted_obs), weight="w")
    assert (
        weighted.loc[weighted["gene"] == "wg", "statistic"].iloc[0]
        > equal.loc[equal["gene"] == "wg", "statistic"].iloc[0]
    )


def test_direction_lets_opposing_guides_cancel():
    rng = np.random.default_rng(4)
    # four strongly active guides of one gene, two moving each way; the one-sided p is small for all.
    genes = ["dg"] * 4 + ["nontargeting"] * 400
    p = np.concatenate([[1e-4, 1e-4, 1e-4, 1e-4], stats.norm.sf(rng.normal(0, 1.0, 400))])
    obs = pd.DataFrame({"Metadata_Gene": genes, "Metadata_sgRNA": [f"g{i}" for i in range(404)], "p": p})
    without = _call(AnnData(np.zeros((404, 1), "float32"), obs=obs.copy()))
    obs["dir"] = [1.0, 1.0, -1.0, -1.0] + [1.0] * 400  # the gene's guides disagree in direction
    with_dir = _call(AnnData(np.zeros((404, 1), "float32"), obs=obs), direction="dir")
    stat_without = without.loc[without["gene"] == "dg", "statistic"].iloc[0]
    stat_with = with_dir.loc[with_dir["gene"] == "dg", "statistic"].iloc[0]
    assert stat_with < stat_without  # cancellation pulls the combined statistic down
    assert abs(stat_with) < 1.0  # opposing guides roughly cancel


def test_repeated_guide_ids_warn(adata):
    adata.obs.loc[adata.obs["Metadata_sgRNA"] == "hit0_1", "Metadata_sgRNA"] = "hit0_0"  # a duplicate id
    with pytest.warns(UserWarning, match="repeats ids"):
        _call(adata)
