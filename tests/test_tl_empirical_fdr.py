"""Empirical FDR calibrated against control genes.

The control genes form the null, so a gene scores a hit when it sits past what they reach.
Tested on a fabricated per-gene table whose null, weak and strong genes are known, so the calibration and the two quantities can be checked without the network.
"""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantispy as mt


@pytest.fixture
def genes():
    """One row per gene: 300 control genes and 150 null genes at N(0, 1), 60 strong genes at N(6, 1)."""
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 300), rng.normal(0, 1, 150), rng.normal(6, 1, 60)])
    names = [f"ctrl{i}" for i in range(300)] + [f"null{i}" for i in range(150)] + [f"hit{i}" for i in range(60)]
    # The gene identity lives in a Metadata_ column, not the index: group=None resolves it there.
    obs = pd.DataFrame({"score": scores, "Metadata_Perturbation": names}, index=names)
    adata = AnnData(scores.reshape(-1, 1).astype(float), obs=obs)
    controls = {name for name in names if name.startswith("ctrl")}
    return adata, controls


def test_it_calls_the_strong_genes_and_spares_the_null_ones(genes):
    adata, controls = genes
    mt.tl.empirical_fdr(adata, control_genes=controls, score="score", alpha=0.05, criterion="q")

    hit = adata.obs["empirical_fdr"]
    strong = adata.obs_names.str.startswith("hit")
    null = adata.obs_names.str.startswith("null")
    control = adata.obs_names.str.startswith("ctrl")
    assert hit[strong].mean() > 0.8  # the shifted genes are mostly called
    assert hit[null].mean() < 0.2  # the null genes mostly are not
    assert not hit[control].any()  # a control gene is never a hit


def test_control_rows_get_no_pvalue_and_the_outputs_are_written(genes):
    adata, controls = genes
    mt.tl.empirical_fdr(adata, control_genes=controls, score="score")

    control = adata.obs["empirical_fdr_control"].to_numpy()
    assert control.sum() == 300
    assert adata.obs["empirical_fdr_pvalue"][control].isna().all()
    assert adata.obs["empirical_fdr_qvalue"][control].isna().all()
    assert adata.obs["empirical_fdr_pvalue"][~control].notna().all()
    table = adata.uns["mantispy"]["empirical_fdr"]
    assert list(table.columns) == ["group", "score", "is_control", "pvalue", "qvalue", "is_hit"]
    assert len(table) == adata.n_obs


def test_the_pvalue_is_calibrated_and_the_qvalue_is_higher(genes):
    adata, controls = genes
    mt.tl.empirical_fdr(adata, control_genes=controls, score="score", criterion="p")

    null = adata.obs_names.str.startswith("null")
    p = adata.obs["empirical_fdr_pvalue"]
    # The null genes are drawn from the control distribution, so about alpha of them fall below alpha.
    assert p[null].le(0.05).mean() < 0.15
    # The q-value accounts for the 210 tested genes, so it never sits below the p-value.
    q = adata.obs["empirical_fdr_qvalue"]
    tested = ~adata.obs["empirical_fdr_control"].to_numpy()
    assert (q[tested].to_numpy() >= p[tested].to_numpy() - 1e-9).all()


def test_the_pvalue_falls_as_the_score_rises(genes):
    adata, controls = genes
    mt.tl.empirical_fdr(adata, control_genes=controls, score="score")
    tested = adata.obs[~adata.obs["empirical_fdr_control"].to_numpy()].sort_values("score")
    p = tested["empirical_fdr_pvalue"].to_numpy()
    assert np.all(np.diff(p) <= 1e-12)  # non-increasing with score


def test_copy_leaves_the_original_untouched(genes):
    adata, controls = genes
    out = mt.tl.empirical_fdr(adata, control_genes=controls, score="score", copy=True)
    assert out is not None
    assert "empirical_fdr" not in adata.obs
    assert "empirical_fdr" in out.obs


def test_a_missing_score_column_is_reported(genes):
    adata, controls = genes
    with pytest.raises(KeyError, match="score"):
        mt.tl.empirical_fdr(adata, control_genes=controls, score="nope")


def test_a_bare_string_control_set_is_rejected(genes):
    adata, _ = genes
    with pytest.raises(TypeError, match="collection"):
        mt.tl.empirical_fdr(adata, control_genes="ctrl0", score="score")


def test_no_control_present_raises(genes):
    adata, _ = genes
    with pytest.raises(ValueError, match="no control gene"):
        mt.tl.empirical_fdr(adata, control_genes={"absent"}, score="score")


def test_it_reads_the_gene_from_a_group_column():
    """With group= the gene names come from an obs column rather than the index."""
    obs = pd.DataFrame(
        {"gene": ["a", "b", "c", "d"], "score": [0.0, 0.1, 5.0, 0.2]},
        index=["w1", "w2", "w3", "w4"],
    )
    adata = AnnData(np.zeros((4, 1)), obs=obs)
    mt.tl.empirical_fdr(adata, control_genes={"a", "b"}, score="score", group="gene", alpha=0.5)
    assert adata.obs["empirical_fdr_control"].tolist() == [True, True, False, False]
    assert bool(adata.obs["empirical_fdr"].to_numpy()[2])  # the high-scoring c is a hit


def test_a_nan_score_is_never_a_hit(genes):
    adata, controls = genes
    adata.obs.loc["hit0", "score"] = np.nan
    mt.tl.empirical_fdr(adata, control_genes=controls, score="score")
    assert np.isnan(adata.obs["empirical_fdr_pvalue"]["hit0"])
    assert not bool(adata.obs["empirical_fdr"]["hit0"])


def test_no_pvalue_is_zero(genes):
    """Add-one smoothing keeps the strongest genes off a zero tail probability."""
    adata, controls = genes
    mt.tl.empirical_fdr(adata, control_genes=controls, score="score")
    p = adata.obs["empirical_fdr_pvalue"].to_numpy()
    assert np.nanmin(p) > 0


def test_few_controls_warn(genes):
    adata, _ = genes
    with pytest.warns(UserWarning, match="control genes"):
        mt.tl.empirical_fdr(adata, control_genes={"ctrl0", "ctrl1", "ctrl2"}, score="score")


def test_all_control_scores_missing_raises(genes):
    adata, controls = genes
    adata.obs.loc[list(controls), "score"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        mt.tl.empirical_fdr(adata, control_genes=controls, score="score")


def test_a_missing_group_column_is_reported(genes):
    adata, controls = genes
    with pytest.raises(KeyError, match="group"):
        mt.tl.empirical_fdr(adata, control_genes=controls, score="score", group="nope")


def test_group_none_reads_the_perturbation_column_not_the_index(genes):
    """The gene identity comes from Metadata_Perturbation; the row index is never read as the gene."""
    adata, controls = genes
    # A row index that disagrees with the perturbation would wrongly flag no controls if the index were read.
    adata.obs_names = [f"row{i}" for i in range(adata.n_obs)]
    mt.tl.empirical_fdr(adata, control_genes=controls, score="score")
    assert adata.obs["empirical_fdr_control"].to_numpy().sum() == 300


def test_group_none_without_a_perturbation_column_is_reported(genes):
    adata, controls = genes
    del adata.obs["Metadata_Perturbation"]
    with pytest.raises(KeyError, match="group="):
        mt.tl.empirical_fdr(adata, control_genes=controls, score="score")
