"""Per-cell perturbation score: a classifier against the reference cells, scored out of fold.

Fabricated cells: non-targeting controls at the origin, a gene whose responders are shifted and whose
non-responders look like the controls, and a gene with no phenotype.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import mantispy as mt


def _cells(n_ref=600, n_gene=200, n_features=30, guide_effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    blocks, genes, guides = [], [], []
    blocks.append(rng.normal(0, 1, (n_ref, n_features)))
    genes += ["nontargeting"] * n_ref
    guides += [f"nt{i % 30}" for i in range(n_ref)]
    responders = rng.normal(0, 1, (n_gene, n_features))
    responders[: n_gene // 2, :5] += 2.0  # half the cells respond, half escaped the knockout
    blocks.append(responders)
    genes += ["HIT"] * n_gene
    guides += [f"hit{i % 4}" for i in range(n_gene)]
    quiet = rng.normal(0, 1, (n_gene, n_features))
    for guide in range(4):  # each guide of the quiet gene carries its own offset, a guide effect and no gene effect
        quiet[guide::4] += guide_effect * rng.normal(0, 1, n_features)
    blocks.append(quiet)
    genes += ["QUIET"] * n_gene
    guides += [f"quiet{i % 4}" for i in range(n_gene)]
    obs = pd.DataFrame(
        {
            "Metadata_Gene": genes,
            "Metadata_sgRNA": guides,
            "Metadata_Control_Type": ["negcon" if g == "nontargeting" else "treatment" for g in genes],
        },
        index=[f"c{i}" for i in range(len(genes))],
    )
    return AnnData(np.vstack(blocks).astype("float32"), obs=obs)


def _cells_with_classes(n_hit_interphase=200, n_hit_mitotic=200, n_features=30, seed=0):
    """Cells split into two levels of ``Metadata_Class``; HIT separates in interphase, not in mitotic."""
    rng = np.random.default_rng(seed)
    blocks, genes, classes = [], [], []
    for level, effect, n_hit in (("interphase", 2.0, n_hit_interphase), ("mitotic", 0.0, n_hit_mitotic)):
        blocks.append(rng.normal(0, 1, (400, n_features)))
        genes += ["nontargeting"] * 400
        classes += [level] * 400
        hit = rng.normal(0, 1, (n_hit, n_features))
        hit[:, :5] += effect
        blocks.append(hit)
        genes += ["HIT"] * n_hit
        classes += [level] * n_hit
    obs = pd.DataFrame(
        {
            "Metadata_Gene": genes,
            "Metadata_Class": classes,
            "Metadata_Control_Type": ["negcon" if g == "nontargeting" else "treatment" for g in genes],
        },
        index=[f"c{i}" for i in range(len(genes))],
    )
    return AnnData(np.vstack(blocks).astype("float32"), obs=obs)


def test_by_scores_each_group_within_each_level():
    adata = _cells_with_classes()
    mt.tl.perturbation_score(adata, by="Metadata_Class", replicate=None)
    table = adata.uns["mantispy"]["perturbation_score"]
    assert "Metadata_Class" in table.columns
    assert set(zip(table["group"], table["Metadata_Class"], strict=True)) == {
        ("HIT", "interphase"),
        ("HIT", "mitotic"),
    }
    t = table.set_index(["group", "Metadata_Class"])
    assert t.loc[("HIT", "interphase"), "auc"] > 0.7  # separable only among interphase cells
    assert abs(t.loc[("HIT", "mitotic"), "auc"] - 0.5) < 0.15
    score = adata.obs["perturbation_score"]
    assert score[adata.obs["Metadata_Gene"] == "nontargeting"].isna().all()
    is_hit = adata.obs["Metadata_Gene"] == "HIT"
    inter = score[is_hit & (adata.obs["Metadata_Class"] == "interphase")].mean()
    mito = score[is_hit & (adata.obs["Metadata_Class"] == "mitotic")].mean()
    assert inter > mito + 0.2


def test_by_level_where_a_group_has_too_few_cells_is_unscored_there(caplog):
    adata = _cells_with_classes(n_hit_mitotic=20)
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.perturbation_score(adata, by="Metadata_Class", replicate=None, min_cells=50)
    t = adata.uns["mantispy"]["perturbation_score"].set_index(["group", "Metadata_Class"])
    assert np.isnan(t.loc[("HIT", "mitotic"), "auc"])
    assert t.loc[("HIT", "mitotic"), "n_cells"] == 20
    assert t.loc[("HIT", "interphase"), "auc"] > 0.7  # the well-populated level is still scored
    assert "left unscored" in caplog.text
    # The too-few-cells level leaves those cells unscored, the populated one does not.
    is_hit = adata.obs["Metadata_Gene"] == "HIT"
    assert adata.obs["perturbation_score"][is_hit & (adata.obs["Metadata_Class"] == "mitotic")].isna().all()
    assert adata.obs["perturbation_score"][is_hit & (adata.obs["Metadata_Class"] == "interphase")].notna().all()


def test_by_none_matches_omitting_by():
    adata = _cells_with_classes()
    omitted = mt.tl.perturbation_score(adata, replicate=None, copy=True)
    explicit = mt.tl.perturbation_score(adata, by=None, replicate=None, copy=True)
    np.testing.assert_array_equal(
        omitted.obs["perturbation_score"].to_numpy(), explicit.obs["perturbation_score"].to_numpy()
    )
    assert "Metadata_Class" not in explicit.uns["mantispy"]["perturbation_score"].columns


def test_a_level_score_does_not_depend_on_the_other_levels():
    adata = _cells_with_classes()
    mt.tl.perturbation_score(adata, by="Metadata_Class", replicate=None)
    only_interphase = adata[adata.obs["Metadata_Class"] == "interphase"].copy()
    mt.tl.perturbation_score(only_interphase, by="Metadata_Class", replicate=None)
    np.testing.assert_array_equal(
        only_interphase.obs["perturbation_score"].to_numpy(),
        adata.obs.loc[only_interphase.obs_names, "perturbation_score"].to_numpy(),
    )


def test_a_gene_with_a_phenotype_scores_high_and_its_responders_above_its_escapers():
    adata = _cells()
    mt.tl.perturbation_score(adata)
    table = adata.uns["mantispy"]["perturbation_score"].set_index("group")
    assert list(table.columns) == ["n_cells", "n_reference", "n_folds", "auc"]
    assert table.loc["HIT", "auc"] > 0.7
    assert abs(table.loc["QUIET", "auc"] - 0.5) < 0.1
    assert table.loc["HIT", "n_cells"] == table.loc["HIT", "n_reference"] == 200

    score = adata.obs["perturbation_score"]
    assert score[adata.obs["Metadata_Gene"] == "nontargeting"].isna().all()
    hit = score[adata.obs["Metadata_Gene"] == "HIT"].to_numpy()
    assert hit[:100].mean() > hit[100:].mean() + 0.2
    assert np.all((hit >= 0) & (hit <= 1))


def test_folds_by_guide_do_not_reward_a_guide_effect():
    adata = _cells(guide_effect=1.0)
    mt.tl.perturbation_score(adata, replicate=None, key_added="cells")
    mt.tl.perturbation_score(adata, replicate="Metadata_sgRNA", key_added="guides")
    by_cells = adata.uns["mantispy"]["cells"].set_index("group")
    by_guides = adata.uns["mantispy"]["guides"].set_index("group")
    # Random folds learn each guide's offset and call it a phenotype; held-out guides expose it.
    assert by_cells.loc["QUIET", "auc"] > 0.75
    assert by_guides.loc["QUIET", "auc"] < by_cells.loc["QUIET", "auc"] - 0.15
    assert by_guides.loc["HIT", "auc"] > 0.7
    assert by_guides.loc["HIT", "n_folds"] == 4  # capped by the gene's four guides


def test_a_group_score_does_not_depend_on_the_other_groups():
    adata = _cells()
    mt.tl.perturbation_score(adata)
    alone = adata[adata.obs["Metadata_Gene"] != "QUIET"].copy()
    mt.tl.perturbation_score(alone)
    np.testing.assert_array_equal(
        alone.obs["perturbation_score"].to_numpy(),
        adata.obs.loc[alone.obs_names, "perturbation_score"].to_numpy(),
    )


def test_small_groups_and_unsplittable_groups_are_unscored_with_a_warning(caplog):
    adata = _cells()
    adata.obs["Metadata_sgRNA"] = adata.obs["Metadata_sgRNA"].astype(str)
    adata.obs.loc[adata.obs["Metadata_Gene"] == "QUIET", "Metadata_sgRNA"] = "one_guide"
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.perturbation_score(adata, replicate="Metadata_sgRNA", min_cells=201)
    assert "perturbation_score: 2 of 2 groups left unscored" in caplog.text
    assert adata.uns["mantispy"]["perturbation_score"]["auc"].isna().all()
    assert adata.obs["perturbation_score"].isna().all()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="scverse.mantispy"):
        mt.tl.perturbation_score(adata, replicate="Metadata_sgRNA")
    assert "1 of 2 groups left unscored, 0 below min_cells=50 and 1 without two folds" in caplog.text
    table = adata.uns["mantispy"]["perturbation_score"].set_index("group")
    assert np.isnan(table.loc["QUIET", "auc"]) and table.loc["HIT", "auc"] > 0.7


def test_fewer_reference_cells_than_group_cells_samples_them_all():
    adata = _cells(n_ref=120)
    mt.tl.perturbation_score(adata)
    table = adata.uns["mantispy"]["perturbation_score"].set_index("group")
    assert table.loc["HIT", "n_reference"] == 120
    assert table.loc["HIT", "auc"] > 0.7  # class_weight="balanced" carries the unequal classes


def test_missing_values_and_an_empty_reference_raise():
    adata = _cells()
    adata.X[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        mt.tl.perturbation_score(adata)
    adata = _cells()
    adata.obs["Metadata_Control_Type"] = "treatment"
    with pytest.raises(ValueError, match="selects no cells"):
        mt.tl.perturbation_score(adata)


@pytest.mark.parametrize("seed", range(4))
def test_the_fit_is_liblinears(seed):
    from sklearn.linear_model import LogisticRegression

    from mantispy._core._liblinear import fit_l2_logistic

    rng = np.random.default_rng(seed)
    n0, n1, p = rng.integers(30, 400, size=2).tolist() + [int(rng.integers(2, 40))]
    x = rng.normal(1.0, 2.0, (n0 + n1, p))
    y = rng.permutation(np.r_[np.zeros(n0, int), np.ones(n1, int)])
    x[y == 1, :2] += 0.8
    model = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear").fit(x, y)

    order = np.argsort(y, kind="stable")  # liblinear's row order, the negative class first
    weight = y.size / (2 * np.bincount(y))
    w = fit_l2_logistic(np.c_[x[order], np.ones(y.size)], 2.0 * y[order] - 1, weight[y[order]], 1e-4, 2000)
    np.testing.assert_allclose(w, np.r_[model.coef_.ravel(), model.intercept_], atol=1e-6)


def test_scores_match_scikit_learn_with_folds_by_guide_and_unequal_classes():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

    adata = _cells(n_ref=120)  # every reference cell is drawn, so the sample is known
    mt.tl.perturbation_score(adata)
    for gene in ("HIT", "QUIET"):
        rows = adata.obs["Metadata_Gene"].isin([gene, "nontargeting"]).to_numpy()
        y = (adata.obs["Metadata_Gene"][rows] == gene).to_numpy().astype(int)
        guides = adata.obs["Metadata_sgRNA"][rows].to_numpy()
        expected = cross_val_predict(
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
            adata.X[rows],
            y,
            groups=guides,
            cv=StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=0),
            method="predict_proba",
        )[:, 1]
        score = adata.obs["perturbation_score"].to_numpy()[rows]
        np.testing.assert_allclose(score[y == 1], expected[y == 1], atol=1e-6)


def test_a_constant_feature_changes_nothing_and_an_all_constant_group_is_unscored():
    adata = _cells()
    mt.tl.perturbation_score(adata)
    padded = AnnData(np.c_[adata.X, np.full(adata.n_obs, 3.0, dtype="float32")], obs=adata.obs.copy())
    mt.tl.perturbation_score(padded)
    np.testing.assert_array_equal(padded.obs["perturbation_score"], adata.obs["perturbation_score"])

    flat = AnnData(np.ones_like(adata.X), obs=adata.obs.copy())
    mt.tl.perturbation_score(flat)
    table = flat.uns["mantispy"]["perturbation_score"]
    assert table["auc"].isna().all() and (table["n_folds"] == 0).all()
    assert flat.obs["perturbation_score"].isna().all()


def test_a_backed_object_scores_as_in_memory_in_batches(tmp_path, monkeypatch):
    import anndata as ad

    from mantispy.tl import _perturbation_score

    _cells().write_h5ad(tmp_path / "cells.h5ad")
    adata = ad.read_h5ad(tmp_path / "cells.h5ad")
    mt.tl.perturbation_score(adata)
    backed = ad.read_h5ad(tmp_path / "cells.h5ad", backed="r")
    monkeypatch.setattr(_perturbation_score, "_BACKED_BATCH_ROWS", 100)  # one group per batch
    mt.tl.perturbation_score(backed, key_added="backed")
    np.testing.assert_array_equal(backed.obs["backed"], adata.obs["perturbation_score"])


def test_n_jobs_is_gone():
    with pytest.raises(TypeError, match="n_jobs"):
        mt.tl.perturbation_score(_cells(), n_jobs=2)


def test_folds_over_cells_warn_at_object_resolution():
    adata = _cells()
    adata.uns["mantispy"] = {"resolution": "object"}
    with pytest.warns(UserWarning, match="replicate=None draws folds over cells"):
        mt.tl.perturbation_score(adata, replicate=None)
    # The default folds by the guide, so it does not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        mt.tl.perturbation_score(adata)


def test_the_default_folds_by_the_guide():
    adata = _cells(guide_effect=1.0)
    mt.tl.perturbation_score(adata, key_added="default")
    mt.tl.perturbation_score(adata, replicate="Metadata_sgRNA", key_added="guides")
    default = adata.uns["mantispy"]["default"].set_index("group")
    guides = adata.uns["mantispy"]["guides"].set_index("group")
    # The default is replicate="Metadata_sgRNA", so its table matches an explicit guide fold.
    assert default.loc["HIT", "n_folds"] == 4
    np.testing.assert_array_equal(default["auc"].to_numpy(), guides["auc"].to_numpy())
    np.testing.assert_array_equal(adata.obs["default"].to_numpy(), adata.obs["guides"].to_numpy())


def test_a_missing_replicate_column_names_the_argument():
    adata = _cells()
    del adata.obs["Metadata_sgRNA"]
    with pytest.raises((KeyError, ValueError), match="replicate"):
        mt.tl.perturbation_score(adata)
