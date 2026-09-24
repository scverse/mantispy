"""Reducing an ``obsm`` representation instead of ``X`` (issue #105).

``pp.tvn``/``pp.harmony`` write a corrected embedding to ``obsm``; ``use_rep`` lets
``tl.consensus`` and ``tl.aggregate`` reduce that representation into the result's ``X``.
"""

import numpy as np
import pytest

import mantispy as mt


@pytest.fixture
def profiles():
    """Well-level profiles with a four-axis embedding in ``obsm``."""
    cells = mt.ds.synthetic_plate(
        n_plates=2, n_wells=48, n_cells=20, n_features=15, n_perturbations=4, effect_size=3.0, seed=0
    )
    wells = mt.tl.aggregate(cells, min_cells=0)
    wells.obsm["X_emb"] = np.random.default_rng(0).normal(size=(wells.n_obs, 4))
    return wells


@pytest.fixture
def embedded_cells():
    """Single cells with a four-axis embedding in ``obsm``."""
    cells = mt.ds.synthetic_plate(n_plates=2, n_wells=24, n_cells=15, n_features=20, seed=0)
    cells.obsm["X_emb"] = np.random.default_rng(1).normal(size=(cells.n_obs, 4))
    return cells


def test_consensus_median_reduces_the_embedding(profiles):
    result = mt.tl.consensus(profiles, use_rep="X_emb", method="median")
    n_groups = profiles.obs["Metadata_Perturbation"].nunique()
    assert result.X.shape == (n_groups, 4)

    emb = profiles.obsm["X_emb"]
    for index, key in enumerate(result.obs["Metadata_Perturbation"]):
        rows = (profiles.obs["Metadata_Perturbation"] == key).to_numpy()
        np.testing.assert_allclose(result.X[index], np.median(emb[rows], axis=0), rtol=1e-5, atol=1e-5)


def test_consensus_modz_reduces_the_embedding(profiles):
    result = mt.tl.consensus(profiles, use_rep="X_emb", method="modz")
    n_groups = profiles.obs["Metadata_Perturbation"].nunique()
    assert result.X.shape == (n_groups, 4)
    assert result.n_vars == 4
    assert list(result.var.index) == ["0", "1", "2", "3"]


@pytest.mark.parametrize("func", ["median", "mean"])
def test_aggregate_reduces_the_embedding(embedded_cells, func):
    wells = mt.tl.aggregate(embedded_cells, func=func, use_rep="X_emb", min_cells=0)
    reduce = np.median if func == "median" else np.mean
    emb = embedded_cells.obsm["X_emb"]

    assert wells.X.shape[1] == 4
    for index in range(wells.n_obs):
        plate = wells.obs["Metadata_Plate"].iloc[index]
        well = wells.obs["Metadata_Well"].iloc[index]
        rows = (
            (embedded_cells.obs["Metadata_Plate"] == plate) & (embedded_cells.obs["Metadata_Well"] == well)
        ).to_numpy()
        np.testing.assert_allclose(wells.X[index], reduce(emb[rows], axis=0), rtol=1e-5, atol=1e-5)


def test_var_is_a_range_index_over_the_embedding_axes(embedded_cells):
    wells = mt.tl.aggregate(embedded_cells, use_rep="X_emb", min_cells=0)
    assert wells.n_vars == 4
    assert list(wells.var.index) == ["0", "1", "2", "3"]
    # obs still matches the by-group metadata: one row per well, every well present.
    assert wells.n_obs == 48
    assert set(wells.obs["Metadata_Well"]) == set(embedded_cells.obs["Metadata_Well"])


def test_a_missing_representation_raises(embedded_cells, profiles):
    with pytest.raises(ValueError, match="no obsm 'X_missing'"):
        mt.tl.aggregate(embedded_cells, use_rep="X_missing", min_cells=0)
    with pytest.raises(ValueError, match="no obsm 'X_missing'"):
        mt.tl.consensus(profiles, use_rep="X_missing")


def test_use_rep_and_layer_are_mutually_exclusive(embedded_cells):
    embedded_cells.layers["other"] = embedded_cells.X.copy()
    with pytest.raises(ValueError, match="mutually exclusive"):
        mt.tl.aggregate(embedded_cells, use_rep="X_emb", layer="other", min_cells=0)


def test_default_use_rep_none_still_reduces_x(embedded_cells, profiles):
    """Regression guard: the default reduces ``X`` and carries ``var`` unchanged."""
    default = mt.tl.aggregate(embedded_cells, min_cells=0)
    explicit = mt.tl.aggregate(embedded_cells, use_rep=None, min_cells=0)
    np.testing.assert_array_equal(np.asarray(default.X), np.asarray(explicit.X))
    assert default.n_vars == embedded_cells.n_vars
    assert list(default.var.index) == list(embedded_cells.var.index)

    cons = mt.tl.consensus(profiles)
    assert cons.n_vars == profiles.n_vars
    assert list(cons.var.index) == list(profiles.var.index)
