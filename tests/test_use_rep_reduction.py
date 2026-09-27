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


def test_reduced_embedding_survives_a_write_round_trip(embedded_cells, tmp_path):
    """The reduced ``var`` carries the schema, so ``mt.io.write`` and a reload succeed (issue #128)."""
    wells = mt.tl.aggregate(embedded_cells, use_rep="X_emb", min_cells=0)
    assert mt.io.validate(wells).ok
    mt.io.write(wells, tmp_path / "wells.h5ad")
    reloaded = mt.io.read(tmp_path / "wells.h5ad")
    np.testing.assert_array_equal(np.asarray(reloaded.X), np.asarray(wells.X))
    assert list(reloaded.var.index) == list(wells.var.index)


def test_a_missing_representation_raises(embedded_cells, profiles):
    with pytest.raises(ValueError, match="no obsm 'X_missing'"):
        mt.tl.aggregate(embedded_cells, use_rep="X_missing", min_cells=0)
    with pytest.raises(ValueError, match="no obsm 'X_missing'"):
        mt.tl.consensus(profiles, use_rep="X_missing")


def test_use_rep_and_layer_are_mutually_exclusive(embedded_cells):
    embedded_cells.layers["other"] = embedded_cells.X.copy()
    with pytest.raises(ValueError, match="mutually exclusive"):
        mt.tl.aggregate(embedded_cells, use_rep="X_emb", layer="other", min_cells=0)
